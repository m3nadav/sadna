"""
Fine-tune Fusion + THM + ToF multistream: same data flow as train_fusion_tof_multistream_cnn.py
(split, dataset, modality dropout, missing placeholders) plus thermal (thm_*) with:

- Deeper/wider classification head: Linear → LayerNorm → ReLU → Dropout (×2 blocks) → logits.
- Partial unfreeze: joint acc+rot backbone — train conv2_block + feature classifier stem; freeze acc_zscore + conv1.
  ToF Conv3D — train conv3 + truncated classifier (384→128); freeze conv1 + conv2.
  THM — train fc1 + fc2 only (64-d features); freeze norm, conv blocks, fc3.

Warm-starts from Fusion_ToF_Multistream_CNN.pth: shape-safe load; THM branch has no keys in that
checkpoint and starts from thermal-only weights loaded separately.

Saves to models/deep_learning_models/Fusion_Thm_ToF_Multistream_CNN_finetune.pth.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

from utils import (
    ACC_COLS,
    BFRB_GESTURES,
    ROT_COLS,
    apply_label_encoding,
    compute_acc_zscore_stats,
    final_robust_split,
    finalize_training_checkpoint,
    load_train_data,
    quat_row_valid_mask,
    set_default_seeds,
)

set_default_seeds()

FUSION_CHECKPOINT_PATH = "models/deep_learning_models/Fusion_Multistream_CNN.pth"
TOF_CHECKPOINT_PATH = "models/deep_learning_models/ToF_CNN_Conv3D.pth"
THM_CHECKPOINT_PATH = "models/deep_learning_models/Multistream_CNN_thermal_only.pth"
TRIPLE_WARMSTART_PATH = "models/deep_learning_models/Fusion_ToF_Multistream_CNN.pth"
SAVE_PATH = "models/deep_learning_models/Fusion_Thm_ToF_Multistream_CNN_finetune.pth"

THM_FEAT_DIM = 64
TRAIN_MODALITY_DROPOUT_THM = 0.1

# Head: two hidden layers (256 → 128) with LayerNorm + dropout between
HEAD_HIDDEN_DIMS = (256, 128)
HEAD_DROPOUT = 0.2

# Optimizer: backbone LR much smaller than head
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4

from train_fusion_multistream_cnn import JointAccRotMultistreamNet, truncate_joint_classifier_to_features
from train_tof_cnn_conv3d import ToFCNN3D, get_tof_columns, _tof_frame_to_5x8x8, _normalize_tof_01, TOF_MAX
from train_fusion_tof_multistream_cnn import (
    compute_sequence_sensor_stats,
    TRAIN_MODALITY_DROPOUT_ROT,
    TRAIN_MODALITY_DROPOUT_TOF,
    TOF_FEAT_DIM,
    FUSION_INERTIAL_DIM,
)
from train_multistream_cnn_thermal import (
    get_thm_columns,
    is_valid_thm_array,
    compute_train_thermal_zscore_params,
    MultistreamCNNThermalNet,
    DROPOUT_P as THM_DROPOUT_DEFAULT,
    THM_MIN_C,
    THM_MAX_C,
)


def make_deep_fusion_head(in_dim: int, num_classes: int, hidden_dims, dropout_p: float) -> nn.Sequential:
    layers = []
    d_in = in_dim
    for d_out in hidden_dims:
        layers.extend(
            [
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
            ]
        )
        d_in = d_out
    layers.append(nn.Linear(d_in, num_classes))
    return nn.Sequential(*layers)


class JointPartialUnfreeze(nn.Module):
    """Joint acc+rot CNN: freeze acc_zscore + conv1; train conv2 + 64-d feature MLP stem."""

    def __init__(self, joint: JointAccRotMultistreamNet):
        super().__init__()
        self.joint = joint
        self._set_trainable_policy()

    def _set_trainable_policy(self):
        for p in self.joint.acc_zscore.parameters():
            p.requires_grad = False
        for p in self.joint.conv1_block.parameters():
            p.requires_grad = False
        for p in self.joint.conv2_block.parameters():
            p.requires_grad = True
        for p in self.joint.classifier.parameters():
            p.requires_grad = True

    def forward(self, acc, rot, rot_mask):
        return self.joint(acc, rot, rot_mask)


def make_joint_finetune_inertial(num_classes, checkpoint_path, device, trainset_df=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and "input_zscore_mean" in ckpt and "input_zscore_std" in ckpt:
        m = np.asarray(ckpt["input_zscore_mean"], dtype=np.float32)
        s = np.asarray(ckpt["input_zscore_std"], dtype=np.float32)
    elif "acc_zscore.mean" in state:
        m = state["acc_zscore.mean"].detach().cpu().numpy()
        s = state["acc_zscore.std"].detach().cpu().numpy()
    elif trainset_df is not None:
        m, s = compute_acc_zscore_stats(trainset_df)
    else:
        raise ValueError(
            "Fusion checkpoint missing input_zscore_* or acc_zscore.*; pass trainset_df or retrain."
        )
    model = JointAccRotMultistreamNet(num_classes=num_classes, norm_mean=m, norm_std=s)
    model.load_state_dict(state, strict=True)
    truncate_joint_classifier_to_features(model)
    return JointPartialUnfreeze(model).to(device)


def make_tof_finetune_extractor(num_classes, checkpoint_path, device, dropout_p=0.1):
    """Truncated ToF Conv3D → 128-d; freeze conv1–2, train conv3 + Linear(384,128)+PReLU."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        nc = ckpt.get("num_classes", num_classes)
        dropout_p = ckpt.get("dropout_p", dropout_p)
    else:
        sd = ckpt
        nc = num_classes
    tof_model = ToFCNN3D(num_classes=nc, dropout_p=dropout_p)
    tof_model.load_state_dict(sd)
    tof_model.classifier = nn.Sequential(
        tof_model.classifier[0],
        tof_model.classifier[1],
    )
    for p in tof_model.conv1.parameters():
        p.requires_grad = False
    for p in tof_model.conv2.parameters():
        p.requires_grad = False
    for p in tof_model.conv3.parameters():
        p.requires_grad = True
    for p in tof_model.classifier.parameters():
        p.requires_grad = True
    return tof_model.to(device)


class ThermalTruncatedFeat64(nn.Module):
    """MultistreamCNNThermalNet through fc2 → (B, 64); no fc3."""

    def __init__(self, net: MultistreamCNNThermalNet):
        super().__init__()
        self.net = net

    def forward(self, thm):
        x = self.net.norm(thm).unsqueeze(1)
        x = self.net.conv1_block(x)
        x = self.net.drop(x)
        x = self.net.conv2_block(x)
        x = self.net.drop(x)
        x = self.net.max_pool(x)
        x = self.net.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.net.relu(self.net.fc1(x))
        x = self.net.drop(x)
        x = self.net.relu(self.net.fc2(x))
        return x


def make_thm_finetune_extractor(
    num_classes,
    checkpoint_path,
    device,
    trainset_df=None,
    thm_cols=None,
    dropout_p=None,
):
    """64-d thermal features; freeze conv + fc3, train fc1/fc2."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        nc = ckpt.get("num_classes", num_classes)
        dp = ckpt.get("dropout_p", dropout_p if dropout_p is not None else THM_DROPOUT_DEFAULT)
        mean = ckpt.get("thm_mean")
        std = ckpt.get("thm_std")
    else:
        sd = ckpt
        nc = num_classes
        dp = dropout_p if dropout_p is not None else THM_DROPOUT_DEFAULT
        mean = std = None

    if mean is None or std is None:
        if trainset_df is None or not thm_cols:
            raise ValueError(
                "Thermal checkpoint missing thm_mean/thm_std; pass trainset_df and thm_cols to recompute."
            )
        mean_arr, std_arr = compute_train_thermal_zscore_params(trainset_df, thm_cols)
        mean = mean_arr.tolist()
        std = std_arr.tolist()

    thm_model = MultistreamCNNThermalNet(num_classes=nc, mean=mean, std=std, dropout_p=dp)
    thm_model.load_state_dict(sd, strict=True)

    for p in thm_model.norm.parameters():
        p.requires_grad = False
    for p in thm_model.conv1_block.parameters():
        p.requires_grad = False
    for p in thm_model.conv2_block.parameters():
        p.requires_grad = False
    for p in thm_model.fc1.parameters():
        p.requires_grad = True
    for p in thm_model.fc2.parameters():
        p.requires_grad = True
    for p in thm_model.fc3.parameters():
        p.requires_grad = False

    return ThermalTruncatedFeat64(thm_model).to(device)


def _sequence_has_valid_rot(data):
    rot_np = data[ROT_COLS].to_numpy(dtype=np.float64)
    return bool(quat_row_valid_mask(rot_np).any())


def _sequence_has_valid_tof(data, tof_cols):
    return (data[tof_cols] != -1).any().any()


def _sequence_has_valid_thm(data, thm_cols):
    raw = data[thm_cols].values.astype(np.float64)
    return bool(is_valid_thm_array(raw).any())


class AccRotToFThmSequenceDataset(Dataset):
    """Acc, rot, ToF, thermal per sequence; flags for missing rot / tof / thm (no sequence drop)."""

    def __init__(self, dataframe, tof_cols, thm_cols, train_mean_raw):
        self.df = dataframe.reset_index(drop=True)
        self.tof_cols = tof_cols
        self.thm_cols = thm_cols
        self.train_mean_raw = np.asarray(train_mean_raw, dtype=np.float64).reshape(len(thm_cols))
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        acc = torch.tensor(data[ACC_COLS].values, dtype=torch.float32).T
        rot_np = data[ROT_COLS].to_numpy(dtype=np.float64)
        valid = quat_row_valid_mask(rot_np)
        rot_filled = np.nan_to_num(rot_np, nan=0.0).astype(np.float32)
        rot_filled[~valid] = 0.0
        rot = torch.tensor(rot_filled, dtype=torch.float32).T
        rot_mask = torch.tensor(valid, dtype=torch.float32).unsqueeze(0)
        has_rot = torch.tensor(_sequence_has_valid_rot(data), dtype=torch.bool)

        has_tof = torch.tensor(_sequence_has_valid_tof(data, self.tof_cols), dtype=torch.bool)
        tof = data[self.tof_cols].values.astype(np.float64)
        tof[tof == -1] = TOF_MAX
        tof[np.isnan(tof)] = TOF_MAX
        tof = np.nan_to_num(tof, nan=TOF_MAX, posinf=TOF_MAX, neginf=0.0)
        tof_per_frame = _tof_frame_to_5x8x8(tof)
        tof_per_frame = _normalize_tof_01(tof_per_frame)
        tof_tensor = torch.clamp(torch.tensor(tof_per_frame, dtype=torch.float32), 0.0, 1.0)
        if not has_tof.item():
            tof_tensor = torch.zeros(1, 5, 8, 8, dtype=torch.float32)

        has_thm = torch.tensor(_sequence_has_valid_thm(data, self.thm_cols), dtype=torch.bool)
        raw_thm = data[self.thm_cols].values.astype(np.float64)
        mask = is_valid_thm_array(raw_thm)
        imputed = raw_thm.copy()
        for c in range(raw_thm.shape[1]):
            col_mask = mask[:, c]
            if not col_mask.all():
                imputed[~col_mask, c] = self.train_mean_raw[c]
        thm = torch.tensor(imputed.T, dtype=torch.float32)
        if not has_thm.item():
            thm = torch.zeros(5, 1, dtype=torch.float32)

        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, rot, rot_mask, tof_tensor, thm, has_rot, has_tof, has_thm, label


def collate_acc_rot_tof_thm(train_mean_raw):
    """Pad acc/rot/tof like fusion; pad thm (B,5,max_T) with per-channel train mean (raw °C)."""

    tm = torch.as_tensor(train_mean_raw, dtype=torch.float32).view(5, 1)

    def collate_fn(batch):
        accs, rots, masks, tofs, thms, has_rots, has_tofs, has_thms, labels = zip(*batch)
        acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
        rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
        mask_padded = pad_sequence([m.T for m in masks], batch_first=True, padding_value=0).transpose(1, 2)
        tof_padded = pad_sequence(list(tofs), batch_first=True, padding_value=0.0)
        lengths = torch.tensor([t.shape[0] for t in tofs], dtype=torch.long)

        max_t = max(t.shape[1] for t in thms)
        b = len(thms)
        thm_padded = tm.expand(5, max_t).unsqueeze(0).expand(b, 5, max_t).clone()
        for i, t in enumerate(thms):
            ti = t.shape[1]
            thm_padded[i, :, :ti] = t

        has_rot_b = torch.stack(list(has_rots))
        has_tof_b = torch.stack(list(has_tofs))
        has_thm_b = torch.stack(list(has_thms))
        labels_stacked = torch.stack(labels)
        return (
            acc_padded,
            rot_padded,
            mask_padded,
            tof_padded,
            lengths,
            thm_padded,
            has_rot_b,
            has_tof_b,
            has_thm_b,
            labels_stacked,
        )

    return collate_fn


def compute_sequence_thm_stats(df, thm_cols, split_name=""):
    """Count sequences with no valid thermal reading."""
    if not thm_cols:
        return
    n_seq = df["sequence_id"].nunique()
    if n_seq == 0:
        print(f"=== {split_name}: no sequences (THM) ===")
        return

    def _miss_thm(g):
        raw = g[thm_cols].values.astype(np.float64)
        return not is_valid_thm_array(raw).any()

    gr = df.groupby("sequence_id", sort=False)
    miss = gr.apply(_miss_thm).astype(bool)
    title = split_name or "split"
    print(
        f"\n--- Thermal availability: {title} ({n_seq} sequences) ---\n"
        f"  Missing THM (no valid reading in sequence): {int(miss.sum())} ({100 * miss.mean():.2f}%)"
    )


class ThmToFFusionMultistreamCNNDeepFinetune(nn.Module):
    """Partial-unfreeze joint inertial + ToF + THM (fc1/fc2 only), deep head, missing_tof / missing_thm."""

    def __init__(
        self,
        joint_partial,
        tof_feat,
        thm_feat,
        num_classes=18,
        head_hidden_dims=HEAD_HIDDEN_DIMS,
        head_dropout=HEAD_DROPOUT,
    ):
        super().__init__()
        self.joint_partial = joint_partial
        self.tof_feat = tof_feat
        self.thm_feat = thm_feat
        self.missing_tof = nn.Parameter(torch.zeros(TOF_FEAT_DIM))
        self.missing_thm = nn.Parameter(torch.zeros(THM_FEAT_DIM))
        in_dim = FUSION_INERTIAL_DIM + TOF_FEAT_DIM + THM_FEAT_DIM
        self.fusion_classifier = make_deep_fusion_head(
            in_dim, num_classes, head_hidden_dims, head_dropout
        )
        self.train_dropout_rot = TRAIN_MODALITY_DROPOUT_ROT
        self.train_dropout_tof = TRAIN_MODALITY_DROPOUT_TOF
        self.train_dropout_thm = TRAIN_MODALITY_DROPOUT_THM

    def forward(self, acc, rot, rot_mask, tof_padded, lengths, thm_padded, has_rot, has_tof, has_thm):
        B = acc.shape[0]
        device = acc.device
        dtype = acc.dtype

        has_rot = has_rot.to(device=device, dtype=torch.bool)
        has_tof = has_tof.to(device=device, dtype=torch.bool)
        has_thm = has_thm.to(device=device, dtype=torch.bool)

        if self.training and (
            self.train_dropout_rot > 0 or self.train_dropout_tof > 0 or self.train_dropout_thm > 0
        ):
            if self.train_dropout_rot > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_rot) & has_rot
                has_rot = has_rot & ~drop
            if self.train_dropout_tof > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_tof) & has_tof
                has_tof = has_tof & ~drop
            if self.train_dropout_thm > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_thm) & has_thm
                has_thm = has_thm & ~drop

        has_rot_f = has_rot.to(dtype=dtype, device=device).view(B, 1, 1)
        rot_mask_eff = rot_mask.to(device=device, dtype=dtype) * has_rot_f
        fr = self.joint_partial(acc, rot, rot_mask_eff)

        tf = torch.zeros(B, TOF_FEAT_DIM, device=device, dtype=dtype)
        if has_tof.any():
            idx = has_tof.nonzero(as_tuple=True)[0]
            tf[idx] = self.tof_feat(tof_padded[idx], lengths[idx])
        if (~has_tof).any():
            idx = (~has_tof).nonzero(as_tuple=True)[0]
            tf[idx] = self.missing_tof.unsqueeze(0).expand(len(idx), -1)

        hf = torch.zeros(B, THM_FEAT_DIM, device=device, dtype=dtype)
        if has_thm.any():
            idx = has_thm.nonzero(as_tuple=True)[0]
            hf[idx] = self.thm_feat(thm_padded[idx])
        if (~has_thm).any():
            idx = (~has_thm).nonzero(as_tuple=True)[0]
            hf[idx] = self.missing_thm.unsqueeze(0).expand(len(idx), -1)

        x = torch.cat([fr, tf, hf], dim=1)
        return self.fusion_classifier(x)

    def fused_features(self, acc, rot, rot_mask, tof_padded, lengths, thm_padded, has_rot, has_tof, has_thm):
        """Fused dim before fusion_classifier (eval-friendly when not training)."""
        B = acc.shape[0]
        device = acc.device
        dtype = acc.dtype

        has_rot = has_rot.to(device=device, dtype=torch.bool)
        has_tof = has_tof.to(device=device, dtype=torch.bool)
        has_thm = has_thm.to(device=device, dtype=torch.bool)

        if self.training and (
            self.train_dropout_rot > 0 or self.train_dropout_tof > 0 or self.train_dropout_thm > 0
        ):
            if self.train_dropout_rot > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_rot) & has_rot
                has_rot = has_rot & ~drop
            if self.train_dropout_tof > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_tof) & has_tof
                has_tof = has_tof & ~drop
            if self.train_dropout_thm > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_thm) & has_thm
                has_thm = has_thm & ~drop

        has_rot_f = has_rot.to(dtype=dtype, device=device).view(B, 1, 1)
        rot_mask_eff = rot_mask.to(device=device, dtype=dtype) * has_rot_f
        fr = self.joint_partial(acc, rot, rot_mask_eff)

        tf = torch.zeros(B, TOF_FEAT_DIM, device=device, dtype=dtype)
        if has_tof.any():
            idx = has_tof.nonzero(as_tuple=True)[0]
            tf[idx] = self.tof_feat(tof_padded[idx], lengths[idx])
        if (~has_tof).any():
            idx = (~has_tof).nonzero(as_tuple=True)[0]
            tf[idx] = self.missing_tof.unsqueeze(0).expand(len(idx), -1)

        hf = torch.zeros(B, THM_FEAT_DIM, device=device, dtype=dtype)
        if has_thm.any():
            idx = has_thm.nonzero(as_tuple=True)[0]
            hf[idx] = self.thm_feat(thm_padded[idx])
        if (~has_thm).any():
            idx = (~has_thm).nonzero(as_tuple=True)[0]
            hf[idx] = self.missing_thm.unsqueeze(0).expand(len(idx), -1)

        return torch.cat([fr, tf, hf], dim=1)


def load_state_dict_shape_safe(model: nn.Module, state_dict: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Load checkpoint weights into model, skipping keys with missing name or shape mismatch.
    strict=False alone still errors on shape mismatch; this filters first.
    Returns (loaded_keys, skipped_shape_mismatch, missing_in_checkpoint).
    """
    target = model.state_dict()
    to_load = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in target:
            continue
        if target[k].shape != v.shape:
            skipped.append(k)
            continue
        to_load[k] = v
    missing = [k for k in target if k not in to_load]
    model.load_state_dict(to_load, strict=False)
    loaded = list(to_load.keys())
    return loaded, skipped, missing


def collect_param_groups(model: ThmToFFusionMultistreamCNNDeepFinetune):
    """Head + missing vs. joint inertial vs. ToF vs. THM (fc) tunables."""
    head_params = list(model.fusion_classifier.parameters()) + [
        model.missing_tof,
        model.missing_thm,
    ]

    fusion_bb = [p for p in model.joint_partial.joint.parameters() if p.requires_grad]

    tof_params = [p for p in model.tof_feat.parameters() if p.requires_grad]
    thm_params = [p for p in model.thm_feat.parameters() if p.requires_grad]

    head_ids = {id(p) for p in head_params}
    fusion_bb = [p for p in fusion_bb if id(p) not in head_ids]
    tof_params = [p for p in tof_params if id(p) not in head_ids]
    thm_params = [p for p in thm_params if id(p) not in head_ids]

    return head_params, fusion_bb, tof_params, thm_params


def main():
    train_df = load_train_data()

    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)
    print(
        f"Split (no sensor drop): train {len(trainset_df)} rows / {trainset_df['sequence_id'].nunique()} seq, "
        f"val {len(valset_df)} / {valset_df['sequence_id'].nunique()}, "
        f"test {len(testset_df)} / {testset_df['sequence_id'].nunique()}"
    )

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    tof_cols = get_tof_columns(train_df)
    thm_cols = get_thm_columns(train_df)
    if not thm_cols:
        raise ValueError("No thm_* columns in train data; cannot train THM fusion fine-tune.")

    compute_sequence_sensor_stats(trainset_df, tof_cols, "TRAIN (before training)")
    compute_sequence_sensor_stats(valset_df, tof_cols, "VAL")
    compute_sequence_sensor_stats(testset_df, tof_cols, "TEST")
    compute_sequence_thm_stats(trainset_df, thm_cols, "TRAIN (before training)")
    compute_sequence_thm_stats(valset_df, thm_cols, "VAL")
    compute_sequence_thm_stats(testset_df, thm_cols, "TEST")

    train_mean_raw, train_std_raw = compute_train_thermal_zscore_params(trainset_df, thm_cols)
    print(f"Thermal z-score train stats {thm_cols}: mean={train_mean_raw}, std={train_std_raw}")

    train_seq_ids = trainset_df["sequence_id"].unique().tolist()
    val_seq_ids = valset_df["sequence_id"].unique().tolist()
    test_seq_ids = testset_df["sequence_id"].unique().tolist()

    num_classes = len(le.classes_)
    print(f"\nNum classes: {num_classes}")
    print(
        f"Training-time modality dropout (stochastic, only on present sensors): "
        f"rot p={TRAIN_MODALITY_DROPOUT_ROT}, tof p={TRAIN_MODALITY_DROPOUT_TOF}, thm p={TRAIN_MODALITY_DROPOUT_THM}"
    )
    print(
        f"Fine-tune: head hidden {HEAD_HIDDEN_DIMS}, dropout {HEAD_DROPOUT} | "
        f"LR head {LR_HEAD}, LR backbone {LR_BACKBONE}"
    )

    collate_fn = collate_acc_rot_tof_thm(train_mean_raw)
    train_ds = AccRotToFThmSequenceDataset(trainset_df, tof_cols, thm_cols, train_mean_raw)
    val_ds = AccRotToFThmSequenceDataset(valset_df, tof_cols, thm_cols, train_mean_raw)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading joint inertial backbone (partial unfreeze) from {FUSION_CHECKPOINT_PATH}...")
    joint_partial = make_joint_finetune_inertial(num_classes, FUSION_CHECKPOINT_PATH, device, trainset_df)
    print(f"Loading ToF backbone (partial unfreeze) from {TOF_CHECKPOINT_PATH}...")
    tof_feat = make_tof_finetune_extractor(num_classes, TOF_CHECKPOINT_PATH, device)
    print(f"Loading THM backbone (fc layers only trainable) from {THM_CHECKPOINT_PATH}...")
    thm_feat = make_thm_finetune_extractor(
        num_classes, THM_CHECKPOINT_PATH, device, trainset_df=trainset_df, thm_cols=thm_cols
    )

    model = ThmToFFusionMultistreamCNNDeepFinetune(
        joint_partial,
        tof_feat,
        thm_feat,
        num_classes=num_classes,
        head_hidden_dims=HEAD_HIDDEN_DIMS,
        head_dropout=HEAD_DROPOUT,
    )
    model = model.to(device)

    if os.path.isfile(TRIPLE_WARMSTART_PATH):
        warm = torch.load(TRIPLE_WARMSTART_PATH, map_location=device, weights_only=False)
        warm_sd = warm["model_state_dict"] if isinstance(warm, dict) and "model_state_dict" in warm else warm
        loaded, skipped_shape, missing = load_state_dict_shape_safe(model, warm_sd)
        print(
            f"Warm-start from {TRIPLE_WARMSTART_PATH} (shape-safe): "
            f"loaded {len(loaded)} tensors, skipped shape mismatch {len(skipped_shape)}, "
            f"not in ckpt / left uninitialized {len(missing)}"
        )
        if skipped_shape:
            print(f"  Skipped (e.g. old fusion head): {skipped_shape[:8]}{'...' if len(skipped_shape) > 8 else ''}")
    else:
        print(f"No warm-start file at {TRIPLE_WARMSTART_PATH}; training from fusion+ToF+THM init only.")

    head_p, fusion_p, tof_p, thm_p = collect_param_groups(model)
    optimizer = optim.AdamW(
        [
            {"params": head_p, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
            {"params": fusion_p, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
            {"params": tof_p, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
            {"params": thm_p, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        ]
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head_p)
    n_fusion_bb = sum(p.numel() for p in fusion_p)
    n_tof_bb = sum(p.numel() for p in tof_p)
    n_thm_bb = sum(p.numel() for p in thm_p)
    print(
        f"Model: {n_total:,} total params | trainable {n_trainable:,} "
        f"(head+missing {n_head:,}, fusion BB {n_fusion_bb:,}, ToF BB {n_tof_bb:,}, THM fc {n_thm_bb:,})"
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    thm_bounds = (THM_MIN_C, THM_MAX_C)

    def train_step(batch):
        acc, rot, rot_mask, tof_padded, lengths, thm_padded, has_rot, has_tof, has_thm, labels = batch
        acc = acc.to(device)
        rot = rot.to(device)
        rot_mask = rot_mask.to(device)
        tof_padded = tof_padded.to(device)
        lengths = lengths.to(device)
        thm_padded = thm_padded.to(device)
        has_rot = has_rot.to(device)
        has_tof = has_tof.to(device)
        has_thm = has_thm.to(device)
        labels = labels.to(device)
        outputs = model(
            acc, rot, rot_mask, tof_padded, lengths, thm_padded, has_rot, has_tof, has_thm
        )
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()
        batch_acc = (outputs.argmax(1) == labels).float().mean()
        return loss.item(), batch_acc.item()

    def val_step(batch):
        v_acc, v_rot, v_rm, v_tof, v_len, v_thm, v_hr, v_ht, v_hth, v_labels = batch
        v_acc = v_acc.to(device)
        v_rot = v_rot.to(device)
        v_rm = v_rm.to(device)
        v_tof = v_tof.to(device)
        v_len = v_len.to(device)
        v_thm = v_thm.to(device)
        v_hr = v_hr.to(device)
        v_ht = v_ht.to(device)
        v_hth = v_hth.to(device)
        v_labels = v_labels.to(device)
        val_outputs = model(
            v_acc, v_rot, v_rm, v_tof, v_len, v_thm, v_hr, v_ht, v_hth
        )
        return criterion(val_outputs, v_labels).item(), (
            val_outputs.argmax(1) == v_labels
        ).float().mean().item()

    history = finalize_training_checkpoint(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        gesture_map,
        train_seq_ids,
        val_seq_ids,
        test_seq_ids,
        SAVE_PATH,
        train_step,
        val_step,
        num_epochs=100,
        early_stop_patience=15,
        extra_checkpoint_keys={
            "thm_cols": thm_cols,
            "thm_mean": train_mean_raw.tolist(),
            "thm_std": train_std_raw.tolist(),
            "thm_valid_bounds": thm_bounds,
        },
    )
    print(f"\nSaved fine-tuned checkpoint to {SAVE_PATH}")
    return model, history, le


make_fusion_finetune_128 = make_joint_finetune_inertial

if __name__ == "__main__":
    model, history, le = main()
