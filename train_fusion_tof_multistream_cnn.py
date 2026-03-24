"""
Train a fusion model: frozen joint inertial (64-dim) and ToF (128-dim) features, with a learnable
missing_tof placeholder when ToF is absent. Joint backbone uses acc + rot + per-timestep quat mask.

Sequences are NOT dropped for missing quaternion or missing ToF; has_rot/has_tof gate sequence-level
availability (rot_mask is zeroed when has_rot is False). Optional modality dropout during training.

Default paths: Fusion_Multistream_CNN.pth, ToF_CNN.pth.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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
from train_fusion_multistream_cnn import (
    JointAccRotMultistreamNet,
    truncate_joint_classifier_to_features,
)
from train_tof_cnn import (
    ToFCNN,
    get_tof_columns,
    _tof_frame_to_5x8x8,
    _normalize_tof_01,
    TOF_MAX,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()

FUSION_CHECKPOINT_PATH = "models/deep_learning_models/Fusion_Multistream_CNN.pth"
TOF_CHECKPOINT_PATH = "models/deep_learning_models/ToF_CNN.pth"

# Extra stochastic “missing sensor” during training (only applied when sensor is actually present)
TRAIN_MODALITY_DROPOUT_ROT = 0.1
TRAIN_MODALITY_DROPOUT_TOF = 0.1

# Feature dimensions (must match frozen backbones)
TOF_FEAT_DIM = 128
FUSION_INERTIAL_DIM = 64


# ---------------------------------------------------------------------------
# Sequence-level sensor availability (for reporting and optional dropout tuning)
# ---------------------------------------------------------------------------
def compute_sequence_sensor_stats(df, tof_cols, split_name=""):
    """Count sequences missing ROT (any NaN in quaternion) and/or ToF (all -1)."""
    n_seq = df["sequence_id"].nunique()
    if n_seq == 0:
        print(f"=== {split_name}: no sequences ===")
        return {
            "n_seq": 0,
            "missing_rot": 0,
            "missing_tof": 0,
            "both": 0,
            "p_missing_rot": 0.0,
            "p_missing_tof": 0.0,
        }

    def _miss_rot(g):
        return g[ROT_COLS].isna().any().any()

    def _miss_tof(g):
        return not (g[tof_cols] != -1).any().any()

    gr = df.groupby("sequence_id", sort=False)
    miss_rot = gr.apply(_miss_rot)
    miss_tof = gr.apply(_miss_tof)
    mr = miss_rot.astype(bool)
    mt = miss_tof.astype(bool)
    both = int((mr & mt).sum())
    only_rot = int((mr & ~mt).sum())
    only_tof = int((~mr & mt).sum())
    neither = int((~mr & ~mt).sum())

    title = split_name or "split"
    print(f"\n--- Sequence sensor availability: {title} ({n_seq} sequences) ---")
    print(f"  Missing ROT (any NaN in sequence): {int(mr.sum())} ({100 * mr.mean():.2f}%)")
    print(f"  Missing ToF (all pixels -1 in sequence): {int(mt.sum())} ({100 * mt.mean():.2f}%)")
    print(f"  Both missing: {both}")
    print(f"  Missing ROT only: {only_rot}")
    print(f"  Missing ToF only: {only_tof}")
    print(f"  Neither missing: {neither}")

    return {
        "n_seq": n_seq,
        "missing_rot": int(mr.sum()),
        "missing_tof": int(mt.sum()),
        "both": both,
        "p_missing_rot": float(mr.mean()),
        "p_missing_tof": float(mt.mean()),
    }


# ---------------------------------------------------------------------------
# Frozen 64-dim joint inertial extractor from trained Fusion_Multistream_CNN (joint arch)
# ---------------------------------------------------------------------------
class JointFrozenInertial(nn.Module):
    """Frozen truncated JointAccRotMultistreamNet; forward(acc, rot, rot_mask) -> (B, 64)."""

    def __init__(self, joint_net: JointAccRotMultistreamNet):
        super().__init__()
        self.joint = joint_net
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, acc, rot, rot_mask):
        return self.joint(acc, rot, rot_mask)


def make_joint_frozen_inertial(num_classes, checkpoint_path, device, trainset_df=None):
    """Load joint Fusion_Multistream_CNN checkpoint; truncate classifier to 64-d features; freeze."""
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
            "Fusion checkpoint missing input_zscore_* or acc_zscore.* buffers; pass trainset_df or retrain."
        )
    model = JointAccRotMultistreamNet(num_classes=num_classes, norm_mean=m, norm_std=s)
    model.load_state_dict(state, strict=True)
    truncate_joint_classifier_to_features(model)
    for p in model.parameters():
        p.requires_grad = False
    return JointFrozenInertial(model).to(device)


def make_tof_feature_extractor(num_classes, checkpoint_path, device, dropout_p=0.1):
    """Load ToFCNN; keep Linear(384->128) + PReLU for 128-dim features; freeze all."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        nc = ckpt.get("num_classes", num_classes)
        dropout_p = ckpt.get("dropout_p", dropout_p)
    else:
        sd = ckpt
        nc = num_classes
    model = ToFCNN(num_classes=nc, dropout_p=dropout_p)
    model.load_state_dict(sd)
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
    )
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


# ---------------------------------------------------------------------------
# Fusion head with missing ToF placeholder
# ---------------------------------------------------------------------------
class ToFFusionMultistreamCNN(nn.Module):
    """
    Frozen joint inertial (64-d) and ToF backbones; trainable missing_tof (128) and head.
    has_rot[i]=False -> zero entire rot_mask for that sample before joint forward.
    has_tof[i]=False -> use missing_tof instead of ToF backbone for that sample.
    """

    def __init__(self, joint_inertial, tof_feat, num_classes=18, hidden_dim=64):
        super(ToFFusionMultistreamCNN, self).__init__()
        self.joint_inertial = joint_inertial
        self.tof_feat = tof_feat
        self.missing_tof = nn.Parameter(torch.zeros(TOF_FEAT_DIM))
        self.fusion_classifier = nn.Sequential(
            nn.Linear(FUSION_INERTIAL_DIM + TOF_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.train_dropout_rot = TRAIN_MODALITY_DROPOUT_ROT
        self.train_dropout_tof = TRAIN_MODALITY_DROPOUT_TOF

    def forward(self, acc, rot, rot_mask, tof_padded, lengths, has_rot, has_tof):
        """
        rot_mask: (B, 1, T) float 0/1 per-timestep valid quat (before has_rot gating).
        has_rot, has_tof: (B,) bool — sequence-level sensor presence.
        """
        B = acc.shape[0]
        device = acc.device
        dtype = acc.dtype

        has_rot = has_rot.to(device=device, dtype=torch.bool)
        has_tof = has_tof.to(device=device, dtype=torch.bool)

        if self.training and (self.train_dropout_rot > 0 or self.train_dropout_tof > 0):
            if self.train_dropout_rot > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_rot) & has_rot
                has_rot = has_rot & ~drop
            if self.train_dropout_tof > 0:
                drop = (torch.rand(B, device=device) < self.train_dropout_tof) & has_tof
                has_tof = has_tof & ~drop

        has_rot_f = has_rot.to(dtype=dtype, device=device).view(B, 1, 1)
        rot_mask_eff = rot_mask.to(device=device, dtype=dtype) * has_rot_f

        inertial = self.joint_inertial(acc, rot, rot_mask_eff)

        tf = torch.zeros(B, TOF_FEAT_DIM, device=device, dtype=dtype)
        if has_tof.any():
            idx = has_tof.nonzero(as_tuple=True)[0]
            tf[idx] = self.tof_feat(tof_padded[idx], lengths[idx])
        if (~has_tof).any():
            idx = (~has_tof).nonzero(as_tuple=True)[0]
            tf[idx] = self.missing_tof.unsqueeze(0).expand(len(idx), -1)

        x = torch.cat([inertial, tf], dim=1)
        return self.fusion_classifier(x)


# ---------------------------------------------------------------------------
# Dataset: acc, rot, rot_mask, ToF, flags, label
# ---------------------------------------------------------------------------
def _sequence_has_valid_rot(data):
    """At least one timestep with valid unit quaternion (matches joint training semantics)."""
    rot_np = data[ROT_COLS].to_numpy(dtype=np.float64)
    return bool(quat_row_valid_mask(rot_np).any())


def _sequence_has_valid_tof(data, tof_cols):
    return (data[tof_cols] != -1).any().any()


class AccRotToFSequenceDataset(Dataset):
    def __init__(self, dataframe, tof_cols):
        self.df = dataframe.reset_index(drop=True)
        self.tof_cols = tof_cols
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

        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, rot, rot_mask, tof_tensor, has_rot, has_tof, label


def collate_acc_rot_tof(batch):
    accs, rots, masks, tofs, has_rots, has_tofs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
    mask_padded = pad_sequence([m.T for m in masks], batch_first=True, padding_value=0).transpose(1, 2)
    tof_padded = pad_sequence(list(tofs), batch_first=True, padding_value=0.0)
    lengths = torch.tensor([t.shape[0] for t in tofs], dtype=torch.long)
    has_rot_b = torch.stack(list(has_rots))
    has_tof_b = torch.stack(list(has_tofs))
    labels_stacked = torch.stack(labels)
    return (
        acc_padded,
        rot_padded,
        mask_padded,
        tof_padded,
        lengths,
        has_rot_b,
        has_tof_b,
        labels_stacked,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    compute_sequence_sensor_stats(trainset_df, tof_cols, "TRAIN (before training)")
    compute_sequence_sensor_stats(valset_df, tof_cols, "VAL")
    compute_sequence_sensor_stats(testset_df, tof_cols, "TEST")

    train_seq_ids = trainset_df["sequence_id"].unique().tolist()
    val_seq_ids = valset_df["sequence_id"].unique().tolist()
    test_seq_ids = testset_df["sequence_id"].unique().tolist()

    num_classes = len(le.classes_)
    print(f"\nNum classes: {num_classes}")
    print(
        f"Training-time modality dropout (stochastic, only on present sensors): "
        f"rot p={TRAIN_MODALITY_DROPOUT_ROT}, tof p={TRAIN_MODALITY_DROPOUT_TOF}"
    )

    train_ds = AccRotToFSequenceDataset(trainset_df, tof_cols)
    val_ds = AccRotToFSequenceDataset(valset_df, tof_cols)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_acc_rot_tof,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_acc_rot_tof,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading frozen joint inertial backbone from {FUSION_CHECKPOINT_PATH}...")
    joint_inertial = make_joint_frozen_inertial(
        num_classes, FUSION_CHECKPOINT_PATH, device, trainset_df
    )
    print(f"Loading frozen ToF backbone from {TOF_CHECKPOINT_PATH}...")
    tof_feat = make_tof_feature_extractor(num_classes, TOF_CHECKPOINT_PATH, device)

    model = ToFFusionMultistreamCNN(joint_inertial, tof_feat, num_classes=num_classes, hidden_dim=64)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total parameters, {n_trainable:,} trainable (head + missing_tof)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Fusion_ToF_Multistream_CNN.pth"

    def train_step(batch):
        acc, rot, rot_mask, tof_padded, lengths, has_rot, has_tof, labels = batch
        acc = acc.to(device)
        rot = rot.to(device)
        rot_mask = rot_mask.to(device)
        tof_padded = tof_padded.to(device)
        lengths = lengths.to(device)
        has_rot = has_rot.to(device)
        has_tof = has_tof.to(device)
        labels = labels.to(device)
        outputs = model(acc, rot, rot_mask, tof_padded, lengths, has_rot, has_tof)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()
        batch_acc = (outputs.argmax(1) == labels).float().mean()
        return loss.item(), batch_acc.item()

    def val_step(batch):
        v_acc, v_rot, v_rm, v_tof, v_len, v_hr, v_ht, v_labels = batch
        v_acc = v_acc.to(device)
        v_rot = v_rot.to(device)
        v_rm = v_rm.to(device)
        v_tof = v_tof.to(device)
        v_len = v_len.to(device)
        v_hr = v_hr.to(device)
        v_ht = v_ht.to(device)
        v_labels = v_labels.to(device)
        val_outputs = model(v_acc, v_rot, v_rm, v_tof, v_len, v_hr, v_ht)
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
        save_path,
        train_step,
        val_step,
        num_epochs=100,
        early_stop_patience=15,
    )
    return model, history, le


# Backward-compat aliases for analysis / external scripts
make_fusion_frozen_128 = make_joint_frozen_inertial

if __name__ == "__main__":
    model, history, le = main()
