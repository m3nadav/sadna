"""
Fine-tune Fusion_ToF_Multistream: same data flow as train_fusion_tof_multistream_cnn.py
(split, dataset, modality dropout, missing placeholders) but:

- Deeper/wider classification head: Linear → LayerNorm → ReLU → Dropout (×2 blocks) → logits.
- Partial unfreeze: acc/rot backbones — train conv2_block + feature classifier stem; freeze norm + conv1.
  ToF — train conv3 + truncated classifier (384→128); freeze conv1 + conv2.
- Two AdamW param groups: head + missing embeddings at LR_HEAD; backbone tunables at LR_BACKBONE (much smaller).

Warm-starts from Fusion_ToF_Multistream_CNN.pth: loads only tensors whose shapes match (skips the old
small head; deep head stays randomly initialized). Backbones + missing embeddings load when compatible.

Saves to models/deep_learning_models/Fusion_ToF_Multistream_CNN_finetune.pth (does not overwrite the baseline checkpoint).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

FUSION_CHECKPOINT_PATH = "models/deep_learning_models/Fusion_Multistream_CNN.pth"
TOF_CHECKPOINT_PATH = "models/deep_learning_models/ToF_CNN.pth"
TRIPLE_WARMSTART_PATH = "models/deep_learning_models/Fusion_ToF_Multistream_CNN.pth"
SAVE_PATH = "models/deep_learning_models/Fusion_ToF_Multistream_CNN_finetune.pth"

# Head: two hidden layers (256 → 128) with LayerNorm + dropout between
HEAD_HIDDEN_DIMS = (256, 128)
HEAD_DROPOUT = 0.2

# Optimizer: backbone LR much smaller than head
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4

from train_multistream_cnn import (
    load_train_data,
    final_robust_split,
    apply_label_encoding,
    BFRB_GESTURES,
    MultistreamCNNInertialNet,
)
from train_multistream_cnn_rot import MultistreamCNNRotNet
from train_fusion_multistream_cnn import FusionMultistreamCNN
from train_tof_cnn import ToFCNN, get_tof_columns
from train_fusion_tof_multistream_cnn import (
    compute_sequence_sensor_stats,
    AccRotToFSequenceDataset,
    collate_acc_rot_tof,
    save_model_and_metadata,
    train_model,
    TRAIN_MODALITY_DROPOUT_ROT,
    TRAIN_MODALITY_DROPOUT_TOF,
    ROT_FEAT_DIM,
    TOF_FEAT_DIM,
    FUSION_INERTIAL_DIM,
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


class FusionPartialUnfreeze128(nn.Module):
    """acc + rot feature extractors: freeze early conv; train conv2 + 64-d feature MLP stem."""

    def __init__(self, fusion_model: FusionMultistreamCNN):
        super().__init__()
        self.acc_backbone = fusion_model.acc_backbone
        self.rot_backbone = fusion_model.rot_backbone
        self._set_trainable_policy()

    def _set_trainable_policy(self):
        for backbone in (self.acc_backbone, self.rot_backbone):
            for p in backbone.norm.parameters():
                p.requires_grad = False
            for p in backbone.conv1_block.parameters():
                p.requires_grad = False
            for p in backbone.conv2_block.parameters():
                p.requires_grad = True
            for p in backbone.classifier.parameters():
                p.requires_grad = True

    def forward(self, acc, rot):
        acc_feat = self.acc_backbone(acc)
        rot_feat = self.rot_backbone(rot)
        return torch.cat([acc_feat, rot_feat], dim=1)


def make_fusion_finetune_128(num_classes, checkpoint_path, device):
    acc_backbone = MultistreamCNNInertialNet(num_classes=num_classes)
    acc_backbone.classifier = nn.Sequential(
        acc_backbone.classifier[0],
        acc_backbone.classifier[1],
        acc_backbone.classifier[2],
        acc_backbone.classifier[3],
    )
    rot_backbone = MultistreamCNNRotNet(num_classes=num_classes)
    rot_backbone.classifier = nn.Sequential(
        rot_backbone.classifier[0],
        rot_backbone.classifier[1],
        rot_backbone.classifier[2],
        rot_backbone.classifier[3],
    )
    fusion = FusionMultistreamCNN(acc_backbone, rot_backbone, num_classes=num_classes, hidden_dim=64)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    fusion.load_state_dict(state)
    return FusionPartialUnfreeze128(fusion).to(device)


def make_tof_finetune_extractor(num_classes, checkpoint_path, device, dropout_p=0.1):
    """Truncated ToF → 128-d; freeze conv1–2, train conv3 + Linear(384,128)+PReLU."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        nc = ckpt.get("num_classes", num_classes)
        dropout_p = ckpt.get("dropout_p", dropout_p)
    else:
        sd = ckpt
        nc = num_classes
    tof_model = ToFCNN(num_classes=nc, dropout_p=dropout_p)
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


class ToFFusionMultistreamCNNDeepFinetune(nn.Module):
    """
    Like ToFFusionMultistreamCNN: partial-unfreeze fusion + ToF extractors, deep LN+dropout head,
    trainable missing_rot / missing_tof.
    """

    def __init__(self, fusion_128, tof_feat, num_classes=18, head_hidden_dims=HEAD_HIDDEN_DIMS, head_dropout=HEAD_DROPOUT):
        super().__init__()
        self.fusion_128 = fusion_128
        self.tof_feat = tof_feat
        self.missing_rot = nn.Parameter(torch.zeros(ROT_FEAT_DIM))
        self.missing_tof = nn.Parameter(torch.zeros(TOF_FEAT_DIM))
        in_dim = FUSION_INERTIAL_DIM + TOF_FEAT_DIM
        self.fusion_classifier = make_deep_fusion_head(
            in_dim, num_classes, head_hidden_dims, head_dropout
        )
        self.train_dropout_rot = TRAIN_MODALITY_DROPOUT_ROT
        self.train_dropout_tof = TRAIN_MODALITY_DROPOUT_TOF

    def forward(self, acc, rot, tof_padded, lengths, has_rot, has_tof):
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

        acc_feat = self.fusion_128.acc_backbone(acc)
        rot_feat = torch.zeros(B, ROT_FEAT_DIM, device=device, dtype=dtype)
        if has_rot.any():
            idx = has_rot.nonzero(as_tuple=True)[0]
            rot_feat[idx] = self.fusion_128.rot_backbone(rot[idx])
        if (~has_rot).any():
            idx = (~has_rot).nonzero(as_tuple=True)[0]
            rot_feat[idx] = self.missing_rot.unsqueeze(0).expand(len(idx), -1)

        fr = torch.cat([acc_feat, rot_feat], dim=1)

        tf = torch.zeros(B, TOF_FEAT_DIM, device=device, dtype=dtype)
        if has_tof.any():
            idx = has_tof.nonzero(as_tuple=True)[0]
            tf[idx] = self.tof_feat(tof_padded[idx], lengths[idx])
        if (~has_tof).any():
            idx = (~has_tof).nonzero(as_tuple=True)[0]
            tf[idx] = self.missing_tof.unsqueeze(0).expand(len(idx), -1)

        x = torch.cat([fr, tf], dim=1)
        return self.fusion_classifier(x)

    def fused_features(self, acc, rot, tof_padded, lengths, has_rot, has_tof):
        """
        256-dim fused representation (acc+rot + ToF) before fusion_classifier.
        Use under model.eval() for specialist feature extraction (no modality dropout).
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

        acc_feat = self.fusion_128.acc_backbone(acc)
        rot_feat = torch.zeros(B, ROT_FEAT_DIM, device=device, dtype=dtype)
        if has_rot.any():
            idx = has_rot.nonzero(as_tuple=True)[0]
            rot_feat[idx] = self.fusion_128.rot_backbone(rot[idx])
        if (~has_rot).any():
            idx = (~has_rot).nonzero(as_tuple=True)[0]
            rot_feat[idx] = self.missing_rot.unsqueeze(0).expand(len(idx), -1)

        fr = torch.cat([acc_feat, rot_feat], dim=1)

        tf = torch.zeros(B, TOF_FEAT_DIM, device=device, dtype=dtype)
        if has_tof.any():
            idx = has_tof.nonzero(as_tuple=True)[0]
            tf[idx] = self.tof_feat(tof_padded[idx], lengths[idx])
        if (~has_tof).any():
            idx = (~has_tof).nonzero(as_tuple=True)[0]
            tf[idx] = self.missing_tof.unsqueeze(0).expand(len(idx), -1)

        return torch.cat([fr, tf], dim=1)


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


def collect_param_groups(model: ToFFusionMultistreamCNNDeepFinetune):
    """Head + missing embeddings vs. fusion IMU backbones vs. ToF trunk."""
    head_params = list(model.fusion_classifier.parameters()) + [model.missing_rot, model.missing_tof]

    fusion_bb = []
    for backbone in (model.fusion_128.acc_backbone, model.fusion_128.rot_backbone):
        fusion_bb.extend([p for p in backbone.parameters() if p.requires_grad])

    tof_params = [p for p in model.tof_feat.parameters() if p.requires_grad]

    head_ids = {id(p) for p in head_params}
    fusion_bb = [p for p in fusion_bb if id(p) not in head_ids]
    tof_params = [p for p in tof_params if id(p) not in head_ids]

    return head_params, fusion_bb, tof_params


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
    print(
        f"Fine-tune: head hidden {HEAD_HIDDEN_DIMS}, dropout {HEAD_DROPOUT} | "
        f"LR head {LR_HEAD}, LR backbone {LR_BACKBONE}"
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

    print(f"Loading fusion backbone (partial unfreeze) from {FUSION_CHECKPOINT_PATH}...")
    fusion_128 = make_fusion_finetune_128(num_classes, FUSION_CHECKPOINT_PATH, device)
    print(f"Loading ToF backbone (partial unfreeze) from {TOF_CHECKPOINT_PATH}...")
    tof_feat = make_tof_finetune_extractor(num_classes, TOF_CHECKPOINT_PATH, device)

    model = ToFFusionMultistreamCNNDeepFinetune(
        fusion_128,
        tof_feat,
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
        print(f"No warm-start file at {TRIPLE_WARMSTART_PATH}; training from fusion+ToF init only.")

    head_p, fusion_p, tof_p = collect_param_groups(model)
    optimizer = optim.AdamW(
        [
            {"params": head_p, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
            {"params": fusion_p, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
            {"params": tof_p, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        ]
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head_p)
    n_fusion_bb = sum(p.numel() for p in fusion_p)
    n_tof_bb = sum(p.numel() for p in tof_p)
    print(
        f"Model: {n_total:,} total params | trainable {n_trainable:,} "
        f"(head+missing {n_head:,}, fusion BB {n_fusion_bb:,}, ToF BB {n_tof_bb:,})"
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=100,
        early_stop_patience=15,
        save_path=SAVE_PATH,
    )

    ckpt_best = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_best)
    save_model_and_metadata(
        model,
        optimizer,
        gesture_map,
        history,
        train_seq_ids,
        val_seq_ids,
        test_seq_ids,
        scheduler,
        filepath=SAVE_PATH,
    )
    print(f"\nSaved fine-tuned checkpoint to {SAVE_PATH}")
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
