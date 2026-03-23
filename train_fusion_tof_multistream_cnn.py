"""
Train a fusion model: frozen acc+rot (128-dim) and ToF (128-dim) features, with learnable
placeholders when rotation or ToF is missing for a sequence. Acceleration is always used.

Sequences are NOT dropped for missing quaternion or missing ToF; per-sequence flags select
real backbone features vs. trainable missing embeddings (64-dim for rot half, 128-dim for ToF).

Optional training-time modality dropout simulates missing sensors on otherwise-complete sequences
so missing embeddings stay useful (important when the dataset has few fully ToF-missing sequences).

Default paths: Fusion_Multistream_CNN.pth, ToF_CNN.pth.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

FUSION_CHECKPOINT_PATH = "models/deep_learning_models/Fusion_Multistream_CNN.pth"
TOF_CHECKPOINT_PATH = "models/deep_learning_models/ToF_CNN.pth"

# Extra stochastic “missing sensor” during training (only applied when sensor is actually present)
TRAIN_MODALITY_DROPOUT_ROT = 0.1
TRAIN_MODALITY_DROPOUT_TOF = 0.1

# Feature dimensions (must match frozen backbones)
ROT_FEAT_DIM = 64
TOF_FEAT_DIM = 128
FUSION_INERTIAL_DIM = 128  # acc 64 + rot 64

from train_multistream_cnn import (
    load_train_data,
    final_robust_split,
    apply_label_encoding,
    BFRB_GESTURES,
    MultistreamCNNInertialNet,
    compute_acc_zscore_stats,
)
from train_multistream_cnn_rot import MultistreamCNNRotNet
from train_fusion_multistream_cnn import FusionMultistreamCNN, ROT_COLS
from train_tof_cnn import (
    ToFCNN,
    get_tof_columns,
    _tof_frame_to_5x8x8,
    _normalize_tof_01,
    TOF_MAX,
)


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
# Frozen 128-dim extractor from trained Fusion_Multistream_CNN
# ---------------------------------------------------------------------------
class FusionFrozen128(nn.Module):
    """Uses frozen acc_backbone + rot_backbone from a trained FusionMultistreamCNN; output (B, 128)."""

    def __init__(self, fusion_model: FusionMultistreamCNN):
        super().__init__()
        self.acc_backbone = fusion_model.acc_backbone
        self.rot_backbone = fusion_model.rot_backbone
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, acc, rot):
        acc_feat = self.acc_backbone(acc)
        rot_feat = self.rot_backbone(rot)
        return torch.cat([acc_feat, rot_feat], dim=1)


def make_fusion_frozen_128(num_classes, checkpoint_path, device, trainset_df=None):
    """Load Fusion_Multistream_CNN and return frozen 128-dim feature extractor."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    acc_sd = {k[len("acc_backbone."):]: v for k, v in state.items() if k.startswith("acc_backbone.")}
    if isinstance(ckpt, dict) and "input_zscore_mean" in ckpt and "input_zscore_std" in ckpt:
        m = np.asarray(ckpt["input_zscore_mean"], dtype=np.float32)
        s = np.asarray(ckpt["input_zscore_std"], dtype=np.float32)
    elif "norm.mean" in acc_sd:
        m = acc_sd["norm.mean"].detach().cpu().numpy()
        s = acc_sd["norm.std"].detach().cpu().numpy()
    elif trainset_df is not None:
        m, s = compute_acc_zscore_stats(trainset_df)
    else:
        raise ValueError(
            "Fusion checkpoint missing input_zscore_* or acc_backbone norm stats; pass trainset_df or retrain."
        )
    acc_backbone = MultistreamCNNInertialNet(num_classes=num_classes, norm_mean=m, norm_std=s)
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
    fusion.load_state_dict(state)
    extractor = FusionFrozen128(fusion)
    return extractor.to(device)


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
# Fusion head with missing-modality placeholders
# ---------------------------------------------------------------------------
class ToFFusionMultistreamCNN(nn.Module):
    """
    Frozen acc/rot and ToF backbones; trainable missing_rot (64), missing_tof (128), and head.
    has_rot[i]=False -> use missing_rot instead of rot_backbone for sample i.
    has_tof[i]=False -> use missing_tof instead of tof backbone for sample i.
    """

    def __init__(self, fusion_128, tof_feat, num_classes=18, hidden_dim=64):
        super(ToFFusionMultistreamCNN, self).__init__()
        self.fusion_128 = fusion_128
        self.tof_feat = tof_feat
        self.missing_rot = nn.Parameter(torch.zeros(ROT_FEAT_DIM))
        self.missing_tof = nn.Parameter(torch.zeros(TOF_FEAT_DIM))
        self.fusion_classifier = nn.Sequential(
            nn.Linear(FUSION_INERTIAL_DIM + TOF_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.train_dropout_rot = TRAIN_MODALITY_DROPOUT_ROT
        self.train_dropout_tof = TRAIN_MODALITY_DROPOUT_TOF

    def forward(self, acc, rot, tof_padded, lengths, has_rot, has_tof):
        """
        has_rot, has_tof: (B,) bool tensors — True if real sensor data should be used.
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

        x = torch.cat([fr, tf], dim=1)
        return self.fusion_classifier(x)


# ---------------------------------------------------------------------------
# Dataset: acc, rot, ToF, flags, label
# ---------------------------------------------------------------------------
def _sequence_has_valid_rot(data):
    return not data[ROT_COLS].isna().any().any()


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
        acc = torch.tensor(data[["acc_x", "acc_y", "acc_z"]].values, dtype=torch.float32).T
        rot_np = data[ROT_COLS].values.T.astype(np.float32)
        rot = torch.tensor(np.nan_to_num(rot_np, nan=0.0), dtype=torch.float32)
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
        return acc, rot, tof_tensor, has_rot, has_tof, label


def collate_acc_rot_tof(batch):
    accs, rots, tofs, has_rots, has_tofs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
    tof_padded = pad_sequence(list(tofs), batch_first=True, padding_value=0.0)
    lengths = torch.tensor([t.shape[0] for t in tofs], dtype=torch.long)
    has_rot_b = torch.stack(list(has_rots))
    has_tof_b = torch.stack(list(has_tofs))
    labels_stacked = torch.stack(labels)
    return acc_padded, rot_padded, tof_padded, lengths, has_rot_b, has_tof_b, labels_stacked


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def save_model_and_metadata(
    model,
    optimizer,
    gesture_map,
    history,
    train_seq_ids,
    val_seq_ids,
    test_seq_ids,
    scheduler,
    filepath="model_checkpoint.pth",
):
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "gesture_map": gesture_map,
        "history": history,
        "num_classes": len(gesture_map),
        "train_sequence_ids": train_seq_ids,
        "val_sequence_ids": val_seq_ids,
        "test_sequence_ids": test_seq_ids,
    }
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to {filepath}")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=100,
    early_stop_patience=15,
    save_path="best_fusion_tof_multistream_cnn.pth",
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for batch in train_loader:
            acc, rot, tof_padded, lengths, has_rot, has_tof, labels = batch
            acc = acc.to(device)
            rot = rot.to(device)
            tof_padded = tof_padded.to(device)
            lengths = lengths.to(device)
            has_rot = has_rot.to(device)
            has_tof = has_tof.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(acc, rot, tof_padded, lengths, has_rot, has_tof)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            batch_acc = (outputs.argmax(1) == labels).float().mean()
            epoch_loss += loss.item()
            epoch_acc += batch_acc.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches

        model.eval()
        val_loss = 0
        val_acc = 0
        val_batches = 0
        with torch.no_grad():
            for v_acc, v_rot, v_tof, v_len, v_hr, v_ht, v_labels in val_loader:
                v_acc = v_acc.to(device)
                v_rot = v_rot.to(device)
                v_tof = v_tof.to(device)
                v_len = v_len.to(device)
                v_hr = v_hr.to(device)
                v_ht = v_ht.to(device)
                v_labels = v_labels.to(device)
                val_outputs = model(v_acc, v_rot, v_tof, v_len, v_hr, v_ht)
                val_loss += criterion(val_outputs, v_labels).item()
                val_acc += (val_outputs.argmax(1) == v_labels).float().mean().item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc * 100)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc * 100)
        history["lr"].append(current_lr)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"| Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc*100:.2f}% "
            f"| Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc*100:.2f}% "
            f"| LR: {current_lr:.6f} | Best Val: {best_val_acc*100:.2f}%"
        )

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    return history


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

    print(f"Loading frozen fusion backbone from {FUSION_CHECKPOINT_PATH}...")
    fusion_128 = make_fusion_frozen_128(num_classes, FUSION_CHECKPOINT_PATH, device, trainset_df)
    print(f"Loading frozen ToF backbone from {TOF_CHECKPOINT_PATH}...")
    tof_feat = make_tof_feature_extractor(num_classes, TOF_CHECKPOINT_PATH, device)

    model = ToFFusionMultistreamCNN(fusion_128, tof_feat, num_classes=num_classes, hidden_dim=64)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total parameters, {n_trainable:,} trainable (head + missing embeddings)")

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
        save_path=save_path,
    )

    ckpt_best = torch.load(save_path, map_location=device, weights_only=False)
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
        filepath=save_path,
    )
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
