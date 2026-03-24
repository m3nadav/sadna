"""
Train ToF CNN: 2D CNN on Time-of-Flight sensor data (5 sensors × 8×8) with the same
train/val/test split as the acceleration-based models. Sequences with ToF data missing
for the entire sequence are dropped.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import (
    BFRB_GESTURES,
    RANDOM_STATE,
    apply_label_encoding,
    filter_sequences_by_sensor_validity,
    final_robust_split,
    finalize_training_checkpoint,
    load_train_data,
    set_default_seeds,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()

# ToF measurement range (mm); -1 denotes missing
TOF_MIN, TOF_MAX = 0.0, 249.0


# ---------------------------------------------------------------------------
# ToF columns and drop sequences with ToF missing for entire sequence
# ---------------------------------------------------------------------------
def get_tof_columns(df):
    return [c for c in df.columns if c.startswith("tof_")]


# ---------------------------------------------------------------------------
# ToF CNN model (input is already normalized to [0, 1] in dataset)
# ---------------------------------------------------------------------------
def fuse_pairwise(x):
    """Fuse channels pairwise by summing: (B, C, H, W) -> (B, C//2, H, W)."""
    B, C, H, W = x.shape
    assert C % 2 == 0
    x = x.view(B, C // 2, 2, H, W)
    return x.sum(dim=2)


# Flattened feature size after conv3: 32 * 4 * 3 = 384 (with smaller kernels below)
TOF_CNN_FEAT_SIZE = 384


class ToFCNN(nn.Module):
    """
    2D CNN for ToF sensors (input already normalized to [0, 1] per frame):
    - Conv1: 8 filters 2×4, stride 2×2 → PReLU → Dropout (MC)
    - Fuse pairwise -> 4 channels
    - Conv2: 16 filters 2×2, stride 1×1 → PReLU → Dropout (MC)
    - Fuse pairwise -> 8 channels
    - Conv3: 32 filters 2×2, stride 1×1 (with padding) → PReLU → Dropout (MC)
    - FC: 128, 100, num_classes (PReLU + Dropout after each hidden layer)
    Input: (B, max_T, 5, 8, 8) and lengths (B,) — padded ToF sequences; pooling over T is masked by lengths.

    Dropout layers follow Monte Carlo Dropout (same as nn.Dropout): active in train mode.
    For MC uncertainty at inference, run multiple forwards with model.train() so dropout stays active.
    """

    def __init__(self, num_classes=13, dropout_p=0.1):
        super(ToFCNN, self).__init__()
        self.dropout_p = dropout_p
        # Conv1: 5 -> 8, kernel (2, 4), stride (2, 2) -> (4, 3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=(2, 4), stride=(2, 2)),
            nn.BatchNorm2d(8),
            nn.PReLU(num_parameters=8),
            nn.Dropout(p=dropout_p),
        )
        # After conv1: (B, 8, 4, 3). Fuse pairwise -> (B, 4, 4, 3)
        # Conv2: 4 -> 16, kernel 2×2, stride 1×1 -> (3, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(num_parameters=16),
            nn.Dropout(p=dropout_p),
        )
        # After conv2: (B, 16, 3, 2). Fuse pairwise -> (B, 8, 3, 2)
        # Conv3: 8 -> 32, kernel 2×2, stride 1×1, padding (1, 1) -> (4, 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=2, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(num_parameters=32),
            nn.Dropout(p=dropout_p),
        )
        # After conv3: (B, 32, 4, 3). Flatten -> 384
        self.classifier = nn.Sequential(
            nn.Linear(TOF_CNN_FEAT_SIZE, 128),
            nn.PReLU(num_parameters=128),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 100),
            nn.PReLU(num_parameters=100),
            nn.Dropout(p=dropout_p),
            nn.Linear(100, num_classes),
        )

    def forward(self, tof_padded, lengths):
        """
        tof_padded: (B, max_T, 5, 8, 8), values in [0, 1]; padding is 0.
        lengths: (B,) actual sequence length per batch item.
        """
        # Ensure input is in [0, 1] and finite (avoids NaN from bad data or padding)
        tof_padded = torch.clamp(tof_padded, 0.0, 1.0)
        lengths = lengths.clamp(min=1)  # avoid div-by-zero in masked mean
        B, max_T, C, H, W = tof_padded.shape
        # Run 2D CNN on each frame
        x = tof_padded.view(B * max_T, C, H, W)
        x = self.conv1(x)       # (B*max_T, 8, 4, 3)
        x = fuse_pairwise(x)    # (B*max_T, 4, 4, 3)
        x = self.conv2(x)       # (B*max_T, 16, 3, 2)
        x = fuse_pairwise(x)    # (B*max_T, 8, 3, 2)
        x = self.conv3(x)       # (B*max_T, 32, 4, 3)
        x = torch.flatten(x, 1) # (B*max_T, 384)
        x = x.view(B, max_T, -1)  # (B, max_T, 384)
        # Masked mean over time (ignore padding)
        mask = torch.arange(max_T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_T)
        mask = mask.unsqueeze(-1).to(x.dtype)  # (B, max_T, 1)
        lengths_clamp = lengths.unsqueeze(1).clamp(min=1).to(x.dtype)  # (B, 1)
        x = (x * mask).sum(dim=1) / lengths_clamp  # (B, 384)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset: variable-length (T, 5, 8, 8) per sequence, normalized to [0, 1]
# ---------------------------------------------------------------------------
def _tof_frame_to_5x8x8(tof_flat):
    """Convert 320-dim ToF vector to (5, 8, 8). tof_flat: (320,) or (T, 320)."""
    if tof_flat.ndim == 1:
        tof_flat = tof_flat.reshape(1, -1)
    T = tof_flat.shape[0]
    # Order: tof_1_v0..tof_1_v63, tof_2_v0..tof_2_v63, ...
    x = tof_flat.reshape(T, 5, 64)
    x = x.reshape(T, 5, 8, 8)
    return x


def _normalize_tof_01(x):
    """Scale ToF values from [TOF_MIN, TOF_MAX] to [0, 1]. Replaces NaN/Inf so output is always finite."""
    x = np.asarray(x, dtype=np.float64)
    # Replace invalid values before clip (NaN/Inf would otherwise propagate)
    x = np.nan_to_num(x, nan=TOF_MAX, posinf=TOF_MAX, neginf=TOF_MIN)
    x = np.clip(x, TOF_MIN, TOF_MAX)
    denom = float(TOF_MAX - TOF_MIN)
    if denom <= 0:
        denom = 1.0
    out = (x - TOF_MIN) / denom
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


class ToFSequenceDataset(Dataset):
    """Sequence-level dataset: (T, 5, 8, 8) per sequence, normalized to [0, 1]; one label per sequence."""

    def __init__(self, dataframe, tof_cols=None):
        self.df = dataframe.reset_index(drop=True)
        if tof_cols is None:
            tof_cols = get_tof_columns(self.df)
        self.tof_cols = tof_cols
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        tof = data[self.tof_cols].values.astype(np.float64)  # (T, 320)
        # -1 and NaN mean no reading (treat as very far); replace with max range before normalizing
        tof[tof == -1] = TOF_MAX
        tof[np.isnan(tof)] = TOF_MAX
        tof = np.nan_to_num(tof, nan=TOF_MAX, posinf=TOF_MAX, neginf=TOF_MIN)
        tof_per_frame = _tof_frame_to_5x8x8(tof)  # (T, 5, 8, 8)
        # Normalize to [0, 1] before training (guaranteed finite and in range)
        tof_per_frame = _normalize_tof_01(tof_per_frame)
        x = torch.tensor(tof_per_frame, dtype=torch.float32)  # (T, 5, 8, 8)
        x = torch.clamp(x, 0.0, 1.0)  # ensure no stray value slips through
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return x, label


def collate_tof_only(batch):
    """Pad ToF-only sequences (T,5,8,8) to (B,max_T,5,8,8), lengths, and labels — no other sensors."""
    tof_list, labels = zip(*batch)
    tof_padded = pad_sequence(tof_list, batch_first=True, padding_value=0.0)  # (B, max_T, 5, 8, 8)
    lengths = torch.tensor([t.shape[0] for t in tof_list], dtype=torch.long)
    labels_stacked = torch.stack(labels)
    return tof_padded, lengths, labels_stacked


collate_tof = collate_tof_only


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
    print(f"Split (before dropping missing ToF): train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    tof_cols = get_tof_columns(train_df)
    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = filter_sequences_by_sensor_validity(
        trainset_df, tof_cols=tof_cols
    )
    valset_df, val_seq_ids = filter_sequences_by_sensor_validity(valset_df, tof_cols=tof_cols)
    testset_df, test_seq_ids = filter_sequences_by_sensor_validity(testset_df, tof_cols=tof_cols)

    print(
        f"Dropped sequences with ToF missing for entire sequence: "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_ds = ToFSequenceDataset(trainset_df, tof_cols)
    val_ds = ToFSequenceDataset(valset_df, tof_cols)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_tof_only,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_tof_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToFCNN(num_classes=num_classes)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/ToF_CNN.pth"

    def train_step(batch):
        tof_padded, lengths, labels = batch
        tof_padded = tof_padded.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        outputs = model(tof_padded, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_acc = (outputs.argmax(1) == labels).float().mean()
        return loss.item(), batch_acc.item()

    def val_step(batch):
        tof_padded, lengths, v_labels = batch
        tof_padded = tof_padded.to(device)
        lengths = lengths.to(device)
        v_labels = v_labels.to(device)
        val_outputs = model(tof_padded, lengths)
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


if __name__ == "__main__":
    model, history, le = main()
