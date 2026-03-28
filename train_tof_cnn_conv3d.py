"""
Train ToF CNN (Conv3d): 3D CNN over (T, H, W) with 5 sensor channels on Time-of-Flight
sensor data (5 sensors × 8×8). Uses length-sorted batching to minimise padding. Same
train/val/test split as the acceleration-based models. Sequences with ToF data missing
for the entire sequence are dropped.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
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
# ToF Conv3d model (input is already normalized to [0, 1] in dataset)
# ---------------------------------------------------------------------------

# Flattened spatial feature size after conv3: 32 * 4 * 3 = 384
TOF_CNN_FEAT_SIZE = 384


class ToFCNN3D(nn.Module):
    """
    3D CNN for ToF sensors (input already normalized to [0, 1] per frame).
    Convolves jointly over (T, H, W) with 5 input channels (one per sensor).
    Spatial feature extractor:
    - Conv3d block 1: 5 → 16, kernel (3,2,4), stride (1,2,2), pad (1,0,0) → (T, 4, 3)
    - Conv3d block 2: 16 → 32, kernel (3,2,2), stride (1,1,1), pad (1,0,0) → (T, 3, 2)
    - Conv3d block 3: 32 → 32, kernel (3,2,2), stride (1,1,1), pad (1,1,1) → (T, 4, 3)
    Temporal aggregation:
    - Flatten spatial per time-step → (B, T, 384)
    - Masked mean over T (ignores padded positions)
    Classifier:
    - FC: 384 → 128 → 100 → num_classes (PReLU + Dropout after each hidden layer)
    Input: (B, max_T, 5, 8, 8) and lengths (B,) — padded ToF sequences.

    Dropout layers follow Monte Carlo Dropout (same as nn.Dropout): active in train mode.
    For MC uncertainty at inference, run multiple forwards with model.train() so dropout stays active.
    """

    def __init__(self, num_classes=13, dropout_p=0.1):
        super(ToFCNN3D, self).__init__()
        self.dropout_p = dropout_p
        # Conv3d block 1: (B, 5, T, 8, 8) -> (B, 16, T, 4, 3)
        self.conv1 = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=(3, 2, 4), stride=(1, 2, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(16),
            nn.PReLU(num_parameters=16),
            nn.Dropout(p=dropout_p),
        )
        # Conv3d block 2: (B, 16, T, 4, 3) -> (B, 32, T, 3, 2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 2, 2), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.PReLU(num_parameters=32),
            nn.Dropout(p=dropout_p),
        )
        # Conv3d block 3: (B, 32, T, 3, 2) -> (B, 32, T, 4, 3)
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 2, 2), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.PReLU(num_parameters=32),
            nn.Dropout(p=dropout_p),
        )
        # After conv3: spatial 4×3, 32 channels → flatten → 384 per time-step
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
        tof_padded = torch.clamp(tof_padded, 0.0, 1.0)
        lengths = lengths.clamp(min=1)
        B, max_T, C, H, W = tof_padded.shape
        # (B, T, 5, 8, 8) -> (B, 5, T, 8, 8) for Conv3d
        x = tof_padded.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)  # (B, 16, T, 4, 3)
        x = self.conv2(x)  # (B, 32, T, 3, 2)
        x = self.conv3(x)  # (B, 32, T, 4, 3)
        # Flatten spatial dims per temporal position
        B, C_out, T_out, H_out, W_out = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, T_out, -1)  # (B, T, 384)
        # Masked mean over time (ignore padding)
        mask = torch.arange(T_out, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(x.dtype)  # (B, T, 1)
        x = (x * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1).to(x.dtype)  # (B, 384)
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
        self.seq_lengths = [len(data) for _, data in self.sequences]

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
# Length-sorted batch sampler (minimises padding within each batch)
# ---------------------------------------------------------------------------
class LengthSortedBatchSampler(Sampler):
    """Sort sequences by length and group into contiguous batches so that
    sequences within each batch have similar lengths, reducing padding waste.
    Batch order is shuffled each epoch for training; kept sequential for eval."""

    def __init__(self, lengths, batch_size, shuffle_batches=True):
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [
            sorted_indices[i : i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]
        self.shuffle_batches = shuffle_batches

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle_batches:
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self.batches)


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
    train_sampler = LengthSortedBatchSampler(
        train_ds.seq_lengths, batch_size=16, shuffle_batches=True
    )
    val_sampler = LengthSortedBatchSampler(
        val_ds.seq_lengths, batch_size=16, shuffle_batches=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_tof_only,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=collate_tof_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToFCNN3D(num_classes=num_classes)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=7
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/ToF_CNN_Conv3D.pth"

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
        num_epochs=50,
        early_stop_patience=10,
    )
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
