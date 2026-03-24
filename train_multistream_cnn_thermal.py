"""
Train Multistream CNN on thermal (thermopile) data only: thm_1 … thm_5.
Same data loading, train/val/test split, and training loop as train_multistream_cnn.py /
train_multistream_cnn_rot.py. Training drops sequences with no temporal variation in any thermal
channel (see utils.filter_sequences_by_sensor_validity); analyze scripts may still use
drop_sequences_with_no_thermal (at least one in-range reading).
Inputs are imputed (invalid -> train mean per channel) and z-score normalized (train stats).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from utils import (
    BFRB_GESTURES,
    RANDOM_STATE,
    ZSCORE_EPS,
    apply_label_encoding,
    filter_sequences_by_sensor_validity,
    final_robust_split,
    finalize_training_checkpoint,
    load_train_data,
    make_supervised_single_tensor_batch_steps,
    set_default_seeds,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()

# Plausible °C band for MLX90632-style readings (tunable after inspecting analyze_thermal_columns)
THM_MIN_C = 5.0
THM_MAX_C = 60.0
DROPOUT_P = 0.2


# ---------------------------------------------------------------------------
# Thermal columns, validity, analysis, sequence drop
# ---------------------------------------------------------------------------
def get_thm_columns(df):
    """Return sorted thermal column names (thm_*)."""
    cols = [c for c in df.columns if c.startswith("thm_")]
    return sorted(cols)


def is_valid_thm_array(values, thm_min=THM_MIN_C, thm_max=THM_MAX_C):
    """Vectorized validity mask: finite, in [thm_min, thm_max], not -1 (missing sentinel)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) & (v != -1.0) & (v >= thm_min) & (v <= thm_max)
    return ok


def analyze_thermal_columns(df, thm_cols):
    """Print thermal column stats and sequence-level coverage."""
    print("\n--- Thermal sensor analysis ---")
    print(f"Columns ({len(thm_cols)}): {thm_cols}")
    if not thm_cols:
        print("No thm_* columns found.")
        return

    for c in thm_cols:
        s = df[c]
        nan_frac = float(s.isna().mean())
        vals = s.values
        n_minus_one = int(np.sum(vals == -1)) if vals.dtype != object else 0
        finite = vals[np.isfinite(vals.astype(np.float64))]
        print(f"  {c}: NaN fraction={nan_frac:.4f}", end="")
        if n_minus_one > 0:
            print(f", count==-1: {n_minus_one}", end="")
        print()
        if finite.size > 0:
            print(
                f"      finite min={finite.min():.4f} max={finite.max():.4f} "
                f"q01={np.quantile(finite, 0.01):.4f} q50={np.quantile(finite, 0.5):.4f} "
                f"q99={np.quantile(finite, 0.99):.4f}"
            )

    n_seq = df["sequence_id"].nunique()
    n_with_data = 0
    n_no_data = 0
    for _, g in df.groupby("sequence_id", sort=False):
        if is_valid_thm_array(g[thm_cols].values).any():
            n_with_data += 1
        else:
            n_no_data += 1
    print(f"Sequences: total={n_seq}, with at least one valid thermal reading={n_with_data}, all-invalid={n_no_data}")
    print("--- End thermal analysis ---\n")


def drop_sequences_with_no_thermal(df, thm_cols, thm_min=THM_MIN_C, thm_max=THM_MAX_C):
    """
    Keep sequences that have at least one valid thermal reading (any sensor, any frame).
    Returns filtered dataframe and list of kept sequence_id.
    """
    if not thm_cols:
        return df.iloc[0:0].copy(), []

    valid_seq_ids = []
    for seq_id, g in df.groupby("sequence_id", sort=False):
        if is_valid_thm_array(g[thm_cols].values, thm_min, thm_max).any():
            valid_seq_ids.append(seq_id)
    return df[df["sequence_id"].isin(valid_seq_ids)].reset_index(drop=True), valid_seq_ids


def compute_train_thermal_zscore_params(train_df, thm_cols, thm_min=THM_MIN_C, thm_max=THM_MAX_C):
    """
    Per-channel mean and std from training rows using only valid values.
    Imputation uses the same per-channel mean (raw °C).
    """
    means = np.zeros(len(thm_cols), dtype=np.float64)
    stds = np.ones(len(thm_cols), dtype=np.float64)
    for i, c in enumerate(thm_cols):
        vals = train_df[c].values.astype(np.float64)
        mask = is_valid_thm_array(vals, thm_min, thm_max)
        if mask.any():
            means[i] = np.mean(vals[mask])
            s = np.std(vals[mask], ddof=0)
            stds[i] = float(s) if s > ZSCORE_EPS else 1.0
        else:
            means[i] = 0.0
            stds[i] = 1.0
    return means, stds


# ---------------------------------------------------------------------------
# Model: Z-score + MultistreamCNNThermalNet (Conv2d + dropout)
# ---------------------------------------------------------------------------
class ZScoreNormalize(nn.Module):
    """Per-channel z-score: (x - mean) / (std + eps). Mean/std shaped for (B, C, T)."""

    def __init__(self, mean, std, eps=ZSCORE_EPS):
        super().__init__()
        # (1, C, 1) broadcasts over batch and time
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


class MultistreamCNNThermalNet(nn.Module):
    """
    Thermal-only CNN: (B,5,T) -> z-score -> (B,1,5,T) Conv2d blocks + dropout + FC with dropout.
    conv1 uses kernel (5,4) to cover all five thermopile channels (mirrors rot's (4,4)).
    """

    def __init__(self, num_classes=18, mean=None, std=None, dropout_p=DROPOUT_P):
        super().__init__()
        self.dropout_p = dropout_p
        self.norm = ZScoreNormalize(mean, std)

        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 4), stride=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 6), stride=(2, 2), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(p=dropout_p)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, thm):
        x = self.norm(thm).unsqueeze(1)
        x = self.conv1_block(x)
        x = self.drop(x)
        x = self.conv2_block(x)
        x = self.drop(x)
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# Dataset and collate
# ---------------------------------------------------------------------------
class ThermalSequenceDataset(Dataset):
    """Sequence-level (5, T) thermal, invalid imputed to train mean (raw), z-score in model."""

    def __init__(self, dataframe, thm_cols, train_mean_raw):
        self.df = dataframe.reset_index(drop=True)
        self.thm_cols = thm_cols
        self.train_mean_raw = np.asarray(train_mean_raw, dtype=np.float64).reshape(len(thm_cols))
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        raw = data[self.thm_cols].values.astype(np.float64)  # (T, 5)
        mask = is_valid_thm_array(raw)
        imputed = raw.copy()
        for c in range(raw.shape[1]):
            col_mask = mask[:, c]
            if not col_mask.all():
                imputed[~col_mask, c] = self.train_mean_raw[c]
        thm = torch.tensor(imputed.T, dtype=torch.float32)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return thm, label


def make_thermal_collate_fn(train_mean_raw):
    """
    Build a collate that batches thermal-only tensors (B, 5, max_T) and labels.
    Padding uses per-channel train mean (raw °C) so z-score maps pads to ~0.
    """

    tm = torch.as_tensor(train_mean_raw, dtype=torch.float32).view(5, 1)

    def collate_thermal_only(batch):
        thms, labels = zip(*batch)
        max_t = max(t.shape[1] for t in thms)
        b = len(thms)
        out = tm.expand(5, max_t).unsqueeze(0).expand(b, 5, max_t).clone()
        for i, t in enumerate(thms):
            ti = t.shape[1]
            out[i, :, :ti] = t
        return out, torch.stack(labels)

    return collate_thermal_only


make_collate_fn = make_thermal_collate_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_df = load_train_data()
    thm_cols = get_thm_columns(train_df)
    if not thm_cols:
        raise ValueError("No thm_* columns in train.csv; cannot train thermal model.")

    analyze_thermal_columns(train_df, thm_cols)

    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)
    print(
        f"Split (before dropping static thermal sequences): train {len(trainset_df)} rows, "
        f"val {len(valset_df)} rows, test {len(testset_df)} rows"
    )

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = filter_sequences_by_sensor_validity(
        trainset_df, thm_cols=thm_cols
    )
    valset_df, val_seq_ids = filter_sequences_by_sensor_validity(valset_df, thm_cols=thm_cols)
    testset_df, test_seq_ids = filter_sequences_by_sensor_validity(testset_df, thm_cols=thm_cols)

    print(
        f"Dropped sequences with no thermal variation (per-channel nunique<=1): "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    if len(trainset_df) == 0:
        raise RuntimeError(
            "Training set has no rows after dropping sequences without thermal variation."
        )

    train_mean_raw, train_std_raw = compute_train_thermal_zscore_params(trainset_df, thm_cols)
    print(f"Z-score train stats per channel {thm_cols}: mean={train_mean_raw}, std={train_std_raw}")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_ds = ThermalSequenceDataset(trainset_df, thm_cols, train_mean_raw)
    val_ds = ThermalSequenceDataset(valset_df, thm_cols, train_mean_raw)
    collate = make_thermal_collate_fn(train_mean_raw)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultistreamCNNThermalNet(
        num_classes=num_classes,
        mean=train_mean_raw,
        std=train_std_raw,
        dropout_p=DROPOUT_P,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Multistream_CNN_thermal_only.pth"
    thm_bounds = (THM_MIN_C, THM_MAX_C)

    train_step, val_step = make_supervised_single_tensor_batch_steps(
        model, criterion, optimizer, device
    )

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
        extra_checkpoint_keys={
            "thm_cols": thm_cols,
            "thm_mean": train_mean_raw.tolist(),
            "thm_std": train_std_raw.tolist(),
            "thm_valid_bounds": thm_bounds,
            "dropout_p": DROPOUT_P,
        },
    )
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
