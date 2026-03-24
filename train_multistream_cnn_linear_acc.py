"""
Train Multistream CNN on world-frame linear acceleration (gravity-compensated).
Uses IMUProcessor to rotate body-frame acc to world frame and subtract gravity;
then same architecture as MultistreamCNNInertialNet (acc-only) and same training flow.
Same train/val/test split; drop sequences with missing quaternion data (as in rot model).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumulative_trapezoid

from utils import (
    ACC_COLS,
    BFRB_GESTURES,
    ROT_COLS,
    RANDOM_STATE,
    ZSCORE_EPS,
    ZScoreNormalize,
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


# ---------------------------------------------------------------------------
# IMUProcessor from Project.ipynb: body-frame acc -> world-frame linear acc
# ---------------------------------------------------------------------------
class IMUProcessor:
    """Handles coordinate transformations and integration of IMU data."""

    GRAVITY_MAGNITUDE = 9.80665

    def __init__(self, df: pd.DataFrame, dt: float = 1.0, silent: bool = False):
        self.df = df
        self.dt = dt
        self.silent = silent
        self._process_data()

    def _process_data(self):
        quat_cols = ROT_COLS
        quats = self.df[quat_cols].values
        norms = np.linalg.norm(quats, axis=1)
        valid_mask = norms > 0

        acc_body_full = self.df[ACC_COLS].values
        self.acc_linear = acc_body_full.copy()

        if not np.all(valid_mask) and not self.silent:
            print(
                "-" * 80,
                f"Warning: Found {np.sum(~valid_mask)} zero-norm quaternions.",
                "Using body-frame acceleration for invalid rows.",
                "-" * 80,
                sep="\n",
            )

        if np.any(valid_mask):
            quats_valid = quats[valid_mask]
            norms_valid = norms[valid_mask]
            quats_valid = quats_valid / norms_valid[:, np.newaxis]
            acc_body_valid = acc_body_full[valid_mask]
            rotations = R.from_quat(quats_valid)
            acc_world = rotations.apply(acc_body_valid)
            acc_linear_valid = acc_world - np.array([0, 0, self.GRAVITY_MAGNITUDE])
            self.acc_linear[valid_mask] = acc_linear_valid

        self.velocity = cumulative_trapezoid(self.acc_linear, dx=self.dt, axis=0, initial=0)
        self.position = cumulative_trapezoid(self.velocity, dx=self.dt, axis=0, initial=0)
        self.time_axis = np.arange(len(self.df)) * self.dt
        self.phases = self.df["phase"].values
        self.acc_mag = np.linalg.norm(self.acc_linear, axis=1)


def compute_linear_acc_train_stats(trainset_df, dt=1.0, silent=True):
    """Mean and population std (ddof=0) of IMUProcessor.acc_linear over all training timesteps."""
    parts = []
    for _, data in trainset_df.groupby("sequence_id"):
        processor = IMUProcessor(data, dt=dt, silent=silent)
        parts.append(processor.acc_linear.astype(np.float32))
    if not parts:
        raise ValueError("trainset_df has no sequences for linear-acc stats")
    all_lin = np.concatenate(parts, axis=0)
    mean = all_lin.mean(axis=0).astype(np.float32)
    std = all_lin.std(axis=0, ddof=0).astype(np.float32)
    std = np.maximum(std, ZSCORE_EPS)
    return mean, std


# ---------------------------------------------------------------------------
# Model: Z-score (train stats on linear acc) + MultistreamCNNInertialNet (acc-only)
# ---------------------------------------------------------------------------
class MultistreamCNNInertialNet(nn.Module):
    """Same architecture as acc-only Multistream CNN; input is world-frame linear acc (3, T)."""

    def __init__(self, num_classes=18, norm_mean=None, norm_std=None):
        super(MultistreamCNNInertialNet, self).__init__()
        if norm_mean is None or norm_std is None:
            raise ValueError(
                "norm_mean and norm_std are required (from compute_linear_acc_train_stats on train split)"
            )
        self.norm = ZScoreNormalize(norm_mean, norm_std)
        self.conv1_block = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=4, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=6, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, acc):
        x = self.norm(acc)
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset: per sequence run IMUProcessor, return (acc_linear (3,T), label)
# ---------------------------------------------------------------------------
class LinearAccSequenceDataset(Dataset):
    """
    Sequence-level dataset: reads ACC_COLS + ROT_COLS per sequence, runs IMUProcessor to obtain
    world-frame linear acceleration, returns only that (3, T) tensor and the label (no raw rot in output).
    """

    def __init__(self, dataframe, dt=1.0, silent=True):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))
        self.dt = dt
        self.silent = silent

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, data = self.sequences[idx]
        processor = IMUProcessor(data, dt=self.dt, silent=self.silent)
        acc_linear = processor.acc_linear.astype(np.float32)  # (T, 3) world linear acc
        acc = torch.tensor(acc_linear.T, dtype=torch.float32)  # (3, T)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, label


def collate_world_linear_acc_only(batch):
    """Pad world-frame linear acc (3,T) only; stack labels — quaternion used only inside __getitem__."""
    accs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, labels_stacked


collate_fn = collate_world_linear_acc_only


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
    print(f"Split (before dropping missing quaternion): train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = filter_sequences_by_sensor_validity(
        trainset_df, rot_cols=ROT_COLS
    )
    valset_df, val_seq_ids = filter_sequences_by_sensor_validity(valset_df, rot_cols=ROT_COLS)
    testset_df, test_seq_ids = filter_sequences_by_sensor_validity(testset_df, rot_cols=ROT_COLS)

    print(
        f"Dropped sequences with missing quaternions: "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_ds = LinearAccSequenceDataset(trainset_df, silent=True)
    val_ds = LinearAccSequenceDataset(valset_df, silent=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_world_linear_acc_only,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_world_linear_acc_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lin_mean, lin_std = compute_linear_acc_train_stats(trainset_df, dt=1.0, silent=True)
    model = MultistreamCNNInertialNet(num_classes=num_classes, norm_mean=lin_mean, norm_std=lin_std)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Multistream_CNN_linear_acc.pth"

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
    )
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
