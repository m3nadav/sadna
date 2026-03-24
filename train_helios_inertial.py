"""
Train HeliosInertialNet (initial model architecture) using the same imports,
dataset, train/val/test splits and functions as in Project.ipynb.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils import (
    ACC_COLS,
    BFRB_GESTURES,
    RANDOM_STATE,
    ZSCORE_EPS,
    ZScoreNormalize,
    apply_label_encoding,
    compute_acc_zscore_stats,
    final_robust_split,
    finalize_training_checkpoint,
    load_train_data,
    make_supervised_single_tensor_batch_steps,
    set_default_seeds,
)

# Optional: match notebook display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()

# ---------------------------------------------------------------------------
# Model: Z-score (train stats) + HeliosInertialNet (same as Project.ipynb)
# ---------------------------------------------------------------------------
class HeliosInertialNet(nn.Module):
    """Initial model architecture from Project.ipynb (acc-only, Conv + LSTM + FC)."""

    def __init__(self, num_classes=18, norm_mean=None, norm_std=None):
        super(HeliosInertialNet, self).__init__()
        if norm_mean is None or norm_std is None:
            raise ValueError("norm_mean and norm_std are required (from compute_acc_zscore_stats on train split)")
        self.input_norm = ZScoreNormalize(norm_mean, norm_std)
        # (B,3,T) -> (B,1,3,T): 2D conv over acc axes x time, then (B,64,T) for remaining 1D stack
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, acc):
        x = self.input_norm(acc).unsqueeze(1)
        x = self.conv1(x).squeeze(2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x_avg = self.global_avg_pool(x).squeeze(-1)
        x_max = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([x_avg, x_max], dim=1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Dataset and collate (same as Project.ipynb)
# ---------------------------------------------------------------------------
class HeliosDataset(Dataset):
    """Sequence-level dataset: body-frame acceleration (3, T) and label only (HeliosInertialNet input)."""

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        acc = torch.tensor(data[ACC_COLS].values, dtype=torch.float32).T
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, label


def helios_collate_fn(batch):
    """Batch (B,3,T) padded acc and labels — acceleration only for HeliosInertialNet."""
    accs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, labels_stacked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load data
    train_df = load_train_data()

    # 2. BFRB column and split (same as Project.ipynb)
    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)
    print(f"Split: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    # 3. Label encoding
    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )
    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    # 4. Sequence IDs for train/val/test (same as Project.ipynb, for checkpoint)
    train_seq_ids = trainset_df["sequence_id"].unique().tolist()
    val_seq_ids = valset_df["sequence_id"].unique().tolist()
    test_seq_ids = testset_df["sequence_id"].unique().tolist()

    # 5. Datasets and loaders (same as Project.ipynb)
    train_ds = HeliosDataset(trainset_df)
    val_ds = HeliosDataset(valset_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=helios_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=helios_collate_fn,
    )

    # 6. Model, criterion, optimizer, scheduler (same as Project.ipynb)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    acc_mean, acc_std = compute_acc_zscore_stats(trainset_df)
    model = HeliosInertialNet(num_classes=num_classes, norm_mean=acc_mean, norm_std=acc_std)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    # 7. Train (saves best state_dict to save_path during training)
    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/helios_inertial_net_2d_conv_z_score_normalized.pth"

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
