"""
Train MultistreamCNNInertialNet (acc-only CNN) using the same imports,
dataset, train/val/test splits and functions as in Project.ipynb.
Model architecture matches the MultistreamCNNInertialNet cell in Project.ipynb.
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

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()

# ---------------------------------------------------------------------------
# Model: Z-score (train stats) + MultistreamCNNInertialNet (from Project.ipynb cell)
# ---------------------------------------------------------------------------
class MultistreamCNNInertialNet(nn.Module):
    """Acc-only Multistream CNN: (B,3,T)->(B,1,3,T) 2D convs + pool + FC."""

    def __init__(self, num_classes=18, norm_mean=None, norm_std=None):
        super(MultistreamCNNInertialNet, self).__init__()
        if norm_mean is None or norm_std is None:
            raise ValueError("norm_mean and norm_std are required (from compute_acc_zscore_stats on train split)")
        self.norm = ZScoreNormalize(norm_mean, norm_std)

        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 4), stride=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 6), stride=(2, 2), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool2d(
            kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, acc):
        x = self.norm(acc).unsqueeze(1)
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset and collate (same as Project.ipynb / train_helios_inertial.py)
# ---------------------------------------------------------------------------
class InertialSequenceDataset(Dataset):
    """Sequence-level dataset: acceleration channels only (3, T) and label."""

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


def collate_acc_only(batch):
    """Pad variable-length acc sequences to (B,3,T); stack labels — acc-only for MultistreamCNNInertialNet."""
    accs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, labels_stacked


collate_fn = collate_acc_only


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
    print(f"Split: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )
    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_seq_ids = trainset_df["sequence_id"].unique().tolist()
    val_seq_ids = valset_df["sequence_id"].unique().tolist()
    test_seq_ids = testset_df["sequence_id"].unique().tolist()

    train_ds = InertialSequenceDataset(trainset_df)
    val_ds = InertialSequenceDataset(valset_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_acc_only,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_acc_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    acc_mean, acc_std = compute_acc_zscore_stats(trainset_df)
    model = MultistreamCNNInertialNet(num_classes=num_classes, norm_mean=acc_mean, norm_std=acc_std)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Multistream_CNN_acc_only.pth"

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
