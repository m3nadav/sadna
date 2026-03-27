"""
Train Multistream CNN on rotation (quaternion) data only: rot_x, rot_y, rot_z, rot_w.
Architecture mirrors MultistreamCNNInertialNet (acc-only) but with 4 input channels.
Uses same data loading, split, and training setup as train_multistream_cnn.py.
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
    ROT_COLS,
    RANDOM_STATE,
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
# Model: MinMaxNormalize + MultistreamCNNRotNet (mirrors acc-only architecture)
# ---------------------------------------------------------------------------
from train_fusion_multistream_cnn import MinMaxNormalize


class MultistreamCNNRotNet(nn.Module):
    """
    Multistream CNN on rotation (quaternion) data only.
    Same 2D conv stack as MultistreamCNNInertialNet: (B,4,T)->(B,1,4,T), then conv2d blocks.
    conv1 uses kernel (4,4) to cover the four quaternion axes (acc model uses (3,4) for three axes).
    """

    def __init__(self, num_classes=18):
        super(MultistreamCNNRotNet, self).__init__()

        self.norm = MinMaxNormalize(-1.0, 1.0)

        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(1, 4)),
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

    def forward(self, rot):
        x = self.norm(rot).unsqueeze(1)
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset and collate for rotation sequences
# ---------------------------------------------------------------------------
class RotSequenceDataset(Dataset):
    """Sequence-level dataset returning rot (4, T) and label. Uses ROT_COLS."""

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id, data = self.sequences[idx]
        # Sequences kept by filter_sequences_by_sensor_validity(rot_cols): ≥1 row with unit-norm quat (~1±1e-6).
        rot_data = data[ROT_COLS].values.astype(np.float32)
        rot = torch.tensor(rot_data, dtype=torch.float32).T  # (4, T)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return rot, label


def collate_rot_only(batch):
    """Pad variable-length quaternion sequences to (B,4,T); stack labels — rotation only."""
    rots, labels = zip(*batch)
    rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return rot_padded, labels_stacked


collate_fn = collate_rot_only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_df = load_train_data()

    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    # Same subject-based train/val/test split as multistream CNN (acc)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)
    print(f"Split (before dropping missing quaternion): train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    # Drop sequences with no usable rotation (see utils.filter_sequences_by_sensor_validity rot rule)
    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = filter_sequences_by_sensor_validity(
        trainset_df, rot_cols=ROT_COLS
    )
    valset_df, val_seq_ids = filter_sequences_by_sensor_validity(valset_df, rot_cols=ROT_COLS)
    testset_df, test_seq_ids = filter_sequences_by_sensor_validity(testset_df, rot_cols=ROT_COLS)

    print(
        f"Dropped sequences with no unit-norm quaternion row: "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_ds = RotSequenceDataset(trainset_df)
    val_ds = RotSequenceDataset(valset_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_rot_only,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_rot_only,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultistreamCNNRotNet(num_classes=num_classes)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Multistream_CNN_rot_only.pth"

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
