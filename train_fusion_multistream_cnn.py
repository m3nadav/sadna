"""
Train a joint early-fusion CNN: z-scored acc (3) + min-max quaternions (4) + per-timestep
validity mask (1) stacked as (8, T), single 2D conv trunk, then MLP classifier.
Uses all sequences (no drop for missing quaternion at sequence level); invalid quat timesteps
are zeroed with mask 0. Checkpoint feeds frozen 64-d extractors for ToF/THM fusion.
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
    ZScoreNormalize,
    apply_label_encoding,
    compute_acc_zscore_stats,
    final_robust_split,
    finalize_training_checkpoint,
    load_train_data,
    quat_row_valid_mask,
    set_default_seeds,
)
from train_multistream_cnn_rot import MinMaxNormalize

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()

SAVE_PATH = "models/deep_learning_models/Fusion_Multistream_CNN.pth"


class JointAccRotMultistreamNet(nn.Module):
    """Joint acc+rot CNN: (B,3,T), (B,4,T), (B,1,T) mask -> logits. Mask: 1 = valid unit quat row."""

    def __init__(self, num_classes=18, norm_mean=None, norm_std=None):
        super(JointAccRotMultistreamNet, self).__init__()
        if norm_mean is None or norm_std is None:
            raise ValueError("norm_mean and norm_std are required (from compute_acc_zscore_stats on train split)")
        self.acc_zscore = ZScoreNormalize(norm_mean, norm_std)
        self.rot_norm = MinMaxNormalize(-1.0, 1.0)

        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 4), stride=(1, 4)),
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

    def forward(self, acc, rot, rot_mask):
        """
        acc: (B, 3, T), rot: (B, 4, T), rot_mask: (B, 1, T) float 0/1.
        """
        acc_n = self.acc_zscore(acc)
        m = rot_mask.to(dtype=acc.dtype, device=acc.device)
        rot_clean = rot * m.expand_as(rot)
        rot_n = self.rot_norm(rot_clean)
        x = torch.cat([acc_n, rot_n, m], dim=1)
        x = x.unsqueeze(1)
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def truncate_joint_classifier_to_features(model):
    """Replace classifier with layers 0..3 (64-d output). Modifies model in place."""
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        model.classifier[2],
        model.classifier[3],
    )


# ---------------------------------------------------------------------------
# Dataset: (acc, rot, rot_mask, label); invalid quat timesteps zeroed, mask 0
# ---------------------------------------------------------------------------
class AccRotSequenceDataset(Dataset):
    """Sequence-level dataset returning acc (3,T), rot (4,T), rot_mask (1,T), label."""

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        acc = torch.tensor(data[ACC_COLS].values, dtype=torch.float32).T
        rot_np = data[ROT_COLS].values.astype(np.float64)
        valid = quat_row_valid_mask(rot_np)
        rot_filled = np.nan_to_num(rot_np, nan=0.0).astype(np.float32)
        rot_filled[~valid] = 0.0
        rot = torch.tensor(rot_filled, dtype=torch.float32).T
        mask = torch.tensor(valid, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, rot, mask, label


def collate_fn(batch):
    """Pad acc, rot, mask; stack labels."""
    accs, rots, masks, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
    mask_padded = pad_sequence([m.T for m in masks], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, rot_padded, mask_padded, labels_stacked


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
        f"Split (all sequences, no quaternion sequence filter): train {len(trainset_df)} rows, "
        f"val {len(valset_df)} rows, test {len(testset_df)} rows"
    )

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    train_seq_ids = trainset_df["sequence_id"].unique().tolist()
    val_seq_ids = valset_df["sequence_id"].unique().tolist()
    test_seq_ids = testset_df["sequence_id"].unique().tolist()

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    acc_mean, acc_std = compute_acc_zscore_stats(trainset_df)

    train_ds = AccRotSequenceDataset(trainset_df)
    val_ds = AccRotSequenceDataset(valset_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = JointAccRotMultistreamNet(
        num_classes=num_classes, norm_mean=acc_mean, norm_std=acc_std
    )
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total parameters, {n_trainable:,} trainable (joint end-to-end)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = SAVE_PATH

    def train_step(batch):
        acc, rot, rot_mask, labels = batch
        acc = acc.to(device)
        rot = rot.to(device)
        rot_mask = rot_mask.to(device)
        labels = labels.to(device)
        outputs = model(acc, rot, rot_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_acc = (outputs.argmax(1) == labels).float().mean()
        return loss.item(), batch_acc.item()

    def val_step(batch):
        v_acc, v_rot, v_mask, v_labels = batch
        v_acc = v_acc.to(device)
        v_rot = v_rot.to(device)
        v_mask = v_mask.to(device)
        v_labels = v_labels.to(device)
        val_outputs = model(v_acc, v_rot, v_mask)
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
