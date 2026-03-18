"""
Train a fusion model that takes the 64-dim features (last layer before classification)
from both the acc and rot Multistream CNNs, concatenates them to 128-dim, and classifies
with a 2-layer head to 18 classes. All acc and rot backbone layers are frozen; only the
fusion classifier is trained. Same train/val/test split and flow as the other models;
only sequences with complete quaternion data are used.
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

# Paths to pretrained checkpoints
ACC_CHECKPOINT_PATH = "models/deep_learning_models/Multistream_CNN_acc_only.pth"
ROT_CHECKPOINT_PATH = "models/deep_learning_models/Multistream_CNN_rot_only.pth"
ROT_COLS = ["rot_x", "rot_y", "rot_z", "rot_w"]


# ---------------------------------------------------------------------------
# Imports from acc and rot training scripts
# ---------------------------------------------------------------------------
from train_multistream_cnn import (
    load_train_data,
    final_robust_split,
    apply_label_encoding,
    BFRB_GESTURES,
    MultistreamCNNInertialNet,
)
from train_multistream_cnn_rot import (
    MultistreamCNNRotNet,
    drop_sequences_with_missing_quaternions,
)


# ---------------------------------------------------------------------------
# Feature extractors: acc and rot models with classifier replaced by feature-only part (64-dim)
# ---------------------------------------------------------------------------
def make_acc_feature_extractor(num_classes, checkpoint_path, device):
    """Load acc model and replace classifier with feature part only (output 64-dim). Freeze all."""
    model = MultistreamCNNInertialNet(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    # classifier is Sequential(Linear(64,128), ReLU, Linear(128,64), ReLU, Linear(64,num_classes))
    # Keep only up to the last ReLU so output is 64-dim
    model.classifier = nn.Sequential(
        model.classifier[0],  # Linear(64, 128)
        model.classifier[1],  # ReLU
        model.classifier[2],  # Linear(128, 64)
        model.classifier[3],  # ReLU
    )
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def make_rot_feature_extractor(num_classes, checkpoint_path, device):
    """Load rot model and replace classifier with feature part only (output 64-dim). Freeze all."""
    model = MultistreamCNNRotNet(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        model.classifier[2],
        model.classifier[3],
    )
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


# ---------------------------------------------------------------------------
# Fusion model: concat(acc_features, rot_features) -> 2-layer head -> num_classes
# ---------------------------------------------------------------------------
class FusionMultistreamCNN(nn.Module):
    """
    Concatenates 64-dim features from acc and rot backbones (frozen) and classifies
    with a 2-layer head: Linear(128, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes).
    """

    def __init__(self, acc_backbone, rot_backbone, num_classes=18, hidden_dim=64):
        super(FusionMultistreamCNN, self).__init__()
        self.acc_backbone = acc_backbone
        self.rot_backbone = rot_backbone
        self.fusion_classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, acc, rot):
        acc_feat = self.acc_backbone(acc)   # (batch, 64)
        rot_feat = self.rot_backbone(rot)   # (batch, 64)
        x = torch.cat([acc_feat, rot_feat], dim=1)  # (batch, 128)
        return self.fusion_classifier(x)


# ---------------------------------------------------------------------------
# Dataset: (acc, rot, label) per sequence; only sequences with complete quaternion
# ---------------------------------------------------------------------------
class AccRotSequenceDataset(Dataset):
    """Sequence-level dataset returning (acc (3,T), rot (4,T), label)."""

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id, data = self.sequences[idx]
        acc = torch.tensor(data[["acc_x", "acc_y", "acc_z"]].values, dtype=torch.float32).T
        rot = torch.tensor(data[ROT_COLS].values, dtype=torch.float32).T
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, rot, label


def collate_fn(batch):
    """Pad acc and rot sequences; stack labels."""
    accs, rots, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, rot_padded, labels_stacked


# ---------------------------------------------------------------------------
# Training loop (same structure as train_multistream_cnn.py)
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
    """Save full checkpoint. Only fusion classifier is trainable; backbones are frozen."""
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
    save_path="best_fusion_multistream_cnn.pth",
):
    """Run training with early stopping and best-model save."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for batch in train_loader:
            acc, rot, labels = batch
            acc = acc.to(device)
            rot = rot.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(acc, rot)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            for v_acc, v_rot, v_labels in val_loader:
                v_acc = v_acc.to(device)
                v_rot = v_rot.to(device)
                v_labels = v_labels.to(device)
                val_outputs = model(v_acc, v_rot)
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
    print(f"Split (before dropping missing quaternion): train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    # Same as rot model: only use sequences with complete quaternion data
    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = drop_sequences_with_missing_quaternions(trainset_df)
    valset_df, val_seq_ids = drop_sequences_with_missing_quaternions(valset_df)
    testset_df, test_seq_ids = drop_sequences_with_missing_quaternions(testset_df)

    print(
        f"Dropped sequences with missing quaternions: "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

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

    # Load frozen backbones (feature extractors)
    print("Loading acc backbone (frozen)...")
    acc_backbone = make_acc_feature_extractor(num_classes, ACC_CHECKPOINT_PATH, device)
    print("Loading rot backbone (frozen)...")
    rot_backbone = make_rot_feature_extractor(num_classes, ROT_CHECKPOINT_PATH, device)

    model = FusionMultistreamCNN(acc_backbone, rot_backbone, num_classes=num_classes, hidden_dim=64)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total parameters, {n_trainable:,} trainable (fusion classifier only)")

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
    save_path = "models/deep_learning_models/Fusion_Multistream_CNN.pth"
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

    model.load_state_dict(torch.load(save_path, map_location=device))
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
