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
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ZSCORE_EPS = 1e-6


def compute_acc_zscore_stats(trainset_df, cols=None):
    """Per-channel mean and population std (ddof=0) on all training rows."""
    if cols is None:
        cols = ACC_COLS
    sub = trainset_df[cols]
    mean = sub.mean(axis=0).to_numpy(dtype=np.float32)
    std = sub.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.maximum(std, ZSCORE_EPS)
    return mean, std


# ---------------------------------------------------------------------------
# Data loading (same as Project.ipynb)
# ---------------------------------------------------------------------------
def load_train_data():
    """Load training CSV from kagglehub or ./data."""
    try:
        path = "./data"
        train_df = pd.read_csv(os.path.join(path, "train.csv"))
        print(f"Loaded train.csv from {path}")
    except FileNotFoundError:
        try:
            import kagglehub
            os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser(".kaggle")
            path = kagglehub.competition_download("cmi-detect-behavior-with-sensor-data")
            train_df = pd.read_csv(os.path.join(path, "train.csv"))
            print(f"Loaded train.csv via kagglehub from {path}")
        except Exception as e:
            raise FileNotFoundError(
                "Neither ./data/train.csv nor kagglehub data found. "
                "Download the competition data or place train.csv in ./data/"
            ) from e
    return train_df


# ---------------------------------------------------------------------------
# BFRB flag and train/val/test split (same as Project.ipynb)
# ---------------------------------------------------------------------------
BFRB_GESTURES = [
    "Above ear - pull hair",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Cheek - pinch skin",
    "Neck - pinch skin",
    "Neck - scratch",
]


def final_robust_split(df, subject_col="subject", bfrb_col="is_bfrb", test_size=0.15, val_size=0.15):
    """Same as Project.ipynb: subject-based split by BFRB proportion."""
    sub_logic = df.groupby(subject_col)[bfrb_col].mean().reset_index()
    sub_logic["broad_cat"] = (sub_logic[bfrb_col] > 0.5).astype(int)
    unique_subs = sub_logic.sample(frac=1, random_state=RANDOM_STATE)
    n_total = len(unique_subs)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    test_subs = unique_subs.iloc[:n_test][subject_col]
    val_subs = unique_subs.iloc[n_test : n_test + n_val][subject_col]
    train_subs = unique_subs.iloc[n_test + n_val :][subject_col]
    train_set = df[df[subject_col].isin(train_subs)]
    val_set = df[df[subject_col].isin(val_subs)]
    test_set = df[df[subject_col].isin(test_subs)]
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Label encoding (same as Project.ipynb)
# ---------------------------------------------------------------------------
def apply_label_encoding(train_df, trainset_df, valset_df, testset_df):
    """Fit LabelEncoder on full train_df and add gesture_encoded to splits."""
    le = LabelEncoder()
    unique_gestures = train_df["gesture"].unique()
    le.fit(unique_gestures)
    gesture_map = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df = train_df.copy()
    train_df["gesture_encoded"] = train_df["gesture"].map(gesture_map)
    trainset_df = trainset_df.copy()
    trainset_df["gesture_encoded"] = trainset_df["gesture"].map(gesture_map)
    valset_df = valset_df.copy()
    valset_df["gesture_encoded"] = valset_df["gesture"].map(gesture_map)
    testset_df = testset_df.copy()
    testset_df["gesture_encoded"] = testset_df["gesture"].map(gesture_map)
    return train_df, trainset_df, valset_df, testset_df, le, gesture_map


# ---------------------------------------------------------------------------
# Model: Z-score (train stats) + MultistreamCNNInertialNet (from Project.ipynb cell)
# ---------------------------------------------------------------------------
class ZScoreNormalize(nn.Module):
    """Per-channel z-score; mean/std from training data, stored as buffers for checkpointing."""

    def __init__(self, mean, std, eps=ZSCORE_EPS):
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1).clamp_min(eps)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x):
        m = self.mean.view(1, -1, 1)
        s = self.std.view(1, -1, 1)
        return (x - m) / s


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
    """Sequence-level dataset returning acc (3, T) and label. Same as Project.ipynb."""

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id, data = self.sequences[idx]
        acc = torch.tensor(data[["acc_x", "acc_y", "acc_z"]].values, dtype=torch.float32).T
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, label


def collate_fn(batch):
    """Collate for Multistream CNN: pad variable-length acc sequences, stack labels."""
    accs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, labels_stacked


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
    """Save full checkpoint (state_dict, optimizer, gesture_map, history, split IDs)."""
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
    # Explicit z-score stats (also in model_state_dict as norm.mean / norm.std) for analyze scripts
    if hasattr(model, "norm") and hasattr(model.norm, "mean"):
        checkpoint["input_zscore_mean"] = model.norm.mean.detach().cpu().numpy()
        checkpoint["input_zscore_std"] = model.norm.std.detach().cpu().numpy()
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
    save_path="best_multistream_cnn.pth",
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
            acc, labels = batch
            acc = acc.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(acc)
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
            for v_acc, v_labels in val_loader:
                v_acc = v_acc.to(device)
                v_labels = v_labels.to(device)
                val_outputs = model(v_acc)
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
