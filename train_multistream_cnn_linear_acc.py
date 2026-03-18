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
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumulative_trapezoid

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


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
        quat_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
        quats = self.df[quat_cols].values
        norms = np.linalg.norm(quats, axis=1)
        valid_mask = norms > 0

        acc_body_full = self.df[["acc_x", "acc_y", "acc_z"]].values
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


# ---------------------------------------------------------------------------
# Data loading and split (same as train_multistream_cnn.py / train_multistream_cnn_rot.py)
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


def drop_sequences_with_missing_quaternions(df, rot_cols=None):
    """Keep only sequences where every row has non-NaN values in all quaternion columns."""
    if rot_cols is None:
        rot_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    seq_has_missing = df.groupby("sequence_id")[rot_cols].apply(
        lambda g: g.isna().any().any()
    )
    valid_seq_ids = seq_has_missing[~seq_has_missing].index.tolist()
    return df[df["sequence_id"].isin(valid_seq_ids)].reset_index(drop=True), valid_seq_ids


# ---------------------------------------------------------------------------
# Model: same as MultistreamCNNInertialNet (acc-only)
# ---------------------------------------------------------------------------
class MinMaxNormalize(nn.Module):
    """Custom layer to scale inputs to [0, 1] based on provided bounds."""

    def __init__(self, min_val, max_val):
        super(MinMaxNormalize, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x = torch.clamp(x, self.min_val, self.max_val)
        return (x - self.min_val) / (self.max_val - self.min_val + 1e-6)


class MultistreamCNNInertialNet(nn.Module):
    """Same architecture as acc-only Multistream CNN; input is world-frame linear acc (3, T)."""

    def __init__(self, num_classes=18):
        super(MultistreamCNNInertialNet, self).__init__()
        self.norm = MinMaxNormalize(-20.0, 20.0)
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
    """Sequence-level dataset: IMUProcessor yields world-frame linear acc (3, T) and label."""

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
        acc_linear = processor.acc_linear.astype(np.float32)  # (T, 3)
        acc = torch.tensor(acc_linear.T, dtype=torch.float32)  # (3, T)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, label


def collate_fn(batch):
    """Pad variable-length acc sequences; stack labels."""
    accs, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, labels_stacked


# ---------------------------------------------------------------------------
# Training loop (same as train_multistream_cnn.py)
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
    save_path="best_linear_acc.pth",
):
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
    print(f"Split (before dropping missing quaternion): train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

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

    train_ds = LinearAccSequenceDataset(trainset_df, silent=True)
    val_ds = LinearAccSequenceDataset(valset_df, silent=True)
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

    model = MultistreamCNNInertialNet(num_classes=num_classes)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Multistream_CNN_linear_acc.pth"
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
