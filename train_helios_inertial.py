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
from sklearn.preprocessing import LabelEncoder

# Optional: match notebook display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


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
# Model: MinMaxNormalize + HeliosInertialNet (same as Project.ipynb)
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


class HeliosInertialNet(nn.Module):
    """Initial model architecture from Project.ipynb (acc-only, Conv + LSTM + FC)."""

    def __init__(self, num_classes=18):
        super(HeliosInertialNet, self).__init__()
        self.input_norm = MinMaxNormalize(-20.0, 20.0)
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
    """Same as Project.ipynb: sequence-level dataset returning acc, rot, tof, thm, label."""

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id, data = self.sequences[idx]
        acc = torch.tensor(data[["acc_x", "acc_y", "acc_z"]].values, dtype=torch.float32).T
        rot_cols = [c for c in data.columns if "rot_" in c]
        rot = torch.tensor(data[rot_cols].values, dtype=torch.float32).T
        tof_cols = [c for c in data.columns if "tof_" in c]
        tof = torch.tensor(data[tof_cols].mean().values, dtype=torch.float32)
        thm_cols = [c for c in data.columns if "thm_" in c]
        thm = torch.tensor(data[thm_cols].iloc[0].values, dtype=torch.float32)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return acc, rot, tof, thm, label


def helios_collate_fn(batch):
    """Collate for HeliosInertialNet (acc + labels only). Same as Project.ipynb."""
    accs, rots, tofs, thms, labels = zip(*batch)
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    labels_stacked = torch.stack(labels)
    return acc_padded, labels_stacked


# ---------------------------------------------------------------------------
# Training loop (same as Project.ipynb)
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
    """
    Saves the model's state_dict, optimizer's state_dict, gesture map, training history,
    the sequence IDs for train, validation, and test sets, and the scheduler's state.
    Same as Project.ipynb "Storing the Model" section.
    """
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
    save_path="best_model.pth",
):
    """Run training with early stopping and best-model save. Same logic as Project.ipynb."""
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
    model = HeliosInertialNet(num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    # 7. Train (saves best state_dict to save_path during training)
    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/helios_inertial_net_2d_conv.pth"
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

    # 8. Load best weights then save full checkpoint (Storing the Model, Project.ipynb)
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
