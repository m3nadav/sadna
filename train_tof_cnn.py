"""
Train ToF CNN: 2D CNN on Time-of-Flight sensor data (5 sensors × 8×8) with the same
train/val/test split as the acceleration-based models. Sequences with ToF data missing
for the entire sequence are dropped.
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

# ToF measurement range (mm); -1 denotes missing
TOF_MIN, TOF_MAX = 0.0, 249.0


# ---------------------------------------------------------------------------
# Data loading (same as Project.ipynb / train_multistream_cnn)
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
# BFRB flag and train/val/test split (same as Project.ipynb / acceleration models)
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
# ToF columns and drop sequences with ToF missing for entire sequence
# ---------------------------------------------------------------------------
def get_tof_columns(df):
    return [c for c in df.columns if c.startswith("tof_")]


def drop_sequences_with_missing_tof(df, tof_cols=None):
    """
    Keep only sequences that have at least one valid ToF reading (non -1) in at least one frame.
    A sequence is invalid only if ALL values throughout the sequence are -1 (no pixel recorded anything).
    Same train/val/test split as acceleration models; we only drop these fully missing sequences.
    """
    if tof_cols is None:
        tof_cols = get_tof_columns(df)
    # Per sequence: is there any valid (non -1) value?
    seq_has_any_valid = df.groupby("sequence_id")[tof_cols].apply(
        lambda g: (g != -1).any().any()
    )
    valid_seq_ids = seq_has_any_valid[seq_has_any_valid].index.tolist()
    return df[df["sequence_id"].isin(valid_seq_ids)].reset_index(drop=True), valid_seq_ids


# ---------------------------------------------------------------------------
# ToF CNN model (input is already normalized to [0, 1] in dataset)
# ---------------------------------------------------------------------------
def fuse_pairwise(x):
    """Fuse channels pairwise by summing: (B, C, H, W) -> (B, C//2, H, W)."""
    B, C, H, W = x.shape
    assert C % 2 == 0
    x = x.view(B, C // 2, 2, H, W)
    return x.sum(dim=2)


# Flattened feature size after conv3: 32 * 4 * 3 = 384 (with smaller kernels below)
TOF_CNN_FEAT_SIZE = 384


class ToFCNN(nn.Module):
    """
    2D CNN for ToF sensors (input already normalized to [0, 1] per frame):
    - Conv1: 8 filters 2×4, stride 2×2
    - Fuse pairwise -> 4 channels
    - Conv2: 16 filters 2×2, stride 1×1
    - Fuse pairwise -> 8 channels
    - Conv3: 32 filters 2×2, stride 1×1 (with padding)
    - FC: 128, 100, num_classes
    Input: (B, max_T, 5, 8, 8) and lengths (B,) — padded ToF sequences; pooling over T is masked by lengths.
    """

    def __init__(self, num_classes=13):
        super(ToFCNN, self).__init__()
        # Conv1: 5 -> 8, kernel (2, 4), stride (2, 2) -> (4, 3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=(2, 4), stride=(2, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # After conv1: (B, 8, 4, 3). Fuse pairwise -> (B, 4, 4, 3)
        # Conv2: 4 -> 16, kernel 2×2, stride 1×1 -> (3, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # After conv2: (B, 16, 3, 2). Fuse pairwise -> (B, 8, 3, 2)
        # Conv3: 8 -> 32, kernel 2×2, stride 1×1, padding (1, 1) -> (4, 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=2, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # After conv3: (B, 32, 4, 3). Flatten -> 384
        self.classifier = nn.Sequential(
            nn.Linear(TOF_CNN_FEAT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, tof_padded, lengths):
        """
        tof_padded: (B, max_T, 5, 8, 8), values in [0, 1]; padding is 0.
        lengths: (B,) actual sequence length per batch item.
        """
        # Ensure input is in [0, 1] and finite (avoids NaN from bad data or padding)
        tof_padded = torch.clamp(tof_padded, 0.0, 1.0)
        lengths = lengths.clamp(min=1)  # avoid div-by-zero in masked mean
        B, max_T, C, H, W = tof_padded.shape
        # Run 2D CNN on each frame
        x = tof_padded.view(B * max_T, C, H, W)
        x = self.conv1(x)       # (B*max_T, 8, 4, 3)
        x = fuse_pairwise(x)    # (B*max_T, 4, 4, 3)
        x = self.conv2(x)       # (B*max_T, 16, 3, 2)
        x = fuse_pairwise(x)    # (B*max_T, 8, 3, 2)
        x = self.conv3(x)       # (B*max_T, 32, 4, 3)
        x = torch.flatten(x, 1) # (B*max_T, 384)
        x = x.view(B, max_T, -1)  # (B, max_T, 384)
        # Masked mean over time (ignore padding)
        mask = torch.arange(max_T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_T)
        mask = mask.unsqueeze(-1).to(x.dtype)  # (B, max_T, 1)
        lengths_clamp = lengths.unsqueeze(1).clamp(min=1).to(x.dtype)  # (B, 1)
        x = (x * mask).sum(dim=1) / lengths_clamp  # (B, 384)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset: variable-length (T, 5, 8, 8) per sequence, normalized to [0, 1]
# ---------------------------------------------------------------------------
def _tof_frame_to_5x8x8(tof_flat):
    """Convert 320-dim ToF vector to (5, 8, 8). tof_flat: (320,) or (T, 320)."""
    if tof_flat.ndim == 1:
        tof_flat = tof_flat.reshape(1, -1)
    T = tof_flat.shape[0]
    # Order: tof_1_v0..tof_1_v63, tof_2_v0..tof_2_v63, ...
    x = tof_flat.reshape(T, 5, 64)
    x = x.reshape(T, 5, 8, 8)
    return x


def _normalize_tof_01(x):
    """Scale ToF values from [TOF_MIN, TOF_MAX] to [0, 1]. Replaces NaN/Inf so output is always finite."""
    x = np.asarray(x, dtype=np.float64)
    # Replace invalid values before clip (NaN/Inf would otherwise propagate)
    x = np.nan_to_num(x, nan=TOF_MAX, posinf=TOF_MAX, neginf=TOF_MIN)
    x = np.clip(x, TOF_MIN, TOF_MAX)
    denom = float(TOF_MAX - TOF_MIN)
    if denom <= 0:
        denom = 1.0
    out = (x - TOF_MIN) / denom
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


class ToFSequenceDataset(Dataset):
    """Sequence-level dataset: (T, 5, 8, 8) per sequence, normalized to [0, 1]; one label per sequence."""

    def __init__(self, dataframe, tof_cols=None):
        self.df = dataframe.reset_index(drop=True)
        if tof_cols is None:
            tof_cols = get_tof_columns(self.df)
        self.tof_cols = tof_cols
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        tof = data[self.tof_cols].values.astype(np.float64)  # (T, 320)
        # -1 and NaN mean no reading (treat as very far); replace with max range before normalizing
        tof[tof == -1] = TOF_MAX
        tof[np.isnan(tof)] = TOF_MAX
        tof = np.nan_to_num(tof, nan=TOF_MAX, posinf=TOF_MAX, neginf=TOF_MIN)
        tof_per_frame = _tof_frame_to_5x8x8(tof)  # (T, 5, 8, 8)
        # Normalize to [0, 1] before training (guaranteed finite and in range)
        tof_per_frame = _normalize_tof_01(tof_per_frame)
        x = torch.tensor(tof_per_frame, dtype=torch.float32)  # (T, 5, 8, 8)
        x = torch.clamp(x, 0.0, 1.0)  # ensure no stray value slips through
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return x, label


def collate_tof(batch):
    """Pad variable-length ToF sequences to max_T with 0, same idea as train_multistream_cnn collate."""
    tof_list, labels = zip(*batch)
    tof_padded = pad_sequence(tof_list, batch_first=True, padding_value=0.0)  # (B, max_T, 5, 8, 8)
    lengths = torch.tensor([t.shape[0] for t in tof_list], dtype=torch.long)
    labels_stacked = torch.stack(labels)
    return tof_padded, lengths, labels_stacked


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
    save_path="best_tof_cnn.pth",
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
            tof_padded, lengths, labels = batch
            tof_padded = tof_padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(tof_padded, lengths)
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
            for tof_padded, lengths, v_labels in val_loader:
                tof_padded = tof_padded.to(device)
                lengths = lengths.to(device)
                v_labels = v_labels.to(device)
                val_outputs = model(tof_padded, lengths)
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
    print(f"Split (before dropping missing ToF): train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    tof_cols = get_tof_columns(train_df)
    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = drop_sequences_with_missing_tof(trainset_df, tof_cols)
    valset_df, val_seq_ids = drop_sequences_with_missing_tof(valset_df, tof_cols)
    testset_df, test_seq_ids = drop_sequences_with_missing_tof(testset_df, tof_cols)

    print(
        f"Dropped sequences with ToF missing for entire sequence: "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_ds = ToFSequenceDataset(trainset_df, tof_cols)
    val_ds = ToFSequenceDataset(valset_df, tof_cols)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_tof,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_tof,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToFCNN(num_classes=num_classes)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/ToF_CNN.pth"
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
