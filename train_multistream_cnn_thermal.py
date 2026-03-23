"""
Train Multistream CNN on thermal (thermopile) data only: thm_1 … thm_5.
Same data loading, train/val/test split, and training loop as train_multistream_cnn.py /
train_multistream_cnn_rot.py. Sequences with no valid thermal readings are dropped per split.
Inputs are imputed (invalid -> train mean per channel) and z-score normalized (train stats).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Plausible °C band for MLX90632-style readings (tunable after inspecting analyze_thermal_columns)
THM_MIN_C = 5.0
THM_MAX_C = 60.0
ZSCORE_EPS = 1e-6
DROPOUT_P = 0.2


# ---------------------------------------------------------------------------
# Data loading (same as Project.ipynb / train_multistream_cnn.py)
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
    """Same as Project.ipynb / train_multistream_cnn.py: subject-based split by BFRB proportion."""
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
# Thermal columns, validity, analysis, sequence drop
# ---------------------------------------------------------------------------
def get_thm_columns(df):
    """Return sorted thermal column names (thm_*)."""
    cols = [c for c in df.columns if c.startswith("thm_")]
    return sorted(cols)


def is_valid_thm_array(values, thm_min=THM_MIN_C, thm_max=THM_MAX_C):
    """Vectorized validity mask: finite, in [thm_min, thm_max], not -1 (missing sentinel)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) & (v != -1.0) & (v >= thm_min) & (v <= thm_max)
    return ok


def analyze_thermal_columns(df, thm_cols):
    """Print thermal column stats and sequence-level coverage."""
    print("\n--- Thermal sensor analysis ---")
    print(f"Columns ({len(thm_cols)}): {thm_cols}")
    if not thm_cols:
        print("No thm_* columns found.")
        return

    for c in thm_cols:
        s = df[c]
        nan_frac = float(s.isna().mean())
        vals = s.values
        n_minus_one = int(np.sum(vals == -1)) if vals.dtype != object else 0
        finite = vals[np.isfinite(vals.astype(np.float64))]
        print(f"  {c}: NaN fraction={nan_frac:.4f}", end="")
        if n_minus_one > 0:
            print(f", count==-1: {n_minus_one}", end="")
        print()
        if finite.size > 0:
            print(
                f"      finite min={finite.min():.4f} max={finite.max():.4f} "
                f"q01={np.quantile(finite, 0.01):.4f} q50={np.quantile(finite, 0.5):.4f} "
                f"q99={np.quantile(finite, 0.99):.4f}"
            )

    n_seq = df["sequence_id"].nunique()
    n_with_data = 0
    n_no_data = 0
    for _, g in df.groupby("sequence_id", sort=False):
        if is_valid_thm_array(g[thm_cols].values).any():
            n_with_data += 1
        else:
            n_no_data += 1
    print(f"Sequences: total={n_seq}, with at least one valid thermal reading={n_with_data}, all-invalid={n_no_data}")
    print("--- End thermal analysis ---\n")


def drop_sequences_with_no_thermal(df, thm_cols, thm_min=THM_MIN_C, thm_max=THM_MAX_C):
    """
    Keep sequences that have at least one valid thermal reading (any sensor, any frame).
    Returns filtered dataframe and list of kept sequence_id.
    """
    if not thm_cols:
        return df.iloc[0:0].copy(), []

    valid_seq_ids = []
    for seq_id, g in df.groupby("sequence_id", sort=False):
        if is_valid_thm_array(g[thm_cols].values, thm_min, thm_max).any():
            valid_seq_ids.append(seq_id)
    return df[df["sequence_id"].isin(valid_seq_ids)].reset_index(drop=True), valid_seq_ids


def compute_train_thermal_zscore_params(train_df, thm_cols, thm_min=THM_MIN_C, thm_max=THM_MAX_C):
    """
    Per-channel mean and std from training rows using only valid values.
    Imputation uses the same per-channel mean (raw °C).
    """
    means = np.zeros(len(thm_cols), dtype=np.float64)
    stds = np.ones(len(thm_cols), dtype=np.float64)
    for i, c in enumerate(thm_cols):
        vals = train_df[c].values.astype(np.float64)
        mask = is_valid_thm_array(vals, thm_min, thm_max)
        if mask.any():
            means[i] = np.mean(vals[mask])
            s = np.std(vals[mask], ddof=0)
            stds[i] = float(s) if s > ZSCORE_EPS else 1.0
        else:
            means[i] = 0.0
            stds[i] = 1.0
    return means, stds


# ---------------------------------------------------------------------------
# Model: Z-score + MultistreamCNNThermalNet (Conv2d + dropout)
# ---------------------------------------------------------------------------
class ZScoreNormalize(nn.Module):
    """Per-channel z-score: (x - mean) / (std + eps). Mean/std shaped for (B, C, T)."""

    def __init__(self, mean, std, eps=ZSCORE_EPS):
        super().__init__()
        # (1, C, 1) broadcasts over batch and time
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1))
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


class MultistreamCNNThermalNet(nn.Module):
    """
    Thermal-only CNN: (B,5,T) -> z-score -> (B,1,5,T) Conv2d blocks + dropout + FC with dropout.
    conv1 uses kernel (5,4) to cover all five thermopile channels (mirrors rot's (4,4)).
    """

    def __init__(self, num_classes=18, mean=None, std=None, dropout_p=DROPOUT_P):
        super().__init__()
        self.dropout_p = dropout_p
        self.norm = ZScoreNormalize(mean, std)

        self.conv1_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 4), stride=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 6), stride=(2, 2), padding=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(p=dropout_p)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, thm):
        x = self.norm(thm).unsqueeze(1)
        x = self.conv1_block(x)
        x = self.drop(x)
        x = self.conv2_block(x)
        x = self.drop(x)
        x = self.max_pool(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# Dataset and collate
# ---------------------------------------------------------------------------
class ThermalSequenceDataset(Dataset):
    """Sequence-level (5, T) thermal, invalid imputed to train mean (raw), z-score in model."""

    def __init__(self, dataframe, thm_cols, train_mean_raw):
        self.df = dataframe.reset_index(drop=True)
        self.thm_cols = thm_cols
        self.train_mean_raw = np.asarray(train_mean_raw, dtype=np.float64).reshape(len(thm_cols))
        self.sequences = list(self.df.groupby("sequence_id"))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _seq_id, data = self.sequences[idx]
        raw = data[self.thm_cols].values.astype(np.float64)  # (T, 5)
        mask = is_valid_thm_array(raw)
        imputed = raw.copy()
        for c in range(raw.shape[1]):
            col_mask = mask[:, c]
            if not col_mask.all():
                imputed[~col_mask, c] = self.train_mean_raw[c]
        thm = torch.tensor(imputed.T, dtype=torch.float32)
        label = torch.tensor(data["gesture_encoded"].iloc[0], dtype=torch.long)
        return thm, label


def make_collate_fn(train_mean_raw):
    """
    Pad (5, T) sequences to batch (B, 5, max_T). Padded timesteps use per-channel train mean (raw °C)
    so z-score in the model maps them to ~0 (same as imputed missing cells).
    """

    tm = torch.as_tensor(train_mean_raw, dtype=torch.float32).view(5, 1)

    def collate_fn(batch):
        thms, labels = zip(*batch)
        max_t = max(t.shape[1] for t in thms)
        b = len(thms)
        out = tm.expand(5, max_t).unsqueeze(0).expand(b, 5, max_t).clone()
        for i, t in enumerate(thms):
            ti = t.shape[1]
            out[i, :, :ti] = t
        return out, torch.stack(labels)

    return collate_fn


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
    thm_cols=None,
    thm_mean=None,
    thm_std=None,
    thm_valid_bounds=None,
    dropout_p=None,
):
    """Save full checkpoint including thermal normalization and dropout hyperparameter."""
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
        "thm_cols": thm_cols,
        "thm_mean": thm_mean,
        "thm_std": thm_std,
        "thm_valid_bounds": thm_valid_bounds,
        "dropout_p": dropout_p,
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
    save_path="best_multistream_cnn_thermal.pth",
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
            thm, labels = batch
            thm = thm.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(thm)
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
            for v_thm, v_labels in val_loader:
                v_thm = v_thm.to(device)
                v_labels = v_labels.to(device)
                val_outputs = model(v_thm)
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
    thm_cols = get_thm_columns(train_df)
    if not thm_cols:
        raise ValueError("No thm_* columns in train.csv; cannot train thermal model.")

    analyze_thermal_columns(train_df, thm_cols)

    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)
    print(
        f"Split (before dropping no-thermal sequences): train {len(trainset_df)} rows, "
        f"val {len(valset_df)} rows, test {len(testset_df)} rows"
    )

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    n_train_seq_before = trainset_df["sequence_id"].nunique()
    n_val_seq_before = valset_df["sequence_id"].nunique()
    n_test_seq_before = testset_df["sequence_id"].nunique()

    trainset_df, train_seq_ids = drop_sequences_with_no_thermal(trainset_df, thm_cols)
    valset_df, val_seq_ids = drop_sequences_with_no_thermal(valset_df, thm_cols)
    testset_df, test_seq_ids = drop_sequences_with_no_thermal(testset_df, thm_cols)

    print(
        f"Dropped sequences with no valid thermal data: "
        f"train {n_train_seq_before} -> {len(train_seq_ids)}, "
        f"val {n_val_seq_before} -> {len(val_seq_ids)}, "
        f"test {n_test_seq_before} -> {len(test_seq_ids)}"
    )
    print(f"After drop: train {len(trainset_df)} rows, val {len(valset_df)} rows, test {len(testset_df)} rows")

    if len(trainset_df) == 0:
        raise RuntimeError("Training set has no rows after dropping sequences without thermal data.")

    train_mean_raw, train_std_raw = compute_train_thermal_zscore_params(trainset_df, thm_cols)
    print(f"Z-score train stats per channel {thm_cols}: mean={train_mean_raw}, std={train_std_raw}")

    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")

    train_ds = ThermalSequenceDataset(trainset_df, thm_cols, train_mean_raw)
    val_ds = ThermalSequenceDataset(valset_df, thm_cols, train_mean_raw)
    collate = make_collate_fn(train_mean_raw)
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultistreamCNNThermalNet(
        num_classes=num_classes,
        mean=train_mean_raw,
        std=train_std_raw,
        dropout_p=DROPOUT_P,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = "models/deep_learning_models/Multistream_CNN_thermal_only.pth"
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
    thm_bounds = (THM_MIN_C, THM_MAX_C)
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
        thm_cols=thm_cols,
        thm_mean=train_mean_raw.tolist(),
        thm_std=train_std_raw.tolist(),
        thm_valid_bounds=thm_bounds,
        dropout_p=DROPOUT_P,
    )
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
