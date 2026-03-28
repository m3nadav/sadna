"""
Train a joint early-fusion CNN: z-scored acc (3) + min-max quaternions (4) + per-timestep
validity mask (1) stacked as (8, T), single 2D conv trunk, then MLP classifier.
Uses all sequences (no drop for missing quaternion at sequence level); invalid quat timesteps
are zeroed with mask 0. Checkpoint feeds frozen 64-d extractors for ToF/THM fusion.

Canonical home for sensor preprocessing modules used across the pipeline:
  - ZScoreNormalize, compute_acc_zscore_stats  (accelerometer z-scoring)
  - MinMaxNormalize                             (quaternion clamping)
  - quat_row_valid_mask                         (per-timestep quaternion validity)
  - ZSCORE_EPS, ROT_QUAT_UNIT_NORM_TOL         (associated constants)

Self-contained — no dependency on utils.py.
"""
import os
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ===========================================================================
# Constants
# ===========================================================================
RANDOM_STATE = 42
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ROT_COLS = ["rot_x", "rot_y", "rot_z", "rot_w"]

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

ZSCORE_EPS = 1e-6
ROT_QUAT_UNIT_NORM_TOL = 1e-6

SAVE_PATH = "models/deep_learning_models/Fusion_Multistream_CNN.pth"

_ZSCORE_SUBMODULE_PATHS = (
    ("norm",),
    ("input_norm",),
    ("acc_backbone", "norm"),
    ("acc_zscore",),
)

# ===========================================================================
# Sensor preprocessing
# ===========================================================================


def quat_row_valid_mask(rot_rows_xyzw):
    """
    Per-timestep quaternion validity: finite components and L2 norm within
    ROT_QUAT_UNIT_NORM_TOL of 1.

    Parameters
    ----------
    rot_rows_xyzw : np.ndarray
        Shape (T, 4) — rows are (rot_x, rot_y, rot_z, rot_w) per timestep.

    Returns
    -------
    np.ndarray
        Shape (T,), dtype bool.
    """
    sub = np.asarray(rot_rows_xyzw, dtype=np.float64)
    if sub.size == 0:
        return np.zeros((0,), dtype=bool)
    if sub.ndim != 2 or sub.shape[1] != 4:
        raise ValueError(f"quat_row_valid_mask expects (T, 4), got {sub.shape}")
    norms = np.linalg.norm(sub, axis=1)
    return np.isfinite(norms) & (np.abs(norms - 1.0) <= ROT_QUAT_UNIT_NORM_TOL)


def compute_acc_zscore_stats(trainset_df, cols=None):
    """Per-channel mean and population std (ddof=0) on all training rows."""
    if cols is None:
        cols = ACC_COLS
    sub = trainset_df[cols]
    mean = sub.mean(axis=0).to_numpy(dtype=np.float32)
    std = sub.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.maximum(std, ZSCORE_EPS)
    return mean, std


class ZScoreNormalize(nn.Module):
    """Per-channel z-score; mean/std from training data, stored as buffers."""

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


class MinMaxNormalize(nn.Module):
    """Clamp inputs to [min_val, max_val].  For unit quaternions use (-1, 1)."""

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


# ===========================================================================
# Data loading & splitting helpers
# ===========================================================================


def set_default_seeds(random_state=RANDOM_STATE):
    np.random.seed(random_state)
    torch.manual_seed(random_state)


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


def final_robust_split(df, subject_col="subject", bfrb_col="is_bfrb", test_size=0.15, val_size=0.15):
    """Subject-based split by BFRB proportion."""
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


# ===========================================================================
# Checkpointing & training loop
# ===========================================================================


def _add_input_zscore_to_checkpoint(checkpoint, model):
    for path in _ZSCORE_SUBMODULE_PATHS:
        mod = model
        for name in path:
            mod = getattr(mod, name, None)
            if mod is None:
                break
        else:
            if hasattr(mod, "mean") and hasattr(mod, "std"):
                checkpoint["input_zscore_mean"] = mod.mean.detach().cpu().numpy()
                checkpoint["input_zscore_std"] = mod.std.detach().cpu().numpy()
                return


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
    extra_checkpoint_keys=None,
):
    """Save full checkpoint; optional extra keys."""
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
    _add_input_zscore_to_checkpoint(checkpoint, model)
    if extra_checkpoint_keys:
        for k, v in extra_checkpoint_keys.items():
            if v is not None:
                checkpoint[k] = v
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to {filepath}")


def load_best_weights_into_model(model, path, device):
    """Load weights saved during training (raw state_dict on disk)."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    train_step: Callable[..., Tuple[float, float]],
    val_step: Callable[..., Tuple[float, float]],
    *,
    num_epochs=100,
    early_stop_patience=15,
    save_path="best_model.pth",
):
    """
    Epoch loop with ReduceLROnPlateau on val loss, early stopping on val accuracy.
    Best weights saved as a raw state_dict at save_path.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss_item, acc_item = train_step(batch)
            epoch_loss += loss_item
            epoch_acc += acc_item
            num_batches += 1

        if num_batches == 0:
            raise ValueError("train_loader produced no batches")

        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                v_loss_item, v_acc_item = val_step(batch)
                val_loss += v_loss_item
                val_acc += v_acc_item
                val_batches += 1

        if val_batches == 0:
            raise ValueError("val_loader produced no batches")

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
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"| Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc * 100:.2f}% "
            f"| Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc * 100:.2f}% "
            f"| LR: {current_lr:.6f} | Best Val: {best_val_acc * 100:.2f}%"
        )

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc * 100:.2f}%")
    return history


def finalize_training_checkpoint(
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
    filepath,
    train_step: Callable[..., Tuple[float, float]],
    val_step: Callable[..., Tuple[float, float]],
    *,
    num_epochs=100,
    early_stop_patience=15,
    extra_checkpoint_keys=None,
):
    """Run training, reload best weights, then save full checkpoint."""
    history = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        train_step,
        val_step,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        save_path=filepath,
    )
    load_best_weights_into_model(model, filepath, device)
    save_model_and_metadata(
        model,
        optimizer,
        gesture_map,
        history,
        train_seq_ids,
        val_seq_ids,
        test_seq_ids,
        scheduler,
        filepath=filepath,
        extra_checkpoint_keys=extra_checkpoint_keys,
    )
    return history


# ===========================================================================
# Model
# ===========================================================================


class JointAccRotMultistreamNet(nn.Module):
    """Joint acc+rot CNN: (B,3,T), (B,4,T), (B,1,T) mask -> logits."""

    def __init__(self, num_classes=18, norm_mean=None, norm_std=None):
        super().__init__()
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

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=(0, 1))
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, acc, rot, rot_mask, pad_mask=None):
        """
        acc: (B, 3, T), rot: (B, 4, T), rot_mask: (B, 1, T) float 0/1.
        pad_mask: (B, 1, T) float 0/1 — 1 for real timesteps, 0 for batch-padding.

        Normalization is applied first, then pad_mask zeros out padded timesteps
        so that padding lives at exactly 0 in the normalized feature space.
        """
        acc_n = self.acc_zscore(acc)
        m = rot_mask.to(dtype=acc.dtype, device=acc.device)
        rot_clean = rot * m.expand_as(rot)
        rot_n = self.rot_norm(rot_clean)

        if pad_mask is not None:
            p = pad_mask.to(dtype=acc_n.dtype, device=acc_n.device)
            acc_n = acc_n * p
            rot_n = rot_n * p
            m = m * p

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


# ===========================================================================
# Dataset
# ===========================================================================


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
    """Pad acc, rot, mask; build pad_mask from sequence lengths; stack labels."""
    accs, rots, masks, labels = zip(*batch)
    lengths = [a.shape[1] for a in accs]
    acc_padded = pad_sequence([a.T for a in accs], batch_first=True, padding_value=0).transpose(1, 2)
    rot_padded = pad_sequence([r.T for r in rots], batch_first=True, padding_value=0).transpose(1, 2)
    mask_padded = pad_sequence([m.T for m in masks], batch_first=True, padding_value=0).transpose(1, 2)
    max_len = acc_padded.shape[2]
    pad_mask = torch.zeros(len(lengths), 1, max_len)
    for i, l in enumerate(lengths):
        pad_mask[i, :, :l] = 1.0
    labels_stacked = torch.stack(labels)
    return acc_padded, rot_padded, mask_padded, pad_mask, labels_stacked


# ===========================================================================
# Main
# ===========================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

set_default_seeds()


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
        optimizer, mode="min", factor=0.8, patience=7
    )

    os.makedirs("models/deep_learning_models", exist_ok=True)
    save_path = SAVE_PATH

    def train_step(batch):
        acc, rot, rot_mask, pad_mask, labels = batch
        acc = acc.to(device)
        rot = rot.to(device)
        rot_mask = rot_mask.to(device)
        pad_mask = pad_mask.to(device)
        labels = labels.to(device)
        outputs = model(acc, rot, rot_mask, pad_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_acc = (outputs.argmax(1) == labels).float().mean()
        return loss.item(), batch_acc.item()

    def val_step(batch):
        v_acc, v_rot, v_mask, v_pad_mask, v_labels = batch
        v_acc = v_acc.to(device)
        v_rot = v_rot.to(device)
        v_mask = v_mask.to(device)
        v_pad_mask = v_pad_mask.to(device)
        v_labels = v_labels.to(device)
        val_outputs = model(v_acc, v_rot, v_mask, v_pad_mask)
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
        early_stop_patience=10,
    )
    return model, history, le


if __name__ == "__main__":
    model, history, le = main()
