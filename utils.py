"""Shared helpers for train_*.py: data loading, splits, label encoding, checkpoints, training loop."""
import os
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

RANDOM_STATE = 42
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ZSCORE_EPS = 1e-6
ROT_COLS = ["rot_x", "rot_y", "rot_z", "rot_w"]
# Unit quaternion: L2 norm within [1 - tol, 1 + tol] (per row, all rot_cols components finite).
ROT_QUAT_UNIT_NORM_TOL = 1e-6


def quat_row_valid_mask(rot_rows_xyzw):
    """
    Per-timestep quaternion validity: finite components and L2 norm within ROT_QUAT_UNIT_NORM_TOL of 1.

    Parameters
    ----------
    rot_rows_xyzw : np.ndarray
        Shape (T, 4) — rows are (rot_x, rot_y, rot_z, rot_w) per timestep (same order as ROT_COLS).

    Returns
    -------
    np.ndarray
        Shape (T,), dtype bool. Padding positions in batched tensors should use 0 in the mask after pad.
    """
    sub = np.asarray(rot_rows_xyzw, dtype=np.float64)
    if sub.size == 0:
        return np.zeros((0,), dtype=bool)
    if sub.ndim != 2 or sub.shape[1] != 4:
        raise ValueError(f"quat_row_valid_mask expects (T, 4), got {sub.shape}")
    norms = np.linalg.norm(sub, axis=1)
    return np.isfinite(norms) & (np.abs(norms - 1.0) <= ROT_QUAT_UNIT_NORM_TOL)


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

_ZSCORE_SUBMODULE_PATHS = (
    ("norm",),
    ("input_norm",),
    ("acc_backbone", "norm"),
    ("acc_zscore",),
)


def set_default_seeds(random_state=RANDOM_STATE):
    np.random.seed(random_state)
    torch.manual_seed(random_state)


def compute_acc_zscore_stats(trainset_df, cols=None):
    """Per-channel mean and population std (ddof=0) on all training rows."""
    if cols is None:
        cols = ACC_COLS
    sub = trainset_df[cols]
    mean = sub.mean(axis=0).to_numpy(dtype=np.float32)
    std = sub.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.maximum(std, ZSCORE_EPS)
    return mean, std


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


def filter_sequences_by_sensor_validity(
    df,
    *,
    sequence_id_col="sequence_id",
    rot_cols=None,
    tof_cols=None,
    thm_cols=None,
):
    """
    Keep sequences that satisfy every non-None sensor rule (AND across rules):

    - **rot_cols**: at least one timestep has a finite quaternion whose Euclidean norm is within
      ``ROT_QUAT_UNIT_NORM_TOL`` of 1 (i.e. approximately a unit quaternion).
    - **tof_cols**: at least one ToF cell is not ``-1`` somewhere in the sequence (any sensor/pixel).
    - **thm_cols**: at least one thermal column has more than one distinct value over the sequence
      (``nunique(dropna=True) > 1``).

    Pass ``None`` for a modality to skip that check. If all three are ``None``, no rows are removed.
    """
    if rot_cols is not None and len(rot_cols) == 0:
        raise ValueError("rot_cols must be non-empty when provided")
    if tof_cols is not None and len(tof_cols) == 0:
        raise ValueError("tof_cols must be non-empty when provided")
    if thm_cols is not None and len(thm_cols) == 0:
        raise ValueError("thm_cols must be non-empty when provided")

    if rot_cols is None and tof_cols is None and thm_cols is None:
        all_ids = df[sequence_id_col].unique().tolist()
        return df.reset_index(drop=True), all_ids

    gr = df.groupby(sequence_id_col, sort=False)

    def _rot_ok(g):
        sub = g[rot_cols].to_numpy(dtype=np.float64)
        if sub.size == 0:
            return False
        norms = np.linalg.norm(sub, axis=1)
        unit_row = np.isfinite(norms) & (np.abs(norms - 1.0) <= ROT_QUAT_UNIT_NORM_TOL)
        return bool(unit_row.any())

    def _tof_ok(g):
        return bool((g[tof_cols] != -1).any().any())

    def _thm_ok(g):
        nu = g[thm_cols].nunique(dropna=True)
        return bool((nu > 1).any())

    keep = None
    if rot_cols is not None:
        keep = gr.apply(_rot_ok)
    if tof_cols is not None:
        ok = gr.apply(_tof_ok)
        keep = ok if keep is None else (keep & ok)
    if thm_cols is not None:
        ok = gr.apply(_thm_ok)
        keep = ok if keep is None else (keep & ok)

    valid_seq_ids = keep[keep].index.tolist()
    out = df[df[sequence_id_col].isin(valid_seq_ids)].reset_index(drop=True)
    return out, valid_seq_ids


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
    """Save full checkpoint; optional extra keys (e.g. thermal metadata, dropout_p)."""
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
    """
    Load weights saved during training (raw state_dict on disk).

    Uses ``weights_only=False`` for trusted local checkpoints (PyTorch 2.6+ default
    would otherwise reject full pickles). Accepts either a bare ``state_dict`` or a
    dict containing ``model_state_dict`` (e.g. re-loading an already-full checkpoint).
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def make_supervised_single_tensor_batch_steps(
    model,
    criterion,
    optimizer,
    device,
    *,
    grad_clip_max_norm=1.0,
):
    """
    Build ``(train_step, val_step)`` for loaders that yield ``(features, labels)`` with
    ``model(features) -> logits`` (CrossEntropy loss, argmax accuracy).

    Used by acc-only, rot-only, thermal-only, Helios, and linear-acc trainers.
    """
    dev = device

    def train_step(batch):
        x, labels = batch
        x = x.to(dev)
        labels = labels.to(dev)
        outputs = model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        batch_acc = (outputs.argmax(1) == labels).float().mean()
        return loss.item(), batch_acc.item()

    def val_step(batch):
        vx, v_labels = batch
        vx = vx.to(dev)
        v_labels = v_labels.to(dev)
        val_outputs = model(vx)
        return criterion(val_outputs, v_labels).item(), (
            val_outputs.argmax(1) == v_labels
        ).float().mean().item()

    return train_step, val_step


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
    Epoch loop: ReduceLROnPlateau on val loss, early stopping on val accuracy, best
    weights saved as a raw ``state_dict`` at ``save_path``.

    ``train_step(batch)`` must ``backward()`` and ``optimizer.step()`` and return
    ``(loss_item, batch_mean_acc)``. ``val_step(batch)`` must return the same under
    ``torch.no_grad()`` (the engine wraps the val loop).
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

    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc * 100:.2f}%")
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
    """
    Run ``train_with_early_stopping``, reload best weights from ``filepath`` with
    ``load_best_weights_into_model``, then ``save_model_and_metadata`` (full checkpoint).
    """
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
