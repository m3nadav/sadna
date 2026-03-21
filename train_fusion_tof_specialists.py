"""
Train small MLP specialists on frozen 256-d fused features from Fusion_ToF_Multistream_CNN_finetune.

Pipeline:
1. Ensures validation analysis exists (runs analyze_fusion_tof_multistream_cnn.py --auto val finetune if needed).
2. Loads val predictions.npz to derive per-block gating: top-k (smallest k with high recall on val
   for samples whose true label is in the block) and margin threshold tau (percentile of p1-p2 on
   within-block errors).
3. Specialist blocks are chosen from validation misclassification structure (tight confusion clusters).
4. Each specialist is trained only on train sequences whose true label lies in that block.

Gating (inference): if (p_(1) - p_(2) < tau) and top-1 & top-2 global class indices are both in the
block, replace main argmax with specialist argmax (local head mapped back to global labels).

Saves:
  models/deep_learning_models/fusion_tof_specialists_gating.json
  models/deep_learning_models/fusion_tof_specialist_<block_name>.pth
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_multistream_cnn import load_train_data, final_robust_split, apply_label_encoding, BFRB_GESTURES
from train_tof_cnn import get_tof_columns
from train_fusion_tof_multistream_cnn import AccRotToFSequenceDataset, collate_acc_rot_tof
from train_fusion_tof_multistream_cnn_finetune import (
    SAVE_PATH as FINETUNE_CHECKPOINT_PATH,
    make_fusion_finetune_128,
    make_tof_finetune_extractor,
    ToFFusionMultistreamCNNDeepFinetune,
    HEAD_HIDDEN_DIMS,
    HEAD_DROPOUT,
    FUSION_CHECKPOINT_PATH,
    TOF_CHECKPOINT_PATH,
)

VAL_ANALYSIS_DIR = Path("analysis_results/fusion_tof_multistream_cnn_finetune_val")
VAL_PREDICTIONS_NPZ = VAL_ANALYSIS_DIR / "predictions.npz"
GATING_JSON = Path("models/deep_learning_models/fusion_tof_specialists_gating.json")
SPECIALIST_DIR = Path("models/deep_learning_models")

# Blocks derived from validation confusion / misclassification clusters (finetune val run).
SPECIALIST_BLOCK_SPECS: list[dict] = [
    {
        "name": "neck_scratch_pinch",
        "gestures": [
            "Neck - scratch",
            "Neck - pinch skin",
        ],
    },
    {
        "name": "upper_face_bfrb",
        "gestures": [
            "Above ear - pull hair",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Cheek - pinch skin",
        ],
    },
    {
        "name": "leg_skin_write_leg",
        "gestures": [
            "Pinch knee/leg skin",
            "Scratch knee/leg skin",
            "Write name on leg",
        ],
    },
    {
        "name": "air_arm_motion",
        "gestures": [
            "Pull air toward your face",
            "Write name in air",
            "Wave hello",
        ],
    },
]

FEAT_DIM = 256
SPECIALIST_HIDDEN = 64
SPECIALIST_DROPOUT = 0.2
RECALL_TARGET = 0.92
MARGIN_PERCENTILE_ON_WBE = 75.0  # within-block errors: tau = percentile(margin)
MARGIN_FALLBACK_PERCENTILE = 50.0
MAX_TOP_K = 5
RANDOM_STATE = 42


def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def ensure_val_analysis():
    VAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not VAL_PREDICTIONS_NPZ.is_file():
        print("Running validation analysis for finetune model (generates predictions.npz)...")
        subprocess.run(
            [sys.executable, "analyze_fusion_tof_multistream_cnn.py", "--auto", "val", "finetune"],
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    if not VAL_PREDICTIONS_NPZ.is_file():
        raise FileNotFoundError(f"Missing {VAL_PREDICTIONS_NPZ} after analysis run.")


def filter_df_by_label_set(df: pd.DataFrame, allowed_codes: set[int]) -> pd.DataFrame:
    keep_ids = []
    for sid, g in df.groupby("sequence_id", sort=False):
        c = int(g["gesture_encoded"].iloc[0])
        if c in allowed_codes:
            keep_ids.append(sid)
    return df[df["sequence_id"].isin(keep_ids)].reset_index(drop=True)


class SpecialistMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_frozen_finetune_model(num_classes: int, device: torch.device) -> ToFFusionMultistreamCNNDeepFinetune:
    fusion_128 = make_fusion_finetune_128(num_classes, FUSION_CHECKPOINT_PATH, device)
    tof_feat = make_tof_finetune_extractor(num_classes, TOF_CHECKPOINT_PATH, device)
    model = ToFFusionMultistreamCNNDeepFinetune(
        fusion_128,
        tof_feat,
        num_classes=num_classes,
        head_hidden_dims=HEAD_HIDDEN_DIMS,
        head_dropout=HEAD_DROPOUT,
    ).to(device)
    ckpt = torch.load(FINETUNE_CHECKPOINT_PATH, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def forward_fused_features(
    backbone: ToFFusionMultistreamCNNDeepFinetune,
    acc,
    rot,
    tof_padded,
    lengths,
    has_rot,
    has_tof,
    device,
):
    acc = acc.to(device)
    rot = rot.to(device)
    tof_padded = tof_padded.to(device)
    lengths = lengths.to(device)
    has_rot = has_rot.to(device)
    has_tof = has_tof.to(device)
    return backbone.fused_features(acc, rot, tof_padded, lengths, has_rot, has_tof)


def derive_top_k_for_block(
    probs: np.ndarray,
    labels: np.ndarray,
    block_indices: set[int],
) -> tuple[int, float]:
    """Smallest k in [2, MAX_TOP_K] such that recall (true in top-k | true in block) >= RECALL_TARGET."""
    labels = labels.astype(int)
    order = np.argsort(-probs, axis=1)
    in_block = np.array([int(y) in block_indices for y in labels])
    if in_block.sum() == 0:
        return 2, float("nan")
    best_k, best_r = MAX_TOP_K, 0.0
    for k in range(2, MAX_TOP_K + 1):
        hits = []
        for i in range(len(labels)):
            if not in_block[i]:
                continue
            topk = set(order[i, :k].tolist())
            hits.append(int(labels[i]) in topk)
        r = float(np.mean(hits)) if hits else 0.0
        best_r = r
        if r >= RECALL_TARGET:
            return k, r
        best_k = k
    return best_k, best_r


def derive_margin_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    block_indices: set[int],
) -> tuple[float, dict]:
    """Ambiguity threshold on validation: percentile of (p1-p2) for within-block errors."""
    labels = labels.astype(int)
    preds = preds.astype(int)
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    wbe = np.array(
        [
            (int(y) in block_indices) and (int(p) in block_indices) and (int(y) != int(p))
            for y, p in zip(labels, preds)
        ]
    )
    info = {
        "n_val": int(len(labels)),
        "n_within_block_errors": int(wbe.sum()),
    }
    if wbe.sum() >= 5:
        tau = float(np.percentile(margin[wbe], MARGIN_PERCENTILE_ON_WBE))
        info["tau_source"] = f"within_block_errors_p{int(MARGIN_PERCENTILE_ON_WBE)}"
    else:
        in_b = np.array([int(y) in block_indices for y in labels])
        if in_b.sum() >= 5:
            tau = float(np.percentile(margin[in_b], MARGIN_FALLBACK_PERCENTILE))
            info["tau_source"] = f"all_block_true_labels_p{int(MARGIN_FALLBACK_PERCENTILE)}"
        else:
            tau = float(np.percentile(margin, 40))
            info["tau_source"] = "global_p40_fallback"
    info["tau"] = tau
    return tau, info


def train_one_specialist(
    backbone: ToFFusionMultistreamCNNDeepFinetune,
    train_loader: DataLoader,
    val_loader: DataLoader,
    global_to_local: dict[int, int],
    num_local: int,
    device: torch.device,
    epochs: int = 80,
    patience: int = 12,
    lr: float = 1e-3,
) -> SpecialistMLP:
    spec = SpecialistMLP(FEAT_DIM, SPECIALIST_HIDDEN, num_local, SPECIALIST_DROPOUT).to(device)
    opt = optim.AdamW(spec.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_state = None
    best_val = 0.0
    bad = 0

    for ep in range(epochs):
        spec.train()
        tr_loss = 0.0
        tr_n = 0
        tr_acc = 0.0
        for acc, rot, tof, lens, hr, ht, y in train_loader:
            y = y.to(device)
            y_local = torch.tensor([global_to_local[int(c)] for c in y.cpu().tolist()], device=device)
            with torch.no_grad():
                z = forward_fused_features(backbone, acc, rot, tof, lens, hr, ht, device)
            opt.zero_grad()
            logits = spec(z)
            loss = crit(logits, y_local)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            tr_n += 1
            tr_acc += (logits.argmax(1) == y_local).float().mean().item()

        spec.eval()
        va = 0.0
        vn = 0
        with torch.no_grad():
            for acc, rot, tof, lens, hr, ht, y in val_loader:
                y = y.to(device)
                y_local = torch.tensor([global_to_local[int(c)] for c in y.cpu().tolist()], device=device)
                z = forward_fused_features(backbone, acc, rot, tof, lens, hr, ht, device)
                logits = spec(z)
                va += (logits.argmax(1) == y_local).float().mean().item()
                vn += 1
        val_acc = va / max(vn, 1)
        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in spec.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"    ep {ep+1}/{epochs} train_loss={tr_loss/max(tr_n,1):.4f} "
                f"train_acc={tr_acc/max(tr_n,1):.3f} val_acc={val_acc:.3f} best={best_val:.3f}"
            )
        if bad >= patience:
            print(f"    early stop at ep {ep+1}")
            break

    if best_state is not None:
        spec.load_state_dict(best_state)
    return spec


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    ensure_val_analysis()

    data = np.load(VAL_PREDICTIONS_NPZ)
    val_probs = np.asarray(data["all_probs"], dtype=np.float64)
    val_labels = np.asarray(data["all_labels"], dtype=int)
    val_preds = np.asarray(data["all_preds"], dtype=int)

    train_df = load_train_data()
    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)
    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )
    class_names = list(le.classes_)
    num_classes = len(class_names)
    name_to_idx = {g: gesture_map[g] for g in class_names}

    tof_cols = get_tof_columns(train_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading frozen finetune backbone from {FINETUNE_CHECKPOINT_PATH}...")
    backbone = load_frozen_finetune_model(num_classes, device)

    blocks_out: list[dict] = []
    SPECIALIST_DIR.mkdir(parents=True, exist_ok=True)

    for spec in SPECIALIST_BLOCK_SPECS:
        name = spec["name"]
        gestures = spec["gestures"]
        missing = [g for g in gestures if g not in name_to_idx]
        if missing:
            raise KeyError(f"Block {name}: unknown gestures {missing}")
        global_indices = sorted(name_to_idx[g] for g in gestures)
        block_set = set(global_indices)
        global_to_local = {g: i for i, g in enumerate(global_indices)}
        num_local = len(global_indices)

        k, recall_k = derive_top_k_for_block(val_probs, val_labels, block_set)
        tau, tau_info = derive_margin_threshold(val_probs, val_labels, val_preds, block_set)

        print(f"\n=== Block {name} ===")
        print(f"  classes ({num_local}): {[class_names[i] for i in global_indices]}")
        print(f"  gating: top_k={k} (recall true-in-topk | y in block on val: {recall_k:.3f})")
        print(f"  gating: margin tau={tau:.4f} ({tau_info})")

        train_sub = filter_df_by_label_set(trainset_df, block_set)
        val_sub = filter_df_by_label_set(valset_df, block_set)
        print(f"  train seq: {train_sub['sequence_id'].nunique()}, val seq: {val_sub['sequence_id'].nunique()}")
        if train_sub["sequence_id"].nunique() < 2:
            print("  skip: not enough train sequences")
            continue

        train_ds = AccRotToFSequenceDataset(train_sub, tof_cols)
        val_ds = AccRotToFSequenceDataset(val_sub, tof_cols)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_acc_rot_tof)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_acc_rot_tof)

        specialist = train_one_specialist(
            backbone, train_loader, val_loader, global_to_local, num_local, device
        )
        ckpt_path = SPECIALIST_DIR / f"fusion_tof_specialist_{name}.pth"
        torch.save(specialist.state_dict(), ckpt_path)
        print(f"  saved {ckpt_path}")

        local_to_global = [int(g) for g in global_indices]
        blocks_out.append(
            {
                "name": name,
                "gestures": [class_names[i] for i in global_indices],
                "class_indices": global_indices,
                "local_to_global": local_to_global,
                "top_k": int(k),
                "recall_true_in_topk_given_y_in_block_val": float(recall_k),
                "margin_threshold_tau": float(tau),
                "margin_threshold_meta": tau_info,
                "checkpoint": str(ckpt_path),
            }
        )

    config = {
        "backbone_checkpoint": FINETUNE_CHECKPOINT_PATH,
        "fusion_pretrained": FUSION_CHECKPOINT_PATH,
        "tof_pretrained": TOF_CHECKPOINT_PATH,
        "feat_dim": FEAT_DIM,
        "class_names": class_names,
        "gesture_map": {k: int(v) for k, v in gesture_map.items()},
        "gating_rule": (
            "Let p1>=p2 be the two largest main-model softmax probabilities and c1,c2 the corresponding class indices. "
            "If (p1-p2) < tau_B AND {c1,c2} ⊆ B.class_indices, run specialist MLP on frozen fused_features (256-d) "
            "and replace prediction with local_to_global[argmax(specialist_logits)]. Otherwise keep main argmax. "
            "Field top_k is the smallest k in [2,5] on validation such that P(true ∈ top-k | y in block) >= recall_target; "
            "use it to audit coverage or for extended gating (e.g. require true ∈ top-k before override)."
        ),
        "recall_target_for_k": RECALL_TARGET,
        "blocks": blocks_out,
    }
    GATING_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(GATING_JSON, "w") as f:
        json.dump(to_json_serializable(config), f, indent=2)
    print(f"\nWrote gating + metadata: {GATING_JSON.resolve()}")


if __name__ == "__main__":
    main()
