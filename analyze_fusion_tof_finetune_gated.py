"""
Evaluate Fusion_ToF_Multistream_CNN_finetune with specialist MLP gating (fusion_tof_specialists_gating.json).

For each sequence: main finetune forward → if margin p1-p2 < tau_B and top-1 & top-2 classes ∈ block B,
replace prediction with specialist_B(argmax on fused 256-d features). Blocks are disjoint.

Runs the same reporting pipeline as analyze_fusion_tof_multistream_cnn.py (reports, CM, ROC, etc.):
- Classification / confusion / region metrics use **gated** predictions.
- Confidence histograms and ROC use **main model** logits/probs (documented); gated max-prob would mix two heads.

Outputs:
  analysis_results/fusion_tof_finetune_gated_val_conv2d/
  analysis_results/fusion_tof_finetune_gated_test_conv2d/

Usage:
  python analyze_fusion_tof_finetune_gated.py           # both val and test
  python analyze_fusion_tof_finetune_gated.py val       # validation only
  python analyze_fusion_tof_finetune_gated.py test     # test only
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import analyze_fusion_tof_multistream_cnn as afc
from analyze_fusion_tof_multistream_cnn import (
    load_model_and_data,
    FINETUNE_CHECKPOINT_PATH,
)
from train_fusion_tof_specialists import SpecialistMLP, SPECIALIST_HIDDEN, SPECIALIST_DROPOUT, FEAT_DIM

GATING_JSON = "models/deep_learning_models/fusion_tof_specialists_gating.json"


def output_dir_for_gated_split(split: str) -> str:
    if split == "val":
        return "analysis_results/fusion_tof_finetune_gated_val_conv2d"
    return "analysis_results/fusion_tof_finetune_gated_test_conv2d"


def load_gating_and_specialists(device: torch.device, config_path: str = GATING_JSON):
    with open(config_path) as f:
        cfg = json.load(f)
    blocks = cfg["blocks"]
    specialists = {}
    for b in blocks:
        name = b["name"]
        n_local = len(b["local_to_global"])
        m = SpecialistMLP(FEAT_DIM, SPECIALIST_HIDDEN, n_local, SPECIALIST_DROPOUT).to(device)
        ck = b["checkpoint"]
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
        m.eval()
        specialists[name] = m
    return cfg, specialists, blocks


@torch.no_grad()
def run_gated_evaluation(model, specialists: dict, blocks: list, loader, device):
    all_labels = []
    all_outputs_main = []
    all_preds_main = []
    all_preds_gated = []
    specialist_applied = []
    per_block_fires = {b["name"]: 0 for b in blocks}

    for acc, rot, tof_padded, lengths, has_rot, has_tof, labels in loader:
        acc = acc.to(device)
        rot = rot.to(device)
        tof_padded = tof_padded.to(device)
        lengths = lengths.to(device)
        has_rot = has_rot.to(device)
        has_tof = has_tof.to(device)
        labels = labels.to(device)

        outputs = model(acc, rot, tof_padded, lengths, has_rot, has_tof)
        probs = F.softmax(outputs, dim=1)
        top2 = torch.topk(probs, k=2, dim=1)
        p1, p2 = top2.values[:, 0], top2.values[:, 1]
        c1, c2 = top2.indices[:, 0], top2.indices[:, 1]
        margin = p1 - p2
        z = model.fused_features(acc, rot, tof_padded, lengths, has_rot, has_tof)

        pred_main = outputs.argmax(dim=1)
        pred = pred_main.clone()
        applied = torch.zeros(outputs.shape[0], dtype=torch.bool, device=device)

        for b in blocks:
            name = b["name"]
            tau = float(b["margin_threshold_tau"])
            idx_set = torch.tensor(b["class_indices"], device=device, dtype=torch.long)
            spec = specialists[name]
            ltg = b["local_to_global"]

            m_ok = margin < tau
            c1_ok = torch.isin(c1, idx_set)
            c2_ok = torch.isin(c2, idx_set)
            mask = m_ok & c1_ok & c2_ok
            if not mask.any():
                continue

            logits_s = spec(z[mask])
            loc = logits_s.argmax(dim=1)
            new_idx = torch.tensor([ltg[j] for j in loc.cpu().tolist()], device=device, dtype=torch.long)
            pred[mask] = new_idx
            applied[mask] = True
            per_block_fires[name] += int(mask.sum().item())

        all_outputs_main.append(outputs.cpu().numpy())
        all_preds_main.append(pred_main.cpu().numpy())
        all_preds_gated.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        specialist_applied.append(applied.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs_main = np.vstack(all_outputs_main)
    all_preds_main = np.concatenate(all_preds_main)
    all_preds_gated = np.concatenate(all_preds_gated)
    specialist_applied = np.concatenate(specialist_applied)

    return (
        all_preds_gated,
        all_preds_main,
        all_labels,
        all_outputs_main,
        specialist_applied,
        per_block_fires,
    )


def run_one_split(split: str, cfg: dict, specialists: dict, blocks: list):
    split_tag = "Validation" if split == "val" else "Test"
    out_dir = output_dir_for_gated_split(split)
    os.makedirs(out_dir, exist_ok=True)

    # Analysis helpers write to ACTIVE_OUTPUT_DIR, not OUTPUT_DIR (see analyze_fusion_tof_multistream_cnn).
    afc.OUTPUT_DIR = out_dir
    afc.ACTIVE_OUTPUT_DIR = out_dir

    model, eval_loader, target_names, num_classes, le, _ = load_model_and_data(
        FINETUNE_CHECKPOINT_PATH,
        split=split,
        variant="finetune",
    )
    device = next(model.parameters()).device

    print(f"\n=== Gated evaluation: {split_tag} → {out_dir} ===")
    preds_g, preds_m, labels, out_main, spec_on, fires = run_gated_evaluation(
        model, specialists, blocks, eval_loader, device
    )
    n = len(labels)
    acc_main = (preds_m == labels).mean()
    acc_g = (preds_g == labels).mean()
    print(f"Accuracy main (finetune only): {acc_main:.4f}")
    print(f"Accuracy gated (finetune + specialists): {acc_g:.4f}")
    print(f"Specialist gate fired on {spec_on.sum()} / {n} sequences ({100*spec_on.mean():.2f}%)")
    for bn, c in fires.items():
        print(f"  fires [{bn}]: {c}")

    all_probs = F.softmax(torch.tensor(out_main, dtype=torch.float32), dim=1).numpy()

    # Method summary
    lines = [
        f"Fusion ToF Finetune + specialist gating — {split_tag} set",
        "",
        f"Backbone: {cfg.get('backbone_checkpoint', '')}",
        f"Gating: {GATING_JSON}",
        "",
        str(cfg.get("gating_rule", "")),
        "",
        f"Accuracy (main finetune argmax): {acc_main:.4f}",
        f"Accuracy (gated): {acc_g:.4f}",
        f"Sequences with specialist override: {int(spec_on.sum())} / {n} ({100*spec_on.mean():.2f}%)",
        "",
        "Per-block gate fires (each sequence counted at most once; blocks are class-disjoint):",
    ]
    for bn, c in sorted(fires.items()):
        lines.append(f"  {bn}: {c}")
    lines.extend(
        [
            "",
            "Note: confidence_analysis and ROC use MAIN model logits only; "
            "classification_report and confusion_matrix use GATED predictions.",
        ]
    )
    with open(os.path.join(out_dir, "gated_method_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    cm, _ = afc.analysis_classification_report_and_cm(
        labels.tolist(), preds_g.tolist(), target_names, split_label=split_tag
    )
    rep_path = os.path.join(out_dir, "classification_report.txt")
    with open(rep_path) as f:
        txt = f.read()
    txt = txt.replace(
        "Fusion ToF Multistream CNN",
        "Fusion ToF Finetune + specialist gating (gated preds)",
        1,
    )
    with open(rep_path, "w") as f:
        f.write(txt)
    afc.analysis_misclassifications(cm, target_names)
    afc.analysis_per_class_metrics(labels.tolist(), preds_g.tolist(), target_names)
    afc.analysis_confidence(out_main, labels.tolist(), preds_g.tolist(), target_names)
    afc.analysis_region_accuracy(labels.tolist(), preds_g.tolist(), target_names)
    afc.analysis_roc_curves(labels.tolist(), all_probs, target_names)

    afc.ensure_output_dir()
    np.savez(
        os.path.join(out_dir, "predictions.npz"),
        all_labels=labels,
        all_preds_gated=preds_g,
        all_preds_main=preds_m,
        specialist_applied=spec_on,
        all_outputs=out_main,
        all_probs=all_probs,
    )
    print(f"Saved: {os.path.join(out_dir, 'predictions.npz')}")
    print(f"All results under: {os.path.abspath(out_dir)}")


def main():
    if not os.path.isfile(GATING_JSON):
        raise FileNotFoundError(f"Missing {GATING_JSON}. Run train_fusion_tof_specialists.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, specialists, blocks = load_gating_and_specialists(device)

    splits = ["val", "test"]
    if len(sys.argv) >= 2:
        s = sys.argv[1].lower().strip()
        if s in ("val", "validation"):
            splits = ["val"]
        elif s in ("test",):
            splits = ["test"]
        else:
            print("Usage: python analyze_fusion_tof_finetune_gated.py [val|test]")
            sys.exit(1)

    for split in splits:
        run_one_split(split, cfg, specialists, blocks)


if __name__ == "__main__":
    main()
