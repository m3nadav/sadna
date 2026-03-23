"""
Analyze Fusion_ToF_Multistream_CNN: same evaluations as analyze_fusion_multistream_cnn.py
(classification report, confusion matrix, misclassifications, per-class metrics,
confidence, region accuracy, ROC/AUC, predictions.npz).

Test or validation set: same subject split as training; no drop for missing ROT/ToF —
uses per-sequence has_rot / has_tof flags (matches train_fusion_tof_multistream_cnn.py).

When run as a script, you are prompted for test vs validation and baseline vs finetune model.
Artifacts are written under a directory derived from `OUTPUT_DIR` (base name) plus suffixes:
  baseline + test: `OUTPUT_DIR`
  baseline + val:  `OUTPUT_DIR` + `_val`
  finetune + test: `OUTPUT_DIR` + `_finetune`
  finetune + val:  `OUTPUT_DIR` + `_finetune_val`
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from train_multistream_cnn import (
    load_train_data,
    final_robust_split,
    apply_label_encoding,
    BFRB_GESTURES,
)
from train_tof_cnn import get_tof_columns
from train_fusion_tof_multistream_cnn import (
    make_fusion_frozen_128,
    make_tof_feature_extractor,
    ToFFusionMultistreamCNN,
    AccRotToFSequenceDataset,
    collate_acc_rot_tof,
    compute_sequence_sensor_stats,
    FUSION_CHECKPOINT_PATH,
    TOF_CHECKPOINT_PATH,
)
from train_fusion_tof_multistream_cnn_finetune import (
    make_fusion_finetune_128,
    make_tof_finetune_extractor,
    ToFFusionMultistreamCNNDeepFinetune,
    HEAD_HIDDEN_DIMS,
    HEAD_DROPOUT,
)

TRIPLE_CHECKPOINT_PATH = "models/deep_learning_models/Fusion_ToF_Multistream_CNN.pth"
FINETUNE_CHECKPOINT_PATH = "models/deep_learning_models/Fusion_ToF_Multistream_CNN_finetune.pth"
OUTPUT_DIR = "analysis_results/fusion_tof_multistream_cnn_conv2d"
# Effective path for this run; `main()` sets from `output_dir_for_split` before saving.
ACTIVE_OUTPUT_DIR = OUTPUT_DIR


def output_dir_for_split(split: str, variant: str = "baseline") -> str:
    """Resolve save directory: base `OUTPUT_DIR` plus val / finetune suffixes."""
    base = OUTPUT_DIR.rstrip("/")
    if variant == "finetune":
        if split == "val":
            return f"{base}_finetune_val"
        return f"{base}_finetune"
    if split == "val":
        return f"{base}_val"
    return base


def ensure_output_dir():
    os.makedirs(ACTIVE_OUTPUT_DIR, exist_ok=True)


def load_model_and_data(
    checkpoint_path=TRIPLE_CHECKPOINT_PATH,
    split: str = "test",
    variant: str = "baseline",
):
    """Rebuild split, eval set (no sensor drop), build model and load checkpoint (baseline or finetune arch)."""
    train_df = load_train_data()
    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )

    tof_cols = get_tof_columns(train_df)
    if split == "val":
        eval_df = valset_df
        split_tag = "Validation"
    else:
        eval_df = testset_df
        split_tag = "Test"
    print(
        f"{split_tag} set (all sequences): {len(eval_df)} rows, "
        f"{eval_df['sequence_id'].nunique()} sequences"
    )
    compute_sequence_sensor_stats(eval_df, tof_cols, f"{split_tag.upper()} (evaluation)")

    num_classes = len(le.classes_)
    target_names = le.classes_

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if variant == "finetune":
        print(
            f"Building finetune model (deep head + partial-unfreeze backbones) from "
            f"{FUSION_CHECKPOINT_PATH} + {TOF_CHECKPOINT_PATH}, weights from {checkpoint_path}..."
        )
        fusion_128 = make_fusion_finetune_128(num_classes, FUSION_CHECKPOINT_PATH, device, trainset_df)
        tof_feat = make_tof_finetune_extractor(num_classes, TOF_CHECKPOINT_PATH, device)
        model = ToFFusionMultistreamCNNDeepFinetune(
            fusion_128,
            tof_feat,
            num_classes=num_classes,
            head_hidden_dims=HEAD_HIDDEN_DIMS,
            head_dropout=HEAD_DROPOUT,
        )
    else:
        print(f"Building baseline model from {FUSION_CHECKPOINT_PATH} and {TOF_CHECKPOINT_PATH}...")
        fusion_128 = make_fusion_frozen_128(num_classes, FUSION_CHECKPOINT_PATH, device, trainset_df)
        tof_feat = make_tof_feature_extractor(num_classes, TOF_CHECKPOINT_PATH, device)
        model = ToFFusionMultistreamCNN(fusion_128, tof_feat, num_classes=num_classes, hidden_dim=64)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    eval_ds = AccRotToFSequenceDataset(eval_df, tof_cols)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_acc_rot_tof,
    )
    return model, eval_loader, target_names, num_classes, le, split_tag


def run_evaluation(model, test_loader, device):
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for acc, rot, tof_padded, lengths, has_rot, has_tof, labels in test_loader:
            acc = acc.to(device)
            rot = rot.to(device)
            tof_padded = tof_padded.to(device)
            lengths = lengths.to(device)
            has_rot = has_rot.to(device)
            has_tof = has_tof.to(device)
            outputs = model(acc, rot, tof_padded, lengths, has_rot, has_tof)
            all_outputs.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels, np.array(all_outputs)


def analysis_classification_report_and_cm(all_labels, all_preds, target_names, split_label="Test"):
    ensure_output_dir()
    report = classification_report(all_labels, all_preds, target_names=target_names)
    path_txt = os.path.join(ACTIVE_OUTPUT_DIR, "classification_report.txt")
    with open(path_txt, "w") as f:
        f.write(f"{split_label} Set Performance Analysis (Fusion ToF Multistream CNN)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Saved: {path_txt}")

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="viridis",
    )
    plt.title("Confusion Matrix: Fusion ToF Multistream CNN — BFRB vs. Non-BFRB Gestures")
    plt.ylabel("True Gesture")
    plt.xlabel("Predicted Gesture")
    path_fig = os.path.join(ACTIVE_OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")
    return cm, report


def analysis_misclassifications(cm, target_names):
    ensure_output_dir()
    misclassifications = []
    n = len(target_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                misclassifications.append(
                    {
                        "True_Gesture": target_names[i],
                        "Predicted_Gesture": target_names[j],
                        "Count": int(cm[i, j]),
                    }
                )
    misclassifications.sort(key=lambda x: x["Count"], reverse=True)

    lines = [
        "--- Analyzing Specific Misclassifications ---",
        "",
        "Top 10 Misclassified Pairs:",
    ]
    for i, mc in enumerate(misclassifications[:10]):
        lines.append(
            f"{i+1}. True: '{mc['True_Gesture']}' was misclassified as "
            f"Predicted: '{mc['Predicted_Gesture']}' (Count: {mc['Count']})"
        )
    lines.append("")
    lines.append("--- Key Observations from Misclassifications ---")

    predicted_as_eyelash = [mc for mc in misclassifications if mc["Predicted_Gesture"] == "Eyelash - pull hair"]
    if predicted_as_eyelash:
        to_join = [f"'{mc['True_Gesture']}'" for mc in predicted_as_eyelash[:3]]
        lines.append(
            f"\n'Eyelash - pull hair' is a common misprediction target. Gestures like {', '.join(to_join)} are often mistaken for it."
        )

    predicted_as_forehead = [mc for mc in misclassifications if mc["Predicted_Gesture"] == "Forehead - pull hairline"]
    if predicted_as_forehead:
        to_join = [f"'{mc['True_Gesture']}'" for mc in predicted_as_forehead[:3]]
        lines.append(
            f"'Forehead - pull hairline' also gets confused with others, including {', '.join(to_join)}."
        )

    lines.append("")
    lines.append("Further analysis often involves looking for:")
    lines.append("- Reciprocal confusions (e.g., A mistaken for B, and B mistaken for A)")
    lines.append("- Specific BFRB vs. non-BFRB confusions")

    text = "\n".join(lines)
    path_txt = os.path.join(ACTIVE_OUTPUT_DIR, "misclassifications_analysis.txt")
    with open(path_txt, "w") as f:
        f.write(text)
    print(f"Saved: {path_txt}")

    df_mc = pd.DataFrame(misclassifications)
    path_csv = os.path.join(ACTIVE_OUTPUT_DIR, "misclassifications.csv")
    df_mc.to_csv(path_csv, index=False)
    print(f"Saved: {path_csv}")
    return misclassifications


def analysis_per_class_metrics(all_labels, all_preds, target_names):
    ensure_output_dir()
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(target_names))), zero_division=0
    )
    metrics_df = pd.DataFrame(
        {
            "Class": target_names,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    ).sort_values("F1", ascending=True)

    path_csv = os.path.join(ACTIVE_OUTPUT_DIR, "per_class_metrics.csv")
    metrics_df.to_csv(path_csv, index=False)
    print(f"Saved: {path_csv}")

    x = np.arange(len(metrics_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, max(6, len(target_names) * 0.45)))
    ax.barh(x - width, metrics_df["Precision"], width, label="Precision", color="steelblue")
    ax.barh(x, metrics_df["Recall"], width, label="Recall", color="darkorange")
    ax.barh(x + width, metrics_df["F1"], width, label="F1", color="seagreen")
    ax.set_yticks(x)
    ax.set_yticklabels(metrics_df["Class"], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Score")
    ax.set_title("Per-Class Precision, Recall & F1 (sorted by F1) — Fusion ToF Multistream CNN")
    ax.legend()
    plt.tight_layout()
    path_fig = os.path.join(ACTIVE_OUTPUT_DIR, "per_class_precision_recall_f1.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    low_f1 = metrics_df[metrics_df["F1"] < 0.5][["Class", "Precision", "Recall", "F1", "Support"]]
    path_low = os.path.join(ACTIVE_OUTPUT_DIR, "classes_f1_below_0.5.txt")
    with open(path_low, "w") as f:
        f.write("Classes with F1 < 0.5:\n")
        f.write(low_f1.to_string(index=False))
    print(f"Saved: {path_low}")
    return metrics_df


def analysis_confidence(all_outputs, all_labels, all_preds, target_names):
    ensure_output_dir()
    all_probs = F.softmax(torch.tensor(all_outputs, dtype=torch.float32), dim=1).numpy()
    max_confidence = all_probs.max(axis=1)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    correct_mask = all_labels_arr == all_preds_arr

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        max_confidence[correct_mask],
        bins=30,
        alpha=0.6,
        color="seagreen",
        label=f"Correct (n={correct_mask.sum()})",
    )
    axes[0].hist(
        max_confidence[~correct_mask],
        bins=30,
        alpha=0.6,
        color="tomato",
        label=f"Wrong (n={(~correct_mask).sum()})",
    )
    axes[0].set_xlabel("Max Softmax Probability (Confidence)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Confidence: Correct vs. Wrong — Fusion ToF Multistream CNN")
    axes[0].legend()

    for mask, label, color in [(correct_mask, "Correct", "seagreen"), (~correct_mask, "Wrong", "tomato")]:
        if mask.sum() == 0:
            continue
        sorted_conf = np.sort(max_confidence[mask])
        cdf = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        axes[1].plot(sorted_conf, cdf, label=label, color=color)
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("Fraction of samples below threshold")
    axes[1].set_title("CDF of Confidence (Correct vs. Wrong)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path_fig = os.path.join(ACTIVE_OUTPUT_DIR, "confidence_analysis.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    mean_correct = max_confidence[correct_mask].mean() if correct_mask.any() else float("nan")
    mean_wrong = max_confidence[~correct_mask].mean() if (~correct_mask).any() else float("nan")
    frac_wrong_high_conf = (max_confidence[~correct_mask] > 0.9).mean() if (~correct_mask).any() else 0.0

    lines = [
        f"Mean confidence — Correct: {mean_correct:.3f} | Wrong: {mean_wrong:.3f}",
        f"Fraction of wrong predictions with confidence > 0.9: {frac_wrong_high_conf:.1%}  (high = model is confidently wrong)",
    ]
    text = "\n".join(lines)
    path_txt = os.path.join(ACTIVE_OUTPUT_DIR, "confidence_summary.txt")
    with open(path_txt, "w") as f:
        f.write(text)
    print(f"Saved: {path_txt}")
    return all_probs


def analysis_region_accuracy(all_labels, all_preds, target_names):
    ensure_output_dir()

    def get_region(class_name):
        return class_name.split(" - ")[0].strip() if " - " in class_name else class_name

    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    true_class_names = [target_names[i] for i in all_labels_arr]
    pred_class_names = [target_names[i] for i in all_preds_arr]
    true_regions = [get_region(n) for n in true_class_names]

    region_results = pd.DataFrame(
        {
            "true_class": true_class_names,
            "pred_class": pred_class_names,
            "true_region": true_regions,
            "correct": all_labels_arr == all_preds_arr,
        }
    )
    region_acc = (
        region_results.groupby("true_region")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct", "count": "total"})
    )
    region_acc["accuracy"] = region_acc["correct"] / region_acc["total"]
    region_acc = region_acc.sort_values("accuracy")

    path_csv = os.path.join(ACTIVE_OUTPUT_DIR, "region_accuracy.csv")
    region_acc.to_csv(path_csv)
    print(f"Saved: {path_csv}")

    fig, ax = plt.subplots(figsize=(9, max(4, len(region_acc) * 0.5)))
    colors = ["tomato" if a < 0.5 else "gold" if a < 0.7 else "seagreen" for a in region_acc["accuracy"]]
    ax.barh(region_acc.index, region_acc["accuracy"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy Grouped by Body Region — Fusion ToF Multistream CNN")
    for i, (region, row) in enumerate(region_acc.iterrows()):
        ax.text(row["accuracy"] + 0.01, i, f"{row['accuracy']:.0%}  (n={int(row['total'])})", va="center", fontsize=8)
    plt.tight_layout()
    path_fig = os.path.join(ACTIVE_OUTPUT_DIR, "region_accuracy.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    path_txt = os.path.join(ACTIVE_OUTPUT_DIR, "region_accuracy_table.txt")
    with open(path_txt, "w") as f:
        f.write("Per-region accuracy table:\n")
        f.write(region_acc[["correct", "total", "accuracy"]].to_string())
    print(f"Saved: {path_txt}")
    return region_acc


def analysis_roc_curves(all_labels, all_probs, target_names):
    ensure_output_dir()
    n_classes = len(target_names)
    y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    y_score = all_probs

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = plt.get_cmap("tab20")
    ncols = 3
    nrows = int(np.ceil((n_classes + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i in range(n_classes):
        ax = axes[i]
        ax.plot(fpr[i], tpr[i], color=cmap(i), lw=1.8, label=f"AUC = {roc_auc[i]:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_title(target_names[i], fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)

    macro_ax = axes[n_classes]
    macro_ax.plot(all_fpr, mean_tpr, "navy", lw=2, label=f"Macro AUC = {macro_auc:.3f}")
    macro_ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    macro_ax.set_title("Macro-Average ROC", fontsize=9)
    macro_ax.legend(fontsize=8, loc="lower right")
    macro_ax.set_xlabel("FPR")
    macro_ax.set_ylabel("TPR")

    for j in range(n_classes + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("One-vs-Rest ROC Curves per Gesture Class — Fusion ToF Multistream CNN", fontsize=12, y=1.01)
    plt.tight_layout()
    path_fig = os.path.join(ACTIVE_OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    auc_summary = pd.DataFrame(
        {
            "Class": target_names,
            "AUC": [roc_auc[i] for i in range(n_classes)],
        }
    ).sort_values("AUC")
    path_csv = os.path.join(ACTIVE_OUTPUT_DIR, "roc_auc_per_class.csv")
    auc_summary.to_csv(path_csv, index=False)
    print(f"Saved: {path_csv}")

    path_txt = os.path.join(ACTIVE_OUTPUT_DIR, "roc_auc_summary.txt")
    with open(path_txt, "w") as f:
        f.write(f"Macro-average AUC: {macro_auc:.4f}\n")
        f.write("\nPer-class AUC (lowest first):\n")
        f.write(auc_summary.to_string(index=False))
    print(f"Saved: {path_txt}")
    return macro_auc, auc_summary


def save_predictions(all_labels, all_preds, all_outputs, all_probs):
    ensure_output_dir()
    np.savez(
        os.path.join(ACTIVE_OUTPUT_DIR, "predictions.npz"),
        all_labels=np.array(all_labels),
        all_preds=np.array(all_preds),
        all_outputs=all_outputs,
        all_probs=all_probs,
    )
    print(f"Saved: {os.path.join(ACTIVE_OUTPUT_DIR, 'predictions.npz')}")


def main(checkpoint_path=None, split="test", variant="baseline"):
    global ACTIVE_OUTPUT_DIR
    ACTIVE_OUTPUT_DIR = output_dir_for_split(split, variant)
    if checkpoint_path is None:
        checkpoint_path = FINETUNE_CHECKPOINT_PATH if variant == "finetune" else TRIPLE_CHECKPOINT_PATH
    print(f"Loading model ({variant}) and {split} data... (checkpoint: {checkpoint_path}, output: {ACTIVE_OUTPUT_DIR})")
    model, eval_loader, target_names, num_classes, le, split_tag = load_model_and_data(
        checkpoint_path, split=split, variant=variant
    )
    device = next(model.parameters()).device

    print("Running evaluation...")
    all_preds, all_labels, all_outputs = run_evaluation(model, eval_loader, device)
    all_probs = F.softmax(torch.tensor(all_outputs, dtype=torch.float32), dim=1).numpy()

    print(f"\n{split_tag} Set Performance:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    ensure_output_dir()
    print("\n--- Saving analysis results ---")

    cm, _ = analysis_classification_report_and_cm(
        all_labels, all_preds, target_names, split_label=split_tag
    )
    analysis_misclassifications(cm, target_names)
    analysis_per_class_metrics(all_labels, all_preds, target_names)
    analysis_confidence(all_outputs, all_labels, all_preds, target_names)
    analysis_region_accuracy(all_labels, all_preds, target_names)
    analysis_roc_curves(all_labels, all_probs, target_names)
    save_predictions(all_labels, all_preds, all_outputs, all_probs)

    print(f"\nAll results saved under: {os.path.abspath(ACTIVE_OUTPUT_DIR)}")


def _prompt_split() -> str:
    while True:
        raw = input("Evaluate on test or validation set? [t]est / [v]al (default t): ").strip().lower()
        if raw in ("", "t", "test"):
            return "test"
        if raw in ("v", "val", "validation"):
            return "val"
        print("Invalid choice. Enter 't' for test or 'v' for validation.")


def _prompt_variant() -> str:
    while True:
        raw = input("Model: [b]aseline or [f]inetune (default b): ").strip().lower()
        if raw in ("", "b", "baseline"):
            return "baseline"
        if raw in ("f", "finetune"):
            return "finetune"
        print("Invalid choice. Enter 'b' for baseline or 'f' for finetune.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--auto":
        split = sys.argv[2] if len(sys.argv) > 2 else "test"
        variant = sys.argv[3] if len(sys.argv) > 3 else "baseline"
        main(split=split, variant=variant)
    else:
        split = _prompt_split()
        variant = _prompt_variant()
        main(split=split, variant=variant)
