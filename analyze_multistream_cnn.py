"""
Analyze Multistream CNN checkpoints (accelerometer or thermal): same evaluations as in Project.ipynb;
save figures, CSV, and text under the selected profile's OUTPUT_DIR.

Run without args for an interactive prompt, or pass: 1/acc or 2/thermal.
"""
import os
import sys
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

# Reuse data loading, split, model and dataset from training script
from train_multistream_cnn import (
    RANDOM_STATE,
    load_train_data,
    final_robust_split,
    apply_label_encoding,
    BFRB_GESTURES,
    MultistreamCNNInertialNet,
    InertialSequenceDataset,
    collate_fn,
    compute_acc_zscore_stats,
)

# Presets: checkpoint path, analysis output folder, and plot/report titles.
MODEL_PROFILES = {
    "acc": {
        "key": "acc",
        "checkpoint": "models/deep_learning_models/Multistream_CNN_acc_only.pth",
        "output_dir": os.path.join("analysis_results", "Multistream_CNN_acc_only"),
        "plot_title": "Multistream CNN (acc-only)",
    },
    "thermal": {
        "key": "thermal",
        "checkpoint": "models/deep_learning_models/Multistream_CNN_thermal_only.pth",
        "output_dir": os.path.join("analysis_results", "Multistream_CNN_thermal_only"),
        "plot_title": "Multistream CNN (thermal)",
    },
}

# Set in main() from the chosen profile; all saved files go under OUTPUT_DIR.
OUTPUT_DIR = MODEL_PROFILES["acc"]["output_dir"]
PROFILE = None


def plot_title():
    """Human-readable model label for figures and report headers."""
    return PROFILE["plot_title"] if PROFILE else "Multistream CNN"


def prompt_model_profile():
    print("\nWhich model do you want to analyze?")
    print("  1 — Accelerometer (Multistream_CNN_acc_only.pth)")
    print("  2 — Thermal / thermopile (Multistream_CNN_thermal_only.pth)")
    while True:
        raw = input("Enter 1 or 2: ").strip()
        if raw == "1":
            return MODEL_PROFILES["acc"]
        if raw == "2":
            return MODEL_PROFILES["thermal"]
        print("Invalid choice. Please enter 1 or 2.")


def resolve_model_profile():
    """Interactive menu, or CLI: python analyze_multistream_cnn.py [1|2|acc|thermal]."""
    if len(sys.argv) > 1:
        a = sys.argv[1].lower().strip()
        if a in ("1", "acc", "accelerometer", "imu"):
            return MODEL_PROFILES["acc"]
        if a in ("2", "thermal", "thm", "temperature"):
            return MODEL_PROFILES["thermal"]
        print(f"Unknown argument {sys.argv[1]!r}; expected 1, 2, acc, or thermal. Using interactive prompt.\n")
    return prompt_model_profile()


def out_path(*parts):
    """Path under OUTPUT_DIR; use this for every saved analysis file."""
    return os.path.join(OUTPUT_DIR, *parts)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_data(profile, checkpoint_path=None):
    """Load checkpoint (full or state_dict only), rebuild split and label encoding."""
    checkpoint_path = checkpoint_path or profile["checkpoint"]
    train_df = load_train_data()
    train_df["is_bfrb"] = train_df["gesture"].isin(BFRB_GESTURES)
    trainset_df, valset_df, testset_df = final_robust_split(train_df)
    trainset_df = trainset_df.reset_index(drop=True)
    valset_df = valset_df.reset_index(drop=True)
    testset_df = testset_df.reset_index(drop=True)

    train_df, trainset_df, valset_df, testset_df, le, gesture_map = apply_label_encoding(
        train_df, trainset_df, valset_df, testset_df
    )
    num_classes = len(le.classes_)
    target_names = le.classes_

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if profile["key"] == "acc":
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        n_cls = ckpt["num_classes"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else num_classes
        if isinstance(ckpt, dict) and "input_zscore_mean" in ckpt and "input_zscore_std" in ckpt:
            m = np.asarray(ckpt["input_zscore_mean"], dtype=np.float32)
            s = np.asarray(ckpt["input_zscore_std"], dtype=np.float32)
        elif "norm.mean" in sd:
            m = sd["norm.mean"].detach().cpu().numpy()
            s = sd["norm.std"].detach().cpu().numpy()
        else:
            m, s = compute_acc_zscore_stats(trainset_df)
        model = MultistreamCNNInertialNet(num_classes=n_cls, norm_mean=m, norm_std=s)
        model.load_state_dict(sd, strict="norm.mean" in sd)
        model = model.to(device)
        model.eval()

        test_ds = InertialSequenceDataset(testset_df)
        test_loader = DataLoader(
            test_ds,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return model, test_loader, target_names, num_classes, le

    # Thermal: same split + drop sequences with no valid thermal data (matches training script).
    from train_multistream_cnn_thermal import (
        get_thm_columns,
        drop_sequences_with_no_thermal,
        compute_train_thermal_zscore_params,
        MultistreamCNNThermalNet,
        ThermalSequenceDataset,
        make_collate_fn,
        DROPOUT_P as THERMAL_DROPOUT_DEFAULT,
    )

    thm_cols = get_thm_columns(train_df)
    if not thm_cols:
        raise ValueError("No thm_* columns in train.csv; cannot evaluate thermal model.")

    trainset_df, _ = drop_sequences_with_no_thermal(trainset_df, thm_cols)
    valset_df, _ = drop_sequences_with_no_thermal(valset_df, thm_cols)
    testset_df, _ = drop_sequences_with_no_thermal(testset_df, thm_cols)

    if len(testset_df) == 0:
        raise RuntimeError("Test split has no rows after dropping sequences without thermal data.")

    num_classes_ckpt = ckpt.get("num_classes", num_classes)
    thm_cols_ckpt = ckpt.get("thm_cols")
    if thm_cols_ckpt:
        thm_cols = list(thm_cols_ckpt)

    thm_mean = ckpt.get("thm_mean")
    thm_std = ckpt.get("thm_std")
    if thm_mean is None or thm_std is None:
        train_mean_raw, train_std_raw = compute_train_thermal_zscore_params(trainset_df, thm_cols)
    else:
        train_mean_raw = np.asarray(thm_mean, dtype=np.float64)
        train_std_raw = np.asarray(thm_std, dtype=np.float64)

    dropout_p = ckpt.get("dropout_p", THERMAL_DROPOUT_DEFAULT)

    model = MultistreamCNNThermalNet(
        num_classes=num_classes_ckpt,
        mean=train_mean_raw,
        std=train_std_raw,
        dropout_p=float(dropout_p),
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    collate_thermal = make_collate_fn(train_mean_raw)
    test_ds = ThermalSequenceDataset(testset_df, thm_cols, train_mean_raw)
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_thermal,
    )
    return model, test_loader, target_names, num_classes_ckpt, le


def run_evaluation(model, test_loader, device):
    """Run evaluation loop; return all_preds, all_labels, all_outputs (logits)."""
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            all_outputs.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels, np.array(all_outputs)


# ---------------------------------------------------------------------------
# 1. Classification report + confusion matrix
# ---------------------------------------------------------------------------
def analysis_classification_report_and_cm(all_labels, all_preds, target_names):
    ensure_output_dir()
    report = classification_report(all_labels, all_preds, target_names=target_names)
    path_txt = out_path("classification_report.txt")
    with open(path_txt, "w") as f:
        f.write(f"Test Set Performance Analysis ({plot_title()})\n")
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
    plt.title(f"Confusion Matrix: {plot_title()} — BFRB vs. Non-BFRB Gestures")
    plt.ylabel("True Gesture")
    plt.xlabel("Predicted Gesture")
    path_fig = out_path("confusion_matrix.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")
    return cm, report


# ---------------------------------------------------------------------------
# 2. Misclassification analysis
# ---------------------------------------------------------------------------
def analysis_misclassifications(cm, target_names):
    ensure_output_dir()
    misclassifications = []
    n = len(target_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                misclassifications.append({
                    "True_Gesture": target_names[i],
                    "Predicted_Gesture": target_names[j],
                    "Count": int(cm[i, j]),
                })
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
        lines.append(f"\n'Eyelash - pull hair' is a common misprediction target. Gestures like {', '.join(to_join)} are often mistaken for it.")

    predicted_as_forehead = [mc for mc in misclassifications if mc["Predicted_Gesture"] == "Forehead - pull hairline"]
    if predicted_as_forehead:
        to_join = [f"'{mc['True_Gesture']}'" for mc in predicted_as_forehead[:3]]
        lines.append(f"'Forehead - pull hairline' also gets confused with others, including {', '.join(to_join)}.")

    lines.append("")
    lines.append("Further analysis often involves looking for:")
    lines.append("- Reciprocal confusions (e.g., A mistaken for B, and B mistaken for A)")
    lines.append("- Specific BFRB vs. non-BFRB confusions")

    text = "\n".join(lines)
    path_txt = out_path("misclassifications_analysis.txt")
    with open(path_txt, "w") as f:
        f.write(text)
    print(f"Saved: {path_txt}")

    df_mc = pd.DataFrame(misclassifications)
    path_csv = out_path("misclassifications.csv")
    df_mc.to_csv(path_csv, index=False)
    print(f"Saved: {path_csv}")
    return misclassifications


# ---------------------------------------------------------------------------
# 3. Per-class Precision, Recall, F1
# ---------------------------------------------------------------------------
def analysis_per_class_metrics(all_labels, all_preds, target_names):
    ensure_output_dir()
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(target_names))), zero_division=0
    )
    metrics_df = pd.DataFrame({
        "Class": target_names,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support,
    }).sort_values("F1", ascending=True)

    path_csv = out_path("per_class_metrics.csv")
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
    ax.set_title(f"Per-Class Precision, Recall & F1 (sorted by F1) — {plot_title()}")
    ax.legend()
    plt.tight_layout()
    path_fig = out_path("per_class_precision_recall_f1.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    low_f1 = metrics_df[metrics_df["F1"] < 0.5][["Class", "Precision", "Recall", "F1", "Support"]]
    path_low = out_path("classes_f1_below_0.5.txt")
    with open(path_low, "w") as f:
        f.write("Classes with F1 < 0.5:\n")
        f.write(low_f1.to_string(index=False))
    print(f"Saved: {path_low}")
    return metrics_df


# ---------------------------------------------------------------------------
# 4. Confidence analysis (correct vs wrong)
# ---------------------------------------------------------------------------
def analysis_confidence(all_outputs, all_labels, all_preds, target_names):
    ensure_output_dir()
    all_probs = F.softmax(torch.tensor(all_outputs, dtype=torch.float32), dim=1).numpy()
    max_confidence = all_probs.max(axis=1)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    correct_mask = all_labels_arr == all_preds_arr

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(max_confidence[correct_mask], bins=30, alpha=0.6, color="seagreen", label=f"Correct (n={correct_mask.sum()})")
    axes[0].hist(max_confidence[~correct_mask], bins=30, alpha=0.6, color="tomato", label=f"Wrong (n={(~correct_mask).sum()})")
    axes[0].set_xlabel("Max Softmax Probability (Confidence)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Confidence: Correct vs. Wrong")
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
    path_fig = out_path("confidence_analysis.png")
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
    path_txt = out_path("confidence_summary.txt")
    with open(path_txt, "w") as f:
        f.write(text)
    print(f"Saved: {path_txt}")
    return all_probs


# ---------------------------------------------------------------------------
# 5. Accuracy by body region
# ---------------------------------------------------------------------------
def analysis_region_accuracy(all_labels, all_preds, target_names):
    ensure_output_dir()

    def get_region(class_name):
        return class_name.split(" - ")[0].strip() if " - " in class_name else class_name

    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    true_class_names = [target_names[i] for i in all_labels_arr]
    pred_class_names = [target_names[i] for i in all_preds_arr]
    true_regions = [get_region(n) for n in true_class_names]

    region_results = pd.DataFrame({
        "true_class": true_class_names,
        "pred_class": pred_class_names,
        "true_region": true_regions,
        "correct": all_labels_arr == all_preds_arr,
    })
    region_acc = (
        region_results.groupby("true_region")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct", "count": "total"})
    )
    region_acc["accuracy"] = region_acc["correct"] / region_acc["total"]
    region_acc = region_acc.sort_values("accuracy")

    path_csv = out_path("region_accuracy.csv")
    region_acc.to_csv(path_csv)
    print(f"Saved: {path_csv}")

    fig, ax = plt.subplots(figsize=(9, max(4, len(region_acc) * 0.5)))
    colors = ["tomato" if a < 0.5 else "gold" if a < 0.7 else "seagreen" for a in region_acc["accuracy"]]
    ax.barh(region_acc.index, region_acc["accuracy"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Accuracy Grouped by Body Region — {plot_title()}")
    for i, (region, row) in enumerate(region_acc.iterrows()):
        ax.text(row["accuracy"] + 0.01, i, f"{row['accuracy']:.0%}  (n={int(row['total'])})", va="center", fontsize=8)
    plt.tight_layout()
    path_fig = out_path("region_accuracy.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    path_txt = out_path("region_accuracy_table.txt")
    with open(path_txt, "w") as f:
        f.write("Per-region accuracy table:\n")
        f.write(region_acc[["correct", "total", "accuracy"]].to_string())
    print(f"Saved: {path_txt}")
    return region_acc


# ---------------------------------------------------------------------------
# 6. Per-class ROC and macro AUC
# ---------------------------------------------------------------------------
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

    plt.suptitle(f"One-vs-Rest ROC Curves per Gesture Class — {plot_title()}", fontsize=12, y=1.01)
    plt.tight_layout()
    path_fig = out_path("roc_curves.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_fig}")

    auc_summary = pd.DataFrame({
        "Class": target_names,
        "AUC": [roc_auc[i] for i in range(n_classes)],
    }).sort_values("AUC")
    path_csv = out_path("roc_auc_per_class.csv")
    auc_summary.to_csv(path_csv, index=False)
    print(f"Saved: {path_csv}")

    path_txt = out_path("roc_auc_summary.txt")
    with open(path_txt, "w") as f:
        f.write(f"Macro-average AUC: {macro_auc:.4f}\n")
        f.write("\nPer-class AUC (lowest first):\n")
        f.write(auc_summary.to_string(index=False))
    print(f"Saved: {path_txt}")
    return macro_auc, auc_summary


# ---------------------------------------------------------------------------
# Save raw predictions for reproducibility
# ---------------------------------------------------------------------------
def save_predictions(all_labels, all_preds, all_outputs, all_probs):
    ensure_output_dir()
    np.savez(
        out_path("predictions.npz"),
        all_labels=np.array(all_labels),
        all_preds=np.array(all_preds),
        all_outputs=all_outputs,
        all_probs=all_probs,
    )
    print(f"Saved: {out_path('predictions.npz')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(checkpoint_path=None, profile=None):
    global OUTPUT_DIR, PROFILE
    if profile is None:
        profile = resolve_model_profile()
    PROFILE = profile
    OUTPUT_DIR = profile["output_dir"]

    cp = checkpoint_path if checkpoint_path is not None else profile["checkpoint"]
    print(f"Analyzing: {profile['plot_title']}")
    print(f"Checkpoint: {cp}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Loading model and test data...")
    model, test_loader, target_names, num_classes, le = load_model_and_data(profile, cp)
    device = next(model.parameters()).device

    print("Running evaluation...")
    all_preds, all_labels, all_outputs = run_evaluation(model, test_loader, device)
    all_probs = F.softmax(torch.tensor(all_outputs, dtype=torch.float32), dim=1).numpy()

    print("\nTest Set Performance:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    ensure_output_dir()
    print("\n--- Saving analysis results ---")

    cm, _ = analysis_classification_report_and_cm(all_labels, all_preds, target_names)
    analysis_misclassifications(cm, target_names)
    analysis_per_class_metrics(all_labels, all_preds, target_names)
    analysis_confidence(all_outputs, all_labels, all_preds, target_names)
    analysis_region_accuracy(all_labels, all_preds, target_names)
    analysis_roc_curves(all_labels, all_probs, target_names)
    save_predictions(all_labels, all_preds, all_outputs, all_probs)

    print(f"\nAll results saved under: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
