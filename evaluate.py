"""
evaluate.py — Metrics, confusion matrices, and prediction maps.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score, confusion_matrix, classification_report, cohen_kappa_score


CLASS_NAMES = ["No risk", "Very Low", "Low", "Medium", "High"]
N_CLASSES = 5

# White for 0, then green → yellow → orange → red
CMAP = mcolors.ListedColormap(["#ffffff", "#d9f0a3", "#addd8e", "#f0960a", "#d7191c"])


def compute_metrics(
    preds: np.ndarray,              # (5, H, W) int8 with -1 for no-data
    targets: np.ndarray,            # (H, W, 5) int8 with -1 for unlabeled
    valid_mask: np.ndarray,         # (H, W) bool
    target_names: list[str],
) -> dict:
    """Compute Macro F1, QWK, per-class F1, and confusion matrix for each depth."""
    out = {}
    macro_f1_per_target = []
    qwk_per_target = []
    for i, tname in enumerate(target_names):
        y_true = targets[:, :, i]
        y_pred = preds[i]

        mask = (y_true >= 0) & valid_mask & (y_pred >= 0)
        if not mask.any():
            print(f"  WARNING: no valid pixels for {tname}")
            continue

        yt = y_true[mask].astype(int)
        yp = y_pred[mask].astype(int)

        labels = list(range(N_CLASSES))
        macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        per_class_f1 = f1_score(yt, yp, labels=labels, average=None, zero_division=0).tolist()
        cm = confusion_matrix(yt, yp, labels=labels).tolist()

        # QWK: penalises errors quadratically by ordinal distance (class 4 vs 3 << class 4 vs 0)
        try:
            qwk = float(cohen_kappa_score(yt, yp, weights="quadratic", labels=labels))
        except Exception:
            qwk = 0.0

        out[tname] = {
            "macro_f1": macro_f1,
            "qwk": qwk,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "n_pixels": int(mask.sum()),
        }
        macro_f1_per_target.append(macro_f1)
        qwk_per_target.append(qwk)

    out["macro_f1_mean"] = float(np.mean(macro_f1_per_target)) if macro_f1_per_target else 0.0
    out["qwk_mean"] = float(np.mean(qwk_per_target)) if qwk_per_target else 0.0
    return out


def save_metrics(metrics: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {out_path}")


def print_summary(metrics: dict, target_names: list[str]) -> None:
    """Print a clean summary table to stdout."""
    print("\n" + "=" * 72)
    print(f"  Mean Macro F1 across all depths : {metrics['macro_f1_mean']:.4f}")
    print(f"  Mean QWK     across all depths  : {metrics['qwk_mean']:.4f}  ← primary metric")
    print("=" * 72)
    print(f"\n  {'Depth':<12} {'Macro F1':>10}  {'QWK':>8}  {'Per-class F1 (cls 0..4)':<40}")
    print("  " + "─" * 70)
    for tname in target_names:
        m = metrics[tname]
        per_class = "  ".join(f"{f:.2f}" for f in m["per_class_f1"])
        print(f"  {tname:<12} {m['macro_f1']:>10.4f}  {m['qwk']:>8.4f}  {per_class}")
    print()


def plot_predictions(
    preds: np.ndarray,              # (5, H, W) int8
    targets: np.ndarray,            # (H, W, 5) int8
    valid_mask: np.ndarray,
    target_names: list[str],
    out_path: str | Path,
    region_name: str = "northumbria",
) -> None:
    """Side-by-side maps: predicted vs ground truth for each depth."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_truth = (targets >= 0).any()
    ncols = 2 if has_truth else 1
    fig, axes = plt.subplots(len(target_names), ncols, figsize=(7 * ncols, 4 * len(target_names)))
    if len(target_names) == 1:
        axes = np.array([axes])

    fig.suptitle(f"Flood Risk Predictions — {region_name.title()}", fontsize=14, fontweight="bold")

    for row, tname in enumerate(target_names):
        # Predicted
        ax = axes[row, 0] if ncols > 1 else axes[row]
        p = np.where(valid_mask, preds[row], np.nan)
        ax.imshow(p, origin="lower", cmap=CMAP, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
        ax.set_title(f"{tname} — Predicted", fontsize=9)
        ax.axis("off")

        if has_truth:
            ax2 = axes[row, 1]
            t = np.where((targets[:, :, row] >= 0) & valid_mask, targets[:, :, row], np.nan)
            ax2.imshow(t, origin="lower", cmap=CMAP, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
            ax2.set_title(f"{tname} — Ground Truth", fontsize=9)
            ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Map → {out_path}")


def plot_confusion_matrices(
    metrics: dict,
    target_names: list[str],
    out_path: str | Path,
) -> None:
    """Confusion matrix per depth."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(target_names), figsize=(4 * len(target_names), 4))
    if len(target_names) == 1:
        axes = [axes]

    for ax, tname in zip(axes, target_names):
        cm = np.array(metrics[tname]["confusion_matrix"])
        # Normalise by row (per true class)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{tname}\nQWK = {metrics[tname]['qwk']:.3f}  |  Macro F1 = {metrics[tname]['macro_f1']:.3f}", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(range(N_CLASSES))
        ax.set_yticklabels(range(N_CLASSES))
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrices → {out_path}")
