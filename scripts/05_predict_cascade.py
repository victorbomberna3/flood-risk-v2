"""
05_predict_cascade.py — Apply trained cascade to Northumbria and evaluate.

Usage:
    python scripts/05_predict_cascade.py --config configs/default.yaml

Key flags:
    --hand-pctile 10     Percentile of Severn class-0 HAND to use as threshold.
                         10 → T = 10th pctile → 90% of class-0 pixels have HAND > T.
                         Pixels with HAND > T are forced to class 0 (overrides Stage A/B).
    --hand-threshold 5.0 Override: use this HAND value directly (metres). Ignores --hand-pctile.
    --no-hand-override   Disable HAND post-processing entirely.
"""
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config
from cascade.models import combine_cascade
from cascade.rounder import soft_probabilities
from cascade.metrics import evaluate_per_depth
from evaluate import compute_metrics, save_metrics, print_summary
from evaluate import plot_predictions, plot_confusion_matrices

RISK_VARS = ["risk_0_2m", "risk_0_3m", "risk_0_6m", "risk_0_9m", "risk_1_2m"]


# ── HAND threshold ────────────────────────────────────────────────────────────

def compute_hand_threshold(severn_npz: Path, pctile: float):
    """Derive a HAND threshold from Severn class-0 labeled pixels.

    Returns T such that `pctile`% of Severn class-0 HAND values are below T.
    Any prediction pixel with HAND > T will be forced to class 0.
    """
    if not severn_npz.exists():
        print("  [HAND] Severn raster not found — skipping HAND override")
        return None
    data = np.load(severn_npz, allow_pickle=False)
    if "terrain__hand" not in data.files:
        print("  [HAND] terrain__hand missing from Severn raster — skipping")
        return None

    valid_idx = np.where(data["valid_mask"].astype(bool).ravel())[0]
    hand = data["terrain__hand"].ravel()[valid_idx].astype(np.float32)
    hand = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0)

    targets = data["targets"].reshape(-1, 5)[valid_idx]
    labeled = (targets >= 0).any(axis=1)
    # class 0 = labeled, and every labeled depth has target == 0
    safe_tgt = targets.copy()
    safe_tgt[targets < 0] = 0
    max_tgt = safe_tgt.max(axis=1)
    class0 = labeled & (max_tgt == 0)

    if class0.sum() < 10:
        print("  [HAND] Too few class-0 pixels in Severn — skipping")
        return None

    hand_c0 = hand[class0]
    threshold = float(np.percentile(hand_c0, pctile))
    print(
        f"  [HAND] {class0.sum():,} Severn class-0 pixels | "
        f"HAND {pctile:.0f}th-pctile = {threshold:.2f} m  →  override threshold"
    )
    print(
        f"  [HAND] class-0 HAND stats: "
        f"p10={np.percentile(hand_c0,10):.1f}  p25={np.percentile(hand_c0,25):.1f}  "
        f"median={np.median(hand_c0):.1f}  p75={np.percentile(hand_c0,75):.1f}  "
        f"p90={np.percentile(hand_c0,90):.1f}"
    )
    return threshold


# ── Data loading ─────────────────────────────────────────────────────────────

def load_pixels(npz_path: Path, feature_names: list[str]) -> tuple:
    """Load valid pixels from NPZ and return tabular arrays.

    Returns
    -------
    (X, x_coords, y_coords, targets, H, W, valid_idx, valid_mask, x_1d, y_1d)
    """
    data = np.load(npz_path, allow_pickle=False)
    valid_mask = data["valid_mask"].astype(bool)
    H, W = valid_mask.shape
    valid_idx = np.where(valid_mask.ravel())[0]

    x_1d = data["x"]
    y_1d = data["y"]
    xx, yy = np.meshgrid(x_1d, y_1d)
    x_coords = xx.ravel()[valid_idx]
    y_coords = yy.ravel()[valid_idx]
    del xx, yy

    needed = set(feature_names)
    feats: dict = {}
    for k in data.files:
        if k.startswith("terrain__"):
            name = k[len("terrain__"):]
            if name in needed:
                arr = data[k].ravel()[valid_idx].astype(np.float32)
                feats[name] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    ordered = [n for n in feature_names if n in feats]
    X = pd.DataFrame({n: feats[n] for n in ordered})

    # Targets: (H, W, 5) → (N_valid, 5)
    targets = data["targets"].reshape(-1, 5)[valid_idx]

    return X, x_coords, y_coords, targets, H, W, valid_idx, valid_mask, x_1d, y_1d


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_stage_a(models: list, X: pd.DataFrame) -> np.ndarray:
    """Average Stage A predictions across all fold models."""
    p = np.zeros(len(X), dtype=np.float64)
    for m in models:
        p += m.predict(X) / len(models)
    return p


def infer_stage_b(b: dict, X: pd.DataFrame) -> np.ndarray:
    """Return blended (n, 5) class probabilities for one depth target."""
    w = b["ensemble_weight"]
    classes = np.array([0, 1, 2, 3, 4])

    # CatBoost ensemble
    cat_proba = np.zeros((len(X), 5), dtype=np.float64)
    for model in b["cat_models"]:
        proba = model.predict_proba(X)
        col = {int(c): j for j, c in enumerate(model.classes_)}
        aligned = np.column_stack([
            proba[:, col[c]] if c in col else np.zeros(len(X))
            for c in classes
        ])
        cat_proba += aligned / len(b["cat_models"])

    # LGB regressor ensemble
    lgb_cont = np.zeros(len(X), dtype=np.float64)
    for model in b["lgb_models"]:
        lgb_cont += model.predict(X) / len(b["lgb_models"])
    lgb_proba = soft_probabilities(
        lgb_cont, labels=list(classes), sigma=b["lgb_sigma"]
    )

    return (w * cat_proba + (1.0 - w) * lgb_proba).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply trained cascade to Northumbria and evaluate"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to cascade_model.pkl (auto-detected if None)")
    # HAND override
    parser.add_argument("--hand-pctile", type=float, default=10.0,
                        help="Percentile of Severn class-0 HAND values used as threshold "
                             "(default 10 → 10th pctile). Lower = more pixels forced to class 0.")
    parser.add_argument("--hand-threshold", type=float, default=None,
                        help="Direct HAND threshold in metres (overrides --hand-pctile).")
    parser.add_argument("--no-hand-override", action="store_true",
                        help="Disable HAND post-processing entirely.")
    parser.add_argument("--tau-override", type=float, default=None,
                        help="Override Stage A threshold tau_a at inference (e.g. 0.3). "
                             "Default: use value saved in checkpoint.")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = Path(config["paths"]["raster_cache"])
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    out_dir = Path(config["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)

    # Load cascade artifact
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint
        else checkpoint_dir / "cascade_model.pkl"
    )
    if not ckpt_path.exists():
        print(f"ERROR: cascade model not found at {ckpt_path}")
        print("       Run scripts/04_train_cascade.py first.")
        sys.exit(1)

    print(f"\n  Loading cascade model: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        artifact = pickle.load(f)

    feature_names = artifact["feature_names"]
    tau_a = artifact["tau_a"]
    if args.tau_override is not None:
        print(f"  tau_a override: {tau_a:.3f} → {args.tau_override:.3f}")
        tau_a = args.tau_override
    print(f"  OOF QWK (Severn 5-fold): {artifact['oof_qwk']:.4f}")
    print(f"  Stage A threshold (tau): {tau_a:.3f}")
    print(f"  Features: {len(feature_names)}")

    # ── HAND threshold (computed from Severn class-0 pixels) ─────────────────
    hand_threshold = None
    if not args.no_hand_override:
        print("\n  Computing HAND override threshold from Severn raster...")
        if args.hand_threshold is not None:
            hand_threshold = args.hand_threshold
            print(f"  [HAND] Using manual threshold: {hand_threshold:.2f} m")
        else:
            hand_threshold = compute_hand_threshold(
                cache_dir / "raster_severn.npz",
                pctile=args.hand_pctile,
            )

    # Load Northumbria raster
    print("\n  Loading Northumbria raster...")
    (X, x_coords, y_coords, targets,
     H, W, valid_idx, valid_mask, x_1d, y_1d) = load_pixels(
        cache_dir / "raster_northumbria.npz", feature_names
    )
    print(f"  {len(X):,} valid pixels")

    # Build HAND override mask for Northumbria valid pixels
    hand_override = np.zeros(len(X), dtype=bool)
    if hand_threshold is not None and "hand" in X.columns:
        hand_vals = X["hand"].values
        hand_override = hand_vals > hand_threshold
        n_ov = hand_override.sum()
        print(
            f"  [HAND] Override: {n_ov:,} / {len(X):,} Northumbria pixels "
            f"({100 * n_ov / len(X):.1f}%) → forced class 0"
        )
    elif hand_threshold is not None and "hand" not in X.columns:
        print("  [HAND] 'hand' not in feature list — override skipped")

    # Stage A
    print("\n  Stage A inference (binary risk gate)...")
    p_a = infer_stage_a(artifact["stage_a"]["models"], X)
    risk_frac = (p_a >= tau_a).mean() * 100
    print(f"  p_a mean={p_a.mean():.3f}  →  {risk_frac:.1f}% pixels gated as 'at risk'")

    # Stage B + combine
    print("\n  Stage B inference (per depth)...")
    preds_flat = np.full((5, H * W), -1, dtype=np.int8)

    for d_idx, depth in enumerate(RISK_VARS):
        print(f"    {depth}...", end="  ", flush=True)
        proba_b = infer_stage_b(artifact["stage_b"][depth], X)
        pred = combine_cascade(p_a, proba_b, tau_a).astype(np.int8)

        # Apply HAND override: physics-based class-0 rule
        pred[hand_override] = 0

        preds_flat[d_idx, valid_idx] = pred

        dist = np.bincount(pred.astype(int), minlength=5).tolist()
        print(f"class dist (0-4): {dist}")

    preds = preds_flat.reshape(5, H, W)

    # Save parquet
    parquet_path = out_dir / "predictions_cascade_northumbria.parquet"
    df_rows = {"x": x_coords, "y": y_coords}
    for d_idx, depth in enumerate(RISK_VARS):
        df_rows[f"pred_{depth}"] = preds[d_idx].ravel()[valid_idx]
    pd.DataFrame(df_rows).to_parquet(parquet_path, index=False)
    print(f"\n  Saved {len(valid_idx):,} predictions → {parquet_path}")

    # Reconstruct full (H, W, 5) target array for evaluate.py compatibility
    targets_full = np.full((H * W, 5), -1, dtype=np.int8)
    targets_full[valid_idx] = targets
    targets_hw5 = targets_full.reshape(H, W, 5)

    # Metrics
    print("\n  Computing metrics...")
    metrics = compute_metrics(
        preds=preds,
        targets=targets_hw5,
        valid_mask=valid_mask,
        target_names=config["targets"],
    )
    save_metrics(metrics, out_dir / "metrics_cascade.json")
    print_summary(metrics, config["targets"])

    # Also report per-depth QWK from cascade.metrics for labeled pixels
    print("  Per-depth cascade evaluation (labeled pixels only):")
    y_true_dict, y_pred_dict = {}, {}
    for d_idx, depth in enumerate(RISK_VARS):
        labeled = targets[:, d_idx] >= 0
        if labeled.any():
            y_true_dict[depth] = targets[labeled, d_idx].astype(int)
            y_pred_dict[depth] = preds[d_idx].ravel()[valid_idx][labeled].astype(int)
    if y_true_dict:
        df_eval = evaluate_per_depth(y_true_dict, y_pred_dict)
        print(df_eval[["qwk", "macro_f1"]].to_string())

    # Maps
    print("\n  Generating maps...")
    plot_predictions(
        preds=preds,
        targets=targets_hw5,
        valid_mask=valid_mask,
        target_names=config["targets"],
        out_path=out_dir / "maps" / "predictions_cascade_northumbria.png",
        region_name="northumbria (cascade)",
    )
    plot_confusion_matrices(
        metrics=metrics,
        target_names=config["targets"],
        out_path=out_dir / "maps" / "confusion_cascade_northumbria.png",
    )

    print("\n" + "=" * 60)
    print(f"  Cascade QWK   (Northumbria): {metrics['qwk_mean']:.4f}")
    print(f"  Cascade Macro F1           : {metrics['macro_f1_mean']:.4f}")
    print(f"  Outputs in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
