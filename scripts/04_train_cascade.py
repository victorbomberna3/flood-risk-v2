"""
04_train_cascade.py — Train two-stage GBM cascade on the Severn raster cache.

Stage A : LightGBM binary  (any risk vs no risk)
Stage B : CatBoost MultiClass + LightGBM regressor ensemble, per depth target

The raster cache must already exist (run 01_build_rasters.py first).
No need to rebuild rasters — we reuse the 29-feature NPZ files.

Required packages (not in base requirements.txt):
    pip install catboost optuna lightgbm

Usage:
    python scripts/04_train_cascade.py --config configs/default.yaml

Key flags:
    --n-trials-a 25      Optuna trials for Stage A  (default 25)
    --n-trials-b 15      Optuna trials per Stage B depth  (default 15)
    --timeout-a  2700    Stage A HPO wall-clock limit in seconds  (45 min)
    --timeout-b  1200    Stage B HPO limit per depth  (20 min)
    --lgb-device cpu     LightGBM device: cpu or gpu
    --catboost-device GPU  CatBoost task_type: CPU or GPU
    --max-pixels 5000000   Subsample labeled pixels (0 = use all, safe on Colab)
"""
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config, set_seed
from cascade.cv import assign_blocks, make_spatial_folds, fold_labels, folds_from_labels
from cascade.models import (
    train_stage_a,
    train_stage_b_catboost,
    train_stage_b_lgb_regression,
    ensemble_stage_b,
    tune_tau_a,
    combine_cascade,
)
from cascade.metrics import evaluate_per_depth

RISK_VARS = ["risk_0_2m", "risk_0_3m", "risk_0_6m", "risk_0_9m", "risk_1_2m"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_pixels(npz_path: Path, feature_names: list[str]) -> tuple:
    """Memory-efficient load: materialise only valid-pixel rows.

    Returns
    -------
    (X, x_coords, y_coords, targets, available_features)
        X             pd.DataFrame (N_valid, n_feat)
        x_coords      (N_valid,) projected x in metres
        y_coords      (N_valid,) projected y in metres
        targets       (N_valid, 5) int8, -1 = unlabeled
        available     list[str] feature names actually present
    """
    data = np.load(npz_path, allow_pickle=False)
    valid_mask = data["valid_mask"].astype(bool)
    valid_idx = np.where(valid_mask.ravel())[0]
    H, W = valid_mask.shape

    # Projected coordinates — used only for spatial block assignment
    x_1d = data["x"]
    y_1d = data["y"]
    xx, yy = np.meshgrid(x_1d, y_1d)
    x_coords = xx.ravel()[valid_idx]
    y_coords = yy.ravel()[valid_idx]
    del xx, yy

    # Feature matrix
    needed = set(feature_names)
    feats: dict = {}
    available: list[str] = []
    for k in data.files:
        if k.startswith("terrain__"):
            name = k[len("terrain__"):]
            if name in needed:
                arr = data[k].ravel()[valid_idx].astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                feats[name] = arr
                available.append(name)

    ordered = [n for n in feature_names if n in feats]
    X = pd.DataFrame({n: feats[n] for n in ordered})

    # Targets: (H, W, 5) → (H*W, 5) → (N_valid, 5)
    targets = data["targets"].reshape(-1, 5)[valid_idx]

    region = str(data["region"][0])
    print(f"  {region}: {len(X):,} valid pixels, {len(ordered)} features")
    return X, x_coords, y_coords, targets, ordered


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train two-stage GBM cascade on Severn raster cache"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-trials-a", type=int, default=25)
    parser.add_argument("--n-trials-b", type=int, default=15)
    parser.add_argument("--timeout-a", type=int, default=2700,
                        help="Stage A HPO wall-clock limit (s)")
    parser.add_argument("--timeout-b", type=int, default=1200,
                        help="Stage B HPO limit per depth (s)")
    parser.add_argument("--lgb-device", default="cpu",
                        help="LightGBM device_type: cpu or gpu")
    parser.add_argument("--catboost-device", default="GPU",
                        help="CatBoost task_type: CPU or GPU")
    parser.add_argument("--max-pixels", type=int, default=0,
                        help="Max labeled pixels (0 = all). Use 5000000 if OOM.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(42)

    cache_dir = Path(config["paths"]["raster_cache"])
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    feature_names = config["features"]["terrain"] + config["features"]["weather"]

    # ── 1. Load raster ───────────────────────────────────────────────────────
    print("\n[1/6] Loading Severn raster cache...")
    X, x_coords, y_coords, targets_flat, available = load_pixels(
        cache_dir / "raster_severn.npz", feature_names
    )

    # ── 2. Filter to labeled pixels ──────────────────────────────────────────
    print("\n[2/6] Filtering to labeled pixels...")
    labeled_any = (targets_flat >= 0).any(axis=1)
    idx_lab = np.where(labeled_any)[0]

    if args.max_pixels > 0 and len(idx_lab) > args.max_pixels:
        rng = np.random.default_rng(42)
        idx_lab = np.sort(rng.choice(idx_lab, size=args.max_pixels, replace=False))
        print(f"  Subsampled to {len(idx_lab):,} labeled pixels")

    X_lab = X.iloc[idx_lab].reset_index(drop=True)
    tgt_lab = targets_flat[idx_lab]          # (N_lab, 5)
    x_lab = x_coords[idx_lab]
    y_lab = y_coords[idx_lab]
    print(f"  Labeled: {len(idx_lab):,} / {len(X):,} valid pixels")

    # Stage A binary target: any depth > 0 risk
    any_risk = np.zeros(len(idx_lab), dtype=int)
    for d in range(5):
        ok = tgt_lab[:, d] >= 0
        any_risk[ok & (tgt_lab[:, d] > 0)] = 1
    print(
        f"  any_risk=1: {any_risk.sum():,}   "
        f"any_risk=0: {(any_risk == 0).sum():,}"
    )

    # ── 3. Spatial CV ────────────────────────────────────────────────────────
    print("\n[3/6] Building 5-fold spatial CV (20 km blocks)...")
    block_id = assign_blocks(x_lab, y_lab, block_size_m=20_000)
    folds = make_spatial_folds(block_id, n_splits=5)
    fold_lab_arr = fold_labels(block_id, folds)
    n_blocks = len(np.unique(block_id))
    print(f"  {n_blocks} blocks → 5 spatial folds")
    for i, (tr, va) in enumerate(folds):
        print(f"  Fold {i}: {len(tr):,} train / {len(va):,} val pixels")

    # ── 4. Stage A ───────────────────────────────────────────────────────────
    print("\n[4/6] Stage A — LightGBM binary classifier")
    print("=" * 60)
    stage_a = train_stage_a(
        X_lab, any_risk, folds,
        feature_name=list(X_lab.columns),
        n_trials=args.n_trials_a,
        timeout=args.timeout_a,
        device=args.lgb_device,
    )
    print(f"\n  Stage A CV AUC: {stage_a['cv_auc']:.4f}")

    # ── 5. Stage B (per depth) ───────────────────────────────────────────────
    print("\n[5/6] Stage B — CatBoost + LightGBM per depth")
    print("=" * 60)
    stage_b: dict = {}

    for d_idx, depth in enumerate(RISK_VARS):
        print(f"\n  ── {depth} ──")

        # Positive pixels for this depth
        depth_ok = tgt_lab[:, d_idx] >= 0
        pos_mask = depth_ok & (any_risk == 1)
        idx_pos = np.where(pos_mask)[0]

        X_pos = X_lab.iloc[idx_pos].reset_index(drop=True)
        y_pos = tgt_lab[idx_pos, d_idx].astype(int)

        # Align folds to positive pixels using the same spatial blocks
        fold_pos = folds_from_labels(fold_lab_arr[idx_pos])

        cls, cnts = np.unique(y_pos, return_counts=True)
        print(
            f"  {len(idx_pos):,} pos pixels — "
            + "  ".join(f"cls{c}:{n:,}" for c, n in zip(cls, cnts))
        )

        # CatBoost
        cat_res = train_stage_b_catboost(
            X_pos, y_pos, fold_pos,
            target_name=depth,
            n_trials=args.n_trials_b,
            timeout=args.timeout_b,
            task_type=args.catboost_device,
        )

        # LightGBM regressor
        lgb_res = train_stage_b_lgb_regression(
            X_pos, y_pos, fold_pos,
            target_name=depth,
            n_trials=args.n_trials_b,
            timeout=args.timeout_b,
            device=args.lgb_device,
        )

        # Ensemble
        best_w, oof_proba = ensemble_stage_b(
            cat_res["oof_proba"], lgb_res["oof_proba"], y_pos
        )

        stage_b[depth] = {
            "cat": cat_res, "lgb": lgb_res,
            "ensemble_weight": best_w,
            "oof_proba": oof_proba,
            "idx_pos": idx_pos,
            "y_pos": y_pos,
        }
        print(
            f"  CatBoost QWK={cat_res['cv_qwk']:.4f}  "
            f"LGB QWK={lgb_res['cv_qwk']:.4f}  "
            f"Ensemble w_cat={best_w:.2f}"
        )

    # ── 6. Tune tau_a + OOF evaluation ──────────────────────────────────────
    print("\n[6/6] Tuning Stage A threshold and OOF evaluation...")

    # Use pixels labeled for all 5 depths simultaneously
    all_labeled = (tgt_lab >= 0).all(axis=1)
    idx_all = np.where(all_labeled)[0]
    print(f"  Pixels labeled for all 5 depths: {all_labeled.sum():,}")

    probs_b_oof: dict = {}
    y_true_oof: dict = {}
    for d_idx, depth in enumerate(RISK_VARS):
        # Fill full-labeled-set proba (uniform default for non-positive pixels)
        proba_full = np.full((len(idx_lab), 5), 0.2, dtype=np.float32)
        idx_pos = stage_b[depth]["idx_pos"]
        proba_full[idx_pos] = stage_b[depth]["oof_proba"]

        probs_b_oof[depth] = proba_full[all_labeled]
        y_true_oof[depth] = tgt_lab[all_labeled, d_idx].astype(int)

    p_a_all = stage_a["oof"][all_labeled]
    best_tau, best_qwk = tune_tau_a(p_a_all, probs_b_oof, y_true_oof)
    print(f"  Best tau_a = {best_tau:.3f}   OOF mean QWK = {best_qwk:.4f}")

    oof_preds = {
        depth: combine_cascade(p_a_all, probs_b_oof[depth], best_tau)
        for depth in RISK_VARS
    }
    metrics_df = evaluate_per_depth(y_true_oof, oof_preds)
    print("\n  OOF metrics (Severn 5-fold spatial CV):")
    print(metrics_df[["qwk", "macro_f1"]].to_string())

    # ── Save artifact ────────────────────────────────────────────────────────
    artifact = {
        "feature_names": available,
        "tau_a": best_tau,
        "oof_qwk": best_qwk,
        "stage_a": {
            "models": stage_a["models"],
            "best_params": stage_a["best_params"],
            "cv_auc": stage_a["cv_auc"],
        },
        "stage_b": {
            depth: {
                "cat_models": stage_b[depth]["cat"]["models"],
                "lgb_models": stage_b[depth]["lgb"]["models"],
                "lgb_rounder": stage_b[depth]["lgb"]["rounder"],
                "lgb_sigma": stage_b[depth]["lgb"].get("sigma", 0.6),
                "ensemble_weight": stage_b[depth]["ensemble_weight"],
                "cat_cv_qwk": stage_b[depth]["cat"]["cv_qwk"],
                "lgb_cv_qwk": stage_b[depth]["lgb"]["cv_qwk"],
            }
            for depth in RISK_VARS
        },
    }
    out_path = checkpoint_dir / "cascade_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    print("\n" + "=" * 60)
    print(f"  OOF mean QWK (Severn 5-fold): {best_qwk:.4f}")
    print(f"  Stage A CV AUC              : {stage_a['cv_auc']:.4f}")
    print(f"  Cascade model saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
