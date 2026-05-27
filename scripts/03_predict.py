"""
03_predict.py — Inference on Northumbria using best Severn checkpoint.

Usage:
    python scripts/03_predict.py --config configs/default.yaml
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config, set_seed, get_device
from data_io import load_raster
from normalize import stack_channels
from model import build_model
from predict import predict_region, save_predictions
from evaluate import compute_metrics, save_metrics, print_summary, plot_predictions, plot_confusion_matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to model_best.pt (auto-detected if None)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    cache_dir = Path(config["paths"]["raster_cache"])
    test_region = config["regions"]["test"]
    out_dir = Path(config["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint
    if args.checkpoint is None:
        ckpt_path = Path(config["paths"]["checkpoints"]) / "model_best.pt"
    else:
        ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}. Run scripts/02_train.py first.")
        sys.exit(1)

    print(f"\n  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"  Saved at epoch {ckpt['epoch']}, val Macro F1 = {ckpt['val_metrics']['macro_f1_mean']:.4f}")

    # Load test raster
    feature_names = config["features"]["terrain"] + config["features"]["weather"]
    print(f"\n  Loading {test_region} raster...")
    raster = load_raster(cache_dir / f"raster_{test_region}.npz", feature_names=feature_names)
    print(f"  {raster}")

    # Stack channels — apply identical preprocessing as training
    channels = stack_channels(raster.terrain, feature_names, valid_mask=raster.valid_mask)
    raster.terrain = {}
    channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
    channels = np.clip(channels, -500, 500)
    print(f"  Channels: {channels.shape}")

    # Build model and load weights
    model = build_model(config, in_channels=channels.shape[0]).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Predict
    print("\n  Running inference...")
    preds = predict_region(
        model=model,
        channels=channels,
        valid_mask=raster.valid_mask,
        feature_names=feature_names,
        n_tasks=len(config["targets"]),
        n_classes=config["n_classes"],
        patch_size=config["patches"]["size"],
        stride=config["patches"]["stride_inference"],
        batch_size=config["inference"]["batch_size"],
        device=device,
        tta=config["inference"]["tta"],
    )
    print(f"  Predictions shape: {preds.shape}")

    # Save predictions as parquet
    save_predictions(
        preds=preds,
        valid_mask=raster.valid_mask,
        x_coords=raster.x,
        y_coords=raster.y,
        target_names=config["targets"],
        out_path=out_dir / f"predictions_{test_region}.parquet",
    )

    # Compute metrics (Northumbria has labels for ~70% of pixels)
    print("\n  Computing metrics...")
    metrics = compute_metrics(
        preds=preds,
        targets=raster.targets,
        valid_mask=raster.valid_mask,
        target_names=config["targets"],
    )
    save_metrics(metrics, out_dir / "metrics.json")
    print_summary(metrics, config["targets"])

    # Generate visualisation maps
    print("  Generating maps...")
    plot_predictions(
        preds=preds,
        targets=raster.targets,
        valid_mask=raster.valid_mask,
        target_names=config["targets"],
        out_path=out_dir / "maps" / f"predictions_vs_truth_{test_region}.png",
        region_name=test_region,
    )
    plot_confusion_matrices(
        metrics=metrics,
        target_names=config["targets"],
        out_path=out_dir / "maps" / f"confusion_matrices_{test_region}.png",
    )

    print("\n" + "=" * 60)
    print(f"  Mean QWK      (Northumbria): {metrics['qwk_mean']:.4f}  ← primary")
    print(f"  Mean Macro F1 (Northumbria): {metrics['macro_f1_mean']:.4f}")
    print(f"  Outputs in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
