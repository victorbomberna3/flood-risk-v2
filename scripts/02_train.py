"""
02_train.py — Train U-Net on Severn raster.

Usage:
    python scripts/02_train.py --config configs/default.yaml
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config, set_seed, get_device, count_parameters, report_gpu_memory
from data_io import load_raster
from normalize import stack_channels
from dataset import build_train_dataset
from model import build_model
from losses import build_loss
from trainer import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(report_gpu_memory())

    cache_dir = Path(config["paths"]["raster_cache"])
    train_region = config["regions"]["train"]

    print(f"\n  Loading {train_region} raster...")
    raster = load_raster(cache_dir / f"raster_{train_region}.npz")
    print(f"  {raster}")

    # Channel order from config
    feature_names = config["features"]["terrain"] + config["features"]["weather"]

    # Verify all features exist
    missing = [f for f in feature_names if f not in raster.terrain]
    if missing:
        print(f"  WARNING: missing features {missing} — will be zero-filled")
    available = [f for f in feature_names if f in raster.terrain]
    print(f"  Using {len(available)} features (of {len(feature_names)} configured)")

    channels = stack_channels(raster.terrain, feature_names, valid_mask=raster.valid_mask)
    print(f"  Channel tensor shape: {channels.shape}  ({channels.dtype})")
    print(f"  Memory: {channels.nbytes / 1e9:.2f} GB")

    # Build datasets
    print("\n  Building train/val datasets...")
    train_ds, val_ds, info = build_train_dataset(
        channels=channels,
        targets=raster.targets,
        valid_mask=raster.valid_mask,
        feature_names=feature_names,
        patch_size=config["patches"]["size"],
        stride=config["patches"]["stride_train"],
        min_valid_frac=config["patches"]["min_valid_fraction"],
        min_labeled_frac=config["patches"]["min_labeled_fraction"],
        val_fraction=config["validation"]["val_fraction"],
        block_size=config["validation"]["block_size_pixels"] * 4,
        augment_train=config["augmentation"]["rotations_90"] or config["augmentation"]["flips"],
        seed=config["training"]["seed"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config["training"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["training"]["num_workers"] > 0,
    )

    # Build model
    print("\n  Building model...")
    model = build_model(config, in_channels=channels.shape[0]).to(device)
    n_params = count_parameters(model)
    print(f"  Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Build loss with class weights from full Severn targets
    print("\n  Computing class weights...")
    loss_fn = build_loss(raster.targets, config, device)

    # Train
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    print("\n" + "=" * 60)
    print(f"  Best val Macro F1: {result['best_metric']:.4f}")
    print(f"  Best checkpoint:   {result['best_checkpoint']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
