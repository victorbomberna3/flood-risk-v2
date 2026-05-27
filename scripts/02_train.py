"""
02_train.py — Train U-Net on Severn raster.

Usage:
    python scripts/02_train.py --config configs/default.yaml
"""
import gc
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

    # Channel order from config
    feature_names = config["features"]["terrain"] + config["features"]["weather"]

    print(f"\n  Loading {train_region} raster...")
    raster = load_raster(cache_dir / f"raster_{train_region}.npz", feature_names=feature_names)
    print(f"  {raster}")

    available = [f for f in feature_names if f in raster.terrain]
    print(f"  Using {len(available)} features (of {len(feature_names)} configured)")

    # Build channels into a disk-backed memmap — avoids allocating 8 GB in RAM
    # on top of the already-loaded source arrays.
    h, w = raster.valid_mask.shape
    mmap_path = "/tmp/channels.dat"
    channels = np.memmap(mmap_path, dtype=np.float32, mode="w+",
                         shape=(len(feature_names), h, w))
    for i, name in enumerate(feature_names):
        arr = raster.terrain.pop(name, np.zeros((h, w), dtype=np.float32)).astype(np.float32)
        arr = np.where(raster.valid_mask, arr, np.float32(0.0))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -500.0, 500.0, out=arr)
        channels[i] = arr
        del arr
    raster.terrain = {}
    gc.collect()
    channels.flush()
    print(f"  Channel tensor shape: {channels.shape}  ({channels.dtype})")
    print(f"  Disk-backed memmap: {channels.nbytes / 1e9:.2f} GB")

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
