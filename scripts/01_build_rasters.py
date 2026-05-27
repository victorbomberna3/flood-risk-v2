"""
01_build_rasters.py — Build raster tensors from NetCDF files.

Run this once. The output .npz files are cached so subsequent training
runs skip the slow feature engineering step (~5 minutes per region).

Usage (from project root):
    python scripts/01_build_rasters.py --config configs/default.yaml
"""
import sys
import argparse
from pathlib import Path

# Make src importable when running this script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config
from data_io import build_region_raster, save_raster
from features import engineer_features
from normalize import normalize_region, NormStats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    drive_data = Path(config["paths"]["drive_data"])
    cache_dir = Path(config["paths"]["raster_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    for region_key in ("train", "test"):
        region = config["regions"][region_key]
        terrain_path = drive_data / f"flood_risk_terrain_{region}.nc"
        weather_path = drive_data / f"era5_land_{region}.nc"

        if not terrain_path.exists():
            print(f"ERROR: {terrain_path} not found")
            print("Check paths.drive_data in your config or move the .nc file there.")
            sys.exit(1)

        print(f"\n{'=' * 60}\n  {region.upper()}\n{'=' * 60}")
        raster = build_region_raster(
            region=region,
            terrain_path=terrain_path,
            weather_path=weather_path,
            feature_engineer=engineer_features,
        )

        # Per-region normalisation
        normalised, stats = normalize_region(
            features=raster.terrain,
            valid_mask=raster.valid_mask,
            region=region,
            method=config["normalize"]["method"],
            save_path=cache_dir / f"norm_stats_{region}.json",
        )
        raster.terrain = normalised

        # Save raster cache
        save_raster(raster, cache_dir / f"raster_{region}.npz")

    print("\n" + "=" * 60)
    print("  Raster build complete.")
    print(f"  Cache directory: {cache_dir}")
    print("  Next: python scripts/02_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
