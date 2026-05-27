"""
data_io.py — Load NetCDF files and build 2-D raster tensors.

Unlike the v1 tabular approach, we keep data as proper (H, W, C) arrays
so the U-Net can exploit spatial structure.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import xarray as xr


RISK_VARS = ["risk_0_2m", "risk_0_3m", "risk_0_6m", "risk_0_9m", "risk_1_2m"]

TERRAIN_RAW_VARS = ["dtm", "clc_type", "waw", "imd", "rciw", "flow_dir", "flow_acc"]

WEATHER_VARS = ["tp", "sro", "swvl1_mean", "t2m_mean", "u10_mean", "v10_mean", "d2m_mean"]


@dataclass
class RegionRaster:
    """Container for a region's raster data."""
    region: str
    x: np.ndarray              # (W,) x-coordinates (EPSG:3035 metres)
    y: np.ndarray              # (H,) y-coordinates
    terrain: dict              # var_name -> (H, W) float32 array
    targets: np.ndarray        # (H, W, 5) int8, -1 = NaN/unlabeled
    valid_mask: np.ndarray     # (H, W) bool — land pixels with valid dtm

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.terrain.values())).shape

    @property
    def n_land_pixels(self) -> int:
        return int(self.valid_mask.sum())

    def __repr__(self) -> str:
        h, w = self.shape
        return (f"RegionRaster(region={self.region}, shape={h}x{w}, "
                f"land_pixels={self.n_land_pixels:,})")


def load_terrain_raster(nc_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """
    Load terrain NetCDF → 2-D arrays.

    Returns:
        x: (W,) x-coordinates
        y: (H,) y-coordinates
        terrain_dict: {var_name: (H, W) array}
        targets: (H, W, 5) int8 with -1 for unlabeled
        valid_mask: (H, W) bool — pixels with valid dtm (= land)
    """
    nc_path = Path(nc_path)
    print(f"  Loading terrain: {nc_path.name}")
    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    x = ds["x"].values.astype(np.float32)
    y = ds["y"].values.astype(np.float32)

    # Land mask = pixels with valid elevation
    dtm = ds["dtm"].values.astype(np.float32)   # (H, W)
    valid_mask = ~np.isnan(dtm)

    # Load all terrain variables
    terrain_dict = {}
    for var in TERRAIN_RAW_VARS:
        if var in ds.data_vars:
            arr = ds[var].values.astype(np.float32)
            terrain_dict[var] = arr
        else:
            print(f"    WARNING: {var} not found in terrain file")

    # Load targets — convert NaN to -1 so we can store as int8
    targets = np.full((dtm.shape[0], dtm.shape[1], 5), -1, dtype=np.int8)
    for i, rv in enumerate(RISK_VARS):
        if rv in ds.data_vars:
            arr = ds[rv].values
            mask = ~np.isnan(arr)
            targets[:, :, i] = np.where(mask, arr.astype(np.int8), -1)

    ds.close()
    return x, y, terrain_dict, targets, valid_mask


def load_weather_arrays(nc_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load weather NetCDF.

    Returns:
        x_w: (W_w,) x-coordinates of weather grid (coarser than terrain)
        y_w: (H_w,) y-coordinates
        weather_dict: {var_name: (T, H_w, W_w) array} — full time series
    """
    nc_path = Path(nc_path)
    print(f"  Loading weather: {nc_path.name}")
    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    x_w = ds["x"].values.astype(np.float32)
    y_w = ds["y"].values.astype(np.float32)

    weather_dict = {}
    for var in WEATHER_VARS:
        if var in ds.data_vars:
            weather_dict[var] = ds[var].values.astype(np.float32)
        else:
            print(f"    WARNING: {var} not found in weather file")

    ds.close()
    return x_w, y_w, weather_dict


def build_region_raster(
    region: str,
    terrain_path: str | Path,
    weather_path: str | Path,
    feature_engineer,                # callable: (raster, weather_dict, x_w, y_w) -> dict
) -> RegionRaster:
    """
    Build a complete RegionRaster for a region:
      1. Load terrain NetCDF → 2-D arrays
      2. Load weather NetCDF → time series
      3. Engineer features on the raster (delegated to features.py)
    """
    print(f"\n[build_region_raster] {region.upper()}")
    x, y, terrain_dict, targets, valid_mask = load_terrain_raster(terrain_path)
    x_w, y_w, weather_dict = load_weather_arrays(weather_path)

    h, w = terrain_dict["dtm"].shape
    print(f"  Terrain grid: {h}x{w}  |  Weather grid: {weather_dict['tp'].shape[1]}x{weather_dict['tp'].shape[2]}")
    print(f"  Land pixels: {int(valid_mask.sum()):,} ({100*valid_mask.mean():.1f}%)")

    # Delegate feature engineering to features.py
    print("  Engineering features ...")
    engineered = feature_engineer(
        terrain_dict, valid_mask, x, y,
        weather_dict, x_w, y_w,
    )
    print(f"  Engineered {len(engineered)} feature channels")

    return RegionRaster(
        region=region,
        x=x, y=y,
        terrain=engineered,
        targets=targets,
        valid_mask=valid_mask,
    )


def save_raster(raster: RegionRaster, out_path: str | Path) -> None:
    """Save a RegionRaster to .npz for fast reload."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        region=np.array([raster.region]),
        x=raster.x, y=raster.y,
        targets=raster.targets,
        valid_mask=raster.valid_mask,
        **{f"terrain__{k}": v for k, v in raster.terrain.items()},
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path.name} ({size_mb:.0f} MB)")


def load_raster(in_path: str | Path, feature_names: list[str] | None = None) -> RegionRaster:
    """Load a saved RegionRaster from .npz.

    feature_names: if given, only load these terrain features (saves RAM).
    """
    in_path = Path(in_path)
    data = np.load(in_path, allow_pickle=False)
    needed = set(feature_names) if feature_names is not None else None
    terrain = {
        k[len("terrain__"):]: data[k]
        for k in data.files
        if k.startswith("terrain__") and (needed is None or k[len("terrain__"):] in needed)
    }
    region = str(data["region"][0])
    return RegionRaster(
        region=region,
        x=data["x"], y=data["y"],
        terrain=terrain,
        targets=data["targets"],
        valid_mask=data["valid_mask"].astype(bool),
    )
