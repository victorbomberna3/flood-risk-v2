"""
normalize.py — Per-region robust normalisation.

THIS IS THE MOST IMPORTANT FILE FOR CROSS-REGION TRANSFER.

The previous approach (XGBoost or CNN) implicitly assumed Severn and
Northumbria have similar feature distributions. They don't:
  - Severn has elevations 0-500m, Northumbria 0-800m
  - Severn has milder, wetter climate
  - Severn has different land cover composition

If you train on Severn's raw features and test on Northumbria's raw features,
the model sees out-of-distribution inputs at test time.

Fix: normalise EACH region INDEPENDENTLY using robust statistics (median, IQR).
After normalisation, Severn's "median elevation" maps to 0 and Northumbria's
"median elevation" also maps to 0. The model now sees the same distribution.

Why robust (median/IQR) vs z-score (mean/std):
  - Flood data is heavy-tailed (HAND, flow_acc, precipitation extremes)
  - Mean/std are dominated by outliers
  - Median/IQR ignore the tails when computing the centre/spread
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class NormStats:
    """Per-region normalisation statistics."""
    region: str
    features: list[str] = field(default_factory=list)
    centers: dict = field(default_factory=dict)   # feature -> median
    scales: dict = field(default_factory=dict)    # feature -> IQR (or std)
    method: str = "robust"

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "features": self.features,
            "centers": {k: float(v) for k, v in self.centers.items()},
            "scales": {k: float(v) for k, v in self.scales.items()},
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Features that should NOT be normalised (already in [0, 1] or categorical)
SKIP_NORMALIZE = {
    "aspect_sin", "aspect_cos", "flow_dir_sin", "flow_dir_cos",
    "near_water",  # binary
}


def fit_norm_stats(
    features: dict,                # name -> (H, W) array
    valid_mask: np.ndarray,        # (H, W) bool
    region: str,
    method: str = "robust",
) -> NormStats:
    """
    Compute per-feature centre and scale, using ONLY valid (land) pixels.

    Important: we use only land pixels for statistics — including NaN/sea
    would dominate the median.
    """
    stats = NormStats(region=region, method=method)

    for name, arr in features.items():
        if name in SKIP_NORMALIZE:
            stats.centers[name] = 0.0
            stats.scales[name] = 1.0
            stats.features.append(name)
            continue

        vals = arr[valid_mask]
        if vals.size == 0:
            stats.centers[name] = 0.0
            stats.scales[name] = 1.0
        elif method == "robust":
            median = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            stats.centers[name] = float(median)
            stats.scales[name] = float(max(iqr, 1e-6))
        elif method == "zscore":
            stats.centers[name] = float(np.mean(vals))
            stats.scales[name] = float(max(np.std(vals), 1e-6))
        else:
            raise ValueError(f"Unknown method: {method}")
        stats.features.append(name)

    return stats


def apply_normalization(features: dict, stats: NormStats) -> dict:
    """
    Apply z-score-style transform using the given stats.

    Returns a new dict with normalised arrays (same shapes).
    """
    out = {}
    for name, arr in features.items():
        if name not in stats.centers:
            print(f"  WARNING: no stats for feature {name}, copying as-is")
            out[name] = arr
            continue
        c = stats.centers[name]
        s = stats.scales[name]
        out[name] = ((arr - c) / s).astype(np.float32)
    return out


def normalize_region(
    features: dict,
    valid_mask: np.ndarray,
    region: str,
    method: str = "robust",
    save_path: str | Path | None = None,
) -> tuple[dict, NormStats]:
    """
    One-step: fit + transform + save stats.

    Returns (normalised_features, stats).
    """
    stats = fit_norm_stats(features, valid_mask, region=region, method=method)
    normalised = apply_normalization(features, stats)

    if save_path is not None:
        stats.save(save_path)
        print(f"    Saved norm stats → {save_path}")

    # Quick diagnostic
    sample_feat = next(iter(normalised.keys()))
    vals = normalised[sample_feat][valid_mask]
    print(f"    Diagnostic ({sample_feat}): median={np.median(vals):.3f}, IQR={np.percentile(vals, 75)-np.percentile(vals, 25):.3f}")

    return normalised, stats


def stack_channels(
    features: dict,
    channel_order: list[str],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Stack feature dict into (C, H, W) tensor in the order specified by config.

    Missing features are filled with zeros (and warned about).
    Invalid pixels (sea/NaN) are zeroed out so the model doesn't see garbage.
    """
    h, w = next(iter(features.values())).shape
    channels = []
    for name in channel_order:
        if name in features:
            arr = features[name].astype(np.float32)
        else:
            print(f"  WARNING: requested feature '{name}' not available, zeros used")
            arr = np.zeros((h, w), dtype=np.float32)
        if valid_mask is not None:
            arr = np.where(valid_mask, arr, 0.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        channels.append(arr)
    stacked = np.stack(channels, axis=0)  # (C, H, W)
    return stacked
