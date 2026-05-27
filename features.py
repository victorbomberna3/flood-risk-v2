"""
features.py — Engineer features on 2-D raster arrays.

Ports the v1 tabular feature engineering to raster operations. All features
are computed on (H, W) arrays so they preserve spatial structure for the U-Net.

The set of features is intentionally identical to v1 (HAND, TWI, slope,
aspect, curvature, roughness, rolling weather stats) — those are good.
The change is keeping them as 2-D arrays instead of flattening to rows.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_edt
from scipy.spatial import cKDTree


RAIN_WINDOWS = [1, 3, 7, 14, 30]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_log1p(arr: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.nan_to_num(arr, nan=0.0), 0, None)).astype(np.float32)


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling sum over axis 0 (time). arr shape: (T, ...).

    Returns array of same shape; first `window` entries are partial sums.
    """
    cs = np.cumsum(arr, axis=0)
    roll = cs.copy()
    roll[window:] = cs[window:] - cs[:-window]
    return roll


# ── Terrain feature engineering ──────────────────────────────────────────────

def engineer_terrain(terrain_dict: dict, valid_mask: np.ndarray) -> dict:
    """
    Compute terrain features from raw DTM + flow_acc + flow_dir.
    All outputs are (H, W) float32 arrays.

    Critically: NO absolute elevation, NO absolute coordinates. Everything
    is relative or local so it transfers across regions.
    """
    dtm = terrain_dict["dtm"]
    h, w = dtm.shape
    dtm_filled = np.nan_to_num(dtm, nan=0.0).astype(np.float32)
    valid_f = valid_mask.astype(np.float32)

    out = {}

    # Relative elevation at 3 scales (60m / 220m / 1km neighbourhoods)
    for size, label in [(3, "3"), (11, "11"), (51, "51")]:
        ns = uniform_filter(dtm_filled, size=size)
        nc = uniform_filter(valid_f, size=size)
        with np.errstate(invalid="ignore"):
            local_mean = ns / np.maximum(nc, 1e-6)
        out[f"dtm_rel_{label}"] = np.where(valid_mask, dtm_filled - local_mean, 0.0).astype(np.float32)

    # Roughness: local std of elevation (texture)
    dtm_sq = dtm_filled ** 2
    for size, label in [(3, "3"), (11, "11")]:
        ms = uniform_filter(dtm_sq, size=size)
        mv = uniform_filter(dtm_filled, size=size)
        var = np.maximum(ms - mv ** 2, 0.0)
        out[f"roughness_{label}"] = np.sqrt(var).astype(np.float32)

    # Gradient-based: slope, aspect, curvature
    # 20m pixel spacing → use as cell size
    gy, gx = np.gradient(dtm_filled, 20.0, 20.0)

    slope = np.arctan(np.sqrt(gx ** 2 + gy ** 2)).astype(np.float32)
    out["slope"] = slope

    aspect = np.arctan2(gx, gy).astype(np.float32)
    out["aspect_sin"] = np.sin(aspect).astype(np.float32)
    out["aspect_cos"] = np.cos(aspect).astype(np.float32)

    # Curvature: Laplacian (negative=concave/pools, positive=convex/ridges)
    d2zdx2 = np.gradient(gx, 20.0, axis=1)
    d2zdy2 = np.gradient(gy, 20.0, axis=0)
    out["curvature"] = (d2zdx2 + d2zdy2).astype(np.float32)

    # Flow accumulation features
    flow_acc = terrain_dict.get("flow_acc")
    if flow_acc is not None:
        flow_acc_filled = np.nan_to_num(flow_acc, nan=0.0)
        out["log_flow_acc"] = _safe_log1p(flow_acc_filled)

        # Spatial lag: neighbourhood mean of log_flow_acc
        log_fa = out["log_flow_acc"]
        out["flow_acc_lag_3x3"]  = uniform_filter(log_fa, size=3).astype(np.float32)
        out["flow_acc_lag_11x11"] = uniform_filter(log_fa, size=11).astype(np.float32)

        # Spatial anomaly: how much more upstream area does this pixel have
        # than its immediate neighbourhood? Identifies local convergence hotspots.
        out["flow_acc_anomaly"] = (log_fa - out["flow_acc_lag_11x11"]).astype(np.float32)

        # TWI = ln(flow_acc / tan(slope))
        tan_slope = np.tan(slope).clip(min=1e-3)
        fa_clip = np.clip(flow_acc_filled, 1, None)
        out["twi"] = np.log(fa_clip / tan_slope).astype(np.float32)

        # HAND + distance to channel via distance transform
        fa_land = flow_acc_filled[valid_mask]
        if len(fa_land) > 0:
            ch_thresh = np.percentile(fa_land, 99.0)  # top 1% = main channels
            channel_2d = flow_acc_filled >= ch_thresh
            if channel_2d.any():
                dist_arr, nearest_idx = distance_transform_edt(~channel_2d, return_indices=True)
                ch_dtm = dtm_filled[nearest_idx[0], nearest_idx[1]]
                hand = (dtm_filled - ch_dtm).astype(np.float32)
                # log distance is better-scaled than raw metres
                out["hand"] = hand
                out["log_dist_channel"] = _safe_log1p(dist_arr.astype(np.float32))
            else:
                out["hand"] = np.zeros_like(dtm_filled)
                out["log_dist_channel"] = np.zeros_like(dtm_filled)
        else:
            out["hand"] = np.zeros_like(dtm_filled)
            out["log_dist_channel"] = np.zeros_like(dtm_filled)

    # Flow direction (circular)
    flow_dir = terrain_dict.get("flow_dir")
    if flow_dir is not None:
        fd_filled = np.nan_to_num(flow_dir, nan=0.0)
        fd_rad = np.deg2rad(fd_filled)
        out["flow_dir_sin"] = np.sin(fd_rad).astype(np.float32)
        out["flow_dir_cos"] = np.cos(fd_rad).astype(np.float32)

    # Categorical / continuous as-is
    for var in ["waw", "imd", "rciw"]:
        if var in terrain_dict:
            out[var] = np.nan_to_num(terrain_dict[var], nan=0.0).astype(np.float32)

    # Near-water binary
    if "rciw" in terrain_dict:
        out["near_water"] = (np.nan_to_num(terrain_dict["rciw"], nan=0.0) > 0).astype(np.float32)

    return out


# ── Weather feature engineering ──────────────────────────────────────────────

def engineer_weather(weather_dict: dict) -> dict:
    """
    Compute weather features from time series.

    Each output is a (H_w, W_w) array — one value per weather grid cell,
    aggregated over the full 10-year time series.

    Targets are annual exceedance probabilities → static climatology features
    are appropriate (no LSTM needed, see notebook for reasoning).
    """
    out = {}

    # Precipitation rolling windows
    if "tp" in weather_dict:
        tp = weather_dict["tp"]  # (T, H, W)
        for w in RAIN_WINDOWS:
            roll = _rolling_sum(tp, w)
            out[f"precip_roll{w}d_max"] = roll.max(axis=0).astype(np.float32)
            out[f"precip_roll{w}d_mean"] = roll.mean(axis=0).astype(np.float32)
            if w in (7, 30):
                out[f"precip_p95_{w}d"] = np.percentile(roll, 95, axis=0).astype(np.float32)

        # Intensity ratio: peak / climatology — region-invariant extremeness
        if "precip_roll30d_mean" in out:
            denom = np.clip(out["precip_roll30d_mean"], 1e-8, None)
            out["precip_intensity"] = np.clip(out["precip_roll1d_max"] / denom, 0, 100).astype(np.float32)

    # Runoff
    if "sro" in weather_dict:
        sro = weather_dict["sro"]
        for w in [7, 30]:
            roll = _rolling_sum(sro, w)
            out[f"runoff_roll{w}d_max"] = roll.max(axis=0).astype(np.float32)
            out[f"runoff_roll{w}d_mean"] = roll.mean(axis=0).astype(np.float32)
        if "runoff_roll30d_mean" in out:
            denom = np.clip(out["runoff_roll30d_mean"], 1e-8, None)
            out["runoff_intensity"] = np.clip(out["runoff_roll7d_max"] / denom, 0, 100).astype(np.float32)

    # Soil moisture
    if "swvl1_mean" in weather_dict:
        swvl = weather_dict["swvl1_mean"]  # (T, H, W)
        out["soil_moist_max"] = np.nan_to_num(np.nanmax(swvl, axis=0), nan=0.0).astype(np.float32)
        out["soil_moist_mean"] = np.nan_to_num(np.nanmean(swvl, axis=0), nan=0.0).astype(np.float32)
        out["soil_moist_std"] = np.nan_to_num(np.nanstd(swvl, axis=0), nan=0.0).astype(np.float32)

    # Wind / temp / dewpoint — INTENTIONALLY OMITTED.
    # These act as regional proxies (Severn is mild/wet, Northumbria is cold/windy).
    # Including them teaches the model to identify the region, not predict floods.

    return out


# ── Spatial alignment: weather grid → terrain grid ───────────────────────────

def align_weather_to_terrain(
    weather_features: dict,    # var -> (H_w, W_w)
    x_w: np.ndarray,           # weather x-coords
    y_w: np.ndarray,           # weather y-coords
    x_t: np.ndarray,           # terrain x-coords
    y_t: np.ndarray,           # terrain y-coords
) -> dict:
    """
    Upsample weather features (coarse grid) to terrain resolution
    using nearest-neighbour lookup.

    Why nearest-neighbour over bilinear:
      - Bilinear smooths extreme rainfall values — exactly what we want to keep
      - Nearest is honest about the underlying resolution
      - 9km weather cells contain ~450 terrain pixels each; smoothing within
        a cell would create artificial gradients
    """
    # Build KD-tree on weather cell centres
    h_w, w_w = next(iter(weather_features.values())).shape
    xx_w, yy_w = np.meshgrid(x_w, y_w)
    w_coords = np.column_stack([xx_w.ravel(), yy_w.ravel()])

    # Build terrain coordinates
    h_t, w_t = len(y_t), len(x_t)
    xx_t, yy_t = np.meshgrid(x_t, y_t)
    t_coords = np.column_stack([xx_t.ravel(), yy_t.ravel()])

    # Nearest-neighbour lookup (terrain → weather)
    tree = cKDTree(w_coords)
    _, idx = tree.query(t_coords, k=1)

    # Apply lookup for each feature
    out = {}
    for var, arr_w in weather_features.items():
        out[var] = arr_w.ravel()[idx].reshape(h_t, w_t).astype(np.float32)

    return out


# ── Top-level entry point ────────────────────────────────────────────────────

def engineer_features(
    terrain_dict: dict,
    valid_mask: np.ndarray,
    x_t: np.ndarray,
    y_t: np.ndarray,
    weather_dict: dict,
    x_w: np.ndarray,
    y_w: np.ndarray,
) -> dict:
    """
    Engineer all features and return a merged dict of (H_t, W_t) arrays
    ready to be stacked into U-Net input channels.
    """
    terrain_feats = engineer_terrain(terrain_dict, valid_mask)
    print(f"    terrain features: {len(terrain_feats)}")

    weather_coarse = engineer_weather(weather_dict)
    print(f"    weather features (coarse): {len(weather_coarse)}")

    weather_fine = align_weather_to_terrain(weather_coarse, x_w, y_w, x_t, y_t)
    print(f"    weather features (upsampled to terrain): {len(weather_fine)}")

    return {**terrain_feats, **weather_fine}
