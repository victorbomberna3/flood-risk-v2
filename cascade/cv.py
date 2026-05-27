"""Spatial block cross-validation.

Pixels are strongly spatially autocorrelated, so random k-fold leaks
information across the train/validation boundary and inflates scores. We
instead group pixels into square spatial blocks and split on whole blocks with
GroupKFold. The same machinery provides the "mini-Northumbria" high-elevation
hold-out used as an in-region domain-shift stress test.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def assign_blocks(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    block_size_m: int = 20_000,
) -> np.ndarray:
    """Assign each pixel to a square spatial block ID.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Projected coordinates (metres, EPSG:3035) per pixel.
    block_size_m : int
        Block edge length in metres. 20 km blocks balance having enough blocks
        for 5 folds against keeping spatial autocorrelation inside a block.

    Returns
    -------
    np.ndarray
        Integer block ID per pixel.
    """
    bx = np.floor(np.asarray(x_coords) / block_size_m).astype(np.int64)
    by = np.floor(np.asarray(y_coords) / block_size_m).astype(np.int64)
    return (by * 100_000 + bx).astype(np.int64)


def make_spatial_folds(
    block_id: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build n_splits spatial-block folds.

    Parameters
    ----------
    block_id : np.ndarray
        Block ID per pixel from assign_blocks.
    n_splits : int
        Number of folds.
    seed : int
        Accepted for uniform call signature; GroupKFold is deterministic.

    Returns
    -------
    list of (train_idx, valid_idx)
        Positional index arrays into the pixel arrays.
    """
    del seed
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(block_id), 1), dtype=np.int8)
    folds = list(gkf.split(dummy_X, groups=block_id))
    for i, (tr, va) in enumerate(folds):
        logger.info(
            "Fold %d: %d train / %d valid pixels, %d valid blocks",
            i, len(tr), len(va), len(np.unique(block_id[va])),
        )
    return folds


def fold_labels(
    block_id: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Per-pixel validation-fold index from a fold list.

    Parameters
    ----------
    block_id : np.ndarray
        Block ID per pixel (only its length is used here).
    folds : list of (train_idx, valid_idx)
        Folds from make_spatial_folds.

    Returns
    -------
    np.ndarray
        Integer fold index per pixel. Pixels never validated are -1.

    Notes
    -----
    Used to align Stage A and Stage B folds: Stage B trains/validates on the
    same spatial blocks as Stage A, avoiding cross-stage leakage.
    """
    lab = np.full(len(block_id), -1, dtype=np.int64)
    for i, (_, va) in enumerate(folds):
        lab[va] = i
    return lab


def folds_from_labels(
    labels: np.ndarray,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rebuild (train_idx, valid_idx) folds from a per-pixel fold-label array.

    Parameters
    ----------
    labels : np.ndarray
        Per-pixel validation-fold index (subset of fold_labels output).
    n_splits : int
        Number of folds.

    Returns
    -------
    list of (train_idx, valid_idx)
        Positional indices into labels.
    """
    labels = np.asarray(labels)
    all_idx = np.arange(len(labels))
    folds = []
    for i in range(n_splits):
        va = all_idx[labels == i]
        tr = all_idx[(labels != i) & (labels >= 0)]
        folds.append((tr, va))
    return folds


def find_mini_northumbria_block(
    block_id: np.ndarray,
    dtm: np.ndarray,
    min_block_pixels: int = 5_000,
) -> int:
    """Pick the highest mean-elevation block as a Northumbria proxy.

    Northumbria sits ~100 m higher than Severn on average. Holding out the
    highest-elevation Severn block approximates the domain shift at test time.
    """
    block_id = np.asarray(block_id)
    dtm = np.asarray(dtm, dtype=np.float64)
    uniq = np.unique(block_id)
    best_id, best_mean = None, -np.inf
    for b in uniq:
        m = block_id == b
        if m.sum() < min_block_pixels:
            continue
        mean_elev = np.nanmean(dtm[m])
        if mean_elev > best_mean:
            best_mean, best_id = mean_elev, int(b)
    if best_id is None:
        raise ValueError(
            f"No block has >= {min_block_pixels} pixels; lower min_block_pixels."
        )
    logger.info("Mini-Northumbria block %d, mean elevation %.1f m", best_id, best_mean)
    return best_id


__all__ = [
    "assign_blocks", "make_spatial_folds", "fold_labels",
    "folds_from_labels", "find_mini_northumbria_block",
]
