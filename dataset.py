"""
dataset.py — PyTorch Dataset for U-Net training & inference.

Key responsibilities:
  1. Extract 256x256 patches from large rasters
  2. Filter patches with too much sea/NaN
  3. Spatial block-based train/val split (NOT random)
  4. Augmentation (D4 dihedral group: 4 rotations x 2 flips)
  5. Handle directional features correctly under rotation/flipping
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


# Directional channels — these need special handling under rotation/flips.
# The CONFIG_FEATURE_ORDER (passed in) determines channel indices.
DIRECTIONAL_PAIRS = [
    ("aspect_sin", "aspect_cos"),
    ("flow_dir_sin", "flow_dir_cos"),
]


def make_patch_grid(
    h: int, w: int, size: int, stride: int,
) -> list[tuple[int, int]]:
    """Generate top-left corners for sliding-window patches."""
    ys = list(range(0, h - size + 1, stride))
    xs = list(range(0, w - size + 1, stride))
    # Include the last row/column if not aligned
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)
    return [(y, x) for y in ys for x in xs]


def patch_quality(
    patch_valid: np.ndarray,
    patch_targets: np.ndarray | None,
    min_valid_frac: float,
    min_labeled_frac: float,
) -> bool:
    """Return True if this patch is usable for training."""
    valid_frac = patch_valid.mean()
    if valid_frac < min_valid_frac:
        return False
    if patch_targets is not None and min_labeled_frac > 0:
        # patch_targets shape (H, W, 5), -1 = unlabeled
        labeled = (patch_targets >= 0).any(axis=-1)
        if labeled.mean() < min_labeled_frac:
            return False
    return True


def split_patches_spatial_block(
    patches: list[tuple[int, int]],
    h: int, w: int,
    val_fraction: float = 0.2,
    block_size: int = 1024,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Split patches into train/val by SPATIAL BLOCKS, not randomly.

    Why: random pixel splits leak through spatial autocorrelation
    (adjacent pixels are highly correlated). Spatial blocks give a
    realistic estimate of cross-region transfer performance.

    Procedure: tile the raster into block_size x block_size blocks,
    randomly assign each block to train or val.
    """
    rng = np.random.default_rng(seed)
    n_block_y = (h + block_size - 1) // block_size
    n_block_x = (w + block_size - 1) // block_size
    n_blocks = n_block_y * n_block_x

    # Randomly assign each block to train (0) or val (1)
    block_assign = rng.random(n_blocks) < val_fraction
    block_assign = block_assign.reshape(n_block_y, n_block_x)

    train_patches, val_patches = [], []
    for (py, px) in patches:
        by = py // block_size
        bx = px // block_size
        by = min(by, n_block_y - 1)
        bx = min(bx, n_block_x - 1)
        if block_assign[by, bx]:
            val_patches.append((py, px))
        else:
            train_patches.append((py, px))

    return train_patches, val_patches


def apply_dihedral(
    x: np.ndarray,                # (C, H, W)
    y: np.ndarray | None,         # (5, H, W) or None
    transform_id: int,            # 0..7
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Apply one of 8 dihedral transformations (D4 group).

    transform_id encoding:
      0: identity
      1: rot90
      2: rot180
      3: rot270
      4: flip horizontal
      5: flip horizontal + rot90
      6: flip horizontal + rot180
      7: flip horizontal + rot270

    Directional sin/cos channels are updated accordingly so that
    "north" remains "north" relative to the rotated patch.
    """
    rot = transform_id % 4
    flip = transform_id // 4

    if flip == 1:
        x = x[:, :, ::-1].copy()
        if y is not None: y = y[:, :, ::-1].copy()

    if rot > 0:
        x = np.rot90(x, k=rot, axes=(1, 2)).copy()
        if y is not None: y = np.rot90(y, k=rot, axes=(1, 2)).copy()

    # Update directional channels — flip changes x-direction,
    # rotation rotates the (sin, cos) pair by 90deg increments.
    for sin_name, cos_name in DIRECTIONAL_PAIRS:
        if sin_name in feature_names and cos_name in feature_names:
            i_sin = feature_names.index(sin_name)
            i_cos = feature_names.index(cos_name)
            sin_ch = x[i_sin].copy()
            cos_ch = x[i_cos].copy()

            # Horizontal flip: sin -> -sin (assuming sin tracks x-component)
            if flip == 1:
                sin_ch = -sin_ch

            # Rotation by k*90deg: (sin, cos) -> rotation matrix application.
            # Convention: aspect_sin = sin(angle), aspect_cos = cos(angle).
            # Rotating the image by 90deg CCW shifts angles by -90deg.
            # new_sin = sin(angle - 90deg*k) = ... use rotation matrix
            if rot > 0:
                theta = -np.pi / 2 * rot  # CCW rotation
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                new_sin = sin_ch * cos_t - cos_ch * sin_t
                new_cos = sin_ch * sin_t + cos_ch * cos_t
                sin_ch, cos_ch = new_sin, new_cos

            x[i_sin] = sin_ch
            x[i_cos] = cos_ch

    return x, y


class PatchDataset(Dataset):
    """
    Yields (input_tensor, target_tensor, valid_mask) for one patch.

    input_tensor: (C, H, W) float32, with valid_mask applied (NaN→0)
    target_tensor: (5, H, W) int64 with -1 for unlabeled pixels
    valid_mask: (H, W) bool — pixels with valid features
    """
    def __init__(
        self,
        channels: np.ndarray,         # (C, H, W) float32 — normalised features
        targets: np.ndarray,           # (H, W, 5) int8 with -1 for unlabeled
        valid_mask: np.ndarray,        # (H, W) bool
        patches: list[tuple[int, int]],
        feature_names: list[str],
        patch_size: int = 256,
        augment: bool = False,
        labels_required: bool = True,
    ):
        self.channels = channels
        self.targets = np.transpose(targets, (2, 0, 1))  # (5, H, W)
        self.valid_mask = valid_mask
        self.patches = patches
        self.feature_names = feature_names
        self.patch_size = patch_size
        self.augment = augment
        self.labels_required = labels_required

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx):
        y0, x0 = self.patches[idx]
        ps = self.patch_size
        y1, x1 = y0 + ps, x0 + ps

        x = self.channels[:, y0:y1, x0:x1].copy()             # (C, H, W)
        t = self.targets[:, y0:y1, x0:x1].astype(np.int64).copy()   # (5, H, W)
        vm = self.valid_mask[y0:y1, x0:x1].copy()             # (H, W)

        # Mask invalid pixels in channels (set to 0)
        x = np.where(vm[None, :, :], x, 0.0).astype(np.float32)

        # Augmentation
        if self.augment:
            tid = np.random.randint(0, 8)
            x, t = apply_dihedral(x, t, tid, self.feature_names)
            # vm rotates too
            rot = tid % 4
            flip = tid // 4
            if flip == 1: vm = vm[:, ::-1].copy()
            if rot > 0:  vm = np.rot90(vm, k=rot).copy()

        return (
            torch.from_numpy(x),
            torch.from_numpy(t),
            torch.from_numpy(vm),
        )


def build_train_dataset(
    channels: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    feature_names: list[str],
    patch_size: int = 256,
    stride: int = 128,
    min_valid_frac: float = 0.5,
    min_labeled_frac: float = 0.1,
    val_fraction: float = 0.2,
    block_size: int = 1024,
    augment_train: bool = True,
    seed: int = 42,
) -> tuple[PatchDataset, PatchDataset, dict]:
    """
    Build train + val datasets from a single region's raster.

    Spatial block split avoids data leakage.
    """
    _, h, w = channels.shape
    candidates = make_patch_grid(h, w, patch_size, stride)
    print(f"  Generated {len(candidates):,} candidate patches")

    # Filter by quality
    targets_hwc = np.transpose(targets, (1, 2, 0)) if targets.ndim == 3 and targets.shape[0] == 5 else targets
    keep = []
    for (py, px) in candidates:
        ps = patch_size
        pv = valid_mask[py:py+ps, px:px+ps]
        pt = targets[py:py+ps, px:px+ps, :] if targets.ndim == 3 else None
        if patch_quality(pv, pt, min_valid_frac, min_labeled_frac):
            keep.append((py, px))
    print(f"  After quality filter: {len(keep):,} patches")

    train_p, val_p = split_patches_spatial_block(keep, h, w, val_fraction, block_size, seed)
    print(f"  Spatial split: {len(train_p):,} train  |  {len(val_p):,} val")

    train_ds = PatchDataset(
        channels, targets, valid_mask, train_p, feature_names,
        patch_size, augment=augment_train, labels_required=True,
    )
    val_ds = PatchDataset(
        channels, targets, valid_mask, val_p, feature_names,
        patch_size, augment=False, labels_required=True,
    )

    info = {"n_train": len(train_p), "n_val": len(val_p), "n_candidates": len(candidates)}
    return train_ds, val_ds, info


def build_inference_dataset(
    channels: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    feature_names: list[str],
    patch_size: int = 256,
    stride: int = 128,
    min_valid_frac: float = 0.05,   # lenient — predict almost everywhere
) -> PatchDataset:
    """Build full-coverage dataset for inference on a test region."""
    _, h, w = channels.shape
    candidates = make_patch_grid(h, w, patch_size, stride)

    keep = []
    for (py, px) in candidates:
        ps = patch_size
        pv = valid_mask[py:py+ps, px:px+ps]
        if pv.mean() >= min_valid_frac:
            keep.append((py, px))
    print(f"  Inference patches: {len(keep):,}")

    return PatchDataset(
        channels, targets, valid_mask, keep, feature_names,
        patch_size, augment=False, labels_required=False,
    )
