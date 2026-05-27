"""
predict.py — Sliding-window inference with patch averaging.

Inference strategy:
  1. Tile Northumbria into overlapping patches (50% overlap)
  2. Predict each patch
  3. Average predictions in overlap regions
  4. Reconstruct full prediction raster
  5. Optional TTA: average over 8 dihedral transformations
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PatchDataset, apply_dihedral, make_patch_grid


@torch.no_grad()
def predict_region(
    model: torch.nn.Module,
    channels: np.ndarray,           # (C, H, W) — normalised features
    valid_mask: np.ndarray,         # (H, W) bool
    feature_names: list[str],
    n_tasks: int = 5,
    n_classes: int = 5,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    tta: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Run sliding-window inference. Returns (5, H, W) int8 prediction array.

    Approach: maintain (5, n_classes, H, W) accumulator of softmax probabilities,
    averaged over all overlapping patches and (optionally) 8 TTA variants.
    """
    model.eval()
    _, h, w = channels.shape

    # Accumulator: average softmax probabilities per task
    probs = np.zeros((n_tasks, n_classes, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    patches = make_patch_grid(h, w, patch_size, stride)
    # Filter to patches that have at least some land
    patches = [(py, px) for (py, px) in patches if valid_mask[py:py+patch_size, px:px+patch_size].mean() > 0.05]
    print(f"  Inference: {len(patches):,} patches, batch_size={batch_size}, tta={tta}")

    n_transforms = 8 if tta else 1

    # Process in batches
    for batch_start in tqdm(range(0, len(patches), batch_size), disable=not show_progress):
        batch_patches = patches[batch_start:batch_start + batch_size]
        # Build batch tensor (B, C, H, W) on CPU first, then move
        batch_x = np.stack([
            np.where(valid_mask[py:py+patch_size, px:px+patch_size][None, :, :],
                     channels[:, py:py+patch_size, px:px+patch_size], 0.0)
            for (py, px) in batch_patches
        ]).astype(np.float32)   # (B, C, H, W)

        # TTA: average softmax across transformations
        batch_probs = [np.zeros((len(batch_patches), n_classes, patch_size, patch_size), dtype=np.float32)
                       for _ in range(n_tasks)]

        for tid in range(n_transforms):
            # Apply transform to whole batch
            x_aug = np.stack([apply_dihedral(b, None, tid, feature_names)[0] for b in batch_x])
            x_t = torch.from_numpy(x_aug).to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(x_t)  # list of (B, n_classes, H, W)

            # Inverse-transform predictions
            inv_tid = _inverse_transform_id(tid)
            for ti, logits in enumerate(outputs):
                p = F.softmax(logits, dim=1).cpu().float().numpy()  # (B, n_classes, H, W)
                # Apply inverse transform per sample
                p_inv = np.stack([
                    apply_dihedral(p[b], None, inv_tid, _spatial_only_features())[0]
                    for b in range(p.shape[0])
                ])
                batch_probs[ti] += p_inv / n_transforms

        # Accumulate into global probs
        for i, (py, px) in enumerate(batch_patches):
            for ti in range(n_tasks):
                probs[ti, :, py:py+patch_size, px:px+patch_size] += batch_probs[ti][i]
            counts[py:py+patch_size, px:px+patch_size] += 1

    # Average over overlapping patches
    counts_safe = np.maximum(counts, 1.0)
    probs /= counts_safe[None, None, :, :]

    # Argmax to get class predictions
    preds = probs.argmax(axis=1).astype(np.int8)  # (5, H, W)

    # Mark non-land as -1
    preds = np.where(valid_mask[None, :, :], preds, -1)
    return preds


def _inverse_transform_id(tid: int) -> int:
    """
    Compute the inverse of a dihedral transform.

    Our apply_dihedral applies F^f first, then R^r:  g = R^r ∘ F^f
    Inverse:  g⁻¹ = F^f ∘ R^(-r)

    Convert g⁻¹ back to standard "F first, then R" form:
      - f = 0:  g⁻¹ = R^(4-r) ∘ I       → tid_inv = (4 - r) % 4
      - f = 1:  F ∘ R^(-r) = R^r ∘ F     (using F·R^k = R^(-k)·F)
                → tid_inv = 4 + r
    """
    rot = tid % 4
    flip = tid // 4
    if flip:
        return rot + 4         # f'=1, r'=r
    return (4 - rot) % 4       # f'=0, r'=(4-r) mod 4



def _spatial_only_features() -> list[str]:
    """Return empty list — for probability tensors there are no directional channels."""
    return []


def save_predictions(
    preds: np.ndarray,              # (5, H, W) int8
    valid_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    target_names: list[str],
    out_path: str | Path,
) -> None:
    """Save predictions as a parquet file matching the v1 format."""
    import pandas as pd
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = valid_mask.shape
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Only export valid pixels (saves space)
    mask_flat = valid_mask.ravel()
    df_data = {
        "x": xx.ravel()[mask_flat],
        "y": yy.ravel()[mask_flat],
    }
    for i, tname in enumerate(target_names):
        df_data[f"pred_{tname}"] = preds[i].ravel()[mask_flat]
    df = pd.DataFrame(df_data)
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df):,} predictions → {out_path}")
