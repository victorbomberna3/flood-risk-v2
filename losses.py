"""
losses.py — Masked weighted cross-entropy.

Critical features:
  1. Multi-task: sum CE across 5 depths, optionally weighted per task.
  2. Class weighting: inverse frequency to combat heavy class imbalance.
  3. NaN masking: invalid/unlabeled pixels (-1) don't contribute to loss.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    targets: np.ndarray,           # (H, W, 5) int8 with -1 for unlabeled
    n_classes: int = 5,
    method: str = "inverse_freq",
    beta: float = 0.999,            # for effective_num
) -> dict[int, np.ndarray]:
    """
    Compute per-task class weights from the full Severn raster.

    Returns dict {task_idx: weight_vector of length n_classes}.
    """
    weights = {}
    for t_idx in range(targets.shape[-1]):
        y = targets[:, :, t_idx]
        mask = y >= 0
        if not mask.any():
            weights[t_idx] = np.ones(n_classes, dtype=np.float32)
            continue
        counts = np.bincount(y[mask], minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1)
        if method == "inverse_freq":
            w = counts.sum() / (n_classes * counts)
        elif method == "effective_num":
            # Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples"
            eff = (1 - np.power(beta, counts)) / (1 - beta)
            w = 1.0 / eff
            w = w * (n_classes / w.sum())
        else:
            raise ValueError(f"Unknown class_weight_method: {method}")
        weights[t_idx] = w.astype(np.float32)
    return weights


class MaskedMultiTaskLoss(nn.Module):
    """
    Sum of weighted cross-entropy losses across 5 flood depths.

    Args:
        class_weights: dict {task_idx: (n_classes,) tensor}
        task_weights:  (5,) tensor — per-depth weighting
        ignore_index:  pixels with target == this value are ignored
        focal_gamma:   if > 0, apply focal loss instead of standard CE
    """
    def __init__(
        self,
        class_weights: dict[int, torch.Tensor],
        task_weights: torch.Tensor,
        ignore_index: int = -1,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.task_weights = nn.Parameter(task_weights, requires_grad=False)
        # Register class weights as buffers (one per task)
        for t_idx, w in class_weights.items():
            self.register_buffer(f"cw_{t_idx}", w, persistent=False)

    def forward(
        self,
        outputs: list[torch.Tensor],   # list of 5 (B, n_classes, H, W)
        targets: torch.Tensor,         # (B, 5, H, W) int64
        valid_mask: torch.Tensor,      # (B, H, W) bool
    ) -> tuple[torch.Tensor, dict]:
        total = torch.zeros(1, device=outputs[0].device)
        active_weight = torch.zeros(1, device=outputs[0].device)
        per_task_losses = {}

        for t_idx, logits in enumerate(outputs):
            tgt = targets[:, t_idx]                   # (B, H, W)
            tgt = torch.where(valid_mask, tgt, torch.full_like(tgt, self.ignore_index))

            if (tgt != self.ignore_index).sum() == 0:
                continue

            cw = getattr(self, f"cw_{t_idx}")

            if self.focal_gamma > 0:
                loss = self._focal_loss(logits, tgt, cw, self.focal_gamma)
            else:
                loss = F.cross_entropy(
                    logits, tgt,
                    weight=cw,
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )

            per_task_losses[t_idx] = loss.detach()
            total = total + self.task_weights[t_idx] * loss
            active_weight = active_weight + self.task_weights[t_idx]

        return total / active_weight.clamp(min=1e-6), per_task_losses

    @staticmethod
    def _focal_loss(
        logits: torch.Tensor,       # (B, C, H, W)
        targets: torch.Tensor,      # (B, H, W) with -1 ignored
        class_weights: torch.Tensor,  # (C,)
        gamma: float = 2.0,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        # gather log_prob for the true class
        # Replace ignore_index with 0 temporarily, mask the result
        valid = targets != ignore_index
        safe_tgt = torch.where(valid, targets, torch.zeros_like(targets))
        log_pt = log_probs.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = -((1 - pt) ** gamma) * log_pt
        # apply class weights
        w = class_weights[safe_tgt]
        focal = focal * w
        focal = focal * valid.float()
        return focal.sum() / valid.float().sum().clamp(min=1.0)


def build_loss(
    targets: np.ndarray,
    config: dict,
    device: torch.device,
) -> MaskedMultiTaskLoss:
    """Construct loss module from raw targets + config."""
    cw_method = config["loss"]["class_weight_method"]
    cw_np = compute_class_weights(targets, n_classes=config["n_classes"], method=cw_method)
    class_weights = {k: torch.from_numpy(v).to(device) for k, v in cw_np.items()}

    tw = torch.tensor(
        [config["loss"]["task_weights"][t] for t in config["targets"]],
        dtype=torch.float32, device=device,
    )

    focal_gamma = config["loss"].get("focal_gamma", 0.0) if config["loss"]["type"] == "focal" else 0.0

    return MaskedMultiTaskLoss(
        class_weights=class_weights,
        task_weights=tw,
        ignore_index=-1,
        focal_gamma=focal_gamma,
    )
