"""Threshold optimisation for regress-then-round ordinal prediction.

A regressor predicts a continuous score; rounding at the naive half-integer
boundaries (1.5, 2.5, 3.5) is rarely QWK-optimal under class imbalance.
OptimizedRounder searches for boundaries that maximise QWK on out-of-fold
predictions via Nelder-Mead.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


class OptimizedRounder:
    """Find ordinal cut points that maximise Quadratic Weighted Kappa.

    Parameters
    ----------
    n_classes : int
        Number of ordinal classes. Default 5 (the 0..4 combined scale).
    labels : sequence of int, optional
        Class labels assigned between cut points. Defaults to [0, 1, 2, 3, 4].
    """

    def __init__(
        self,
        n_classes: int = 5,
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        self.n_classes = n_classes
        self.labels = list(labels) if labels is not None else list(range(n_classes))
        if len(self.labels) != n_classes:
            raise ValueError("len(labels) must equal n_classes")
        self.coef_: np.ndarray = np.array(
            [(self.labels[i] + self.labels[i + 1]) / 2.0 for i in range(n_classes - 1)],
            dtype=np.float64,
        )

    def _digitize(self, x: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
        idx = np.digitize(np.asarray(x, dtype=np.float64), np.sort(boundaries))
        lab = np.asarray(self.labels)
        idx = np.clip(idx, 0, len(lab) - 1)
        return lab[idx]

    def _loss(self, boundaries: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = self._digitize(x, boundaries)
        return -cohen_kappa_score(y, y_hat, weights="quadratic")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OptimizedRounder":
        """Optimise boundaries against y using Nelder-Mead."""
        res = minimize(
            self._loss,
            self.coef_,
            args=(np.asarray(x, dtype=np.float64), np.asarray(y)),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
        )
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply fitted boundaries to map scores to ordinal labels."""
        return self._digitize(x, self.coef_)

    @property
    def coefficients(self) -> np.ndarray:
        return self.coef_


def soft_probabilities(
    values: np.ndarray,
    labels: Sequence[int] = (0, 1, 2, 3, 4),
    sigma: float = 0.6,
) -> np.ndarray:
    """Convert continuous regression outputs to a soft class distribution.

    Places each prediction on the same probability scale as a classifier so
    LightGBM-regression outputs can be averaged with CatBoost class
    probabilities in the Stage B ensemble.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Continuous predictions.
    labels : sequence of int
        Ordinal class labels.
    sigma : float
        Gaussian bandwidth. Smaller is sharper.

    Returns
    -------
    np.ndarray, shape (n, n_classes)
        Row-normalised soft probabilities, P(k) ∝ exp(-(v-k)²/(2σ²)).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    lab = np.asarray(labels, dtype=np.float64).reshape(1, -1)
    logits = -((v - lab) ** 2) / (2.0 * sigma ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


__all__ = ["OptimizedRounder", "soft_probabilities"]
