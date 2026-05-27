"""Evaluation metrics for the flood-risk cascade.

Primary metric: Quadratic Weighted Kappa (QWK).
Secondary: macro-F1, per-class F1.

Class convention
----------------
0 = No Risk, 1 = Very Low, 2 = Low, 3 = Medium, 4 = High.
Unlabeled pixels (-1) must be filtered before calling these functions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa.

    Returns 0.0 if a single class is present (kappa undefined) to keep
    CV aggregation robust.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size < 2:
        return 0.0
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels))


def evaluate_cascade(
    y_true_5class: np.ndarray,
    y_pred_5class: np.ndarray,
) -> Dict[str, float]:
    """Score one depth target on the combined 0..4 space.

    Returns
    -------
    dict
        qwk, macro_f1, f1_class_{k} for k in 0..4.
    """
    y_true = np.asarray(y_true_5class).ravel()
    y_pred = np.asarray(y_pred_5class).ravel()
    out: Dict[str, float] = {
        "qwk": qwk(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(y_true.size),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    present = set(np.unique(y_true).tolist())
    for k, val in zip([0, 1, 2, 3, 4], per_class):
        out[f"f1_class_{k}"] = float(val) if k in present else float("nan")
    return out


def evaluate_per_depth(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Per-depth metric table plus a mean row.

    Returns
    -------
    pd.DataFrame
        One row per target plus a mean row.
    """
    rows = {t: evaluate_cascade(y_true_dict[t], y_pred_dict[t]) for t in y_true_dict}
    df = pd.DataFrame(rows).T
    df.loc["mean"] = df.mean(numeric_only=True)
    return df


__all__ = ["qwk", "evaluate_cascade", "evaluate_per_depth"]
