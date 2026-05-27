"""Two-stage cascade models with Optuna-bounded hyperparameter search.

Stage A : LightGBM binary (any risk vs no risk).
Stage B : CatBoost MultiClass + LightGBM regress-then-round, ensembled.

All HPO is bounded by both n_trials and timeout so a single run never
exceeds the available time budget.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from .rounder import OptimizedRounder, soft_probabilities

logger = logging.getLogger(__name__)

Fold = Tuple[np.ndarray, np.ndarray]
SEED = 42


def _progress_cb(tag: str):
    """Optuna callback that prints each trial score + elapsed."""
    t0 = time.time()

    def cb(study, trial):
        v = trial.value if trial.value is not None else float("nan")
        print(
            f"  [{tag}] trial {trial.number + 1:2d}  score={v:.4f}  "
            f"best={study.best_value:.4f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    return cb


# ── Stage A: binary risk-vs-no-risk (LightGBM) ──────────────────────────────

def train_stage_a(
    X,
    y: np.ndarray,
    folds: List[Fold],
    n_trials: int = 25,
    timeout: int = 2700,
    feature_name: Optional[Sequence[str]] = None,
    categorical_feature: Optional[Sequence[str]] = None,
    device: str = "cpu",
    seed: int = SEED,
) -> dict:
    """Tune and fit Stage A binary classifier with spatial OOF.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Binary target any_risk (1 = at least one depth at risk).
    folds : list of (train_idx, valid_idx)
    n_trials : int
        Optuna trial cap.
    timeout : int
        Optuna wall-clock cap (seconds).
    feature_name, categorical_feature : optional
        Column names and categorical subset for LightGBM.
    device : str
        'cpu' or 'gpu'.
    seed : int

    Returns
    -------
    dict
        best_params, models (per-fold), oof (OOF probability), cv_auc.
    """
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.asarray(y)
    cat = list(categorical_feature) if categorical_feature else "auto"

    def _dataset(idx):
        return lgb.Dataset(
            X.iloc[idx],
            label=y[idx],
            feature_name=list(feature_name) if feature_name else "auto",
            categorical_feature=cat,
            free_raw_data=False,
        )

    tr_idx, va_idx = folds[0]
    dtr, dva = _dataset(tr_idx), _dataset(va_idx)

    def objective(trial) -> float:
        params = {
            "objective": "binary", "metric": "auc", "is_unbalance": True,
            "verbosity": -1, "seed": seed, "num_threads": 0, "device_type": device,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        booster = lgb.train(
            params, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        p = booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration)
        return roc_auc_score(y[va_idx], p)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    print("Stage A: HPO start", flush=True)
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout,
        show_progress_bar=False, callbacks=[_progress_cb("Stage A")],
    )
    best = {
        **study.best_params,
        "objective": "binary", "metric": "auc", "is_unbalance": True,
        "verbosity": -1, "seed": seed, "bagging_freq": 5,
        "num_threads": 0, "device_type": device,
    }
    print(f"Stage A: HPO done  best AUC={study.best_value:.4f}; refitting 5 folds", flush=True)

    models, oof = [], np.full(len(y), np.nan)
    aucs = []
    for i, (tr, va) in enumerate(folds):
        booster = lgb.train(
            best, _dataset(tr), num_boost_round=3000, valid_sets=[_dataset(va)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        aucs.append(roc_auc_score(y[va], oof[va]))
        models.append(booster)
        print(f"  [Stage A] fold {i}  AUC={aucs[-1]:.4f}", flush=True)

    return {
        "best_params": best, "models": models,
        "oof": oof, "cv_auc": float(np.mean(aucs)), "study": study,
    }


# ── Stage B: CatBoost MultiClass ────────────────────────────────────────────

def train_stage_b_catboost(
    X,
    y: np.ndarray,
    folds: List[Fold],
    target_name: str,
    cat_features: Optional[Sequence[int]] = None,
    n_trials: int = 15,
    timeout: int = 1200,
    task_type: str = "GPU",
    devices: str = "0",
    seed: int = SEED,
) -> dict:
    """Tune and fit CatBoost MultiClass on positive pixels.

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
        Ordinal labels 0..4.
    folds : list of (train_idx, valid_idx)
    target_name : str
        Depth name for logging.
    cat_features : optional
        Column indices treated as native categoricals.
    n_trials, timeout : int
        HPO caps.
    task_type : str
        'CPU' or 'GPU'.
    seed : int

    Returns
    -------
    dict
        best_params, models, oof_proba (n, 5), classes, cv_qwk.
    """
    from catboost import CatBoostClassifier, Pool
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.asarray(y)
    cat_features = list(cat_features) if cat_features else None
    gpu_kwargs: dict = {"task_type": task_type}
    if task_type == "GPU":
        gpu_kwargs["devices"] = devices

    def _pool(idx):
        return Pool(X.iloc[idx], label=y[idx], cat_features=cat_features)

    tr_idx, va_idx = folds[0]
    ptr, pva = _pool(tr_idx), _pool(va_idx)

    def objective(trial) -> float:
        params = {
            "loss_function": "MultiClass", "eval_metric": "WKappa",
            "auto_class_weights": "Balanced", "iterations": 2000,
            "od_type": "Iter", "od_wait": 50, "random_seed": seed,
            "verbose": 0, "allow_writing_files": False,
            **gpu_kwargs,
            "depth": trial.suggest_int("depth", 6, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        model = CatBoostClassifier(**params)
        model.fit(ptr, eval_set=pva, use_best_model=True)
        pred = model.predict(X.iloc[va_idx]).ravel().astype(int)
        return cohen_kappa_score(y[va_idx], pred, weights="quadratic")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    print(f"Stage B[{target_name}] CatBoost: HPO start", flush=True)
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout,
        show_progress_bar=False,
        callbacks=[_progress_cb(f"B-cat {target_name}")],
    )
    best = {
        **study.best_params,
        "loss_function": "MultiClass", "eval_metric": "WKappa",
        "auto_class_weights": "Balanced", "iterations": 2000,
        "od_type": "Iter", "od_wait": 50, "random_seed": seed,
        "verbose": 0, "allow_writing_files": False,
        **gpu_kwargs,
    }
    print(
        f"Stage B[{target_name}] CatBoost: HPO done  best QWK={study.best_value:.4f}; "
        "refitting 5 folds", flush=True,
    )

    classes = np.array([0, 1, 2, 3, 4])
    models, oof = [], np.full((len(y), 5), np.nan, dtype=np.float32)
    qwks = []
    for i, (tr, va) in enumerate(folds):
        model = CatBoostClassifier(**best)
        model.fit(_pool(tr), eval_set=_pool(va), use_best_model=True)
        proba = model.predict_proba(X.iloc[va])
        col = {int(c): j for j, c in enumerate(model.classes_)}
        n_va = len(va)
        oof[va] = np.column_stack([
            proba[:, col[c]] if c in col else np.zeros(n_va, dtype=np.float32)
            for c in classes
        ])
        pred = classes[oof[va].argmax(axis=1)]
        qwks.append(cohen_kappa_score(y[va], pred, weights="quadratic"))
        models.append(model)
        print(f"  [B-cat {target_name}] fold {i}  QWK={qwks[-1]:.4f}", flush=True)

    return {
        "best_params": best, "models": models,
        "oof_proba": oof, "classes": classes,
        "cv_qwk": float(np.mean(qwks)), "study": study,
    }


# ── Stage B: LightGBM regress-then-round ────────────────────────────────────

def train_stage_b_lgb_regression(
    X,
    y: np.ndarray,
    folds: List[Fold],
    target_name: str,
    cat_features: Optional[Sequence[str]] = None,
    n_trials: int = 15,
    timeout: int = 1200,
    sigma: float = 0.6,
    device: str = "cpu",
    seed: int = SEED,
) -> dict:
    """Train LightGBM regressor on 0..4, tune QWK boundaries, emit soft probs.

    Provides ensemble diversity against CatBoost. An OptimizedRounder learns
    non-integer cut points on OOF predictions, and outputs are spread to a
    5-class soft distribution for ensembling.

    Returns
    -------
    dict
        best_params, models, oof_cont, oof_proba, rounder, cv_qwk.
    """
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.asarray(y, dtype=float)
    cat = list(cat_features) if cat_features else "auto"

    def _dataset(idx):
        return lgb.Dataset(
            X.iloc[idx], label=y[idx], categorical_feature=cat, free_raw_data=False
        )

    tr_idx, va_idx = folds[0]
    dtr, dva = _dataset(tr_idx), _dataset(va_idx)

    def objective(trial) -> float:
        params = {
            "objective": "regression", "metric": "l2", "verbosity": -1,
            "seed": seed, "num_threads": 0, "bagging_freq": 5, "device_type": device,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        booster = lgb.train(
            params, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        p = booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration)
        yp = np.clip(np.round(p), 0, 4).astype(int)
        return cohen_kappa_score(y[va_idx].astype(int), yp, weights="quadratic")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    print(f"Stage B[{target_name}] LGB-reg: HPO start", flush=True)
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout,
        show_progress_bar=False,
        callbacks=[_progress_cb(f"B-lgb {target_name}")],
    )
    best = {
        **study.best_params,
        "objective": "regression", "metric": "l2",
        "verbosity": -1, "seed": seed, "num_threads": 0,
        "bagging_freq": 5, "device_type": device,
    }
    print(
        f"Stage B[{target_name}] LGB-reg: HPO done  best QWK={study.best_value:.4f}; "
        "refitting 5 folds", flush=True,
    )

    models, oof_cont = [], np.full(len(y), np.nan)
    for i, (tr, va) in enumerate(folds):
        booster = lgb.train(
            best, _dataset(tr), num_boost_round=3000, valid_sets=[_dataset(va)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_cont[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)
        models.append(booster)
        print(f"  [B-lgb {target_name}] fold {i} done", flush=True)

    rounder = OptimizedRounder(n_classes=5, labels=[0, 1, 2, 3, 4]).fit(
        oof_cont, y.astype(int)
    )
    oof_pred = rounder.predict(oof_cont)
    cv_qwk = float(cohen_kappa_score(y.astype(int), oof_pred, weights="quadratic"))
    oof_proba = soft_probabilities(oof_cont, labels=(0, 1, 2, 3, 4), sigma=sigma)

    return {
        "best_params": best, "models": models,
        "oof_cont": oof_cont, "oof_proba": oof_proba,
        "rounder": rounder, "cv_qwk": cv_qwk, "sigma": sigma, "study": study,
    }


# ── Cascade combination ──────────────────────────────────────────────────────

def combine_cascade(
    p_a: np.ndarray,
    probs_b: np.ndarray,
    tau_a: float,
) -> np.ndarray:
    """Combine Stage A probability and Stage B class probs into 0..4 labels.

    Parameters
    ----------
    p_a : np.ndarray, shape (n,)
        Stage A probability of "at risk".
    probs_b : np.ndarray, shape (n, 5)
        Stage B class probabilities over the 0..4 scale.
    tau_a : float
        Decision threshold. p_a < tau_a → class 0; otherwise Stage B argmax.

    Returns
    -------
    np.ndarray, shape (n,)
        Combined labels in {0, 1, 2, 3, 4}.
    """
    p_a = np.asarray(p_a).ravel()
    probs_b = np.asarray(probs_b)
    severity = probs_b.argmax(axis=1)
    return np.where(p_a < tau_a, 0, severity).astype(int)


def tune_tau_a(
    p_a_oof: np.ndarray,
    probs_b_oof: Dict[str, np.ndarray],
    y_true_5class: Dict[str, np.ndarray],
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Sweep Stage A threshold to maximise mean QWK across depth targets.

    Parameters
    ----------
    p_a_oof : np.ndarray
        Stage A OOF probabilities aligned to candidate pixels.
    probs_b_oof : dict[str, np.ndarray]
        Per-target Stage B OOF class probabilities, shape (n, 5).
    y_true_5class : dict[str, np.ndarray]
        Per-target true labels in {0..4}.
    grid : np.ndarray, optional
        Threshold candidates. Defaults to arange(0.05, 0.95, 0.01).

    Returns
    -------
    (best_tau, best_mean_qwk) : tuple of float
    """
    if grid is None:
        grid = np.arange(0.05, 0.95 + 1e-9, 0.01)
    targets = list(y_true_5class.keys())
    best_tau, best_score = float(grid[0]), -np.inf
    for tau in grid:
        scores = []
        for t in targets:
            pred = combine_cascade(p_a_oof, probs_b_oof[t], tau)
            scores.append(cohen_kappa_score(
                y_true_5class[t], pred, weights="quadratic"
            ))
        mean_q = float(np.mean(scores))
        if mean_q > best_score:
            best_score, best_tau = mean_q, float(tau)
    logger.info("tune_tau_a: best tau=%.3f mean QWK=%.4f", best_tau, best_score)
    return best_tau, best_score


def ensemble_stage_b(
    cat_proba: np.ndarray,
    lgb_proba: np.ndarray,
    y_true: np.ndarray,
    weights: Sequence[float] = (0.3, 0.5, 0.7),
) -> Tuple[float, np.ndarray]:
    """Pick the CatBoost/LGB blend weight that maximises OOF QWK.

    Parameters
    ----------
    cat_proba, lgb_proba : np.ndarray, shape (n, 5)
    y_true : np.ndarray
        True 0..4 labels for the positive pixels.
    weights : sequence of float
        Candidate CatBoost weights w; LGB gets 1-w.

    Returns
    -------
    (best_w, best_proba) : tuple
    """
    classes = np.array([0, 1, 2, 3, 4])
    best_w, best_q, best_p = float(weights[0]), -np.inf, None
    for w in weights:
        blend = w * cat_proba + (1.0 - w) * lgb_proba
        pred = classes[blend.argmax(axis=1)]
        q = cohen_kappa_score(y_true, pred, weights="quadratic")
        if q > best_q:
            best_q, best_w, best_p = q, w, blend
    logger.info("ensemble_stage_b: best CatBoost weight=%.2f QWK=%.4f", best_w, best_q)
    return float(best_w), best_p


__all__ = [
    "train_stage_a",
    "train_stage_b_catboost",
    "train_stage_b_lgb_regression",
    "combine_cascade",
    "tune_tau_a",
    "ensemble_stage_b",
]
