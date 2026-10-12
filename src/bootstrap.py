"""Non-parametric bootstrap confidence intervals for evaluation metrics.

The validation split has 460 images, so a single accuracy or recall
number carries several points of sampling uncertainty. These helpers
resample the evaluation set with replacement and report percentile
intervals, plus paired intervals for the difference between two models
scored on the same images (the comparison that actually matters when
choosing a model).

Resampling indices are drawn once from a seeded generator and shared
across metrics and models, so results are deterministic and paired
comparisons use identical resamples.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

Metric = Callable[[np.ndarray, np.ndarray], float]

DAMAGE_LABEL = 1


def accuracy_at(threshold: float) -> Metric:
    def metric(y_true: np.ndarray, p_damage: np.ndarray) -> float:
        return float(((p_damage >= threshold).astype(int) == y_true).mean())
    return metric


def recall_at(threshold: float) -> Metric:
    def metric(y_true: np.ndarray, p_damage: np.ndarray) -> float:
        positives = y_true == DAMAGE_LABEL
        if not positives.any():
            return float("nan")
        return float((p_damage[positives] >= threshold).mean())
    return metric


def roc_auc(y_true: np.ndarray, p_damage: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_damage))


def bootstrap_indices(n_samples: int, n_resamples: int = 2000, seed: int = 42) -> np.ndarray:
    """Return an (n_resamples, n_samples) array of row indices drawn with replacement."""
    if n_samples < 1 or n_resamples < 1:
        raise ValueError("n_samples and n_resamples must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_samples, size=(n_resamples, n_samples))


def _interval(point: float, values: np.ndarray, confidence: float, n_resamples: int) -> dict:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        raise ValueError("metric is undefined on every bootstrap resample")
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(valid, [alpha, 1 - alpha])
    return {
        "point": point,
        "lower": float(lower),
        "upper": float(upper),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        "confidence": confidence,
        "n_resamples": n_resamples,
        # Resamples where the metric is undefined (e.g. no positives) are dropped.
        "n_valid": int(valid.size),
    }


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")


def bootstrap_ci(
    y_true: np.ndarray,
    p_damage: np.ndarray,
    metric: Metric,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    indices: np.ndarray | None = None,
) -> dict:
    """Percentile bootstrap interval for metric(y_true, p_damage)."""
    _check_confidence(confidence)
    y_true, p_damage = np.asarray(y_true), np.asarray(p_damage)
    if indices is None:
        indices = bootstrap_indices(len(y_true), n_resamples, seed)
    values = np.array([metric(y_true[idx], p_damage[idx]) for idx in indices])
    return _interval(metric(y_true, p_damage), values, confidence, len(indices))


def paired_bootstrap_diff(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: Metric,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    indices: np.ndarray | None = None,
) -> dict:
    """Interval for metric(model A) - metric(model B) on the same resampled images.

    Pairing removes the shared image-difficulty variance, so the
    difference interval is much tighter than comparing two independent
    intervals. ``p_a_not_better`` is the fraction of resamples where A
    does not beat B.
    """
    _check_confidence(confidence)
    y_true, p_a, p_b = np.asarray(y_true), np.asarray(p_a), np.asarray(p_b)
    if indices is None:
        indices = bootstrap_indices(len(y_true), n_resamples, seed)
    diffs = np.array([metric(y_true[idx], p_a[idx]) - metric(y_true[idx], p_b[idx]) for idx in indices])
    result = _interval(metric(y_true, p_a) - metric(y_true, p_b), diffs, confidence, len(indices))
    result["p_a_not_better"] = float((diffs[~np.isnan(diffs)] <= 0).mean())
    return result
