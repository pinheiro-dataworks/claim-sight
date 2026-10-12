"""Tests for the bootstrap confidence-interval helpers."""
import numpy as np
import pytest

from src.bootstrap import (
    accuracy_at,
    bootstrap_ci,
    bootstrap_indices,
    paired_bootstrap_diff,
    recall_at,
    roc_auc,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y_true = np.repeat([1, 0], 100)
    # Informative but imperfect scores: positives shifted upward.
    p_damage = np.clip(rng.normal(0.35 + 0.3 * y_true, 0.2), 0, 1)
    return y_true, p_damage


def test_bootstrap_indices_shape_and_range():
    idx = bootstrap_indices(50, n_resamples=300, seed=1)
    assert idx.shape == (300, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_bootstrap_indices_deterministic_with_seed():
    np.testing.assert_array_equal(bootstrap_indices(30, 10, seed=7), bootstrap_indices(30, 10, seed=7))
    assert not np.array_equal(bootstrap_indices(30, 10, seed=7), bootstrap_indices(30, 10, seed=8))


@pytest.mark.parametrize("n_samples, n_resamples", [(0, 10), (10, 0)])
def test_bootstrap_indices_rejects_empty(n_samples, n_resamples):
    with pytest.raises(ValueError):
        bootstrap_indices(n_samples, n_resamples)


@pytest.mark.parametrize("metric", [accuracy_at(0.5), recall_at(0.5), recall_at(0.25), roc_auc])
def test_interval_contains_point_estimate(scores, metric):
    y_true, p_damage = scores
    ci = bootstrap_ci(y_true, p_damage, metric, n_resamples=1000, seed=42)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert ci["upper"] - ci["lower"] > 0
    assert ci["n_resamples"] == ci["n_valid"] == 1000


def test_bootstrap_ci_deterministic_with_seed(scores):
    y_true, p_damage = scores
    a = bootstrap_ci(y_true, p_damage, roc_auc, n_resamples=500, seed=3)
    b = bootstrap_ci(y_true, p_damage, roc_auc, n_resamples=500, seed=3)
    c = bootstrap_ci(y_true, p_damage, roc_auc, n_resamples=500, seed=4)
    assert a == b
    assert (a["lower"], a["upper"]) != (c["lower"], c["upper"])


def test_higher_confidence_gives_wider_interval(scores):
    y_true, p_damage = scores
    narrow = bootstrap_ci(y_true, p_damage, accuracy_at(0.5), confidence=0.8, n_resamples=1000)
    wide = bootstrap_ci(y_true, p_damage, accuracy_at(0.5), confidence=0.99, n_resamples=1000)
    assert wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"]


def test_point_estimates_match_direct_computation():
    y_true = np.array([1, 1, 1, 0, 0])
    p_damage = np.array([0.9, 0.3, 0.2, 0.6, 0.1])
    assert accuracy_at(0.5)(y_true, p_damage) == pytest.approx(0.4)
    assert recall_at(0.5)(y_true, p_damage) == pytest.approx(1 / 3)
    assert recall_at(0.25)(y_true, p_damage) == pytest.approx(2 / 3)


def test_undefined_resamples_are_dropped_not_crashed():
    y_true = np.array([1, 0, 0, 0, 0, 0])  # many resamples contain no positive
    p_damage = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.2])
    ci = bootstrap_ci(y_true, p_damage, roc_auc, n_resamples=500, seed=0)
    assert 0 < ci["n_valid"] < 500
    assert np.isnan(recall_at(0.5)(np.zeros(3, dtype=int), np.ones(3)))


def test_metric_undefined_everywhere_raises():
    with pytest.raises(ValueError, match="undefined"):
        bootstrap_ci(np.zeros(5, dtype=int), np.full(5, 0.5), roc_auc, n_resamples=20)


def test_invalid_confidence_raises(scores):
    y_true, p_damage = scores
    with pytest.raises(ValueError):
        bootstrap_ci(y_true, p_damage, roc_auc, confidence=1.5)
    with pytest.raises(ValueError):
        paired_bootstrap_diff(y_true, p_damage, p_damage, roc_auc, confidence=0)


def test_paired_diff_of_identical_models_is_exactly_zero(scores):
    y_true, p_damage = scores
    diff = paired_bootstrap_diff(y_true, p_damage, p_damage, roc_auc, n_resamples=200)
    assert diff["point"] == diff["lower"] == diff["upper"] == 0.0
    assert diff["p_a_not_better"] == 1.0


def test_paired_diff_detects_a_better_model(scores):
    y_true, p_damage = scores
    noisy = np.clip(p_damage + np.random.default_rng(1).normal(0, 0.4, len(p_damage)), 0, 1)
    diff = paired_bootstrap_diff(y_true, p_damage, noisy, roc_auc, n_resamples=1000)
    assert diff["lower"] <= diff["point"] <= diff["upper"]
    assert diff["lower"] > 0
    assert diff["p_a_not_better"] < 0.05
