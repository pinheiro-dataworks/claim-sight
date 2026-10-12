"""Tests for the classical baseline: feature contract and an end-to-end smoke run."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.baseline import (
    FEATURE_DIM,
    FEATURE_GROUPS,
    build_baselines,
    extract_features,
    load_split_features,
    run_baselines,
)
from src.predictions import load_predictions


def _random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("h, w", [(480, 640), (640, 480), (224, 224), (50, 900), (1, 1)])
def test_feature_vector_has_fixed_shape_for_any_input_size(h, w):
    features = extract_features(_random_image(h, w))
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_feature_dim_matches_documented_groups():
    assert FEATURE_DIM == sum(FEATURE_GROUPS.values()) == 1879


def test_features_are_deterministic():
    image = _random_image(300, 400, seed=7)
    assert np.array_equal(extract_features(image), extract_features(image))


def test_hsv_histograms_ignore_letterbox_padding():
    """A uniform image must put all its mass in one bin per channel, whatever its aspect ratio."""
    n_hsv = FEATURE_GROUPS["hsv_histogram"]
    wide = np.full((50, 400, 3), (40, 160, 90), dtype=np.uint8)
    hsv = extract_features(wide)[:n_hsv].reshape(3, -1)
    np.testing.assert_allclose(hsv.sum(axis=1), 1.0, rtol=1e-5)
    assert (hsv.max(axis=1) > 0.9).all()  # no mass leaked to the black padding bins


def test_edge_features_respond_to_scratches():
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    scratched = flat.copy()
    scratched[::20, :] = 0
    edge_start = FEATURE_GROUPS["hsv_histogram"] + FEATURE_GROUPS["hog"]
    assert extract_features(scratched)[edge_start] > extract_features(flat)[edge_start]


def test_load_split_features_keeps_dataset_order_and_labels(synthetic_data_root):
    X, y, images = load_split_features(synthetic_data_root, "validation", n_jobs=1)
    assert X.shape == (len(images), FEATURE_DIM)
    assert images == sorted(images)  # 00-damage first, then 01-whole, each sorted
    assert all(img.startswith("validation/") for img in images)
    assert set(y[[i for i, img in enumerate(images) if "00-damage" in img]]) == {1}


def test_build_baselines_is_seeded():
    models = build_baselines(seed=123, cv_folds=3)
    assert set(models) == {"logreg", "hist_gb"}
    assert models["hist_gb"].random_state == 123
    assert models["logreg"][-1].cv.random_state == 123


def test_run_baselines_end_to_end(synthetic_data_root, tmp_path):
    summary = run_baselines(synthetic_data_root, out_dir=tmp_path, cv_folds=3, n_jobs=1)
    assert summary["feature_dim"] == FEATURE_DIM
    for key in ("logreg", "hist_gb"):
        images, y_true, p_damage = load_predictions(tmp_path / f"validation_{key}.csv")
        assert len(images) == len(y_true) == summary["n_validation"]
        assert ((p_damage >= 0) & (p_damage <= 1)).all()
    # Synthetic scratches are a strong edge signal. Only the linear model is
    # checked: 24 training images are below HistGradientBoosting's default
    # min_samples_leaf, so it legitimately predicts a constant here.
    _, y_true, p_damage = load_predictions(tmp_path / "validation_logreg.csv")
    assert roc_auc_score(y_true, p_damage) >= 0.8
    meta = json.loads((tmp_path / "baseline_meta.json").read_text(encoding="utf-8"))
    assert set(meta["models"]) == {"logreg", "hist_gb"}
