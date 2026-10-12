"""Tests for the cross-model comparison report."""
import numpy as np
import pytest

from src.compare_models import compare, confident_subset, error_overlap, write_reports
from src.predictions import save_predictions

IMAGES = [f"validation/x/{i}.jpg" for i in range(8)]
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0])


@pytest.fixture
def predictions_dir(tmp_path):
    save_predictions("resnet50", IMAGES, Y_TRUE, np.array([.95, .9, .8, .3, .2, .1, .05, .6]), tmp_path)
    save_predictions("logreg", IMAGES, Y_TRUE, np.array([.7, .6, .4, .2, .45, .5, .3, .1]), tmp_path)
    save_predictions("hist_gb", IMAGES, Y_TRUE, np.array([.99, .4, .6, .7, .2, .01, .3, .55]), tmp_path)
    return tmp_path


def test_confident_subset_coverage_and_accuracy():
    result = confident_subset(np.array([1, 0, 1, 0]), np.array([0.95, 0.05, 0.6, 0.92]))
    assert result["n"] == 3
    assert result["coverage"] == pytest.approx(0.75)
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_confident_subset_empty():
    assert confident_subset(np.array([1]), np.array([0.5]))["accuracy"] is None


def test_error_overlap_partitions_all_images():
    overlap = error_overlap(Y_TRUE, np.full(8, 0.9), np.where(Y_TRUE == 1, 0.9, 0.1))
    assert sum(overlap.values()) == len(Y_TRUE)
    assert overlap == {"both_correct": 4, "only_a_correct": 0, "only_b_correct": 4, "both_wrong": 0}


def test_compare_scores_every_model_and_picks_best_baseline(predictions_dir, tmp_path):
    result = compare(predictions_dir, metrics_dir=tmp_path / "missing")
    assert list(result["models"]) == ["resnet50", "hist_gb", "logreg"]
    resnet = result["models"]["resnet50"]
    assert resnet["operating_points"]["0.5"]["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert resnet["operating_points"]["0.25"]["recall_damage"] == 1.0
    assert resnet["matches_versioned_metrics"] is None
    best = max(("hist_gb", "logreg"), key=lambda k: result["models"][k]["roc_auc"])
    assert result["error_overlap"]["b"] == best

    json_path, md_path = write_reports(result, tmp_path / "reports")
    assert json_path.exists()
    assert "| ResNet50 | CNN |" in md_path.read_text(encoding="utf-8")


def test_compare_rejects_mismatched_evaluation_sets(predictions_dir):
    save_predictions("efficientnet_b0", IMAGES, 1 - Y_TRUE, np.full(8, 0.5), predictions_dir)
    with pytest.raises(ValueError, match="different images or labels"):
        compare(predictions_dir)


def test_compare_requires_at_least_one_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare(tmp_path)
