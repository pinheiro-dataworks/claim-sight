"""Tests for the confidence-interval report over saved predictions."""
import numpy as np

from src.confidence_intervals import compute_intervals, write_reports
from src.predictions import save_predictions


def _scores():
    rng = np.random.default_rng(0)
    y_true = np.repeat([1, 0], 100)
    return y_true, np.clip(rng.normal(0.35 + 0.3 * y_true, 0.2), 0, 1)


def test_compute_intervals_report(tmp_path):
    y_true, p_damage = _scores()
    images = [f"validation/x/{i}.jpg" for i in range(len(y_true))]
    save_predictions("resnet50", images, y_true, p_damage, tmp_path)
    save_predictions("logreg", images, y_true, np.clip(p_damage * 0.8 + 0.1, 0, 1), tmp_path)

    result = compute_intervals(tmp_path, n_resamples=200, seed=5)
    assert set(result["models"]) == {"resnet50", "logreg"}
    assert set(result["models"]["resnet50"]) == {"accuracy@0.5", "roc_auc", "recall@0.5", "recall@0.25"}
    assert set(result["paired_differences"]) == {"resnet50-logreg"}
    assert result == compute_intervals(tmp_path, n_resamples=200, seed=5)

    json_path, md_path = write_reports(result, tmp_path / "reports")
    text = md_path.read_text(encoding="utf-8")
    assert json_path.exists()
    assert "| ResNet50 |" in text and "Paired differences" in text
