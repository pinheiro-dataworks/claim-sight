"""Tests for the per-image predictions artifact and the CNN export path."""
import numpy as np
import pytest
import torch

from src.model import build_model
from src.predictions import export_dl_predictions, load_predictions, save_predictions


def test_save_load_round_trip(tmp_path):
    images = ["validation/00-damage/a.jpg", "validation/01-whole/b.jpg"]
    y_true = np.array([1, 0])
    p_damage = np.array([0.912345678, 0.000000012])
    path = save_predictions("toy", images, y_true, p_damage, out_dir=tmp_path)
    assert path.name == "validation_toy.csv"
    images_back, y_back, p_back = load_predictions(path)
    assert images_back == images
    np.testing.assert_array_equal(y_back, y_true)
    np.testing.assert_allclose(p_back, p_damage, atol=1e-8)


def test_save_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        save_predictions("toy", ["a"], np.array([1, 0]), np.array([0.5]), out_dir=tmp_path)


def test_export_dl_predictions_covers_validation_split(synthetic_data_root, tmp_path):
    torch.manual_seed(0)
    checkpoint = tmp_path / "untrained.pt"
    torch.save(build_model("efficientnet_b0", pretrained=False).state_dict(), checkpoint)

    path = export_dl_predictions(
        "efficientnet_b0", data_root=synthetic_data_root, checkpoint=checkpoint, out_dir=tmp_path,
    )
    images, y_true, p_damage = load_predictions(path)
    n_expected = sum(1 for _ in (synthetic_data_root / "validation").glob("*/*.jpg"))
    assert len(images) == n_expected
    assert set(y_true) == {0, 1}
    assert ((p_damage >= 0) & (p_damage <= 1)).all()
