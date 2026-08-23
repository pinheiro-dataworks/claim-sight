"""Unit tests for src/dataset.py's CarDamageDataset — built from the
same two real images bundled in samples/ for the Streamlit demo, laid
out in a temporary directory matching the expected class-folder shape
(00-damage/, 01-whole/), so no downloaded dataset is required.
"""
import shutil
from pathlib import Path

import pytest
import torch

from src.dataset import CarDamageDataset, eval_transform

SAMPLES_DIR = Path("samples")


@pytest.fixture
def split_dir(tmp_path):
    """A tmp_path/00-damage/, tmp_path/01-whole/ layout with one real image each."""
    (tmp_path / "00-damage").mkdir()
    (tmp_path / "01-whole").mkdir()
    shutil.copy(SAMPLES_DIR / "damage_example.jpg", tmp_path / "00-damage" / "damage_example.jpg")
    shutil.copy(SAMPLES_DIR / "whole_example.jpg", tmp_path / "01-whole" / "whole_example.jpg")
    return tmp_path


def test_dataset_length_matches_files_on_disk(split_dir):
    ds = CarDamageDataset(split_dir, eval_transform)
    assert len(ds) == 2


def test_dataset_assigns_damage_as_positive_class(split_dir):
    ds = CarDamageDataset(split_dir, eval_transform)
    labels = {path.parent.name: label for path, label in ds.samples}
    assert labels["00-damage"] == 1
    assert labels["01-whole"] == 0


def test_getitem_returns_normalized_tensor_and_label(split_dir):
    ds = CarDamageDataset(split_dir, eval_transform)
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert label.dtype == torch.long
    assert label.item() in (0, 1)


def test_getitem_raises_on_unreadable_file(tmp_path):
    (tmp_path / "00-damage").mkdir()
    (tmp_path / "01-whole").mkdir()
    bogus = tmp_path / "00-damage" / "not_an_image.jpg"
    bogus.write_bytes(b"this is not image data")

    ds = CarDamageDataset(tmp_path, eval_transform)
    with pytest.raises(ValueError, match="Could not read image"):
        ds[0]
