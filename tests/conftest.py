"""Shared fixtures: a tiny synthetic dataset laid out like data/raw/.

"Damage" images are smooth panels with random dark scratch lines; "whole"
images are the same panels without them. That gives the classical
features a real, learnable signal while keeping the fixture fast and
free of any downloaded data.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

N_PER_CLASS = 12


def _synthetic_image(rng: np.random.Generator, damaged: bool) -> np.ndarray:
    h, w = int(rng.integers(120, 200)), int(rng.integers(160, 260))
    base = rng.integers(60, 200, size=3)
    gradient = np.linspace(0.8, 1.2, w)[None, :, None]
    image = np.clip(np.ones((h, w, 3)) * base * gradient, 0, 255).astype(np.uint8)
    if damaged:
        for _ in range(int(rng.integers(6, 12))):
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(image, p1, p2, (20, 20, 20), thickness=2)
    return image


@pytest.fixture(scope="session")
def synthetic_data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    for split in ("training", "validation"):
        for class_folder, damaged in (("00-damage", True), ("01-whole", False)):
            folder = root / split / class_folder
            folder.mkdir(parents=True)
            for i in range(N_PER_CLASS):
                cv2.imwrite(str(folder / f"{i:04d}.jpg"), _synthetic_image(rng, damaged))
    return root
