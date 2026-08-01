"""Augmentation policy and PyTorch Dataset for the ClaimSight classifier.

Golden rule: augmentation exists only on the training split. Validation
and the live API see exactly the same deterministic pipeline
(src.preprocessing.preprocess), with nothing beyond ImageNet
normalization applied on top.

Augmentation choices are deliberately conservative and each one maps to
a real source of variability in claim photos taken by policyholders on
phones — see the inline comments. Nothing aggressive (no heavy cutout,
no elastic distortion) that could erase the very damage signal the
model needs to learn, and no vertical flips (an upside-down car is not
a realistic photo).
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess

# damage = positive class (index 1) — recall on this class is the
# business-critical metric (see src/evaluate.py).
CLASS_MAP = {"00-damage": 1, "01-whole": 0}
CLASS_NAMES = ["01-whole", "00-damage"]  # index -> name

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),                                    # camera angle varies
    A.Rotate(limit=15, p=0.5),                                  # slight handheld tilt
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # lighting
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.GaussNoise(p=0.2),                                        # low-quality phone camera
    A.CoarseDropout(
        num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3
    ),  # discourages relying on one single region
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

eval_transform = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


class CarDamageDataset(Dataset):
    """Binary car damage dataset.

    Applies the shared, deterministic preprocess() first, then
    split-specific Albumentations transforms (augmentation for training,
    normalization-only for validation).
    """

    def __init__(self, split_dir: str | Path, transform: A.Compose):
        self.samples: list[tuple[Path, int]] = []
        for class_folder, label in CLASS_MAP.items():
            folder = Path(split_dir) / class_folder
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((path, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {path}")
        image_rgb = preprocess(image_bgr)  # <- single source of truth
        tensor = self.transform(image=image_rgb)["image"]
        return tensor, torch.tensor(label, dtype=torch.long)


def get_loaders(
    data_root: str | Path = "data/raw",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders over data/raw/{training,validation}."""
    train_ds = CarDamageDataset(Path(data_root) / "training", train_transform)
    val_ds = CarDamageDataset(Path(data_root) / "validation", eval_transform)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl
