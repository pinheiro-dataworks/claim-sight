"""Deterministic image preprocessing — the single source of truth.

This module is imported by training (src/dataset.py), evaluation
(src/evaluate.py, src/explain.py) and serving (app/main.py). There is
exactly one code path that turns raw image bytes into a model-ready
tensor, so the transform a model is trained on and the transform it sees
in production can never drift apart (zero training/serving skew).

Pipeline: BGR -> RGB -> aspect-preserving letterbox resize to 224x224 ->
bilateral filter (edge-preserving denoise) -> CLAHE on the L channel of
LAB color space (local contrast normalization).
"""
from __future__ import annotations

import cv2
import numpy as np

TARGET_SIZE = 224

# ImageNet normalization statistics — required because both backbones
# (ResNet50, EfficientNet-B0) were pretrained on ImageNet with these stats.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Deterministic preprocessing shared by training, evaluation and serving.

    Same input -> same output, always. No randomness anywhere in this
    function (augmentation lives in src/dataset.py and is applied only
    to the training split, strictly after this step).

    Args:
        image_bgr: raw image as loaded by cv2.imread / cv2.imdecode,
            shape (H, W, 3), dtype uint8, channel order BGR.

    Returns:
        RGB uint8 array of shape (224, 224, 3).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = _resize_with_padding(image_rgb, TARGET_SIZE)
    image_rgb = cv2.bilateralFilter(image_rgb, d=5, sigmaColor=50, sigmaSpace=5)
    image_rgb = _apply_clahe(image_rgb)
    return image_rgb


def letterbox_box(height: int, width: int, size: int = TARGET_SIZE) -> tuple[int, int, int, int]:
    """Return (top, left, content_h, content_w) of the image inside the letterbox.

    Exposed so downstream consumers (e.g. the classical baseline's
    color and edge statistics) can exclude the constant padding without
    re-deriving the resize arithmetic.
    """
    scale = size / max(height, width)
    content_h, content_w = max(1, round(height * scale)), max(1, round(width * scale))
    return (size - content_h) // 2, (size - content_w) // 2, content_h, content_w


def _resize_with_padding(image: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio, then letterbox-pad to a (size x size) square.

    Preserving aspect ratio matters here specifically because a stretched
    dent no longer looks like a dent — distorting geometry would corrupt
    the exact visual signal the model is trained to detect.
    """
    h, w = image.shape[:2]
    top, left, new_h, new_w = letterbox_box(h, w, size)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    bottom = size - new_h - top
    right = size - new_w - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def _apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB space for local contrast.

    Claim photos are shot on phones under wildly inconsistent lighting;
    CLAHE normalizes that variability without blowing out color channels
    the way a naive global histogram equalization would.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
