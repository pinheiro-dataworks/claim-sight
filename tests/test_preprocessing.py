"""Contract tests for src.preprocessing.preprocess().

These are the verifiable evidence behind the "zero training/serving skew"
claim: the same function, given the same bytes, always produces the same
tensor-ready array, with a fixed shape and dtype regardless of input size
or aspect ratio.
"""
import numpy as np
import pytest

from src.preprocessing import TARGET_SIZE, letterbox_box, preprocess


def _random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_output_shape_is_fixed_square():
    img = _random_image(480, 640)
    out = preprocess(img)
    assert out.shape == (TARGET_SIZE, TARGET_SIZE, 3)


def test_output_dtype_is_uint8():
    img = _random_image(300, 300)
    out = preprocess(img)
    assert out.dtype == np.uint8


def test_deterministic_same_input_same_output():
    img = _random_image(512, 384, seed=42)
    out_a = preprocess(img)
    out_b = preprocess(img)
    assert np.array_equal(out_a, out_b)


@pytest.mark.parametrize(
    "h, w",
    [(100, 100), (480, 640), (640, 480), (224, 224), (50, 900), (900, 50), (1, 1)],
)
def test_handles_arbitrary_aspect_ratios(h, w):
    img = _random_image(h, w, seed=h * w)
    out = preprocess(img)
    assert out.shape == (TARGET_SIZE, TARGET_SIZE, 3)
    assert out.dtype == np.uint8


def test_does_not_mutate_input():
    img = _random_image(224, 224)
    original = img.copy()
    preprocess(img)
    assert np.array_equal(img, original)


def test_preserves_aspect_ratio_no_stretch():
    """A wide image should pad top/bottom (letterbox), not stretch content
    to fill the square — this is what protects dent geometry."""
    # Solid bright red image, full width and height (no aspect distortion
    # inside the content itself — only the letterbox padding varies).
    h, w = 100, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # BGR: pure red band across full width, full height
    out = preprocess(img)
    # CLAHE's tile interpolation can bleed a faint non-zero value into
    # padding near tile boundaries, so use a strong brightness threshold
    # (not >0) to distinguish real content rows from letterbox padding.
    row_brightness = out[:, :, 0].mean(axis=1)  # red channel, per row
    bright_rows = np.where(row_brightness > 127)[0]
    # Only ~56 of 224 rows are real content (scale = 224/400 * 100 ≈ 56);
    # if the image had been stretched to fill the square, all 224 rows
    # would be bright. Assert padding clearly dominates.
    assert bright_rows.size < TARGET_SIZE * 0.8


@pytest.mark.parametrize("h, w", [(100, 400), (400, 100), (224, 224), (1, 900)])
def test_letterbox_box_matches_padding_layout(h, w):
    top, left, content_h, content_w = letterbox_box(h, w)
    assert max(content_h, content_w) == TARGET_SIZE
    assert 0 <= top and top + content_h <= TARGET_SIZE
    assert 0 <= left and left + content_w <= TARGET_SIZE
    # A white source image must be non-black exactly inside the reported box.
    out = preprocess(np.full((h, w, 3), 255, dtype=np.uint8))
    assert out[top:top + content_h, left:left + content_w].min() > 200
