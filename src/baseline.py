"""Classical (non-deep-learning) baseline for the damage classifier.

Quantifies how much of the CNNs' performance comes from learned deep
features, as opposed to what hand-crafted color, texture and edge
descriptors already capture.

Train/serve parity: features are computed on the output of the same
src.preprocessing.preprocess() the CNNs consume, so both model families
see identical pixels.

Feature vector (1,879 dims):
    - HSV color histogram: 32 bins per channel, L1-normalized per
      channel, letterbox padding masked out (96).
    - HOG on the grayscale image downscaled to 128x128: 16x16-pixel
      cells, 2x2-cell blocks, 9 orientations, L2-Hys (1,764).
    - Edge statistics inside the content region: Canny edge density,
      a 4x4 grid of local Canny densities, and Sobel gradient-magnitude
      mean and standard deviation (19).

Splits: models are fit on data/raw/training and scored on
data/raw/validation, using the exact file lists of CarDamageDataset.
No hyperparameter is selected on the validation split: the logistic
regression's C is chosen by stratified k-fold CV on the training split,
and gradient boosting early-stops on a held-out fraction of training.

Usage:
    python -m src.baseline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import CarDamageDataset, eval_transform
from src.predictions import PREDICTIONS_DIR, relative_image_paths, save_predictions
from src.preprocessing import letterbox_box, preprocess

HSV_BINS = 32
HOG_SIZE = 128
HOG_CELL = 16
HOG_ORIENTATIONS = 9
EDGE_GRID = 4
CANNY_LOW, CANNY_HIGH = 50, 150

_HOG_BLOCKS = HOG_SIZE // HOG_CELL - 1
FEATURE_GROUPS = {
    "hsv_histogram": 3 * HSV_BINS,
    "hog": _HOG_BLOCKS * _HOG_BLOCKS * 4 * HOG_ORIENTATIONS,
    "edge_stats": 1 + EDGE_GRID * EDGE_GRID + 2,
}
FEATURE_DIM = sum(FEATURE_GROUPS.values())

BASELINE_KEYS = ("logreg", "hist_gb")


def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    """Map a raw BGR image to the fixed-length classical feature vector."""
    h, w = image_bgr.shape[:2]
    image_rgb = preprocess(image_bgr)
    box = letterbox_box(h, w)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return np.concatenate([
        _hsv_histogram(image_rgb, box),
        _hog(gray),
        _edge_stats(gray, box),
    ]).astype(np.float32)


def _hsv_histogram(image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, content_h, content_w = box
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    mask[top:top + content_h, left:left + content_w] = 255
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hists = []
    for channel, upper in ((0, 180), (1, 256), (2, 256)):  # OpenCV 8-bit hue range is [0, 180)
        hist = cv2.calcHist([hsv], [channel], mask, [HSV_BINS], [0, upper]).ravel()
        hists.append(hist / max(float(hist.sum()), 1.0))
    return np.concatenate(hists)


def _hog(gray: np.ndarray) -> np.ndarray:
    # HOG runs on the full letterboxed frame (what the CNN sees); the
    # constant padding has zero gradient and contributes empty cells.
    small = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return hog(
        small,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(HOG_CELL, HOG_CELL),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def _edge_stats(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, content_h, content_w = box
    content = gray[top:top + content_h, left:left + content_w]
    edges = cv2.Canny(content, CANNY_LOW, CANNY_HIGH) > 0
    # Local densities: damage is a localized texture anomaly, so a global
    # density alone would dilute it against the rest of the bodywork.
    cells = [
        float(cell.mean()) if cell.size else 0.0
        for band in np.array_split(edges, EDGE_GRID, axis=0)
        for cell in np.array_split(band, EDGE_GRID, axis=1)
    ]
    gx = cv2.Sobel(content, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(content, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return np.array([edges.mean(), *cells, magnitude.mean(), magnitude.std()], dtype=np.float32)


def _features_from_path(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return extract_features(image_bgr)


def load_split_features(
    data_root: str | Path, split: str, n_jobs: int = -1
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features for a split, in CarDamageDataset's file order.

    Returns (X, y, relative_image_paths).
    """
    data_root = Path(data_root)
    samples = CarDamageDataset(data_root / split, eval_transform).samples
    features = Parallel(n_jobs=n_jobs)(delayed(_features_from_path)(path) for path, _ in samples)
    X = np.stack(features)
    y = np.array([label for _, label in samples], dtype=int)
    return X, y, relative_image_paths(samples, data_root)


def build_baselines(seed: int = 42, cv_folds: int = 5) -> dict[str, Pipeline | HistGradientBoostingClassifier]:
    """The two classical models, with every source of randomness seeded."""
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-4, 1, 11),
            l1_ratios=(0.0,),  # pure L2: thousands of correlated HOG bins, no sparsity prior
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
            scoring="roc_auc",
            max_iter=5000,
            use_legacy_attributes=False,
        ),
    )
    hist_gb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    return {"logreg": logreg, "hist_gb": hist_gb}


def _fitted_hyperparameters(key: str, model) -> dict:
    if key == "logreg":
        return {"C": float(np.ravel(model[-1].C_)[0])}
    return {"n_iter": int(model.n_iter_)}


def run_baselines(
    data_root: str | Path = "data/raw",
    out_dir: Path = PREDICTIONS_DIR,
    seed: int = 42,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> dict:
    """Extract features, fit both baselines on training, save validation predictions."""
    t0 = time.perf_counter()
    X_train, y_train, _ = load_split_features(data_root, "training", n_jobs=n_jobs)
    X_val, y_val, val_images = load_split_features(data_root, "validation", n_jobs=n_jobs)
    extraction_s = time.perf_counter() - t0

    summary = {
        "seed": seed,
        "feature_dim": FEATURE_DIM,
        "feature_groups": FEATURE_GROUPS,
        "n_train": len(y_train),
        "n_validation": len(y_val),
        "feature_extraction_seconds": round(extraction_s, 2),
        "models": {},
    }
    for key, model in build_baselines(seed=seed, cv_folds=cv_folds).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        p_damage = model.predict_proba(X_val)[:, 1]
        path = save_predictions(key, val_images, y_val, p_damage, out_dir)
        summary["models"][key] = {
            "fit_seconds": round(fit_s, 2),
            "hyperparameters": _fitted_hyperparameters(key, model),
            "predictions": path.as_posix(),
        }
        print(f"{key}: fit in {fit_s:.1f}s -> {path}")

    meta_path = Path(out_dir) / "baseline_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the classical (HSV + HOG + Canny) baselines.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_baselines(args.data_root, out_dir=args.out_dir, seed=args.seed)
