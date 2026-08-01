"""Dataset ingestion and structural validation.

Source: Car Damage Detection (Kaggle, anujms/car-damage-detection).
Expected layout after download:

    data/raw/
    ├── training/
    │   ├── 00-damage/
    │   └── 01-whole/
    └── validation/
        ├── 00-damage/
        └── 01-whole/

Only binary folder labels are provided — no masks, no bounding boxes.
That constraint is what shapes the rest of the project's architecture
(binary classifier + Grad-CAM weak localization, not segmentation).

Download is idempotent: if data/raw already has valid structure with
images in every expected folder, the Kaggle API call is skipped
entirely. This lets the same script run safely in CI or on a fresh
clone without re-downloading ~130MB every time.
"""
from __future__ import annotations

from pathlib import Path

DATASET_SLUG = "anujms/car-damage-detection"
RAW_DIR = Path("data/raw")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
EXPECTED_DIRS = [
    "training/00-damage",
    "training/01-whole",
    "validation/00-damage",
    "validation/01-whole",
]


def validate_structure(root: Path = RAW_DIR) -> bool:
    """Check that every expected class folder exists and holds images."""
    for rel in EXPECTED_DIRS:
        folder = root / rel
        if not folder.is_dir():
            return False
        has_image = any(
            p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir() if p.is_file()
        )
        if not has_image:
            return False
    return True


def download_dataset(force: bool = False, root: Path = RAW_DIR) -> Path:
    """Download and extract the Kaggle dataset if not already present.

    Requires KAGGLE_USERNAME/KAGGLE_KEY env vars or ~/.kaggle/kaggle.json
    (see https://www.kaggle.com/docs/api). Returns the directory
    containing the extracted `training/` and `validation/` splits.
    """
    if not force and validate_structure(root):
        print(f"Dataset already present and valid at {root} — skipping download.")
        return root

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    root.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET_SLUG, path=str(root), unzip=True)

    if not validate_structure(root):
        raise RuntimeError(
            f"Downloaded dataset at {root} failed structural validation — "
            f"expected folders {EXPECTED_DIRS}."
        )
    return root


def dataset_summary(root: Path = RAW_DIR) -> dict[str, int]:
    """Return real image counts per split/class — feeds the README and EDA.

    Never hand-typed: this is the only source of truth for dataset size
    numbers quoted anywhere in this project (README, landing page, model card).
    """
    summary: dict[str, int] = {}
    for rel in EXPECTED_DIRS:
        folder = root / rel
        count = sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        summary[rel] = count
    return summary


if __name__ == "__main__":
    download_dataset()
    summary = dataset_summary()
    for split_class, count in summary.items():
        print(f"{split_class}: {count} images")
    total = sum(summary.values())
    print(f"TOTAL: {total} images")
