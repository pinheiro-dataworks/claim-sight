"""Per-image validation predictions, persisted as CSV.

Every model compared under reports/ (both CNNs and the classical
baselines) is reduced to the same artifact: one row per validation
image with its true label and P(damage). Model comparison and bootstrap
confidence intervals are computed from these files alone, so they are
reproducible without checkpoints or the raw dataset, and every model is
scored by exactly the same code.

Usage:
    python -m src.predictions --arch all
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import CarDamageDataset, eval_transform
from src.evaluate import get_predictions
from src.model import SUPPORTED_ARCHS, build_model

PREDICTIONS_DIR = Path("reports/predictions")
COLUMNS = ("image", "y_true", "p_damage")


def predictions_path(model_key: str, out_dir: Path = PREDICTIONS_DIR) -> Path:
    return Path(out_dir) / f"validation_{model_key}.csv"


def save_predictions(
    model_key: str,
    images: list[str],
    y_true: np.ndarray,
    p_damage: np.ndarray,
    out_dir: Path = PREDICTIONS_DIR,
) -> Path:
    """Write one CSV row per image: relative path, true label, P(damage)."""
    if not len(images) == len(y_true) == len(p_damage):
        raise ValueError("images, y_true and p_damage must have the same length")
    out_path = predictions_path(model_key, out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for image, label, prob in zip(images, y_true, p_damage):
            writer.writerow((image, int(label), f"{float(prob):.8f}"))
    return out_path


def load_predictions(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read a predictions CSV back into (images, y_true, p_damage)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    images = [r["image"] for r in rows]
    y_true = np.array([int(r["y_true"]) for r in rows], dtype=int)
    p_damage = np.array([float(r["p_damage"]) for r in rows], dtype=float)
    return images, y_true, p_damage


def relative_image_paths(samples: list[tuple[Path, int]], data_root: Path) -> list[str]:
    """Dataset-relative POSIX paths, so CSVs are identical across machines and OSes."""
    return [Path(path).relative_to(data_root).as_posix() for path, _ in samples]


def export_dl_predictions(
    arch: str,
    data_root: str | Path = "data/raw",
    checkpoint: str | Path | None = None,
    out_dir: Path = PREDICTIONS_DIR,
    batch_size: int = 32,
) -> Path:
    """Run a trained checkpoint over the validation split and save its predictions."""
    data_root = Path(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint or f"models/best_{arch}.pt")

    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_ds = CarDamageDataset(data_root / "validation", eval_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, p_damage = get_predictions(model, val_dl, device)

    images = relative_image_paths(val_ds.samples, data_root)
    return save_predictions(arch, images, y_true, p_damage, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export per-image validation predictions of trained CNNs.")
    parser.add_argument("--arch", choices=[*SUPPORTED_ARCHS, "all"], default="all")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out-dir", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()

    for name in SUPPORTED_ARCHS if args.arch == "all" else (args.arch,):
        print(f"Saved: {export_dl_predictions(name, data_root=args.data_root, out_dir=args.out_dir)}")
