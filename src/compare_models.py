"""Side-by-side comparison of every model with saved validation predictions.

Reads reports/predictions/validation_<model>.csv for the CNNs and the
classical baselines, scores all of them with the same
src.evaluate.compute_metrics(), and writes reports/model_comparison.json
plus a Markdown rendering. Refuses to compare files that do not cover
exactly the same images with the same labels.

Usage:
    python -m src.compare_models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluate import compute_metrics
from src.predictions import PREDICTIONS_DIR, load_predictions, predictions_path

REPORTS_DIR = Path("reports")
VERSIONED_METRICS_DIR = Path("outputs/metrics")

MODEL_ORDER = ("resnet50", "efficientnet_b0", "hist_gb", "logreg")
DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "hist_gb": "HistGradientBoosting (HSV + HOG + Canny)",
    "logreg": "Logistic regression (HSV + HOG + Canny)",
}
DL_KEYS = ("resnet50", "efficientnet_b0")
BASELINE_KEYS = ("hist_gb", "logreg")
THRESHOLDS = (0.5, 0.25)
CONFIDENT_MARGIN = 0.4  # "confident" = P(damage) <= 0.1 or >= 0.9


def operating_point(y_true: np.ndarray, p_damage: np.ndarray, threshold: float) -> dict:
    m = compute_metrics(y_true, p_damage, threshold=threshold)
    damage = m["classification_report"]["00-damage"]
    return {
        "accuracy": m["accuracy"],
        "recall_damage": damage["recall"],
        "precision_damage": damage["precision"],
        "confusion_matrix": m["confusion_matrix"],
        "roc_auc": m["roc_auc"],
    }


def confident_subset(y_true: np.ndarray, p_damage: np.ndarray, margin: float = CONFIDENT_MARGIN) -> dict:
    """Coverage and accuracy on the images the model is confident about.

    Measures where a cheap model could short-circuit the pipeline: if
    its confident predictions are as accurate as the CNN's, only the
    remaining uncertain images need the expensive model.
    """
    mask = np.abs(p_damage - 0.5) >= margin
    correct = (p_damage >= 0.5).astype(int) == y_true
    return {
        "margin": margin,
        "coverage": float(mask.mean()),
        "n": int(mask.sum()),
        "accuracy": float(correct[mask].mean()) if mask.any() else None,
    }


def error_overlap(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, threshold: float = 0.5) -> dict:
    a_ok = (p_a >= threshold).astype(int) == y_true
    b_ok = (p_b >= threshold).astype(int) == y_true
    return {
        "both_correct": int((a_ok & b_ok).sum()),
        "only_a_correct": int((a_ok & ~b_ok).sum()),
        "only_b_correct": int((~a_ok & b_ok).sum()),
        "both_wrong": int((~a_ok & ~b_ok).sum()),
    }


def load_all(predictions_dir: Path = PREDICTIONS_DIR) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """Load every available model's predictions and verify they share one evaluation set."""
    loaded = {
        key: load_predictions(predictions_path(key, predictions_dir))
        for key in MODEL_ORDER
        if predictions_path(key, predictions_dir).exists()
    }
    if not loaded:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    ref_key, (ref_images, ref_y, _) = next(iter(loaded.items()))
    for key, (images, y_true, _) in loaded.items():
        if images != ref_images or not np.array_equal(y_true, ref_y):
            raise ValueError(f"{key} and {ref_key} predictions cover different images or labels")
    return loaded


def _versioned_check(key: str, metrics_at_half: dict, metrics_dir: Path) -> bool | None:
    """True if recomputed CNN metrics match the committed outputs/metrics JSON."""
    path = metrics_dir / f"metrics_{key}.json"
    if key not in DL_KEYS or not path.exists():
        return None
    versioned = json.loads(path.read_text(encoding="utf-8"))["default_threshold"]
    return (
        versioned["confusion_matrix"] == metrics_at_half["confusion_matrix"]
        and abs(versioned["roc_auc"] - metrics_at_half["roc_auc"]) < 1e-6
    )


def compare(
    predictions_dir: Path = PREDICTIONS_DIR,
    metrics_dir: Path = VERSIONED_METRICS_DIR,
) -> dict:
    loaded = load_all(predictions_dir)
    first_images, y_true, _ = next(iter(loaded.values()))
    result = {"n_images": len(first_images), "thresholds": list(THRESHOLDS), "models": {}}

    for key, (_, _, p_damage) in loaded.items():
        points = {f"{t:g}": operating_point(y_true, p_damage, t) for t in THRESHOLDS}
        roc_auc = points["0.5"]["roc_auc"]
        matches = _versioned_check(key, points["0.5"], metrics_dir)
        for point in points.values():
            del point["roc_auc"]  # threshold-independent; reported once per model
        result["models"][key] = {
            "name": DISPLAY_NAMES[key],
            "family": "deep_learning" if key in DL_KEYS else "classical",
            "roc_auc": roc_auc,
            "operating_points": points,
            "confident_subset": confident_subset(y_true, p_damage),
            "matches_versioned_metrics": matches,
        }

    baselines = [k for k in BASELINE_KEYS if k in result["models"]]
    if "resnet50" in loaded and baselines:
        best = max(baselines, key=lambda k: result["models"][k]["roc_auc"])
        result["error_overlap"] = {
            "a": "resnet50",
            "b": best,
            "threshold": 0.5,
            **error_overlap(y_true, loaded["resnet50"][2], loaded[best][2]),
        }
    return result


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.2f}%"


def render_markdown(result: dict) -> str:
    lines = [
        "# Model comparison — validation split",
        "",
        (
            f"Generated by `python -m src.compare_models` from `reports/predictions/` "
            f"({result['n_images']} held-out images). Do not edit by hand."
        ),
        "",
        (
            "| Model | Family | Accuracy @0.5 | ROC-AUC | Recall @0.5 | Recall @0.25 "
            "| Precision @0.5 | Precision @0.25 | FN @0.5 | FN @0.25 |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in result["models"].values():
        half, quarter = m["operating_points"]["0.5"], m["operating_points"]["0.25"]
        family = "CNN" if m["family"] == "deep_learning" else "classical"
        lines.append(
            f"| {m['name']} | {family} | {_pct(half['accuracy'])} | {m['roc_auc']:.4f} "
            f"| {_pct(half['recall_damage'])} | {_pct(quarter['recall_damage'])} "
            f"| {_pct(half['precision_damage'])} | {_pct(quarter['precision_damage'])} "
            f"| {half['confusion_matrix']['fn']} | {quarter['confusion_matrix']['fn']} |"
        )

    lines += [
        "",
        f"## Confident predictions (P(damage) ≤ {0.5 - CONFIDENT_MARGIN:.1f} or ≥ {0.5 + CONFIDENT_MARGIN:.1f})",
        "",
        "| Model | Coverage | Images | Accuracy on confident subset |",
        "|---|---|---|---|",
    ]
    for m in result["models"].values():
        c = m["confident_subset"]
        lines.append(f"| {m['name']} | {_pct(c['coverage'])} | {c['n']} | {_pct(c['accuracy'])} |")

    if "error_overlap" in result:
        o = result["error_overlap"]
        a, b = DISPLAY_NAMES[o["a"]], DISPLAY_NAMES[o["b"]]
        lines += [
            "",
            f"## Error overlap @0.5 — {a} vs. {b}",
            "",
            "| Both correct | Only " + a + " correct | Only " + b + " correct | Both wrong |",
            "|---|---|---|---|",
            f"| {o['both_correct']} | {o['only_a_correct']} | {o['only_b_correct']} | {o['both_wrong']} |",
        ]

    checks = {k: m["matches_versioned_metrics"] for k, m in result["models"].items()
              if m["matches_versioned_metrics"] is not None}
    if checks:
        lines += ["", "CNN predictions reproduce the committed `outputs/metrics/` results: "
                  + ", ".join(f"{DISPLAY_NAMES[k]} {'yes' if ok else 'NO'}" for k, ok in checks.items()) + "."]
    return "\n".join(lines) + "\n"


def write_reports(result: dict, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "model_comparison.json"
    md_path = reports_dir / "model_comparison.md"
    # Explicit LF so generated artifacts are byte-identical on every OS.
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all models with saved validation predictions.")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    comparison = compare(args.predictions_dir)
    for path in write_reports(comparison, args.reports_dir):
        print(f"Saved: {path}")
