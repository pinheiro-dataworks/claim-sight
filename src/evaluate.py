"""Business-oriented evaluation: confusion matrix, per-class metrics,
ROC-AUC, decision-threshold tuning, and an explicit false-negative audit.

Accuracy alone is not enough here because the cost of errors is
asymmetric: a damaged vehicle misclassified as intact (a false negative
on the `damage` class) can wrongly close a legitimate claim, while a
false positive only costs one extra human review. Everything below is
built around that asymmetry — recall on `damage` is the metric that
matters most, and the default 0.5 argmax threshold is treated as a
starting point to be tuned, not a given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.dataset import CarDamageDataset, eval_transform
from src.model import build_model

FIGURES_DIR = Path("outputs/figures")
METRICS_DIR = Path("outputs/metrics")
DAMAGE_LABEL = 1  # matches CLASS_MAP in src/dataset.py
CLASS_NAMES_BY_LABEL = {0: "01-whole", 1: "00-damage"}


@torch.no_grad()
def get_predictions(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob_damage) over an entire loader."""
    model.eval().to(device)
    ys, probs = [], []
    for x, y in loader:
        out = torch.softmax(model(x.to(device)), dim=1)
        probs.extend(out[:, DAMAGE_LABEL].cpu().numpy())
        ys.extend(y.numpy())
    return np.array(ys), np.array(probs)


def compute_metrics(y_true: np.ndarray, y_prob_damage: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute confusion matrix, per-class report and ROC-AUC at a given threshold."""
    y_pred = (y_prob_damage >= threshold).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["01-whole", "00-damage"],
        output_dict=True, zero_division=0,
    )
    auc = roc_auc_score(y_true, y_prob_damage)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows/cols: whole, damage
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
        "roc_auc": float(auc),
        "accuracy": float((y_pred == y_true).mean()),
    }


def scan_thresholds(y_true: np.ndarray, y_prob_damage: np.ndarray, steps: int = 19) -> list[dict]:
    """Sweep thresholds and record the recall/precision trade-off for the damage class."""
    rows = []
    for t in np.linspace(0.05, 0.95, steps):
        m = compute_metrics(y_true, y_prob_damage, threshold=float(t))
        damage_stats = m["classification_report"]["00-damage"]
        rows.append({
            "threshold": round(float(t), 3),
            "recall_damage": round(damage_stats["recall"], 4),
            "precision_damage": round(damage_stats["precision"], 4),
            "f1_damage": round(damage_stats["f1-score"], 4),
            "fn": m["confusion_matrix"]["fn"],
            "fp": m["confusion_matrix"]["fp"],
        })
    return rows


def choose_recall_priority_threshold(
    threshold_table: list[dict], min_recall: float = 0.95
) -> dict:
    """Pick the highest threshold that still keeps damage recall >= min_recall.

    A higher threshold means fewer whole cars wrongly flagged (better
    precision, less human-review overhead) without giving up the recall
    floor the business requires on the costlier error type.
    """
    candidates = [row for row in threshold_table if row["recall_damage"] >= min_recall]
    if not candidates:
        # No threshold reaches the target recall — fall back to the one
        # with the highest achievable recall instead of silently failing.
        return max(threshold_table, key=lambda r: r["recall_damage"])
    return max(candidates, key=lambda r: r["threshold"])


def plot_confusion_matrix(cm: dict, arch: str, threshold: float) -> Path:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Oranges", cbar=False,
        xticklabels=["whole", "damage"], yticklabels=["whole", "damage"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {arch} @ threshold={threshold:.2f}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"confusion_matrix_{arch}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(y_true, y_prob_damage, arch: str, auc: float) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_prob_damage)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#FFA23C", linewidth=2, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8993A4", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve (damage class) — {arch}")
    ax.legend(loc="lower right")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"roc_curve_{arch}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_threshold_tradeoff(threshold_table: list[dict], arch: str) -> Path:
    thresholds = [r["threshold"] for r in threshold_table]
    recall = [r["recall_damage"] for r in threshold_table]
    precision = [r["precision_damage"] for r in threshold_table]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thresholds, recall, color="#FFA23C", linewidth=2, label="Recall (damage)")
    ax.plot(thresholds, precision, color="#3FD6B0", linewidth=2, label="Precision (damage)")
    ax.axvline(0.5, color="#8993A4", linestyle=":", linewidth=1, label="Default (0.5)")
    ax.set_xlabel("Decision threshold on P(damage)")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision/recall vs. threshold — {arch}")
    ax.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"threshold_tradeoff_{arch}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def find_false_negatives(model, dataset: CarDamageDataset, device) -> list[str]:
    """Return file paths of damage images the model misclassified as whole."""
    model.eval().to(device)
    fn_paths = []
    with torch.no_grad():
        for (path, label), (tensor, _) in zip(dataset.samples, dataset):
            if label != DAMAGE_LABEL:
                continue
            pred = model(tensor.unsqueeze(0).to(device)).argmax(1).item()
            if pred != DAMAGE_LABEL:
                fn_paths.append(str(path))
    return fn_paths


def evaluate_architecture(
    arch: str,
    data_root: str = "data/raw",
    checkpoint: str | None = None,
    min_recall: float = 0.95,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint or f"models/best_{arch}.pt")

    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    val_ds = CarDamageDataset(Path(data_root) / "validation", eval_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_prob = get_predictions(model, val_dl, device)

    default_metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    threshold_table = scan_thresholds(y_true, y_prob)
    chosen = choose_recall_priority_threshold(threshold_table, min_recall=min_recall)
    tuned_metrics = compute_metrics(y_true, y_prob, threshold=chosen["threshold"])

    cm_path = plot_confusion_matrix(default_metrics["confusion_matrix"], arch, threshold=0.5)
    roc_path = plot_roc_curve(y_true, y_prob, arch, default_metrics["roc_auc"])
    tradeoff_path = plot_threshold_tradeoff(threshold_table, arch)

    fn_paths = find_false_negatives(model, val_ds, device)

    summary = {
        "arch": str(arch),
        "n_validation": int(len(val_ds)),
        "default_threshold": default_metrics,
        "threshold_table": threshold_table,
        "recall_priority_threshold": chosen,
        "recall_priority_metrics": tuned_metrics,
        "false_negative_count": len(fn_paths),
        "false_negative_paths": fn_paths,
        "figures": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
            "threshold_tradeoff": str(tradeoff_path),
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"metrics_{arch}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"=== {arch} evaluation ===")
    print(f"val_accuracy (threshold=0.5): {default_metrics['accuracy']:.4f}")
    print(f"ROC-AUC (damage): {default_metrics['roc_auc']:.4f}")
    print(
        f"Recall-priority threshold: {chosen['threshold']:.2f} "
        f"(recall={chosen['recall_damage']:.4f}, precision={chosen['precision_damage']:.4f})"
    )
    print(f"False negatives at default threshold: {default_metrics['confusion_matrix']['fn']}")
    print(f"Saved: {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ClaimSight checkpoint.")
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--min-recall", type=float, default=0.95)
    args = parser.parse_args()

    evaluate_architecture(
        args.arch, data_root=args.data_root,
        checkpoint=args.checkpoint, min_recall=args.min_recall,
    )
