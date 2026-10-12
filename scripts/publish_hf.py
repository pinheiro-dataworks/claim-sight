"""Publish the trained ClaimSight checkpoint(s) to the Hugging Face Hub.

Prerequisites (not automated here, on purpose — this script never
handles credentials directly):

    1. Create a free account at https://huggingface.co
    2. Create a write-scoped access token at
       https://huggingface.co/settings/tokens
    3. Authenticate this machine:  huggingface-cli login

Usage:
    python -m scripts.publish_hf --repo-id PinheiroDataworks/claimsight-damage-detection
    python -m scripts.publish_hf --repo-id PinheiroDataworks/claimsight-damage-detection --arch efficientnet_b0

    # Render the model card to MODEL_CARD.md only (no upload, no credentials):
    python -m scripts.publish_hf --repo-id PinheiroDataworks/claimsight-damage-detection --card-only

The card embeds reports/model_comparison.json and
reports/confidence_intervals.json when present, so the published
numbers always match the versioned evaluation artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo

MODEL_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "hist_gb": "HistGradientBoosting (HSV + HOG + Canny)",
    "logreg": "Logistic regression (HSV + HOG + Canny)",
}


def _ci(intervals: dict | None, model: str, metric: str, pct: bool) -> str:
    """Format a '[lower, upper]' interval cell, or an em dash if unavailable."""
    ci = (intervals or {}).get("models", {}).get(model, {}).get(metric)
    if ci is None:
        return "—"
    if pct:
        return f"[{100 * ci['lower']:.2f}%, {100 * ci['upper']:.2f}%]"
    return f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"


def _evaluation_table(arch: str, metrics: dict, intervals: dict | None) -> str:
    default = metrics["default_threshold"]
    damage = default["classification_report"]["00-damage"]
    cm = default["confusion_matrix"]
    at_025 = next(r for r in metrics["threshold_table"] if abs(r["threshold"] - 0.25) < 1e-9)
    rows = [
        ("Accuracy (@ threshold 0.5)", f"{100 * default['accuracy']:.2f}%", _ci(intervals, arch, "accuracy@0.5", True)),
        ("ROC-AUC (damage class)", f"{default['roc_auc']:.4f}", _ci(intervals, arch, "roc_auc", False)),
        ("Recall — damage (@ 0.5)", f"{100 * damage['recall']:.2f}%", _ci(intervals, arch, "recall@0.5", True)),
        ("Recall — damage (@ 0.25)", f"{100 * at_025['recall_damage']:.2f}%", _ci(intervals, arch, "recall@0.25", True)),
        ("Precision — damage (@ 0.5)", f"{100 * damage['precision']:.2f}%", "—"),
        ("Confusion matrix @ 0.5 (TN/FP/FN/TP)", f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}", "—"),
    ]
    lines = ["| Metric | Value | 95% CI |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {ci} |" for name, value, ci in rows]
    if intervals:
        lines += ["", (
            f"95% intervals: percentile bootstrap, {intervals['n_resamples']:,} resamples of the "
            f"{intervals['n_images']} validation images, seed {intervals['seed']}."
        )]
    return "\n".join(lines)


def _baseline_section(arch: str, comparison: dict | None, intervals: dict | None, github_repo: str) -> str:
    if not comparison:
        return ""
    lines = [
        "## Baseline comparison",
        "",
        "Same training/validation split for every model. The classical baselines use hand-crafted",
        "features (HSV histogram, HOG, Canny/Sobel edge statistics) computed after the same",
        "`preprocess()` the CNNs consume.",
        "",
        "| Model | Accuracy @0.5 | ROC-AUC [95% CI] | Recall @0.5 | Recall @0.25 |",
        "|---|---|---|---|---|",
    ]
    for key, m in comparison["models"].items():
        name = f"**{MODEL_NAMES[key]}** (this checkpoint)" if key == arch else MODEL_NAMES[key]
        half, quarter = m["operating_points"]["0.5"], m["operating_points"]["0.25"]
        lines.append(
            f"| {name} | {100 * half['accuracy']:.2f}% | {m['roc_auc']:.4f} {_ci(intervals, key, 'roc_auc', False)} "
            f"| {100 * half['recall_damage']:.2f}% | {100 * quarter['recall_damage']:.2f}% |"
        )
    diffs = (intervals or {}).get("paired_differences", {})
    best = max((k for k in comparison["models"] if k in ("hist_gb", "logreg")),
               key=lambda k: comparison["models"][k]["roc_auc"], default=None)
    pair = diffs.get(f"{arch}-{best}") if best else None
    if pair:
        d = pair["roc_auc"]
        lines += ["", (
            f"Paired bootstrap difference in ROC-AUC, {MODEL_NAMES[arch]} minus {MODEL_NAMES[best]}: "
            f"{d['point']:+.4f} (95% CI {d['lower']:+.4f} to {d['upper']:+.4f})."
        )]
    lines += ["", (
        "Caveat: the dataset has no separate test split. The CNNs used the validation loss for early "
        "stopping and checkpoint selection, so their numbers are slightly optimistic relative to the "
        f"baselines. Methodology and interpretation: https://github.com/{github_repo}/tree/main/reports"
    ), ""]
    return "\n".join(lines)


def build_model_card(
    arch: str,
    metrics: dict,
    meta: dict,
    other_arch: str | None,
    other_metrics: dict | None,
    repo_id: str,
    github_repo: str,
    comparison: dict | None = None,
    intervals: dict | None = None,
) -> str:
    acc = metrics["default_threshold"]["accuracy"]
    auc = metrics["default_threshold"]["roc_auc"]
    recall = metrics["default_threshold"]["classification_report"]["00-damage"]["recall"]
    precision = metrics["default_threshold"]["classification_report"]["00-damage"]["precision"]
    tuned = metrics["recall_priority_threshold"]

    comparison_row = ""
    if other_arch and other_metrics:
        o_acc = other_metrics["default_threshold"]["accuracy"]
        o_auc = other_metrics["default_threshold"]["roc_auc"]
        comparison_row = f"| {other_arch} | {o_acc:.4f} | {o_auc:.4f} |\n"

    # With the full comparison report available, the richer baseline table
    # (which already includes both CNNs) replaces the two-row architecture table.
    baseline_section = _baseline_section(arch, comparison, intervals, github_repo)
    architecture_table = "" if baseline_section else (
        "| Architecture | Val. accuracy | ROC-AUC |\n|---|---|---|\n"
        f"| **{arch}** (this checkpoint) | {acc:.4f} | {auc:.4f} |\n{comparison_row}"
    )

    return f"""---
license: mit
tags:
  - computer-vision
  - image-classification
  - pytorch
  - insurance
  - claims-triage
  - grad-cam
pipeline_tag: image-classification
---

# ClaimSight — Vehicle Damage Triage Classifier ({arch})

Binary image classifier that flags whether a submitted vehicle photo
shows visible damage (`00-damage`) or an intact vehicle (`01-whole`),
built as a **claims-triage decision-support tool** — not an autonomous
adjuster. Every flagged prediction is intended to route to a human
reviewer.

Full project, training code, and API: https://github.com/{github_repo}

## Intended use

- First-pass triage of policyholder-submitted claim photos, to route
  obviously-intact vehicles away from a manual review queue.
- **Not** a final claims-adjudication system. **Not** a repair-cost
  estimator. Every prediction should be reviewed by a human before any
  claim decision is made.

## How to use

```python
import torch
from huggingface_hub import hf_hub_download

# clone github.com/{github_repo} for src/model.py, src/preprocessing.py
from src.model import build_model
from src.preprocessing import preprocess
from src.dataset import eval_transform

checkpoint_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="best_{arch}.pt",
)
model = build_model("{arch}", pretrained=False)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

import cv2
image_bgr = cv2.imread("claim_photo.jpg")
image_rgb = preprocess(image_bgr)                      # same function used in training
tensor = eval_transform(image=image_rgb)["image"].unsqueeze(0)
probs = torch.softmax(model(tensor), dim=1).squeeze()
print({{"00-damage": probs[1].item(), "01-whole": probs[0].item()}})
```

## Training data

Car Damage Detection (Kaggle, `anujms/car-damage-detection`) — 2,300
images, binary folder labels only (no masks/bounding boxes), split
1,840 train / 460 validation, balanced within each split.

## Architecture & training

Transfer learning ({arch}, ImageNet-pretrained) in two phases: frozen
backbone with a fresh head first, then fine-tuning of the last block at
a reduced learning rate. `ReduceLROnPlateau` + early stopping on
validation loss. Full details, augmentation policy and reproducibility
notes: see the project README.

## Evaluation (real, on the held-out validation split)

{_evaluation_table(arch, metrics, intervals)}

Recall on the `damage` class is the priority metric: a false negative
(damaged vehicle classified as intact) can wrongly close a legitimate
claim, while a false positive only costs one extra human review. A
recall-priority operating point was chosen by sweeping the decision
threshold: at threshold **{tuned['threshold']:.2f}**, damage recall is
**{tuned['recall_damage']:.4f}** at precision **{tuned['precision_damage']:.4f}**
(vs. {recall:.4f} recall / {precision:.4f} precision at the default 0.5).

{architecture_table}
{baseline_section}
## Explainability

Grad-CAM (`pytorch-grad-cam`) on the last convolutional layer is used
as a **weak-localization** signal — a coarse heatmap of the region that
most influenced the decision. It is **not** pixel-level segmentation:
there is no mask ground truth in this dataset, so there is no IoU/Dice.
See the project repo's `outputs/gradcam/` for overlays on both correct
and incorrect predictions, including a documented shortcut-learning
check.

## Limitations

- Binary output only — no severity or damaged-part classification.
- Grad-CAM is weak localization, not segmentation.
- Modest dataset size (2,300 images, one source) — real domain-shift
  risk against a real insurer's photo distribution (different brands,
  angles, lighting, phone cameras).
- Observed false positives on reflective, studio-lit dealership photos.
- No independent test split: reported metrics come from the validation
  split that was also used for early stopping.
- Not validated as a repair-cost estimator — triage signal only.
- Requires human review in every deployment path.

Tracked limitations and planned work: https://github.com/{github_repo}/issues

## Reproducibility

Fixed seed (`torch.manual_seed(42)`), deterministic cuDNN settings,
pinned `requirements.txt`. Metadata for this exact run:

```json
{json.dumps(meta, indent=2)}
```
"""


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render_card(
    repo_id: str,
    arch: str,
    also_publish_other: bool = True,
    models_dir: Path = Path("models"),
    metrics_dir: Path = Path("outputs/metrics"),
    reports_dir: Path = Path("reports"),
    github_repo: str = "pinheiro-dataworks/claim-sight",
) -> str:
    """Build the model card from versioned artifacts only (no checkpoint, no network)."""
    meta_path = models_dir / f"best_{arch}.meta.json"
    metrics_path = metrics_dir / f"metrics_{arch}.json"
    for p in (meta_path, metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — train and evaluate {arch} first.")

    other_arch = "efficientnet_b0" if arch == "resnet50" else "resnet50"
    return build_model_card(
        arch,
        _read_json(metrics_path),
        _read_json(meta_path),
        other_arch if also_publish_other else None,
        _read_json(metrics_dir / f"metrics_{other_arch}.json"),
        repo_id=repo_id,
        github_repo=github_repo,
        comparison=_read_json(reports_dir / "model_comparison.json"),
        intervals=_read_json(reports_dir / "confidence_intervals.json"),
    )


def publish(
    repo_id: str,
    arch: str,
    also_publish_other: bool = True,
    models_dir: Path = Path("models"),
    metrics_dir: Path = Path("outputs/metrics"),
    github_repo: str = "pinheiro-dataworks/claim-sight",
) -> None:
    checkpoint_path = models_dir / f"best_{arch}.pt"
    meta_path = models_dir / f"best_{arch}.meta.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {checkpoint_path} — train {arch} first.")

    metrics = _read_json(metrics_dir / f"metrics_{arch}.json")
    card = render_card(
        repo_id, arch, also_publish_other, models_dir, metrics_dir, github_repo=github_repo,
    )
    other_arch = "efficientnet_b0" if arch == "resnet50" else "resnet50"

    create_repo(repo_id, exist_ok=True)
    api = HfApi()

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=f"best_{arch}.pt",
        repo_id=repo_id,
        commit_message=f"Publish {arch} checkpoint — val_accuracy={metrics['default_threshold']['accuracy']:.4f}",
    )
    api.upload_file(
        path_or_fileobj=str(meta_path),
        path_in_repo=f"best_{arch}.meta.json",
        repo_id=repo_id,
    )

    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Update model card",
    )

    print(f"Published to https://huggingface.co/{repo_id}")

    if also_publish_other and (models_dir / f"best_{other_arch}.pt").exists():
        other_checkpoint = models_dir / f"best_{other_arch}.pt"
        other_meta = models_dir / f"best_{other_arch}.meta.json"
        api.upload_file(
            path_or_fileobj=str(other_checkpoint),
            path_in_repo=f"best_{other_arch}.pt",
            repo_id=repo_id,
            commit_message=f"Also publish {other_arch} checkpoint for comparison",
        )
        if other_meta.exists():
            api.upload_file(path_or_fileobj=str(other_meta), path_in_repo=f"best_{other_arch}.meta.json", repo_id=repo_id)
        print(f"Also published {other_arch} checkpoint to the same repo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a ClaimSight checkpoint to the Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="e.g. PinheiroDataworks/claimsight-damage-detection")
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--no-other-arch", action="store_true", help="Skip publishing the second architecture too.")
    parser.add_argument(
        "--github-repo", default="pinheiro-dataworks/claim-sight",
        help="owner/repo for the model card's 'full project' link (default: this project's own repo).",
    )
    parser.add_argument(
        "--card-only", action="store_true",
        help="Write the model card to --card-path and exit; nothing is uploaded.",
    )
    parser.add_argument("--card-path", type=Path, default=Path("MODEL_CARD.md"))
    args = parser.parse_args()

    if args.card_only:
        card = render_card(
            args.repo_id, args.arch, also_publish_other=not args.no_other_arch, github_repo=args.github_repo,
        )
        args.card_path.write_text(card, encoding="utf-8", newline="")
        print(f"Saved: {args.card_path}")
    else:
        publish(args.repo_id, args.arch, also_publish_other=not args.no_other_arch, github_repo=args.github_repo)
