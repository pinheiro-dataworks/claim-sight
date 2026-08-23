"""Publish the trained ClaimSight checkpoint(s) to the Hugging Face Hub.

Prerequisites (not automated here, on purpose — this script never
handles credentials directly):

    1. Create a free account at https://huggingface.co
    2. Create a write-scoped access token at
       https://huggingface.co/settings/tokens
    3. Authenticate this machine:  huggingface-cli login

Usage:
    python -m scripts.publish_hf --repo-id <your-username>/claimsight-damage-detection
    python -m scripts.publish_hf --repo-id <your-username>/claimsight-damage-detection --arch efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def build_model_card(
    arch: str,
    metrics: dict,
    meta: dict,
    other_arch: str | None,
    other_metrics: dict | None,
    repo_id: str,
    github_repo: str,
) -> str:
    cm = metrics["default_threshold"]["confusion_matrix"]
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

| Metric | Value |
|---|---|
| Validation accuracy | {acc:.4f} |
| ROC-AUC (damage class) | {auc:.4f} |
| Recall — damage class (@ threshold 0.5) | {recall:.4f} |
| Precision — damage class (@ threshold 0.5) | {precision:.4f} |
| Confusion matrix (TN/FP/FN/TP) | {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |

Recall on the `damage` class is the priority metric: a false negative
(damaged vehicle classified as intact) can wrongly close a legitimate
claim, while a false positive only costs one extra human review. A
recall-priority operating point was chosen by sweeping the decision
threshold: at threshold **{tuned['threshold']:.2f}**, damage recall is
**{tuned['recall_damage']:.4f}** at precision **{tuned['precision_damage']:.4f}**
(vs. {recall:.4f} recall / {precision:.4f} precision at the default 0.5).

| Architecture | Val. accuracy | ROC-AUC |
|---|---|---|
| **{arch}** (this checkpoint) | {acc:.4f} | {auc:.4f} |
{comparison_row}

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
- Not validated as a repair-cost estimator — triage signal only.
- Requires human review in every deployment path.

## Reproducibility

Fixed seed (`torch.manual_seed(42)`), deterministic cuDNN settings,
pinned `requirements.txt`. Metadata for this exact run:

```json
{json.dumps(meta, indent=2)}
```
"""


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
    metrics_path = metrics_dir / f"metrics_{arch}.json"

    for p in (checkpoint_path, meta_path, metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — train and evaluate {arch} first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    other_arch = "efficientnet_b0" if arch == "resnet50" else "resnet50"
    other_metrics_path = metrics_dir / f"metrics_{other_arch}.json"
    other_metrics = json.loads(other_metrics_path.read_text(encoding="utf-8")) if other_metrics_path.exists() else None

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

    card = build_model_card(
        arch, metrics, meta, other_arch if also_publish_other else None, other_metrics,
        repo_id=repo_id, github_repo=github_repo,
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
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/claimsight-damage-detection")
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--no-other-arch", action="store_true", help="Skip publishing the second architecture too.")
    parser.add_argument(
        "--github-repo", default="pinheiro-dataworks/claim-sight",
        help="owner/repo for the model card's 'full project' link (default: this project's own repo).",
    )
    args = parser.parse_args()

    publish(args.repo_id, args.arch, also_publish_other=not args.no_other_arch, github_repo=args.github_repo)
