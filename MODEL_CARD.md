---
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

# ClaimSight — Vehicle Damage Triage Classifier (resnet50)

Binary image classifier that flags whether a submitted vehicle photo
shows visible damage (`00-damage`) or an intact vehicle (`01-whole`),
built as a **claims-triage decision-support tool** — not an autonomous
adjuster. Every flagged prediction is intended to route to a human
reviewer.

Full project, training code, and API: https://github.com/pinheiro-dataworks/claim-sight

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

# clone github.com/pinheiro-dataworks/claim-sight for src/model.py, src/preprocessing.py
from src.model import build_model
from src.preprocessing import preprocess
from src.dataset import eval_transform

checkpoint_path = hf_hub_download(
    repo_id="PinheiroDataworks/claimsight-damage-detection",
    filename="best_resnet50.pt",
)
model = build_model("resnet50", pretrained=False)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

import cv2
image_bgr = cv2.imread("claim_photo.jpg")
image_rgb = preprocess(image_bgr)                      # same function used in training
tensor = eval_transform(image=image_rgb)["image"].unsqueeze(0)
probs = torch.softmax(model(tensor), dim=1).squeeze()
print({"00-damage": probs[1].item(), "01-whole": probs[0].item()})
```

## Training data

Car Damage Detection (Kaggle, `anujms/car-damage-detection`) — 2,300
images, binary folder labels only (no masks/bounding boxes), split
1,840 train / 460 validation, balanced within each split.

## Architecture & training

Transfer learning (resnet50, ImageNet-pretrained) in two phases: frozen
backbone with a fresh head first, then fine-tuning of the last block at
a reduced learning rate. `ReduceLROnPlateau` + early stopping on
validation loss. Full details, augmentation policy and reproducibility
notes: see the project README.

## Evaluation (real, on the held-out validation split)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy (@ threshold 0.5) | 94.35% | [92.17%, 96.30%] |
| ROC-AUC (damage class) | 0.9858 | [0.9751, 0.9935] |
| Recall — damage (@ 0.5) | 95.65% | [92.69%, 98.21%] |
| Recall — damage (@ 0.25) | 98.70% | [97.00%, 100.00%] |
| Precision — damage (@ 0.5) | 93.22% | — |
| Confusion matrix @ 0.5 (TN/FP/FN/TP) | 214/16/10/220 | — |

95% intervals: percentile bootstrap, 2,000 resamples of the 460 validation images, seed 42.

Recall on the `damage` class is the priority metric: a false negative
(damaged vehicle classified as intact) can wrongly close a legitimate
claim, while a false positive only costs one extra human review. A
recall-priority operating point was chosen by sweeping the decision
threshold: at threshold **0.25**, damage recall is
**0.9870** at precision **0.9080**
(vs. 0.9565 recall / 0.9322 precision at the default 0.5).


## Baseline comparison

Same training/validation split for every model. The classical baselines use hand-crafted
features (HSV histogram, HOG, Canny/Sobel edge statistics) computed after the same
`preprocess()` the CNNs consume.

| Model | Accuracy @0.5 | ROC-AUC [95% CI] | Recall @0.5 | Recall @0.25 |
|---|---|---|---|---|
| **ResNet50** (this checkpoint) | 94.35% | 0.9858 [0.9751, 0.9935] | 95.65% | 98.70% |
| EfficientNet-B0 | 88.70% | 0.9620 [0.9469, 0.9761] | 88.70% | 95.65% |
| HistGradientBoosting (HSV + HOG + Canny) | 84.57% | 0.9240 [0.8999, 0.9457] | 81.74% | 94.35% |
| Logistic regression (HSV + HOG + Canny) | 82.17% | 0.9097 [0.8832, 0.9333] | 78.70% | 96.96% |

Paired bootstrap difference in ROC-AUC, ResNet50 minus HistGradientBoosting (HSV + HOG + Canny): +0.0618 (95% CI +0.0418 to +0.0843).

Caveat: the dataset has no separate test split. The CNNs used the validation loss for early stopping and checkpoint selection, so their numbers are slightly optimistic relative to the baselines. Methodology and interpretation: https://github.com/pinheiro-dataworks/claim-sight/tree/main/reports

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

Tracked limitations and planned work: https://github.com/pinheiro-dataworks/claim-sight/issues

## Reproducibility

Fixed seed (`torch.manual_seed(42)`), deterministic cuDNN settings,
pinned `requirements.txt`. Metadata for this exact run:

```json
{
  "arch": "resnet50",
  "seed": 42,
  "device": "cuda",
  "phase1_epochs_ran": 8,
  "phase2_epochs_ran": 15,
  "best_val_loss": 0.14994667431582576,
  "checkpoint": "models\\best_resnet50.pt",
  "torch_version": "2.6.0+cu124",
  "trained_at_utc": "2026-08-01T22:51:16Z"
}
```
