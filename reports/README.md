# Reports

Evaluation artifacts that compare every ClaimSight model on the same
460-image validation split. All tables are generated from versioned
per-image predictions, so they can be regenerated without checkpoints or
the raw dataset.

| File | Produced by | Content |
|---|---|---|
| `predictions/validation_<model>.csv` | `python -m src.predictions` (CNNs), `python -m src.baseline` (classical) | One row per validation image: relative path, true label, P(damage) |
| `predictions/baseline_meta.json` | `python -m src.baseline` | Feature layout, fitted hyperparameters, fit times |
| `model_comparison.{json,md}` | `python -m src.compare_models` | Accuracy, ROC-AUC, recall/precision @0.5 and @0.25, confident-subset coverage, error overlap |

Reproduce from scratch (needs `data/raw/` and the checkpoints in `models/`):

```bash
pip install -r requirements-dev.txt
python -m src.predictions --arch all
python -m src.baseline
python -m src.compare_models
```

## Classical baseline

The baseline answers a simple question: how much of the CNNs'
performance comes from learned deep features rather than from
descriptors any computer-vision pipeline would try first?

- **Features (1,879 dims)**, computed on the output of the same
  `preprocess()` the CNNs consume: HSV color histogram (96, letterbox
  padding masked out), HOG on 128×128 grayscale (1,764), and Canny/Sobel
  edge statistics including a 4×4 grid of local edge densities (19).
- **Models**: `StandardScaler` + L2 `LogisticRegressionCV` (C selected by
  stratified 5-fold CV on the training split) and
  `HistGradientBoostingClassifier` (early stopping on a 15% slice of the
  training split). Seed 42 throughout; two consecutive runs produce
  byte-identical prediction files.
- **Splits**: fit on the 1,840 training images, scored on the 460
  validation images, with `CarDamageDataset`'s exact file lists.

## Results

See [`model_comparison.md`](model_comparison.md) for the full generated
tables. Headline numbers:

| Model | Accuracy @0.5 | ROC-AUC | Recall @0.5 | Recall @0.25 | FP @0.25 |
|---|---|---|---|---|---|
| ResNet50 | 94.35% | 0.9858 | 95.65% | 98.70% | 23 |
| EfficientNet-B0 | 88.70% | 0.9620 | 88.70% | 95.65% | 48 |
| HistGradientBoosting (HSV + HOG + Canny) | 84.57% | 0.9240 | 81.74% | 94.35% | 74 |
| Logistic regression (HSV + HOG + Canny) | 82.17% | 0.9097 | 78.70% | 96.96% | 105 |

## Interpretation

**What deep learning adds.** The classical features are far from
useless: a ROC-AUC of 0.92 means color, texture and edge statistics
already rank most damaged cars above most intact ones. ResNet50 still
improves on the best baseline by 0.062 ROC-AUC and 9.8 accuracy points,
and the difference is concentrated on the costly error type: at the
default threshold it misses 10 damaged cars against 42 for
HistGradientBoosting. The baselines can match ResNet50's recall only by
lowering the threshold, which multiplies false alarms: logistic
regression reaches 96.96% recall @0.25 but flags 105 intact cars for
review, against 23 for ResNet50 at a higher recall (98.70%). In claims
triage that is the gap between a filter that removes review work and
one that mostly forwards it.

The error-overlap table supports the same reading. On the 460 images,
ResNet50 is right where HistGradientBoosting is wrong 62 times, and the
reverse happens 17 times. A plausible explanation, not yet verified
image by image, is that reflections, panel gaps and specular highlights
produce edge and HOG signatures similar to scratches and dents; separating
them requires context about car parts and shape that the CNN learns and
global descriptors do not encode.

**When the baseline would be enough.**

- *Ranking, not routing.* If the system only orders a human review queue
  (every claim is still reviewed), a 0.92 ROC-AUC model already puts most
  damaged cars first, and it trains in seconds on a CPU.
- *No deep-learning runtime.* The baseline needs only OpenCV,
  scikit-image and scikit-learn; the ResNet50 checkpoint is ~94 MB and
  needs PyTorch at inference time.
- *Monitoring and sanity checks.* A cheap, independent model is useful
  for detecting drift or silent failures in the CNN, because the two
  models make largely different errors (17 images only the baseline gets
  right).

It is **not** a good first stage of a cascade: HistGradientBoosting is
confident (P ≤ 0.1 or ≥ 0.9) on only 46.3% of images, at 96.24% accuracy
there, while ResNet50 is confident on 80.2% at 98.64%. Short-circuiting
on the baseline would save CNN compute on fewer than half the images and
add errors on them.

## Caveats

- **Validation is not a pristine test set.** The dataset ships only
  training and validation splits. The CNNs used the validation loss for
  early stopping and checkpoint selection; the baselines did not touch
  it. CNN numbers are therefore slightly optimistic relative to the
  baselines. With a 0.062 ROC-AUC gap the conclusion is unlikely to
  change, but an independent test set is needed to measure the exact
  margin.
- **Threshold 0.25 is not a matched operating point.** Each model's
  probabilities are calibrated differently, so recall @0.25 compares
  different precision levels. ROC-AUC and the false-positive counts
  above are the fairer comparisons.
- **Single split, single seed.** Sampling uncertainty on 460 images is
  not shown in these tables.
