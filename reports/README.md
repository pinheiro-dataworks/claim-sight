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
| `confidence_intervals.{json,md}` | `python -m src.confidence_intervals` | 95% bootstrap intervals per model and paired differences against ResNet50 |

Reproduce from scratch (needs `data/raw/` and the checkpoints in `models/`):

```bash
pip install -r requirements-dev.txt
python -m src.predictions --arch all
python -m src.baseline
python -m src.compare_models
python -m src.confidence_intervals
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

## Confidence intervals

95% percentile bootstrap intervals: 2,000 resamples of the 460 validation
images, seed 42, with the same resampled index sets shared by every model
and metric (full table: [`confidence_intervals.md`](confidence_intervals.md)).

| Model | Accuracy @0.5 | ROC-AUC | Recall @0.5 | Recall @0.25 |
|---|---|---|---|---|
| ResNet50 | 94.35% [92.17, 96.30] | 0.9858 [0.9751, 0.9935] | 95.65% [92.69, 98.21] | 98.70% [97.00, 100.00] |
| EfficientNet-B0 | 88.70% [85.87, 91.52] | 0.9620 [0.9469, 0.9761] | 88.70% [84.28, 92.51] | 95.65% [92.80, 97.96] |
| HistGradientBoosting | 84.57% [81.30, 87.61] | 0.9240 [0.8999, 0.9457] | 81.74% [76.57, 86.70] | 94.35% [91.21, 97.14] |
| Logistic regression | 82.17% [78.70, 85.43] | 0.9097 [0.8832, 0.9333] | 78.70% [73.54, 84.05] | 96.96% [94.56, 99.08] |

Comparing two marginal intervals understates what the data can resolve,
because both models are scored on the same images. The paired intervals
for ResNet50 minus each model exclude zero on every metric:

| Compared with | Δ Accuracy @0.5 (pts) | Δ ROC-AUC | Δ Recall @0.5 (pts) |
|---|---|---|---|
| EfficientNet-B0 | +5.65 [+3.04, +8.48] | +0.0238 [+0.0118, +0.0362] | +6.96 [+2.84, +11.16] |
| HistGradientBoosting | +9.78 [+6.30, +13.26] | +0.0618 [+0.0418, +0.0843] | +13.91 [+8.37, +19.73] |
| Logistic regression | +12.17 [+8.70, +15.87] | +0.0761 [+0.0547, +0.1007] | +16.96 [+11.54, +22.36] |

## Interpretation

**What deep learning adds.** The classical features are far from
useless: a ROC-AUC of 0.92 means color, texture and edge statistics
already rank most damaged cars above most intact ones. ResNet50 still
improves on the best baseline by 0.062 ROC-AUC (paired 95% CI
+0.042 to +0.084) and 9.8 accuracy points (+6.3 to +13.3), and the difference is concentrated on the costly error type: at the
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
  for detecting drift or silent failures in the CNN, because their errors
  only partly overlap: the baseline is right on 17 of ResNet50's 26
  errors at threshold 0.5.

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
  baselines. The paired ROC-AUC interval against the best baseline
  (+0.042 to +0.084) leaves room for that bias without changing the
  conclusion, but an independent test set is needed to measure the exact
  margin.
- **Threshold 0.25 is not a matched operating point.** Each model's
  probabilities are calibrated differently, so recall @0.25 compares
  different precision levels. ROC-AUC and the false-positive counts
  above are the fairer comparisons.
- **Intervals cover evaluation sampling only.** The bootstrap resamples
  the 460 validation images; it does not capture training variance
  (different seeds or data orderings), which would widen the intervals.
- **Percentile intervals near the boundary.** ResNet50 recall @0.25
  misses only 3 of 230 damaged cars, so its upper bound reaches 100% and
  the interval is less reliable than the others.
