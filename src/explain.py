"""Grad-CAM explainability — weak localization, not segmentation.

Grad-CAM (via pytorch-grad-cam) is computed on the last convolutional
layer and shows *where* the model looked to reach its decision. It is
deliberately generated for correct AND incorrect predictions across both
classes (TP, FP, TN, FN), specifically so shortcut learning can be
caught: if the heatmap for a correct "damage" prediction lights up the
license plate, a watermark, or the background instead of the dent, the
model has learned a spurious correlation rather than the actual damage
signal — and that becomes a documented limitation, not a hidden one.

What this module does NOT produce: a segmentation mask. There is no
pixel-level ground truth in this dataset, so there is no IoU/Dice here,
only a coarse activation heatmap intended for human-reviewed triage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import eval_transform
from src.model import build_model, get_target_layer
from src.preprocessing import preprocess

GRADCAM_DIR = Path("outputs/gradcam")
DAMAGE_LABEL = 1
OUTCOME_LABELS = {
    "TP": "True Positive (damage, correctly flagged)",
    "FN": "False Negative (damage, MISSED — costliest error)",
    "FP": "False Positive (whole, wrongly flagged)",
    "TN": "True Negative (whole, correctly cleared)",
}


def _load_model(arch: str, checkpoint: str | None, device):
    model = build_model(arch, pretrained=False)
    checkpoint_path = Path(checkpoint or f"models/best_{arch}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


def generate_overlay(model, arch: str, image_path: str | Path, device, target_class: int | None = None):
    """Run one image through preprocess() + Grad-CAM and return (overlay_rgb, pred_idx, prob)."""
    image_bgr = cv2.imread(str(image_path))
    raw = preprocess(image_bgr)  # same function used in training and serving
    rgb_float = raw.astype(np.float32) / 255.0
    input_tensor = eval_transform(image=raw)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1).squeeze()
        pred = int(probs.argmax().item())

    target = target_class if target_class is not None else pred
    target_layer = get_target_layer(model, arch)
    cam = GradCAM(model=model, target_layers=[target_layer])
    # At inference time the model is reloaded from a state_dict with every
    # parameter's requires_grad reset to whatever build_model() set at
    # construction (frozen backbone) — that flag is not part of the
    # checkpoint. Grad-CAM needs a live autograd graph to back-propagate
    # through, so the input tensor must explicitly require grad regardless
    # of which layers were fine-tuned during training.
    input_tensor.requires_grad_(True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target)])[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return overlay, pred, float(probs[target].item())


def categorize_validation_set(model, val_root: Path, device) -> dict[str, list[Path]]:
    """Bucket every validation image into TP / FP / TN / FN at threshold=0.5."""
    buckets: dict[str, list[Path]] = {"TP": [], "FP": [], "TN": [], "FN": []}
    class_dirs = {"00-damage": DAMAGE_LABEL, "01-whole": 0}

    with torch.no_grad():
        for class_folder, true_label in class_dirs.items():
            folder = val_root / class_folder
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                image_bgr = cv2.imread(str(path))
                raw = preprocess(image_bgr)
                tensor = eval_transform(image=raw)["image"].unsqueeze(0).to(device)
                pred = int(model(tensor).argmax(1).item())

                if true_label == DAMAGE_LABEL and pred == DAMAGE_LABEL:
                    buckets["TP"].append(path)
                elif true_label == DAMAGE_LABEL and pred != DAMAGE_LABEL:
                    buckets["FN"].append(path)
                elif true_label != DAMAGE_LABEL and pred == DAMAGE_LABEL:
                    buckets["FP"].append(path)
                else:
                    buckets["TN"].append(path)
    return buckets


def build_summary_grid(model, arch: str, buckets: dict[str, list[Path]], device, n_per_bucket: int = 3) -> Path:
    """One figure: rows = outcome categories, columns = example overlays."""
    rows = [k for k in ("TP", "FN", "FP", "TN") if buckets[k]]
    n_cols = n_per_bucket
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.2 * n_cols, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    for r, outcome in enumerate(rows):
        samples = buckets[outcome][:n_cols]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(samples):
                overlay, pred, conf = generate_overlay(model, arch, samples[c], device)
                ax.imshow(overlay)
                ax.set_title(f"pred={('damage' if pred == 1 else 'whole')} ({conf:.2f})", fontsize=9)
            ax.axis("off")
        axes[r, 0].set_ylabel(outcome, fontsize=11, rotation=0, labelpad=40)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Grad-CAM by confusion-matrix outcome — {arch}", fontsize=13)
    plt.tight_layout()
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRADCAM_DIR / f"summary_grid_{arch}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_individual_overlays(model, arch: str, buckets: dict[str, list[Path]], device, n_per_bucket: int = 4) -> dict:
    """Save individual before/after overlay PNGs, grouped by outcome folder."""
    saved: dict[str, list[str]] = {k: [] for k in buckets}
    for outcome, paths in buckets.items():
        out_dir = GRADCAM_DIR / arch / outcome
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in paths[:n_per_bucket]:
            overlay, _pred, _conf = generate_overlay(model, arch, path, device)
            out_path = out_dir / f"{path.stem}_gradcam.png"
            plt.imsave(out_path, overlay)
            saved[outcome].append(str(out_path))
    return saved


def run_explainability(arch: str, data_root: str = "data/raw", checkpoint: str | None = None, n_per_bucket: int = 4) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(arch, checkpoint, device)
    val_root = Path(data_root) / "validation"

    buckets = categorize_validation_set(model, val_root, device)
    counts = {k: len(v) for k, v in buckets.items()}
    print(f"Validation outcome counts (threshold=0.5): {counts}")

    grid_path = build_summary_grid(model, arch, buckets, device, n_per_bucket=min(3, n_per_bucket))
    saved_individual = save_individual_overlays(model, arch, buckets, device, n_per_bucket=n_per_bucket)

    summary = {
        "arch": arch,
        "outcome_counts": counts,
        "summary_grid": str(grid_path),
        "individual_overlays": saved_individual,
    }
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRADCAM_DIR / f"summary_{arch}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Grad-CAM summary grid: {grid_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a trained checkpoint.")
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-per-bucket", type=int, default=4)
    args = parser.parse_args()

    run_explainability(args.arch, data_root=args.data_root, checkpoint=args.checkpoint, n_per_bucket=args.n_per_bucket)
