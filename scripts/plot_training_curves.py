"""Plot train/validation loss and accuracy curves for both architectures,
side by side, from the history JSON files written by src/train.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HISTORY_DIR = Path("outputs/history")
FIGURES_DIR = Path("outputs/figures")
ARCHS = ["resnet50", "efficientnet_b0"]
COLORS = {"resnet50": "#FFA23C", "efficientnet_b0": "#3FD6B0"}


def plot_comparison() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for arch in ARCHS:
        path = HISTORY_DIR / f"history_{arch}.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text(encoding="utf-8"))
        epochs = list(range(1, len(history) + 1))
        val_loss = [h["val_loss"] for h in history]
        val_acc = [h["val_acc"] for h in history]

        # Mark the phase-2 boundary (fine-tuning start)
        phase2_start = next((i for i, h in enumerate(history) if h["phase"] == "phase2_finetune"), None)

        axes[0].plot(epochs, val_loss, label=arch, color=COLORS[arch], linewidth=2)
        axes[1].plot(epochs, val_acc, label=arch, color=COLORS[arch], linewidth=2)
        if phase2_start is not None:
            for ax in axes:
                ax.axvline(phase2_start + 0.5, color=COLORS[arch], linestyle=":", alpha=0.5, linewidth=1)

    axes[0].set_xlabel("Epoch (cumulative across both phases)")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss per epoch")
    axes[0].legend()

    axes[1].set_xlabel("Epoch (cumulative across both phases)")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("Validation accuracy per epoch")
    axes[1].legend()

    fig.suptitle("Training curves — dotted line marks Phase 2 (fine-tuning) start", fontsize=11)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "training_curves_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = plot_comparison()
    print(f"Saved: {path}")
