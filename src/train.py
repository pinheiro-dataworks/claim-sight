"""Two-phase transfer-learning training loop.

Phase 1 — head only, backbone frozen: the new classification head starts
with random weights. Training everything at once would let its large,
noisy gradients wreck the pretrained ImageNet features in the backbone
(catastrophic forgetting), so the backbone stays frozen and only the
head trains, at a relatively high learning rate.

Phase 2 — fine-tune the last block: with the head stabilized, the final
convolutional block is unfrozen and trained at a much lower learning
rate. Early layers encode generic features (edges, textures) that
transfer well as-is; only the late layers need to specialize toward
"what damage looks like."

Reproducibility: a fixed seed, deterministic cuDNN settings, and a
per-run metadata JSON (architecture, seed, epochs, timestamp, git-free
package versions) saved next to every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.dataset import get_loaders
from src.model import build_model, get_fine_tune_layer_names

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs/history")


def set_seed(seed: int = 42) -> None:
    """Fix all random sources for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class PhaseResult:
    epochs_ran: int
    best_val_loss: float
    history: list[dict] = field(default_factory=list)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        running_loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


def run_phase(
    model: nn.Module,
    train_dl,
    val_dl,
    *,
    phase_name: str,
    arch: str,
    epochs: int,
    lr: float,
    device: torch.device,
    patience: int = 5,
    checkpoint_path: Path,
) -> PhaseResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loss_acc(model, val_dl, criterion, device)
        scheduler.step(val_loss)
        dt = time.time() - t0

        record = {
            "phase": phase_name, "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "seconds": round(dt, 1),
        }
        history.append(record)
        print(
            f"[{arch} | {phase_name}] epoch {epoch + 1}/{epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.4f} | {dt:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1} ({phase_name}).")
                break

    return PhaseResult(epochs_ran=len(history), best_val_loss=best_val_loss, history=history)


def train_architecture(
    arch: str,
    *,
    data_root: str = "data/raw",
    batch_size: int = 32,
    phase1_epochs: int = 8,
    phase1_lr: float = 1e-3,
    phase2_epochs: int = 15,
    phase2_lr: float = 1e-5,
    patience: int = 5,
    seed: int = 42,
    num_workers: int = 2,
) -> dict:
    """Run both training phases for one architecture and return a summary dict."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training {arch} on {device} ===")

    train_dl, val_dl = get_loaders(data_root, batch_size=batch_size, num_workers=num_workers)

    model = build_model(arch).to(device)
    checkpoint_path = MODELS_DIR / f"best_{arch}.pt"

    phase1 = run_phase(
        model, train_dl, val_dl,
        phase_name="phase1_head", arch=arch, epochs=phase1_epochs, lr=phase1_lr,
        device=device, patience=patience, checkpoint_path=checkpoint_path,
    )

    # Load the best Phase 1 checkpoint before unfreezing for Phase 2.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    unfreeze_prefixes = get_fine_tune_layer_names(arch)
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
            p.requires_grad = True

    phase2 = run_phase(
        model, train_dl, val_dl,
        phase_name="phase2_finetune", arch=arch, epochs=phase2_epochs, lr=phase2_lr,
        device=device, patience=patience, checkpoint_path=checkpoint_path,
    )

    full_history = phase1.history + phase2.history
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUTS_DIR / f"history_{arch}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)

    best_val_loss = min(phase1.best_val_loss, phase2.best_val_loss)
    metadata = {
        "arch": arch,
        "seed": seed,
        "device": str(device),
        "phase1_epochs_ran": phase1.epochs_ran,
        "phase2_epochs_ran": phase2.epochs_ran,
        "best_val_loss": best_val_loss,
        "checkpoint": str(checkpoint_path),
        "torch_version": torch.__version__,
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = MODELS_DIR / f"best_{arch}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"=== {arch} done. Best val_loss={best_val_loss:.4f}. Checkpoint: {checkpoint_path} ===")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ClaimSight damage classifier.")
    parser.add_argument("--arch", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase1-epochs", type=int, default=8)
    parser.add_argument("--phase2-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    train_architecture(
        args.arch,
        data_root=args.data_root,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )
