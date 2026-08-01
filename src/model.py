"""Transfer-learning model builder.

Two backbones are supported for an honest side-by-side comparison:
ResNet50 (IMAGENET1K_V2) and EfficientNet-B0 (IMAGENET1K_V1). Both get a
fresh binary classification head; everything else starts frozen so
Phase 1 of training (src/train.py) can adapt the head without wrecking
pretrained ImageNet features with large, noisy gradients
(catastrophic forgetting).
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ("resnet50", "efficientnet_b0")


def build_model(arch: str = "resnet50", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a transfer-learning model with a fresh classification head.

    The backbone starts fully frozen (requires_grad=False everywhere);
    src/train.py selectively unfreezes the last block for Phase 2
    fine-tuning.
    """
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Supported: {SUPPORTED_ARCHS}")

    return model


def get_fine_tune_layer_names(arch: str) -> tuple[str, ...]:
    """Parameter name prefixes to unfreeze during Phase 2 fine-tuning."""
    if arch == "resnet50":
        return ("layer4", "fc")
    if arch == "efficientnet_b0":
        return ("features.8", "classifier")
    raise ValueError(f"Unknown architecture: {arch!r}. Supported: {SUPPORTED_ARCHS}")


def get_target_layer(model: nn.Module, arch: str):
    """Return the last convolutional layer used as the Grad-CAM target."""
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unknown architecture: {arch!r}. Supported: {SUPPORTED_ARCHS}")
