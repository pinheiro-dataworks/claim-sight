"""Unit tests for src/model.py — architecture construction and the
metadata helpers used by training (fine-tune layer names) and
explainability (Grad-CAM target layer).
"""
import pytest
import torch.nn as nn

from src.model import SUPPORTED_ARCHS, build_model, get_fine_tune_layer_names, get_target_layer


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_build_model_freezes_backbone_but_not_the_fresh_head(arch):
    model = build_model(arch, pretrained=False)
    assert isinstance(model, nn.Module)
    params = list(model.parameters())
    # the backbone is frozen before the head is replaced, so some
    # parameters must be frozen (backbone) and some must not (the new
    # head, created fresh afterward with its default requires_grad=True)
    assert any(not p.requires_grad for p in params), "backbone should start frozen"
    assert any(p.requires_grad for p in params), "the fresh classification head should be trainable"


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("not-a-real-arch")


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_fine_tune_layer_names_are_nonempty(arch):
    names = get_fine_tune_layer_names(arch)
    assert isinstance(names, tuple)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


def test_fine_tune_layer_names_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_fine_tune_layer_names("not-a-real-arch")


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_get_target_layer_returns_a_module(arch):
    model = build_model(arch, pretrained=False)
    layer = get_target_layer(model, arch)
    assert isinstance(layer, nn.Module)


def test_get_target_layer_rejects_unknown_architecture():
    model = build_model("resnet50", pretrained=False)
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_target_layer(model, "not-a-real-arch")
