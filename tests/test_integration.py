"""Integration tests — download the real published checkpoint from the
Hugging Face Hub and verify actual trained behavior, not just API
shape (that's what test_api.py's contract tests are for).

Opt-in: needs network access, so these are marked `integration` and
excluded from the default test run (see pyproject.toml's `-m` config
note and .github/workflows/tests.yml). Run explicitly with:

    pytest tests/test_integration.py -m integration -v

Uses EfficientNet-B0 (~16MB) rather than the deployed ResNet50 (~94MB)
specifically to keep this fast and light, per the "download a small
checkpoint when needed" principle — this is a smoke test that a real
trained artifact loads and behaves sanely end-to-end, not a benchmark
of the specific deployed model (that's already covered by the real
numbers in outputs/metrics/metrics_resnet50.json).
"""
import pytest
import torch
from fastapi.testclient import TestClient
from huggingface_hub import hf_hub_download

from src.dataset import eval_transform
from src.model import build_model
from src.preprocessing import preprocess

pytestmark = pytest.mark.integration

HF_REPO = "PinheiroDataworks/claimsight-damage-detection"
ARCH = "efficientnet_b0"

# Known-good values, verified directly against this exact published
# checkpoint (see project history) — a generous tolerance below the
# measured confidence, not an exact-float match, so minor numerical
# drift across torch/hardware doesn't make this test flaky.
MIN_CONFIDENCE = 0.7


@pytest.fixture(scope="module")
def real_model():
    checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename=f"best_{ARCH}.pt")
    model = build_model(ARCH, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def _predict(model, image_path: str) -> tuple[str, float]:
    import cv2

    image_bgr = cv2.imread(image_path)
    image_rgb = preprocess(image_bgr)
    tensor = eval_transform(image=image_rgb)["image"].unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    pred_idx = int(probs.argmax())
    return ("00-damage" if pred_idx == 1 else "01-whole"), float(probs[pred_idx])


def test_real_checkpoint_correctly_classifies_damage_example(real_model):
    prediction, confidence = _predict(real_model, "samples/damage_example.jpg")
    assert prediction == "00-damage"
    assert confidence >= MIN_CONFIDENCE


def test_real_checkpoint_correctly_classifies_whole_example(real_model):
    prediction, confidence = _predict(real_model, "samples/whole_example.jpg")
    assert prediction == "01-whole"
    assert confidence >= MIN_CONFIDENCE


def test_full_api_with_real_downloaded_checkpoint(monkeypatch):
    """End-to-end: real HF Hub download, real weights, real API contract,
    including the Grad-CAM hook mechanics — the most complete check."""
    monkeypatch.setenv("CLAIMSIGHT_ARCH", ARCH)
    monkeypatch.setenv("CLAIMSIGHT_HF_REPO", HF_REPO)

    import importlib

    import app.main as main_module

    importlib.reload(main_module)  # re-run module-level model loading with the env vars above

    assert main_module.model is not None, "real checkpoint should have loaded via CLAIMSIGHT_HF_REPO"

    with TestClient(main_module.app) as client, open("samples/damage_example.jpg", "rb") as f:
        res = client.post("/predict-with-gradcam", files={"file": ("car.jpg", f, "image/jpeg")})

    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "00-damage"
    assert body["confidence"] >= MIN_CONFIDENCE
    assert body["gradcam_overlay_base64"]
