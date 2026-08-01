"""API contract tests — health check, valid prediction, and the two
documented error paths (bad content-type, corrupted bytes).

Requires a trained checkpoint at models/best_<CLAIMSIGHT_ARCH>.pt
(defaults to resnet50). Run `python -m src.train` first if this file
is skipped/failing with a 503.
"""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

_ARCH = os.environ.get("CLAIMSIGHT_ARCH", "resnet50")
_CHECKPOINT = Path(f"models/best_{_ARCH}.pt")

pytestmark = pytest.mark.skipif(
    not _CHECKPOINT.exists(),
    reason=f"No trained checkpoint at {_CHECKPOINT} — run `python -m src.train` first.",
)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _fake_jpeg_bytes(size=(300, 200), color=(120, 60, 40)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model"] in {"resnet50", "efficientnet_b0"}


def test_predict_valid_image_returns_full_schema(client):
    files = {"file": ("car.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    res = client.post("/predict", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] in {"00-damage", "01-whole"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"00-damage", "01-whole"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert "threshold_used" in body


def test_predict_rejects_unsupported_content_type(client):
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    res = client.post("/predict", files=files)
    assert res.status_code == 415


def test_predict_rejects_corrupted_image_bytes(client):
    files = {"file": ("broken.jpg", b"not a real jpeg", "image/jpeg")}
    res = client.post("/predict", files=files)
    assert res.status_code == 422


def test_predict_with_gradcam_returns_overlay(client):
    files = {"file": ("car.png", _fake_jpeg_bytes(), "image/png")}
    res = client.post("/predict-with-gradcam", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["gradcam_overlay_base64"]
    assert body["prediction"] in {"00-damage", "01-whole"}


def test_demo_page_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "ClaimSight" in res.text
