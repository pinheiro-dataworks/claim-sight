"""ClaimSight serving layer — FastAPI.

The single most important line in this file is the import of
`preprocess` from `src.preprocessing`: it is the exact same function
used in training and evaluation, so there is no separate "serving-side"
reimplementation that could silently drift from what the model was
trained on (training/serving skew).

Endpoints:
    GET  /health               liveness + which architecture is loaded
    POST /predict               JSON prediction (multipart image upload)
    POST /predict-with-gradcam  same, plus a base64 Grad-CAM overlay
    GET  /                      minimal interactive HTML demo

Model source: a local checkpoint under models/, or — if
CLAIMSIGHT_HF_REPO is set — downloaded from the Hugging Face Hub on
startup via hf_hub_download, so the deployed API never needs a model
binary committed to git.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import GradCamPredictionResponse, HealthResponse, PredictionResponse
from src.dataset import eval_transform
from src.model import build_model, get_target_layer
from src.preprocessing import preprocess  # <- same function as training

ARCH = os.environ.get("CLAIMSIGHT_ARCH", "resnet50")
# Default decision threshold on P(damage). 0.5 is the statistically
# neutral default; src/evaluate.py documents a recall-priority
# alternative and the trade-off it implies — see outputs/metrics/.
THRESHOLD = float(os.environ.get("CLAIMSIGHT_THRESHOLD", "0.5"))
HF_REPO = os.environ.get("CLAIMSIGHT_HF_REPO")  # e.g. "username/claimsight-damage-detection"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
CLASSES = {0: "01-whole", 1: "00-damage"}
DAMAGE_LABEL = 1

app = FastAPI(
    title="ClaimSight API",
    description="Vehicle damage triage — binary classification with Grad-CAM explainability.",
    version="1.0.0",
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # populated in _load_model(), called at import time below


def _resolve_checkpoint_path() -> Path:
    local_path = Path(f"models/best_{ARCH}.pt")
    if local_path.exists():
        return local_path
    if HF_REPO:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=HF_REPO, filename=f"best_{ARCH}.pt")
        return Path(downloaded)
    raise FileNotFoundError(
        f"No checkpoint found at {local_path} and CLAIMSIGHT_HF_REPO is not set. "
        "Train a model first (python -m src.train --arch {ARCH}) or point "
        "CLAIMSIGHT_HF_REPO at a published Hugging Face model repo."
    )


def _load_model():
    global model
    checkpoint_path = _resolve_checkpoint_path()
    m = build_model(ARCH, pretrained=False)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.to(device).eval()
    model = m
    return m


try:
    _load_model()
except FileNotFoundError as exc:
    # Allow the module to import (useful for tests / OpenAPI schema
    # generation) even before a model has been trained; endpoints that
    # need the model will raise a clear 503 instead of failing at import.
    print(f"[claimsight] WARNING: model not loaded — {exc}")


def _require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. See server logs.")
    return model


def _decode_upload(raw: bytes) -> np.ndarray:
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image — file is not a valid JPEG/PNG.")
    return image_bgr


def _predict(image_bgr: np.ndarray) -> tuple[str, float, dict[str, float], torch.Tensor]:
    m = _require_model()
    image_rgb = preprocess(image_bgr)
    tensor = eval_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(m(tensor), dim=1).squeeze()
    prob_damage = float(probs[DAMAGE_LABEL].item())
    pred_idx = DAMAGE_LABEL if prob_damage >= THRESHOLD else 0
    prediction = CLASSES[pred_idx]
    confidence = round(float(probs[pred_idx].item()), 4)
    probabilities = {
        CLASSES[0]: round(float(probs[0].item()), 4),
        CLASSES[1]: round(float(probs[1].item()), 4),
    }
    return prediction, confidence, probabilities, tensor


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model=ARCH, device=str(device))


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}. Use image/jpeg or image/png.")
    raw = await file.read()
    image_bgr = _decode_upload(raw)
    prediction, confidence, probabilities, _ = _predict(image_bgr)
    return PredictionResponse(
        prediction=prediction, confidence=confidence,
        probabilities=probabilities, threshold_used=THRESHOLD,
    )


@app.post("/predict-with-gradcam", response_model=GradCamPredictionResponse)
async def predict_with_gradcam(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}. Use image/jpeg or image/png.")
    raw = await file.read()
    image_bgr = _decode_upload(raw)
    m = _require_model()

    prediction, confidence, probabilities, tensor = _predict(image_bgr)

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    image_rgb = preprocess(image_bgr)
    rgb_float = image_rgb.astype(np.float32) / 255.0
    pred_idx = 1 if prediction == "00-damage" else 0
    target_layer = get_target_layer(m, ARCH)
    cam = GradCAM(model=m, target_layers=[target_layer])
    # See src/explain.py for why this is required: a freshly loaded model
    # has every parameter's requires_grad reset by build_model(), so the
    # input tensor itself must require grad for Grad-CAM's backward pass
    # to have a graph to walk.
    tensor.requires_grad_(True)
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return GradCamPredictionResponse(
        prediction=prediction, confidence=confidence,
        probabilities=probabilities, threshold_used=THRESHOLD,
        gradcam_overlay_base64=overlay_b64,
    )


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return _DEMO_HTML


_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClaimSight — Live Demo</title>
<style>
  :root{--void:#10141A;--panel:#171D26;--line:#2A3140;--ink:#EAEEF4;--ink-muted:#8993A4;
        --amber:#FFA23C;--teal:#3FD6B0;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--void);color:var(--ink);font-family:-apple-system,Segoe UI,Inter,sans-serif;padding:0;}
  header{display:flex;align-items:center;gap:14px;padding:16px 28px;border-bottom:1px solid var(--line);background:var(--panel);}
  header img{height:34px;}
  header .name{font-weight:700;font-size:18px;letter-spacing:-0.01em;}
  header .name span{color:var(--amber);}
  header .tag{font-family:monospace;font-size:11px;color:var(--ink-muted);margin-left:6px;}
  main{max-width:760px;margin:0 auto;padding:36px 24px 60px;}
  h1{font-size:22px;margin:0 0 8px;}
  p.lead{color:var(--ink-muted);font-size:14.5px;margin:0 0 26px;}
  .dropzone{border:1.5px dashed var(--line);border-radius:10px;padding:28px;text-align:center;cursor:pointer;background:var(--panel);}
  .dropzone:hover{border-color:var(--amber);}
  input[type=file]{display:none;}
  button{font-family:monospace;font-size:13px;background:var(--amber);color:#1A1200;border:none;
         padding:11px 22px;border-radius:6px;font-weight:600;cursor:pointer;margin-top:16px;}
  button:disabled{opacity:0.5;cursor:not-allowed;}
  .result{margin-top:28px;display:none;}
  .result.show{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
  .result img{width:100%;border-radius:8px;border:1px solid var(--line);}
  .badge{display:inline-block;font-family:monospace;font-size:13px;padding:6px 14px;border-radius:100px;font-weight:600;margin-bottom:10px;}
  .badge.damage{background:rgba(255,162,60,0.15);color:var(--amber);border:1px solid rgba(255,162,60,0.4);}
  .badge.whole{background:rgba(63,214,176,0.15);color:var(--teal);border:1px solid rgba(63,214,176,0.4);}
  .bar-row{display:flex;align-items:center;gap:10px;font-family:monospace;font-size:12px;margin-top:6px;color:var(--ink-muted);}
  .bar-track{flex:1;height:7px;background:var(--line);border-radius:4px;overflow:hidden;}
  .bar-fill{height:100%;background:var(--amber);}
  .disclaimer{margin-top:34px;border:1px solid rgba(255,162,60,0.35);background:rgba(255,162,60,0.08);
              border-radius:8px;padding:14px 18px;font-size:12.5px;color:var(--ink-muted);}
  .disclaimer strong{color:var(--ink);}
  .footer-link{color:var(--ink-muted);font-size:12px;margin-top:24px;display:block;}
  .footer-link a{color:var(--teal);}
</style>
</head>
<body>
<header>
  <img src="/static/logo_renands_nav.png" alt="renan DS" onerror="this.style.display='none'">
  <div><span class="name">Claim<span>Sight</span></span><span class="tag">/ LIVE DEMO</span></div>
</header>
<main>
  <h1>Upload a vehicle photo</h1>
  <p class="lead">Runs the actual served model — same <code>preprocess()</code> pipeline used in training, plus a Grad-CAM overlay showing what the model looked at.</p>

  <label class="dropzone" id="dropzone">
    <input type="file" id="fileInput" accept="image/jpeg,image/png">
    <div id="dropzoneText">Click to choose a JPEG or PNG photo</div>
  </label>
  <br>
  <button id="analyzeBtn" disabled>Analyze</button>

  <div class="result" id="result">
    <div>
      <div id="badge" class="badge"></div>
      <div class="bar-row"><span style="width:70px">00-damage</span><div class="bar-track"><div class="bar-fill" id="barDamage"></div></div><span id="valDamage"></span></div>
      <div class="bar-row"><span style="width:70px">01-whole</span><div class="bar-track"><div class="bar-fill" id="barWhole" style="background:var(--teal)"></div></div><span id="valWhole"></span></div>
      <p style="font-size:12px;color:var(--ink-muted);margin-top:14px;">Original</p>
      <img id="origImg">
    </div>
    <div>
      <p style="font-size:12px;color:var(--ink-muted);margin-top:0;">Grad-CAM — region that most influenced the decision (weak localization, not a segmentation mask)</p>
      <img id="camImg">
    </div>
  </div>

  <div class="disclaimer">
    <strong>ClaimSight is a decision-support prototype for claims triage.</strong>
    It does not issue, deny, or finalize any insurance claim. Every flagged prediction requires human review.
  </div>
  <span class="footer-link">API contract: <a href="/docs">/docs</a> (Swagger) · <a href="/health">/health</a></span>
</main>
<script>
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const dropzoneText = document.getElementById('dropzoneText');
const analyzeBtn = document.getElementById('analyzeBtn');
const result = document.getElementById('result');
let selectedFile = null;

fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0];
  dropzoneText.textContent = selectedFile ? selectedFile.name : 'Click to choose a JPEG or PNG photo';
  analyzeBtn.disabled = !selectedFile;
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict-with-gradcam', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    const badge = document.getElementById('badge');
    badge.textContent = data.prediction + '  ·  confidence ' + (data.confidence * 100).toFixed(1) + '%';
    badge.className = 'badge ' + (data.prediction === '00-damage' ? 'damage' : 'whole');

    const pDamage = data.probabilities['00-damage'];
    const pWhole = data.probabilities['01-whole'];
    document.getElementById('barDamage').style.width = (pDamage * 100) + '%';
    document.getElementById('barWhole').style.width = (pWhole * 100) + '%';
    document.getElementById('valDamage').textContent = (pDamage * 100).toFixed(1) + '%';
    document.getElementById('valWhole').textContent = (pWhole * 100).toFixed(1) + '%';

    document.getElementById('origImg').src = URL.createObjectURL(selectedFile);
    document.getElementById('camImg').src = 'data:image/png;base64,' + data.gradcam_overlay_base64;

    result.classList.add('show');
  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
  }
});
</script>
</body>
</html>"""
