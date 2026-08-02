"""ClaimSight — live interactive demo (Streamlit Community Cloud).

Deployed for free at share.streamlit.io. This is a *second* front door
to the same model as app/main.py's FastAPI service — it reuses the
exact same preprocess(), build_model(), and Grad-CAM code, just with a
different UI, because Hugging Face Spaces' Docker/Gradio SDKs now
require a paid PRO plan on their free CPU tier and this project's
scope commits to zero deployment cost.

The FastAPI app (app/main.py) remains the tested, documented serving
contract — run it locally with `uvicorn app.main:app` (see README).
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import eval_transform
from src.model import build_model, get_target_layer
from src.preprocessing import preprocess

HF_REPO = os.environ.get("CLAIMSIGHT_HF_REPO", "PinheiroDataworks/claimsight-damage-detection")
ARCH = os.environ.get("CLAIMSIGHT_ARCH", "resnet50")
THRESHOLD = float(os.environ.get("CLAIMSIGHT_THRESHOLD", "0.5"))
CLASSES = {0: "01-whole", 1: "00-damage"}
DAMAGE_LABEL = 1

st.set_page_config(page_title="ClaimSight — Live Demo", page_icon="🚗", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #10141A; }
    .cs-badge { display:inline-block; font-family:monospace; font-size:14px; padding:6px 16px;
                border-radius:100px; font-weight:600; margin-bottom:10px; }
    .cs-badge.damage { background:rgba(255,162,60,0.15); color:#FFA23C; border:1px solid rgba(255,162,60,0.4); }
    .cs-badge.whole { background:rgba(63,214,176,0.15); color:#3FD6B0; border:1px solid rgba(63,214,176,0.4); }
    .cs-disclaimer { border:1px solid rgba(255,162,60,0.35); background:rgba(255,162,60,0.08);
                      border-radius:8px; padding:14px 18px; font-size:13px; color:#8993A4; margin-top:24px; }
    .cs-disclaimer b { color:#EAEEF4; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model from Hugging Face Hub…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = f"models/best_{ARCH}.pt"
    if os.path.exists(local_path):
        checkpoint_path = local_path
    else:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename=f"best_{ARCH}.pt")
    model = build_model(ARCH, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model, device


def predict_with_gradcam(model, device, image_bgr: np.ndarray):
    image_rgb = preprocess(image_bgr)  # same function used in training and the FastAPI app
    tensor = eval_transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    prob_damage = float(probs[DAMAGE_LABEL].item())
    pred_idx = DAMAGE_LABEL if prob_damage >= THRESHOLD else 0

    rgb_float = image_rgb.astype(np.float32) / 255.0
    target_layer = get_target_layer(model, ARCH)
    cam = GradCAM(model=model, target_layers=[target_layer])
    tensor.requires_grad_(True)  # required: a freshly loaded model has requires_grad=False everywhere
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return {
        "prediction": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "prob_damage": prob_damage,
        "prob_whole": float(probs[0].item()),
        "overlay": overlay,
        "preprocessed": image_rgb,
    }


st.markdown("### 🚗 Claim<span style='color:#FFA23C'>Sight</span> — live demo", unsafe_allow_html=True)
st.caption(
    f"Vehicle damage triage · ResNet50, 94.35% validation accuracy, ROC-AUC 0.986 · "
    f"model loaded from [huggingface.co/{HF_REPO}](https://huggingface.co/{HF_REPO})"
)

uploaded = st.file_uploader("Upload a vehicle photo (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode this file as an image. Please upload a JPEG or PNG.")
    else:
        model, device = load_model()
        with st.spinner("Running inference…"):
            result = predict_with_gradcam(model, device, image_bgr)

        badge_class = "damage" if result["prediction"] == "00-damage" else "whole"
        st.markdown(
            f'<span class="cs-badge {badge_class}">{result["prediction"]} '
            f'· confidence {result["confidence"]:.1%}</span>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("P(00-damage)", f"{result['prob_damage']:.1%}")
        col2.metric("P(01-whole)", f"{result['prob_whole']:.1%}")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with img_col2:
            st.image(result["overlay"], caption="Grad-CAM — region that most influenced the decision", use_container_width=True)

        st.caption(
            "Grad-CAM is a coarse activation heatmap (weak localization), not a pixel-level "
            "segmentation mask — see the project README for the full explainability discussion."
        )

st.markdown(
    """
    <div class="cs-disclaimer">
    <b>ClaimSight is a decision-support prototype for claims triage.</b>
    It does not issue, deny, or finalize any insurance claim. Every flagged prediction requires human review.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "[GitHub repo](https://github.com/pinheiro-dataworks/claim-sight) · "
    f"[Model card](https://huggingface.co/{HF_REPO}) · "
    "Tested FastAPI equivalent: `uvicorn app.main:app` (see README)"
)
