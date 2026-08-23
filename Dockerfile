# ClaimSight API — containerized FastAPI service. CPU-only by default.
#
# Note: Hugging Face Spaces' Docker/Gradio SDKs now require a paid PRO
# plan on the free CPU tier (they didn't when this was written), so the
# live public demo is deployed on Streamlit Community Cloud instead
# (streamlit_app.py) — free, no card required. This Dockerfile still
# builds and runs the exact same tested API; use it on any container
# host you have free/paid capacity on (Cloud Run, Fly.io, your own
# infra, or an HF Space if you have PRO).
#
# The model checkpoint is NOT baked into this image — it is downloaded
# from the Hugging Face Hub at container startup via CLAIMSIGHT_HF_REPO,
# so the image stays small and the model can be updated without a rebuild.
FROM python:3.11-slim

# opencv-python-headless (see requirements.txt) still wants libglib at
# runtime on minimal base images, even without any GUI functions used.
# Package name depends on the base image's Debian release: newer ones
# (trixie+) renamed it to the "t64" (64-bit time_t) variant and only
# keep a broken transitional stub under the old name — try the new
# name first, fall back to the old one for older bases.
RUN apt-get update && \
    (apt-get install -y --no-install-recommends libglib2.0-0t64 \
     || apt-get install -y --no-install-recommends libglib2.0-0) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app

# Hugging Face Spaces expects the app to listen on 7860.
ENV CLAIMSIGHT_ARCH=resnet50 \
    PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
