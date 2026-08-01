# ClaimSight API — deployable for free on Hugging Face Spaces (Docker SDK)
# or any container host. CPU-only by default (Spaces' free tier has no GPU).
#
# The model checkpoint is NOT baked into this image — it is downloaded
# from the Hugging Face Hub at container startup via CLAIMSIGHT_HF_REPO,
# so the image stays small and the model can be updated without a rebuild.
FROM python:3.11-slim

# opencv needs libgl/libglib at runtime even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
