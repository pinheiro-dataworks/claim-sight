"""Pydantic request/response models — the API's explicit contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="'00-damage' or '01-whole'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    threshold_used: float = Field(..., description="Decision threshold applied to P(damage)")


class GradCamPredictionResponse(PredictionResponse):
    gradcam_overlay_base64: str = Field(
        ..., description="PNG overlay, base64-encoded, showing the region that most influenced the decision"
    )
