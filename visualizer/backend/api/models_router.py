"""FastAPI router for inspecting models, architectures, and layer parameters."""

from __future__ import annotations

from typing import List
from fastapi import APIRouter, HTTPException

from visualizer.backend.adapters.registry import registry
from visualizer.backend.schemas.api_models import (
    LayerParameters,
    ModelArchitecture,
    ModelSummary,
)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=List[ModelSummary])
def list_models() -> List[ModelSummary]:
    """List all available loaded model checkpoints."""
    return registry.list_models()


@router.get("/{model_id}", response_model=ModelArchitecture)
def get_model_architecture(model_id: str) -> ModelArchitecture:
    """Get detailed architecture, layer dimensions, activation functions, and metadata."""
    adapter = registry.get_model(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return adapter.get_architecture()


@router.get("/{model_id}/layers/{layer_idx}", response_model=LayerParameters)
def get_layer_parameters(model_id: str, layer_idx: int) -> LayerParameters:
    """Get weight matrix and bias vector for a specific layer."""
    adapter = registry.get_model(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    try:
        return adapter.get_layer_parameters(layer_idx)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
