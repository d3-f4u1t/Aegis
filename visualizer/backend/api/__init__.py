"""API routers package."""

from visualizer.backend.api.experiments_router import router as experiments_router
from visualizer.backend.api.inference_router import router as inference_router
from visualizer.backend.api.models_router import router as models_router
from visualizer.backend.api.telemetry_router import router as telemetry_router

__all__ = [
    "models_router",
    "inference_router",
    "experiments_router",
    "telemetry_router",
]
