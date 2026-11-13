"""Main FastAPI application entry point for the Neural Network Visualization Laboratory."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from visualizer.backend.adapters.registry import registry
from visualizer.backend.api.experiments_router import router as experiments_router
from visualizer.backend.api.inference_router import router as inference_router
from visualizer.backend.api.models_router import router as models_router
from visualizer.backend.api.telemetry_router import router as telemetry_router

app = FastAPI(
    title="Neural Network Visualization Laboratory",
    description="Interactive academic visualizer for from-scratch neural network website traffic anomaly detection.",
    version="1.0.0",
)

# Enable CORS for local Vite dev server and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(models_router)
app.include_router(inference_router)
app.include_router(experiments_router)
app.include_router(telemetry_router)


@app.get("/api/v1/health", tags=["system"])
def health_check() -> dict:
    """System health check and loaded model count."""
    return {
        "status": "healthy",
        "models_count": len(registry.list_models()),
        "default_model": registry.get_default_model_id(),
        "version": "1.0.0",
    }


# Serve built frontend if dist directory exists
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("visualizer.backend.main:app", host="127.0.0.1", port=8000, reload=True)
