"""FastAPI router for step-by-step forward tracing and multi-model comparison."""

from __future__ import annotations

import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visualizer.backend.adapters.registry import registry
from visualizer.backend.artifact_readers.dataset_reader import dataset_reader
from visualizer.backend.schemas.api_models import (
    ComparisonResponse,
    ForwardTraceResult,
    ModelComparisonItem,
    TraceRequest,
)

router = APIRouter(prefix="/api/v1", tags=["inference"])


class MultiModelCompareRequest(BaseModel):
    features: List[float]
    is_scaled: bool = False
    sample_id: Optional[str] = None
    ground_truth_label: Optional[int] = None
    threshold: float = 0.5


@router.post("/models/{model_id}/trace", response_model=ForwardTraceResult)
def trace_model_forward(model_id: str, request: TraceRequest) -> ForwardTraceResult:
    """Execute fully observable step-by-step forward pass through the network."""
    adapter = registry.get_model(model_id)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    if request.is_scaled:
        scaled_features = request.features
        raw_features = None
    else:
        raw_features = request.features
        scaled_features = registry.scale_features(model_id, raw_features)

    try:
        return adapter.trace_forward(
            scaled_features=scaled_features,
            raw_features=raw_features,
            threshold=request.threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Forward trace error: {e}")


@router.post("/compare", response_model=ComparisonResponse)
def compare_models(request: MultiModelCompareRequest) -> ComparisonResponse:
    """Run all primary Phase 5 model families on the same sample to compare predictions and latency."""
    compare_ids = [
        ("phase5_exp1_indistribution_threshold", "Threshold Baseline (Rate Rule)"),
        ("phase5_exp1_indistribution_logistic_regression", "Logistic Regression"),
        ("phase5_exp1_indistribution_scalar_mlp", "Scalar Reference MLP [13, 16, 8, 1]"),
        ("phase5_exp1_indistribution_numpy_mlp", "NumPy Vectorized MLP [13, 16, 8, 1]"),
    ]

    # Resolve ground truth if sample_id provided
    ground_truth = request.ground_truth_label
    if request.sample_id and ground_truth is None:
        sample = dataset_reader.get_sample_by_id(request.sample_id)
        if sample:
            ground_truth = sample.ground_truth_label

    results: List[ModelComparisonItem] = []
    base_pred: Optional[int] = None

    for m_id, display_name in compare_ids:
        adapter = registry.get_model(m_id)
        if not adapter:
            continue

        if request.is_scaled:
            scaled_f = request.features
        else:
            scaled_f = registry.scale_features(m_id, request.features)

        t0 = time.perf_counter()
        prob = adapter.predict_proba(scaled_f)
        pred = adapter.predict(scaled_f, threshold=request.threshold)
        lat = (time.perf_counter() - t0) * 1000.0

        if base_pred is None and adapter.engine == "numpy":
            base_pred = pred

        is_correct = (pred == ground_truth) if ground_truth is not None else True

        results.append(
            ModelComparisonItem(
                model_id=m_id,
                model_name=display_name,
                engine=adapter.engine,
                prediction=pred,
                probability=float(prob),
                threshold=request.threshold,
                is_correct=is_correct,
                latency_ms=lat,
                agreement=True,  # will update below
            )
        )

    # Compute agreement relative to the NumPy MLP reference
    ref_pred = next((r.prediction for r in results if "numpy" in r.engine), None)
    if ref_pred is not None:
        for r in results:
            r.agreement = (r.prediction == ref_pred)

    return ComparisonResponse(
        sample_id=request.sample_id,
        ground_truth_label=ground_truth,
        results=results,
    )
