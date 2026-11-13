"""FastAPI router for inspecting curated traffic samples and anomaly failure cases."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException

from visualizer.backend.artifact_readers.dataset_reader import dataset_reader
from visualizer.backend.schemas.api_models import TrafficSampleItem

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])


@router.get("/samples", response_model=List[TrafficSampleItem])
def get_traffic_samples(scenario: Optional[str] = None) -> List[TrafficSampleItem]:
    """Retrieve curated real behavioural traffic windows, optionally filtered by scenario."""
    return dataset_reader.get_samples(scenario=scenario)


@router.get("/samples/{sample_id}", response_model=TrafficSampleItem)
def get_single_sample(sample_id: str) -> TrafficSampleItem:
    """Retrieve details and 13 features for a single sample."""
    sample = dataset_reader.get_sample_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
    return sample
