"""FastAPI router for experimental evaluations and scenario performance breakdown."""

from __future__ import annotations

from typing import Any, List
from fastapi import APIRouter, HTTPException

from visualizer.backend.artifact_readers.experiment_reader import experiment_reader
from visualizer.backend.schemas.api_models import ExperimentReport

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.get("", response_model=List[dict])
def list_experiments() -> List[dict]:
    """List summary cards for all recorded Phase 5 experiments."""
    return experiment_reader.list_experiments()


@router.get("/{experiment_id}", response_model=ExperimentReport)
def get_experiment_report(experiment_id: str) -> ExperimentReport:
    """Retrieve full metrics, confusion matrix, PR-AUC, and scenario breakdown."""
    report = experiment_reader.get_experiment(experiment_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_id}' not found. Available: {[e['experiment_id'] for e in experiment_reader.list_experiments()]}",
        )
    return report
