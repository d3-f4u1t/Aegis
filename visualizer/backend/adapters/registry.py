"""Model Registry: discovers, loads, and manages access to all trained model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from training.preprocessing import StandardScaler
from visualizer.backend.adapters.base import ModelAdapter
from visualizer.backend.adapters.baseline_adapter import (
    LogisticRegressionAdapter,
    ThresholdAdapter,
)
from visualizer.backend.adapters.numpy_adapter import NumpyMLPAdapter
from visualizer.backend.adapters.scalar_adapter import ScalarMLPAdapter
from visualizer.backend.schemas.api_models import ModelSummary


class ModelRegistry:
    """Central repository of all available model checkpoints and associated preprocessors."""

    def __init__(self, results_dir: str | Path | None = None) -> None:
        self.results_dir = Path(results_dir) if results_dir else Path("experiments/results")
        self._models: Dict[str, ModelAdapter] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._discover_and_load()

    def _discover_and_load(self) -> None:
        """Inspect results_dir and instantiate adapters for supported checkpoint formats."""
        if not self.results_dir.exists():
            return

        # First pass: find all preprocessing artifacts
        for p in self.results_dir.glob("*preprocessing.json"):
            try:
                scaler = StandardScaler.load(p)
                # Store by stem prefix, e.g. "phase5_exp1_indistribution"
                prefix = p.stem.replace("_preprocessing", "")
                self._scalers[prefix] = scaler
            except Exception:
                continue

        # Second pass: instantiate model adapters
        for json_path in sorted(self.results_dir.glob("*.json")):
            filename = json_path.name

            # Skip non-model files
            if filename.endswith("_preprocessing.json") or filename == "phase5_summary.json":
                continue

            model_id = json_path.stem
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)

                # Differentiate by JSON structure
                if "format_version" in raw:
                    fv = raw["format_version"]
                    engine = raw.get("engine", "scalar" if fv == 1 else "numpy")
                    if fv == 1 or engine == "scalar":
                        adapter = ScalarMLPAdapter(json_path, model_id=model_id)
                        self._models[model_id] = adapter
                    elif fv == 2 or engine == "numpy":
                        adapter = NumpyMLPAdapter(json_path, model_id=model_id)
                        self._models[model_id] = adapter
                elif "weights" in raw and "bias" in raw and "lr" in raw:
                    adapter = LogisticRegressionAdapter(json_path, model_id=model_id)
                    self._models[model_id] = adapter
                elif "feature_index" in raw and "threshold" in raw:
                    adapter = ThresholdAdapter(json_path, model_id=model_id)
                    self._models[model_id] = adapter
            except Exception as e:
                # Silently skip invalid or incompatible JSON files
                continue

    def list_models(self) -> List[ModelSummary]:
        """Return summaries of all loaded models."""
        return [adapter.get_summary() for adapter in self._models.values()]

    def get_model(self, model_id: str) -> Optional[ModelAdapter]:
        """Return the model adapter for the given ID, or None if not found."""
        return self._models.get(model_id)

    def get_scaler(self, model_id: str) -> Optional[StandardScaler]:
        """Find the matching StandardScaler for a model, if one exists."""
        # Try matching by prefix
        for prefix, scaler in self._scalers.items():
            if model_id.startswith(prefix):
                return scaler
        # Fallback to exp1 scaler if available
        return self._scalers.get("phase5_exp1_indistribution")

    def scale_features(self, model_id: str, raw_features: Sequence[float]) -> List[float]:
        """Scale raw features using the model's corresponding scaler if present."""
        scaler = self.get_scaler(model_id)
        if scaler and scaler.params:
            return scaler.transform([raw_features])[0]
        return list(raw_features)

    def get_default_model_id(self) -> str:
        """Return preferred default model for UI."""
        preferred = [
            "phase5_exp1_indistribution_numpy_mlp",
            "phase5_exp1_indistribution_scalar_mlp",
            "phase4_validation_mlp",
        ]
        for p in preferred:
            if p in self._models:
                return p
        if self._models:
            return next(iter(self._models.keys()))
        return "default"


# Global singleton registry
registry = ModelRegistry()
