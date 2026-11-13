"""Abstract base adapter for neural network and baseline models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

from visualizer.backend.schemas.api_models import (
    ForwardTraceResult,
    LayerParameters,
    ModelArchitecture,
    ModelSummary,
)


class ModelAdapter(ABC):
    """Unified interface wrapping scalar, numpy, and baseline model artifacts."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier of the loaded model checkpoint."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name."""
        ...

    @property
    @abstractmethod
    def engine(self) -> str:
        """Execution engine name: 'scalar', 'numpy', 'logistic_regression', 'threshold'."""
        ...

    @abstractmethod
    def get_summary(self) -> ModelSummary:
        """Returns brief summary for model listing."""
        ...

    @abstractmethod
    def get_architecture(self) -> ModelArchitecture:
        """Returns detailed model architecture, layer sizes, activations, parameters."""
        ...

    @abstractmethod
    def get_layer_parameters(self, layer_idx: int) -> LayerParameters:
        """Returns weight matrix and bias vector for a specific layer."""
        ...

    @abstractmethod
    def predict_proba(self, features: Sequence[float]) -> float:
        """Returns probability of attack P(attack | features)."""
        ...

    @abstractmethod
    def predict(self, features: Sequence[float], threshold: float = 0.5) -> int:
        """Returns binary classification (0=normal, 1=attack)."""
        ...

    @abstractmethod
    def trace_forward(
        self,
        scaled_features: Sequence[float],
        raw_features: Sequence[float] | None = None,
        threshold: float = 0.5,
    ) -> ForwardTraceResult:
        """Executes a fully observable step-by-step forward pass."""
        ...
