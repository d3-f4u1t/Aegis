"""Model adapters package."""

from visualizer.backend.adapters.base import ModelAdapter
from visualizer.backend.adapters.baseline_adapter import (
    LogisticRegressionAdapter,
    ThresholdAdapter,
)
from visualizer.backend.adapters.numpy_adapter import NumpyMLPAdapter
from visualizer.backend.adapters.registry import ModelRegistry, registry
from visualizer.backend.adapters.scalar_adapter import ScalarMLPAdapter

__all__ = [
    "ModelAdapter",
    "ScalarMLPAdapter",
    "NumpyMLPAdapter",
    "LogisticRegressionAdapter",
    "ThresholdAdapter",
    "ModelRegistry",
    "registry",
]
