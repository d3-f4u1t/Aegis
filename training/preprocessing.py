"""Preprocessing pipeline.

Fits normalization statistics on training data only and applies them unchanged
during inference. Statistics travel with the model artifact alongside the
feature schema version.

This module is used by both the offline training pipeline and the live
inference pipeline. The same statistics must be applied in both contexts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from telemetry.feature_extractor import FULL_FEATURE_ORDER, n_full_features


@dataclass
class PreprocessingParams:
    """Normalization parameters fitted on training data."""

    feature_names: list[str] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    schema_version: str = "1"

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "means": [float(m) for m in self.means],
            "stds": [float(s) for s in self.stds],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessingParams":
        return cls(
            feature_names=list(data.get("feature_names", [])),
            means=[float(m) for m in data.get("means", [])],
            stds=[float(s) for s in data.get("stds", [])],
            schema_version=str(data.get("schema_version", "1")),
        )


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    Fitted on training data only. The same mean and std are applied at
    inference. Features with zero variance are left unscaled (std set to 1.0)
    to avoid division by zero.
    """

    def __init__(self) -> None:
        self.params: PreprocessingParams | None = None

    def fit(self, X: Sequence[Sequence[float]], feature_names: Sequence[str], schema_version: str = "1") -> PreprocessingParams:
        """Fit normalization statistics on training data.

        Args:
            X: Training feature vectors, shape (n_samples, n_features).
            feature_names: Ordered list of feature names, must match X columns.
            schema_version: Version of the feature schema used.

        Returns:
            The fitted PreprocessingParams.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X_arr.shape[1]} columns but {len(feature_names)} feature names provided"
            )

        means = np.mean(X_arr, axis=0).tolist()
        stds = np.std(X_arr, axis=0).tolist()
        # Avoid division by zero for constant features
        stds = [s if s > 0.0 else 1.0 for s in stds]

        self.params = PreprocessingParams(
            feature_names=list(feature_names),
            means=means,
            stds=stds,
            schema_version=schema_version,
        )
        return self.params

    def transform(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        """Apply fitted normalization to feature vectors.

        Raises ValueError if the scaler has not been fitted.
        """
        if self.params is None:
            raise ValueError("StandardScaler has not been fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        means = np.array(self.params.means, dtype=float)
        stds = np.array(self.params.stds, dtype=float)

        if X_arr.shape[1] != means.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[1]} columns but scaler was fitted on {means.shape[0]} features"
            )

        standardized = (X_arr - means) / stds
        return standardized.tolist()

    def fit_transform(
        self, X: Sequence[Sequence[float]], feature_names: Sequence[str], schema_version: str = "1"
    ) -> list[list[float]]:
        """Fit on X then transform X. Convenience for training."""
        self.fit(X, feature_names, schema_version)
        return self.transform(X)

    def save(self, path: str | Path) -> Path:
        """Save preprocessing parameters to a JSON file."""
        if self.params is None:
            raise ValueError("Nothing to save: scaler has not been fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.params.to_dict(), fh, indent=2)
            fh.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        """Load preprocessing parameters from a JSON file."""
        path = Path(path)
        scaler = cls()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        scaler.params = PreprocessingParams.from_dict(data)
        return scaler

    def schema_matches(self, schema_version: str, feature_names: Sequence[str]) -> bool:
        """Check whether the fitted schema matches the expected schema."""
        if self.params is None:
            return False
        return (
            self.params.schema_version == schema_version
            and list(self.params.feature_names) == list(feature_names)
        )
