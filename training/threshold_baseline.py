"""Threshold-based detection baseline.

A simple rule-based detector that flags a window as anomalous when a single
feature exceeds a fixed threshold. For example:

    requests_per_second > rate_threshold
    status_4xx_ratio > error_ratio_threshold

This baseline exists to establish whether the from-scratch MLP and logistic
regression provide measurable value over simple, interpretable rules.

No machine learning is involved; the threshold is chosen on validation data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from training.evaluate import operating_point


class ThresholdBaseline:
    """Single-feature threshold detector.

    Flags a sample as anomalous when ``feature_value >= threshold``.

    The threshold is selected on validation data using
    :func:`training.evaluate.operating_point`.
    """

    def __init__(self, feature_index: int, threshold: float = 0.0) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.feature_name: str | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Sequence[str] | None = None,
        max_fpr: float = 0.05,
    ) -> "ThresholdBaseline":
        """Select the threshold on validation data.

        Searches candidate thresholds from the observed feature values and
        picks the one that maximizes recall subject to FPR <= max_fpr.

        Args:
            X: Validation features.
            y: Validation labels.
            feature_names: Optional feature names for schema tracking.
            max_fpr: Maximum acceptable false-positive rate.

        Returns:
            self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        feature_values = X_arr[:, self.feature_index]
        scores = feature_values.tolist()  # higher = more anomalous

        op = operating_point(list(y_arr), scores, max_fpr=max_fpr)
        self.threshold = float(op["threshold"])
        self.feature_name = (
            feature_names[self.feature_index] if feature_names else None
        )
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Return binary predictions (1 = anomalous)."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return (X_arr[:, self.feature_index] >= self.threshold).astype(int).tolist()

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return the raw feature value as a pseudo-probability (not calibrated)."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr[:, self.feature_index].tolist()

    def get_params(self) -> dict:
        return {
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "feature_name": self.feature_name,
        }

    @classmethod
    def from_params(cls, params: dict) -> "ThresholdBaseline":
        model = cls(
            feature_index=int(params["feature_index"]),
            threshold=float(params["threshold"]),
        )
        model.feature_name = params.get("feature_name")
        return model
