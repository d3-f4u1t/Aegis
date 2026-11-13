"""Logistic regression baseline, implemented from scratch.

A simple linear classifier:

    p = sigmoid(w · x + b)
    L = -mean(y * log(p) + (1 - y) * log(1 - p))

Gradients:

    dL/dw = (1/n) * X.T @ (p - y)
    dL/db = (1/n) * sum(p - y)

Trained with gradient descent. No deep-learning framework; only NumPy and the
standard library. This baseline exists to establish whether the from-scratch
MLP provides measurable value over a simple linear model.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class LogisticRegression:
    """Binary logistic regression trained by gradient descent."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000, seed: int = 0) -> None:
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_features: int | None = None
        self.feature_names: list[str] | None = None
        self.schema_version: str = "1"

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Sequence[str] | None = None,
        schema_version: str = "1",
    ) -> "LogisticRegression":
        """Fit logistic regression on training data.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Binary targets (0 or 1), shape (n_samples,).
            feature_names: Ordered feature names (for schema tracking).
            schema_version: Feature schema version.

        Returns:
            self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.seed)

        # Initialize weights small, bias zero.
        self.weights = rng.normal(0.0, 0.01, size=n_features)
        self.bias = 0.0
        self.n_features = n_features
        self.feature_names = list(feature_names) if feature_names else None
        self.schema_version = schema_version

        for _ in range(self.epochs):
            z = X_arr @ self.weights + self.bias
            p = self._sigmoid(z)

            # Gradients of mean BCE loss.
            error = p - y_arr
            grad_w = (X_arr.T @ error) / n_samples
            grad_b = float(np.mean(error))

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """Return P(class=1) for each sample."""
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X_arr.shape[1]} features but model expects {self.n_features}"
            )
        z = X_arr @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> dict:
        """Return model parameters as a dict (for checkpointing)."""
        if self.weights is None:
            raise ValueError("Model not fitted.")
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "n_features": int(self.n_features),
            "feature_names": self.feature_names,
            "schema_version": self.schema_version,
            "lr": self.lr,
            "epochs": self.epochs,
            "seed": self.seed,
        }

    @classmethod
    def from_params(cls, params: dict) -> "LogisticRegression":
        """Create a model from saved parameters."""
        model = cls(lr=params.get("lr", 0.01), epochs=params.get("epochs", 1000), seed=params.get("seed", 0))
        model.weights = np.array(params["weights"], dtype=float)
        model.bias = float(params["bias"])
        model.n_features = int(params["n_features"])
        model.feature_names = params.get("feature_names")
        model.schema_version = params.get("schema_version", "1")
        return model
