"""Vectorized loss functions and their derivatives.

Contract: ``loss(y_true, y_pred) -> (loss_value, grad_wrt_prediction)``.

``y_pred`` is the model's raw output (logits or probabilities depending on the
loss). Gradients are averaged over the batch, so they represent
``d(mean_batch_loss)/d(prediction)`` and can be passed directly to
``MLP.backward``.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from .activations import sigmoid


class Loss:
    name = "loss"

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


def _as_batch(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y


class MeanSquaredError(Loss):
    """Mean of squared differences over the batch and outputs.

    L = mean((P - T)^2)
    dL/dP = 2 * (P - T) / (batch * n_outputs)
    """

    name = "mse"

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, np.ndarray]:
        y_true = _as_batch(y_true)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        diff = y_pred - y_true
        loss = float(np.mean(diff * diff))
        grad = 2.0 * diff / diff.size
        return loss, grad


class BinaryCrossEntropyWithLogits(Loss):
    """Stable binary cross-entropy for a single logit per sample.

    L = mean(max(z, 0) - z*y + log(1 + exp(-|z|)))
    dL/dz = (sigmoid(z) - y) / batch

    The algebraically rearranged form keeps the loss finite for any logit, and
    the gradient avoids multiplying a huge number by a vanishing sigmoid
    derivative.
    """

    name = "bce_with_logits"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = _as_batch(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        if y_pred.shape[1] != 1:
            raise ValueError("binary cross-entropy expects exactly one logit")
        if np.any((y_true < 0.0) | (y_true > 1.0)):
            raise ValueError("binary targets must be in [0, 1]")

        z = y_pred
        y = y_true
        batch = z.shape[0]

        loss = float(
            np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))
        )
        grad = (sigmoid(z) - y) / batch
        return loss, grad


class BinaryCrossEntropy(Loss):
    """Textbook BCE applied to probabilities. Kept for comparison/teaching.

    L = mean(-[y log p + (1-y) log(1-p)])
    dL/dp = (p - y) / (p * (1 - p)) / batch

    The gradient is w.r.t. probability ``p``, *not* w.r.t. logit ``z``. Pairing
    this loss with a raw logit output without an extra ``sigmoid'(z)`` factor
    produces wrong gradients. Use ``BinaryCrossEntropyWithLogits`` for logit
    outputs.
    """

    name = "bce"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = _as_batch(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        if y_pred.shape[1] != 1:
            raise ValueError("binary cross-entropy expects exactly one probability")
        if np.any((y_true < 0.0) | (y_true > 1.0)):
            raise ValueError("binary targets must be in [0, 1]")

        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        y = y_true
        batch = p.shape[0]

        loss = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
        grad = (p - y) / (p * (1.0 - p)) / batch
        return loss, grad
