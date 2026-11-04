"""Loss functions and their derivatives.

Contract: ``loss(y_true, y_pred) -> (loss_value, grad_wrt_prediction)``.

The gradient is with respect to the model's *raw output* (logits). Putting the
sigmoid inside the loss is what keeps binary cross-entropy numerically stable
when a logit becomes very large in magnitude: the combined gradient is simply
``sigmoid(z) - y`` rather than a huge number multiplied by a vanishing
sigmoid derivative.
"""

from __future__ import annotations

import math
from typing import Sequence

from .activations import sigmoid


class Loss:
    name = "loss"

    def __call__(
        self, y_true: float, y_pred: Sequence[float]
    ) -> tuple[float, list[float]]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class MeanSquaredError(Loss):
    """Mean of squared differences. Useful for regression and for gradient tests."""

    name = "mse"

    def __call__(
        self, y_true: float | Sequence[float], y_pred: Sequence[float]
    ) -> tuple[float, list[float]]:
        targets = (
            [float(y_true)]
            if isinstance(y_true, (int, float))
            else [float(v) for v in y_true]
        )
        if len(targets) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        n = len(targets)
        diffs = [p - t for t, p in zip(targets, y_pred)]
        loss = math.fsum(d * d for d in diffs) / n
        grad = [2.0 * d / n for d in diffs]
        return loss, grad


class BinaryCrossEntropyWithLogits(Loss):
    """Stable binary cross-entropy for a single logit.

    loss(z, y) = max(z, 0) - z*y + log(1 + exp(-|z|))

    The first two terms are the algebraically rearranged form of
    -[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))], and the last term is the
    softplus remainder that keeps it finite for any z. The gradient reduces to
    ``sigmoid(z) - y``.
    """

    name = "bce_with_logits"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def __call__(
        self, y_true: float, y_pred: Sequence[float]
    ) -> tuple[float, list[float]]:
        if len(y_pred) != 1:
            raise ValueError("binary cross-entropy expects exactly one logit")
        y = float(y_true)
        if not (0.0 <= y <= 1.0):
            raise ValueError("binary targets must be in [0, 1]")
        z = float(y_pred[0])
        # max(z,0) - z*y = max(-z,0) when y=0, etc. This form is stable.
        loss = max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))
        grad = sigmoid(z) - y
        return loss, [grad]


class BinaryCrossEntropy(Loss):
    """Textbook BCE applied to probabilities. Kept for comparison/teaching.

    Less stable than BinaryCrossEntropyWithLogits near 0 and 1; prefer that one
    for training. Provided so the difference can be measured directly.
    """

    name = "bce"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def __call__(
        self, y_true: float, y_pred: Sequence[float]
    ) -> tuple[float, list[float]]:
        if len(y_pred) != 1:
            raise ValueError("binary cross-entropy expects exactly one probability")
        y = float(y_true)
        p = min(max(float(y_pred[0]), self.eps), 1.0 - self.eps)
        loss = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        grad = (p - y) / (p * (1.0 - p))
        return loss, [grad]
