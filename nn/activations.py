"""Activation functions, implemented explicitly.

Every activation exposes the same two operations:

    forward(z)  -> a        the activation value
    derivative(z) -> da/dz  the local derivative, evaluated at the *input* z

Backpropagation needs the derivative, not the output. Keeping the interface
input-based means every layer can apply the same chain rule:

    delta = grad_output * activation.derivative(z)

Nothing here delegates to a deep-learning framework; this is plain Python and
the standard library's ``math`` module.
"""

from __future__ import annotations

import math
from typing import Sequence


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid: 1 / (1 + exp(-z)).

    The naive formula overflows for very negative z (exp(-z) -> inf). Splitting
    on the sign of z keeps every exponential argument negative.
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class Activation:
    """Base class. Identity by default."""

    name = "identity"

    def forward(self, z: float) -> float:
        return z

    def derivative(self, z: float) -> float:
        return 1.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class Identity(Activation):
    name = "identity"


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z: float) -> float:
        return sigmoid(z)

    def derivative(self, z: float) -> float:
        s = sigmoid(z)
        return s * (1.0 - s)


class Tanh(Activation):
    name = "tanh"

    def forward(self, z: float) -> float:
        return math.tanh(z)

    def derivative(self, z: float) -> float:
        t = math.tanh(z)
        return 1.0 - t * t


class ReLU(Activation):
    name = "relu"

    def forward(self, z: float) -> float:
        return z if z > 0.0 else 0.0

    def derivative(self, z: float) -> float:
        # At z == 0 the function is not differentiable; we follow the common
        # convention of using the subgradient 0.
        return 1.0 if z > 0.0 else 0.0


_ACTIVATIONS = {
    "identity": Identity,
    "linear": Identity,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
}


def get_activation(activation: str | Activation) -> Activation:
    """Turn a name into an activation instance (or pass an instance through)."""
    if isinstance(activation, Activation):
        return activation
    try:
        return _ACTIVATIONS[activation.lower()]()
    except KeyError as exc:
        known = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"unknown activation {activation!r}; known: {known}") from exc


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a vector of logits.

    Subtracting the maximum logit leaves the result unchanged mathematically
    but prevents exp() from overflowing. Used with the multi-class
    cross-entropy loss, which arrives in a later phase.
    """
    if len(logits) == 0:
        raise ValueError("softmax requires at least one logit")
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = math.fsum(exps)
    return [e / total for e in exps]
