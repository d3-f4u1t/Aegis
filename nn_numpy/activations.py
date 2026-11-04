"""Vectorized activation functions and their derivatives.

Each activation exposes:

    forward(z)  -> a        activation value, same shape as z
    derivative(z) -> da/dz  local derivative evaluated at z, same shape as z

All operations are element-wise NumPy array operations. ``z`` may be any shape;
the derivative is applied element-wise, which is correct because these
activations are pointwise nonlinearities.

The derivative interface is intentionally input-based (evaluated at ``z``, not
at the output ``a``) so backpropagation applies the same chain rule everywhere:

    delta = grad_output * activation.derivative(z)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid applied element-wise.

    Splits on the sign of ``z`` so every exponential argument is non-positive,
    preventing overflow for very negative ``z``. For very large positive ``z``,
    ``exp(-z)`` underflows to 0 and the result correctly rounds to 1.0.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class Activation:
    """Base class: identity activation."""

    name = "identity"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(z, dtype=float))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class Identity(Activation):
    name = "identity"


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        s = sigmoid(z)
        return s * (1.0 - s)


class Tanh(Activation):
    name = "tanh"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        t = np.tanh(z)
        return 1.0 - t * t


class ReLU(Activation):
    name = "relu"

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # At z == 0 the function is not differentiable; we follow the common
        # convention of using the subgradient 0. Matches the scalar engine.
        return (z > 0.0).astype(float)


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


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax applied along the last axis.

    Subtracts the per-row maximum before exponenting so the computation does not
    overflow for large logits. Used with the multi-class cross-entropy loss,
    which arrives in a later phase.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        raise ValueError("softmax requires at least one logit")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)
