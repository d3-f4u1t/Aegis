"""Vectorized parameters and dense layers.

A dense layer computes:

    Z = X @ W + b        (pre-activation, shape ``(batch, n_outputs)``)
    A = activation(Z)    (activation output, same shape)

where ``W`` has shape ``(n_inputs, n_outputs)`` and ``b`` has shape
``(n_outputs,)``. This is the standard weight-matrix convention: column ``j`` of
``W`` holds the weights feeding output neuron ``j``.

A :class:`Parameter` wraps a NumPy array and its accumulated gradient. Gradients
are accumulated into ``Parameter.grad`` during backward passes and reset by
``zero_grad()``; only the optimizer mutates ``Parameter.value``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .activations import Activation, get_activation


class Parameter:
    """A learnable NumPy array plus its accumulated gradient."""

    __slots__ = ("value", "grad")

    def __init__(self, value: np.ndarray) -> None:
        self.value = np.array(value, dtype=float, copy=True)
        self.grad = np.zeros_like(self.value)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Parameter(shape={self.value.shape}, "
            f"value_norm={float(np.linalg.norm(self.value)):.6g}, "
            f"grad_norm={float(np.linalg.norm(self.grad)):.6g})"
        )


def init_scale(scheme: str, fan_in: int, fan_out: int) -> float:
    """Standard deviation for weight initialisation.

    Mirrors the scalar engine exactly:
    he:     sqrt(2 / fan_in)        recommended for ReLU
    xavier: sqrt(2 / (fan_in + fan_out)) recommended for sigmoid/tanh
    small:  0.01
    """
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    if scheme == "he":
        return math.sqrt(2.0 / fan_in)
    if scheme == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if scheme == "small":
        return 0.01
    raise ValueError(f"unknown init scheme {scheme!r}")


class DenseLayer:
    """A fully connected layer: every input feeds every output neuron.

    Forward:

        Z = X @ W + b
        A = activation(Z)

    Backward, given ``grad_outputs = dL/dA`` of shape ``(batch, n_outputs)``
    where ``grad_outputs`` is the gradient of the *mean* batch loss:

        delta = grad_outputs * activation'(Z)                    (batch, n_outputs)
        dL/dW = X.T @ delta                                      (n_inputs, n_outputs)
        dL/db = sum(delta, axis=0)                               (n_outputs,)
        dL/dX = delta @ W.T                                      (batch, n_inputs)

    This matches the scalar engine, which divides each per-sample gradient by
    the batch size before backward; the net effect is the mean-batch gradient.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: str | Activation,
        rng: np.random.RandomState,
        init: str | None = None,
    ) -> None:
        if n_inputs <= 0 or n_outputs <= 0:
            raise ValueError("layer dimensions must be positive")
        act = get_activation(activation)
        if init is None:
            init = "he" if act.name == "relu" else "xavier"
        scale = init_scale(init, fan_in=n_inputs, fan_out=n_outputs)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = act
        self.init = init
        self.scale = scale

        # Weight matrix: (n_inputs, n_outputs). Bias vector: (n_outputs,).
        self.W = Parameter(rng.standard_normal((n_inputs, n_outputs)) * scale)
        self.b = Parameter(np.zeros(n_outputs, dtype=float))

        # Cached for backpropagation.
        self._X: np.ndarray | None = None
        self._Z: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_inputs:
            raise ValueError(
                f"expected {self.n_inputs} inputs per sample, got {X.shape[1]}"
            )
        self._X = X
        self._Z = X @ self.W.value + self.b.value
        return self.activation.forward(self._Z)

    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        """Return ``dL/dX`` and accumulate ``dL/dW`` and ``dL/db``."""
        if self._X is None or self._Z is None:
            raise RuntimeError("backward called before forward")
        grad_outputs = np.asarray(grad_outputs, dtype=float)
        if grad_outputs.shape != self._Z.shape:
            raise ValueError(
                f"expected gradient shape {self._Z.shape}, got {grad_outputs.shape}"
            )

        batch = self._X.shape[0]
        delta = grad_outputs * self.activation.derivative(self._Z)

        # ``grad_outputs`` is already the gradient of the *mean* batch loss
        # (dL_mean/dA), as produced by the loss functions. The scalar engine
        # divides each per-sample gradient by the batch size before backward;
        # the net effect is identical, so we must NOT divide again here.
        self.W.grad = self._X.T @ delta
        self.b.grad = delta.sum(axis=0)

        grad_input = delta @ self.W.value.T
        return grad_input

    def parameters(self) -> Iterable[Parameter]:
        yield self.W
        yield self.b

    @property
    def n_parameters(self) -> int:
        return self.n_inputs * self.n_outputs + self.n_outputs

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DenseLayer({self.n_inputs} -> {self.n_outputs}, "
            f"activation={self.activation.name}, init={self.init})"
        )
