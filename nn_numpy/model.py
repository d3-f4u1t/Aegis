"""Vectorized multilayer perceptron.

Mirrors the scalar ``nn.model.MLP`` interface. ``sizes`` includes the input and
output dimensions, e.g. ``[8, 32, 16, 1]``. Hidden layers use ``hidden_activation``;
the final layer uses ``output_activation`` (defaults to identity so it emits
logits).

``forward`` accepts either a single sample (1D array of shape ``(n_inputs,)``)
or a batch (2D array of shape ``(batch, n_inputs)``). A single sample returns a
1D array; a batch returns a 2D array.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .activations import get_activation, sigmoid
from .layers import DenseLayer, Parameter


class MLP:
    """Fully connected feed-forward network built from NumPy dense layers."""

    def __init__(
        self,
        sizes: Sequence[int],
        hidden_activation: str = "relu",
        output_activation: str = "identity",
        seed: int = 0,
        init: str | None = None,
    ) -> None:
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least input and output dimensions")
        if any(int(n) <= 0 for n in sizes):
            raise ValueError("every layer size must be positive")

        self.sizes = [int(n) for n in sizes]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.seed = seed
        self.init = init

        # Note: np.random.RandomState produces a different sequence than the
        # scalar engine's random.Random for the same seed. Initialization is
        # reproducible *within* this engine, but not byte-identical to the
        # scalar engine. Parity tests copy scalar parameters into this model.
        rng = np.random.RandomState(seed)

        self.layers: list[DenseLayer] = []
        last = len(self.sizes) - 1
        for i in range(last):
            activation = output_activation if i == last - 1 else hidden_activation
            self.layers.append(
                DenseLayer(
                    self.sizes[i],
                    self.sizes[i + 1],
                    activation,
                    rng,
                    init=init,
                )
            )

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        if X.shape[1] != self.sizes[0]:
            raise ValueError(
                f"expected {self.sizes[0]} inputs per sample, got {X.shape[1]}"
            )
        values = X
        for layer in self.layers:
            values = layer.forward(values)
        if single:
            return values[0]
        return values

    def backward(self, grad_outputs: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad_outputs, dtype=float)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> Iterable[Parameter]:
        for layer in self.layers:
            yield from layer.parameters()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    @property
    def n_parameters(self) -> int:
        return sum(layer.n_parameters for layer in self.layers)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(class=1) for a network with a single logit output."""
        if self.sizes[-1] != 1:
            raise ValueError("predict_proba requires exactly one output")
        logits = self.forward(X)
        probs = sigmoid(logits)
        if np.asarray(X).ndim == 1:
            return probs[0]
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def describe(self) -> str:
        lines = [f"MLP(sizes={self.sizes}, params={self.n_parameters})"]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MLP(sizes={self.sizes}, params={self.n_parameters})"
