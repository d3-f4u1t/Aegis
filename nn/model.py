"""A small multilayer perceptron built from the layers in this package.

Design note: for binary classification the network emits a single *logit*
(pre-activation value). The sigmoid lives inside the loss function, which is
numerically stable and lets the gradient simplify to ``p - y``. This is the
same convention modern frameworks use (BCEWithLogits). ``predict_proba``
applies the sigmoid for inference.

For multi-class problems the output layer will emit one logit per class and a
softmax cross-entropy loss will consume them directly.
"""

from __future__ import annotations

import random
from typing import Sequence

from .activations import get_activation, sigmoid
from .layers import DenseLayer, Parameter


class MLP:
    """Fully connected feed-forward network.

    ``sizes`` includes the input and output dimensions, e.g. [8, 32, 16, 1].
    The hidden activation applies to every hidden layer; the output activation
    defaults to identity (logits).
    """

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

        rng = random.Random(seed)
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

    def forward(self, inputs: Sequence[float]) -> list[float]:
        values = [float(x) for x in inputs]
        if len(values) != self.sizes[0]:
            raise ValueError(
                f"expected {self.sizes[0]} inputs, got {len(values)}"
            )
        for layer in self.layers:
            values = layer.forward(values)
        return values

    def backward(self, grad_outputs: Sequence[float]) -> list[float]:
        grad = list(grad_outputs)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    @property
    def n_parameters(self) -> int:
        return sum(layer.n_parameters for layer in self.layers)

    def predict_proba(self, inputs: Sequence[float]) -> float:
        """P(class=1) for a network whose output layer is a single logit."""
        if self.sizes[-1] != 1:
            raise ValueError("predict_proba requires exactly one output")
        return sigmoid(self.forward(inputs)[0])

    def predict(self, inputs: Sequence[float], threshold: float = 0.5) -> int:
        return int(self.predict_proba(inputs) >= threshold)

    def describe(self) -> str:
        lines = [f"MLP(sizes={self.sizes}, params={self.n_parameters})"]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer!r}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MLP(sizes={self.sizes}, params={self.n_parameters})"
