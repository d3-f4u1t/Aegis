"""Neurons and dense layers, built by hand.

A neuron computes:

    z = b + sum_i(w_i * x_i)
    a = activation(z)

The forward pass caches ``x`` and ``z`` so the backward pass can apply the
chain rule without recomputing anything.

Gradients are *accumulated* into each Parameter, never applied inside
``backward``. The optimizer is the only component that changes parameter
values. This separation is what makes mini-batch training (several backward
passes before one update) correct.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .activations import Activation, get_activation


class Parameter:
    """A single learnable scalar plus the gradient of the loss w.r.t. it.

    Scalar parameters keep the first implementation easy to inspect and
    gradient-check. When NumPy is introduced, the same interface can hold an
    array instead of a float.
    """

    __slots__ = ("value", "grad")

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.grad = 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Parameter(value={self.value:.6g}, grad={self.grad:.6g})"


def init_scale(scheme: str, fan_in: int, fan_out: int) -> float:
    """Standard deviation for weight initialisation.

    he:     sqrt(2 / fan_in)        recommended for ReLU
    xavier: sqrt(2 / (fan_in+fan_out)) recommended for sigmoid/tanh
    small:  0.01                    a deliberately conservative fallback
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


class Neuron:
    """One artificial neuron with an activation function."""

    def __init__(
        self,
        n_inputs: int,
        activation: Activation,
        rng: random.Random,
        init: str = "he",
    ) -> None:
        if n_inputs <= 0:
            raise ValueError("a neuron needs at least one input")
        scale = init_scale(init, fan_in=n_inputs, fan_out=1)
        self.activation = activation
        self.weights = [Parameter(rng.gauss(0.0, scale)) for _ in range(n_inputs)]
        self.bias = Parameter(0.0)
        self._inputs: list[float] | None = None
        self._z: float | None = None

    def forward(self, inputs: Sequence[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} inputs, got {len(inputs)}"
            )
        z = self.bias.value
        for w, x in zip(self.weights, inputs):
            z += w.value * x
        self._inputs = [float(x) for x in inputs]
        self._z = z
        return self.activation.forward(z)

    def backward(self, grad_output: float) -> list[float]:
        """Accumulate parameter gradients and return d(loss)/d(inputs).

        ``grad_output`` is d(loss)/d(activation output) for this neuron.
        """
        if self._inputs is None or self._z is None:
            raise RuntimeError("backward called before forward")
        delta = grad_output * self.activation.derivative(self._z)
        grads = [0.0] * len(self.weights)
        for i, (w, x) in enumerate(zip(self.weights, self._inputs)):
            w.grad += delta * x
            grads[i] = delta * w.value
        self.bias.grad += delta
        return grads

    def parameters(self) -> Iterable[Parameter]:
        yield from self.weights
        yield self.bias


class DenseLayer:
    """A fully connected layer: every input feeds every neuron."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: str | Activation,
        rng: random.Random,
        init: str | None = None,
    ) -> None:
        if n_outputs <= 0:
            raise ValueError("a layer needs at least one output")
        act = get_activation(activation)
        if init is None:
            init = "he" if act.name == "relu" else "xavier"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = act
        self.neurons = [
            Neuron(n_inputs, act, rng, init=init) for _ in range(n_outputs)
        ]

    def forward(self, inputs: Sequence[float]) -> list[float]:
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, grad_outputs: Sequence[float]) -> list[float]:
        if len(grad_outputs) != self.n_outputs:
            raise ValueError(
                f"expected {self.n_outputs} output gradients, got {len(grad_outputs)}"
            )
        grad_inputs = [0.0] * self.n_inputs
        for neuron, grad_out in zip(self.neurons, grad_outputs):
            for i, g in enumerate(neuron.backward(grad_out)):
                grad_inputs[i] += g
        return grad_inputs

    def parameters(self) -> Iterable[Parameter]:
        for neuron in self.neurons:
            yield from neuron.parameters()

    @property
    def n_parameters(self) -> int:
        return self.n_outputs * (self.n_inputs + 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DenseLayer({self.n_inputs} -> {self.n_outputs}, "
            f"activation={self.activation.name})"
        )
