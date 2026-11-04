"""NumPy-vectorized numerical engine.

This package is a vectorized re-implementation of the scalar engine in ``nn/``.
The scalar engine is preserved unchanged as the audited mathematical reference.
This package exists to:

* reproduce the scalar engine's forward propagation, loss calculations and
  backpropagation using NumPy arrays,
* serve as a performance-optimized numerical kernel,
* be parity-tested against the scalar engine before any reliance on it.

Design contract (mirrors the scalar engine):

* A neuron / layer computes ``z = X @ W + b`` then ``a = activation(z)``.
* Gradients are *accumulated* into each ``Parameter``; only the optimizer
  mutates parameter values.
* The binary output layer emits logits; the sigmoid lives inside
  ``BinaryCrossEntropyWithLogits`` so the gradient simplifies to
  ``sigmoid(z) - y``.
* Mini-batch gradients are averaged over the batch (gradient of the *mean*
  batch loss).

Only ``numpy`` is used; no PyTorch, TensorFlow, Keras, JAX or automatic
differentiation. Every operation is hand-written.
"""

from __future__ import annotations

from .activations import (
    Activation,
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    get_activation,
    sigmoid,
    softmax,
)
from .layers import DenseLayer, Parameter, init_scale
from .losses import (
    BinaryCrossEntropy,
    BinaryCrossEntropyWithLogits,
    Loss,
    MeanSquaredError,
)
from .model import MLP
from .optimizers import SGD, Adam, Momentum, Optimizer
from .serialization import load_model, save_model

__version__ = "0.1.0"

__all__ = [
    "Activation",
    "Adam",
    "BinaryCrossEntropy",
    "BinaryCrossEntropyWithLogits",
    "DenseLayer",
    "Loss",
    "MLP",
    "MeanSquaredError",
    "Momentum",
    "Optimizer",
    "Parameter",
    "ReLU",
    "SGD",
    "Sigmoid",
    "Tanh",
    "get_activation",
    "init_scale",
    "load_model",
    "save_model",
    "sigmoid",
    "softmax",
]
