"""From-scratch neural-network engine.

Phase 1: scalar Python implementation of neurons, dense layers, forward and
backward propagation, losses, and optimizers. No deep-learning framework, and
no automatic differentiation.
"""

from .activations import Activation, ReLU, Sigmoid, Tanh, get_activation, sigmoid, softmax
from .layers import DenseLayer, Neuron, Parameter
from .losses import BinaryCrossEntropy, BinaryCrossEntropyWithLogits, MeanSquaredError
from .model import MLP
from .optimizers import SGD, Adam, Momentum
from .serialization import load_model, save_model

__version__ = "0.1.0"

__all__ = [
    "Activation",
    "Adam",
    "BinaryCrossEntropy",
    "BinaryCrossEntropyWithLogits",
    "DenseLayer",
    "MLP",
    "MeanSquaredError",
    "Momentum",
    "Neuron",
    "Parameter",
    "ReLU",
    "SGD",
    "Sigmoid",
    "Tanh",
    "get_activation",
    "load_model",
    "save_model",
    "sigmoid",
    "softmax",
]
