import random

import pytest

from nn.activations import Identity
from nn.layers import Neuron
from nn.model import MLP


def test_single_neuron_forward_and_backward_by_hand():
    neuron = Neuron(1, Identity(), rng=random.Random(0), init="small")
    # Force exact parameters so the arithmetic is checkable on paper.
    neuron.weights[0].value = 2.0
    neuron.bias.value = 1.0

    output = neuron.forward([3.0])  # z = 1 + 2*3 = 7
    assert output == pytest.approx(7.0)

    grad_input = neuron.backward(0.5)  # dL/dz = 0.5
    assert neuron.weights[0].grad == pytest.approx(1.5)  # 0.5 * x = 1.5
    assert neuron.bias.grad == pytest.approx(0.5)
    assert grad_input[0] == pytest.approx(1.0)  # 0.5 * w = 1.0


def test_hand_built_network_output():
    model = MLP(
        [2, 2, 1],
        hidden_activation="identity",
        output_activation="identity",
        seed=0,
    )
    first, second = model.layers[0].neurons
    first.weights[0].value, first.weights[1].value = 1.0, 0.0
    first.bias.value = 0.0
    second.weights[0].value, second.weights[1].value = 0.0, 1.0
    second.bias.value = 0.0

    output_neuron = model.layers[1].neurons[0]
    output_neuron.weights[0].value = 1.0
    output_neuron.weights[1].value = 1.0
    output_neuron.bias.value = -1.0

    assert model.forward([2.0, 3.0]) == [pytest.approx(4.0)]  # 2 + 3 - 1


def test_parameter_count():
    model = MLP([2, 3, 1], seed=0)
    # 3 neurons x (2 weights + 1 bias) + 1 neuron x (3 weights + 1 bias)
    assert model.n_parameters == 13
    assert len(model.parameters()) == 13


def test_shapes_are_checked():
    model = MLP([2, 3, 1], seed=0)
    with pytest.raises(ValueError, match="expected 2 inputs"):
        model.forward([1.0])
    with pytest.raises(ValueError, match="expected 3 output gradients"):
        model.layers[0].backward([1.0])


def test_backward_requires_forward():
    model = MLP([2, 2, 1], seed=0)
    with pytest.raises(RuntimeError, match="before forward"):
        model.layers[0].backward([1.0, 1.0])
