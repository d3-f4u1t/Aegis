import math

import pytest

from nn.activations import Identity, ReLU, Sigmoid, Tanh, sigmoid, softmax


def numeric_derivative(f, z: float, h: float = 1e-6) -> float:
    return (f(z + h) - f(z - h)) / (2.0 * h)


@pytest.mark.parametrize("activation", [Sigmoid(), Tanh()])
@pytest.mark.parametrize("z", [-3.0, -1.0, 0.0, 0.5, 2.0])
def test_derivative_matches_numeric(activation, z):
    expected = numeric_derivative(activation.forward, z)
    assert activation.derivative(z) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("z", [-1000.0, -50.0, 0.0, 50.0, 1000.0])
def test_sigmoid_is_stable_and_bounded(z):
    value = sigmoid(z)
    assert 0.0 <= value <= 1.0
    assert math.isfinite(value)


def test_sigmoid_known_values():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(2.0) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_relu_and_identity():
    relu = ReLU()
    assert relu.forward(-2.0) == 0.0
    assert relu.forward(3.0) == 3.0
    assert relu.derivative(-2.0) == 0.0
    assert relu.derivative(3.0) == 1.0
    assert Identity().forward(-2.0) == -2.0
    assert Identity().derivative(0.0) == 1.0


def test_softmax_is_a_distribution():
    probabilities = softmax([1.0, 2.0, 3.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_softmax_handles_large_logits():
    probabilities = softmax([1000.0, 1001.0, 1002.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in probabilities)


def test_unknown_activation_is_rejected():
    from nn.activations import get_activation

    with pytest.raises(ValueError, match="unknown activation"):
        get_activation("swish")
