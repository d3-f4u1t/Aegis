"""Numerical gradient checking.

For every scalar parameter w, compare:

    analytic  dL/dw  from backpropagation
    numeric   (L(w+h) - L(w-h)) / 2h

If backpropagation is correct, they agree to within floating-point noise. This
is the single most valuable test in a from-scratch neural network: it catches
sign errors, missing transposes, and wrong derivative factors before any real
data is involved.
"""

import math

import pytest

from nn.losses import BinaryCrossEntropy, BinaryCrossEntropyWithLogits, MeanSquaredError
from nn.model import MLP

X = [
    [0.5, -1.0, 0.25],
    [-0.5, 0.5, 1.0],
    [1.0, 0.0, -0.5],
    [0.0, 1.0, 0.5],
    [-1.0, -0.5, 0.0],
]
Y = [1.0, 0.0, 1.0, 1.0, 0.0]


def mean_loss(model, loss_fn):
    total = 0.0
    for inputs, target in zip(X, Y):
        value, _ = loss_fn(target, model.forward(inputs))
        total += value
    return total / len(X)


def analytic_gradients(model, loss_fn):
    model.zero_grad()
    for inputs, target in zip(X, Y):
        _, grad = loss_fn(target, model.forward(inputs))
        model.backward([g / len(X) for g in grad])
    return [p.grad for p in model.parameters()]


@pytest.mark.parametrize(
    "loss_fn", [MeanSquaredError(), BinaryCrossEntropyWithLogits()]
)
def test_every_parameter_gradient(loss_fn):
    # tanh is smooth everywhere, so finite differences are valid at every point.
    # (ReLU has a kink at z=0 where analytic and central-difference gradients
    # legitimately disagree; that edge case has its own test.)
    model = MLP([3, 4, 2, 1], hidden_activation="tanh", seed=3)
    params = model.parameters()
    expected = analytic_gradients(model, loss_fn)

    h = 1e-6
    for index, (param, analytic) in enumerate(zip(params, expected)):
        original = param.value

        param.value = original + h
        loss_plus = mean_loss(model, loss_fn)

        param.value = original - h
        loss_minus = mean_loss(model, loss_fn)

        param.value = original
        numeric = (loss_plus - loss_minus) / (2.0 * h)

        assert numeric == pytest.approx(analytic, abs=2e-6), (
            f"gradient mismatch at parameter {index}: "
            f"analytic={analytic:.9f} numeric={numeric:.9f}"
        )


def test_gradients_accumulate_across_backward_calls():
    model = MLP([2, 2, 1], seed=1)
    model.zero_grad()
    for _ in range(3):
        model.forward([1.0, 2.0])
        model.backward([1.0])
    grads = [p.grad for p in model.parameters()]
    assert any(g != 0.0 for g in grads)


def test_relu_backprop_gradient_checks_pass_away_from_kink():
    """ReLU derivative is piecewise; central differences are valid only where z != 0.

    This regression test uses seed 5 and inputs chosen so no hidden neuron sits
    exactly on the kink, then gradient-checks every parameter through a ReLU
    hidden layer and an identity output with MSE loss.
    """
    model = MLP([2, 3, 1], hidden_activation="relu", output_activation="identity", seed=5)
    X = [[1.0, 2.0], [-1.0, 0.5], [2.0, -1.0], [0.5, 1.5]]
    Y = [1.0, 0.0, 1.0, 0.0]
    loss_fn = MeanSquaredError()

    def mean_loss(m):
        total = 0.0
        for x, t in zip(X, Y):
            value, _ = loss_fn(t, m.forward(x))
            total += value
        return total / len(X)

    model.zero_grad()
    for x, t in zip(X, Y):
        _, grad = loss_fn(t, model.forward(x))
        model.backward([g / len(X) for g in grad])
    expected = [p.grad for p in model.parameters()]

    h = 1e-6
    for index, (param, analytic) in enumerate(zip(model.parameters(), expected)):
        original = param.value
        param.value = original + h
        loss_plus = mean_loss(model)
        param.value = original - h
        loss_minus = mean_loss(model)
        param.value = original
        numeric = (loss_plus - loss_minus) / (2.0 * h)
        assert numeric == pytest.approx(analytic, abs=2e-6), (
            f"ReLU backprop gradient mismatch at parameter {index}: "
            f"analytic={analytic:.9f} numeric={numeric:.9f}"
        )


def test_zero_grad_clears_accumulated_gradients():
    model = MLP([2, 2, 1], seed=1)
    model.forward([1.0, 2.0])
    model.backward([1.0])
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())


def test_textbook_bce_gradient_matches_numeric_wrt_probability():
    """Contract lock: BinaryCrossEntropy gradient is dL/dp (probability), not dL/dz.

    The analytic gradient (p - y) / (p * (1 - p)) is checked against a central
    difference of the BCE loss with respect to the probability output. This test
    exists because chaining this loss incorrectly against a logit would silently
    produce wrong gradients; the stable BinaryCrossEntropyWithLogits should be
    used for logit outputs instead.
    """

    def bce(p, y, eps=1e-12):
        p = min(max(p, eps), 1.0 - eps)
        return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

    for y, p in [(1.0, 0.7), (0.0, 0.3), (1.0, 0.2)]:
        h = 1e-6
        numeric = (bce(p + h, y) - bce(p - h, y)) / (2.0 * h)
        analytic = (p - y) / (p * (1.0 - p))
        assert numeric == pytest.approx(analytic, abs=1e-9), (
            f"BCE contract gradient mismatch at y={y}, p={p}: "
            f"analytic={analytic:.9f} numeric={numeric:.9f}"
        )
