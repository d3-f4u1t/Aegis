"""Numerical gradient checking.

For every scalar parameter w, compare:

    analytic  dL/dw  from backpropagation
    numeric   (L(w+h) - L(w-h)) / 2h

If backpropagation is correct, they agree to within floating-point noise. This
is the single most valuable test in a from-scratch neural network: it catches
sign errors, missing transposes, and wrong derivative factors before any real
data is involved.
"""

import pytest

from nn.losses import BinaryCrossEntropyWithLogits, MeanSquaredError
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


def test_zero_grad_clears_accumulated_gradients():
    model = MLP([2, 2, 1], seed=1)
    model.forward([1.0, 2.0])
    model.backward([1.0])
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())
