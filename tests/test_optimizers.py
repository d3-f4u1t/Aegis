import pytest

from nn.layers import Parameter
from nn.optimizers import SGD, Adam, Momentum


def test_sgd_single_step():
    p = Parameter(10.0)
    p.grad = 2.0
    SGD(lr=0.5).step([p])
    assert p.value == pytest.approx(9.0)


def test_sgd_converges_on_quadratic():
    p = Parameter(0.0)
    optimizer = SGD(lr=0.1)
    for _ in range(100):
        p.grad = 2.0 * (p.value - 3.0)
        optimizer.step([p])
    assert p.value == pytest.approx(3.0, abs=1e-6)


def test_momentum_two_steps():
    p = Parameter(0.0)
    optimizer = Momentum(lr=0.1, beta=0.9)
    p.grad = 1.0
    optimizer.step([p])
    assert p.value == pytest.approx(-0.1)
    p.grad = 1.0
    optimizer.step([p])
    assert p.value == pytest.approx(-0.1 - 0.19)


def test_adam_converges_on_quadratic():
    p = Parameter(0.0)
    optimizer = Adam(lr=0.1)
    for _ in range(200):
        p.grad = 2.0 * (p.value - 3.0)
        optimizer.step([p])
    assert p.value == pytest.approx(3.0, abs=1e-4)


def test_optimizer_does_not_change_gradients():
    p = Parameter(1.0)
    p.grad = 0.5
    Adam(lr=0.01).step([p])
    assert p.grad == pytest.approx(0.5)
