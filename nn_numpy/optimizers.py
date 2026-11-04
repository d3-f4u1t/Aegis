"""Vectorized optimizers.

Each optimizer consumes a list of :class:`nn_numpy.layers.Parameter` objects,
reads their gradients, and updates their values. Gradients are read, never
written here. Optimizer state is keyed by ``id(parameter)`` because the
Parameter objects are stable for the lifetime of the model.

Semantics mirror the scalar engine's optimizers exactly.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

from .layers import Parameter


class Optimizer:
    def step(self, parameters: Iterable[Parameter]) -> None:
        raise NotImplementedError

    def zero_grad(self, parameters: Iterable[Parameter]) -> None:
        for p in parameters:
            p.zero_grad()


class SGD(Optimizer):
    """Plain gradient descent: w <- w - lr * grad_w."""

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr

    def step(self, parameters: Iterable[Parameter]) -> None:
        for p in parameters:
            p.value -= self.lr * p.grad


class Momentum(Optimizer):
    """Heavy-ball momentum: v <- beta*v + grad; w <- w - lr*v."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        self.lr = lr
        self.beta = beta
        self.velocity: dict[int, np.ndarray] = {}

    def step(self, parameters: Iterable[Parameter]) -> None:
        for p in parameters:
            key = id(p)
            v = self.beta * self.velocity.get(key, np.zeros_like(p.value)) + p.grad
            self.velocity[key] = v
            p.value -= self.lr * v


class Adam(Optimizer):
    """Adaptive moment estimation with bias correction.

    m <- b1*m + (1-b1)*g
    v <- b2*v + (1-b2)*g^2
    m_hat <- m / (1-b1^t),  v_hat <- v / (1-b2^t)
    w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, parameters: Iterable[Parameter]) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for p in parameters:
            key = id(p)
            g = p.grad
            m = b1 * self.m.get(key, np.zeros_like(g)) + (1.0 - b1) * g
            v = b2 * self.v.get(key, np.zeros_like(g)) + (1.0 - b2) * g * g
            self.m[key] = m
            self.v[key] = v
            m_hat = m / bc1
            v_hat = v / bc2
            p.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
