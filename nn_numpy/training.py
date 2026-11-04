"""Vectorized training loop: mini-batches, shuffling, validation, history.

Mirrors ``training.train`` from the scalar engine. Each batch does:

    zero_grad -> forward(batch) -> loss(batch) -> backward(batch) -> step

The backward gradient is already averaged over the batch, so a single
``optimizer.step`` applies the gradient of the mean batch loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from nn_numpy.losses import Loss
from nn_numpy.model import MLP
from nn_numpy.optimizers import Optimizer


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def batch_loss(
    model: MLP, X: np.ndarray, y: np.ndarray, loss_fn: Loss
) -> float:
    """Mean loss over a dataset without touching gradients."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    total = 0.0
    for i in range(len(X)):
        value, _ = loss_fn(y[i : i + 1], model.forward(X[i : i + 1]))
        total += value
    return total / len(X)


def evaluate(
    model: MLP, X: np.ndarray, y: np.ndarray, loss_fn: Loss
) -> tuple[float, np.ndarray]:
    """Return (mean loss, raw model outputs) for a dataset."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    total = 0.0
    outputs = []
    for i in range(len(X)):
        output = model.forward(X[i : i + 1])
        value, _ = loss_fn(y[i : i + 1], output)
        total += value
        outputs.append(output)
    return total / len(X), np.vstack(outputs)


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Loss,
    optimizer: Optimizer,
    epochs: int = 100,
    batch_size: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    verbose: bool = False,
    log_every: int = 0,
) -> History:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) == 0:
        raise ValueError("cannot train on an empty dataset")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must be provided together")

    n = len(X)
    if batch_size is None:
        batch_size = n
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.RandomState(seed)
    parameters = list(model.parameters())
    history = History()

    for epoch in range(epochs):
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            Xb = X[idx]
            yb = y[idx]
            current_batch = len(idx)

            model.zero_grad()
            output = model.forward(Xb)
            value, grad = loss_fn(yb, output)
            total += value * current_batch
            model.backward(grad)
            optimizer.step(parameters)

        history.loss.append(total / n)
        if X_val is not None and y_val is not None:
            history.val_loss.append(batch_loss(model, X_val, y_val, loss_fn))

        should_log = verbose and (
            epoch == 0
            or epoch == epochs - 1
            or (log_every > 0 and (epoch + 1) % log_every == 0)
        )
        if should_log:
            message = f"epoch {epoch + 1:4d}/{epochs}  loss={history.loss[-1]:.6f}"
            if history.val_loss:
                message += f"  val_loss={history.val_loss[-1]:.6f}"
            print(message)

    return history
