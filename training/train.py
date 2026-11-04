"""Training loop: mini-batches, shuffling, validation, and history.

The loop is intentionally explicit. Each batch does:

    zero_grad -> forward (per sample) -> loss -> backward (per sample) -> step

Gradients accumulate across the samples of a batch, so each sample's gradient
is divided by the batch size to produce the gradient of the mean batch loss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from nn.losses import Loss
from nn.model import MLP
from nn.optimizers import Optimizer


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def batch_loss(
    model: MLP, X: Sequence[Sequence[float]], y: Sequence[float], loss_fn: Loss
) -> float:
    """Mean loss over a dataset, without touching gradients."""
    if len(X) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    total = 0.0
    for inputs, target in zip(X, y):
        value, _ = loss_fn(target, model.forward(inputs))
        total += value
    return total / len(X)


def evaluate(
    model: MLP, X: Sequence[Sequence[float]], y: Sequence[float], loss_fn: Loss
) -> tuple[float, list[float]]:
    """Return (mean loss, raw model outputs) for a dataset."""
    if len(X) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    total = 0.0
    predictions: list[float] = []
    for inputs, target in zip(X, y):
        output = model.forward(inputs)
        value, _ = loss_fn(target, output)
        total += value
        predictions.append(output[0] if len(output) == 1 else output)
    return total / len(X), predictions


def train(
    model: MLP,
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    loss_fn: Loss,
    optimizer: Optimizer,
    epochs: int = 100,
    batch_size: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    X_val: Sequence[Sequence[float]] | None = None,
    y_val: Sequence[float] | None = None,
    verbose: bool = False,
    log_every: int = 0,
) -> History:
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

    rng = random.Random(seed)
    parameters = model.parameters()
    history = History()

    for epoch in range(epochs):
        order = list(range(n))
        if shuffle:
            rng.shuffle(order)
        total = 0.0
        for start in range(0, n, batch_size):
            indices = order[start : start + batch_size]
            current_batch = len(indices)
            model.zero_grad()
            for index in indices:
                output = model.forward(X[index])
                value, grad = loss_fn(y[index], output)
                total += value
                model.backward([g / current_batch for g in grad])
            optimizer.step(parameters)

        history.loss.append(total / n)
        if X_val is not None and y_val is not None:
            history.val_loss.append(batch_loss(model, X_val, y_val, loss_fn))

        should_log = verbose and (
            epoch == 0 or epoch == epochs - 1 or (log_every > 0 and (epoch + 1) % log_every == 0)
        )
        if should_log:
            message = f"epoch {epoch + 1:4d}/{epochs}  loss={history.loss[-1]:.6f}"
            if history.val_loss:
                message += f"  val_loss={history.val_loss[-1]:.6f}"
            print(message)

    return history
