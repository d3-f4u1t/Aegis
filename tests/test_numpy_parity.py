"""Scalar-versus-NumPy parity tests.

For identical inputs, parameters and configurations, the NumPy engine must
reproduce the audited scalar engine:

* forward outputs
* loss values
* gradients for every weight and bias
* optimizer parameter updates
* training results on small deterministic datasets

Tolerances reflect legitimate floating-point rounding from different operation
ordering (scalar per-sample loops vs NumPy batched matmuls); we never demand
exact equality.

Strategy: build a scalar model and a NumPy model with the same architecture,
then copy the scalar engine's parameter values into the NumPy model so the two
start from identical parameters. (The two engines use different RNG streams, so
the same seed does *not* produce identical initial values; initialization
reproducibility is tested separately within each engine.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nn.losses import BinaryCrossEntropyWithLogits as ScalarBCELogits
from nn.losses import MeanSquaredError as ScalarMSE
from nn.model import MLP as ScalarMLP
from nn.optimizers import Adam as ScalarAdam
from nn.optimizers import Momentum as ScalarMomentum
from nn.optimizers import SGD as ScalarSGD
from training.train import train as scalar_train

from nn_numpy.losses import BinaryCrossEntropyWithLogits as NumpyBCELogits
from nn_numpy.losses import MeanSquaredError as NumpyMSE
from nn_numpy.model import MLP as NumpyMLP
from nn_numpy.optimizers import Adam as NumpyAdam
from nn_numpy.optimizers import Momentum as NumpyMomentum
from nn_numpy.optimizers import SGD as NumpySGD
from nn_numpy.training import train as numpy_train


# ---------------------------------------------------------------------------
# Helpers: mapping between scalar (per-neuron) and NumPy (matrix) layouts.
# ---------------------------------------------------------------------------

def scalar_weight_matrix(layer) -> np.ndarray:
    """Reassemble a scalar dense layer's weights into an (n_inputs, n_outputs) matrix."""
    n_in, n_out = layer.n_inputs, layer.n_outputs
    W = np.zeros((n_in, n_out), dtype=float)
    b = np.zeros(n_out, dtype=float)
    for j, neuron in enumerate(layer.neurons):
        for i, w in enumerate(neuron.weights):
            W[i, j] = w.value
        b[j] = neuron.bias.value
    return W, b


def scalar_grad_matrices(model) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-layer (dL/dW, dL/db) matrices reconstructed from scalar gradients."""
    mats = []
    for layer in model.layers:
        W, b = scalar_weight_matrix(layer)
        Wg = np.zeros_like(W)
        bg = np.zeros_like(b)
        for j, neuron in enumerate(layer.neurons):
            for i, w in enumerate(neuron.weights):
                Wg[i, j] = w.grad
            bg[j] = neuron.bias.grad
        mats.append((Wg, bg))
    return mats


def numpy_grad_matrices(model) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(layer.W.grad.copy(), layer.b.grad.copy()) for layer in model.layers]


def copy_scalar_to_numpy(scalar_model, numpy_model) -> None:
    if len(scalar_model.layers) != len(numpy_model.layers):
        raise ValueError("layer count mismatch")
    for sl, nl in zip(scalar_model.layers, numpy_model.layers):
        W, b = scalar_weight_matrix(sl)
        if W.shape != nl.W.value.shape or b.shape != nl.b.value.shape:
            raise ValueError("parameter shape mismatch")
        nl.W.value = W.copy()
        nl.b.value = b.copy()


def make_pair(sizes, hidden, output, seed):
    sm = ScalarMLP(sizes, hidden_activation=hidden, output_activation=output, seed=seed)
    nm = NumpyMLP(sizes, hidden_activation=hidden, output_activation=output, seed=seed)
    copy_scalar_to_numpy(sm, nm)
    return sm, nm


ARCHITECTURES = [
    ([2, 3, 1], "relu", "identity"),
    ([3, 4, 2, 1], "tanh", "identity"),
    ([4, 5, 3, 1], "sigmoid", "identity"),
    ([2, 8, 1], "relu", "identity"),
    ([5, 6, 4, 1], "tanh", "identity"),
]

LOSS_PAIRS = [
    (ScalarMSE(), NumpyMSE(), "mse"),
    (ScalarBCELogits(), NumpyBCELogits(), "bce_with_logits"),
]

OPTIMIZER_PAIRS = [
    # Classes (not shared instances) so each test gets a fresh optimizer with
    # no lingering Adam/momentum state keyed by parameter id().
    (ScalarSGD, NumpySGD, {"lr": 0.01}, "sgd"),
    (ScalarMomentum, NumpyMomentum, {"lr": 0.01, "beta": 0.9}, "momentum"),
    (ScalarAdam, NumpyAdam, {"lr": 0.01}, "adam"),
]


# ---------------------------------------------------------------------------
# Forward propagation parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes,hidden,output", ARCHITECTURES)
def test_forward_parity(sizes, hidden, output):
    sm, nm = make_pair(sizes, hidden, output, seed=11)
    rng = np.random.RandomState(3)
    X = rng.randn(6, sizes[0])

    scalar_outs = np.array([np.asarray(sm.forward(list(x))) for x in X])
    numpy_outs = nm.forward(X)

    assert numpy_outs.shape == scalar_outs.shape
    assert np.allclose(scalar_outs, numpy_outs, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("sizes,hidden,output", ARCHITECTURES)
def test_forward_parity_single_sample(sizes, hidden, output):
    sm, nm = make_pair(sizes, hidden, output, seed=11)
    rng = np.random.RandomState(7)
    x = rng.randn(sizes[0])

    scalar_out = np.asarray(sm.forward(list(x)))
    numpy_out = nm.forward(x)  # 1D input -> 1D output

    assert numpy_out.shape == scalar_out.shape
    assert np.allclose(scalar_out, numpy_out, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Loss value parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes,hidden,output", ARCHITECTURES)
@pytest.mark.parametrize("loss_s,loss_n,name", LOSS_PAIRS)
def test_loss_parity(sizes, hidden, output, loss_s, loss_n, name):
    sm, nm = make_pair(sizes, hidden, output, seed=11)
    rng = np.random.RandomState(5)
    X = rng.randn(6, sizes[0])
    if name == "bce_with_logits":
        Y = (rng.rand(6, sizes[-1]) > 0.5).astype(float)
    else:
        Y = rng.randn(6, sizes[-1])

    scalar_outs = [np.asarray(sm.forward(list(x))) for x in X]
    scalar_loss = float(
        np.mean(
            [
                loss_s(float(y[0]) if sizes[-1] == 1 else y, out)[0]
                for y, out in zip(Y, scalar_outs)
            ]
        )
    )
    numpy_loss, _ = loss_n(Y, nm.forward(X))

    assert abs(scalar_loss - numpy_loss) < 1e-9


# ---------------------------------------------------------------------------
# Gradient parity: every weight and bias
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes,hidden,output", ARCHITECTURES)
@pytest.mark.parametrize("loss_s,loss_n,name", LOSS_PAIRS)
def test_gradient_parity_every_parameter(sizes, hidden, output, loss_s, loss_n, name):
    sm, nm = make_pair(sizes, hidden, output, seed=11)
    rng = np.random.RandomState(5)
    X = rng.randn(6, sizes[0])
    if name == "bce_with_logits":
        Y = (rng.rand(6, sizes[-1]) > 0.5).astype(float)
    else:
        Y = rng.randn(6, sizes[-1])

    batch = len(X)

    # Scalar: per-sample forward/loss/backward, gradient divided by batch size.
    sm.zero_grad()
    for x, y in zip(X, Y):
        out = sm.forward(list(x))
        _, grad = loss_s(float(y[0]) if sizes[-1] == 1 else y, out)
        sm.backward([g / batch for g in grad])
    s_mats = scalar_grad_matrices(sm)

    # NumPy: single batched forward/loss/backward (gradient already averaged).
    nm.zero_grad()
    out = nm.forward(X)
    _, grad = loss_n(Y, out)
    nm.backward(grad)
    n_mats = numpy_grad_matrices(nm)

    assert len(s_mats) == len(n_mats)
    for (sw, sb), (nw, nb) in zip(s_mats, n_mats):
        assert sw.shape == nw.shape and sb.shape == nb.shape
        assert np.allclose(sw, nw, rtol=1e-6, atol=1e-8), (
            f"weight gradient mismatch ({name})"
        )
        assert np.allclose(sb, nb, rtol=1e-6, atol=1e-8), (
            f"bias gradient mismatch ({name})"
        )


# ---------------------------------------------------------------------------
# Optimizer update parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes,hidden,output", ARCHITECTURES[:2])
@pytest.mark.parametrize("loss_s,loss_n,name", LOSS_PAIRS)
@pytest.mark.parametrize("opt_s_cls,opt_n_cls,opt_kwargs,oname", OPTIMIZER_PAIRS)
def test_optimizer_update_parity(
    sizes, hidden, output, loss_s, loss_n, name, opt_s_cls, opt_n_cls, opt_kwargs, oname
):
    sm, nm = make_pair(sizes, hidden, output, seed=11)
    rng = np.random.RandomState(5)
    X = rng.randn(4, sizes[0])
    if name == "bce_with_logits":
        Y = (rng.rand(4, sizes[-1]) > 0.5).astype(float)
    else:
        Y = rng.randn(4, sizes[-1])

    batch = len(X)

    # Fresh optimizer instances per test (no shared id()-keyed state).
    opt_s = opt_s_cls(**opt_kwargs)
    opt_n = opt_n_cls(**opt_kwargs)

    # One scalar training step (one batch).
    sm.zero_grad()
    for x, y in zip(X, Y):
        out = sm.forward(list(x))
        _, grad = loss_s(float(y[0]) if sizes[-1] == 1 else y, out)
        sm.backward([g / batch for g in grad])
    opt_s.step(sm.parameters())

    # One NumPy training step (one batch).
    nm.zero_grad()
    out = nm.forward(X)
    _, grad = loss_n(Y, out)
    nm.backward(grad)
    opt_n.step(nm.parameters())

    for sl, nl in zip(sm.layers, nm.layers):
        sw, sb = scalar_weight_matrix(sl)
        assert np.allclose(sw, nl.W.value, rtol=1e-6, atol=1e-8), (
            f"weight update mismatch ({oname}, {name})"
        )
        assert np.allclose(sb, nl.b.value, rtol=1e-6, atol=1e-8), (
            f"bias update mismatch ({oname}, {name})"
        )


# ---------------------------------------------------------------------------
# Multi-epoch training parity (deterministic, no shuffling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes,hidden,output", ARCHITECTURES[:3])
@pytest.mark.parametrize("loss_s,loss_n,name", LOSS_PAIRS)
def test_training_parity(sizes, hidden, output, loss_s, loss_n, name):
    sm, nm = make_pair(sizes, hidden, output, seed=11)
    rng = np.random.RandomState(5)
    X = rng.randn(12, sizes[0])
    if name == "bce_with_logits":
        Y = (rng.rand(12, sizes[-1]) > 0.5).astype(float)
    else:
        Y = rng.randn(12, sizes[-1])

    X_list = [list(x) for x in X]
    Y_list = [float(y[0]) if sizes[-1] == 1 else list(y) for y in Y]

    # Fresh optimizer instances per test.
    opt_s = ScalarSGD(lr=0.01)
    opt_n = NumpySGD(lr=0.01)

    # Scalar engine: deterministic order (shuffle=False).
    scalar_train(
        sm,
        X_list,
        Y_list,
        loss_s,
        opt_s,
        epochs=25,
        batch_size=4,
        shuffle=False,
        seed=1,
    )

    # NumPy engine: deterministic order (shuffle=False).
    numpy_train(
        nm,
        X,
        Y,
        loss_n,
        opt_n,
        epochs=25,
        batch_size=4,
        shuffle=False,
        seed=1,
    )

    # Loss histories should track closely (rounding apart).
    # Recompute scalar history the same way numpy does for a fair compare.
    # Here we compare final parameters and the final loss value.
    scalar_final_loss = batch_loss(sm, X_list, Y_list, loss_s)
    numpy_final_loss, _ = evaluate(nm, X, Y, loss_n)
    assert abs(scalar_final_loss - numpy_final_loss) < 1e-6, (
        f"final loss mismatch ({oname}, {name}): {scalar_final_loss} vs {numpy_final_loss}"
    )

    for sl, nl in zip(sm.layers, nm.layers):
        sw, sb = scalar_weight_matrix(sl)
        assert np.allclose(sw, nl.W.value, rtol=1e-5, atol=1e-7), (
            f"trained weight mismatch ({oname}, {name})"
        )
        assert np.allclose(sb, nl.b.value, rtol=1e-5, atol=1e-7), (
            f"trained bias mismatch ({oname}, {name})"
        )


# ---------------------------------------------------------------------------
# Scalar helpers reused above (defined near bottom to keep helpers grouped)
# ---------------------------------------------------------------------------

def batch_loss(model, X, y, loss_fn):
    total = 0.0
    for i in range(len(X)):
        value, _ = loss_fn(y[i], model.forward(X[i]))
        total += value
    return total / len(X)


def evaluate(model, X, y, loss_fn):
    total = 0.0
    outs = []
    for i in range(len(X)):
        # Keep the batch dimension (2D) so the NumPy loss receives matching shapes.
        out = model.forward(X[i : i + 1])
        value, _ = loss_fn(y[i : i + 1], out)
        total += value
        outs.append(out)
    return total / len(X), np.vstack(outs)
