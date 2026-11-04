"""Unit tests for the NumPy engine in isolation.

These tests do not reference the scalar engine. They verify the NumPy
implementation's own behaviour: activations, losses, optimizers, parameter
counts, initialization reproducibility, serialization round-trips, and error
handling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nn_numpy.activations import get_activation, sigmoid, softmax
from nn_numpy.layers import DenseLayer, Parameter, init_scale
from nn_numpy.losses import (
    BinaryCrossEntropy,
    BinaryCrossEntropyWithLogits,
    MeanSquaredError,
)
from nn_numpy.model import MLP
from nn_numpy.optimizers import Adam, Momentum, SGD
from nn_numpy.serialization import load_model, save_model
from nn_numpy.training import train


def numeric_derivative(f, z: float, h: float = 1e-6) -> float:
    return (f(z + h) - f(z - h)) / (2.0 * h)


# ---------------- activations ----------------


@pytest.mark.parametrize("activation", [get_activation("sigmoid"), get_activation("tanh")])
@pytest.mark.parametrize("z", [-3.0, -1.0, 0.0, 0.5, 2.0])
def test_activation_derivative_matches_numeric(activation, z):
    expected = numeric_derivative(activation.forward, z)
    assert activation.derivative(np.array([z]))[0] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("z", [-1000.0, -50.0, 0.0, 50.0, 1000.0])
def test_sigmoid_stable_and_bounded(z):
    value = float(sigmoid(np.array([z]))[0])
    assert 0.0 <= value <= 1.0
    assert math.isfinite(value)


def test_relu_and_identity():
    relu = get_activation("relu")
    assert relu.forward(np.array([-2.0, 0.0, 3.0])) == pytest.approx(
        np.array([0.0, 0.0, 3.0])
    )
    assert relu.derivative(np.array([-2.0, 0.0, 3.0])) == pytest.approx(
        np.array([0.0, 0.0, 1.0])
    )
    ident = get_activation("identity")
    assert ident.forward(np.array([-2.0, 5.0])) == pytest.approx(np.array([-2.0, 5.0]))
    assert np.all(ident.derivative(np.array([0.0])) == 1.0)


def test_softmax_is_a_distribution():
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[2] > probs[1] > probs[0]


def test_softmax_handles_large_logits():
    probs = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(probs))


# ---------------- layers / model ----------------


def test_parameter_count():
    model = MLP([2, 3, 1], seed=0)
    # layer0: 2*3+3 = 9 ; layer1: 3*1+1 = 4 ; total 13
    assert model.n_parameters == 13


def test_forward_shapes():
    model = MLP([3, 4, 2], seed=0)
    X = np.random.RandomState(1).randn(5, 3)
    out = model.forward(X)
    assert out.shape == (5, 2)
    out_single = model.forward(X[0])
    assert out_single.shape == (2,)


def test_backward_requires_forward():
    model = MLP([2, 2, 1], seed=0)
    with pytest.raises(RuntimeError, match="before forward"):
        model.layers[0].backward(np.ones((1, 2)))


def test_zero_grad_clears_gradients():
    model = MLP([2, 2, 1], seed=1)
    model.forward(np.array([[1.0, 2.0]]))
    model.backward(np.ones((1, 1)))
    assert any(np.any(p.grad != 0.0) for p in model.parameters())
    model.zero_grad()
    assert all(np.all(p.grad == 0.0) for p in model.parameters())


def test_same_seed_reproducible_initialisation():
    a = MLP([3, 5, 1], seed=42)
    b = MLP([3, 5, 1], seed=42)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.allclose(pa.value, pb.value)


def test_different_seeds_differ():
    a = MLP([3, 5, 1], seed=42)
    b = MLP([3, 5, 1], seed=43)
    assert not all(
        np.allclose(pa.value, pb.value) for pa, pb in zip(a.parameters(), b.parameters())
    )


# ---------------- losses ----------------


def test_mse_loss_and_gradient():
    loss_fn = MeanSquaredError()
    y_pred = np.array([[0.5], [1.5]])
    y_true = np.array([[1.0], [1.0]])
    loss, grad = loss_fn(y_true, y_pred)
    diff = y_pred - y_true
    assert loss == pytest.approx(float(np.mean(diff * diff)))
    assert grad == pytest.approx(2.0 * diff / diff.size)


def test_bce_with_logits_gradient_matches_numeric():
    loss_fn = BinaryCrossEntropyWithLogits()
    y = np.array([[1.0]])
    for z in [-3.0, -0.5, 0.0, 0.5, 3.0]:
        z_arr = np.array([[z]])

        def loss_at(zv):
            return loss_fn(y, np.array([[zv]]))[0]

        h = 1e-6
        numeric = (loss_at(z + h) - loss_at(z - h)) / (2 * h)
        _, analytic = loss_fn(y, z_arr)
        assert analytic[0, 0] == pytest.approx(numeric, abs=1e-6)


def test_textbook_bce_gradient_is_wrt_probability():
    """Contract: BinaryCrossEntropy gradient is dL/dp, not dL/dz."""
    loss_fn = BinaryCrossEntropy()
    y = np.array([[1.0]])
    p = np.array([[0.7]])
    _, grad = loss_fn(y, p)
    expected = (0.7 - 1.0) / (0.7 * (1.0 - 0.7))
    assert grad[0, 0] == pytest.approx(expected, abs=1e-12)


# ---------------- optimizers ----------------


def test_sgd_step():
    p = Parameter(np.array([10.0]))
    p.grad = np.array([2.0])
    SGD(lr=0.5).step([p])
    assert p.value == pytest.approx(np.array([9.0]))


def test_momentum_two_steps():
    p = Parameter(np.array([0.0]))
    opt = Momentum(lr=0.1, beta=0.9)
    p.grad = np.array([1.0])
    opt.step([p])
    assert p.value == pytest.approx(np.array([-0.1]))
    p.grad = np.array([1.0])
    opt.step([p])
    assert p.value == pytest.approx(np.array([-0.1 - 0.19]))


def test_adam_converges_on_quadratic():
    p = Parameter(np.array([0.0]))
    opt = Adam(lr=0.1)
    for _ in range(200):
        p.grad = np.array([2.0 * (p.value[0] - 3.0)])
        opt.step([p])
    assert p.value[0] == pytest.approx(3.0, abs=1e-4)


# ---------------- serialization ----------------


def test_serialization_round_trip_preserves_outputs(tmp_path):
    model = MLP([3, 4, 2, 1], hidden_activation="tanh", seed=19)
    X = np.array([[0.1, -0.2, 0.3]])
    expected = model.forward(X)

    path = save_model(model, tmp_path / "model.json", metadata={"task": "test"})
    loaded = load_model(path)

    assert np.allclose(loaded.forward(X), expected)
    assert loaded.sizes == model.sizes
    assert loaded.hidden_activation == "tanh"
    assert loaded.metadata["task"] == "test"


def test_rejects_unknown_format_version(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format_version": 999, "engine": "numpy"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_model(path)


# ---------------- training ----------------


def test_xor_converges_numpy():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = MLP([2, 8, 8, 1], hidden_activation="relu", seed=7)
    history = train(
        model,
        X,
        Y,
        BinaryCrossEntropyWithLogits(),
        Adam(lr=0.05),
        epochs=3000,
        batch_size=4,
        shuffle=False,
        seed=7,
    )
    assert history.loss[-1] < 0.02
    probs = model.predict_proba(X).ravel()
    assert probs[0] < 0.1 and probs[1] > 0.9 and probs[2] > 0.9 and probs[3] < 0.1


def test_history_one_entry_per_epoch():
    model = MLP([2, 4, 1], seed=1)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [1.0], [1.0], [0.0]])
    history = train(
        model,
        X,
        Y,
        BinaryCrossEntropyWithLogits(),
        Adam(lr=0.01),
        epochs=5,
        batch_size=2,
        shuffle=False,
        seed=1,
    )
    assert len(history.loss) == 5
