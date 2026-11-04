"""Benchmark: scalar engine vs NumPy engine.

Times forward + backward propagation for identical networks and batches,
reporting wall-clock seconds and the NumPy speedup factor. No PyTorch,
TensorFlow, Keras, JAX or automatic differentiation is used.

Run:

    python benchmarks/benchmark_numpy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np

from nn.losses import BinaryCrossEntropyWithLogits as ScalarBCELogits
from nn.losses import MeanSquaredError as ScalarMSE
from nn.model import MLP as ScalarMLP
from nn.optimizers import Adam as ScalarAdam
from training.train import train as scalar_train

from nn_numpy.losses import BinaryCrossEntropyWithLogits as NumpyBCELogits
from nn_numpy.losses import MeanSquaredError as NumpyMSE
from nn_numpy.model import MLP as NumpyMLP
from nn_numpy.optimizers import Adam as NumpyAdam
from nn_numpy.training import train as numpy_train


def scalar_weight_matrix(layer):
    n_in, n_out = layer.n_inputs, layer.n_outputs
    W = np.zeros((n_in, n_out), dtype=float)
    b = np.zeros(n_out, dtype=float)
    for j, neuron in enumerate(layer.neurons):
        for i, w in enumerate(neuron.weights):
            W[i, j] = w.value
        b[j] = neuron.bias.value
    return W, b


def copy_scalar_to_numpy(sm, nm):
    for sl, nl in zip(sm.layers, nm.layers):
        W, b = scalar_weight_matrix(sl)
        nl.W.value = W.copy()
        nl.b.value = b.copy()


def benchmark_case(sizes, batch_size, n_repeats=20):
    rng = np.random.RandomState(0)
    X = rng.randn(batch_size, sizes[0])
    Y = (rng.rand(batch_size, sizes[-1]) > 0.5).astype(float)
    X_list = [list(x) for x in X]
    Y_list = [float(y[0]) for y in Y]

    # Scalar model (reference).
    sm = ScalarMLP(sizes, hidden_activation="relu", output_activation="identity", seed=11)
    # NumPy model seeded identically then loaded with scalar parameters.
    nm = NumpyMLP(sizes, hidden_activation="relu", output_activation="identity", seed=11)
    copy_scalar_to_numpy(sm, nm)

    loss_s = ScalarBCELogits()
    loss_n = NumpyBCELogits()
    opt_s = ScalarAdam(lr=0.01)
    opt_n = NumpyAdam(lr=0.01)

    # Warmup (JIT-less Python, but fills caches).
    sm.zero_grad()
    for x, y in zip(X_list[:4], Y_list[:4]):
        out = sm.forward(list(x))
        _, g = loss_s(y, out)
        sm.backward([gi / 4 for gi in g])
    opt_s.step(sm.parameters())
    nm.zero_grad()
    out = nm.forward(X[:4])
    _, g = loss_n(Y[:4], out)
    nm.backward(g)
    opt_n.step(nm.parameters())

    # ---- Scalar timing: per-sample forward/loss/backward over a batch ----
    def scalar_run():
        sm.zero_grad()
        for x, y in zip(X_list, Y_list):
            out = sm.forward(list(x))
            _, g = loss_s(y, out)
            sm.backward([gi / batch_size for gi in g])
        opt_s.step(sm.parameters())

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        scalar_run()
    t_scalar = time.perf_counter() - t0

    # ---- NumPy timing: batched forward/loss/backward ----
    def numpy_run():
        nm.zero_grad()
        out = nm.forward(X)
        _, g = loss_n(Y, out)
        nm.backward(g)
        opt_n.step(nm.parameters())

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        numpy_run()
    t_numpy = time.perf_counter() - t0

    speedup = t_scalar / t_numpy if t_numpy > 0 else float("inf")
    return t_scalar, t_numpy, speedup


def main() -> None:
    cases = [
        ([2, 4, 1], 16),
        ([2, 4, 1], 256),
        ([10, 32, 16, 1], 64),
        ([10, 32, 16, 1], 1024),
        ([20, 64, 32, 1], 256),
        ([20, 64, 32, 1], 2048),
    ]
    print(f"{'architecture':<22} {'batch':>7} {'scalar(s)':>10} {'numpy(s)':>10} {'speedup':>9}")
    print("-" * 70)
    for sizes, batch_size in cases:
        t_s, t_n, sp = benchmark_case(sizes, batch_size)
        arch = "x".join(str(s) for s in sizes)
        print(f"{arch:<22} {batch_size:>7} {t_s:>10.4f} {t_n:>10.4f} {sp:>9.2f}x")
    print("\nNotes:")
    print("  * Each timing is one Adam update step (forward + backward) over a batch,")
    print("    repeated 20 times, wall-clock via time.perf_counter().")
    print("  * Scalar engine loops over samples in Python; NumPy engine uses batched")
    print("    matmuls. NumPy wins as batch size and layer width grow, and loses for")
    print("    tiny problems where Python overhead dominates.")


if __name__ == "__main__":
    main()
