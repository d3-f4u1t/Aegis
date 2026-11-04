# nn_numpy — NumPy-vectorized numerical engine

A vectorized re-implementation of the audited scalar engine in `nn/`, built for
performance while preserving the scalar engine's mathematics exactly. The scalar
engine is **not** modified or replaced; it remains the reference implementation.

This package uses only NumPy (no PyTorch, TensorFlow, Keras, JAX or automatic
differentiation). Every forward pass, loss, derivative, backpropagation step and
optimizer update is hand-written over NumPy arrays.

## Contents

| Module | Purpose |
|---|---|
| `nn_numpy/activations.py` | vectorized sigmoid, tanh, ReLU, identity, softmax + derivatives |
| `nn_numpy/layers.py` | `Parameter`, `DenseLayer`, `init_scale` |
| `nn_numpy/model.py` | `MLP` (forward, backward, predict_proba) |
| `nn_numpy/losses.py` | MSE, BCE-with-logits, BCE |
| `nn_numpy/optimizers.py` | SGD, Momentum, Adam |
| `nn_numpy/training.py` | mini-batch train loop, `History` |
| `nn_numpy/serialization.py` | JSON checkpoints |

## See also

- `docs/NUMPY_ENGINE.md` — array shapes, dimensions, matrix operations and gradient calculations.
- `docs/SCALAR_ENGINE_AUDIT.md` — audit of the scalar reference engine.
- `tests/test_numpy_parity.py` — scalar-vs-NumPy parity (forward, loss, gradients, updates, training).
- `tests/test_numpy_engine.py` — NumPy engine unit tests.
- `benchmarks/benchmark_numpy.py` — scalar vs NumPy performance.

## Quick parity check

```powershell
python -m pytest tests/test_numpy_parity.py tests/test_numpy_engine.py -q
```

## Benchmark

```powershell
python benchmarks/benchmark_numpy.py
```
