# NumPy Engine — Array Shapes, Dimensions, and Gradients

This document explains the NumPy-vectorized engine in `nn_numpy/`, which is a
performance-oriented re-implementation of the audited scalar engine in `nn/`.
The scalar engine is preserved unchanged as the mathematical reference. This
document is read alongside `docs/SCALAR_ENGINE_AUDIT.md`.

## Convention: row-major batches, column-major weights

Every computation is expressed in terms of **2D batches**:

```
X : (batch, n_inputs)      input batch
W : (n_inputs, n_outputs)  weight matrix; column j feeds output neuron j
b : (n_outputs,)           bias vector
Z = X @ W + b              : (batch, n_outputs) pre-activation
A = activation(Z)          : (batch, n_outputs) activation output
```

A single sample is handled by reshaping to `(1, n_inputs)` and returning the
first row, so callers may pass either a 1D array `(n_inputs,)` or a 2D batch.

## Forward propagation

For each layer `l`:

```
Z_l = X_l @ W_l + b_l
A_l = f_l(Z_l)
X_{l+1} = A_l
```

`X @ W` is a batched matrix multiply: output element `(i, j)` is the dot product
of input row `i` with weight column `j`. Bias `b` is broadcast across the batch
dimension. This is exactly the scalar engine's per-neuron weighted sum
`z = b + sum_i(w_i * x_i)`, evaluated for every sample and neuron simultaneously.

## Activations

All activations are **pointwise (element-wise) nonlinearities**, so their
derivative is applied element-wise at `z` and has the same shape as `z`:

| Activation | forward(z) | derivative(z) |
|---|---|---|
| Identity | z | 1 |
| Sigmoid | 1 / (1 + e^-z) | s * (1 - s) |
| Tanh | tanh(z) | 1 - t^2 |
| ReLU | max(z, 0) | 1 if z > 0 else 0 |

The sigmoid is computed with a sign-split so no exponential argument is
positive, avoiding overflow. Softmax (for a later multi-class phase) subtracts
the per-row maximum before exponenting.

## Backpropagation

Given `grad_outputs = dL/dA` of shape `(batch, n_outputs)` for a layer:

```
delta        = grad_outputs * activation'(Z)              (batch, n_outputs)
dL/dW        = X.T @ delta / batch                        (n_inputs, n_outputs)
dL/db        = mean(delta, axis=0)                        (n_outputs,)
dL/dX        = delta @ W.T                                (batch, n_inputs)
```

`dL/dW` and `dL/db` are accumulated into `Parameter.grad`; `dL/dX` is returned
to the previous layer so the chain rule propagates backward through the stack.

### Why the batch average?

The scalar engine computes each sample's gradient, divides it by the batch
size, and accumulates across samples:

```
d(mean_batch_L)/dW = (1/B) * sum_samples dL_sample/dW
```

The NumPy engine computes this directly: `X.T @ delta / batch` is the gradient
of the **mean** batch loss. This is why a single `backward` call on a batch is
equivalent to `B` scalar backward calls each divided by `B`. It is verified
per-parameter against the scalar engine in `tests/test_numpy_parity.py`.

## Losses

All losses return `(loss_value, grad)` where `grad = d(mean_batch_L)/d(prediction)`
and is averaged over the batch.

**Mean squared error:**

```
L     = mean((P - T)^2)
dL/dP = 2 * (P - T) / (batch * n_outputs)
```

**Binary cross-entropy with logits (single logit z):**

```
L     = mean(max(z, 0) - z*y + log(1 + exp(-|z|)))
dL/dz = (sigmoid(z) - y) / batch
```

The loss is algebraically rearranged to stay finite for any logit, and the
gradient avoids the unstable "huge number times vanishing sigmoid derivative"
pattern.

**Binary cross-entropy (textbook, on probabilities):**

```
L     = mean(-[y log p + (1-y) log(1-p)])
dL/dp = (p - y) / (p * (1 - p)) / batch
```

This gradient is w.r.t. probability `p`, **not** logit `z`. Use
`BinaryCrossEntropyWithLogits` for logit outputs.

## Optimizers

SGD, heavy-ball Momentum and bias-corrected Adam read `Parameter.grad` and
mutate `Parameter.value`; they never touch the forward cache. Optimizer state
(momentum velocities, Adam moments) is keyed by `id(parameter)`. Semantics match
the scalar engine exactly.

## Checkpoint format

JSON checkpoints store:

* `format_version: 2` (the scalar engine uses version 1)
* `engine: "numpy"`
* `architecture`: sizes, activations, seed, init
* `layers[]`: each layer's full weight matrix `W` and bias vector `b`
* `metadata`: free-form training information

This differs intentionally from the scalar format, which nests per-neuron
weight lists and biases under `layers[].neurons[]`. A checkpoint is loadable only
by the same engine; cross-engine loading requires an explicit conversion.

## Reproducibility

Initialization is reproducible **within** this engine (same seed → same values).
It is not byte-identical to the scalar engine for the same seed, because the
scalar engine uses Python's `random.Random` while this engine uses NumPy's
`RandomState`. Parity tests therefore copy scalar parameter values into the
NumPy model rather than comparing same-seed initialization.
