# Architecture

This document describes the architecture of the neural-network engine. It is
written for the two implementations in this repository:

* `nn/` — scalar Python reference engine (audited, inspectable)
* `nn_numpy/` — NumPy vectorised engine (parity-tested, performance-oriented)

Both implement identical mathematics. `nn/` is the ground truth; `nn_numpy/`
reproduces it with array operations and is verified against `nn/` by parity
tests.

## Core abstraction: the Parameter

Every trainable value is a `Parameter` holding:

* `value` — the current parameter (a float in `nn/`, a NumPy array in
  `nn_numpy/`)
* `grad` — the accumulated gradient of the loss with respect to that parameter

Gradients **accumulate** during backward passes. Only the optimiser mutates
`value`. This separation is what makes mini-batch training correct: several
samples can contribute to the gradient before one update is applied.

## The neuron

One neuron computes:

```
z = b + sum_i(w_i * x_i)
a = activation(z)
```

`forward(inputs)` validates arity, computes `z`, caches `x` and `z`, and
returns `activation(z)`. `backward(grad_output)` receives `dL/da`, computes the
local delta:

```
delta = grad_output * activation'(z)
```

accumulates `dL/dw_i += delta * x_i` and `dL/db += delta`, and returns
`dL/dx_i = delta * w_i` so the previous layer can continue the chain rule.

## The dense layer

A dense layer connects every input to every output neuron. In the scalar engine
this is a loop over neurons; in the NumPy engine it is a single batched
matrix multiply:

```
Z = X @ W + b       Z: (batch, n_outputs)
A = activation(Z)   A: (batch, n_outputs)
```

where `W` is `(n_inputs, n_outputs)` — column `j` feeds output neuron `j` —
and `b` is `(n_outputs,)`, broadcast across the batch.

## Forward propagation

An MLP stacks dense layers:

```
X ──► Dense ──► activation ──► Dense ──► activation ──► … ──► Dense (logit)
```

Layer `i` maps `sizes[i]` inputs to `sizes[i+1]` outputs. The final layer uses
the output activation (default identity, so it emits logits).

## Backward propagation

Given `grad_outputs = dL/dA` for a layer (the gradient of the *mean* batch
loss), the layer computes:

```
delta        = grad_outputs * activation'(Z)
dL/dW        = X.T @ delta             (NumPy) / sum over samples (scalar)
dL/db        = sum(delta, axis=0)      (NumPy) / accumulated (scalar)
dL/dX        = delta @ W.T
```

`dL/dW` and `dL/db` are accumulated into `Parameter.grad`; `dL/dX` is returned
to the previous layer. Layers are visited in reverse order, so the chain rule
propagates from the output back to the input.

## Activations

Pointwise nonlinearities; the derivative is applied element-wise at `z`:

| Activation | forward(z) | derivative(z) |
|---|---|---|
| Identity | z | 1 |
| Sigmoid | 1 / (1 + e⁻ᶻ) | s · (1 − s) |
| Tanh | tanh(z) | 1 − t² |
| ReLU | max(z, 0) | 1 if z > 0 else 0 |

The sigmoid uses a sign-split so no exponential argument is positive, avoiding
overflow. Softmax (for a future multi-class phase) subtracts the per-row
maximum before exponenting.

## Losses

Contract: `loss(y_true, y_pred) → (loss_value, grad)`, where `grad` is
`d(mean_batch_L)/d(prediction)`.

* **Mean squared error:** `L = mean((P − T)²)`, `dL/dP = 2(P − T) / N`
* **Binary cross-entropy with logits:** `L = mean(max(z,0) − z·y + log(1+e⁻|z|))`,
  `dL/dz = (sigmoid(z) − y) / batch`. Numerically stable for extreme logits.
* **Binary cross-entropy (textbook, on probabilities):** `dL/dp = (p − y) /
  (p·(1−p)) / batch`. This is w.r.t. probability `p`, not logit `z`.

## Optimisers

SGD, heavy-ball momentum, and bias-corrected Adam read `Parameter.grad` and
mutate `Parameter.value`. They never touch the forward cache. Optimiser state
(momentum velocity, Adam moments) is keyed by `id(parameter)`.

## Training loop

Per batch:

```
zero_grad → forward(batch) → loss(batch) → backward(batch) → optimiser.step
```

Each sample's gradient is divided by the batch size (scalar) or the loss
gradient is already mean-averaged (NumPy), so one optimiser step applies the
gradient of the mean batch loss. Shuffling uses a seeded RNG for
reproducibility.

## Checkpoints

Self-describing JSON containing `format_version`, `engine`, `architecture`
(sizes, activations, seed, init), per-layer weights and bias, and free-form
`metadata`. Architecture and weights travel together; inference rejects a
mismatched architecture.

The scalar engine uses `format_version: 1` with per-neuron weight lists. The
NumPy engine uses `format_version: 2` with full weight matrices. The formats
are intentionally distinct; a checkpoint is loadable only by its own engine.

## Reproducibility

Initialisation is reproducible within each engine (same seed → same values).
It is **not** byte-identical across engines because `nn/` uses Python's
`random.Random` and `nn_numpy/` uses NumPy's `RandomState`. Parity tests copy
scalar parameter values into the NumPy model rather than comparing same-seed
initialisation.

## Deployment intent (not yet implemented)

The engine is designed for passive, out-of-band monitoring:

```
website ──► [logs / middleware events] ──► [collector] ──► [features]
                                                              │
                                                              ▼
                                                        [preprocessing]
                                                              │
                                                              ▼
                                                        [MLP inference]
                                                              │
                                                              ▼
                                                        [detection policy]
                                                              │
                                                              ▼
                                                        [alert store / UI]
```

The monitoring branch is out of band: stopping it does not affect the website.
Nothing is placed synchronously in the request path.
