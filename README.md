# Website Traffic Anomaly Detection — From-Scratch Neural Network

A from-scratch neural-network engine plus a passive website-ingress monitoring
pipeline. The engine is plain Python today: **no PyTorch, TensorFlow, Keras,
JAX, or automatic differentiation.**

**Current status:** Milestone 1 — the numerical core works and is tested.
Website telemetry, datasets, and detection are not started yet.

## Quick start

Requires Python 3.11+. The runtime engine has no third-party dependencies.

```powershell
python -m pytest                     # run the full test suite
python -m examples.train_xor         # train a tiny MLP, print results, save a checkpoint
```

Expected demo tail:

```text
final loss=0.00000051  accuracy=1.00
model saved to ...\experiments\results\xor_mlp.json
```

## What is implemented

| Component | File | Notes |
|---|---|---|
| Activations + derivatives | `nn/activations.py` | sigmoid, tanh, ReLU, identity, stable softmax |
| Neuron / dense layer | `nn/layers.py` | explicit scalar loops, per-parameter gradients |
| MLP | `nn/model.py` | configurable depth/width, seeded initialisation |
| Losses | `nn/losses.py` | MSE, binary cross-entropy with logits, textbook BCE |
| Optimizers | `nn/optimizers.py` | SGD, momentum, Adam with bias correction |
| Training loop | `training/train.py` | mini-batches, shuffling, validation, history |
| Metrics | `training/metrics.py` | confusion matrix, precision/recall/F1, FPR/FNR, detection rate at a target FPR |
| Checkpoints | `nn/serialization.py` | JSON with architecture + weights + metadata |
| Tests | `tests/` | 46 tests, including per-parameter numerical gradient checks |

The gradient check compares every analytic gradient against a central finite
difference. That is the main correctness guarantee for the hand-written
backpropagation.

## Design decisions (current milestone)

1. **Scalar Python first.** Parameters are individual `Parameter` objects, not
   arrays. This makes the chain rule inspectable and the gradient check
   unambiguous. NumPy replaces the storage later, behind the same layer API.
2. **Logits, not probabilities, at the output.** The binary network emits one
   raw logit and `BinaryCrossEntropyWithLogits` applies the sigmoid internally.
   This is numerically stable and makes the gradient simplify to `p - y`.
   `predict_proba()` applies the sigmoid at inference time.
3. **Gradients accumulate; only the optimizer mutates values.** This keeps
   mini-batch training correct and makes gradient checking straightforward.
4. **Checkpoints are self-describing.** Architecture, weights, and arbitrary
   metadata (dataset, seed, schema version later) travel together. A future
   inference process must reject a mismatch rather than silently use stale
   preprocessing.
5. **Determinism is testable.** Same seed => identical initial parameters;
   save/load preserves outputs exactly.

## Repository layout

```text
nn/                     neural-network engine
training/               training loop, evaluation, metrics
examples/               runnable demos
tests/                  unit and numerical-correctness tests
docs/                   project direction and review notes
experiments/            experiment log; generated results are git-ignored
```

The planned long-term layout (telemetry, detection, dashboard, C kernels) is in
`docs/PROJECT_DIRECTION.md` §10. Directories are added when their
responsibility becomes real, not before.

## Roadmap

- [x] Phase 0–2: neuron, layers, forward pass, losses, backpropagation,
      SGD/momentum/Adam, numerical gradient checks
- [x] Phase 3 (partial): mini-batches, validation loss, metrics, seeds,
      JSON checkpoints
- [ ] Phase 1b: NumPy array implementation with parity tests against the scalar
      engine; benchmark both
- [ ] Phase 3 (rest): preprocessing pipeline fitted on training data only,
      experiment harness, train/validation/test protocol
- [ ] Phase 4: local website + passive structured-log collector + windowed
      feature extraction
- [ ] Phase 5: compatible HTTP dataset, binary supervised classifier, honest
      evaluation
- [ ] Phase 6: passive inference + alert records
- [ ] Phase 7: autoencoder anomaly path
- [ ] Phase 8: profiling and selected C kernels

## Open decisions

These are tracked in `docs/REVIEW_NOTES.md` and must be settled before Phase 5:

1. Which dataset family defines the feature schema (request-level HTTP vs
   flow-level network data — they are not interchangeable).
2. The v1 attack taxonomy (behavioural attacks are observable in logs;
   payload attacks such as SQLi/XSS need URL/query-string features).
3. An alert budget (acceptable alerts/day at a chosen false-positive rate),
   which is the real operating constraint.

## Documents

- `docs/PROJECT_DIRECTION.md` — the full project direction and phase plan.
- `docs/REVIEW_NOTES.md` — engineering review: strengths, risks, and concrete
  recommendations.
- `experiments/experiment_log.md` — what was run, with which configuration and
  result.
