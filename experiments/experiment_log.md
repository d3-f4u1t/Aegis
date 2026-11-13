# Experiment log

One entry per meaningful experiment: question, setup, command, result, and
interpretation. Keep raw artifacts out of version control; record their paths
and the commit hash instead.

---

## 2026-11-04 — Milestone 1: scalar engine correctness

**Question.** Is the hand-written forward pass, backpropagation, and optimizer
implementation numerically correct without automatic differentiation?

**Setup.**

- Engine: scalar Python, float64, no third-party runtime dependencies.
- Network: `[3, 4, 2, 1]`, tanh hidden layers, identity output (logit).
- Gradient check: central differences, `h = 1e-6`, every parameter, for both
  MSE and binary cross-entropy with logits.
- Learning check: XOR, `[2, 8, 8, 1]` ReLU hidden / identity output, Adam
  (`lr = 0.05`), full batch of 4, 3000 epochs, seed 7.

**Command.**

```powershell
python -m pytest
python -m examples.train_xor
```

**Result.**

- 46 tests passed; all analytic gradients matched finite differences within
  `2e-6`.
- XOR: loss `5.1e-7`, all four samples classified correctly.
- Checkpoint saved to `experiments/results/xor_mlp.json` (git-ignored).

**Observations.**

- The first gradient-check run failed at a ReLU bias because the seed produced
  an all-dead hidden layer and the neuron sat exactly at `z = 0`, where the
  analytic subgradient (0) and a central difference disagree by construction.
  The check now uses tanh, which is smooth everywhere; the ReLU convention is
  tested separately.
- Dead-ReLU collapse at initialisation is a real training risk for the traffic
  classifier later. Candidate mitigations: leaky ReLU, better initialisation,
  or checking activation health per layer.

**Next.** NumPy array engine with parity tests against this scalar
implementation, then the preprocessing/experiment harness.

---

## 2026-11-04 — Phase 0: Scalar engine audit

**Question.** Is the existing scalar neural-network engine mathematically correct,
genuinely from scratch, and safe to vectorize with NumPy?

**Method.** Independent re-derivation (not reliance on the test suite) of every
neuron/layer/loss/backprop/optimizer equation, followed by an adversarial
numeric check of every analytic gradient against central finite differences.

**Finding.** No critical defects. The engine is correct and from-scratch:
stdlib `math`/`random`/`json` only; no PyTorch/TensorFlow/Keras/JAX/autograd.
Per-parameter analytic-vs-numeric gradient agreement holds for MSE and
BinaryCrossEntropyWithLogits across all 29 parameters of a `[3,4,2,1]` tanh
network (58 assertions), and for all 13 parameters of a `[2,3,1]` ReLU network
evaluated away from the kink.

**Two Important risks documented and regression-tested:**
1. `ReLU.derivative(0.0)` returns subgradient 0 — central differences at exactly
   `z=0` legitimately disagree; gradient check uses tanh, ReLU covered away from
   the kink.
2. `BinaryCrossEntropy` gradient is dL/dp (probability), not dL/dz — contract
   locked with a numeric test; `BinaryCrossEntropyWithLogits` remains the
   logit-output loss.

**Corrections.** No engine code changed (nothing was incorrect). Added two
regression tests to `tests/test_gradients.py`; suite went from 46 → 48 passed,
0 failed.

**Result.** `docs/SCALAR_ENGINE_AUDIT.md` delivered; verdict PASS — proceed to
NumPy vectorization (Phase 1b) with parity tests against this scalar baseline.
Do not proceed to website integration yet.

---

## 2026-11-13 — Phase 4: Website telemetry pipeline and feature schema v1

**Question.** Can we build a reusable, versioned telemetry pipeline that converts
raw HTTP events into normalized feature vectors matching a documented schema,
with preprocessing statistics fitted on training data only?

**Scope.** Local, controlled environment only. No Vercel integration, no model
training, no public datasets, no C optimization, no autoencoder.

**Implementation.** New `telemetry/` package and supporting modules:

* `configs/feature_schema.yaml` — v1 feature schema: 12 request-level features
  (method one-hot, path length, query count, status code, response bytes,
  duration, content length, header count) + 13 behavioural window features
  (request rate, unique paths/methods, 4xx/5xx counts and ratios, auth failures,
  mean/max request size, mean response time, mean inter-request interval).
* `telemetry/parsers/log_parser.py` — JSONL event parser into `NormalizedEvent`
  records. Client IPs pseudonymized with SHA-256 + salt (aggregation key only,
  never a numerical feature).
* `telemetry/windows/window_aggregator.py` — Fixed-duration window aggregation
  by pseudonymous client identity. Configurable window duration (default 60s).
* `telemetry/feature_extractor.py` — Converts events and windows to fixed-order
  numerical vectors matching the schema exactly (25 features total).
* `training/preprocessing.py` — `StandardScaler` fitted on training data only;
  statistics saved with model artifact; inference applies unchanged.
* `examples/generate_synthetic_events.py` — Synthetic HTTP event generator for
  testing (test-only utility, no web server required).
* `tests/test_telemetry/` — 29 tests covering parser, window aggregator,
  feature extractor, and preprocessing.

**Validation.** Generated 2,226 synthetic events from 12 clients (3 attack-like
with higher frequency and more 4xx/5xx). Pipeline: parse → aggregate into 432
windows → extract 13-feature vectors → standardize → save/load roundtrip.
All steps verified end-to-end.

**Test results.** Full suite: 159 passed, 0 failed (46 scalar + 84 NumPy +
29 telemetry).

**Limitations.**
- No live website yet; validated with synthetic events only.
- Synthetic event generator is test-only, not a traffic generator for production.
- No real access log format parsing yet (nginx/apache combined log format would
  need a separate parser).
- Preprocessing is standard scaling only; no robust scaling or quantization.
- Schema is experimental v1, subject to change based on real telemetry inspection.

**Next.** Update roadmap to mark Phase 4 complete; await approval before Phase 3
(preprocessing harness, experiment harness, train/val/test protocol) or Phase 5
(supervised classifier training).

---


## 2026-11-04 — Phase 1b: NumPy vectorization

**Question.** Can a NumPy-vectorized engine reproduce the audited scalar engine's
forward propagation, loss values, gradients, optimizer updates, and training
results, and is it faster?

**Implementation.** New separated package `nn_numpy/` (scalar `nn/` left
untouched):

* `activations.py` — vectorized sigmoid/tanh/ReLU/identity/softmax + derivatives
* `layers.py` — `Parameter`, `DenseLayer` with `Z = X @ W + b`, batched backprop
* `model.py` — `MLP` (forward/backward/predict_proba), 1D or 2D input
* `losses.py` — MSE, BCE-with-logits, BCE (mean-batch gradients)
* `optimizers.py` — SGD, Momentum, Adam (same semantics as scalar)
* `training.py` — mini-batch train loop, `History`
* `serialization.py` — JSON checkpoints (`format_version: 2`, engine `numpy`)

**Key contract detail.** The NumPy loss returns `d(mean_batch_L)/d(prediction)`
(already averaged over the batch), so `DenseLayer.backward` computes
`dL/dW = X.T @ delta` and `dL/db = sum(delta, axis=0)` with NO additional batch
division. This matches the scalar engine, which divides each per-sample
gradient by the batch size before backward. (An early draft divided twice and
produced gradients 1/B too small; caught by the parity tests.)

**Parity results.** `tests/test_numpy_parity.py` + `tests/test_numpy_engine.py`:
84 tests, all passing, covering 5 architectures × 2 losses × 3 optimizers for
forward outputs, loss values, per-parameter gradients, optimizer updates, and
25-epoch training results, plus unit tests for activations/losses/optimizers/
serialization/training. Full suite: 130 passed, 0 failed (46 scalar + 84 numpy).

**Benchmark.** `benchmarks/benchmark_numpy.py` times one Adam step (forward +
backward) over a batch, 20 repeats, wall-clock:

| Architecture | Batch | Scalar (s) | NumPy (s) | Speedup |
|---|---|---|---|---|
| 2x4x1 | 16 | 0.0067 | 0.0028 | 2.40x |
| 2x4x1 | 256 | 0.0596 | 0.0054 | 11.01x |
| 10x32x16x1 | 64 | 0.2826 | 0.0032 | 87.80x |
| 10x32x16x1 | 1024 | 5.0732 | 0.0399 | 127.21x |
| 20x64x32x1 | 256 | 5.6844 | 0.0588 | 96.60x |
| 20x64x32x1 | 2048 | 41.8011 | 0.1886 | 221.63x |

NumPy is dramatically faster as batch size and layer width grow (batched
matmuls vs Python per-sample loops). For tiny problems the advantage is small
because fixed Python/NumPy overhead dominates — an honest, measured limitation,
not a claim.

**Documentation.** `docs/NUMPY_ENGINE.md` explains array shapes, dimensions,
matrix operations, the backward derivation, loss contracts, optimizer semantics,
checkpoint format differences, and reproducibility.

**Caveat documented.** Same seed does not produce byte-identical initialization
across the two engines (Python `random.Random` vs NumPy `RandomState`),
so parity tests copy scalar parameter values into the NumPy model. This is
intentional and documented.

**Verdict.** PASS — the NumPy engine is parity-correct with the scalar reference
and measurably faster. Ready for the scalar engine to remain as the audited
reference while the NumPy engine becomes the default numerical kernel. Do not
proceed to Phase 2 without approval.

---
