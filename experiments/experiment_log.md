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
  Clarified that request-level (12) and window-level (13) are separate
  observation units with separate feature vectors — NOT combined into a 25-feature vector.
* `telemetry/parsers/log_parser.py` — JSONL event parser into `NormalizedEvent`
  records. Client IPs pseudonymized with SHA-256 + salt (aggregation key only,
  never a numerical feature).
* `telemetry/windows/window_aggregator.py` — Fixed-duration window aggregation
  by pseudonymous client identity. Configurable window duration (default 60s).
* `telemetry/feature_extractor.py` — Converts events and windows to fixed-order
  numerical vectors matching the schema exactly.
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

---

## 2026-11-13 — Experiment harness and baseline comparisons

**Question.** Can we build a reproducible experiment harness that loads telemetry
data, splits by client (no leakage), preprocesses, trains multiple models, and
evaluates with standard detection metrics — and can we establish baseline
comparisons against threshold detection and logistic regression?

**Scope.** Pre-Phase-5 preparation. No Phase-5 model training is performed as
part of this entry; the harness is built and validated end-to-end.

**Implementation.** New `training/` modules:

* `training/dataset.py` — `load_dataset()` parses JSONL events, aggregates into
  windows, extracts window-level features, labels windows by client IP hash.
  `split_by_client()` splits at the client level (all windows from one client
  go entirely into train/val/test — no leakage). `fit_preprocessing()` fits
  StandardScaler on training data only.
* `training/evaluate.py` — Extended metrics: precision-recall curve, PR-AUC
  (trapezoidal integration), operating point selection under FPR constraint,
  threshold sweep.
* `training/logistic_regression.py` — From-scratch logistic regression trained
  by gradient descent (NumPy only, no autograd).
* `training/threshold_baseline.py` — Single-feature threshold detector with
  threshold selected on validation data under FPR constraint.
* `training/experiment.py` — `ExperimentConfig` and `run_experiment()`:
  load → split → preprocess → train → evaluate → save artifacts. Supports
  MLP (scalar or NumPy), logistic regression, and threshold baseline.
* `tests/test_harness/` — 13 tests covering metrics, logistic regression,
  threshold baseline.

**Validation.** End-to-end experiment on synthetic data (2,226 events, 432
windows, 13 attack-client windows, 420 normal-client windows):
- Threshold baseline (requests_per_window): accuracy=1.0, FPR=0.0, recall=1.0
- Logistic regression: accuracy=1.0, FPR=0.0, recall=1.0
- MLP (NumPy, [13,16,8,1]): accuracy=1.0, FPR=0.0, recall=1.0

All three models achieve perfect separation because the synthetic attack
patterns are simplistic (high frequency + more 4xx/5xx + scanning paths) and
easily distinguishable from normal traffic. This does NOT demonstrate
real-world attack detection capability.

**Test results.** Full suite: 172 passed, 0 failed (159 previous + 13 new).

**Limitations.**
- Synthetic data labels are implicit (by client IP range), not stored in events.
- Attack patterns are overly simplistic; real attacks are more varied.
- Window-level samples only; request-level features not yet used for training.
- No train/val/test leakage across clients (split by client ID).
- Experiment harness validated but Phase-5 training not yet executed.

**Next.** Await approval before starting Phase-5 supervised classifier training.

---

## 2026-11-13 — Phase 5: Supervised website traffic classification

**Question.** Does a from-scratch neural network provide measurable value over
simpler approaches (rate thresholds, logistic regression) for behavioural
website traffic classification when evaluated on realistic, non-trivial
synthetic traffic with behavioural overlap across clients and scenarios?

**Scope.** Supervised classification of 13-feature behavioural windows.
Comparison of single-feature rate threshold baseline, from-scratch Logistic
Regression, from-scratch Scalar MLP, and NumPy-vectorized MLP. Controlled
synthetic telemetry with explicit ground-truth labelling. No PyTorch,
TensorFlow, Keras, JAX, or autograd. No Flask or Vercel integration.

**Implementation & Audit Corrections.**
* **Audit findings & fixes:**
  - `training/evaluate.py`: Fixed off-by-one threshold indexing in PR curve
    sweep, removed duplicate initial points, and implemented `average_precision`
    alongside trapezoidal `pr_auc`. Documented the theoretical limitation of
    trapezoidal PR-AUC (linear interpolation overestimates area when recall jumps;
    uninformative baseline yields 0.75 vs true AP of 0.50).
  - `training/experiment.py`: Fixed `_run_threshold_baseline` which evaluated
    `binary_metrics` using a hardcoded `threshold=0.5` rather than the threshold
    selected during validation (`model.threshold`). Added per-scenario breakdown
    and inference latency benchmarking across all models.
  - `training/dataset.py`: Updated `load_dataset` to extract explicit ground truth
    labels and scenario names from telemetry events (`event.raw`) rather than
    relying on implicit IP-range lists. Added `split_stratified_by_client`
    (scenario-aware client partitioning with zero client leakage) and
    `split_scenario_held_out` (complete isolation of designated scenarios for
    novelty testing).
* **Traffic generation:**
  - `training/traffic_generator.py`: Built a comprehensive scenario generator
    spanning 13 behavioural scenarios (6 normal, 7 attack-like) across 45 clients,
    generating 54,294 events aggregated into 2,695 windows (1,255 attack, 1,440 normal).
  - Normal scenarios: casual browsing, active browsing, bursty page browsing,
    repeated visits, occasional 404 errors (15-25% 404s), and high-frequency
    legitimate polling (0.46 req/s, 100% 200 OK, 1 path).
  - Attack scenarios: volumetric flood (1.54 req/s), slow path enumeration
    (0.08 req/s, 81% 404s), credential brute-forcing (85%+ 401s), distributed
    low-rate scanning (0.04 req/s), bursty spikes, high-volume scraping
    (0.87 req/s, 100% 200 OK), and parameter fuzzing.
  - Client IPs drawn from a mixed, realistic pool without subnet-based leakage.
* **Model architecture:**
  - MLP: `[13, 16, 8, 1]` with ReLU hidden activations and identity output
    (logits paired with `BinaryCrossEntropyWithLogits`). Adam optimizer (`lr=0.005`),
    batch size 32, 150 epochs.

**Evaluation & Experimental Results.**

### 1. Experiment 1: In-Distribution Benchmark (Disjoint Clients)
*Dataset: 2,695 windows (1,137 train, 779 val, 779 test). Clients disjoint across splits.*

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | PR-AUC | AP | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Threshold (RPS) | 0.6829 | 0.9778 | 0.4200 | 0.5876 | 0.0111 | 0.5800 | 0.8168 | 0.8138 | 0.42 µs |
| Logistic Regression | 0.9833 | 1.0000 | 0.9690 | 0.9842 | 0.0000 | 0.0310 | 0.9998 | 0.9998 | 0.67 µs |
| Scalar MLP | 0.9949 | 0.9976 | 0.9928 | 0.9952 | 0.0028 | 0.0072 | 0.9999 | 0.9999 | 48.26 µs |
| NumPy MLP | 0.9949 | 1.0000 | 0.9905 | 0.9952 | 0.0000 | 0.0095 | 0.9999 | 0.9999 | 0.28 µs |

*Key finding:* Threshold baseline collapses (recall 42.0%, FNR 58.0%) because slow scans,
brute-force, and low-rate attacks have rates below the threshold. Logistic Regression
achieves 98.33% accuracy but misses 21.7% of slow scans (13/60 FN). Both MLPs achieve
99.49% accuracy and 95.0% recall on slow scans, demonstrating that non-linear feature
interactions (rate × 4xx ratio × path count) resolve ambiguous low-rate traffic without
flagging benign 404s. NumPy MLP is ~170× faster than the Scalar MLP reference engine.

### 2. Experiment 2: Unseen Attack Scenarios (Hold-Out Generalization)
*Held out from training/val: `attack_slow_path_enumeration` and `attack_high_volume_scraping` (360 test samples).*

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | PR-AUC | AP | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Threshold (RPS) | 0.6369 | 0.9738 | 0.4507 | 0.6162 | 0.0222 | 0.5493 | 0.8619 | 0.8590 | 0.69 µs |
| Logistic Regression | 0.9470 | 1.0000 | 0.9181 | 0.9573 | 0.0000 | 0.0819 | 0.9997 | 0.9997 | 0.77 µs |
| NumPy MLP | 0.7782 | 1.0000 | 0.6571 | 0.7930 | 0.0000 | 0.3429 | 0.9488 | 0.9489 | 0.21 µs |

*Critical scientific finding:* The MLP completely failed on unseen `attack_high_volume_scraping`
(0.0% recall, 180/180 missed). Because all attack patterns in the training set exhibited
elevated error codes (4xx/5xx) or auth failures, the non-linear MLP learned that status errors
were a necessary condition for maliciousness. When confronted with zero-error scraping, the MLP
confidently misclassified it as benign. Conversely, Logistic Regression's linear weights on
request rate and path diversity allowed it to detect 99.44% of scraping. This provides concrete
evidence of the vulnerability of supervised classification to novel attack topologies.

### 3. Experiment 3: Novel Benign Pattern Stress Test (Unseen Polling)
*Held out from training/val: `normal_api_polling` (240 test samples).*

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | PR-AUC | AP | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Threshold (RPS) | 0.6027 | 0.5353 | 0.6874 | 0.6019 | 0.4630 | 0.3126 | 0.6933 | 0.6914 | 0.58 µs |
| Logistic Regression | 0.9854 | 0.9975 | 0.9690 | 0.9831 | 0.0019 | 0.0310 | 0.9992 | 0.9992 | 0.79 µs |
| NumPy MLP | 0.9948 | 0.9905 | 0.9976 | 0.9941 | 0.0074 | 0.0024 | 0.9998 | 0.9998 | 0.21 µs |

*Key finding:* The threshold detector falsely flagged 100% (240/240) of unseen polling windows
as attacks (FPR = 46.30%). Both Logistic Regression and the MLP maintained 100% accuracy on
polling (0 false positives), recognizing that high rate without errors or path diversity is benign.

**Answer to the Research Question.**
Does the neural network provide measurable value over simpler approaches?
* **In-distribution:** YES. The MLP improves F1 from 0.9842 (LR) to 0.9952 (MLP) and recall
  on subtle low-rate attacks from 78.33% to 95.00%, resolving non-linear trade-offs that linear
  hyperplanes cannot separate.
* **Out-of-distribution (Novel attack styles):** NO. The supervised MLP overfits to the
  error-signature assumptions of its training distribution, missing 100% of zero-error scraping.
  Simpler linear models proved more robust to novel combinations.
* **Conclusion:** Supervised classification alone is insufficient for robust anomaly detection.
  This experimentally justifies the planned transition to unsupervised anomaly detection
  (Autoencoders in Phase 7).

**Test Results.** Full suite: 187 passed, 0 failed (172 previous + 15 new tests covering
traffic generation, split generalization, average precision, and reproducibility).

**Limitations.**
- Evaluated on controlled synthetic HTTP traffic; real web traffic contains more complex
  temporal drift, bot masquerading, and CDN/reverse-proxy aggregation artifacts.
- Supervised classification cannot guarantee detection of unseen attack families without
  retraining or semi-supervised anomaly detection.
- Request-level payload features (URL query strings, headers) are excluded by design to
  preserve privacy and bounded memory.

**Next.** Mark Phase 5 complete in roadmap. Await approval before beginning Phase 6
(Passive Telemetry Integration).

---
