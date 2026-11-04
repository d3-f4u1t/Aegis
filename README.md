# Website Traffic Anomaly Detection — From-Scratch Neural Network

A from-scratch neural-network engine, written in plain Python and NumPy, intended
as the detection core of a passive, out-of-band website traffic monitoring and
anomaly-detection system.

**This repository currently contains only the neural-network engine.** Website
telemetry collection, feature extraction, datasets, training on traffic data,
and detection integration are **not yet implemented**. Nothing here detects
attacks in live traffic.

## The problem

Website operators need to tell the difference between normal visitors and
anomalous behaviour — for example, a sudden flood of requests from one client,
repeated scanning of invalid paths, or login patterns that look like brute-force
attacks. Most off-the-shelf tools are black boxes: they raise alerts without
exposing how a decision was made, and they are difficult to verify or adapt.

This project takes a different approach: build the neural network from first
principles so that every weight, gradient, and update is understood and
inspectable, then use it as one signal in a passive monitoring pipeline that
observes — but never blocks or modifies — live website traffic.

## Why build from scratch

Most deep-learning projects import a framework (PyTorch, TensorFlow, JAX) and
treat the model as a black box. That is appropriate for many applications, but
it makes the following hard:

* Verifying that backpropagation is implemented correctly.
* Understanding exactly what computation runs on live data.
* Reproducing results across environments without framework version drift.
* Adapting the math to an unusual domain such as HTTP request features.

This engine implements every operation by hand: neurons, dense layers, forward
propagation, activation functions, loss functions, backpropagation, and
optimisers (SGD, momentum, Adam). There is no automatic differentiation. The
code is the documentation.

Two implementations share the same mathematics:

| Implementation | Status | Purpose |
|---|---|---|
| `nn/` — scalar Python | Reference implementation | Audited, easy to inspect, used as the mathematical ground truth |
| `nn_numpy/` — NumPy | Vectorised kernel | Parity-tested against `nn/`; substantially faster on realistic batch sizes |

## What this project is (and is not)

**It is:**

* A from-scratch multilayer perceptron (MLP) engine with forward propagation,
  backpropagation, losses, and optimisers.
* Numerically verified: every analytic gradient is checked against central
  finite differences, and the NumPy engine is parity-tested against the scalar
  reference.
* Designed for a specific deployment pattern: passive, out-of-band monitoring
  that never sits in a website's request path.

**It is not:**

* A deployed website anomaly detector.
* A trained model on real traffic data.
* A firewall, WAF, IPS, or automatic blocking system.
* A claim that any particular attack (SQL injection, brute force, scraping,
  etc.) can be detected.

Attack detection requires labelled traffic data, a defined feature schema, and
trained thresholds — none of which exist in this repository yet. The engine is
the foundation; the detection pipeline is future work (see Roadmap).

## Technical approach

The engine is a small, configurable multilayer perceptron:

```
input features  →  Dense → ReLU/tanh  →  Dense → …  →  Dense (logit output)
                                                   ↳ sigmoid → P(class = 1)
```

* Forward pass: `Z = X @ W + b`, then `A = activation(Z)`.
* Backward pass: the chain rule is applied by hand, layer by layer, in reverse.
  Gradients accumulate into parameters; only the optimiser mutates values.
* Binary output: a single logit. The sigmoid lives inside
  `BinaryCrossEntropyWithLogits`, which keeps the gradient stable
  (`sigmoid(z) - y`) even for extreme logits.
* Checkpoints are self-describing JSON: architecture, weights, bias, and
  metadata travel together.

### Data flow (intended, not yet built)

```
website visitor
      │
      ▼
[web server / reverse proxy]  ──►  structured access logs / middleware events
      │                                              │
      │                                              ▼
      │                                   [passive collector]
      │                                              │
      │                                              ▼
      │                                   [feature extraction]
      │                                              │
      │                                              ▼
      │                                   [preprocessing]
      │                                              │
      │                                              ▼
      │                                   [MLP inference]
      │                                              │
      │                                              ▼
      │                                   [detection policy]
      │                                              │
      │                                              ▼
      │                                   [alert store / UI]
```

The monitoring branch is out of band: if it stops, the website keeps serving
normally. Nothing is placed synchronously in the request path.

## Current development status

| Area | Status |
|---|---|
| Scalar neural-network engine (`nn/`) | Implemented, audited, tested |
| NumPy vectorised engine (`nn_numpy/`) | Implemented, parity-tested vs scalar |
| Training loop, metrics, checkpoints | Implemented (scalar) |
| Verification (gradient checks, parity, XOR demo) | Implemented, 130 tests passing |
| Website telemetry collection | Not started |
| Feature extraction from HTTP logs | Not started |
| Training on traffic data | Not started |
| Detection integration and alerting | Not started |

## Features implemented

* Neurons and dense layers, hand-written forward propagation
* Activations: sigmoid, tanh, ReLU, identity, softmax (each with derivative)
* Losses: mean squared error, binary cross-entropy with logits, textbook BCE
* Optimisers: SGD, momentum, Adam (bias-corrected)
* Backpropagation implemented manually (no automatic differentiation)
* Mini-batch training with shuffling and validation loss
* Model checkpoints (JSON, self-describing)
* Reproducible seeded initialisation
* Numerical gradient verification (central finite differences)
* Scalar ↔ NumPy parity tests (forward, loss, gradients, optimiser updates,
  multi-epoch training)
* Benchmark harness comparing scalar vs NumPy performance
* Deterministic demo: the engine learns XOR (loss ≈ 5e-7, 4/4 correct)

## Planned features (not started)

* Preprocessing pipeline fitted on training data only
* Local website + structured-log telemetry collector
* Windowed behavioural features (request rate, unique paths, error ratios)
* Compatible HTTP dataset (e.g. CSIC 2010) plus locally generated traffic
* Binary supervised classifier for known traffic patterns
* Autoencoder anomaly path for previously unseen behaviour
* Passive inference and stored alert records
* Comparison vs simple rate thresholds and conventional models
* Selected low-level C kernels after CPU profiling

## Repository layout

```
website-traffic-anomaly-detection/
├── .github/                CI workflows and issue templates
├── benchmarks/             Scalar vs NumPy performance benchmark
├── docs/                   Project direction, architecture, audit, roadmap
├── examples/               Runnable demos (e.g. XOR training)
├── experiments/            Experiment log; generated results are git-ignored
├── nn/                     Scalar reference engine (Python, audited)
├── nn_numpy/               NumPy vectorised engine (parity-tested)
├── training/               Scalar training loop, metrics
├── tests/                  Unit and numerical-correctness tests
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE                 (placeholder — license not yet selected)
├── README.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt        (runtime: none — stdlib only)
└── requirements-dev.txt    (pytest, numpy)
```

## Installation

```powershell
# Clone the repository
git clone https://github.com/<your-username>/website-traffic-anomaly-detection.git
cd website-traffic-anomaly-detection

# Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install development dependencies (pytest, numpy)
pip install -r requirements-dev.txt
```

The runtime engine (`nn/` and `nn_numpy/`) uses only the Python standard library
and NumPy respectively. No deep-learning framework is required or used.

## Running the tests

```powershell
python -m pytest
```

Expected output: all tests pass (currently 130). The suite includes:

* Scalar engine tests: activations, forward pass, gradients (numeric checks),
  optimisers, serialization, training, metrics
* NumPy engine tests: activations, losses, optimisers, serialization, training
* Scalar-vs-NumPy parity tests: forward outputs, loss values, per-parameter
  gradients, optimiser updates, multi-epoch training

## Running the examples

```powershell
# Train the from-scratch MLP on XOR and save a checkpoint
python -m examples.train_xor
```

Expected output: final loss ≈ 5e-7, all four XOR samples classified correctly,
checkpoint written to `experiments/results/xor_mlp.json`.

## Reproducing experiments

All experiments are recorded in `experiments/experiment_log.md`. Each entry
states the question, setup, command, result, and interpretation, and references
the commit that produced it. Generated artifacts in `experiments/results/` are
git-ignored; they can be reproduced by running the commands in the log.

To reproduce the NumPy benchmark:

```powershell
python benchmarks/benchmark_numpy.py
```

Results are saved to `benchmarks/results.md`. Timing is wall-clock and varies
between runs; the benchmark reports relative speedup, not absolute performance.

## Roadmap

The phased plan is in `docs/DEVELOPMENT_ROADMAP.md`. In summary:

1. **Phase 0–2 (done):** Scalar engine, losses, backpropagation, optimisers,
   gradient checks
2. **Phase 1b (done):** NumPy vectorised engine with parity tests
3. **Phase 3 (next):** Preprocessing pipeline, experiment harness, train/val/test
   protocol
4. **Phase 4:** Local website + passive telemetry collector + windowed features
5. **Phase 5:** Compatible HTTP dataset, binary supervised classifier, evaluation
6. **Phase 6:** Passive inference + alert records
7. **Phase 7:** Autoencoder anomaly detection
8. **Phase 8:** Profiling and selected C kernels

## Limitations and honest caveats

* **No attack detection is implemented.** The repository contains a neural
  network engine, not a detector. Real-world detection requires labelled traffic
  data and trained thresholds that do not yet exist.
* **An anomaly score is not proof of malicious intent.** Future alerting will
  treat model output as an investigation lead, not a definitive verdict.
* **The scalar engine is slow.** It is intentionally scalar for inspectability;
  the NumPy engine is the performance-oriented kernel.
* **Same seed does not produce identical initialisation across the two
  engines** (Python `random.Random` vs NumPy `RandomState`). Parity is
  established by copying parameter values, not by matching seeds.
* **Benchmark results are wall-clock timings** on one machine and vary between
  runs. They indicate relative behaviour, not absolute performance guarantees.

## Documentation

* `docs/PROJECT_DIRECTION.md` — project scope, non-goals, deployment architecture
* `docs/ARCHITECTURE.md` — engine architecture, data flow, shapes, gradients
* `docs/SCALAR_ENGINE_AUDIT.md` — independent audit of the scalar engine
* `docs/NUMPY_ENGINE.md` — NumPy engine shapes, dimensions, gradient calculations
* `docs/DEVELOPMENT_ROADMAP.md` — phased development plan
* `docs/EXPERIMENTS.md` — index to the experiment log
* `docs/REVIEW_NOTES.md` — engineering review of risks and data-contract decisions

## License

A `LICENSE` placeholder is present. No license has been selected yet; one must
be chosen before any public release. See the repository for the current state.

## Contributing

See `CONTRIBUTING.md` for how to report bugs, suggest features, and submit
changes. All contributions are subject to the project's
`CODE_OF_CONDUCT.md`.

## Security

See `SECURITY.md` for the security policy and responsible disclosure process.
