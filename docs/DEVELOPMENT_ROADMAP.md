# Development Roadmap

The project is developed in phases. Each phase has explicit completion
criteria; the next phase does not begin until the previous one is verified.

## Completed

### Phase 0–2 — Foundations, single neuron and layers, learning algorithm
Neurons, dense layers, forward propagation, activation functions (sigmoid,
tanh, ReLU, identity), losses (MSE, BCE-with-logits, BCE), backpropagation,
SGD/momentum/Adam, numerical gradient checks. A hand-written MLP learns XOR
(loss ≈ 5e-7, 4/4 correct).

### Phase 1b — NumPy vectorisation
A NumPy-based engine reproducing the scalar engine's forward pass, losses, and
backpropagation. Parity-tested against the scalar reference across 5
architectures, 2 losses, and 3 optimisers (forward outputs, loss values,
per-parameter gradients, optimiser updates, multi-epoch training). Benchmarked:
NumPy is up to ~220× faster than scalar on realistic batch sizes.

## In progress / next

### Phase 3 — Training harness (complete)
Preprocessing pipeline (StandardScaler) fitted on training data only.
Experiment harness: `training/dataset.py` (load, split by client, preprocess),
`training/evaluate.py` (PR-AUC, operating point, threshold sweep),
`training/logistic_regression.py` (from-scratch LR),
`training/threshold_baseline.py` (threshold detector),
`training/experiment.py` (end-to-end experiment runner). 13 new tests.
All validated end-to-end on synthetic data.

### Phase 4 — Website telemetry (complete)
A local, controlled telemetry pipeline: synthetic event generator, JSONL parser,
fixed-duration window aggregation, feature extractor matching
`configs/feature_schema.yaml` v1, and preprocessing with training-data-only
statistics. Validated end-to-end with 2,226 synthetic events from 12 clients.
No Flask app, no Vercel integration, no model training.

### Phase 5 — Supervised classification (complete)
Evaluated supervised binary classification on realistic 13-feature behavioural
telemetry with 13 distinct scenarios (6 normal, 7 attack-like) across 45 clients
(54,294 events, 2,695 windows). Compared rate threshold, logistic regression,
scalar MLP, and NumPy MLP. Findings: in-distribution, NumPy MLP achieved 99.49%
accuracy (0.9952 F1) and resolved subtle low-rate scans that logistic regression
partially missed; out-of-distribution (unseen attack patterns), the supervised
MLP overfit to error signatures and missed zero-error scraping, providing
empirical justification for unsupervised anomaly detection in Phase 7. 187 tests
passing. Checkpoints and preprocessing artifacts saved.

### Phase 6 — Passive integration (not started — awaiting approval)
Run inference on telemetry from the local website and generate stored alerts.
Controlled test scenario produces observable alerts while normal requests
continue unaffected.

### Phase 7 — Anomaly detection
Autoencoder trained primarily on benign traffic; reconstruction error as an
anomaly signal. Thresholds selected using validation data.

### Phase 8 — Low-level optimisation
Profile the system, identify bottlenecks, and implement selected numerical
kernels in C if beneficial. Correctness preserved and performance measured.

## Explicitly out of scope for the first release

* A LAN-wide network intrusion detection system.
* A firewall, WAF, IPS, or automatic blocking system.
* Placing the model synchronously in the website's request path.
* Deep-learning frameworks or automatic differentiation for the initial model.
* Continuous training on live traffic.
* Complex RNN, transformer, or distributed architectures at the outset.

## Open decisions (settle before Phase 5)

1. Which dataset family defines the feature schema (request-level HTTP vs
   flow-level network data — they are not interchangeable).
2. The v1 attack taxonomy (behavioural attacks are observable in logs; payload
   attacks such as SQLi/XSS need URL/query-string features).
3. An alert budget (acceptable alerts/day at a chosen false-positive rate).
