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

### Phase 3 — Training harness
Preprocessing pipeline fitted on training data only, experiment harness,
train/validation/test protocol, reproducibility (seeds, schemas, metadata).

## Planned (not started)

### Phase 4 — Website telemetry
A local website and passive structured-log collector. Windowed behavioural
features (request rate, unique paths, 4xx/5xx ratios, mean size, latency).
The collector must not affect website operation.

### Phase 5 — Supervised detection
A compatible HTTP dataset (e.g. CSIC 2010) plus locally generated traffic.
Train a binary supervised classifier for known traffic patterns. Evaluate
honestly (confusion matrix, precision/recall/F1, FPR/FNR, PR-AUC) and compare
against simple rate thresholds and conventional models.

### Phase 6 — Passive integration
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
