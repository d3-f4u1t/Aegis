# Project Direction: From-Scratch Neural Network for Website Traffic Anomaly Detection

**Document type:** Technical project direction and engineering blueprint  
**Status:** Initial scope agreed; implementation not yet started  
**Primary target:** A normal website / web application  
**Initial deployment:** Passive, out-of-band monitoring  
**Detection objective:** Both known attack detection and previously unseen anomalous behaviour  
**Core constraint:** The neural-network engine must be implemented and understood from scratch, not built by importing a deep-learning framework.

---

## 1. Project summary

Build a neural-network engine from first principles, then integrate it into a passive monitoring system for incoming traffic to a website or web application.

The system observes website traffic, extracts request-level and behavioural features, converts those features into numerical vectors, and passes them to a neural network. The model produces predictions or anomaly scores. The monitoring system records and presents alerts but does not block, modify, delay, or otherwise interfere with live requests in the initial version.

The project has two equally important outcomes:

1. **Learning and implementation outcome:** Develop a deep, low-level understanding of neural networks, their mathematics, numerical operations, training algorithms, software architecture, and execution on computer hardware.
2. **Applied engineering outcome:** Use the neural network in a realistic website traffic monitoring pipeline and evaluate whether it can identify known attacks and unusual traffic patterns.

This is initially a website-focused system. A broader server, service, or LAN-wide implementation may be considered later, but must not expand the first implementation's scope.

## 2. Scope decisions

The following decisions are fixed for the initial project.

| Decision | Initial direction |
|---|---|
| Environment | One locally hosted or controlled website/web application |
| Traffic of interest | Incoming website requests and related connection metadata |
| Placement | Passive, out-of-band observer |
| Initial model | A manually implemented multilayer perceptron (MLP) |
| Detection goals | Known attack classification and anomaly detection |
| Training | Offline training using datasets and controlled traffic |
| Live operation | Inference on observed traffic; no automatic blocking |
| Implementation language | Python first; C for selected low-level components later |
| Numerical tools | Begin with scalar Python; introduce NumPy for array operations |
| Deep-learning frameworks | Do not use PyTorch/TensorFlow to implement the initial model or its training |
| Initial data source | Public HTTP/security datasets plus locally generated traffic |
| Future extensions | Autoencoder, temporal models, inline deployment, wider server/network coverage |

### Explicit non-goals for the first version

- Do not build a LAN-wide network intrusion detection system.
- Do not build a firewall, WAF, IPS, or automatic blocking system.
- Do not place the experimental model synchronously in the website's request path.
- Do not begin with deep learning frameworks or automatic differentiation.
- Do not attempt to detect every possible cyberattack.
- Do not claim that an anomaly score proves malicious intent.
- Do not train continuously on live traffic in the initial version.
- Do not begin with a complex RNN, transformer, or distributed architecture.

---

## 3. System context and deployment architecture

### 3.1 Intended traffic path

The website continues to serve requests normally. A monitoring component receives a copy of suitable request telemetry or traffic metadata and processes it independently.

```text
                 Internet / Website visitors
                            |
                            v
                  [Website entry point]
                            |
              +-------------+-------------+
              |                           |
              v                           v
       [Reverse proxy /              [Passive telemetry
        web server]                   / log / event copy]
              |                           |
              v                           v
       [Web application]            [Collector]
                                          |
                                          v
                                   [Parser / normalizer]
                                          |
                                          v
                                   [Session / window
                                      aggregation]
                                          |
                                          v
                                   [Feature extraction]
                                          |
                                          v
                                   [Preprocessing]
                                          |
                                          v
                                   [Neural network
                                      inference]
                                          |
                                          v
                                   [Detection policy]
                                          |
                                          v
                                   [Alert store /
                                    monitoring UI]
```

The telemetry branch must not sit in the critical request path. If the monitoring process stops, the website must continue to operate normally.

### 3.2 Practical first deployment

Start with a local test environment:

- A small website or web application running locally.
- A reverse proxy such as Nginx, if useful for the experiment.
- Access logs or application middleware that emits structured request events.
- A separate Python monitoring process that reads those events.
- A locally stored model checkpoint.
- A simple alert log or dashboard.

For the first prototype, structured web-server or application logs are a practical and safe source of HTTP request information. Later, compare this with packet capture or proxy telemetry.

### 3.3 Passive monitoring methods

Possible telemetry sources:

1. **Application middleware:** The application emits request metadata after or during request handling.
2. **Reverse-proxy logs:** A reverse proxy records request and response metadata.
3. **Mirrored traffic / packet capture:** A separate sensor observes copied packets. This is useful for network metadata, but encrypted HTTPS payloads are not readable before TLS termination.
4. **Proxy event stream:** A reverse proxy exports structured events to a separate collector.

The first implementation should use structured logs or middleware events. It is simpler, avoids unnecessary packet-parsing complexity, and can expose HTTP-level fields after TLS termination.

Do not assume that a passive packet sensor outside TLS termination can see URLs, HTTP methods, headers, or bodies. With HTTPS, those fields require visibility at a trusted TLS-terminating component, such as the web server, reverse proxy, or application.

If a CDN or load balancer is present, the origin server may see intermediary addresses. Only trust forwarded client-address headers when they are set by a trusted proxy and untrusted clients cannot spoof them.

---

## 4. What the detector should detect

The first version has two detection goals.

### 4.1 Known attack / known-class detection

Train a supervised classifier on labelled examples. Initial classes may be:

- Benign / normal
- HTTP flooding or unusually high request rates
- Repeated authentication failures / brute-force-like behaviour
- Path or endpoint scanning
- Suspicious or malformed HTTP request patterns

The exact attack classes depend on dataset quality and what the local test environment can safely reproduce. Do not claim detection of an attack class unless the training and evaluation data actually represent it.

### 4.2 Previously unseen anomalous behaviour

Develop a second detection path that learns a representation of normal traffic and flags unusual observations. A suitable later model is an autoencoder trained primarily on normal traffic.

An autoencoder reconstructs its input. A reconstruction error can be used as an anomaly signal:

\[
E(x,\hat{x}) = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
\]

A threshold is selected using validation data, not chosen arbitrarily. High reconstruction error is a candidate anomaly, not proof of an attack.

### 4.3 Recommended development order

1. Implement a supervised MLP for binary classification: benign versus attack/anomalous.
2. Extend to multiple attack categories only when the data supports it.
3. Implement an autoencoder using the same neural-network engine.
4. Combine or compare the two detection paths.
5. Consider temporal models only after the flow/window pipeline and baselines are reliable.

A supervised classifier cannot reliably identify every unseen attack. The autoencoder can flag unfamiliar patterns but may also flag legitimate changes. Combining both provides complementary signals, not guaranteed coverage.

---

## 5. Traffic data: events, sessions, and behavioural windows

A neural network consumes numerical inputs, not raw concepts such as “a website visitor” or “an attack.” The system must define what one model input represents.

### 5.1 Initial observation units

Use two related observation types:

- **Request-level event:** One HTTP request and its available metadata.
- **Behavioural window:** Aggregated activity associated with a client/session over a fixed time interval.

A single request can look ordinary while a sequence of requests is suspicious. Therefore, behavioural features are essential.

For example, an initial window could be 30 or 60 seconds. Treat the window duration as a configurable experiment parameter and evaluate alternatives.

### 5.2 Candidate request-level features

Only use fields that are available, lawful to collect, and consistently defined.

| Feature | Description |
|---|---|
| HTTP method | Encoded method category, e.g. GET, POST, PUT |
| Path length | Length of URL path, excluding or separately treating query string |
| Query parameter count | Number of query parameters |
| Request byte count | Size of request or logged request metadata |
| Header count | Number of headers, if safely available |
| Content length | Declared/observed request body length, when available |
| Status code | Response status, available after response |
| Response byte count | Response size, if logged |
| Request duration | Server-observed request processing time |
| Protocol/version | HTTP version or protocol category |

Do not store or train directly on passwords, session tokens, authorization headers, cookies, personal data, or complete request bodies. Exclude or redact sensitive values. Prefer derived metadata over raw content.

### 5.3 Candidate behavioural features

Aggregate by a carefully chosen key, such as a trusted client identity or pseudonymous client/session identifier, over a time window.

| Feature | Description |
|---|---|
| Requests per window | Request count during the window |
| Requests per second | Rate derived from window duration |
| Unique paths | Number of distinct paths requested |
| Unique methods | Number of HTTP methods used |
| 4xx count / ratio | Client-error response frequency |
| 5xx count / ratio | Server-error response frequency |
| Authentication failures | Count of failed authentication events, if safely exposed |
| Mean request size | Average request bytes |
| Maximum request size | Largest request in the window |
| Mean response time | Average server response duration |
| Request interval statistics | Mean/variance of time between requests |
| Concurrent requests | Approximate concurrency, if reliable |
| Connection metadata | Counts or protocol statistics, where available |

The first feature schema should remain small, explicit, and testable. Add features only when there is a reason and data to support them.

### 5.4 Client identity and privacy

IP addresses can help group events, but an IP address is not a meaningful continuous numerical feature. Do not simply convert dotted IPv4 addresses into large integers and feed them into the model.

Use IP/client identity for aggregation and investigation, with appropriate privacy controls. Consider keyed pseudonymization for stored identifiers. Account for shared NAT addresses, proxies, changing addresses, and multiple users behind one address.

### 5.5 Feature vector

After extraction and preprocessing, a sample might look conceptually like:

\[
x=[\text{request rate},\text{unique paths},\text{4xx ratio},
\text{mean size},\text{mean latency},\ldots]
\]

The actual order, units, categories, and transformations must be defined in a versioned feature schema. Training and inference must use exactly the same schema.

---

## 6. Data sources and dataset strategy

Use multiple sources because public datasets and a live website represent different environments.

### 6.1 Candidate public datasets

Investigate the following; inspect their documentation, licenses, feature definitions, label meanings, and data quality before adopting them.

- **CSIC 2010 HTTP Dataset:** Relevant to HTTP request anomaly detection and web request patterns.
- **CIC-IDS2017:** Useful for network intrusion and flow-based attack classification. Its feature representation may not match application-level HTTP logs.
- **UNSW-NB15:** Useful for network intrusion classification and flow-oriented experiments. Do not assume its features map directly to website request logs.
- **CSE-CIC-IDS2018:** A larger network intrusion dataset that may support broader experiments, but requires careful feature compatibility checks.

Do not combine datasets by blindly concatenating columns. A feature with the same name may have different units, extraction logic, or semantics across datasets.

### 6.2 Local data collection

Create a controlled local website and collect:

- Normal browsing traffic generated by test users or scripts.
- Repeated requests at different rates.
- Normal login and failed-login sequences using test accounts.
- Requests to valid and invalid paths.
- Controlled, non-destructive scanning and request-pattern simulations in the local lab.

Label the generated scenarios precisely. Record how each sample was produced.

### 6.3 Data leakage prevention

- Split data by time, host, session, or capture scenario where appropriate.
- Avoid putting nearly identical flows or requests from the same activity into both training and test sets.
- Fit normalization statistics on training data only.
- Keep a final test set untouched until model selection is complete.
- Report dataset limitations and possible distribution shift.

---

## 7. Neural-network implementation requirements

### 7.1 From-scratch definition

The initial neural-network engine must implement these operations directly:

- Parameter initialization
- Dense/fully connected layers
- Forward propagation
- Activation functions
- Loss calculations
- Backpropagation and gradient calculation
- Gradient descent and at least one additional optimizer
- Mini-batch training
- Evaluation
- Model parameter serialization and loading

Do not use PyTorch, TensorFlow, Keras, JAX, or automatic differentiation to implement the initial model or training loop.

NumPy may be introduced after scalar Python implementations are understood. NumPy is a numerical array library, not a neural-network framework. Important operations must still be implemented by the project code rather than delegated to a deep-learning library.

### 7.2 Learning sequence

1. Python fundamentals needed for implementation.
2. Scalars, arrays, vectors, matrices, dot products, and matrix multiplication.
3. Floating-point representation, numerical precision, and overflow/underflow.
4. A single artificial neuron: inputs, weights, bias, weighted sum.
5. Activation functions: ReLU, sigmoid, tanh, softmax and their use cases.
6. Dense layers and tensor/array shapes.
7. Multilayer network forward pass.
8. Loss functions: mean squared error and cross-entropy, with appropriate tasks.
9. Derivatives and the chain rule.
10. Backpropagation derived and implemented manually.
11. Gradient descent, learning rate, momentum and Adam.
12. Mini-batches, epochs, shuffling and validation.
13. Initialization, normalization, regularization and overfitting.
14. Numerical gradient checking and unit tests.
15. Model saving/loading and reproducible inference.

### 7.3 Initial MLP architecture

Start with a small configurable MLP. For example:

```text
Input: N traffic features
Dense: 32 neurons + ReLU
Dense: 16 neurons + ReLU
Output: 1 neuron + Sigmoid   # binary classification
```

This is only an initial experiment, not a final architecture. Choose input dimension from the feature schema. Tune hidden dimensions using validation data.

For multi-class classification, use an output layer with one logit per class and an appropriate softmax/cross-entropy implementation. For an autoencoder, use an encoder, bottleneck, and decoder.

### 7.4 Numerical correctness

Implement tests for:

- Output shapes at every layer.
- Correct parameter counts.
- Known small forward-pass examples.
- Finite loss and gradients.
- Backpropagation against finite-difference numerical gradients.
- Training on a tiny dataset that can be overfit intentionally.
- Save/load equivalence.
- Deterministic runs with controlled random seeds.

Do not move to a large dataset until these tests pass.

---

## 8. Training and inference are separate

### 8.1 Offline training pipeline

```text
Dataset / captured historical telemetry
              |
              v
       Parse and validate
              |
              v
       Feature extraction
              |
              v
       Train/validation/test split
              |
              v
       Fit preprocessing on train only
              |
              v
       Neural network training
              |
              v
       Validation and threshold selection
              |
              v
       Save model + preprocessing + schema
```

Training updates weights and biases. It should initially happen offline and reproducibly.

### 8.2 Passive inference pipeline

```text
Website request / proxy event
              |
              v
       Telemetry collector
              |
              v
       Feature extraction
              |
              v
       Same saved preprocessing
              |
              v
       Load model parameters
              |
              v
       Neural network prediction
              |
              v
       Detection policy / threshold
              |
              v
       Alert record and dashboard
```

The inference process must use the same feature order, units, category encodings, normalization values, and missing-value handling as training.

Save these together as a model artifact:

- Model weights and biases.
- Architecture/configuration.
- Feature schema and version.
- Preprocessing parameters.
- Class mapping.
- Model version and training metadata.
- Anomaly threshold and how it was selected.

Do not silently change the feature schema while using an old model.

---

## 9. Detection policy and alerts

The neural network produces a prediction or score. A separate detection policy turns that output into an alert.

An alert should contain, where available:

- Timestamp
- Model version
- Pseudonymous client/session identifier
- Observation/window identifier
- Predicted class
- Model score
- Threshold used
- Relevant non-sensitive feature values
- Detection reason or supporting signals
- Confidence limitations

For combined detection:

- Supervised classifier: class probabilities or logits.
- Autoencoder: reconstruction error.
- Rule-based signals: transparent thresholds such as unusually high request rate.

Keep these signals separate initially. Evaluate each independently before designing a combined decision policy.

An alert is an investigation lead. The initial passive system must not block users, modify requests, or automatically take defensive action.

---

## 10. Software architecture and repository structure

A possible project structure:

```text
website-traffic-anomaly-detection/
|
|-- README.md
|-- PROJECT_SPEC.md
|-- requirements.txt
|-- configs/
|   |-- feature_schema.yaml
|   |-- model.yaml
|   `-- detection.yaml
|
|-- data/
|   |-- raw/                 # excluded from version control if large/sensitive
|   |-- processed/
|   `-- README.md
|
|-- nn/
|   |-- tensor_ops.py        # early scalar/array operations
|   |-- initializers.py
|   |-- activations.py
|   |-- layers.py
|   |-- losses.py
|   |-- optimizers.py
|   |-- model.py
|   |-- backprop.py
|   `-- serialization.py
|
|-- training/
|   |-- dataset.py
|   |-- preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   `-- metrics.py
|
|-- telemetry/
|   |-- collectors/
|   |-- parsers/
|   |-- sessions.py
|   |-- windows.py
|   `-- feature_extractor.py
|
|-- detection/
|   |-- inference.py
|   |-- policy.py
|   `-- alerts.py
|
|-- integration/
|   |-- web_app/
|   |-- reverse_proxy/
|   `-- local_lab/
|
|-- dashboard/
|
|-- low_level/
|   `-- c_kernels/           # later: selected C implementations
|
|-- tests/
|   |-- test_activations.py
|   |-- test_forward.py
|   |-- test_gradients.py
|   |-- test_training.py
|   |-- test_features.py
|   `-- test_inference_pipeline.py
|
`-- experiments/
    |-- experiment_log.md
    `-- results/
```

This is a target structure, not a requirement to create every directory immediately. Add modules when their responsibilities become clear. Keep raw datasets and sensitive telemetry out of public version control.

---

## 11. Low-level software and computer systems learning

After the Python implementation works, investigate how the computations execute on real hardware.

Topics to study:

- Python objects, references, lists, and memory overhead.
- Array memory layout, shape, strides, and contiguous storage.
- Row-major matrix storage and cache locality.
- Floating-point formats and numerical stability.
- CPU registers, cache hierarchy, and memory bandwidth.
- Matrix multiplication algorithms and computational complexity.
- Vectorization and SIMD.
- Profiling and benchmarking.
- Threads, processes, and safe data pipelines.
- C pointers, arrays, allocation, and deallocation.
- Python/C interfaces for replacing selected numerical kernels.
- GPU execution concepts only after CPU profiling establishes a reason.

Implement selected low-level operations in C, such as vector operations or matrix multiplication, and compare correctness and performance against the Python/NumPy versions. Do not rewrite the entire project in C merely for the sake of it.

---

## 12. Evaluation plan

A credible project must measure more than accuracy.

### 12.1 Detection metrics

For classification, report:

- Confusion matrix
- Precision
- Recall
- F1-score
- False-positive rate
- False-negative rate
- Per-class metrics for multi-class experiments
- PR-AUC where class imbalance makes it useful

Accuracy alone is insufficient, especially when benign traffic dominates.

For anomaly detection, report:

- Detection rate at selected false-positive rates
- Precision/recall against labelled evaluation data
- Reconstruction-error distributions for normal and anomalous data
- Threshold selection method
- Performance on data from a different time or traffic scenario

### 12.2 Systems metrics

Measure:

- Inference latency (median and tail percentiles)
- Events processed per second
- CPU and memory usage
- Collector and queue throughput
- Dropped or delayed telemetry events
- End-to-end time from request observation to alert
- Effect on website latency (the passive design should avoid adding request-path latency)

### 12.3 Baselines

Compare the neural network with simple baselines:

- Fixed request-rate thresholds.
- Logistic regression.
- A decision tree or another suitable conventional model, if useful.

The purpose is to establish whether the neural network provides measurable value over simpler approaches.

### 12.4 Generalization tests

Test across:

- Different time periods.
- Different traffic-generation scenarios.
- Different clients or sessions.
- Normal traffic patterns not present in training.
- Attack patterns not used during training, where a valid evaluation set exists.

Be explicit about what “unseen” means. A random split of highly similar records does not prove generalization to unseen attack types or environments.

---

## 13. Security, privacy, and operational safety

- Monitor only systems and traffic for which permission has been granted.
- Use a local lab for attack simulations.
- Do not collect passwords, session tokens, authorization headers, cookies, or unnecessary personal information.
- Redact sensitive query parameters and request fields.
- Limit access to raw telemetry and alert records.
- Define retention periods for captured data.
- Treat forwarded client-IP headers as untrusted unless received through a configured trusted proxy.
- Validate and sanitize log input; request strings may contain hostile content.
- Ensure the monitoring service cannot crash or block the website.
- Use bounded queues and define behaviour when the collector or inference process falls behind.
- Log model errors and telemetry loss.
- Do not treat model output as definitive proof of malicious activity.
- Keep automatic blocking out of the initial scope.

---

## 14. Development phases and completion criteria

### Phase 0 — Foundations

Learn Python fundamentals, numerical representation, vectors, matrices, and basic calculus as needed.

**Complete when:** Able to calculate a neuron's weighted sum and explain every operation.

### Phase 1 — Single neuron and layers

Implement neurons, activations, dense layers, and forward propagation.

**Complete when:** A small network produces verified outputs with correct shapes.

### Phase 2 — Learning algorithm

Implement loss, derivatives, backpropagation, gradient descent, and numerical gradient checks.

**Complete when:** The network can learn a small synthetic task without automatic differentiation.

### Phase 3 — Training engine

Implement datasets, mini-batches, shuffling, validation, metrics, checkpoints, and reproducibility.

**Complete when:** A model can be trained, saved, reloaded, and evaluated consistently.

### Phase 4 — Website telemetry

Build a local website and passive request-event collector. Parse structured events and create request/window features.

**Complete when:** The collector produces documented, repeatable feature vectors without affecting website operation.

### Phase 5 — Dataset and supervised detection

Prepare compatible HTTP/network data, train a binary classifier, and evaluate against held-out data.

**Complete when:** Metrics, limitations, and error cases are documented.

### Phase 6 — Passive integration

Run inference on telemetry from the local website and generate stored alerts.

**Complete when:** A controlled test scenario produces observable alerts, while normal website requests continue unaffected.

### Phase 7 — Anomaly detection extension

Implement an autoencoder and compare its results with the supervised classifier and simple baselines.

**Complete when:** Thresholds are selected using validation data and results are reported on held-out data.

### Phase 8 — Low-level optimization

Profile the system, identify bottlenecks, and implement selected numerical kernels in C if beneficial.

**Complete when:** Correctness is preserved and performance changes are measured.

### Phase 9 — Broader deployment

Only after the website prototype is reliable, consider more web servers, general-purpose services, reverse-proxy integrations, distributed sensors, or inline enforcement.

---

## 15. Research and engineering questions

Keep an experiment log and investigate questions such as:

1. Can a manually implemented MLP identify known anomalous website traffic patterns?
2. How does request-level information compare with time-window behavioural features?
3. How does class imbalance affect false positives and missed detections?
4. Can an autoencoder trained primarily on benign traffic identify some patterns absent from supervised training?
5. How well does the detector generalize across time periods and traffic-generation environments?
6. What is the inference latency and throughput of the from-scratch implementation?
7. Does a neural network provide a meaningful benefit over simple rate-based rules and conventional models?
8. Which features contribute most to detection, and how can alerts be made understandable?

These are questions to test, not conclusions to assume.

---

## 16. Immediate next steps

Do the following in order:

1. Establish the Python environment and repository.
2. Learn the numerical foundations needed for a neuron.
3. Implement a single neuron using plain Python.
4. Implement a small MLP and forward propagation.
5. Derive and implement backpropagation.
6. Add tests and a reproducible training loop.
7. Train on a small synthetic classification problem to verify learning.
8. Build a local website and inspect its request logs.
9. Define and version the first HTTP traffic feature schema.
10. Build the passive telemetry collector and test it independently.
11. Prepare a compatible HTTP dataset and train the first traffic classifier.
12. Integrate inference and alert generation.

Do not start by downloading a large dataset, building a dashboard, or capturing all network packets. First prove that the neural-network engine works correctly on a small, understandable task. Then connect it to the website telemetry pipeline.

---

## 17. Project principles

1. **Understand before abstracting:** Implement basic operations explicitly before wrapping them in reusable abstractions.
2. **No hidden learning machinery:** The initial model's forward pass, gradients, backpropagation, and optimizer must be our own code.
3. **Separate concerns:** Traffic collection, feature extraction, model training, inference, and alerting are separate components.
4. **Reproducibility:** Record feature schemas, dataset versions, random seeds, model configurations, and experiment results.
5. **Measure, don't assume:** Evaluate accuracy, false positives, latency, throughput, and generalization.
6. **Passive first:** Monitoring must not interfere with the website.
7. **Privacy by design:** Collect the minimum telemetry required and redact sensitive information.
8. **Expand only after a working baseline:** Add autoencoders, temporal models, C optimization, or broader deployment only when the previous phase is tested.

## Final definition

The initial project is a **from-scratch neural-network implementation integrated with a passive website-ingress traffic monitoring and anomaly detection pipeline**.

It will begin with a local website, structured HTTP telemetry, a manually implemented MLP, supervised detection of known traffic patterns, and an independent alerting process. It will then extend toward autoencoder-based detection of unfamiliar patterns and lower-level performance engineering.

The system is an experimental detection and monitoring tool, not a guaranteed security product. Its value must be demonstrated through controlled experiments, honest evaluation, and comparison with simpler baselines.
