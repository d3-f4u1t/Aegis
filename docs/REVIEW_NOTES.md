# Engineering review of PROJECT_DIRECTION.md

**Reviewed:** 2026-11-04

## Verdict

The direction is unusually disciplined for a project of this kind. The scope
boundaries are the strongest part: passive-only deployment, a small MLP before
any deep model, no framework for the initial engine, explicit non-goals, and
evaluation against simple baselines. Those choices remove most of the ways this
project could turn into an unverifiable demo.

The main weakness is not the neural network. It is the **data contract**: which
telemetry fields exist, which dataset can train on them, and which attacks are
therefore observable at all. That deserves more design attention than the
model, because it decides whether Phase 5 is meaningful.

## Risks and recommendations

### 1. Dataset features and live features may not be the same features

`CIC-IDS2017`, `UNSW-NB15`, and `CSE-CIC-IDS2018` are flow/packet-level
datasets. Their features (packet lengths, flow durations, TCP flags, inter-arrival
statistics) are not produced by an access log or application middleware.
`CSIC 2010` is request-level and is the only listed dataset that naturally maps
to an HTTP log schema.

**Recommendation.** Pick one feature schema and one primary dataset family for
v1. Prefer request/window features compatible with CSIC 2010 plus locally
generated traffic. Treat flow-level datasets as a separate experiment, not as
extra rows for the same model. Write the field-by-field mapping (source ->
feature -> unit -> transformation) before writing extraction code.

### 2. Some named attack classes are not observable in ordinary logs

"Repeated authentication failures" requires authentication events. Access logs
usually expose only a status code, and many applications return `200` for a
failed login. Likewise, SQL injection and XSS are payload attacks: if raw query
strings and bodies are excluded, they are invisible.

**Recommendation.** Either emit structured auth events from application
middleware, or drop brute-force from the v1 class list. Decide explicitly
whether v1 targets *behavioural* attacks (flood, scan, enumeration) or also
*payload* attacks. If payload attacks are in scope, add encoded URL features
(length, character-class counts, entropy, suspicious-token counts) without
storing raw values.

### 3. Some "request-level" fields are response-level

Status code, response bytes, and request duration only exist after the response.
Header count and content length are frequently absent from default access logs.
Nginx needs `$request_time` explicitly; `$body_bytes_sent` is not the request
size.

**Recommendation.** Define the event schema by source, not by wish:
middleware events, proxy logs, and packet capture each expose different fields.
Mark every feature as available-at-request-time or available-after-response, and
make missing-value handling part of the schema.

### 4. Window aggregates drift for legitimate reasons

Time of day, campaigns, deployments, shared NAT, and crawlers all change window
statistics. A global autoencoder on raw counts will flag normal change.

**Recommendation.** Favour ratios and per-request aggregates over raw counts,
evaluate across time splits, and report results separately for "new client" and
"known client" windows. If drift proves dominant, add per-client short-term
baselines before adding a more complex model.

### 5. Model metrics are not the operating constraint

With a low attack base rate, precision collapses even at high recall. A model
with excellent PR-AUC can still produce an unusable alert volume.

**Recommendation.** Choose an alert budget (alerts/day at a fixed false-positive
rate) before Phase 5, and report detection rate at that operating point. Keep
the threshold and its selection method in the model artifact.

### 6. Integration is deferred too far

Phase 4 builds telemetry, but the first end-to-end path appears in Phase 6.
Schema mismatches and queue/backpressure problems are cheaper to find with a
dummy threshold model than after training.

**Recommendation.** In Phase 4, build a walking skeleton: log event -> window ->
trivial rule -> stored alert. Then swap the rule for the network in Phase 6.
This does not weaken the "understand the NN first" principle; it de-risks the
pipeline independently.

### 7. Smaller items

- **Softmax/cross-entropy:** implement the fused, stable form when multi-class
  work starts; a naive softmax followed by log is a common source of NaNs.
- **Checkpoint safety:** store a feature-schema hash with the weights and make
  inference fail loudly on mismatch.
- **ReLU health:** this milestone hit an all-dead ReLU initialisation. Track the
  fraction of zero activations per layer during training; consider leaky ReLU if
  it recurs.
- **Operational details:** UTC timestamps everywhere, bounded queues with an
  explicit drop policy, log rotation, and a documented timezone assumption.
- **Licensing:** verify the license and redistribution terms of every dataset
  before committing derived data.

## What is already right and should not change

- Passive, out-of-band placement with no request-path dependency.
- No deep-learning framework for the initial engine.
- Numerical gradient checking before real data.
- Time/scenario-based splits and fitting preprocessing on training data only.
- Comparing against rate thresholds and conventional models.
- Treating an anomaly score as an investigation lead, not proof.
