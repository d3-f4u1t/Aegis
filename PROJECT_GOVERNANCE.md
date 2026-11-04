# Project Governance

From-Scratch Neural Network for Website Traffic Anomaly Detection

**Status:** Authoritative source of truth for project identity, scope, architectural decisions, and development rules.  
**Effective:** Immediately. This document governs all future development sessions and must be updated whenever an architectural decision is explicitly approved.

---

## A. Permanent project identity

**Project:** From-Scratch Neural Network for Website Traffic Anomaly Detection.

**Objective:** To understand, implement, and train a neural network from its mathematical foundations, then use it to identify known attacks and unusual behavioural patterns in incoming website traffic.

**Deployment intent:** The eventual system must be a passive, out-of-band monitoring system that does not interfere with normal website request processing.

---

## B. Non-negotiable boundaries

- The core neural-network implementation must be developed from scratch.
- Python is the primary implementation language.
- NumPy is allowed for vectorized numerical operations after the scalar implementation.
- No PyTorch, TensorFlow, Keras, JAX or automatic differentiation for the core implementation.
- C may be introduced later for selected low-level numerical operations.
- Initial detection targets a normal website/web application.
- Initial monitoring is passive.
- Both known attack detection and unfamiliar behavioural anomaly detection are long-term objectives.
- No automatic blocking or request modification in the initial system.
- No unrelated features or infrastructure without a demonstrated need.

---

## C. Architecture principles

Maintain the separation between:

1. Website traffic generation.
2. Telemetry collection.
3. Parsing and normalization.
4. Behavioural window aggregation.
5. Feature extraction.
6. Model training.
7. Model inference.
8. Detection policy.
9. Alert storage and dashboard.

Training and live inference must use a consistent, versioned feature schema.

The scalar neural network remains the mathematical reference for the NumPy implementation.

---

## D. Change-control rules

Before making a significant architectural or scope change:

1. Explain the existing design.
2. Identify the problem with it.
3. Explain why the proposed change is necessary.
4. Compare alternatives.
5. Explain the effect on completed work.
6. Explain the effect on future phases.
7. Wait for my approval before implementing the change.

Do not silently change the project's objective, replace established architecture, or introduce unnecessary dependencies.

Small implementation corrections and bug fixes do not require approval, provided they preserve the established design.

---

## E. Development discipline

- Work phase by phase.
- Every phase must have explicit acceptance criteria.
- Passing tests alone does not automatically mean a phase is complete.
- Maintain reproducible experiments.
- Record actual results, not fabricated claims.
- Preserve meaningful Git history.
- Do not automatically start the next phase after completing a task.
- Report blockers and uncertainties honestly.

---

## F. Mandatory workflow for every new development task

### At the beginning of each task:

1. Read `PROJECT_GOVERNANCE.md`.
2. Read the relevant architecture and roadmap documentation.
3. Inspect the current repository state and recent commits.
4. Identify the current phase.
5. Explain how the requested task fits the overall objective.
6. Identify its acceptance criteria.

### At the end:

1. Report files changed.
2. Report tests actually executed and their results.
3. Report any architectural changes.
4. Update the roadmap and experiment log where appropriate.
5. Identify remaining limitations.
6. Wait for my approval before proceeding to another phase.

---

## G. Conflict resolution

If an existing implementation conflicts with this governance document, do not silently choose either one. Explain the conflict and propose a resolution.

If new experimental evidence suggests that the project direction itself should change, present the evidence and alternatives. Do not implement a change to the project's core objective without my approval.

---

## Final instruction

This document governs future development sessions and must be updated whenever an architectural decision is explicitly approved.

This document is **not** permission to implement additional features now.
