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
