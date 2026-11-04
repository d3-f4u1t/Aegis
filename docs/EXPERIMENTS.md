# Experiments

All experiments are recorded in [`experiments/experiment_log.md`](../experiments/experiment_log.md).

Each log entry states:

* **Question** — what the experiment was trying to answer
* **Setup** — model, data, configuration, command
* **Result** — what happened, including measured values
* **Interpretation** — what the result means and what to do next

Generated artifacts (checkpoints, benchmark outputs) in
`experiments/results/` and `benchmarks/results.md` are **git-ignored**. They are
reproduced by running the commands documented in the log.

## Log entries

| Date | Entry | Key result |
|---|---|---|
| 2026-11-04 | Milestone 1: scalar engine correctness | 46 tests passed; per-parameter gradient checks agree with finite differences; XOR loss 5.1e-7 |
| 2026-11-04 | Phase 0: scalar engine audit | No critical defects; two risks documented and regression-tested; 48 tests passing |
| 2026-11-04 | Phase 1b: NumPy vectorisation | 84 parity + engine tests passing; full suite 130 passing; NumPy up to ~220× faster on realistic batches |

## Reproducing an experiment

1. Check out the commit referenced in the log entry.
2. Run the command listed under "Command."
3. Compare your result to the "Result" section.

## Related documents

* `docs/SCALAR_ENGINE_AUDIT.md` — independent audit of the scalar engine
* `docs/NUMPY_ENGINE.md` — NumPy engine shapes, dimensions, gradients
* `benchmarks/benchmark_numpy.py` — scalar vs NumPy performance harness
* `examples/train_xor.py` — runnable demonstration of learning XOR
