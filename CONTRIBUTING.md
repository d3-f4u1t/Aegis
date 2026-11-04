# Contributing to Website Traffic Anomaly Detection

Thank you for considering a contribution. This is an engineering and research
project that builds a neural network from first principles for passive website
traffic anomaly detection. Contributions are welcome, but they must respect the
project's scope and its from-scratch constraint.

## Before you contribute

Please read:

* `docs/PROJECT_DIRECTION.md` — project scope, goals, and explicit non-goals
* `docs/DEVELOPMENT_ROADMAP.md` — the phased plan and completion criteria
* `docs/SCALAR_ENGINE_AUDIT.md` — how the scalar engine was verified

## Scope and non-goals

Contributions must not:

* Introduce PyTorch, TensorFlow, Keras, JAX, or automatic differentiation for
  the initial model or its training.
* Add automatic blocking, firewall/WAF/IPS behaviour, or inline request-path
  deployment.
* Expand the first implementation beyond one locally hosted website.
* Claim that attack detection is implemented before labelled traffic data and
  trained thresholds exist.

## How to contribute

1. **Open an issue first.** Discuss the change you want to make, especially for
   anything that affects scope, architecture, or the roadmap.
2. **Fork the repository** and create a feature branch from `main`.
3. **Make your change.** Keep it focused and consistent with the existing code
   style (explicit, well-documented, hand-written math).
4. **Add or update tests.** Every trainable parameter should have gradient-check
   coverage; new features need corresponding unit or parity tests.
5. **Run the test suite:** `python -m pytest` must pass (currently 130 tests).
6. **Update documentation** if your change affects architecture, roadmap, or
   usage.
7. **Open a pull request** referencing the issue it addresses.

## Commit style

Use clear, imperative commit messages scoped to a phase or area:

```
Phase 1b: add NumPy dense layer with parity tests
docs: add architecture diagram and data flow
```

## Code review

All pull requests require review. Reviewers will check:

* Mathematical correctness (especially gradients and loss contracts)
* Test coverage and reproducibility
* Consistency with the from-scratch constraint
* No new third-party deep-learning dependencies

## Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you agree to abide by its terms.

## License

No license has been selected yet. Contributions are made with the understanding
that a license will be chosen before any public release. Do not contribute code
you do not have the right to share.
