"""Tests verifying reproducibility and parity of Phase 5 experiments."""

import pytest
from training.experiment import ExperimentConfig, run_experiment


def test_experiment_reproducibility_deterministic():
    """Verify that repeating an experiment with identical config yields identical results."""
    cfg1 = ExperimentConfig(
        experiment_name="reproducibility_check_run1",
        seed=123,
        models=["threshold", "logistic_regression", "numpy_mlp"],
        split_strategy="stratified_client",
        mlp_epochs=20,
        mlp_batch_size=32,
        lr_epochs=50,
        events_path="data/telemetry/synthetic_phase5.jsonl",
        output_dir="experiments/results",
    )
    cfg2 = ExperimentConfig(
        experiment_name="reproducibility_check_run2",
        seed=123,
        models=["threshold", "logistic_regression", "numpy_mlp"],
        split_strategy="stratified_client",
        mlp_epochs=20,
        mlp_batch_size=32,
        lr_epochs=50,
        events_path="data/telemetry/synthetic_phase5.jsonl",
        output_dir="experiments/results",
    )

    res1 = run_experiment(cfg1)
    res2 = run_experiment(cfg2)

    assert res1.dataset_size == res2.dataset_size
    assert res1.n_train == res2.n_train
    assert res1.n_test == res2.n_test

    for model in ["threshold", "logistic_regression", "numpy_mlp"]:
        m1 = res1.model_results[model]
        m2 = res2.model_results[model]
        assert m1["accuracy"] == pytest.approx(m2["accuracy"])
        assert m1["precision"] == pytest.approx(m2["precision"])
        assert m1["recall"] == pytest.approx(m2["recall"])
        assert m1["f1"] == pytest.approx(m2["f1"])
        assert m1["pr_auc"] == pytest.approx(m2["pr_auc"])
        assert m1["average_precision"] == pytest.approx(m2["average_precision"])
