"""Phase 5 Experiment Execution Script.

Runs the complete Phase 5 experimental evaluation suite:
1. Experiment 1: In-Distribution / Unseen-Client Generalization (Threshold, LR, Scalar MLP, NumPy MLP)
2. Experiment 2: Unseen-Attack Generalization (Scenario held-out: slow enumeration & scraping)
3. Experiment 3: Benign Pattern Stress Test (Scenario held-out: high-frequency legitimate polling)

Records full metrics, confusion matrices, PR-AUC, Average Precision,
per-scenario breakdowns, and inference latency.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.experiment import ExperimentConfig, run_experiment


def run_all_phase5_experiments() -> dict[str, dict]:
    results = {}
    data_path = "data/telemetry/synthetic_phase5.jsonl"

    print("=" * 80)
    print("PHASE 5: SUPERVISED WEBSITE TRAFFIC CLASSIFICATION EXPERIMENTS")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 1: In-Distribution Benchmark (Unseen Clients)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Running Experiment 1: In-Distribution Benchmark (Disjoint Clients)...")
    config_exp1 = ExperimentConfig(
        experiment_name="phase5_exp1_indistribution",
        seed=42,
        window_duration_seconds=60.0,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        models=["threshold", "logistic_regression", "scalar_mlp", "numpy_mlp"],
        split_strategy="stratified_client",
        mlp_hidden_sizes=[16, 8],
        mlp_learning_rate=0.005,
        mlp_epochs=150,
        mlp_batch_size=32,
        lr_learning_rate=0.01,
        lr_epochs=400,
        threshold_feature_index=1,  # requests_per_second
        threshold_max_fpr=0.05,
        events_path=data_path,
        output_dir="experiments/results",
    )
    res1 = run_experiment(config_exp1)
    results["exp1_indistribution"] = res1.to_dict()
    print("  Exp 1 completed.")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 2: Scenario-Held-Out (Unseen Attack Scenarios)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Running Experiment 2: Unseen-Attack Generalization...")
    held_out_attacks = ["attack_slow_path_enumeration", "attack_high_volume_scraping"]
    config_exp2 = ExperimentConfig(
        experiment_name="phase5_exp2_unseen_attacks",
        seed=42,
        window_duration_seconds=60.0,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        models=["threshold", "logistic_regression", "numpy_mlp"],
        split_strategy="scenario_held_out",
        held_out_scenarios=held_out_attacks,
        mlp_hidden_sizes=[16, 8],
        mlp_learning_rate=0.005,
        mlp_epochs=150,
        mlp_batch_size=32,
        lr_learning_rate=0.01,
        lr_epochs=400,
        threshold_feature_index=1,
        threshold_max_fpr=0.05,
        events_path=data_path,
        output_dir="experiments/results",
    )
    res2 = run_experiment(config_exp2)
    results["exp2_unseen_attacks"] = res2.to_dict()
    print("  Exp 2 completed.")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 3: Benign Pattern Stress Test (Unseen Polling Scenario)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Running Experiment 3: Benign Pattern Stress Test (Novel Normal)...")
    held_out_normal = ["normal_api_polling"]
    config_exp3 = ExperimentConfig(
        experiment_name="phase5_exp3_unseen_polling",
        seed=42,
        window_duration_seconds=60.0,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        models=["threshold", "logistic_regression", "numpy_mlp"],
        split_strategy="scenario_held_out",
        held_out_scenarios=held_out_normal,
        mlp_hidden_sizes=[16, 8],
        mlp_learning_rate=0.005,
        mlp_epochs=150,
        mlp_batch_size=32,
        lr_learning_rate=0.01,
        lr_epochs=400,
        threshold_feature_index=1,
        threshold_max_fpr=0.05,
        events_path=data_path,
        output_dir="experiments/results",
    )
    res3 = run_experiment(config_exp3)
    results["exp3_unseen_polling"] = res3.to_dict()
    print("  Exp 3 completed.")

    # Save aggregated results
    summary_path = Path("experiments/results/phase5_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved complete experimental results to {summary_path}")

    return results


if __name__ == "__main__":
    run_all_phase5_experiments()
