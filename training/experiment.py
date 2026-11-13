"""Experiment harness for reproducible training and evaluation.

Provides:

* ``ExperimentConfig`` — a reproducible experiment specification (seed, model
  type, hyperparameters, data paths, split ratios).
* ``run_experiment`` — executes an experiment: load data, split, preprocess,
  train, evaluate, save artifacts.
* ``ExperimentResult`` — structured results including metrics, artifacts, and
  configuration hash.

All training uses the from-scratch engines in ``nn/`` and ``nn_numpy/``.
No deep-learning frameworks are used.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from nn.losses import BinaryCrossEntropyWithLogits as ScalarBCELogits
from nn.model import MLP as ScalarMLP
from nn.optimizers import Adam as ScalarAdam
from nn.serialization import save_model as save_scalar_model
from training.train import train as scalar_train

from nn_numpy.losses import BinaryCrossEntropyWithLogits as NumpyBCELogits
from nn_numpy.model import MLP as NumpyMLP
from nn_numpy.optimizers import Adam as NumpyAdam
from nn_numpy.serialization import save_model as save_nn_numpy_model
from nn_numpy.training import train as numpy_train

from training.dataset import (
    Dataset,
    fit_preprocessing,
    load_dataset,
    preprocess_split,
    split_by_client,
    split_scenario_held_out,
    split_stratified_by_client,
)
from telemetry.windows.window_aggregator import WindowConfig
from training.evaluate import (
    average_precision,
    binary_metrics,
    operating_point,
    pr_auc,
    threshold_sweep,
)
from training.logistic_regression import LogisticRegression
from training.threshold_baseline import ThresholdBaseline


@dataclass
class ExperimentConfig:
    """Reproducible experiment specification."""

    experiment_name: str
    seed: int = 42
    window_duration_seconds: float = 60.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Model selection: "mlp", "numpy_mlp", "scalar_mlp", "logistic_regression", "threshold"
    models: list[str] = field(default_factory=lambda: ["mlp", "logistic_regression", "threshold"])

    # Splitting strategy: "client", "stratified_client", or "scenario_held_out"
    split_strategy: str = "stratified_client"
    held_out_scenarios: list[str] = field(default_factory=list)

    # MLP hyperparameters
    mlp_hidden_sizes: list[int] = field(default_factory=lambda: [16, 8])
    mlp_hidden_activation: str = "relu"
    mlp_output_activation: str = "identity"
    mlp_learning_rate: float = 0.01
    mlp_epochs: int = 500
    mlp_batch_size: int = 32
    mlp_use_numpy: bool = True

    # Logistic regression hyperparameters
    lr_learning_rate: float = 0.01
    lr_epochs: int = 1000

    # Threshold baseline
    threshold_feature_index: int = 1  # requests_per_second in window features
    threshold_max_fpr: float = 0.05

    # Data paths
    events_path: str = "data/telemetry/synthetic.jsonl"
    attack_client_hashes: list[str] = field(default_factory=list)

    # Output
    output_dir: str = "experiments/results"

    def config_hash(self) -> str:
        """Return a short hash of the configuration for reproducibility tracking."""
        cfg = asdict(self)
        cfg_json = json.dumps(cfg, sort_keys=True)
        return hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:12]


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    experiment_name: str
    config_hash: str
    timestamp_utc: str
    dataset_size: int
    n_train: int
    n_val: int
    n_test: int
    n_positive_train: int
    n_positive_test: int
    n_features: int
    feature_names: list[str]
    schema_version: str

    model_results: dict[str, dict] = field(default_factory=dict)
    artifact_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a complete experiment: load, split, preprocess, train, evaluate.

    Steps:
    1. Load telemetry events from JSONL.
    2. Aggregate into windows, extract features, label windows.
    3. Split by client into train/validation/test (no leakage).
    4. Fit preprocessing (StandardScaler) on training data only.
    5. Train each configured model.
    6. Evaluate on test data.
    7. Save model artifacts and preprocessing parameters.
    8. Return structured results.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1–2: Load and label data ─────────────────────────────────────
    window_config = WindowConfig(duration_seconds=config.window_duration_seconds)

    dataset = load_dataset(
        config.events_path,
        window_config=window_config,
        attack_client_hashes=set(config.attack_client_hashes),
    )

    # ── Step 3: Split by client / scenario ──────────────────────────────────
    if config.split_strategy == "scenario_held_out":
        train_ds, val_ds, test_ds = split_scenario_held_out(
            dataset,
            held_out_scenarios=config.held_out_scenarios,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
    elif config.split_strategy == "stratified_client":
        train_ds, val_ds, test_ds = split_stratified_by_client(
            dataset,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
    else:
        train_ds, val_ds, test_ds = split_by_client(
            dataset,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )

    # ── Step 4: Fit preprocessing on training data only ───────────────────
    scaler = fit_preprocessing(train_ds)
    train_X, train_y, val_X, val_y, test_X, test_y = preprocess_split(
        scaler, train_ds, val_ds, test_ds
    )

    n_features = len(train_ds.feature_names)

    # ── Step 5–6: Train and evaluate each model ───────────────────────────
    model_results: dict[str, dict] = {}
    artifact_paths: list[str] = []

    for model_name in config.models:
        result = _train_and_evaluate(
            model_name=model_name,
            config=config,
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            test_scenarios=test_ds.scenarios,
            feature_names=train_ds.feature_names,
            schema_version=train_ds.schema_version,
            output_dir=output_dir,
        )
        model_results[model_name] = result["metrics"]
        if result.get("artifact_path"):
            artifact_paths.append(result["artifact_path"])

    # ── Step 7–8: Save preprocessing and return results ───────────────────
    preprocessing_path = output_dir / f"{config.experiment_name}_preprocessing.json"
    scaler.save(preprocessing_path)
    artifact_paths.append(str(preprocessing_path))

    return ExperimentResult(
        experiment_name=config.experiment_name,
        config_hash=config.config_hash(),
        timestamp_utc=timestamp,
        dataset_size=dataset.n_samples,
        n_train=len(train_ds.samples),
        n_val=len(val_ds.samples),
        n_test=len(test_ds.samples),
        n_positive_train=train_ds.n_positive,
        n_positive_test=test_ds.n_positive,
        n_features=n_features,
        feature_names=train_ds.feature_names,
        schema_version=train_ds.schema_version,
        model_results=model_results,
        artifact_paths=artifact_paths,
    )


def _compute_per_scenario_metrics(
    y_true: list[int],
    preds: list[int],
    scenarios: list[str],
) -> dict[str, dict[str, float]]:
    """Compute detection breakdown across distinct behavioural scenarios."""
    breakdown: dict[str, dict[str, float]] = {}
    for scen in sorted(set(scenarios)):
        indices = [i for i, s in enumerate(scenarios) if s == scen]
        sub_true = [y_true[i] for i in indices]
        sub_preds = [preds[i] for i in indices]
        total = len(indices)
        correct = sum(1 for yt, yp in zip(sub_true, sub_preds) if yt == yp)
        tp = sum(1 for yt, yp in zip(sub_true, sub_preds) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(sub_true, sub_preds) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(sub_true, sub_preds) if yt == 1 and yp == 0)
        tn = sum(1 for yt, yp in zip(sub_true, sub_preds) if yt == 0 and yp == 0)
        pos = sum(sub_true)
        breakdown[scen] = {
            "total_samples": float(total),
            "accuracy": correct / total if total > 0 else 0.0,
            "recall": tp / pos if pos > 0 else (1.0 if fn == 0 else 0.0),
            "false_positive_rate": fp / (total - pos) if (total - pos) > 0 else 0.0,
            "tp": float(tp),
            "fp": float(fp),
            "tn": float(tn),
            "fn": float(fn),
        }
    return breakdown


def _train_and_evaluate(
    model_name: str,
    config: ExperimentConfig,
    train_X: list[list[float]],
    train_y: list[float],
    val_X: list[list[float]],
    val_y: list[float],
    test_X: list[list[float]],
    test_y: list[float],
    test_scenarios: list[str],
    feature_names: list[str],
    schema_version: str,
    output_dir: Path,
) -> dict:
    """Train one model and evaluate on test data."""

    if model_name == "threshold":
        return _run_threshold_baseline(
            config, train_X, train_y, val_X, val_y, test_X, test_y,
            test_scenarios, feature_names, schema_version, output_dir,
        )

    if model_name == "logistic_regression":
        return _run_logistic_regression(
            config, train_X, train_y, val_X, val_y, test_X, test_y,
            test_scenarios, feature_names, schema_version, output_dir,
        )

    if model_name in ("mlp", "numpy_mlp"):
        return _run_mlp(
            config, train_X, train_y, val_X, val_y, test_X, test_y,
            test_scenarios, feature_names, schema_version, output_dir,
            use_numpy=True,
        )

    if model_name == "scalar_mlp":
        return _run_mlp(
            config, train_X, train_y, val_X, val_y, test_X, test_y,
            test_scenarios, feature_names, schema_version, output_dir,
            use_numpy=False,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _run_threshold_baseline(
    config, train_X, train_y, val_X, val_y, test_X, test_y,
    test_scenarios, feature_names, schema_version, output_dir,
) -> dict:
    model = ThresholdBaseline(
        feature_index=config.threshold_feature_index,
    )
    model.fit(
        val_X, val_y,
        feature_names=feature_names,
        max_fpr=config.threshold_max_fpr,
    )

    test_scores = model.predict_proba(test_X)
    test_preds = [1 if s >= model.threshold else 0 for s in test_scores]
    y_true_int = [int(y) for y in test_y]

    # Benchmark latency
    t0 = time.perf_counter()
    for _ in range(20):
        _ = model.predict(test_X)
    t1 = time.perf_counter()
    latency_ms = ((t1 - t0) / (20 * max(1, len(test_X)))) * 1000.0

    metrics = binary_metrics(y_true_int, test_scores, threshold=model.threshold)
    metrics["threshold"] = model.threshold
    metrics["feature_name"] = model.feature_name
    metrics["pr_auc"] = pr_auc(y_true_int, test_scores)
    metrics["average_precision"] = average_precision(y_true_int, test_scores)
    metrics["inference_latency_ms_per_sample"] = latency_ms
    metrics["per_scenario"] = _compute_per_scenario_metrics(y_true_int, test_preds, test_scenarios)

    params_path = output_dir / f"{config.experiment_name}_threshold.json"
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(model.get_params(), fh, indent=2)
        fh.write("\n")

    return {"metrics": metrics, "artifact_path": str(params_path)}


def _run_logistic_regression(
    config, train_X, train_y, val_X, val_y, test_X, test_y,
    test_scenarios, feature_names, schema_version, output_dir,
) -> dict:
    model = LogisticRegression(
        lr=config.lr_learning_rate,
        epochs=config.lr_epochs,
        seed=config.seed,
    )
    model.fit(train_X, train_y, feature_names=feature_names, schema_version=schema_version)

    test_probs = model.predict_proba(test_X).tolist()
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    y_true_int = [int(y) for y in test_y]

    # Benchmark latency
    t0 = time.perf_counter()
    for _ in range(20):
        _ = model.predict(test_X)
    t1 = time.perf_counter()
    latency_ms = ((t1 - t0) / (20 * max(1, len(test_X)))) * 1000.0

    metrics = binary_metrics(y_true_int, test_probs, threshold=0.5)
    metrics["pr_auc"] = pr_auc(y_true_int, test_probs)
    metrics["average_precision"] = average_precision(y_true_int, test_probs)
    metrics["inference_latency_ms_per_sample"] = latency_ms
    metrics["per_scenario"] = _compute_per_scenario_metrics(y_true_int, test_preds, test_scenarios)

    params_path = output_dir / f"{config.experiment_name}_logistic_regression.json"
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(model.get_params(), fh, indent=2)
        fh.write("\n")

    return {"metrics": metrics, "artifact_path": str(params_path)}


def _run_mlp(
    config, train_X, train_y, val_X, val_y, test_X, test_y,
    test_scenarios, feature_names, schema_version, output_dir,
    use_numpy: bool = True,
) -> dict:
    sizes = [len(train_X[0])] + list(config.mlp_hidden_sizes) + [1]
    y_true_int = [int(y) for y in test_y]

    if use_numpy:
        model = NumpyMLP(
            sizes=sizes,
            hidden_activation=config.mlp_hidden_activation,
            output_activation=config.mlp_output_activation,
            seed=config.seed,
        )
        loss_fn = NumpyBCELogits()
        optimizer = NumpyAdam(lr=config.mlp_learning_rate)

        history = numpy_train(
            model,
            np.asarray(train_X, dtype=float),
            np.asarray(train_y, dtype=float),
            loss_fn,
            optimizer,
            epochs=config.mlp_epochs,
            batch_size=config.mlp_batch_size,
            shuffle=True,
            seed=config.seed,
            X_val=np.asarray(val_X, dtype=float),
            y_val=np.asarray(val_y, dtype=float),
        )

        test_X_arr = np.asarray(test_X, dtype=float)
        test_probs_raw = model.predict_proba(test_X_arr).tolist()
        # predict_proba on 2D input returns [[p1], [p2], ...]; flatten to [p1, p2, ...]
        test_probs = [float(p[0]) if isinstance(p, list) else float(p) for p in test_probs_raw]
        test_preds = [1 if p >= 0.5 else 0 for p in test_probs]

        t0 = time.perf_counter()
        for _ in range(20):
            _ = model.predict(test_X_arr)
        t1 = time.perf_counter()
        latency_ms = ((t1 - t0) / (20 * max(1, len(test_X)))) * 1000.0

        model_path = output_dir / f"{config.experiment_name}_numpy_mlp.json"
        save_nn_numpy_model(model, model_path, metadata={
            "experiment": config.experiment_name,
            "config_hash": config.config_hash(),
            "schema_version": schema_version,
            "feature_names": feature_names,
        })
        artifact_path = str(model_path)

    else:
        model = ScalarMLP(
            sizes=sizes,
            hidden_activation=config.mlp_hidden_activation,
            output_activation=config.mlp_output_activation,
            seed=config.seed,
        )
        loss_fn = ScalarBCELogits()
        optimizer = ScalarAdam(lr=config.mlp_learning_rate)

        history = scalar_train(
            model,
            train_X,
            train_y,
            loss_fn,
            optimizer,
            epochs=config.mlp_epochs,
            batch_size=config.mlp_batch_size,
            shuffle=True,
            seed=config.seed,
            X_val=val_X,
            y_val=val_y,
        )

        test_probs = [float(model.predict_proba(x)) for x in test_X]
        test_preds = [1 if p >= 0.5 else 0 for p in test_probs]

        t0 = time.perf_counter()
        for _ in range(5):
            _ = [model.predict(x) for x in test_X]
        t1 = time.perf_counter()
        latency_ms = ((t1 - t0) / (5 * max(1, len(test_X)))) * 1000.0

        model_path = output_dir / f"{config.experiment_name}_scalar_mlp.json"
        save_scalar_model(model, model_path, metadata={
            "experiment": config.experiment_name,
            "config_hash": config.config_hash(),
            "schema_version": schema_version,
            "feature_names": feature_names,
        })
        artifact_path = str(model_path)

    metrics = binary_metrics(y_true_int, test_probs, threshold=0.5)
    metrics["pr_auc"] = pr_auc(y_true_int, test_probs)
    metrics["average_precision"] = average_precision(y_true_int, test_probs)
    metrics["inference_latency_ms_per_sample"] = latency_ms
    metrics["per_scenario"] = _compute_per_scenario_metrics(y_true_int, test_preds, test_scenarios)
    metrics["final_train_loss"] = history.loss[-1] if history.loss else None
    if history.val_loss:
        metrics["final_val_loss"] = history.val_loss[-1]

    return {"metrics": metrics, "artifact_path": artifact_path}
