"""Dataset loading and splitting for telemetry experiments.

Loads JSONL telemetry events, parses them, aggregates into behavioural windows,
extracts feature vectors, and assigns labels. Splits are performed at the
client level to avoid leakage: all windows from one client go entirely into
train, validation, or test — never split across sets.

This prevents a client's behaviour from appearing in both training and test,
which would inflate performance estimates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from telemetry.feature_extractor import (
    event_to_vector,
    n_full_features,
    n_request_features,
    n_window_features,
    window_to_vector,
)
from telemetry.parsers.log_parser import NormalizedEvent, parse_jsonl_file
from telemetry.windows.window_aggregator import WindowConfig, aggregate_windows
from training.preprocessing import StandardScaler


@dataclass
class WindowSample:
    """One window sample with features and label."""

    features: list[float]
    label: int  # 0 = normal, 1 = attack/anomalous
    client_ip_hash: str
    window_start_utc: str
    source: str  # "normal" or "attack"
    scenario: str = "unknown"
    attack_event_count: int = 0
    total_events: int = 0


@dataclass
class Dataset:
    """A labelled dataset of window samples."""

    samples: list[WindowSample]
    feature_names: list[str]
    schema_version: str = "1"
    n_features: int = 0

    @property
    def X(self) -> list[list[float]]:
        return [s.features for s in self.samples]

    @property
    def y(self) -> list[float]:
        return [float(s.label) for s in self.samples]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_positive(self) -> int:
        return sum(1 for s in self.samples if s.label == 1)

    @property
    def n_negative(self) -> int:
        return sum(1 for s in self.samples if s.label == 0)

    @property
    def scenarios(self) -> list[str]:
        return [s.scenario for s in self.samples]

    @property
    def unique_scenarios(self) -> set[str]:
        return {s.scenario for s in self.samples}

    def filter_by_scenario(self, scenarios: Sequence[str]) -> Dataset:
        scenario_set = set(scenarios)
        return Dataset(
            samples=[s for s in self.samples if s.scenario in scenario_set],
            feature_names=list(self.feature_names),
            schema_version=self.schema_version,
            n_features=self.n_features,
        )

    def exclude_scenarios(self, scenarios: Sequence[str]) -> Dataset:
        scenario_set = set(scenarios)
        return Dataset(
            samples=[s for s in self.samples if s.scenario not in scenario_set],
            feature_names=list(self.feature_names),
            schema_version=self.schema_version,
            n_features=self.n_features,
        )


def load_dataset(
    events_path: str | Path,
    window_config: WindowConfig | None = None,
    attack_client_hashes: Sequence[str] | None = None,
) -> Dataset:
    """Load events from JSONL, aggregate into windows, extract features, label.

    Labels and scenarios are assigned at the window level:
    1. If the window contains explicit attack events (from event metadata),
       or its client IP hash is in ``attack_client_hashes``, the window is
       labelled 1 (attack).
    2. Otherwise, the window is labelled 0 (normal).
    3. The scenario is determined by the dominant scenario among the window's
       events.

    Args:
        events_path: Path to JSONL telemetry events file.
        window_config: Window aggregation config (default 60s windows).
        attack_client_hashes: Optional set of client IP hashes considered attack-like.

    Returns:
        A Dataset with features and labels.
    """
    if window_config is None:
        window_config = WindowConfig()
    if attack_client_hashes is None:
        attack_client_hashes = set()
    else:
        attack_client_hashes = set(attack_client_hashes)

    events = parse_jsonl_file(str(events_path))
    windows = aggregate_windows(events, window_config)

    from telemetry.feature_extractor import WINDOW_FEATURE_ORDER

    feature_names = list(WINDOW_FEATURE_ORDER)

    samples: list[WindowSample] = []
    for window in windows:
        features = window_to_vector(window)

        is_attack = window.has_attack or (bool(attack_client_hashes) and window.client_ip_hash in attack_client_hashes)
        label = 1 if is_attack else 0
        source = "attack" if label == 1 else "normal"

        if window.scenarios:
            dominant_scenario = max(set(window.scenarios), key=window.scenarios.count)
        else:
            dominant_scenario = "synthetic_attack" if label == 1 else "synthetic_normal"

        samples.append(
            WindowSample(
                features=features,
                label=label,
                client_ip_hash=window.client_ip_hash,
                window_start_utc=window.window_start_utc,
                source=source,
                scenario=dominant_scenario,
                attack_event_count=window.attack_event_count,
                total_events=window.requests_per_window,
            )
        )

    return Dataset(
        samples=samples,
        feature_names=feature_names,
        schema_version="1",
        n_features=len(feature_names),
    )


def split_by_client(
    dataset: Dataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/validation/test at the client level.

    All windows from one client go entirely into one split. This prevents
    leakage where a client's behaviour appears in both training and test.

    Args:
        dataset: The full dataset.
        train_ratio: Fraction of clients for training.
        val_ratio: Fraction of clients for validation.
        test_ratio: Fraction of clients for testing.
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Group samples by client IP hash.
    clients: dict[str, list[WindowSample]] = {}
    for sample in dataset.samples:
        clients.setdefault(sample.client_ip_hash, []).append(sample)

    client_ids = sorted(clients.keys())
    rng = random.Random(seed)
    rng.shuffle(client_ids)

    n_clients = len(client_ids)
    n_train = int(n_clients * train_ratio)
    n_val = int(n_clients * val_ratio)
    # test gets the rest

    train_clients = set(client_ids[:n_train])
    val_clients = set(client_ids[n_train : n_train + n_val])
    test_clients = set(client_ids[n_train + n_val :])

    def _build(client_set: set[str]) -> Dataset:
        samples = [s for s in dataset.samples if s.client_ip_hash in client_set]
        return Dataset(
            samples=samples,
            feature_names=list(dataset.feature_names),
            schema_version=dataset.schema_version,
            n_features=dataset.n_features,
        )

    return _build(train_clients), _build(val_clients), _build(test_clients)


def split_stratified_by_client(
    dataset: Dataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split dataset at client level with stratification by dominant scenario.

    Ensures that every behavioural scenario has clients represented across
    splits according to the specified ratios, while ensuring all windows
    from any single client go exclusively to one split (zero client leakage).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Map client_id to (dominant_scenario, list_of_samples)
    client_samples: dict[str, list[WindowSample]] = {}
    for sample in dataset.samples:
        client_samples.setdefault(sample.client_ip_hash, []).append(sample)

    scenario_to_clients: dict[str, list[str]] = {}
    for client_id, samples in client_samples.items():
        # Dominant scenario for this client
        scenarios = [s.scenario for s in samples]
        dom_scenario = max(set(scenarios), key=scenarios.count)
        scenario_to_clients.setdefault(dom_scenario, []).append(client_id)

    train_clients: set[str] = set()
    val_clients: set[str] = set()
    test_clients: set[str] = set()

    rng = random.Random(seed)

    for scenario, c_ids in sorted(scenario_to_clients.items()):
        c_sorted = sorted(c_ids)
        rng.shuffle(c_sorted)
        n = len(c_sorted)

        if n == 1:
            # Single client for this scenario: default to train
            train_clients.add(c_sorted[0])
        elif n == 2:
            train_clients.add(c_sorted[0])
            test_clients.add(c_sorted[1])
        else:
            n_tr = max(1, int(round(n * train_ratio)))
            n_va = max(1, int(round(n * val_ratio)))
            # If total allocations exceed or match n, ensure at least 1 in test if possible
            if n_tr + n_va >= n and n >= 3:
                n_tr = n - 2
                n_va = 1
            train_clients.update(c_sorted[:n_tr])
            val_clients.update(c_sorted[n_tr : n_tr + n_va])
            test_clients.update(c_sorted[n_tr + n_va :])

    def _build(client_set: set[str]) -> Dataset:
        samples = [s for s in dataset.samples if s.client_ip_hash in client_set]
        return Dataset(
            samples=samples,
            feature_names=list(dataset.feature_names),
            schema_version=dataset.schema_version,
            n_features=dataset.n_features,
        )

    return _build(train_clients), _build(val_clients), _build(test_clients)


def split_scenario_held_out(
    dataset: Dataset,
    held_out_scenarios: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split dataset holding out specified scenarios exclusively for testing.

    Windows belonging to ``held_out_scenarios`` are completely withheld from
    train and validation, appearing only in the test set. Clients behaving under
    known scenarios are split across train/val/test with stratification.

    This tests the model's ability to detect novel attack patterns or handle
    unseen normal behaviours without prior exposure.
    """
    held_out_set = set(held_out_scenarios)
    held_out_samples = [s for s in dataset.samples if s.scenario in held_out_set]
    training_samples = [s for s in dataset.samples if s.scenario not in held_out_set]

    train_base = Dataset(
        samples=training_samples,
        feature_names=list(dataset.feature_names),
        schema_version=dataset.schema_version,
        n_features=dataset.n_features,
    )

    tr_ds, val_ds, te_base = split_stratified_by_client(
        train_base,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Test set combines held-out scenario samples with the held-out clients from known scenarios
    combined_test_samples = te_base.samples + held_out_samples

    test_ds = Dataset(
        samples=combined_test_samples,
        feature_names=list(dataset.feature_names),
        schema_version=dataset.schema_version,
        n_features=dataset.n_features,
    )

    return tr_ds, val_ds, test_ds


def fit_preprocessing(
    train_dataset: Dataset,
) -> StandardScaler:
    """Fit preprocessing (StandardScaler) on training data only.

    Returns the fitted scaler. Validation and test data must use
    ``scaler.transform()`` with these same statistics.
    """
    scaler = StandardScaler()
    scaler.fit(train_dataset.X, train_dataset.feature_names, train_dataset.schema_version)
    return scaler


def preprocess_split(
    scaler: StandardScaler,
    train: Dataset,
    val: Dataset | None = None,
    test: Dataset | None = None,
) -> tuple[list[list[float]], ...]:
    """Apply fitted preprocessing to train/val/test datasets.

    Returns transformed datasets with the same labels and metadata.
    """
    train_X = scaler.fit_transform(train.X, train.feature_names, train.schema_version)

    result = [train_X, train.y]

    if val is not None:
        val_X = scaler.transform(val.X)
        result.extend([val_X, val.y])
    if test is not None:
        test_X = scaler.transform(test.X)
        result.extend([test_X, test.y])

    return tuple(result)
