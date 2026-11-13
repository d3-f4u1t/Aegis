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


def load_dataset(
    events_path: str | Path,
    window_config: WindowConfig | None = None,
    attack_client_hashes: Sequence[str] | None = None,
) -> Dataset:
    """Load events from JSONL, aggregate into windows, extract features, label.

    Labels are assigned at the window level: a window is labelled 1 (attack)
    if its client IP hash is in ``attack_client_hashes``, otherwise 0 (normal).

    Args:
        events_path: Path to JSONL telemetry events file.
        window_config: Window aggregation config (default 60s windows).
        attack_client_hashes: Set of client IP hashes considered attack-like.
            If None, all windows are labelled 0.

    Returns:
        A Dataset with features and labels.
    """
    if window_config is None:
        window_config = WindowConfig()
    if attack_client_hashes is None:
        attack_client_hashes = set()

    events = parse_jsonl_file(str(events_path))
    windows = aggregate_windows(events, window_config)

    feature_names = (
        ["request_" + f for _ in range(0)]  # placeholder, replaced below
    )
    # Window-level samples use window feature order only.
    # Request-level and window-level features are separate observation units
    # and must not be combined into one vector (see configs/feature_schema.yaml).
    from telemetry.feature_extractor import WINDOW_FEATURE_ORDER

    feature_names = list(WINDOW_FEATURE_ORDER)

    samples: list[WindowSample] = []
    for window in windows:
        features = window_to_vector(window)
        label = 1 if window.client_ip_hash in attack_client_hashes else 0
        samples.append(
            WindowSample(
                features=features,
                label=label,
                client_ip_hash=window.client_ip_hash,
                window_start_utc=window.window_start_utc,
                source="attack" if label == 1 else "normal",
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
