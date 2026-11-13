"""Tests for dataset splitting, client leakage prevention, and scenario hold-out."""

import pytest
from training.dataset import (
    Dataset,
    WindowSample,
    split_by_client,
    split_scenario_held_out,
    split_stratified_by_client,
)


def _make_dummy_dataset() -> Dataset:
    samples = []
    # 3 normal casual clients
    for c in ["norm_cas_1", "norm_cas_2", "norm_cas_3"]:
        for w in range(5):
            samples.append(
                WindowSample(
                    features=[float(w)] * 13,
                    label=0,
                    client_ip_hash=c,
                    window_start_utc=f"2026-11-14T00:0{w}:00+00:00",
                    source="normal",
                    scenario="normal_casual_browsing",
                )
            )
    # 3 normal polling clients
    for c in ["norm_pol_1", "norm_pol_2", "norm_pol_3"]:
        for w in range(5):
            samples.append(
                WindowSample(
                    features=[float(w)] * 13,
                    label=0,
                    client_ip_hash=c,
                    window_start_utc=f"2026-11-14T00:0{w}:00+00:00",
                    source="normal",
                    scenario="normal_api_polling",
                )
            )
    # 3 attack flood clients
    for c in ["atk_fld_1", "atk_fld_2", "atk_fld_3"]:
        for w in range(5):
            samples.append(
                WindowSample(
                    features=[float(w)] * 13,
                    label=1,
                    client_ip_hash=c,
                    window_start_utc=f"2026-11-14T00:0{w}:00+00:00",
                    source="attack",
                    scenario="attack_request_flood",
                )
            )
    # 3 attack scan clients
    for c in ["atk_scn_1", "atk_scn_2", "atk_scn_3"]:
        for w in range(5):
            samples.append(
                WindowSample(
                    features=[float(w)] * 13,
                    label=1,
                    client_ip_hash=c,
                    window_start_utc=f"2026-11-14T00:0{w}:00+00:00",
                    source="attack",
                    scenario="attack_slow_path_enumeration",
                )
            )

    return Dataset(
        samples=samples,
        feature_names=[f"f{i}" for i in range(13)],
        schema_version="1",
        n_features=13,
    )


def test_client_splitting_zero_client_leakage():
    ds = _make_dummy_dataset()
    train, val, test = split_by_client(ds, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    tr_clients = {s.client_ip_hash for s in train.samples}
    va_clients = {s.client_ip_hash for s in val.samples}
    te_clients = {s.client_ip_hash for s in test.samples}

    assert tr_clients.isdisjoint(va_clients)
    assert tr_clients.isdisjoint(te_clients)
    assert va_clients.isdisjoint(te_clients)
    assert tr_clients | va_clients | te_clients == {s.client_ip_hash for s in ds.samples}


def test_stratified_client_splitting_preserves_scenarios():
    ds = _make_dummy_dataset()
    train, val, test = split_stratified_by_client(ds, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    tr_clients = {s.client_ip_hash for s in train.samples}
    va_clients = {s.client_ip_hash for s in val.samples}
    te_clients = {s.client_ip_hash for s in test.samples}

    # Zero leakage check
    assert tr_clients.isdisjoint(va_clients)
    assert tr_clients.isdisjoint(te_clients)
    assert va_clients.isdisjoint(te_clients)

    # All scenarios must be represented in train
    assert train.unique_scenarios == ds.unique_scenarios


def test_scenario_held_out_strict_isolation():
    ds = _make_dummy_dataset()
    held_out = ["attack_slow_path_enumeration", "normal_api_polling"]

    train, val, test = split_scenario_held_out(ds, held_out_scenarios=held_out, seed=42)

    # Held out scenarios must NEVER appear in train or validation
    assert "attack_slow_path_enumeration" not in train.unique_scenarios
    assert "attack_slow_path_enumeration" not in val.unique_scenarios
    assert "normal_api_polling" not in train.unique_scenarios
    assert "normal_api_polling" not in val.unique_scenarios

    # Must be in test
    assert "attack_slow_path_enumeration" in test.unique_scenarios
    assert "normal_api_polling" in test.unique_scenarios

    # Known client IDs must not leak
    tr_clients = {s.client_ip_hash for s in train.samples}
    va_clients = {s.client_ip_hash for s in val.samples}
    assert tr_clients.isdisjoint(va_clients)
