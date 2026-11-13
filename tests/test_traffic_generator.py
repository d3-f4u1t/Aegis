"""Tests for realistic synthetic traffic generator."""

from datetime import datetime, timezone
import random
import pytest

from training.traffic_generator import (
    ALL_SCENARIOS,
    ATTACK_SCENARIOS,
    NORMAL_SCENARIOS,
    GeneratorPlan,
    generate_dataset_events,
    generate_realistic_ip_pool,
    generate_scenario_events,
)


def test_scenario_counts():
    assert len(NORMAL_SCENARIOS) == 6
    assert len(ATTACK_SCENARIOS) == 7
    assert len(ALL_SCENARIOS) == 13


def test_ip_pool_generation():
    pool = generate_realistic_ip_pool(20, seed=42)
    assert len(pool) == 20
    # Must have varied subnets
    subnets = {ip.rsplit(".", 1)[0] for ip in pool}
    assert len(subnets) >= 4


def test_all_scenarios_generate_valid_events():
    start = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
    for scenario in ALL_SCENARIOS:
        rng = random.Random(123)
        events = generate_scenario_events(
            scenario=scenario,
            client_ip="192.168.1.50",
            start_time=start,
            duration_seconds=300.0,
            rng=rng,
        )
        assert len(events) > 0
        is_attack = scenario.startswith("attack_")
        expected_label = 1 if is_attack else 0

        for ev in events:
            assert ev["client_ip"] == "192.168.1.50"
            assert ev["scenario"] == scenario
            assert ev["label"] == expected_label
            assert "timestamp_utc" in ev
            assert "http_method" in ev
            assert "path" in ev
            assert isinstance(ev["status_code"], int)
            assert isinstance(ev["response_bytes"], int)
            assert isinstance(ev["request_duration_ms"], float)
            assert isinstance(ev["content_length"], int)
            assert isinstance(ev["header_count"], int)
            assert isinstance(ev["is_auth_failure"], bool)


def test_normal_api_polling_characteristics():
    start = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
    rng = random.Random(42)
    events = generate_scenario_events(
        scenario="normal_api_polling",
        client_ip="192.168.1.50",
        start_time=start,
        duration_seconds=600.0,
        rng=rng,
    )
    # High frequency, 0 errors, single path
    assert len(events) >= 150
    assert all(ev["status_code"] == 200 for ev in events)
    assert all(not ev["is_auth_failure"] for ev in events)
    unique_paths = {ev["path"] for ev in events}
    assert len(unique_paths) == 1


def test_attack_slow_enumeration_characteristics():
    start = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
    rng = random.Random(42)
    events = generate_scenario_events(
        scenario="attack_slow_path_enumeration",
        client_ip="192.168.1.50",
        start_time=start,
        duration_seconds=600.0,
        rng=rng,
    )
    # Low frequency, high 4xx ratio, sensitive paths
    assert len(events) <= 100
    four_xx = [ev for ev in events if 400 <= ev["status_code"] < 500]
    assert len(four_xx) / len(events) >= 0.60


def test_attack_brute_force_characteristics():
    start = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
    rng = random.Random(42)
    events = generate_scenario_events(
        scenario="attack_brute_force",
        client_ip="192.168.1.50",
        start_time=start,
        duration_seconds=600.0,
        rng=rng,
    )
    auth_failures = [ev for ev in events if ev["is_auth_failure"]]
    assert len(auth_failures) / len(events) >= 0.80
    assert all(ev["http_method"] == "POST" for ev in events)


def test_attack_high_volume_scraping_characteristics():
    start = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
    rng = random.Random(42)
    events = generate_scenario_events(
        scenario="attack_high_volume_scraping",
        client_ip="192.168.1.50",
        start_time=start,
        duration_seconds=600.0,
        rng=rng,
    )
    # High frequency, diverse paths, NO 4xx/5xx errors
    assert len(events) >= 300
    assert all(ev["status_code"] == 200 for ev in events)
    unique_paths = {ev["path"] for ev in events}
    assert len(unique_paths) >= 50


def test_generator_determinism():
    plan1 = GeneratorPlan(
        name="test", seed=77, duration_seconds=120.0,
        clients_per_scenario={"normal_casual_browsing": 2, "attack_request_flood": 2}
    )
    plan2 = GeneratorPlan(
        name="test", seed=77, duration_seconds=120.0,
        clients_per_scenario={"normal_casual_browsing": 2, "attack_request_flood": 2}
    )
    evs1 = generate_dataset_events(plan1)
    evs2 = generate_dataset_events(plan2)
    assert len(evs1) == len(evs2)
    for e1, e2 in zip(evs1, evs2):
        assert e1 == e2
