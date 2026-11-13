"""Tests for the window aggregator."""

import pytest

from telemetry.parsers.log_parser import NormalizedEvent
from telemetry.windows.window_aggregator import WindowConfig, aggregate_windows


def _make_event(
    client_ip: str = "1.1.1.1",
    timestamp: str = "2026-11-13T10:00:00+00:00",
    method: str = "GET",
    path: str = "/",
    status: int = 200,
    content_length: int = 100,
    duration_ms: float = 10.0,
    is_auth_failure: bool = False,
) -> NormalizedEvent:
    return NormalizedEvent(
        timestamp_utc=timestamp,
        client_ip_hash=client_ip,
        http_method=method,
        path=path,
        path_length=len(path),
        query_param_count=0,
        status_code=status,
        response_bytes=content_length,
        request_duration_ms=duration_ms,
        content_length=content_length,
        header_count=5,
        is_auth_failure=is_auth_failure,
    )


def test_single_window_single_client():
    events = [
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:00+00:00", path="/a", status=200),
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:30+00:00", path="/b", status=200),
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:45+00:00", path="/a", status=404),
    ]
    records = aggregate_windows(events, WindowConfig(duration_seconds=60.0))
    assert len(records) == 1
    rec = records[0]
    assert rec.client_ip_hash == events[0].client_ip_hash
    assert rec.requests_per_window == 3
    assert rec.unique_paths == 2
    assert rec.unique_methods == 1
    assert rec.status_4xx_count == 1
    assert rec.status_5xx_count == 0
    assert rec.status_4xx_ratio() == pytest.approx(1.0 / 3.0)
    assert rec.status_5xx_ratio() == pytest.approx(0.0)
    assert rec.request_rate_per_second() == pytest.approx(3.0 / 60.0)


def test_multiple_clients_separate_windows():
    events = [
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:00+00:00"),
        _make_event(client_ip="B", timestamp="2026-11-13T10:00:00+00:00"),
    ]
    records = aggregate_windows(events, WindowConfig(duration_seconds=60.0))
    assert len(records) == 2
    client_hashes = {r.client_ip_hash for r in records}
    assert client_hashes == {events[0].client_ip_hash, events[1].client_ip_hash}


def test_events_in_different_windows():
    events = [
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:00+00:00"),
        _make_event(client_ip="A", timestamp="2026-11-13T10:01:10+00:00"),
    ]
    records = aggregate_windows(events, WindowConfig(duration_seconds=60.0))
    assert len(records) == 2


def test_auth_failure_count():
    events = [
        _make_event(client_ip="A", status=401, is_auth_failure=True),
        _make_event(client_ip="A", status=401, is_auth_failure=True),
        _make_event(client_ip="A", status=200, is_auth_failure=False),
    ]
    records = aggregate_windows(events)
    assert records[0].auth_failure_count == 2


def test_mean_inter_request_interval_requires_two_events():
    events = [_make_event(client_ip="A", timestamp="2026-11-13T10:00:00+00:00")]
    records = aggregate_windows(events)
    assert records[0].mean_inter_request_interval_ms() == 0.0

    events = [
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:00+00:00"),
        _make_event(client_ip="A", timestamp="2026-11-13T10:00:10+00:00"),
    ]
    records = aggregate_windows(events)
    assert records[0].mean_inter_request_interval_ms() == pytest.approx(10000.0)


def test_empty_events_returns_empty():
    assert aggregate_windows([]) == []
