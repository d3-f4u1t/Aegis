"""Tests for the log event parser."""

import json
import tempfile
from pathlib import Path

import pytest

from telemetry.parsers.log_parser import (
    NormalizedEvent,
    count_query_params,
    parse_jsonl_event,
    parse_jsonl_file,
    pseudonymize_client_ip,
)


def test_parse_valid_event():
    line = json.dumps({
        "timestamp_utc": "2026-11-13T10:00:00+00:00",
        "client_ip": "192.168.1.1",
        "http_method": "GET",
        "path": "/search?q=test",
        "status_code": 200,
        "response_bytes": 1234,
        "request_duration_ms": 45.5,
        "content_length": 0,
        "header_count": 12,
        "is_auth_failure": False,
    })
    event = parse_jsonl_event(line)
    assert event is not None
    assert event.client_ip_hash == pseudonymize_client_ip("192.168.1.1")
    assert event.http_method == "GET"
    assert event.path == "/search?q=test"
    assert event.path_length == 14
    assert event.query_param_count == 1
    assert event.status_code == 200
    assert event.response_bytes == 1234
    assert event.request_duration_ms == 45.5
    assert event.is_auth_failure is False


def test_parse_invalid_json():
    assert parse_jsonl_event("not json") is None
    assert parse_jsonl_event("") is None
    assert parse_jsonl_event("   ") is None


def test_parse_missing_fields_uses_defaults():
    line = json.dumps({"http_method": "POST", "path": "/login"})
    event = parse_jsonl_event(line)
    assert event is not None
    assert event.status_code == 0
    assert event.response_bytes == 0
    assert event.request_duration_ms == 0.0
    assert event.query_param_count == 0


def test_query_param_count():
    assert count_query_params("/") == 0
    assert count_query_params("/search") == 0
    assert count_query_params("/search?q=test") == 1
    assert count_query_params("/search?q=test&page=2") == 2
    assert count_query_params("/search?q=") == 1


def test_ip_pseudonymization_is_deterministic():
    h1 = pseudonymize_client_ip("192.168.1.1")
    h2 = pseudonymize_client_ip("192.168.1.1")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest
    assert h1 != pseudonymize_client_ip("192.168.1.2")


def test_parse_jsonl_file(tmp_path):
    lines = [
        json.dumps({"timestamp_utc": "2026-11-13T10:00:00+00:00", "client_ip": "1.1.1.1",
                    "http_method": "GET", "path": "/", "status_code": 200}),
        "not json",
        "",
        json.dumps({"timestamp_utc": "2026-11-13T10:01:00+00:00", "client_ip": "2.2.2.2",
                    "http_method": "POST", "path": "/login", "status_code": 401, "is_auth_failure": True}),
    ]
    path = Path(tmp_path) / "events.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")

    events = parse_jsonl_file(str(path))
    assert len(events) == 2
    assert events[0].client_ip_hash == pseudonymize_client_ip("1.1.1.1")
    assert events[1].client_ip_hash == pseudonymize_client_ip("2.2.2.2")
    assert events[1].is_auth_failure is True


def test_normalized_event_is_dataclass():
    event = NormalizedEvent(
        timestamp_utc="2026-11-13T10:00:00+00:00",
        client_ip_hash="abc",
        http_method="GET",
        path="/",
        path_length=1,
        query_param_count=0,
        status_code=200,
        response_bytes=100,
        request_duration_ms=10.0,
        content_length=0,
        header_count=5,
        is_auth_failure=False,
    )
    assert event.http_method == "GET"
    assert event.status_code == 200
