"""Tests for the feature extractor."""

import pytest

from telemetry.feature_extractor import (
    event_to_vector,
    features_to_vector,
    n_full_features,
    n_request_features,
    n_window_features,
    window_to_vector,
)
from telemetry.parsers.log_parser import NormalizedEvent
from telemetry.windows.window_aggregator import WindowRecord


def _sample_event() -> NormalizedEvent:
    return NormalizedEvent(
        timestamp_utc="2026-11-13T10:00:00+00:00",
        client_ip_hash="abc123",
        http_method="GET",
        path="/search?q=test",
        path_length=14,
        query_param_count=1,
        status_code=200,
        response_bytes=1234,
        request_duration_ms=45.5,
        content_length=0,
        header_count=12,
        is_auth_failure=False,
    )


def test_request_feature_count():
    assert n_request_features() == 12


def test_window_feature_count():
    assert n_window_features() == 13


def test_full_feature_count():
    assert n_full_features() == 25


def test_event_to_vector_length_and_values():
    event = _sample_event()
    vec = event_to_vector(event)
    assert len(vec) == n_request_features()

    # GET should be one-hot
    assert vec[0] == 1.0   # http_method_GET
    assert vec[1] == 0.0   # http_method_POST
    assert vec[2] == 0.0   # http_method_PUT
    assert vec[3] == 0.0   # http_method_DELETE
    assert vec[4] == 0.0   # http_method_OTHER
    assert vec[5] == 14.0  # path_length
    assert vec[6] == 1.0   # query_param_count
    assert vec[7] == 200.0 # status_code
    assert vec[8] == 1234.0 # response_bytes
    assert vec[9] == 45.5  # request_duration_ms
    assert vec[10] == 0.0  # content_length
    assert vec[11] == 12.0 # header_count


def test_post_method_one_hot():
    event = _sample_event()
    event.http_method = "POST"
    vec = event_to_vector(event)
    assert vec[0] == 0.0
    assert vec[1] == 1.0


def test_unknown_method_is_other():
    event = _sample_event()
    event.http_method = "PATCH"
    vec = event_to_vector(event)
    assert vec[4] == 1.0  # http_method_OTHER


def test_window_to_vector_length_and_values():
    window = WindowRecord(
        client_ip_hash="abc",
        window_start_utc="2026-11-13T10:00:00+00:00",
        window_end_utc="2026-11-13T10:01:00+00:00",
        window_duration_seconds=60.0,
        requests_per_window=10,
        unique_paths=3,
        unique_methods=2,
        status_4xx_count=2,
        status_5xx_count=1,
        auth_failure_count=1,
        total_request_bytes=5000,
        max_request_size=1000,
        total_duration_ms=500.0,
        total_response_bytes=10000,
        request_timestamps=[
            "2026-11-13T10:00:00+00:00",
            "2026-11-13T10:00:10+00:00",
        ],
    )
    vec = window_to_vector(window)
    assert len(vec) == n_window_features()
    assert vec[0] == 10.0   # requests_per_window
    assert vec[1] == pytest.approx(10.0 / 60.0)  # requests_per_second
    assert vec[2] == 3.0    # unique_paths
    assert vec[3] == 2.0    # unique_methods
    assert vec[4] == 2.0    # status_4xx_count
    assert vec[5] == pytest.approx(2.0 / 10.0)  # status_4xx_ratio
    assert vec[6] == 1.0    # status_5xx_count
    assert vec[7] == pytest.approx(1.0 / 10.0)  # status_5xx_ratio
    assert vec[8] == 1.0    # auth_failure_count
    assert vec[9] == 500.0  # mean_request_size
    assert vec[10] == 1000.0  # max_request_size
    assert vec[11] == 50.0  # mean_response_time_ms
    assert vec[12] == 10000.0  # mean_inter_request_interval_ms


def test_features_to_vector_fills_missing_with_zero():
    features = {"requests_per_window": 5}
    vec = features_to_vector(features)
    # requests_per_window is the first window feature, after 12 request features
    assert vec[12] == 5.0
    assert all(v == 0.0 for i, v in enumerate(vec) if i != 12)


def test_features_to_vector_respects_order():
    features = {"b": 2.0, "a": 1.0}
    vec = features_to_vector(features, feature_order=["a", "b"])
    assert vec == [1.0, 2.0]
