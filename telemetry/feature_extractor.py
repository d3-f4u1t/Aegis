"""Feature extraction.

Converts parsed request events and behavioural window records into numerical
feature vectors matching the versioned feature schema
(configs/feature_schema.yaml v1).

The output vector order is fixed and must match the schema exactly:

Request-level (per event):
    http_method_GET, http_method_POST, http_method_PUT,
    http_method_DELETE, http_method_OTHER,
    path_length, query_param_count, status_code,
    response_bytes, request_duration_ms,
    content_length, header_count

Behavioural window (per window):
    requests_per_window, requests_per_second,
    unique_paths, unique_methods,
    status_4xx_count, status_4xx_ratio,
    status_5xx_count, status_5xx_ratio,
    auth_failure_count,
    mean_request_size, max_request_size,
    mean_response_time_ms, mean_inter_request_interval_ms
"""

from __future__ import annotations

from typing import Sequence

from telemetry.parsers.log_parser import NormalizedEvent
from telemetry.windows.window_aggregator import WindowRecord

# Fixed feature order — must match configs/feature_schema.yaml v1
REQUEST_FEATURE_ORDER: list[str] = [
    "http_method_GET",
    "http_method_POST",
    "http_method_PUT",
    "http_method_DELETE",
    "http_method_OTHER",
    "path_length",
    "query_param_count",
    "status_code",
    "response_bytes",
    "request_duration_ms",
    "content_length",
    "header_count",
]

WINDOW_FEATURE_ORDER: list[str] = [
    "requests_per_window",
    "requests_per_second",
    "unique_paths",
    "unique_methods",
    "status_4xx_count",
    "status_4xx_ratio",
    "status_5xx_count",
    "status_5xx_ratio",
    "auth_failure_count",
    "mean_request_size",
    "max_request_size",
    "mean_response_time_ms",
    "mean_inter_request_interval_ms",
]

FULL_FEATURE_ORDER: list[str] = REQUEST_FEATURE_ORDER + WINDOW_FEATURE_ORDER


def method_one_hot(http_method: str) -> dict[str, int]:
    """Encode an HTTP method as a one-hot vector."""
    method = http_method.upper()
    return {
        "http_method_GET": 1 if method == "GET" else 0,
        "http_method_POST": 1 if method == "POST" else 0,
        "http_method_PUT": 1 if method == "PUT" else 0,
        "http_method_DELETE": 1 if method == "DELETE" else 0,
        "http_method_OTHER": 1 if method not in ("GET", "POST", "PUT", "DELETE") else 0,
    }


def event_to_request_features(event: NormalizedEvent) -> dict[str, float | int]:
    """Convert a single normalized event to request-level features."""
    method_oh = method_one_hot(event.http_method)
    return {
        "http_method_GET": method_oh["http_method_GET"],
        "http_method_POST": method_oh["http_method_POST"],
        "http_method_PUT": method_oh["http_method_PUT"],
        "http_method_DELETE": method_oh["http_method_DELETE"],
        "http_method_OTHER": method_oh["http_method_OTHER"],
        "path_length": event.path_length,
        "query_param_count": event.query_param_count,
        "status_code": event.status_code,
        "response_bytes": event.response_bytes,
        "request_duration_ms": event.request_duration_ms,
        "content_length": event.content_length,
        "header_count": event.header_count,
    }


def window_to_features(window: WindowRecord) -> dict[str, float | int]:
    """Convert a window record to behavioural window features."""
    return {
        "requests_per_window": window.requests_per_window,
        "requests_per_second": window.request_rate_per_second(),
        "unique_paths": window.unique_paths,
        "unique_methods": window.unique_methods,
        "status_4xx_count": window.status_4xx_count,
        "status_4xx_ratio": window.status_4xx_ratio(),
        "status_5xx_count": window.status_5xx_count,
        "status_5xx_ratio": window.status_5xx_ratio(),
        "auth_failure_count": window.auth_failure_count,
        "mean_request_size": window.mean_request_size(),
        "max_request_size": window.max_request_size,
        "mean_response_time_ms": window.mean_response_time_ms(),
        "mean_inter_request_interval_ms": window.mean_inter_request_interval_ms(),
    }


def features_to_vector(
    features: dict[str, float | int],
    feature_order: Sequence[str] = FULL_FEATURE_ORDER,
) -> list[float]:
    """Convert a feature dictionary to a fixed-order numerical vector.

    Missing features are filled with 0.0.
    """
    return [float(features.get(name, 0.0)) for name in feature_order]


def event_to_vector(event: NormalizedEvent) -> list[float]:
    """Convert a single normalized event to a request-level feature vector."""
    features = event_to_request_features(event)
    return features_to_vector(features, REQUEST_FEATURE_ORDER)


def window_to_vector(window: WindowRecord) -> list[float]:
    """Convert a window record to a behavioural window feature vector."""
    features = window_to_features(window)
    return features_to_vector(features, WINDOW_FEATURE_ORDER)


def n_request_features() -> int:
    return len(REQUEST_FEATURE_ORDER)


def n_window_features() -> int:
    return len(WINDOW_FEATURE_ORDER)


def n_full_features() -> int:
    return len(FULL_FEATURE_ORDER)
