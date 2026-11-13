"""Telemetry package public API."""

from .feature_extractor import (
    FULL_FEATURE_ORDER,
    REQUEST_FEATURE_ORDER,
    WINDOW_FEATURE_ORDER,
    event_to_vector,
    features_to_vector,
    n_full_features,
    n_request_features,
    n_window_features,
    window_to_vector,
)
from .parsers.log_parser import NormalizedEvent, parse_jsonl_event, parse_jsonl_file, pseudonymize_client_ip
from .windows.window_aggregator import WindowConfig, WindowRecord, aggregate_windows

__all__ = [
    "FULL_FEATURE_ORDER",
    "REQUEST_FEATURE_ORDER",
    "WINDOW_FEATURE_ORDER",
    "NormalizedEvent",
    "WindowConfig",
    "WindowRecord",
    "aggregate_windows",
    "event_to_vector",
    "features_to_vector",
    "n_full_features",
    "n_request_features",
    "n_window_features",
    "parse_jsonl_event",
    "parse_jsonl_file",
    "pseudonymize_client_ip",
    "window_to_vector",
]
