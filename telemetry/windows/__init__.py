"""Window aggregation package."""

from .window_aggregator import WindowConfig, WindowRecord, aggregate_windows

__all__ = [
    "WindowConfig",
    "WindowRecord",
    "aggregate_windows",
]
