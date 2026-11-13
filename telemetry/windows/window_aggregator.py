"""Behavioural window aggregation.

Aggregates normalized request events into fixed-duration time windows grouped
by pseudonymous client identity. Each window produces a set of behavioural
features that describe the client's activity during that interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from telemetry.parsers.log_parser import NormalizedEvent


@dataclass
class WindowRecord:
    """Aggregated behavioural features for one client over one time window."""

    client_ip_hash: str
    window_start_utc: str
    window_end_utc: str
    window_duration_seconds: float

    # Request-level aggregates
    requests_per_window: int = 0
    unique_paths: int = 0
    unique_methods: int = 0
    status_4xx_count: int = 0
    status_5xx_count: int = 0
    auth_failure_count: int = 0

    # Size aggregates
    total_request_bytes: int = 0
    max_request_size: int = 0

    # Timing aggregates
    total_duration_ms: float = 0.0
    total_response_bytes: int = 0
    request_timestamps: list[str] = field(default_factory=list)

    def request_rate_per_second(self) -> float:
        if self.window_duration_seconds <= 0:
            return 0.0
        return self.requests_per_window / self.window_duration_seconds

    def status_4xx_ratio(self) -> float:
        if self.requests_per_window == 0:
            return 0.0
        return self.status_4xx_count / self.requests_per_window

    def status_5xx_ratio(self) -> float:
        if self.requests_per_window == 0:
            return 0.0
        return self.status_5xx_count / self.requests_per_window

    def mean_request_size(self) -> float:
        if self.requests_per_window == 0:
            return 0.0
        return self.total_request_bytes / self.requests_per_window

    def mean_response_time_ms(self) -> float:
        if self.requests_per_window == 0:
            return 0.0
        return self.total_duration_ms / self.requests_per_window

    def mean_inter_request_interval_ms(self) -> float:
        if len(self.request_timestamps) < 2:
            return 0.0
        times = sorted(
            datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            for ts in self.request_timestamps
        )
        deltas = [
            (times[i + 1] - times[i]).total_seconds() * 1000.0
            for i in range(len(times) - 1)
        ]
        return sum(deltas) / len(deltas) if deltas else 0.0


@dataclass
class WindowConfig:
    """Configuration for window aggregation."""

    duration_seconds: float = 60.0


def _floor_to_window(
    timestamp_str: str, duration_seconds: float
) -> tuple[datetime, datetime]:
    """Floor a timestamp to the start of its window and return (start, end)."""
    ts = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    offset = (ts - epoch).total_seconds()
    window_index = int(offset // duration_seconds)
    start = epoch + timedelta(seconds=window_index * duration_seconds)
    end = start + timedelta(seconds=duration_seconds)
    return start, end


def aggregate_windows(
    events: Iterable[NormalizedEvent], config: WindowConfig | None = None
) -> list[WindowRecord]:
    """Group events into fixed-duration windows by client identity.

    Returns one WindowRecord per (client, window) pair that contains at least
    one event.
    """
    if config is None:
        config = WindowConfig()

    # Group events by (client_ip_hash, window_start)
    groups: dict[tuple[str, datetime], list[NormalizedEvent]] = {}
    for event in events:
        window_start, _ = _floor_to_window(event.timestamp_utc, config.duration_seconds)
        key = (event.client_ip_hash, window_start)
        groups.setdefault(key, []).append(event)

    records: list[WindowRecord] = []
    for (client_ip_hash, window_start), window_events in groups.items():
        window_end = window_start + timedelta(seconds=config.duration_seconds)
        rec = WindowRecord(
            client_ip_hash=client_ip_hash,
            window_start_utc=window_start.isoformat(),
            window_end_utc=window_end.isoformat(),
            window_duration_seconds=config.duration_seconds,
        )

        paths: set[str] = set()
        methods: set[str] = set()

        for ev in window_events:
            rec.requests_per_window += 1
            paths.add(ev.path)
            methods.add(ev.http_method)

            if 400 <= ev.status_code < 500:
                rec.status_4xx_count += 1
            if 500 <= ev.status_code < 600:
                rec.status_5xx_count += 1
            if ev.is_auth_failure:
                rec.auth_failure_count += 1

            rec.total_request_bytes += ev.content_length
            rec.max_request_size = max(rec.max_request_size, ev.content_length)
            rec.total_duration_ms += ev.request_duration_ms
            rec.total_response_bytes += ev.response_bytes
            rec.request_timestamps.append(ev.timestamp_utc)

        rec.unique_paths = len(paths)
        rec.unique_methods = len(methods)
        records.append(rec)

    return records
