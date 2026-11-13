"""Log event parsers.

Parses raw telemetry events (e.g. structured JSON lines from application
middleware or access logs) into normalized event records. Independent of the
monitored website.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SALT = "vibematch-telemetry-salt-v1"  # fixed salt for client IP pseudonymization


@dataclass
class NormalizedEvent:
    """A single normalized request event."""

    timestamp_utc: str
    client_ip_hash: str
    http_method: str
    path: str
    path_length: int
    query_param_count: int
    status_code: int
    response_bytes: int
    request_duration_ms: float
    content_length: int
    header_count: int
    is_auth_failure: bool
    raw: dict[str, Any] = field(default_factory=dict)


def pseudonymize_client_ip(client_ip: str, salt: str = SALT) -> str:
    """Pseudonymize a client IP address for use as an aggregation key.

    Returns a SHA-256 hex digest. The IP itself is never stored as a numerical
    feature.
    """
    return hashlib.sha256(f"{salt}:{client_ip}".encode("utf-8")).hexdigest()


def count_query_params(path: str) -> int:
    """Count query parameters in a URL path."""
    if "?" not in path:
        return 0
    query = path.split("?", 1)[1]
    if not query:
        return 0
    return max(1, len([p for p in query.split("&") if p]))


def parse_jsonl_event(line: str) -> NormalizedEvent | None:
    """Parse a single JSONL telemetry event into a NormalizedEvent.

    Expected input fields (all optional with sensible defaults):

        timestamp_utc   ISO-8601 timestamp (default: now)
        client_ip       client IP string (pseudonymized)
        http_method     GET/POST/PUT/DELETE/etc.
        path            URL path (may include query string)
        status_code     HTTP response status
        response_bytes  response body size in bytes
        request_duration_ms server-observed processing time
        content_length  declared request body length
        header_count    number of request headers
        is_auth_failure true if this was a failed authentication event

    Returns None if the line is not valid JSON or is missing critical fields.
    """
    try:
        raw = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(raw, dict):
        return None

    timestamp = raw.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
    client_ip = raw.get("client_ip", "0.0.0.0")
    http_method = str(raw.get("http_method", "OTHER")).upper()
    path = str(raw.get("path", "/"))
    status_code = int(raw.get("status_code", 0))
    response_bytes = int(raw.get("response_bytes", 0))
    duration_ms = float(raw.get("request_duration_ms", 0.0))
    content_length = int(raw.get("content_length", 0))
    header_count = int(raw.get("header_count", 0))
    is_auth_failure = bool(raw.get("is_auth_failure", False))

    return NormalizedEvent(
        timestamp_utc=timestamp,
        client_ip_hash=pseudonymize_client_ip(client_ip),
        http_method=http_method,
        path=path,
        path_length=len(path),
        query_param_count=count_query_params(path),
        status_code=status_code,
        response_bytes=response_bytes,
        request_duration_ms=duration_ms,
        content_length=content_length,
        header_count=header_count,
        is_auth_failure=is_auth_failure,
        raw=raw,
    )


def parse_jsonl_file(path: str) -> list[NormalizedEvent]:
    """Parse all valid events from a JSONL file, skipping invalid lines."""
    events: list[NormalizedEvent] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = parse_jsonl_event(line)
            if event is not None:
                events.append(event)
    return events
