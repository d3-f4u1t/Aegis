"""Synthetic HTTP telemetry event generator.

Generates synthetic JSONL events that mimic a web server access log for
testing the telemetry collector, parser, window aggregator, and feature
extractor pipeline. This is a test-only utility — no web server is required.

Usage:

    python -m examples.generate_synthetic_events --output data/telemetry/synthetic.jsonl --events 1000
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone


METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
PATHS = [
    "/", "/about", "/search", "/login", "/api/data",
    "/profile", "/settings", "/contact", "/products", "/cart",
]
STATUSES = [200, 200, 200, 200, 301, 302, 400, 401, 403, 404, 500]


def generate_event(
    rng: random.Random,
    timestamp: datetime,
    client_ip: str,
    attack_pattern: bool = False,
) -> dict:
    """Generate a single synthetic HTTP event.

    If attack_pattern is True, the event is more likely to look anomalous
    (high frequency, many 4xx/5xx, scanning-like paths).
    """
    if attack_pattern:
        method = rng.choice(["GET", "POST", "GET", "GET"])
        path = rng.choice(
            ["/admin", "/wp-admin", "/login", "/api/data", "/search?q=test",
             "/../../etc/passwd", "/api/v1/users", "/login"]
        )
        status = rng.choice([200, 401, 403, 404, 500, 200])
        response_bytes = rng.randint(100, 5000)
        duration_ms = rng.uniform(50, 500)
        content_length = rng.randint(0, 2000)
        header_count = rng.randint(5, 30)
        is_auth_failure = status in (401, 403)
    else:
        method = rng.choice(METHODS)
        path = rng.choice(PATHS)
        if rng.random() < 0.3:
            path += "?" + "&".join(f"p{i}=v{i}" for i in range(rng.randint(1, 3)))
        status = rng.choice(STATUSES)
        response_bytes = rng.randint(500, 50000)
        duration_ms = rng.uniform(10, 200)
        content_length = rng.randint(0, 1000)
        header_count = rng.randint(5, 25)
        is_auth_failure = status in (401, 403)

    return {
        "timestamp_utc": timestamp.isoformat(),
        "client_ip": client_ip,
        "http_method": method,
        "path": path,
        "status_code": status,
        "response_bytes": response_bytes,
        "request_duration_ms": round(duration_ms, 2),
        "content_length": content_length,
        "header_count": header_count,
        "is_auth_failure": is_auth_failure,
    }


def generate_session(
    rng: random.Random,
    start: datetime,
    client_ip: str,
    n_events: int,
    attack_pattern: bool = False,
    interval_seconds_mean: float = 30.0,
) -> list[dict]:
    """Generate a session of events for one client over time."""
    events = []
    current = start
    for _ in range(n_events):
        events.append(generate_event(rng, current, client_ip, attack_pattern))
        current += timedelta(seconds=rng.expovariate(1.0 / interval_seconds_mean))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic HTTP telemetry events")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--events", type=int, default=1000, help="Total events to generate")
    parser.add_argument("--clients", type=int, default=10, help="Number of distinct client IPs")
    parser.add_argument("--attack-clients", type=int, default=2,
                        help="Number of clients generating attack-like traffic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start", default=None,
                        help="Start timestamp ISO-8601 (default: now)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = (
        datetime.fromisoformat(args.start)
        if args.start
        else datetime.now(timezone.utc)
    )

    # Generate normal clients
    normal_clients = [f"10.0.0.{i}" for i in range(1, args.clients - args.attack_clients + 1)]
    attack_clients = [f"10.0.10.{i}" for i in range(1, args.attack_clients + 1)]

    all_events = []

    for ip in normal_clients:
        n = rng.randint(20, 200)
        all_events.extend(
            generate_session(rng, start, ip, n, attack_pattern=False)
        )

    for ip in attack_clients:
        n = rng.randint(100, 500)
        all_events.extend(
            generate_session(rng, start, ip, n, attack_pattern=True,
                           interval_seconds_mean=2.0)
        )

    # Sort by timestamp
    all_events.sort(key=lambda e: e["timestamp_utc"])

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        for ev in all_events:
            fh.write(json.dumps(ev) + "\n")

    print(f"Generated {len(all_events)} events from {args.clients} clients "
          f"({args.attack_clients} attack-like) to {args.output}")


if __name__ == "__main__":
    main()
