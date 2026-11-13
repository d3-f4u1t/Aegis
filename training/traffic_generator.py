"""Synthetic HTTP traffic generator with realistic behavioural scenarios.

Generates controlled, reproducible synthetic website traffic for experimental
evaluation of traffic anomaly classifiers. Replaces simplistic binary rate
differences with realistic, diverse behavioural scenarios:

Normal Scenarios:
1. casual_browsing: Slow reading of common pages, long intervals, low rates.
2. active_browsing: Rapid navigation, search queries, product views, mixed GET/POST.
3. bursty_browsing: Page-load bursts (HTML, assets, XHR) followed by reading pauses.
4. repeated_visits: Frequent returning to same pages (dashboard, cart, orders).
5. occasional_404: Benign user encountering dead links or mistyping (10-25% 404s).
6. api_polling: Legitimate client polling (/api/v1/poll, /health) at high rate (0.4-0.8 RPS)
   with 0 errors and 1 unique path (challenges rate-based detectors).

Attack Scenarios:
1. request_flood: Volumetric HTTP flooding, high RPS across paths, server stress.
2. slow_path_enumeration: Stealthy directory scanning at low rate (same rate as casual
   browsing) with high 404/403 ratio (evades rate-based detectors).
3. brute_force: Repeated POST authentication attempts to /login with 85%+ 401s.
4. distributed_low_rate: Multi-client low-rate scanning (2-5 req/window) with 404s.
5. bursty_spikes: Intense short bursts (15-25 requests in 2 seconds) then silence.
6. high_volume_scraping: Automated data harvesting, high RPS, very high unique paths,
   but 95%+ 200 OK (evades error-ratio detectors).
7. unusual_path_diversity: Rapid scanning and parameter fuzzing across diverse paths.

Ground Truth and Attribution:
- Every event records explicit ``label`` (0=normal, 1=attack) and ``scenario``.
- Client IP addresses are drawn from a shared, realistic IP pool without subnet bias.
- Generation is fully deterministic given a random seed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence


NORMAL_SCENARIOS = [
    "normal_casual_browsing",
    "normal_active_browsing",
    "normal_bursty_browsing",
    "normal_repeated_visits",
    "normal_occasional_404",
    "normal_api_polling",
]

ATTACK_SCENARIOS = [
    "attack_request_flood",
    "attack_slow_path_enumeration",
    "attack_brute_force",
    "attack_distributed_low_rate",
    "attack_bursty_spikes",
    "attack_high_volume_scraping",
    "attack_unusual_path_diversity",
]

ALL_SCENARIOS = NORMAL_SCENARIOS + ATTACK_SCENARIOS

# Realistic path catalogs
COMMON_CONTENT_PATHS = [
    "/", "/about", "/contact", "/products", "/products/101", "/products/102",
    "/products/103", "/blog", "/blog/announcement", "/terms", "/privacy",
    "/pricing", "/faq", "/docs", "/docs/quickstart",
]

ACTIVE_USER_PATHS = [
    "/search", "/products/201", "/products/202", "/cart", "/cart/checkout",
    "/account", "/account/orders", "/account/settings", "/api/v1/user/profile",
]

POLLING_PATHS = ["/api/v1/poll", "/health", "/api/v1/notifications/stream"]

SENSITIVE_SCAN_PATHS = [
    "/admin", "/admin/login.php", "/wp-admin", "/wp-login.php", "/.env",
    "/.git/HEAD", "/config.json", "/backup.sql", "/api/v1/debug",
    "/server-status", "/phpmyadmin", "/.aws/credentials", "/etc/passwd",
    "/v2/_catalog", "/actuator/health", "/swagger-ui.html",
]

AUTH_PATHS = ["/login", "/api/v1/auth/login", "/api/v1/auth/token"]


@dataclass
class ScenarioClientConfig:
    """Specification of one client session."""

    client_id: str
    client_ip: str
    scenario: str
    label: int  # 0 = normal, 1 = attack
    duration_seconds: float = 3600.0  # 1 hour by default


def generate_realistic_ip_pool(n_ips: int, seed: int = 100) -> list[str]:
    """Generate realistic mixed IPv4 addresses without class/subnet bias.

    Uses a mix of public and private ranges distributed evenly.
    """
    rng = random.Random(seed)
    ips = set()
    subnets = [
        "192.168.1.", "192.168.2.", "10.0.1.", "10.0.2.", "172.16.1.",
        "198.51.100.", "203.0.113.", "198.18.0.", "100.64.0.",
    ]
    while len(ips) < n_ips:
        prefix = rng.choice(subnets)
        host = rng.randint(2, 254)
        ips.add(f"{prefix}{host}")
    return sorted(ips)


def _make_event(
    timestamp: datetime,
    client_ip: str,
    method: str,
    path: str,
    status_code: int,
    response_bytes: int,
    duration_ms: float,
    content_length: int,
    header_count: int,
    is_auth_failure: bool,
    scenario: str,
    label: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": timestamp.isoformat(),
        "client_ip": client_ip,
        "http_method": method,
        "path": path,
        "status_code": status_code,
        "response_bytes": response_bytes,
        "request_duration_ms": round(duration_ms, 2),
        "content_length": content_length,
        "header_count": header_count,
        "is_auth_failure": is_auth_failure,
        "scenario": scenario,
        "label": label,
    }


def generate_scenario_events(
    scenario: str,
    client_ip: str,
    start_time: datetime,
    duration_seconds: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate a sequence of HTTP request events for a specified scenario."""
    events: list[dict[str, Any]] = []
    end_time = start_time + timedelta(seconds=duration_seconds)
    current = start_time

    # ──────────────────────────────────────────────────────────────────────────
    # NORMAL SCENARIOS (label = 0)
    # ──────────────────────────────────────────────────────────────────────────

    if scenario == "normal_casual_browsing":
        # Slow, thoughtful browsing: intervals 12-28s, common content, 98% 200 OK
        while current < end_time:
            path = rng.choice(COMMON_CONTENT_PATHS)
            status = 200 if rng.random() > 0.02 else 304
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=path,
                    status_code=status,
                    response_bytes=rng.randint(2000, 25000),
                    duration_ms=rng.uniform(15.0, 120.0),
                    content_length=0,
                    header_count=rng.randint(8, 14),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=0,
                )
            )
            interval = rng.uniform(12.0, 28.0)
            current += timedelta(seconds=interval)

    elif scenario == "normal_active_browsing":
        # Active browsing: intervals 2-8s, mixed GET/POST, cart, search
        while current < end_time:
            is_post = rng.random() < 0.15
            method = "POST" if is_post else "GET"
            if is_post:
                path = rng.choice(["/cart", "/cart/checkout", "/account/settings"])
                content_length = rng.randint(50, 450)
                status = 200 if rng.random() > 0.05 else 302
            else:
                path = rng.choice(ACTIVE_USER_PATHS)
                if path == "/search":
                    path += f"?q=item_{rng.randint(1, 50)}&page=1"
                content_length = 0
                status = 200
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method=method,
                    path=path,
                    status_code=status,
                    response_bytes=rng.randint(1000, 30000),
                    duration_ms=rng.uniform(20.0, 180.0),
                    content_length=content_length,
                    header_count=rng.randint(10, 18),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=0,
                )
            )
            interval = rng.uniform(2.5, 7.5)
            current += timedelta(seconds=interval)

    elif scenario == "normal_bursty_browsing":
        # Bursty page navigation: 3-7 requests in 0.3-1.5s, then 20-45s reading pause
        while current < end_time:
            burst_size = rng.randint(3, 7)
            base_page = rng.choice(COMMON_CONTENT_PATHS)
            for b in range(burst_size):
                if b == 0:
                    path = base_page
                    r_bytes = rng.randint(5000, 20000)
                else:
                    path = f"{base_page}/asset_{b}.js"
                    r_bytes = rng.randint(500, 4000)
                events.append(
                    _make_event(
                        timestamp=current,
                        client_ip=client_ip,
                        method="GET",
                        path=path,
                        status_code=200 if rng.random() > 0.05 else 304,
                        response_bytes=r_bytes,
                        duration_ms=rng.uniform(10.0, 90.0),
                        content_length=0,
                        header_count=rng.randint(8, 15),
                        is_auth_failure=False,
                        scenario=scenario,
                        label=0,
                    )
                )
                current += timedelta(seconds=rng.uniform(0.08, 0.35))
            # Reading pause
            current += timedelta(seconds=rng.uniform(20.0, 45.0))

    elif scenario == "normal_repeated_visits":
        # Repeated checks on dashboard / cart / status: 1-2 unique paths, interval 8-22s
        focus_path = rng.choice(["/account/orders", "/cart", "/dashboard", "/status"])
        while current < end_time:
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=focus_path,
                    status_code=200,
                    response_bytes=rng.randint(1200, 8000),
                    duration_ms=rng.uniform(15.0, 80.0),
                    content_length=0,
                    header_count=rng.randint(8, 12),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=0,
                )
            )
            interval = rng.uniform(8.0, 22.0)
            current += timedelta(seconds=interval)

    elif scenario == "normal_occasional_404":
        # Normal user following broken link or mistyping: 15-25% 404s, remainder 200 OK
        while current < end_time:
            is_404 = rng.random() < 0.20
            if is_404:
                path = rng.choice(["/old-page", "/docs/v0", "/missing-item", "/product/notfound", "/images/missing.png"])
                status = 404
                r_bytes = rng.randint(200, 800)
            else:
                path = rng.choice(COMMON_CONTENT_PATHS)
                status = 200
                r_bytes = rng.randint(1500, 15000)
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=path,
                    status_code=status,
                    response_bytes=r_bytes,
                    duration_ms=rng.uniform(12.0, 110.0),
                    content_length=0,
                    header_count=rng.randint(8, 14),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=0,
                )
            )
            interval = rng.uniform(7.0, 20.0)
            current += timedelta(seconds=interval)

    elif scenario == "normal_api_polling":
        # Legitimate single-endpoint polling: 0.4 - 0.7 RPS, 100% 200 OK, 1 path, 0 errors
        polling_endpoint = rng.choice(POLLING_PATHS)
        while current < end_time:
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=polling_endpoint,
                    status_code=200,
                    response_bytes=rng.randint(120, 350),
                    duration_ms=rng.uniform(8.0, 45.0),
                    content_length=0,
                    header_count=rng.randint(6, 10),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=0,
                )
            )
            interval = rng.uniform(1.6, 2.8)
            current += timedelta(seconds=interval)

    # ──────────────────────────────────────────────────────────────────────────
    # ATTACK SCENARIOS (label = 1)
    # ──────────────────────────────────────────────────────────────────────────

    elif scenario == "attack_request_flood":
        # High-rate HTTP flood: 1.0 - 2.5 RPS, stress causes elevated latency & occasional 500
        while current < end_time:
            path = rng.choice(["/", "/search?q=flood", "/api/v1/data", "/products"])
            # Server stress triggers some 500/503 errors under flood
            status = 200 if rng.random() > 0.08 else rng.choice([500, 503])
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method=rng.choice(["GET", "POST", "GET"]),
                    path=path,
                    status_code=status,
                    response_bytes=rng.randint(200, 3000),
                    duration_ms=rng.uniform(120.0, 450.0),  # elevated server load
                    content_length=rng.randint(0, 500),
                    header_count=rng.randint(5, 12),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=1,
                )
            )
            interval = rng.uniform(0.35, 0.95)
            current += timedelta(seconds=interval)

    elif scenario == "attack_slow_path_enumeration":
        # Stealthy directory scanning at LOW rate (7-18s intervals): 75-90% 404/403
        while current < end_time:
            path = rng.choice(SENSITIVE_SCAN_PATHS)
            status = rng.choice([404, 404, 403, 404, 200])
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=path,
                    status_code=status,
                    response_bytes=rng.randint(150, 900),
                    duration_ms=rng.uniform(30.0, 160.0),
                    content_length=0,
                    header_count=rng.randint(6, 11),
                    is_auth_failure=status == 403,
                    scenario=scenario,
                    label=1,
                )
            )
            interval = rng.uniform(7.5, 18.0)  # low rate, mimics casual browsing
            current += timedelta(seconds=interval)

    elif scenario == "attack_brute_force":
        # Repeated credential stuffing / brute-force: POST /login, 85%+ 401, is_auth_failure=True
        login_path = rng.choice(AUTH_PATHS)
        while current < end_time:
            is_success = rng.random() < 0.05
            status = 200 if is_success else 401
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="POST",
                    path=login_path,
                    status_code=status,
                    response_bytes=rng.randint(100, 500),
                    duration_ms=rng.uniform(40.0, 150.0),
                    content_length=rng.randint(150, 350),
                    header_count=rng.randint(10, 16),
                    is_auth_failure=not is_success,
                    scenario=scenario,
                    label=1,
                )
            )
            interval = rng.uniform(1.8, 4.2)
            current += timedelta(seconds=interval)

    elif scenario == "attack_distributed_low_rate":
        # Bot in distributed network: very low rate (15-30s intervals), scanning diverse sensitive paths
        while current < end_time:
            path = rng.choice(SENSITIVE_SCAN_PATHS)
            status = rng.choice([404, 403, 404])
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=path,
                    status_code=status,
                    response_bytes=rng.randint(150, 600),
                    duration_ms=rng.uniform(25.0, 130.0),
                    content_length=0,
                    header_count=rng.randint(5, 10),
                    is_auth_failure=status == 403,
                    scenario=scenario,
                    label=1,
                )
            )
            interval = rng.uniform(16.0, 32.0)  # very low rate
            current += timedelta(seconds=interval)

    elif scenario == "attack_bursty_spikes":
        # Sudden bursts of requests (12-22 in 1.5s), followed by long silence (45-65s)
        while current < end_time:
            burst_len = rng.randint(12, 22)
            for _ in range(burst_len):
                events.append(
                    _make_event(
                        timestamp=current,
                        client_ip=client_ip,
                        method="GET",
                        path=rng.choice(SENSITIVE_SCAN_PATHS + COMMON_CONTENT_PATHS),
                        status_code=rng.choice([200, 404, 403, 404]),
                        response_bytes=rng.randint(200, 2000),
                        duration_ms=rng.uniform(20.0, 100.0),
                        content_length=0,
                        header_count=rng.randint(6, 12),
                        is_auth_failure=False,
                        scenario=scenario,
                        label=1,
                    )
                )
                current += timedelta(seconds=rng.uniform(0.05, 0.18))
            current += timedelta(seconds=rng.uniform(45.0, 65.0))

    elif scenario == "attack_high_volume_scraping":
        # High volume web scraping: rapid traversal across dozens of paths, 98% 200 OK (no 4xx errors!)
        catalog_items = [f"/products/catalog_item_{i}" for i in range(1, 150)]
        while current < end_time:
            path = rng.choice(catalog_items)
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method="GET",
                    path=path,
                    status_code=200,  # 100% successful scrapes!
                    response_bytes=rng.randint(3000, 18000),
                    duration_ms=rng.uniform(15.0, 70.0),
                    content_length=0,
                    header_count=rng.randint(7, 12),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=1,
                )
            )
            interval = rng.uniform(0.7, 1.6)  # rapid scraping
            current += timedelta(seconds=interval)

    elif scenario == "attack_unusual_path_diversity":
        # Path and parameter fuzzing: rapid unique endpoints, high path count, 50% 404s
        fuzz_counter = 0
        while current < end_time:
            fuzz_counter += 1
            path = f"/api/endpoint_{fuzz_counter}?param_{fuzz_counter}=fuzz"
            status = rng.choice([200, 404, 400, 404])
            events.append(
                _make_event(
                    timestamp=current,
                    client_ip=client_ip,
                    method=rng.choice(["GET", "POST"]),
                    path=path,
                    status_code=status,
                    response_bytes=rng.randint(150, 1200),
                    duration_ms=rng.uniform(20.0, 120.0),
                    content_length=rng.randint(0, 300),
                    header_count=rng.randint(5, 10),
                    is_auth_failure=False,
                    scenario=scenario,
                    label=1,
                )
            )
            interval = rng.uniform(1.2, 2.8)
            current += timedelta(seconds=interval)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return events


@dataclass
class GeneratorPlan:
    """Configurable plan for synthetic dataset generation."""

    name: str = "phase5_benchmark"
    seed: int = 42
    start_time_iso: str = "2026-11-14T00:00:00+00:00"
    duration_seconds: float = 3600.0  # 1 hour per client
    clients_per_scenario: dict[str, int] = field(default_factory=dict)

    @classmethod
    def default_phase5_plan(cls, seed: int = 42) -> GeneratorPlan:
        """Balanced benchmark with realistic client counts across all 13 scenarios."""
        scenario_counts = {
            # Normal scenarios (total 24 clients)
            "normal_casual_browsing": 5,
            "normal_active_browsing": 5,
            "normal_bursty_browsing": 4,
            "normal_repeated_visits": 3,
            "normal_occasional_404": 3,
            "normal_api_polling": 4,
            # Attack scenarios (total 21 clients)
            "attack_request_flood": 3,
            "attack_slow_path_enumeration": 3,
            "attack_brute_force": 3,
            "attack_distributed_low_rate": 3,
            "attack_bursty_spikes": 3,
            "attack_high_volume_scraping": 3,
            "attack_unusual_path_diversity": 3,
        }
        return cls(
            name="phase5_benchmark",
            seed=seed,
            duration_seconds=3600.0,
            clients_per_scenario=scenario_counts,
        )


def generate_dataset_events(plan: GeneratorPlan) -> list[dict[str, Any]]:
    """Execute generator plan and return sorted JSONL-ready events."""
    rng = random.Random(plan.seed)
    start_dt = datetime.fromisoformat(plan.start_time_iso)

    total_clients = sum(plan.clients_per_scenario.values())
    ip_pool = generate_realistic_ip_pool(total_clients, seed=plan.seed)
    rng.shuffle(ip_pool)

    all_events: list[dict[str, Any]] = []
    ip_idx = 0

    for scenario, count in sorted(plan.clients_per_scenario.items()):
        is_attack = scenario.startswith("attack_")
        label = 1 if is_attack else 0
        for _ in range(count):
            client_ip = ip_pool[ip_idx]
            ip_idx += 1
            client_events = generate_scenario_events(
                scenario=scenario,
                client_ip=client_ip,
                start_time=start_dt,
                duration_seconds=plan.duration_seconds,
                rng=rng,
            )
            all_events.extend(client_events)

    all_events.sort(key=lambda e: e["timestamp_utc"])
    return all_events


def save_events_to_jsonl(events: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    """Save events to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic Phase 5 synthetic telemetry")
    parser.add_argument("--output", default="data/telemetry/synthetic_phase5.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--duration", type=float, default=3600.0, help="Duration in seconds")
    args = parser.parse_args()

    plan = GeneratorPlan.default_phase5_plan(seed=args.seed)
    plan.duration_seconds = args.duration
    events = generate_dataset_events(plan)
    out_file = save_events_to_jsonl(events, args.output)

    pos = sum(1 for e in events if e.get("label") == 1)
    neg = len(events) - pos
    print(f"Generated {len(events)} events ({neg} normal, {pos} attack) -> {out_file}")


if __name__ == "__main__":
    main()
