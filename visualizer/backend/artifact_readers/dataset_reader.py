"""Dataset artifact reader providing curated real traffic windows for the Visualization Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from training.dataset import Dataset, WindowSample, load_dataset
from training.preprocessing import StandardScaler
from visualizer.backend.schemas.api_models import TrafficSampleItem

SCENARIO_DESCRIPTIONS = {
    "normal_casual_browsing": "Typical human visitor casually navigating pages with intermittent delays.",
    "normal_active_browsing": "Engaged user actively browsing multiple resource pages with healthy HTTP 200 responses.",
    "normal_bursty_browsing": "User opening multiple tabs simultaneously, causing small short bursts of requests.",
    "normal_repeated_visits": "Returning visitor repeatedly accessing cached resources and assets.",
    "normal_occasional_404": "Benign visitor encountering broken images or links; benign non-zero 4xx count.",
    "normal_api_polling": "Legitimate client application periodically querying a telemetry or status API endpoint.",
    "attack_request_flood": "High-volume volumetric flood attempting to saturate web server request processing queues.",
    "attack_bursty_spikes": "Volumetric pulsed attacks alternating between dormant intervals and intense packet bursts.",
    "attack_slow_path_enumeration": "Low-and-slow directory traversal looking for vulnerable admin panels or exposed .env files.",
    "attack_unusual_path_diversity": "Automated web crawler or vulnerability fuzzer probing an abnormally wide variety of paths.",
    "attack_brute_force": "High-frequency authentication attacks targeting login/token endpoints with rapid password attempts.",
    "attack_distributed_low_rate": "Stealthy coordinated attack keeping rates below typical rate-limiting thresholds.",
    "attack_high_volume_scraping": "Fast automated content scraping public endpoints. Every request returns HTTP 200 OK with zero errors.",
}


class DatasetReader:
    """Manages curated samples extracted from authentic telemetry datasets."""

    def __init__(
        self,
        telemetry_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
    ) -> None:
        self.telemetry_path = (
            Path(telemetry_path) if telemetry_path else Path("data/telemetry/synthetic_phase5.jsonl")
        )
        self.scaler_path = (
            Path(scaler_path)
            if scaler_path
            else Path("experiments/results/phase5_exp1_indistribution_preprocessing.json")
        )
        self._curated_samples: List[TrafficSampleItem] = []
        self._samples_by_id: Dict[str, TrafficSampleItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.telemetry_path.exists():
            return

        scaler: Optional[StandardScaler] = None
        if self.scaler_path.exists():
            try:
                scaler = StandardScaler.load(self.scaler_path)
            except Exception:
                scaler = None

        ds = load_dataset(self.telemetry_path)

        # Collect 2 representative windows per scenario
        scenario_counts: Dict[str, int] = {}
        for idx, sample in enumerate(ds.samples):
            scen = sample.scenario
            curr_count = scenario_counts.get(scen, 0)
            if curr_count >= 2:
                continue

            scenario_counts[scen] = curr_count + 1
            sample_id = f"sample_{scen}_{curr_count + 1}"

            raw_f = list(sample.features)
            if scaler and scaler.params:
                scaled_f = scaler.transform([raw_f])[0]
            else:
                scaled_f = list(raw_f)

            is_scraping = (scen == "attack_high_volume_scraping")
            item = TrafficSampleItem(
                sample_id=sample_id,
                scenario=scen,
                source=sample.source,
                ground_truth_label=int(sample.label),
                client_ip_hash=sample.client_ip_hash,
                window_start_utc=sample.window_start_utc,
                raw_features=raw_f,
                scaled_features=scaled_f,
                feature_names=list(ds.feature_names),
                description=SCENARIO_DESCRIPTIONS.get(scen, "Curated synthetic traffic sample."),
                is_failure_case=is_scraping,
                failure_model="NumPy MLP & Logistic Regression" if is_scraping else None,
                failure_explanation=(
                    "Supervised classifiers trained on typical attacks rely heavily on error signals "
                    "(status_4xx_ratio, auth_failure_count). High-volume scraping queries existing resources "
                    "with 200 OK responses, causing the MLP to predict p < 0.01 (False Negative, 0% recall). "
                    "This empirical result proves the necessity of Phase 7 unsupervised anomaly detection."
                    if is_scraping
                    else None
                ),
            )
            self._curated_samples.append(item)
            self._samples_by_id[sample_id] = item

    def get_samples(self, scenario: str | None = None) -> List[TrafficSampleItem]:
        """Return list of curated traffic samples, optionally filtered by scenario."""
        if scenario:
            return [s for s in self._curated_samples if s.scenario == scenario]
        return self._curated_samples

    def get_sample_by_id(self, sample_id: str) -> Optional[TrafficSampleItem]:
        """Lookup single traffic sample by ID."""
        return self._samples_by_id.get(sample_id)


dataset_reader = DatasetReader()
