"""Artifact reader for Phase 5 experiment logs and evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from visualizer.backend.schemas.api_models import (
    ExperimentModelResult,
    ExperimentReport,
    ScenarioMetric,
)

EXPERIMENT_METADATA = {
    "exp1_indistribution": {
        "name": "Experiment 1: In-Distribution Evaluation",
        "description": "Standard 60/20/20 client-stratified split across all 13 traffic scenarios. Tests baseline detection capability when all attack patterns are seen during training.",
        "split_strategy": "Stratified Client Split (60% Train, 20% Val, 20% Test)",
    },
    "exp2_unseen_attacks": {
        "name": "Experiment 2: Zero-Day Unseen Attack Generalization",
        "description": "Holds out brute-force attacks and high-volume scraping completely from training and validation. Evaluates classifier generalization to novel attack vectors.",
        "split_strategy": "Scenario Holdout Split (Held out: attack_brute_force, attack_high_volume_scraping)",
    },
    "exp3_unseen_polling": {
        "name": "Experiment 3: Unseen Benign Automation Generalization",
        "description": "Holds out high-frequency benign API polling from training. Tests whether novel automated benign traffic triggers false positive alarms.",
        "split_strategy": "Scenario Holdout Split (Held out: benign_api_polling)",
    },
}


class ExperimentReader:
    """Loads and caches experimental summaries from JSON artifacts."""

    def __init__(self, summary_path: str | Path | None = None) -> None:
        self.summary_path = (
            Path(summary_path) if summary_path else Path("experiments/results/phase5_summary.json")
        )
        self._reports: Dict[str, ExperimentReport] = {}
        self._load()

    def _load(self) -> None:
        if not self.summary_path.exists():
            return

        with self.summary_path.open("r", encoding="utf-8") as fh:
            raw_data = json.load(fh)

        for exp_key, raw_exp in raw_data.items():
            meta = EXPERIMENT_METADATA.get(
                exp_key,
                {
                    "name": raw_exp.get("experiment_name", exp_key),
                    "description": "Phase 5 telemetry anomaly detection experiment.",
                    "split_strategy": "Unknown",
                },
            )

            models_dict: Dict[str, ExperimentModelResult] = {}
            for model_name, raw_m in raw_exp.get("model_results", {}).items():
                per_scen: Dict[str, ScenarioMetric] = {}
                for s_name, s_data in raw_m.get("per_scenario", {}).items():
                    per_scen[s_name] = ScenarioMetric(
                        total_samples=float(s_data.get("total_samples", 0)),
                        accuracy=float(s_data.get("accuracy", 0.0)),
                        recall=float(s_data.get("recall", 0.0)),
                        false_positive_rate=float(s_data.get("false_positive_rate", 0.0)),
                        tp=float(s_data.get("tp", 0.0)),
                        fp=float(s_data.get("fp", 0.0)),
                        tn=float(s_data.get("tn", 0.0)),
                        fn=float(s_data.get("fn", 0.0)),
                    )

                models_dict[model_name] = ExperimentModelResult(
                    model_name=model_name,
                    accuracy=float(raw_m.get("accuracy", 0.0)),
                    precision=float(raw_m.get("precision", 0.0)),
                    recall=float(raw_m.get("recall", 0.0)),
                    f1=float(raw_m.get("f1", 0.0)),
                    false_positive_rate=float(raw_m.get("false_positive_rate", 0.0)),
                    false_negative_rate=float(raw_m.get("false_negative_rate", 0.0)),
                    tp=float(raw_m.get("tp", 0.0)),
                    fp=float(raw_m.get("fp", 0.0)),
                    tn=float(raw_m.get("tn", 0.0)),
                    fn=float(raw_m.get("fn", 0.0)),
                    pr_auc=float(raw_m.get("pr_auc", 0.0)),
                    average_precision=float(raw_m.get("average_precision", 0.0)),
                    inference_latency_ms=float(raw_m.get("inference_latency_ms_per_sample", 0.0)),
                    per_scenario=per_scen,
                )

            report = ExperimentReport(
                experiment_id=exp_key,
                name=meta["name"],
                description=meta["description"],
                dataset_size=int(raw_exp.get("dataset_size", 0)),
                n_train=int(raw_exp.get("n_train", 0)),
                n_val=int(raw_exp.get("n_val", 0)),
                n_test=int(raw_exp.get("n_test", 0)),
                positives_test=int(raw_exp.get("n_positive_test", 0)),
                split_strategy=meta["split_strategy"],
                models=models_dict,
            )
            self._reports[exp_key] = report

    def list_experiments(self) -> List[dict]:
        """Return list of experiment descriptors for navigation."""
        items = []
        for exp_id, report in self._reports.items():
            items.append({
                "experiment_id": exp_id,
                "name": report.name,
                "description": report.description,
                "dataset_size": report.dataset_size,
                "models_evaluated": list(report.models.keys()),
            })
        return items

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentReport]:
        """Get full experiment report with models and scenario breakdowns."""
        return self._reports.get(experiment_id)


experiment_reader = ExperimentReader()
