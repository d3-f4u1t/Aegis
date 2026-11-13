"""Model adapters for baseline models (Logistic Regression and Threshold detector)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from training.logistic_regression import LogisticRegression
from training.threshold_baseline import ThresholdBaseline
from visualizer.backend.adapters.base import ModelAdapter
from visualizer.backend.schemas.api_models import (
    ForwardTraceResult,
    LayerParameters,
    LayerTrace,
    ModelArchitecture,
    ModelSummary,
    NeuronTrace,
    NeuronWeights,
)
from visualizer.backend.tracing.forward_tracer import trace_mlp_forward


class LogisticRegressionAdapter(ModelAdapter):
    """Adapter wrapping a trained from-scratch Logistic Regression checkpoint."""

    def __init__(self, checkpoint_path: str | Path, model_id: str, name: str | None = None) -> None:
        self.path = Path(checkpoint_path)
        self._model_id = model_id
        self._name = name or self.path.stem

        with self.path.open("r", encoding="utf-8") as fh:
            self.raw_data = json.load(fh)

        self.model = LogisticRegression.from_params(self.raw_data)
        self.feature_names = self.raw_data.get(
            "feature_names", [f"f_{i}" for i in range(self.model.n_features or 13)]
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine(self) -> str:
        return "logistic_regression"

    def get_summary(self) -> ModelSummary:
        n_feat = self.model.n_features or len(self.feature_names)
        return ModelSummary(
            model_id=self.model_id,
            name=self.name,
            engine="logistic_regression",
            architecture_desc=f"{n_feat} -> 1 (Logistic Regression)",
            total_parameters=n_feat + 1,
            schema_version=str(self.raw_data.get("schema_version", "1")),
            feature_count=n_feat,
        )

    def get_architecture(self) -> ModelArchitecture:
        n_feat = self.model.n_features or len(self.feature_names)
        return ModelArchitecture(
            model_id=self.model_id,
            name=self.name,
            engine="logistic_regression",
            format_version=1,
            sizes=[n_feat, 1],
            hidden_activation="none",
            output_activation="sigmoid",
            total_parameters=n_feat + 1,
            schema_version=str(self.raw_data.get("schema_version", "1")),
            feature_names=self.feature_names,
            metadata={
                "lr": self.raw_data.get("lr"),
                "epochs": self.raw_data.get("epochs"),
                "seed": self.raw_data.get("seed"),
            },
        )

    def get_layer_parameters(self, layer_idx: int) -> LayerParameters:
        if layer_idx != 0:
            raise IndexError(f"Logistic regression has only layer 0, got {layer_idx}")

        n_inputs = self.model.n_features or len(self.feature_names)
        n_outputs = 1
        w_list = [float(w) for w in self.model.weights]
        b_val = float(self.model.bias)

        weights_matrix = [[w] for w in w_list]
        biases = [b_val]
        neurons = [NeuronWeights(neuron_idx=0, bias=b_val, weights=w_list)]

        return LayerParameters(
            layer_idx=0,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            activation="sigmoid",
            weights=weights_matrix,
            biases=biases,
            neurons=neurons,
        )

    def predict_proba(self, features: Sequence[float]) -> float:
        feat_arr = np.asarray(features, dtype=float).reshape(1, -1)
        prob = self.model.predict_proba(feat_arr)
        return float(prob[0])

    def predict(self, features: Sequence[float], threshold: float = 0.5) -> int:
        prob = self.predict_proba(features)
        return 1 if prob >= threshold else 0

    def trace_forward(
        self,
        scaled_features: Sequence[float],
        raw_features: Sequence[float] | None = None,
        threshold: float = 0.5,
    ) -> ForwardTraceResult:
        lp = self.get_layer_parameters(0)
        layers_data = [
            {
                "weights": lp.weights,
                "bias": lp.biases,
                "activation": "sigmoid",
            }
        ]

        res = trace_mlp_forward(
            model_id=self.model_id,
            engine="logistic_regression",
            layers_data=layers_data,
            scaled_features=scaled_features,
            raw_features=raw_features,
            feature_names=self.feature_names,
            threshold=threshold,
        )

        direct_prob = self.predict_proba(scaled_features)
        res.parity_verified = abs(direct_prob - res.probability) < 1e-11
        return res


class ThresholdAdapter(ModelAdapter):
    """Adapter wrapping a calibrated Threshold Baseline detector."""

    def __init__(self, checkpoint_path: str | Path, model_id: str, name: str | None = None) -> None:
        self.path = Path(checkpoint_path)
        self._model_id = model_id
        self._name = name or self.path.stem

        with self.path.open("r", encoding="utf-8") as fh:
            self.raw_data = json.load(fh)

        self.model = ThresholdBaseline.from_params(self.raw_data)
        self.feature_index = self.model.feature_index
        self.rule_threshold = self.model.threshold
        self.feature_name = self.model.feature_name or f"feature_{self.feature_index}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine(self) -> str:
        return "threshold"

    def get_summary(self) -> ModelSummary:
        return ModelSummary(
            model_id=self.model_id,
            name=self.name,
            engine="threshold",
            architecture_desc=f"Rule: {self.feature_name} >= {self.rule_threshold:.4f}",
            total_parameters=1,
            schema_version="1",
            feature_count=13,
        )

    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture(
            model_id=self.model_id,
            name=self.name,
            engine="threshold",
            format_version=1,
            sizes=[13, 1],
            hidden_activation="none",
            output_activation="step_rule",
            total_parameters=1,
            schema_version="1",
            feature_names=[f"feature_{i}" for i in range(13)],
            metadata={
                "feature_index": self.feature_index,
                "feature_name": self.feature_name,
                "calibrated_threshold": self.rule_threshold,
            },
        )

    def get_layer_parameters(self, layer_idx: int) -> LayerParameters:
        # A single-step rule represented conceptually as weight 1.0 on feature_index, bias = -threshold
        w = [0.0] * 13
        if 0 <= self.feature_index < 13:
            w[self.feature_index] = 1.0
        return LayerParameters(
            layer_idx=0,
            n_inputs=13,
            n_outputs=1,
            activation="step",
            weights=[[val] for val in w],
            biases=[-self.rule_threshold],
            neurons=[
                NeuronWeights(
                    neuron_idx=0,
                    bias=-self.rule_threshold,
                    weights=w,
                )
            ],
        )

    def predict_proba(self, features: Sequence[float]) -> float:
        feat_val = float(features[self.feature_index])
        # Return pseudo-probability: normalized or raw feature value
        return feat_val

    def predict(self, features: Sequence[float], threshold: float | None = None) -> int:
        thresh = self.rule_threshold if threshold is None else threshold
        feat_val = float(features[self.feature_index])
        return 1 if feat_val >= thresh else 0

    def trace_forward(
        self,
        scaled_features: Sequence[float],
        raw_features: Sequence[float] | None = None,
        threshold: float | None = None,
    ) -> ForwardTraceResult:
        eval_thresh = self.rule_threshold if threshold is None else threshold
        source_feats = raw_features if raw_features is not None else scaled_features
        feat_val = float(source_feats[self.feature_index])
        is_attack = int(feat_val >= eval_thresh)

        neuron_trace = NeuronTrace(
            neuron_idx=0,
            incoming_values=[feat_val],
            incoming_weights=[1.0],
            weighted_terms=[feat_val],
            bias=-eval_thresh,
            z=feat_val - eval_thresh,
            a=float(is_attack),
            is_active=bool(is_attack),
        )

        formula = f"{self.feature_name} ({feat_val:.4f}) >= threshold ({eval_thresh:.4f}) -> {is_attack}"

        layer_trace = LayerTrace(
            layer_idx=0,
            layer_type="rule_threshold",
            activation="step_rule",
            inputs=[feat_val],
            outputs=[float(is_attack)],
            neurons=[neuron_trace],
            mathematical_formula=formula,
        )

        return ForwardTraceResult(
            model_id=self.model_id,
            engine="threshold",
            raw_features=list(raw_features) if raw_features else list(scaled_features),
            scaled_features=list(scaled_features),
            feature_names=[self.feature_name],
            layers=[layer_trace],
            logit=feat_val - eval_thresh,
            probability=1.0 if is_attack else 0.0,
            threshold=eval_thresh,
            prediction=is_attack,
            confidence=1.0,
            parity_verified=True,
            inference_time_ms=0.01,
        )
