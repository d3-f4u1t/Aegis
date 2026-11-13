"""Model adapter for the scalar neural network engine (Format Version 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence

from nn.model import MLP as ScalarMLP
from nn.serialization import load_model as load_scalar_model
from visualizer.backend.adapters.base import ModelAdapter
from visualizer.backend.schemas.api_models import (
    ForwardTraceResult,
    LayerParameters,
    ModelArchitecture,
    ModelSummary,
    NeuronWeights,
)
from visualizer.backend.tracing.forward_tracer import trace_mlp_forward


class ScalarMLPAdapter(ModelAdapter):
    """Adapter wrapping an audited scalar reference MLP checkpoint."""

    def __init__(self, checkpoint_path: str | Path, model_id: str, name: str | None = None) -> None:
        self.path = Path(checkpoint_path)
        self._model_id = model_id
        self._name = name or self.path.stem

        with self.path.open("r", encoding="utf-8") as fh:
            self.raw_data = json.load(fh)

        self.model = load_scalar_model(self.path)
        self.arch_info = self.raw_data.get("architecture", {})
        self.metadata_info = self.raw_data.get("metadata", {})
        self.feature_names = self.metadata_info.get("feature_names", [f"f_{i}" for i in range(self.model.sizes[0])])

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine(self) -> str:
        return "scalar"

    def get_summary(self) -> ModelSummary:
        return ModelSummary(
            model_id=self.model_id,
            name=self.name,
            engine="scalar",
            architecture_desc=" -> ".join(str(s) for s in self.model.sizes),
            total_parameters=self.model.n_parameters,
            schema_version=str(self.metadata_info.get("schema_version", "1")),
            feature_count=self.model.sizes[0],
        )

    def get_architecture(self) -> ModelArchitecture:
        return ModelArchitecture(
            model_id=self.model_id,
            name=self.name,
            engine="scalar",
            format_version=1,
            sizes=list(self.model.sizes),
            hidden_activation=self.model.hidden_activation,
            output_activation=self.model.output_activation,
            total_parameters=self.model.n_parameters,
            schema_version=str(self.metadata_info.get("schema_version", "1")),
            feature_names=self.feature_names,
            metadata=self.metadata_info,
        )

    def get_layer_parameters(self, layer_idx: int) -> LayerParameters:
        if layer_idx < 0 or layer_idx >= len(self.model.layers):
            raise IndexError(f"Layer index {layer_idx} out of range (0..{len(self.model.layers)-1})")

        layer = self.model.layers[layer_idx]
        n_inputs = self.model.sizes[layer_idx]
        n_outputs = self.model.sizes[layer_idx + 1]
        act_name = self.model.output_activation if layer_idx == len(self.model.layers) - 1 else self.model.hidden_activation

        # Reconstruct matrix W (n_inputs, n_outputs) and biases (n_outputs,)
        weights_matrix: List[List[float]] = [[0.0] * n_outputs for _ in range(n_inputs)]
        biases: List[float] = []
        neurons_list: List[NeuronWeights] = []

        for j, neuron in enumerate(layer.neurons):
            b_val = float(neuron.bias.value)
            biases.append(b_val)
            w_vals = [float(w.value) for w in neuron.weights]
            neurons_list.append(NeuronWeights(neuron_idx=j, bias=b_val, weights=w_vals))
            for i in range(n_inputs):
                weights_matrix[i][j] = w_vals[i]

        return LayerParameters(
            layer_idx=layer_idx,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            activation=act_name,
            weights=weights_matrix,
            biases=biases,
            neurons=neurons_list,
        )

    def predict_proba(self, features: Sequence[float]) -> float:
        return float(self.model.predict_proba(features))

    def predict(self, features: Sequence[float], threshold: float = 0.5) -> int:
        return int(self.model.predict(features, threshold=threshold))

    def trace_forward(
        self,
        scaled_features: Sequence[float],
        raw_features: Sequence[float] | None = None,
        threshold: float = 0.5,
    ) -> ForwardTraceResult:
        layers_data = []
        for l_idx in range(len(self.model.layers)):
            lp = self.get_layer_parameters(l_idx)
            layers_data.append({
                "weights": lp.weights,
                "bias": lp.biases,
                "activation": lp.activation,
            })

        res = trace_mlp_forward(
            model_id=self.model_id,
            engine="scalar",
            layers_data=layers_data,
            scaled_features=scaled_features,
            raw_features=raw_features,
            feature_names=self.feature_names,
            threshold=threshold,
        )

        # Parity check against actual engine
        direct_prob = self.predict_proba(scaled_features)
        parity = abs(direct_prob - res.probability) < 1e-11
        res.parity_verified = parity
        return res
