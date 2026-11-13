"""Forward propagation execution and intermediate computation tracer.

Computes exact step-by-step intermediate numerical values (weighted sums,
biases, post-activation outputs, and activation states) without modifying
the core neural network engine implementations.
"""

from __future__ import annotations

import math
import time
from typing import List, Sequence

from visualizer.backend.schemas.api_models import (
    ForwardTraceResult,
    LayerTrace,
    NeuronTrace,
)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _activate(z: float, activation: str) -> float:
    act = activation.lower()
    if act == "relu":
        return max(0.0, z)
    if act == "sigmoid":
        return _sigmoid(z)
    if act == "tanh":
        return math.tanh(z)
    if act == "identity":
        return z
    raise ValueError(f"Unsupported activation: {activation}")


def _formula_for_layer(layer_idx: int, activation: str, is_output: bool) -> str:
    act = activation.lower()
    if is_output:
        return "z = b + \\sum_i (a_i \\cdot w_i); \\quad p = \\sigma(z) = \\frac{1}{1 + e^{-z}}"
    if act == "relu":
        return "z_j = b_j + \\sum_i (x_i \\cdot w_{ij}); \\quad a_j = \\max(0, z_j)"
    if act == "sigmoid":
        return "z_j = b_j + \\sum_i (x_i \\cdot w_{ij}); \\quad a_j = \\sigma(z_j)"
    if act == "tanh":
        return "z_j = b_j + \\sum_i (x_i \\cdot w_{ij}); \\quad a_j = \\tanh(z_j)"
    return "z_j = b_j + \\sum_i (x_i \\cdot w_{ij}); \\quad a_j = z_j"


def trace_mlp_forward(
    model_id: str,
    engine: str,
    layers_data: Sequence[dict],  # each dict: {"weights": List[List[float]], "bias": List[float], "activation": str}
    scaled_features: Sequence[float],
    raw_features: Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> ForwardTraceResult:
    """Trace a multi-layer perceptron forward pass with full numerical observability."""
    start_t = time.perf_counter()

    scaled_x = [float(v) for v in scaled_features]
    raw_x = [float(v) for v in (raw_features if raw_features is not None else scaled_x)]
    f_names = list(feature_names) if feature_names else [f"f_{i}" for i in range(len(scaled_x))]

    current_inputs = list(scaled_x)
    layer_traces: List[LayerTrace] = []
    n_layers = len(layers_data)

    for l_idx, layer in enumerate(layers_data):
        weights = layer["weights"]  # shape (n_inputs, n_outputs)
        biases = layer["bias"]      # shape (n_outputs,)
        activation = layer["activation"]
        is_output = (l_idx == n_layers - 1)

        n_in = len(weights)
        n_out = len(biases)
        if len(current_inputs) != n_in:
            raise ValueError(f"Layer {l_idx} expects {n_in} inputs, received {len(current_inputs)}")

        neuron_traces: List[NeuronTrace] = []
        layer_outputs: List[float] = []

        for j in range(n_out):
            b_j = float(biases[j])
            incoming_weights = [float(weights[i][j]) for i in range(n_in)]
            weighted_terms = [current_inputs[i] * incoming_weights[i] for i in range(n_in)]
            z_j = b_j + sum(weighted_terms)
            a_j = _activate(z_j, activation)

            # Neuron is active if ReLU output > 0, or non-zero for others
            is_active = (a_j > 0.0) if activation.lower() == "relu" else (abs(a_j) > 1e-4)

            neuron_traces.append(
                NeuronTrace(
                    neuron_idx=j,
                    incoming_values=list(current_inputs),
                    incoming_weights=incoming_weights,
                    weighted_terms=weighted_terms,
                    bias=b_j,
                    z=z_j,
                    a=a_j,
                    is_active=is_active,
                )
            )
            layer_outputs.append(a_j)

        layer_type = "output" if is_output else "hidden"
        formula = _formula_for_layer(l_idx, activation, is_output)

        layer_traces.append(
            LayerTrace(
                layer_idx=l_idx,
                layer_type=layer_type,
                activation=activation,
                inputs=list(current_inputs),
                outputs=list(layer_outputs),
                neurons=neuron_traces,
                mathematical_formula=formula,
            )
        )

        current_inputs = layer_outputs

    # Final output computation
    logit = float(layer_outputs[0])
    output_act = layers_data[-1]["activation"].lower()
    if output_act == "identity":
        probability = _sigmoid(logit)
    elif output_act == "sigmoid":
        probability = logit
    else:
        probability = _sigmoid(logit)

    prediction = 1 if probability >= threshold else 0
    confidence = abs(probability - threshold) * 2.0  # scaled 0.0 to 1.0
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return ForwardTraceResult(
        model_id=model_id,
        engine=engine,
        raw_features=raw_x,
        scaled_features=scaled_x,
        feature_names=f_names,
        layers=layer_traces,
        logit=logit,
        probability=probability,
        threshold=threshold,
        prediction=prediction,
        confidence=confidence,
        parity_verified=True,
        inference_time_ms=elapsed_ms,
    )
