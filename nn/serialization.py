"""Model persistence.

A checkpoint is a single JSON document containing:

    format_version  schema version of the file itself
    architecture    everything needed to rebuild the layer shapes
    layers          weights and biases, nested per layer/neuron
    metadata        free-form training information (feature schema, metrics...)

JSON is deliberately chosen for the first version: it is inspectable, diffable,
and has no hidden binary layout. NumPy ``.npz`` (or a compact binary format)
will replace it once the parameter count makes JSON impractical. Whatever the
format, the rule stays: architecture + preprocessing + schema travel with the
weights, or the checkpoint is not loadable in a meaningful way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import MLP

FORMAT_VERSION = 1


def model_to_dict(model: MLP, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "architecture": {
            "sizes": list(model.sizes),
            "hidden_activation": model.hidden_activation,
            "output_activation": model.output_activation,
            "seed": model.seed,
            "init": model.init,
        },
        "layers": [
            {
                "neurons": [
                    {
                        "weights": [w.value for w in neuron.weights],
                        "bias": neuron.bias.value,
                    }
                    for neuron in layer.neurons
                ]
            }
            for layer in model.layers
        ],
        "metadata": dict(metadata or {}),
    }


def save_model(
    model: MLP, path: str | Path, metadata: dict[str, Any] | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model_to_dict(model, metadata)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def load_model(path: str | Path) -> MLP:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version!r}; expected {FORMAT_VERSION}"
        )

    arch = payload["architecture"]
    model = MLP(
        arch["sizes"],
        hidden_activation=arch["hidden_activation"],
        output_activation=arch["output_activation"],
        seed=arch.get("seed", 0),
        init=arch.get("init"),
    )

    layers = payload.get("layers", [])
    if len(layers) != len(model.layers):
        raise ValueError(
            f"checkpoint has {len(layers)} layers, architecture needs {len(model.layers)}"
        )
    for layer, layer_data in zip(model.layers, layers):
        neurons = layer_data.get("neurons", [])
        if len(neurons) != len(layer.neurons):
            raise ValueError("neuron count does not match the architecture")
        for neuron, neuron_data in zip(layer.neurons, neurons):
            weights = neuron_data.get("weights", [])
            if len(weights) != len(neuron.weights):
                raise ValueError("weight count does not match the architecture")
            for param, value in zip(neuron.weights, weights):
                param.value = float(value)
            neuron.bias.value = float(neuron_data.get("bias", 0.0))

    model.metadata = dict(payload.get("metadata", {}))
    return model
