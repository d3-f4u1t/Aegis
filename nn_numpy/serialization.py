"""NumPy model persistence.

Checkpoint format is JSON and mirrors the scalar engine's ``nn.serialization``:

    format_version  schema version (2 for the NumPy engine)
    architecture    sizes, hidden/output activation, seed, init
    layers          per layer: weight matrix W and bias vector b (nested lists)
    metadata        free-form training information

Differences from the scalar checkpoint (documented intentionally):

* The scalar engine stores per-neuron weight lists and biases nested under
  ``layers[].neurons[]``. The NumPy engine stores the full weight matrix
  ``W`` (shape ``(n_inputs, n_outputs)``) and bias vector ``b`` per layer.
* ``format_version`` is 2 so the two formats are never confused.
* NumPy does not store per-neuron groupings; the same architecture rebuilds the
  layer shapes, and the matrices are loaded directly.

A checkpoint saved by one engine is not directly loadable by the other without
a conversion step. This is intentional: the formats have different layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import MLP

FORMAT_VERSION = 2


def model_to_dict(model: MLP, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "engine": "numpy",
        "architecture": {
            "sizes": list(model.sizes),
            "hidden_activation": model.hidden_activation,
            "output_activation": model.output_activation,
            "seed": model.seed,
            "init": model.init,
        },
        "layers": [
            {
                "weights": layer.W.value.tolist(),
                "bias": layer.b.value.tolist(),
            }
            for layer in model.layers
        ],
        "metadata": dict(metadata or {}),
    }


def save_model(model: MLP, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
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
    if payload.get("engine") != "numpy":
        raise ValueError(
            f"checkpoint engine {payload.get('engine')!r} is not 'numpy'"
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
        weights = np.array(layer_data.get("weights", []), dtype=float)
        if weights.shape != (layer.n_inputs, layer.n_outputs):
            raise ValueError("weight matrix shape does not match the architecture")
        layer.W.value = weights
        bias = np.array(layer_data.get("bias", []), dtype=float)
        if bias.shape != (layer.n_outputs,):
            raise ValueError("bias vector shape does not match the architecture")
        layer.b.value = bias

    model.metadata = dict(payload.get("metadata", {}))
    return model
