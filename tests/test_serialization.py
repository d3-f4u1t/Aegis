import json

import pytest

from nn.model import MLP
from nn.serialization import load_model, save_model


def test_round_trip_preserves_outputs_and_metadata(tmp_path):
    model = MLP([3, 4, 2, 1], hidden_activation="tanh", seed=19)
    inputs = [0.1, -0.2, 0.3]
    expected = model.forward(inputs)

    path = save_model(
        model,
        tmp_path / "models" / "model.json",
        metadata={"feature_schema": "v1", "trained_on": "tiny-test"},
    )
    loaded = load_model(path)

    assert loaded.forward(inputs) == pytest.approx(expected)
    assert loaded.sizes == model.sizes
    assert loaded.hidden_activation == "tanh"
    assert loaded.metadata["feature_schema"] == "v1"
    assert loaded.metadata["trained_on"] == "tiny-test"


def test_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_model(path)


def test_rejects_architecture_mismatch(tmp_path):
    model = MLP([2, 2, 1], seed=0)
    path = save_model(model, tmp_path / "model.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["layers"][0]["neurons"][0]["weights"] = [1.0]  # should be 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="weight count"):
        load_model(path)
