"""Unit tests for visualizer model adapters."""

import pytest
from visualizer.backend.adapters.registry import registry
from visualizer.backend.schemas.api_models import LayerParameters, ModelArchitecture, ModelSummary


def test_registry_discovers_models():
    models = registry.list_models()
    assert len(models) >= 10
    model_ids = {m.model_id for m in models}
    assert "phase5_exp1_indistribution_numpy_mlp" in model_ids
    assert "phase5_exp1_indistribution_scalar_mlp" in model_ids
    assert "phase5_exp1_indistribution_logistic_regression" in model_ids
    assert "phase5_exp1_indistribution_threshold" in model_ids


def test_numpy_mlp_adapter():
    adapter = registry.get_model("phase5_exp1_indistribution_numpy_mlp")
    assert adapter is not None
    assert adapter.engine == "numpy"

    summary = adapter.get_summary()
    assert isinstance(summary, ModelSummary)
    assert summary.feature_count == 13
    assert "13 -> 16 -> 8 -> 1" in summary.architecture_desc

    arch = adapter.get_architecture()
    assert isinstance(arch, ModelArchitecture)
    assert arch.sizes == [13, 16, 8, 1]
    assert arch.format_version == 2

    # Layer 0 parameters
    l0 = adapter.get_layer_parameters(0)
    assert isinstance(l0, LayerParameters)
    assert l0.n_inputs == 13
    assert l0.n_outputs == 16
    assert len(l0.weights) == 13
    assert len(l0.weights[0]) == 16
    assert len(l0.biases) == 16
    assert len(l0.neurons) == 16

    # Layer out of bounds
    with pytest.raises(IndexError):
        adapter.get_layer_parameters(99)


def test_scalar_mlp_adapter():
    adapter = registry.get_model("phase5_exp1_indistribution_scalar_mlp")
    assert adapter is not None
    assert adapter.engine == "scalar"

    arch = adapter.get_architecture()
    assert arch.sizes == [13, 16, 8, 1]
    assert arch.format_version == 1

    l1 = adapter.get_layer_parameters(1)
    assert l1.n_inputs == 16
    assert l1.n_outputs == 8
    assert len(l1.weights) == 16
    assert len(l1.biases) == 8


def test_logistic_regression_adapter():
    adapter = registry.get_model("phase5_exp1_indistribution_logistic_regression")
    assert adapter is not None
    assert adapter.engine == "logistic_regression"

    arch = adapter.get_architecture()
    assert arch.sizes == [13, 1]

    l0 = adapter.get_layer_parameters(0)
    assert l0.n_inputs == 13
    assert l0.n_outputs == 1
    assert len(l0.weights) == 13
    assert len(l0.biases) == 1

    with pytest.raises(IndexError):
        adapter.get_layer_parameters(1)


def test_threshold_adapter():
    adapter = registry.get_model("phase5_exp1_indistribution_threshold")
    assert adapter is not None
    assert adapter.engine == "threshold"

    arch = adapter.get_architecture()
    assert arch.sizes == [13, 1]

    pred = adapter.predict([0.0] * 13)
    assert pred in (0, 1)
