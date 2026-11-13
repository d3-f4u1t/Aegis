"""Unit tests for the forward propagation tracer and mathematical parity."""

import pytest
from visualizer.backend.adapters.registry import registry
from visualizer.backend.schemas.api_models import ForwardTraceResult


def test_numpy_mlp_tracer_parity():
    adapter = registry.get_model("phase5_exp1_indistribution_numpy_mlp")
    assert adapter is not None

    sample = [0.1, -0.5, 1.2, 0.0, 2.3, 0.4, 0.0, 0.0, 1.0, 50.0, 120.0, 65.0, 2500.0]
    scaled = registry.scale_features(adapter.model_id, sample)

    trace = adapter.trace_forward(scaled_features=scaled, raw_features=sample)
    assert isinstance(trace, ForwardTraceResult)
    assert trace.parity_verified is True

    direct_prob = adapter.predict_proba(scaled)
    assert abs(trace.probability - direct_prob) < 1e-11

    # Check intermediate layer values
    assert len(trace.layers) == 3
    # Layer 0: 16 neurons
    assert len(trace.layers[0].neurons) == 16
    n0 = trace.layers[0].neurons[0]
    expected_z = n0.bias + sum(t for t in n0.weighted_terms)
    assert abs(n0.z - expected_z) < 1e-12
    # ReLU activation
    assert n0.a == max(0.0, n0.z)


def test_scalar_mlp_tracer_parity():
    adapter = registry.get_model("phase5_exp1_indistribution_scalar_mlp")
    assert adapter is not None

    sample = [0.1, -0.5, 1.2, 0.0, 2.3, 0.4, 0.0, 0.0, 1.0, 50.0, 120.0, 65.0, 2500.0]
    scaled = registry.scale_features(adapter.model_id, sample)

    trace = adapter.trace_forward(scaled_features=scaled, raw_features=sample)
    assert trace.parity_verified is True
    direct_prob = adapter.predict_proba(scaled)
    assert abs(trace.probability - direct_prob) < 1e-11


def test_logistic_regression_tracer_parity():
    adapter = registry.get_model("phase5_exp1_indistribution_logistic_regression")
    assert adapter is not None

    sample = [0.1, -0.5, 1.2, 0.0, 2.3, 0.4, 0.0, 0.0, 1.0, 50.0, 120.0, 65.0, 2500.0]
    scaled = registry.scale_features(adapter.model_id, sample)

    trace = adapter.trace_forward(scaled_features=scaled, raw_features=sample)
    assert trace.parity_verified is True
    direct_prob = adapter.predict_proba(scaled)
    assert abs(trace.probability - direct_prob) < 1e-11


def test_threshold_tracer():
    adapter = registry.get_model("phase5_exp1_indistribution_threshold")
    assert adapter is not None

    sample = [0.0] * 13
    sample[1] = 100.0  # requests_per_second high -> should be attack
    trace = adapter.trace_forward(scaled_features=sample, raw_features=sample)
    assert trace.prediction == 1
    assert trace.probability == 1.0


def test_dimension_mismatch_raises():
    adapter = registry.get_model("phase5_exp1_indistribution_numpy_mlp")
    assert adapter is not None

    with pytest.raises(ValueError):
        adapter.trace_forward(scaled_features=[1.0, 2.0])  # only 2 features instead of 13
