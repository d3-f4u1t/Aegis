"""Tests for threshold baseline."""

import pytest

from training.threshold_baseline import ThresholdBaseline


def test_threshold_baseline_fit_and_predict():
    X = [[0.1], [0.5], [0.9], [0.2], [0.8]]
    y = [0.0, 0.0, 1.0, 0.0, 1.0]
    model = ThresholdBaseline(feature_index=0)
    model.fit(X, y, max_fpr=0.5)
    preds = model.predict(X)
    assert len(preds) == 5
    assert all(p in (0, 1) for p in preds)


def test_threshold_baseline_params_roundtrip():
    model = ThresholdBaseline(feature_index=2, threshold=0.75)
    model.feature_name = "requests_per_second"
    params = model.get_params()
    restored = ThresholdBaseline.from_params(params)
    assert restored.feature_index == 2
    assert restored.threshold == 0.75
    assert restored.feature_name == "requests_per_second"


def test_threshold_predict_proba():
    model = ThresholdBaseline(feature_index=0, threshold=0.5)
    X = [[0.3], [0.7], [0.5]]
    probs = model.predict_proba(X)
    assert probs == [0.3, 0.7, 0.5]
