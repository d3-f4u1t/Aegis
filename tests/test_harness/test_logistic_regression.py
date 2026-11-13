"""Tests for logistic regression baseline."""

import pytest

from training.logistic_regression import LogisticRegression


def test_logistic_regression_perfect_separation():
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]  # XOR - not linearly separable, but LR should try
    model = LogisticRegression(lr=0.1, epochs=5000, seed=42)
    model.fit(X, y, feature_names=["x1", "x2"], schema_version="1")
    probs = model.predict_proba(X).tolist()
    # Just check it runs and produces valid probabilities
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_logistic_regression_linearly_separable():
    X = [[0.0, 0.0], [0.0, 0.1], [1.0, 0.9], [1.0, 1.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    model = LogisticRegression(lr=0.1, epochs=5000, seed=42)
    model.fit(X, y, feature_names=["x1", "x2"], schema_version="1")
    probs = model.predict_proba(X).tolist()
    preds = model.predict(X).tolist()
    assert preds == [0, 0, 1, 1]


def test_logistic_regression_params_roundtrip():
    model = LogisticRegression(lr=0.01, epochs=100, seed=7)
    X = [[0.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0]
    model.fit(X, y, feature_names=["a", "b"], schema_version="1")
    params = model.get_params()
    restored = LogisticRegression.from_params(params)
    assert restored.n_features == 2
    assert restored.bias == model.bias
    for w1, w2 in zip(restored.weights, model.weights):
        assert w1 == pytest.approx(w2)


def test_predict_proba_before_fit_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict_proba([[0.0, 0.0]])
