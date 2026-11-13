"""Tests for the preprocessing pipeline."""

import tempfile
from pathlib import Path

import pytest

from training.preprocessing import PreprocessingParams, StandardScaler


def test_fit_and_transform():
    scaler = StandardScaler()
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    params = scaler.fit(X, ["a", "b"])

    assert params.means == pytest.approx([3.0, 4.0])
    assert params.stds == pytest.approx([1.632993161855452, 1.632993161855452])

    transformed = scaler.transform(X)
    expected = [[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]
    for row, exp_row in zip(transformed, expected):
        assert row == pytest.approx(exp_row, abs=1e-4)


def test_fit_transform_convenience():
    scaler = StandardScaler()
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    transformed = scaler.fit_transform(X, ["a", "b"])
    assert len(transformed) == 3
    assert len(transformed[0]) == 2


def test_zero_variance_feature_does_not_divide_by_zero():
    scaler = StandardScaler()
    X = [[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]]
    scaler.fit(X, ["a", "b"])
    assert scaler.params.stds[0] == 1.0  # constant feature
    transformed = scaler.transform(X)
    assert all(row[0] == 0.0 for row in transformed)


def test_save_and_load_roundtrip(tmp_path):
    scaler = StandardScaler()
    X = [[1.0, 2.0], [3.0, 4.0]]
    scaler.fit(X, ["a", "b"], schema_version="1")

    path = Path(tmp_path) / "preprocessing.json"
    scaler.save(path)

    loaded = StandardScaler.load(path)
    assert loaded.params is not None
    assert loaded.params.schema_version == "1"
    assert loaded.params.feature_names == ["a", "b"]
    assert loaded.params.means == pytest.approx([2.0, 3.0])

    # Transforms should match
    X_test = [[5.0, 6.0]]
    assert scaler.transform(X_test) == loaded.transform(X_test)


def test_schema_mismatch_detection():
    scaler = StandardScaler()
    X = [[1.0, 2.0]]
    scaler.fit(X, ["a", "b"], schema_version="1")

    assert scaler.schema_matches("1", ["a", "b"]) is True
    assert scaler.schema_matches("2", ["a", "b"]) is False
    assert scaler.schema_matches("1", ["a", "c"]) is False


def test_transform_before_fit_raises():
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not been fitted"):
        scaler.transform([[1.0, 2.0]])


def test_preprocessing_params_serialization():
    params = PreprocessingParams(
        feature_names=["a", "b"],
        means=[1.0, 2.0],
        stds=[0.5, 0.5],
        schema_version="1",
    )
    d = params.to_dict()
    assert d["feature_names"] == ["a", "b"]
    assert d["means"] == [1.0, 2.0]
    assert d["stds"] == [0.5, 0.5]
    assert d["schema_version"] == "1"

    restored = PreprocessingParams.from_dict(d)
    assert restored.feature_names == ["a", "b"]
    assert restored.means == [1.0, 2.0]
    assert restored.stds == [0.5, 0.5]
    assert restored.schema_version == "1"
