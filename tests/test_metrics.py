import pytest

from training.metrics import binary_confusion, binary_metrics, detection_rate_at_fpr


def test_confusion_matrix():
    cm = binary_confusion([1, 1, 0, 0], [1, 0, 1, 0])
    assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_binary_metrics():
    metrics = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)


def test_all_correct_predictions():
    metrics = binary_metrics([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["false_negative_rate"] == pytest.approx(0.0)


def test_detection_rate_at_zero_false_positives():
    detection, threshold = detection_rate_at_fpr(
        [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], max_fpr=0.0
    )
    assert detection == pytest.approx(1.0)
    assert threshold == pytest.approx(0.8)
