"""Tests for extended evaluation metrics."""

import pytest

from training.evaluate import (
    average_precision,
    operating_point,
    pr_auc,
    precision_recall_curve,
    threshold_sweep,
)


def test_precision_recall_curve_shapes():
    y_true = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    assert len(precisions) == len(recalls) == len(thresholds)
    assert precisions[0] == 1.0  # threshold above all: no predictions
    assert recalls[0] == 0.0


def test_pr_auc_perfect_separation():
    y_true = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert pr_auc(y_true, scores) == pytest.approx(1.0)


def test_pr_auc_degenerate_all_same_score():
    # All scores identical: single operating point at recall=1.0,
    # precision = proportion of positives = 0.5.
    # PR-AUC = (1-0) * (1+0.5)/2 = 0.75
    y_true = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    auc = pr_auc(y_true, scores)
    assert auc == pytest.approx(0.75)


def test_pr_auc_in_range():
    y_true = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.4, 0.6, 0.3, 0.8]
    auc = pr_auc(y_true, scores)
    assert 0.0 <= auc <= 1.0


def test_operating_point_meets_fpr_constraint():
    y_true = [1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    op = operating_point(y_true, scores, max_fpr=0.5)
    assert op["false_positive_rate"] <= 0.5
    assert 0.0 <= op["threshold"] <= 1.0


def test_threshold_sweep():
    y_true = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    results = threshold_sweep(y_true, scores)
    assert len(results) == 4
    assert results[0]["threshold"] == 0.9
    assert "precision" in results[0]
    assert "recall" in results[0]
    assert "f1" in results[0]
    assert "false_positive_rate" in results[0]


def test_average_precision_perfect_separation():
    y_true = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert average_precision(y_true, scores) == pytest.approx(1.0)


def test_average_precision_degenerate_all_same_score():
    # Unlike trapezoidal PR-AUC which overestimates to 0.75,
    # AP for uninformative classifier equals positive base rate (0.5).
    y_true = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    ap = average_precision(y_true, scores)
    assert ap == pytest.approx(0.50)


def test_average_precision_intermediate():
    y_true = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    # At 0.9: TP=1, FP=0 -> P=1.0, R=0.5 -> delta_r = 0.5 -> contrib = 0.5 * 1.0 = 0.5
    # At 0.8: TP=1, FP=1 -> P=0.5, R=0.5 -> delta_r = 0.0 -> contrib = 0.0
    # At 0.2: TP=2, FP=1 -> P=2/3, R=1.0 -> delta_r = 0.5 -> contrib = 0.5 * (2/3) = 1/3
    # At 0.1: TP=2, FP=2 -> P=0.5, R=1.0 -> delta_r = 0.0 -> contrib = 0.0
    # Total AP = 0.5 + 1/3 = 5/6 = 0.8333...
    ap = average_precision(y_true, scores)
    assert ap == pytest.approx(5.0 / 6.0)
