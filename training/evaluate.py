"""Extended evaluation metrics for binary classification.

Builds on ``training.metrics`` (confusion matrix, precision, recall, F1,
FPR/FNR) and adds:

* Precision-recall curve and PR-AUC (area under the PR curve).
* Threshold sweeping for operating-point selection.
* Baseline comparison helpers.

All functions are pure Python with no deep-learning dependencies.
"""

from __future__ import annotations

import math
from typing import Sequence

from training.metrics import binary_confusion, binary_metrics


def precision_recall_curve(
    y_true: Sequence[int], scores: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    """Compute precision-recall curve points.

    Sweeps every observed score as a candidate threshold (from high to low)
    and returns ``(precisions, recalls, thresholds)`` where each index
    corresponds to the same threshold.

    The first point is at threshold = max(scores) + epsilon (no predictions,
    precision = 1.0 by convention, recall = 0.0). The last point is at
    threshold = min(scores) (all predicted positive).
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot compute curve on empty data")

    positives = sum(1 for v in y_true if v == 1)
    negatives = len(y_true) - positives
    if positives == 0:
        raise ValueError("need at least one positive example")

    # Sort by score descending; ties grouped together.
    scored = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)

    precisions: list[float] = [1.0]  # threshold above all scores: no preds
    recalls: list[float] = [0.0]
    thresholds: list[float] = [float("inf")]

    tp = 0
    fp = 0
    prev_score = None

    for score, label in scored:
        # Only compute metrics when score changes (threshold = score).
        if score != prev_score:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / positives
            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(score)
            prev_score = score

        if label == 1:
            tp += 1
        else:
            fp += 1

    # Final point: all samples predicted positive.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / positives
    precisions.append(precision)
    recalls.append(recall)
    thresholds.append(float(scores[0]) if scores else 0.0)  # min score after sort

    return precisions, recalls, thresholds


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve.

    Uses trapezoidal integration over recall. Returns 0.0 if there are no
    positive examples.
    """
    precisions, recalls, _ = precision_recall_curve(list(y_true), list(scores))

    # Integrate precision over recall using trapezoidal rule.
    # Sort by recall ascending (should already be, but be safe).
    points = sorted(zip(recalls, precisions), key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(points)):
        r0, p0 = points[i - 1]
        r1, p1 = points[i]
        area += (r1 - r0) * (p0 + p1) / 2.0
    return area


def operating_point(
    y_true: Sequence[int], scores: Sequence[float], max_fpr: float = 0.05
) -> dict[str, float]:
    """Select an operating point with FPR <= max_fpr that maximizes recall.

    Sweeps all candidate thresholds and returns the one with the highest
    recall (detection rate) subject to the FPR constraint. If no threshold
    meets the constraint, returns the threshold with the lowest FPR.

    Returns a dict with threshold, precision, recall, f1, fpr, fnr, accuracy.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    best = None
    best_recall = -1.0
    best_fpr = float("inf")

    for threshold in sorted(set(scores), reverse=True):
        metrics = binary_metrics(list(y_true), list(scores), threshold=threshold)
        if metrics["false_positive_rate"] <= max_fpr:
            if metrics["recall"] >= best_recall:
                best_recall = metrics["recall"]
                best = {
                    "threshold": threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "false_negative_rate": metrics["false_negative_rate"],
                    "accuracy": metrics["accuracy"],
                }

    if best is None:
        # No threshold meets the FPR constraint; return the most restrictive.
        metrics = binary_metrics(list(y_true), list(scores), threshold=max(scores))
        best = {
            "threshold": max(scores),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
            "accuracy": metrics["accuracy"],
            "note": "no threshold meets FPR constraint",
        }

    return best


def threshold_sweep(
    y_true: Sequence[int], scores: Sequence[float]
) -> list[dict[str, float]]:
    """Evaluate metrics at every observed score threshold.

    Returns a list of dicts, one per unique score threshold, sorted by
    threshold descending. Each dict contains threshold, precision, recall,
    f1, fpr, fnr, accuracy, tp, fp, tn, fn.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    results: list[dict[str, float]] = []
    for threshold in sorted(set(scores), reverse=True):
        m = binary_metrics(list(y_true), list(scores), threshold=threshold)
        m["threshold"] = threshold
        results.append(m)
    return results
