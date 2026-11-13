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

    Sweeps every observed unique score as a candidate threshold (from high to low)
    and returns ``(precisions, recalls, thresholds)`` where each index
    corresponds to evaluating positive predictions at ``score >= threshold``.

    The initial point is at threshold = inf (no positive predictions,
    precision = 1.0 by standard convention, recall = 0.0). Subsequent points
    correspond to the unique observed scores in descending order.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot compute curve on empty data")

    positives = sum(1 for v in y_true if v == 1)
    if positives == 0:
        raise ValueError("need at least one positive example")

    # Sort samples by score descending; group by identical scores.
    scored = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)

    precisions: list[float] = [1.0]  # threshold above all scores: no positive predictions
    recalls: list[float] = [0.0]
    thresholds: list[float] = [float("inf")]

    tp = 0
    fp = 0
    i = 0
    n = len(scored)

    while i < n:
        current_score = scored[i][0]
        # Accumulate all samples with the same score threshold
        while i < n and scored[i][0] == current_score:
            if scored[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / positives
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(current_score)

    return precisions, recalls, thresholds


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve using the trapezoidal rule.

    Note on calculation:
    Trapezoidal integration linearly connects adjacent (recall, precision)
    points with straight line segments:
        area += (r1 - r0) * (p0 + p1) / 2.0

    Important methodological note:
    Because precision does not vary linearly with recall between empirical
    operating points, trapezoidal integration systematically overestimates
    the true area under the PR curve (Davis & Goadrich, 2006). For example,
    on an uninformative classifier with a 50% positive rate, trapezoidal PR-AUC
    evaluates to 0.75 rather than 0.50. For an un-interpolated, step-wise metric,
    use :func:`average_precision`.
    """
    precisions, recalls, _ = precision_recall_curve(list(y_true), list(scores))

    # Integrate precision over recall using trapezoidal rule.
    points = sorted(zip(recalls, precisions), key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(points)):
        r0, p0 = points[i - 1]
        r1, p1 = points[i]
        area += (r1 - r0) * (p0 + p1) / 2.0
    return area


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Compute Average Precision (AP) summarizing the precision-recall curve.

    Calculated as the weighted mean of precisions achieved at each threshold,
    weighted by the increase in recall from the previous threshold:

        AP = sum_n (R_n - R_{n-1}) * P_n

    This avoids the linear interpolation bias of the trapezoidal rule and
    accurately reflects the expected precision across recall levels. For an
    uninformative random model, AP converges to the base rate of positive examples.
    """
    positives = sum(1 for v in y_true if v == 1)
    if positives == 0:
        return 0.0

    precisions, recalls, _ = precision_recall_curve(list(y_true), list(scores))
    ap = 0.0
    for i in range(1, len(recalls)):
        delta_r = recalls[i] - recalls[i - 1]
        if delta_r > 0:
            ap += delta_r * precisions[i]
    return ap


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
