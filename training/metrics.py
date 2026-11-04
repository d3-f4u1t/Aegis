"""Evaluation metrics.

Accuracy alone is misleading for traffic data: benign traffic dominates, so a
model that never predicts "attack" can score 99%+ accuracy while being useless.
These helpers therefore report the full confusion matrix and the metrics that
matter operationally (recall at a chosen false-positive rate).
"""

from __future__ import annotations

from typing import Sequence


def binary_confusion(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = fp = tn = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            raise ValueError("labels must be 0 or 1")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def binary_metrics(
    y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    """Threshold scores, then compute the standard binary metrics.

    ``scores`` are P(class=1) values, not logits.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    predicted = [1 if s >= threshold else 0 for s in scores]
    cm = binary_confusion(list(y_true), predicted)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, len(y_true))
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def detection_rate_at_fpr(
    y_true: Sequence[int], scores: Sequence[float], max_fpr: float
) -> tuple[float, float]:
    """Best detection rate (recall) achievable with FPR <= max_fpr.

    Sweeps every observed score as a candidate threshold and returns
    ``(detection_rate, threshold)``. This is the metric an operator actually
    chooses an operating point with.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    positives = sum(1 for v in y_true if v == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("need at least one positive and one negative example")

    best_detection = 0.0
    best_threshold = 1.0
    for candidate in sorted(set(scores), reverse=True):
        cm = binary_confusion(
            list(y_true), [1 if s >= candidate else 0 for s in scores]
        )
        fpr = cm["fp"] / negatives
        if fpr <= max_fpr:
            detection = cm["tp"] / positives
            if detection >= best_detection:
                best_detection = detection
                best_threshold = candidate
    return best_detection, best_threshold
