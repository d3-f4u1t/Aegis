"""Training-side utilities: loop, evaluation, metrics, and (later) preprocessing."""

from .metrics import binary_confusion, binary_metrics, detection_rate_at_fpr
from .train import History, batch_loss, evaluate, train

__all__ = [
    "History",
    "batch_loss",
    "binary_confusion",
    "binary_metrics",
    "detection_rate_at_fpr",
    "evaluate",
    "train",
]
