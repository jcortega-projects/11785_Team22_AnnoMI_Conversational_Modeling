"""Metrics computation."""

from __future__ import annotations

from typing import Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

def compute_standard_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    """Compute per-class metrics."""
    labels = ["high", "low"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "high": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "low": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }

def compute_classification_report(y_true: list[str], y_pred: list[str]) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=["high", "low"],
        output_dict=True,
        zero_division=0,
    )



class MetricsCompute:
    """Unified metrics interface."""
    
    @staticmethod
    def all_metrics(
        y_true: list[str],
        y_pred: list[str],
        texts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute all metrics."""
        cm = confusion_matrix(y_true, y_pred, labels=["high", "low"])
        return {
            "standard": compute_standard_metrics(y_true, y_pred),
            "per_class": compute_per_class_metrics(y_true, y_pred),
            "confusion_matrix": cm.tolist(),
            "classification_report": compute_classification_report(y_true, y_pred),
            "num_examples": len(y_true),
        }

