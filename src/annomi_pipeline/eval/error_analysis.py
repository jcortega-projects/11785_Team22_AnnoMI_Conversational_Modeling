"""Error analysis utilities."""

from __future__ import annotations

from typing import Any


def collect_misclassified_examples(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    example_ids: list[str] | None = None,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Collect misclassified examples."""
    pass


def categorize_errors(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, list[str]]:
    """Categorize errors by label pairs."""
    pass


class ErrorAnalysisReport:
    """Error analysis report."""
    
    def __init__(self):
        """Initialize report."""
        pass
    
    def add_misclassified_examples(
        self,
        texts: list[str],
        y_true: list[str],
        y_pred: list[str],
        **kwargs: Any,
    ) -> None:
        """Add misclassified examples."""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        pass
