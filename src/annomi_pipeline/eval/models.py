"""Model abstraction layer."""

from __future__ import annotations

from typing import Any


class SklearnClassifier:
    """Wrapper for sklearn classifiers."""
    
    def __init__(self, estimator: Any, vectorizer: Any = None):
        """Initialize with estimator and optional vectorizer."""
        pass
    
    def fit(self, X: list[str], y: list[str]) -> None:
        """Train the model."""
        pass
    
    def predict(self, X: list[str]) -> list[str]:
        """Make predictions."""
        pass


class DummyClassifier:
    """Dummy baseline classifier."""
    
    def fit(self, X: list[str], y: list[str]) -> None:
        """Learn most frequent class."""
        pass
    
    def predict(self, X: list[str]) -> list[str]:
        """Always predict most frequent class."""
        pass
