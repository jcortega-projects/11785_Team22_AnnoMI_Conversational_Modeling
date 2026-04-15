"""Data loading and splitting utilities."""

from __future__ import annotations

from typing import Any


def load_split_records(split_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Load tokenized datasets."""
    pass


def extract_texts_and_labels(
    records: list[dict[str, Any]],
    label_field: str,
) -> tuple[list[str], list[str | None]]:
    """Extract texts and labels from records."""
    pass
