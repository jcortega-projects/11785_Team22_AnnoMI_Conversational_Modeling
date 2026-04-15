"""Train/validation/test splitting strategies."""

from __future__ import annotations


def stratified_split(
    X: list[str],
    y: list[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Stratified train/val/test split."""
    pass


def group_stratified_split(
    X: list[str],
    y: list[str],
    groups: list[str | int],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Group-stratified split preventing leakage."""
    pass
