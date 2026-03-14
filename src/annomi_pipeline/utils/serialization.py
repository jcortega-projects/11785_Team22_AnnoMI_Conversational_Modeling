"""Serialization-safe helpers and nested field access."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MISSING_VALUES = {None, "", "n/a", "N/A", "nan", "NaN"}


def get_by_path(payload: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path from a nested mapping."""

    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def is_missing(value: Any) -> bool:
    """Return True if the supplied value should be treated as missing."""

    return value in MISSING_VALUES
