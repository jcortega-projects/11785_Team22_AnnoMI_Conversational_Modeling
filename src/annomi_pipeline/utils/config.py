"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to a base directory if needed."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (Path(base_dir) / path).resolve()

