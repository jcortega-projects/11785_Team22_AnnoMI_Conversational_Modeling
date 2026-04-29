"""Helpers for loading OpenAI credentials from safe locations."""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_ENV_PATH_VARS = ("ANNOMI_ENV_FILE", "OPENAI_ENV_FILE")
_DEFAULT_EXTERNAL_ENV_PATHS = (
    Path.home() / ".config" / "annomi-mlp" / ".env",
    Path.home() / ".annomi-mlp.env",
)


def _candidate_env_paths(search_start: Path, explicit_env_file: str | Path | None) -> list[Path]:
    """Build ordered candidate dotenv paths without duplicates."""

    candidates: list[Path] = []
    if explicit_env_file:
        candidates.append(Path(explicit_env_file).expanduser())
    for env_var in _ENV_PATH_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(_DEFAULT_EXTERNAL_ENV_PATHS)

    base_dir = search_start.expanduser().resolve()
    if base_dir.is_file():
        base_dir = base_dir.parent
    candidates.extend(parent / ".env" for parent in (base_dir, *base_dir.parents))

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        unique_candidates.append(resolved)
        seen.add(resolved)
    return unique_candidates


def load_openai_env(
    *,
    search_start: str | Path,
    explicit_env_file: str | Path | None = None,
) -> Path | None:
    """Load `OPENAI_API_KEY` from an external dotenv file when available.

    Preference order:
    1. Existing process environment (`OPENAI_API_KEY`)
    2. An explicit `--env-file` path
    3. `ANNOMI_ENV_FILE` / `OPENAI_ENV_FILE`
    4. Common external secret paths in the user's home directory
    5. Backward-compatible fallback: a `.env` found while walking up from the
       calling script location
    """

    if os.environ.get("OPENAI_API_KEY"):
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        if explicit_env_file or any(os.environ.get(name) for name in _ENV_PATH_VARS):
            LOGGER.warning(
                "python-dotenv is not installed, so dotenv files cannot be loaded automatically."
            )
        return None

    for candidate in _candidate_env_paths(Path(search_start), explicit_env_file):
        if not candidate.exists():
            continue
        load_dotenv(candidate, override=False)
        if os.environ.get("OPENAI_API_KEY"):
            LOGGER.info("Loaded OPENAI_API_KEY from %s", candidate)
            return candidate

    return None
