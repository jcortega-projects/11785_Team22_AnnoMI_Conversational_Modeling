"""Fairness evaluation script."""

from __future__ import annotations

import argparse
import logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Fairness evaluation.")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--output-dir", help="Output directory")
    return parser.parse_args()


def main() -> None:
    """Run fairness evaluation."""
    args = parse_args()
    pass


if __name__ == "__main__":
    main()
