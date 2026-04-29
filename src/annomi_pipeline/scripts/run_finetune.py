"""CLI entrypoint for end-to-end transformer fine-tuning.

Usage
-----
    python -m annomi_pipeline.scripts.run_finetune \\
        --config configs/train_config_roberta_finetune.yaml
"""

from __future__ import annotations

import argparse

from annomi_pipeline.training.train_finetune import train_and_evaluate_finetune
from annomi_pipeline.utils.config import load_yaml
from annomi_pipeline.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer end-to-end for AnnoMI.")
    parser.add_argument("--config", required=True, help="Path to the fine-tuning YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_yaml(args.config)
    train_and_evaluate_finetune(config)


if __name__ == "__main__":
    main()
