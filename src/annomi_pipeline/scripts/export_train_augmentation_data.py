"""Export real training examples into augmentation-ready JSONL.

Input source:
- `configs/data_config.yaml` for the raw dataset path and processed directory
- `data/processed/train.jsonl` produced by Stage 1

Output path:
- defaults to `data/outputs/augmentation/train_real_examples_for_augmentation.jsonl`

Schema:
- keeps the current processed `text` field and `metadata.client_talk_type`
  label path so the file stays close to the existing training schema
- adds `client_text`, `context`, `prior_turns`, and provenance placeholders
  (`source_type`, `original_example_id`, `augmentation_method`, etc.)

Why JSONL:
- the current pipeline already uses JSONL
- nested provenance/context fields fit naturally without lossy flattening
- accepted synthetic examples can later be merged back into training-only data
  without touching validation/test and without discarding metadata
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from annomi_pipeline.data.augmentation_export import build_augmentation_export_records
from annomi_pipeline.data.ingestion import build_conversations, load_annomi_dataframe
from annomi_pipeline.utils.config import load_yaml, resolve_path
from annomi_pipeline.utils.io import read_jsonl, write_jsonl
from annomi_pipeline.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Export Stage 1 training examples into an augmentation-ready JSONL file."
    )
    parser.add_argument("--config", required=True, help="Path to the data YAML configuration.")
    parser.add_argument(
        "--output",
        default="data/outputs/augmentation/train_real_examples_for_augmentation.jsonl",
        help="Destination JSONL export path.",
    )
    return parser.parse_args()


def _resolve_path(path_value: str, config_path: Path) -> Path:
    """Resolve config-relative paths in the same way as Stage 1."""

    config_dir = config_path.parent
    project_dir = config_dir.parent
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    project_relative = resolve_path(path, project_dir)
    config_relative = resolve_path(path, config_dir)
    if project_relative.exists() or not config_relative.exists():
        return project_relative
    return config_relative


def main() -> None:
    """Build the training-only augmentation export from current pipeline artifacts."""

    args = parse_args()
    configure_logging()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)

    paths = config["paths"]
    data_config = config["data"]
    source_csv = _resolve_path(paths["source_csv"], config_path)
    processed_dir = _resolve_path(paths["processed_dir"], config_path)
    train_jsonl = processed_dir / "train.jsonl"
    output_path = _resolve_path(args.output, config_path)

    train_records = read_jsonl(train_jsonl)
    dataframe = load_annomi_dataframe(
        csv_path=source_csv,
        sort_columns=[
            data_config["transcript_id_column"],
            data_config["utterance_id_column"],
        ],
    )
    conversations = build_conversations(dataframe, data_config)

    export_rows = build_augmentation_export_records(
        records=train_records,
        conversations=conversations,
        source_dataset=str(source_csv),
        split_name="train",
    )
    write_jsonl(output_path, export_rows)

    LOGGER.info("Read %s training records from %s", len(train_records), train_jsonl)
    LOGGER.info("Wrote %s augmentation export rows to %s", len(export_rows), output_path)
    if export_rows:
        LOGGER.info("Export schema keys: %s", sorted(export_rows[0].keys()))


if __name__ == "__main__":
    main()
