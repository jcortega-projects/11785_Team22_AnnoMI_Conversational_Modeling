"""CLI entry point for embedding generation.

Reads the processed JSONL splits produced by Stage 1 and writes
``{split}_embeddings.npy`` plus example-id alignment files. Validates that
embedding row counts match the corresponding JSONL records.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from annomi_pipeline.data.embedding import generate_embeddings
from annomi_pipeline.utils.config import load_yaml, resolve_path
from annomi_pipeline.utils.io import read_jsonl
from annomi_pipeline.utils.logging import configure_logging
from annomi_pipeline.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for processed splits.")
    parser.add_argument("--config", required=True, help="Path to embeddings YAML config.")
    return parser.parse_args()


def _resolve(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return resolve_path(path, project_dir)


def main() -> None:
    args = parse_args()
    configure_logging()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    set_global_seed(int(config.get("seed", 42)))

    project_dir = config_path.parent.parent
    paths = config["paths"]

    train_jsonl = _resolve(paths["train_jsonl"], project_dir)
    val_jsonl = _resolve(paths["val_jsonl"], project_dir)
    test_jsonl = _resolve(paths["test_jsonl"], project_dir)
    embeddings_dir = _resolve(paths["embeddings_dir"], project_dir)
    model_dir = _resolve(paths["model_dir"], project_dir)

    split_records = {
        "train": read_jsonl(train_jsonl),
        "val": read_jsonl(val_jsonl),
        "test": read_jsonl(test_jsonl),
    }
    LOGGER.info(
        "Loaded %s/%s/%s train/val/test records",
        len(split_records["train"]),
        len(split_records["val"]),
        len(split_records["test"]),
    )

    manifest = generate_embeddings(
        split_records=split_records,
        embedding_config=config["embedding"],
        embeddings_dir=embeddings_dir,
        model_dir=model_dir,
    )

    # Alignment validation
    import numpy as np
    for split in ("train", "val", "test"):
        matrix = np.load(embeddings_dir / f"{split}_embeddings.npy")
        n_records = len(split_records[split])
        if matrix.shape[0] != n_records:
            raise RuntimeError(
                f"Embedding/record mismatch for {split}: {matrix.shape[0]} vs {n_records}"
            )
        LOGGER.info("%s embeddings shape=%s (aligned)", split, tuple(matrix.shape))

    LOGGER.info("Embedding manifest: %s", manifest)


if __name__ == "__main__":
    main()
