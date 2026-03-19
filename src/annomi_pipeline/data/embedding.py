"""Embedding generation for chunked conversation text."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from annomi_pipeline.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def _extract_texts_and_ids(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Collect ordered texts and example identifiers from chunk records."""

    texts = [str(record["text"]) for record in records]
    example_ids = [str(record["example_id"]) for record in records]
    return texts, example_ids


def _save_embeddings(
    embeddings_dir: Path,
    split_name: str,
    matrix: np.ndarray,
    example_ids: list[str],
) -> None:
    """Persist an embedding matrix and its aligned example ids."""

    np.save(embeddings_dir / f"{split_name}_embeddings.npy", matrix)
    write_json(embeddings_dir / f"{split_name}_example_ids.json", example_ids)


def generate_embeddings(
    split_records: dict[str, list[dict[str, Any]]],
    embedding_config: dict[str, Any],
    embeddings_dir: str | Path,
    model_dir: str | Path,
) -> dict[str, Any]:
    """Generate and save embeddings for the configured backend."""

    backend = embedding_config["type"]
    embeddings_path = ensure_dir(embeddings_dir)
    model_path = ensure_dir(model_dir)
    LOGGER.info("Generating %s embeddings", backend)

    train_texts, train_ids = _extract_texts_and_ids(split_records["train"])
    val_texts, val_ids = _extract_texts_and_ids(split_records["val"])
    test_texts, test_ids = _extract_texts_and_ids(split_records["test"])

    if backend == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=int(embedding_config.get("max_features", 2048)),
            lowercase=bool(embedding_config.get("lowercase", True)),
            ngram_range=tuple(embedding_config.get("ngram_range", [1, 1])),
        )
        train_matrix = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
        val_matrix = vectorizer.transform(val_texts).toarray().astype(np.float32)
        test_matrix = vectorizer.transform(test_texts).toarray().astype(np.float32)
        with (model_path / "tfidf_vectorizer.pkl").open("wb") as handle:
            pickle.dump(vectorizer, handle)
    elif backend == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        model_name = embedding_config["sentence_transformer_model"]
        encoder = SentenceTransformer(model_name)
        batch_size = int(embedding_config.get("batch_size", 32))
        train_matrix = encoder.encode(
            train_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        val_matrix = encoder.encode(
            val_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        test_matrix = encoder.encode(
            test_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported embedding type: {backend}")

    _save_embeddings(embeddings_path, "train", train_matrix, train_ids)
    _save_embeddings(embeddings_path, "val", val_matrix, val_ids)
    _save_embeddings(embeddings_path, "test", test_matrix, test_ids)

    manifest = {
        "backend": backend,
        "train_shape": list(train_matrix.shape),
        "val_shape": list(val_matrix.shape),
        "test_shape": list(test_matrix.shape),
    }
    write_json(embeddings_path / "embedding_manifest.json", manifest)
    return manifest

