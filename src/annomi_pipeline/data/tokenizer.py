"""Tokenization utilities for chunked conversation text."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any
from typing import Protocol

import numpy as np


class TextTokenizer(Protocol):
    """Tokenizer protocol used for token statistics."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a string into a list of tokens."""


@dataclass(slots=True)
class WhitespaceTokenizer:
    """Whitespace-based tokenizer."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class HuggingFaceTokenizer:
    """Thin wrapper around a Hugging Face tokenizer."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)


def build_tokenizer(tokenizer_type: str, hf_model_name: str | None = None) -> TextTokenizer:
    """Instantiate the configured tokenizer backend."""

    if tokenizer_type == "whitespace":
        return WhitespaceTokenizer()
    if tokenizer_type == "huggingface":
        if not hf_model_name:
            raise ValueError("hf_model_name must be provided for Hugging Face tokenization")
        return HuggingFaceTokenizer(hf_model_name)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def compute_token_statistics(lengths: list[int]) -> dict[str, float | int]:
    """Aggregate descriptive token statistics."""

    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}

    array = np.asarray(lengths)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "median": float(median(lengths)),
        "p95": float(np.percentile(array, 95)),
    }


def build_token_report(
    split_records: dict[str, list[dict[str, object]]],
    tokenizer_type: str,
    hf_model_name: str | None = None,
) -> dict[str, dict[str, float | int] | str]:
    """Compute token statistics for every dataset split."""

    tokenizer = build_tokenizer(tokenizer_type=tokenizer_type, hf_model_name=hf_model_name)
    split_lengths: dict[str, list[int]] = {}
    for split_name, records in split_records.items():
        split_lengths[split_name] = [len(tokenizer.tokenize(str(record["text"]))) for record in records]

    combined = [length for lengths in split_lengths.values() for length in lengths]
    return {
        "tokenizer_type": tokenizer_type,
        "hf_model_name": hf_model_name or "",
        "overall": compute_token_statistics(combined),
        **{split_name: compute_token_statistics(lengths) for split_name, lengths in split_lengths.items()},
    }


def build_tokenized_splits(
    split_records: dict[str, list[dict[str, Any]]],
    tokenizer_type: str,
    hf_model_name: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, float | int] | str]]:
    """Tokenize each record and return tokenized splits plus aggregate statistics."""

    tokenizer = build_tokenizer(tokenizer_type=tokenizer_type, hf_model_name=hf_model_name)
    tokenized_splits: dict[str, list[dict[str, Any]]] = {}
    split_lengths: dict[str, list[int]] = {}

    for split_name, records in split_records.items():
        tokenized_records: list[dict[str, Any]] = []
        lengths: list[int] = []
        for record in records:
            tokens = tokenizer.tokenize(str(record["text"]))
            tokenized_record = dict(record)
            tokenized_record["tokens"] = tokens
            tokenized_record["token_count"] = len(tokens)
            tokenized_records.append(tokenized_record)
            lengths.append(len(tokens))
        tokenized_splits[split_name] = tokenized_records
        split_lengths[split_name] = lengths

    combined = [length for lengths in split_lengths.values() for length in lengths]
    token_report = {
        "tokenizer_type": tokenizer_type,
        "hf_model_name": hf_model_name or "",
        "overall": compute_token_statistics(combined),
        **{split_name: compute_token_statistics(lengths) for split_name, lengths in split_lengths.items()},
    }
    return tokenized_splits, token_report
