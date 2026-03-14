"""Unit tests for tokenized Stage 1 outputs."""

from __future__ import annotations

from annomi_pipeline.data.tokenizer import build_tokenized_splits


def test_build_tokenized_splits_adds_tokens_and_token_counts() -> None:
    tokenized_splits, token_report = build_tokenized_splits(
        split_records={
            "train": [{"example_id": "a", "text": "hello world"}],
            "val": [{"example_id": "b", "text": "hello"}],
            "test": [],
        },
        tokenizer_type="whitespace",
    )

    assert tokenized_splits["train"][0]["tokens"] == ["hello", "world"]
    assert tokenized_splits["train"][0]["token_count"] == 2
    assert tokenized_splits["val"][0]["token_count"] == 1
    assert token_report["train"]["count"] == 1
    assert token_report["overall"]["max"] == 2
