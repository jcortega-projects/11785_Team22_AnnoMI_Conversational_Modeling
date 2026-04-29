"""Conversation chunking utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any

from annomi_pipeline.data.ingestion import Conversation, ConversationTurn


def _non_missing(values: list[str | None]) -> list[str]:
    """Filter null-like values while preserving strings."""

    return [
        value
        for value in values
        if value is not None and str(value).strip().lower() not in {"", "n/a", "nan"}
    ]


def _distribution(values: list[str | None]) -> dict[str, int]:
    """Build a deterministic frequency table."""

    counts = Counter(_non_missing(values))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _majority_label(counts: dict[str, int]) -> str | None:
    """Return the highest-frequency label with deterministic tie-breaking."""

    return next(iter(counts)) if counts else None


def _format_chunk_text(turns: list[ConversationTurn]) -> str:
    """Render a speaker-labelled chunk string."""

    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


def build_chunk_record(
    conversation: Conversation,
    turns: list[ConversationTurn],
    chunk_id: int,
) -> dict[str, Any]:
    """Create a JSON-serializable chunk example."""

    therapist_counts = _distribution(
        [turn.attributes.get("main_therapist_behaviour") for turn in turns]
    )
    client_counts = _distribution(
        [turn.attributes.get("client_talk_type") for turn in turns]
    )

    return {
        "example_id": f"{conversation.transcript_id}_{chunk_id}",
        "transcript_id": conversation.transcript_id,
        "chunk_id": chunk_id,
        "start_utterance": turns[0].utterance_id,
        "end_utterance": turns[-1].utterance_id,
        "text": _format_chunk_text(turns),
        "turn_count": len(turns),
        "topic": conversation.topic,
        "mi_quality": conversation.mi_quality,
        "metadata": {
            **conversation.transcript_metadata,
            "utterance_ids": [turn.utterance_id for turn in turns],
            "speaker_sequence": [turn.speaker for turn in turns],
            "timestamps": [turn.timestamp for turn in turns if turn.timestamp],
            "therapist_behavior_counts": therapist_counts,
            "therapist_behavior_mode": _majority_label(therapist_counts),
            "client_talk_counts": client_counts,
            "client_talk_mode": _majority_label(client_counts),
        },
    }


def chunk_conversation(
    conversation: Conversation,
    turns_per_chunk: int,
    overlap: int,
    min_turns: int,
    include_partial_final_chunk: bool,
) -> list[dict[str, Any]]:
    """Split a conversation into fixed-size sliding windows."""

    if turns_per_chunk <= 0:
        raise ValueError("turns_per_chunk must be positive")
    if overlap >= turns_per_chunk:
        raise ValueError("overlap must be smaller than turns_per_chunk")

    step = turns_per_chunk - overlap
    total_turns = len(conversation.turns)
    chunk_records: list[dict[str, Any]] = []
    chunk_id = 0
    start = 0

    while start < total_turns:
        end = min(start + turns_per_chunk, total_turns)
        window = conversation.turns[start:end]
        if len(window) < min_turns:
            break
        if len(window) < turns_per_chunk and not include_partial_final_chunk:
            break
        chunk_records.append(build_chunk_record(conversation, window, chunk_id))
        chunk_id += 1
        if end == total_turns:
            break
        start += step

    return chunk_records


def build_client_utterance_examples(
    conversation: Conversation,
    context_turns: int,
    allowed_labels: set[str],
    label_attribute: str = "client_talk_type",
    speaker_label: str = "client",
) -> list[dict[str, Any]]:
    """Build per-utterance examples targeting client_talk_type prediction.

    Each example corresponds to a single client utterance whose label is in
    ``allowed_labels``. When ``context_turns > 0`` the input text is the
    current client utterance prefixed by up to ``context_turns`` preceding
    turns (causal-only — never future turns). The target row is always the
    client utterance itself; therapist turns may appear in the context but
    never as targets.
    """

    if context_turns < 0:
        raise ValueError("context_turns must be >= 0")

    examples: list[dict[str, Any]] = []
    chunk_id = 0
    for index, turn in enumerate(conversation.turns):
        if str(turn.speaker).strip().lower() != speaker_label:
            continue
        raw_label = turn.attributes.get(label_attribute)
        if raw_label is None:
            continue
        label = str(raw_label).strip().lower()
        if label in {"", "n/a", "nan"}:
            continue
        if label not in allowed_labels:
            continue

        start = max(0, index - context_turns)
        window = conversation.turns[start : index + 1]
        text = (
            turn.text.strip()
            if context_turns == 0
            else _format_chunk_text(window)
        )
        examples.append(
            {
                "example_id": f"{conversation.transcript_id}_{turn.utterance_id}",
                "transcript_id": conversation.transcript_id,
                "chunk_id": chunk_id,
                "utterance_id": turn.utterance_id,
                "start_utterance": window[0].utterance_id,
                "end_utterance": turn.utterance_id,
                "text": text,
                "turn_count": len(window),
                "topic": conversation.topic,
                "mi_quality": conversation.mi_quality,
                "metadata": {
                    **conversation.transcript_metadata,
                    "speaker": turn.speaker,
                    "context_turns": context_turns,
                    "client_talk_type": label,
                    "utterance_ids": [t.utterance_id for t in window],
                    "speaker_sequence": [t.speaker for t in window],
                },
            }
        )
        chunk_id += 1
    return examples


def build_client_utterance_splits(
    conversations: list[Conversation],
    split_ids: dict[str, list[int | str]],
    context_turns: int,
    allowed_labels: set[str],
    label_attribute: str = "client_talk_type",
) -> dict[str, list[dict[str, Any]]]:
    """Apply :func:`build_client_utterance_examples` per split."""

    transcript_to_split = {
        transcript_id: split_name
        for split_name, transcript_ids in split_ids.items()
        for transcript_id in transcript_ids
    }
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for conversation in conversations:
        split_name = transcript_to_split.get(conversation.transcript_id)
        if split_name is None:
            continue
        split_records[split_name].extend(
            build_client_utterance_examples(
                conversation=conversation,
                context_turns=context_turns,
                allowed_labels=allowed_labels,
                label_attribute=label_attribute,
            )
        )
    return split_records


def chunk_splits(
    conversations: list[Conversation],
    split_ids: dict[str, list[int | str]],
    chunk_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Chunk conversations and assign them to the configured data splits."""

    transcript_to_split = {
        transcript_id: split_name
        for split_name, transcript_ids in split_ids.items()
        for transcript_id in transcript_ids
    }
    split_records = {"train": [], "val": [], "test": []}
    for conversation in conversations:
        split_name = transcript_to_split[conversation.transcript_id]
        split_records[split_name].extend(
            chunk_conversation(
                conversation=conversation,
                turns_per_chunk=int(chunk_config["turns_per_chunk"]),
                overlap=int(chunk_config["overlap"]),
                min_turns=int(chunk_config.get("min_turns", 1)),
                include_partial_final_chunk=bool(chunk_config.get("include_partial_final_chunk", True)),
            )
        )
    return split_records

