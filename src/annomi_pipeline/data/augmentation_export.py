"""Build augmentation-seed exports from processed training examples.

The export keeps the existing processed-record fields that the current
training pipeline already consumes (`text` plus `metadata.client_talk_type`)
and adds provenance/context fields that make synthetic generation, QA, and
later train-only merging straightforward.
"""

from __future__ import annotations

from typing import Any

from annomi_pipeline.data.ingestion import Conversation, ConversationTurn


def _format_turn(turn: ConversationTurn) -> str:
    """Render a turn in the same speaker-labelled style used elsewhere."""

    return f"{turn.speaker}: {turn.text}"


def _serialize_turn(turn: ConversationTurn) -> dict[str, Any]:
    """Convert a conversation turn into a JSON-serializable dictionary."""

    return {
        "utterance_id": turn.utterance_id,
        "speaker": turn.speaker,
        "text": turn.text,
        "timestamp": turn.timestamp,
        "attributes": dict(turn.attributes),
    }


def build_augmentation_export_records(
    records: list[dict[str, Any]],
    conversations: list[Conversation],
    source_dataset: str,
    split_name: str = "train",
) -> list[dict[str, Any]]:
    """Expand processed records into augmentation-ready export rows.

    Each row is still compatible with the existing training pipeline because it
    preserves the `text` field and the original `metadata.client_talk_type`
    label path, while also exposing the raw client utterance and full prior
    causal context for downstream synthetic generation.
    """

    conversation_by_id = {conversation.transcript_id: conversation for conversation in conversations}
    turn_index_by_transcript = {
        conversation.transcript_id: {
            turn.utterance_id: index for index, turn in enumerate(conversation.turns)
        }
        for conversation in conversations
    }

    export_rows: list[dict[str, Any]] = []
    for record in records:
        transcript_id = record["transcript_id"]
        utterance_id = record["utterance_id"]

        if transcript_id not in conversation_by_id:
            raise KeyError(f"Transcript {transcript_id!r} not found in reconstructed conversations.")
        if utterance_id not in turn_index_by_transcript[transcript_id]:
            raise KeyError(
                f"Utterance {utterance_id!r} not found in transcript {transcript_id!r}."
            )

        conversation = conversation_by_id[transcript_id]
        turn_index = turn_index_by_transcript[transcript_id][utterance_id]
        target_turn = conversation.turns[turn_index]
        prior_turns = conversation.turns[:turn_index]

        metadata = dict(record.get("metadata", {}))
        label = str(metadata.get("client_talk_type", "")).strip().lower()
        speaker = str(metadata.get("speaker", target_turn.speaker)).strip().lower()

        if speaker != "client":
            raise ValueError(
                f"Processed record {record.get('example_id')} is not a client example: {speaker!r}"
            )
        if target_turn.speaker.strip().lower() != "client":
            raise ValueError(
                "Target turn speaker mismatch for "
                f"{record.get('example_id')}: expected client, found {target_turn.speaker!r}"
            )

        raw_label = str(target_turn.attributes.get("client_talk_type", "")).strip().lower()
        if raw_label and raw_label != label:
            raise ValueError(
                "Target label mismatch for "
                f"{record.get('example_id')}: processed={label!r} raw={raw_label!r}"
            )

        export_rows.append(
            {
                "example_id": record["example_id"],
                "original_example_id": record["example_id"],
                "synthetic_candidate_id": None,
                "split": split_name,
                "source_type": "real",
                "augmentation_method": None,
                "augmentation_source": None,
                "source_dataset": source_dataset,
                "transcript_id": transcript_id,
                "utterance_id": utterance_id,
                "chunk_id": record.get("chunk_id"),
                "start_utterance": record.get("start_utterance"),
                "end_utterance": record.get("end_utterance"),
                "text": record.get("text", ""),
                "client_text": target_turn.text,
                "label": label,
                "turn_count": record.get("turn_count"),
                "topic": record.get("topic"),
                "mi_quality": record.get("mi_quality"),
                "stored_context_turns": metadata.get("context_turns"),
                "context": "\n".join(_format_turn(turn) for turn in prior_turns),
                "prior_turns": [_serialize_turn(turn) for turn in prior_turns],
                "metadata": metadata,
            }
        )

    return export_rows
