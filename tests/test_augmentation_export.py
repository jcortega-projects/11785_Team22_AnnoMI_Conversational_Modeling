"""Tests for augmentation export helpers."""

from __future__ import annotations

from annomi_pipeline.data.augmentation_export import build_augmentation_export_records
from annomi_pipeline.data.ingestion import Conversation, ConversationTurn


def test_build_augmentation_export_records_preserves_processed_fields_and_context() -> None:
    conversations = [
        Conversation(
            transcript_id=7,
            topic="stress",
            mi_quality="high",
            transcript_metadata={"video_title": "Example", "video_url": "https://example.test"},
            turns=[
                ConversationTurn(
                    utterance_id=1,
                    speaker="therapist",
                    text="How has this week been?",
                    timestamp=None,
                    attributes={"main_therapist_behaviour": "question", "client_talk_type": None},
                ),
                ConversationTurn(
                    utterance_id=2,
                    speaker="client",
                    text="Pretty rough.",
                    timestamp=None,
                    attributes={"main_therapist_behaviour": None, "client_talk_type": "neutral"},
                ),
                ConversationTurn(
                    utterance_id=3,
                    speaker="therapist",
                    text="What would you like to change first?",
                    timestamp=None,
                    attributes={"main_therapist_behaviour": "question", "client_talk_type": None},
                ),
                ConversationTurn(
                    utterance_id=4,
                    speaker="client",
                    text="I want to stop drinking every night.",
                    timestamp=None,
                    attributes={"main_therapist_behaviour": None, "client_talk_type": "change"},
                ),
            ],
        )
    ]
    records = [
        {
            "example_id": "7_4",
            "transcript_id": 7,
            "chunk_id": 1,
            "utterance_id": 4,
            "start_utterance": 4,
            "end_utterance": 4,
            "text": "I want to stop drinking every night.",
            "turn_count": 1,
            "topic": "stress",
            "mi_quality": "high",
            "metadata": {
                "video_title": "Example",
                "video_url": "https://example.test",
                "speaker": "client",
                "context_turns": 0,
                "client_talk_type": "change",
                "utterance_ids": [4],
                "speaker_sequence": ["client"],
            },
        }
    ]

    export_rows = build_augmentation_export_records(
        records=records,
        conversations=conversations,
        source_dataset="AnnoMI/AnnoMI-simple.csv",
    )

    assert len(export_rows) == 1
    row = export_rows[0]
    assert row["example_id"] == "7_4"
    assert row["original_example_id"] == "7_4"
    assert row["split"] == "train"
    assert row["source_type"] == "real"
    assert row["augmentation_method"] is None
    assert row["client_text"] == "I want to stop drinking every night."
    assert row["label"] == "change"
    assert row["stored_context_turns"] == 0
    assert row["text"] == "I want to stop drinking every night."
    assert row["context"] == (
        "therapist: How has this week been?\n"
        "client: Pretty rough.\n"
        "therapist: What would you like to change first?"
    )
    assert len(row["prior_turns"]) == 3
    assert row["prior_turns"][0]["speaker"] == "therapist"
    assert row["metadata"]["client_talk_type"] == "change"
