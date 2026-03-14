"""Unit tests for Stage 1 data ingestion helpers."""

from __future__ import annotations

import pandas as pd

from annomi_pipeline.data.ingestion import build_conversations, validate_required_columns


def test_build_conversations_uses_configured_metadata_fields() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "transcript_id": 7,
                "utterance_id": 0,
                "interlocutor": "therapist",
                "utterance_text": "How does that feel?",
                "topic": "sleep",
                "mi_quality": "high",
                "timestamp": "00:00:01",
                "video_title": "demo video",
                "main_therapist_behaviour": "question",
                "client_talk_type": None,
            },
            {
                "transcript_id": 7,
                "utterance_id": 1,
                "interlocutor": "client",
                "utterance_text": "It feels hard.",
                "topic": "sleep",
                "mi_quality": "high",
                "timestamp": "00:00:03",
                "video_title": "demo video",
                "main_therapist_behaviour": None,
                "client_talk_type": "change",
            },
        ]
    )

    validate_required_columns(
        dataframe,
        [
            "transcript_id",
            "utterance_id",
            "interlocutor",
            "utterance_text",
            "topic",
            "mi_quality",
            "timestamp",
            "video_title",
            "main_therapist_behaviour",
            "client_talk_type",
        ],
    )
    conversations = build_conversations(
        dataframe,
        {
            "transcript_id_column": "transcript_id",
            "utterance_id_column": "utterance_id",
            "speaker_column": "interlocutor",
            "text_column": "utterance_text",
            "topic_column": "topic",
            "mi_quality_column": "mi_quality",
            "timestamp_column": "timestamp",
            "transcript_metadata_fields": [
                "video_title",
            ],
            "turn_attribute_fields": [
                "main_therapist_behaviour",
                "client_talk_type",
            ],
        },
    )

    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation.transcript_metadata["video_title"] == "demo video"
    assert conversation.turns[0].timestamp == "00:00:01"
    assert conversation.turns[0].attributes["main_therapist_behaviour"] == "question"
    assert conversation.turns[1].attributes["client_talk_type"] == "change"
