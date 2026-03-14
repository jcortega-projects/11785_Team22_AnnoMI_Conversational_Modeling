"""Unit tests for conversation chunking."""

from __future__ import annotations

from annomi_pipeline.data.chunking import chunk_conversation
from annomi_pipeline.data.ingestion import Conversation, ConversationTurn


def test_chunk_conversation_sliding_window() -> None:
    conversation = Conversation(
        transcript_id=7,
        topic="sleep",
        mi_quality="high",
        transcript_metadata={"video_title": "demo"},
        turns=[
            ConversationTurn(utterance_id=0, speaker="therapist", text="How are you?", timestamp=None, attributes={}),
            ConversationTurn(utterance_id=1, speaker="client", text="Tired.", timestamp=None, attributes={}),
            ConversationTurn(utterance_id=2, speaker="therapist", text="Tell me more.", timestamp=None, attributes={}),
            ConversationTurn(utterance_id=3, speaker="client", text="Work has been rough.", timestamp=None, attributes={}),
            ConversationTurn(utterance_id=4, speaker="therapist", text="That sounds hard.", timestamp=None, attributes={}),
        ],
    )

    chunks = chunk_conversation(
        conversation=conversation,
        turns_per_chunk=3,
        overlap=1,
        min_turns=2,
        include_partial_final_chunk=True,
    )

    assert len(chunks) == 2
    assert chunks[0]["example_id"] == "7_0"
    assert chunks[0]["start_utterance"] == 0
    assert chunks[0]["end_utterance"] == 2
    assert "therapist: How are you?" in chunks[0]["text"]
    assert chunks[1]["start_utterance"] == 2
    assert chunks[1]["end_utterance"] == 4

