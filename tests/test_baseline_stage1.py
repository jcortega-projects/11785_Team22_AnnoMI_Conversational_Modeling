"""Unit tests for the Stage 1 baseline sweep."""

from __future__ import annotations

from annomi_pipeline.data.ingestion import Conversation, ConversationTurn
from annomi_pipeline.stage1.baseline import run_baseline_experiments


def _conversation(transcript_id: int, therapist_text: str, therapist_label: str) -> Conversation:
    return Conversation(
        transcript_id=transcript_id,
        topic="demo",
        mi_quality="high",
        transcript_metadata={"video_title": "demo"},
        turns=[
            ConversationTurn(
                utterance_id=0,
                speaker="therapist",
                text=therapist_text,
                timestamp=None,
                attributes={"main_therapist_behaviour": therapist_label, "client_talk_type": None},
            ),
            ConversationTurn(
                utterance_id=1,
                speaker="client",
                text=f"client response {transcript_id}",
                timestamp=None,
                attributes={"main_therapist_behaviour": None, "client_talk_type": "neutral"},
            ),
        ],
    )


def test_run_baseline_experiments_writes_plot_and_results(tmp_path) -> None:
    conversations = [
        _conversation(1, "ask ask goals", "question"),
        _conversation(2, "reflect reflect feelings", "reflection"),
        _conversation(3, "ask ask options", "question"),
        _conversation(4, "reflect reflect summary", "reflection"),
        _conversation(5, "ask ask plan", "question"),
        _conversation(6, "reflect reflect empathy", "reflection"),
    ]
    split_ids = {"train": [1, 2], "val": [3, 4], "test": [5, 6]}

    payload = run_baseline_experiments(
        conversations=conversations,
        split_ids=split_ids,
        base_chunk_config={
            "turns_per_chunk": 2,
            "overlap": 1,
            "min_turns": 2,
            "include_partial_final_chunk": True,
        },
        baseline_config={
            "label_field": "metadata.therapist_behavior_mode",
            "scoring_split": "val",
            "chunk_window_sizes": [2],
            "vocab_sizes": [16],
            "ngram_range": [1, 1],
            "class_weight": "balanced",
            "max_iter": 200,
            "random_state": 0,
        },
        output_dir=tmp_path,
    )

    assert len(payload["results"]) == 1
    assert payload["best_result"]["val_macro_f1"] == 1.0
    assert payload["best_result"]["test_macro_f1"] == 1.0
    assert (tmp_path / "baseline_results.json").exists()
    assert (tmp_path / "baseline_results.png").exists()
