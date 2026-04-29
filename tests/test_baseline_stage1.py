"""Unit tests for the Stage 1 baseline sweep (client_talk_type task)."""

from __future__ import annotations

from annomi_pipeline.data.ingestion import Conversation, ConversationTurn
from annomi_pipeline.stage1.baseline import run_baseline_experiments


def _conversation(transcript_id: int, client_text: str, client_label: str) -> Conversation:
    return Conversation(
        transcript_id=transcript_id,
        topic="demo",
        mi_quality="high",
        transcript_metadata={"video_title": "demo"},
        turns=[
            ConversationTurn(
                utterance_id=0,
                speaker="therapist",
                text="how are you feeling today",
                timestamp=None,
                attributes={"main_therapist_behaviour": "question", "client_talk_type": None},
            ),
            ConversationTurn(
                utterance_id=1,
                speaker="client",
                text=client_text,
                timestamp=None,
                attributes={"main_therapist_behaviour": None, "client_talk_type": client_label},
            ),
        ],
    )


def test_run_baseline_experiments_writes_plot_and_results(tmp_path) -> None:
    conversations = [
        _conversation(1, "i want to quit smoking now", "change"),
        _conversation(2, "i really want to change my life", "change"),
        _conversation(3, "i cannot stop drinking ever", "sustain"),
        _conversation(4, "i will never be able to quit", "sustain"),
        _conversation(5, "today the weather is fine", "neutral"),
        _conversation(6, "i went to the store yesterday", "neutral"),
        _conversation(7, "i want to change for my kids", "change"),
        _conversation(8, "i cannot quit it is too hard", "sustain"),
        _conversation(9, "the meeting was at noon", "neutral"),
    ]
    split_ids = {
        "train": [1, 3, 5, 7, 8],
        "val": [2, 4],
        "test": [6, 9],
    }

    payload = run_baseline_experiments(
        conversations=conversations,
        split_ids=split_ids,
        allowed_labels={"change", "neutral", "sustain"},
        label_attribute="client_talk_type",
        baseline_config={
            "label_field": "metadata.client_talk_type",
            "scoring_split": "val",
            "context_turns_sweep": [0],
            "vocab_sizes": [32],
            "ngram_range": [1, 1],
            "class_weight": "balanced",
            "max_iter": 500,
            "random_state": 0,
        },
        output_dir=tmp_path,
    )

    assert len(payload["results"]) == 1
    assert payload["best_result"]["val_macro_f1"] > 0.0
    assert (tmp_path / "baseline_results.json").exists()
    assert (tmp_path / "baseline_results.png").exists()
