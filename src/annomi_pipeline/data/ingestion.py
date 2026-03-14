"""Data ingestion and transcript-level splitting for the AnnoMI dataset."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from annomi_pipeline.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversationTurn:
    """A single ordered turn within an AnnoMI conversation."""

    utterance_id: int
    speaker: str
    text: str
    timestamp: str | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Conversation:
    """A reconstructed conversation grouped by transcript identifier."""

    transcript_id: int | str
    topic: str | None
    mi_quality: str | None
    transcript_metadata: dict[str, Any]
    turns: list[ConversationTurn]


def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python primitives."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def load_annomi_dataframe(csv_path: str | Path, sort_columns: list[str]) -> pd.DataFrame:
    """Load and sort the raw AnnoMI CSV."""

    resolved = Path(csv_path).expanduser().resolve()
    LOGGER.info("Loading dataset from %s", resolved)
    dataframe = pd.read_csv(resolved)
    dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)
    LOGGER.info(
        "Loaded %s rows spanning %s transcripts",
        len(dataframe),
        dataframe[sort_columns[0]].nunique(),
    )
    return dataframe


def validate_required_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    """Ensure the configured columns are present in the loaded dataframe."""

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def maybe_copy_source_to_raw(source_csv: str | Path, raw_dir: str | Path) -> Path:
    """Copy the configured source CSV into the raw data directory."""

    source_path = Path(source_csv).expanduser().resolve()
    destination_dir = ensure_dir(raw_dir)
    destination_path = destination_dir / source_path.name
    if source_path != destination_path:
        shutil.copy2(source_path, destination_path)
        LOGGER.info("Copied source dataset to %s", destination_path)
    return destination_path


def build_conversations(dataframe: pd.DataFrame, data_config: dict[str, Any]) -> list[Conversation]:
    """Reconstruct ordered conversations from utterance-level rows."""

    transcript_id_column = data_config["transcript_id_column"]
    utterance_id_column = data_config["utterance_id_column"]
    speaker_column = data_config["speaker_column"]
    text_column = data_config["text_column"]
    topic_column = data_config["topic_column"]
    mi_quality_column = data_config["mi_quality_column"]
    timestamp_column = data_config.get("timestamp_column")
    transcript_metadata_fields = [
        column
        for column in data_config.get("transcript_metadata_fields", [])
        if column in dataframe.columns and column != timestamp_column
    ]
    turn_attribute_fields = [
        column
        for column in data_config.get("turn_attribute_fields", [])
        if column in dataframe.columns and column != timestamp_column
    ]

    conversations: list[Conversation] = []
    for transcript_id, group in dataframe.groupby(transcript_id_column, sort=True):
        ordered = group.sort_values(utterance_id_column)
        first_row = ordered.iloc[0]
        transcript_metadata = {
            field: _clean_value(first_row[field]) for field in transcript_metadata_fields
        }
        turns = [
            ConversationTurn(
                utterance_id=int(row[utterance_id_column]),
                speaker=str(_clean_value(row[speaker_column]) or "unknown"),
                text=str(_clean_value(row[text_column]) or ""),
                timestamp=_clean_value(row[timestamp_column])
                if timestamp_column and timestamp_column in ordered.columns
                else None,
                attributes={
                    field: _clean_value(row[field])
                    for field in turn_attribute_fields
                },
            )
            for _, row in ordered.iterrows()
        ]
        conversations.append(
            Conversation(
                transcript_id=_clean_value(transcript_id),
                topic=_clean_value(first_row[topic_column]),
                mi_quality=_clean_value(first_row[mi_quality_column]),
                transcript_metadata=transcript_metadata,
                turns=turns,
            )
        )
    return conversations


def split_transcript_ids(
    dataframe: pd.DataFrame,
    split_config: dict[str, Any],
    transcript_id_column: str,
    seed: int,
) -> dict[str, list[int | str]]:
    """Create train/validation/test transcript splits without leakage."""

    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    transcript_frame = dataframe.groupby(transcript_id_column, as_index=False).first()
    transcript_ids = transcript_frame[transcript_id_column].tolist()

    stratify_column = split_config.get("stratify_by")
    stratify_values = (
        transcript_frame[stratify_column].tolist()
        if stratify_column and stratify_column in transcript_frame.columns
        else None
    )

    try:
        train_ids, temp_ids, _, temp_labels = train_test_split(
            transcript_ids,
            stratify_values,
            test_size=(1.0 - train_ratio),
            random_state=seed,
            stratify=stratify_values,
        )
    except ValueError:
        LOGGER.warning("Falling back to unstratified split because stratification was not feasible.")
        train_ids, temp_ids = train_test_split(
            transcript_ids,
            test_size=(1.0 - train_ratio),
            random_state=seed,
        )
        temp_labels = None

    val_share_of_temp = val_ratio / (val_ratio + test_ratio)
    try:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_share_of_temp),
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_share_of_temp),
            random_state=seed,
        )

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
