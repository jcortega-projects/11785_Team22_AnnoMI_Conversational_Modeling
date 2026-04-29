"""CLI entrypoint for Stage 1 data preparation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from annomi_pipeline.data.chunking import build_client_utterance_splits
from annomi_pipeline.data.ingestion import (
    build_conversations,
    load_annomi_dataframe,
    maybe_copy_source_to_raw,
    split_transcript_ids,
    validate_required_columns,
)
from annomi_pipeline.data.tokenizer import build_token_report, build_tokenized_splits
from annomi_pipeline.stage1.baseline import run_baseline_experiments
from annomi_pipeline.utils.config import load_yaml, resolve_path
from annomi_pipeline.utils.io import ensure_dir, write_json, write_jsonl
from annomi_pipeline.utils.logging import configure_logging
from annomi_pipeline.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run Stage 1 for the AnnoMI pipeline.")
    parser.add_argument("--config", required=True, help="Path to the data YAML configuration.")
    return parser.parse_args()


def main() -> None:
    """Execute Stage 1 end to end."""

    args = parse_args()
    configure_logging()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    set_global_seed(int(config.get("seed", 42)))

    paths_config = config["paths"]
    data_config = config["data"]
    config_dir = config_path.parent
    project_dir = config_dir.parent

    def _resolve(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path

        project_relative = resolve_path(path, project_dir)
        config_relative = resolve_path(path, config_dir)
        if project_relative.exists() or not config_relative.exists():
            return project_relative
        return config_relative

    processed_dir = ensure_dir(_resolve(paths_config["processed_dir"]))
    output_dir = ensure_dir(_resolve(paths_config["output_dir"]))
    tokenized_dir = (
        ensure_dir(_resolve(paths_config["tokenized_dir"]))
        if config["tokenizer"].get("save_tokenized_splits", False)
        else None
    )

    source_csv = _resolve(paths_config["source_csv"])
    if data_config.get("copy_source_to_raw", False):
        maybe_copy_source_to_raw(source_csv, _resolve(paths_config["raw_dir"]))

    dataframe = load_annomi_dataframe(
        csv_path=source_csv,
        sort_columns=[
            data_config["transcript_id_column"],
            data_config["utterance_id_column"],
        ],
    )
    required_columns = list(
        dict.fromkeys(
            [
                data_config["transcript_id_column"],
                data_config["utterance_id_column"],
                data_config["speaker_column"],
                data_config["text_column"],
                data_config["topic_column"],
                data_config["mi_quality_column"],
                data_config.get("timestamp_column"),
                *data_config.get("transcript_metadata_fields", []),
                *data_config.get("turn_attribute_fields", []),
            ]
        )
    )
    validate_required_columns(
        dataframe,
        [column for column in required_columns if column],
    )
    split_ids = split_transcript_ids(
        dataframe=dataframe,
        split_config=config["splits"],
        transcript_id_column=data_config["transcript_id_column"],
        seed=int(config.get("seed", 42)),
    )
    conversations = build_conversations(dataframe, data_config)

    task_config = config["task"]
    allowed_labels = {str(label).strip().lower() for label in task_config["allowed_labels"]}
    context_turns = int(task_config.get("context_turns", 0))
    split_records = build_client_utterance_splits(
        conversations=conversations,
        split_ids=split_ids,
        context_turns=context_turns,
        allowed_labels=allowed_labels,
        label_attribute=str(task_config.get("target", "client_talk_type")),
    )

    label_to_id = {label: idx for idx, label in enumerate(sorted(allowed_labels))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    write_json(
        processed_dir / "label_mapping.json",
        {"label_to_id": label_to_id, "id_to_label": id_to_label},
    )

    for split_name, records in split_records.items():
        write_jsonl(processed_dir / f"{split_name}.jsonl", records)
        LOGGER.info("Wrote %s utterance examples for %s", len(records), split_name)
        # Validation: every example must have an allowed label and be a client turn.
        for record in records:
            label = record["metadata"]["client_talk_type"]
            assert label in allowed_labels, f"Unexpected label {label} in {split_name}"
            assert str(record["metadata"]["speaker"]).lower() == "client"

    if tokenized_dir is not None:
        tokenized_splits, token_report = build_tokenized_splits(
            split_records=split_records,
            tokenizer_type=config["tokenizer"]["type"],
            hf_model_name=config["tokenizer"].get("hf_model_name"),
        )
        for split_name, records in tokenized_splits.items():
            write_jsonl(tokenized_dir / f"{split_name}.jsonl", records)
            LOGGER.info("Wrote %s tokenized examples for %s", len(records), split_name)
    else:
        token_report = build_token_report(
            split_records=split_records,
            tokenizer_type=config["tokenizer"]["type"],
            hf_model_name=config["tokenizer"].get("hf_model_name"),
        )
    write_json(output_dir / "token_stats.json", token_report)

    baseline_summary = None
    if config.get("baseline", {}).get("enabled", False):
        baseline_config = dict(config["baseline"])
        baseline_config.setdefault("random_state", int(config.get("seed", 42)))
        baseline_summary = run_baseline_experiments(
            conversations=conversations,
            split_ids=split_ids,
            allowed_labels=allowed_labels,
            label_attribute=str(task_config.get("target", "client_talk_type")),
            baseline_config=baseline_config,
            output_dir=output_dir,
        )

    summary = {
        "source_csv": str(source_csv),
        "conversation_count": len(conversations),
        "transcript_counts": {split_name: len(ids) for split_name, ids in split_ids.items()},
        "chunk_counts": {split_name: len(records) for split_name, records in split_records.items()},
        "processed_dir": str(processed_dir),
        "tokenized_dir": str(tokenized_dir) if tokenized_dir else "",
        "token_report_path": str(output_dir / "token_stats.json"),
        "baseline_results_path": str(output_dir / "baseline_results.json")
        if baseline_summary is not None
        else "",
        "baseline_plot_path": str(output_dir / "baseline_results.png")
        if baseline_summary is not None
        else "",
        "baseline_best_result": baseline_summary.get("best_result") if baseline_summary else None,
    }
    write_json(output_dir / "stage1_summary.json", summary)
    LOGGER.info("Stage 1 complete. Summary written to %s", output_dir / "stage1_summary.json")


if __name__ == "__main__":
    main()
