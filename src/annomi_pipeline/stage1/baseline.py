"""TF-IDF and logistic-regression baseline experiments for Stage 1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from annomi_pipeline.data.chunking import chunk_splits
from annomi_pipeline.data.ingestion import Conversation
from annomi_pipeline.utils.io import ensure_dir, write_json
from annomi_pipeline.utils.serialization import get_by_path, is_missing


def _extract_labeled_examples(
    records: list[dict[str, Any]],
    label_field: str,
) -> tuple[list[str], list[str]]:
    """Collect texts and labels from records, skipping missing labels."""

    texts: list[str] = []
    labels: list[str] = []
    for record in records:
        label = get_by_path(record, label_field)
        if is_missing(label):
            continue
        texts.append(str(record["text"]))
        labels.append(str(label))
    return texts, labels


def _filter_known_labels(
    texts: list[str],
    labels: list[str],
    known_labels: set[str],
) -> tuple[list[str], list[str]]:
    """Keep only examples whose labels were seen in training."""

    kept = [(text, label) for text, label in zip(texts, labels) if label in known_labels]
    if not kept:
        return [], []
    kept_texts, kept_labels = zip(*kept)
    return list(kept_texts), list(kept_labels)


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Compute macro F1 while handling empty inputs."""

    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _resolve_overlap(
    turns_per_chunk: int,
    base_chunk_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> int:
    """Scale overlap when sweeping different chunk window sizes."""

    if turns_per_chunk <= 1:
        return 0

    if "overlap_ratio" in baseline_config:
        overlap = int(round(float(baseline_config["overlap_ratio"]) * turns_per_chunk))
        return min(turns_per_chunk - 1, max(0, overlap))

    base_turns = int(base_chunk_config["turns_per_chunk"])
    base_overlap = int(base_chunk_config["overlap"])
    if base_turns <= 0:
        return 0

    scaled_overlap = int(round((base_overlap / base_turns) * turns_per_chunk))
    return min(turns_per_chunk - 1, max(0, scaled_overlap))


def _build_experiment_chunk_config(
    base_chunk_config: dict[str, Any],
    turns_per_chunk: int,
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    """Create the chunking config for a specific baseline experiment."""

    config = dict(base_chunk_config)
    config["turns_per_chunk"] = int(turns_per_chunk)
    config["overlap"] = _resolve_overlap(int(turns_per_chunk), base_chunk_config, baseline_config)
    return config


def save_baseline_plot(
    results: list[dict[str, Any]],
    scoring_split: str,
    output_path: str | Path,
) -> None:
    """Render the preliminary baseline plot used as Figure 2."""

    if not results:
        return

    score_key = f"{scoring_split}_macro_f1"
    output = Path(output_path)
    ensure_dir(output.parent)

    plt.figure(figsize=(9, 5.5))
    for max_features in sorted({int(result["max_features"]) for result in results}):
        subset = sorted(
            [result for result in results if int(result["max_features"]) == max_features],
            key=lambda result: int(result["turns_per_chunk"]),
        )
        x_values = [int(result["turns_per_chunk"]) for result in subset]
        y_values = [float(result[score_key]) for result in subset]
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            label=f"max_features={max_features}",
        )
        for x_value, y_value in zip(x_values, y_values):
            plt.annotate(
                f"{y_value:.2f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    plt.xlabel("Chunk window size (utterances)")
    plt.ylabel(f"{scoring_split.title()} macro F1")
    plt.title("Preliminary baseline performance using TF-IDF + Logistic Regression")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Vocabulary size")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def run_baseline_experiments(
    conversations: list[Conversation],
    split_ids: dict[str, list[int | str]],
    base_chunk_config: dict[str, Any],
    baseline_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the Stage 1 TF-IDF/logistic-regression baseline sweep."""

    label_field = str(baseline_config["label_field"])
    scoring_split = str(baseline_config.get("scoring_split", "val"))
    if scoring_split not in {"val", "test"}:
        raise ValueError("baseline.scoring_split must be either 'val' or 'test'")

    output_path = ensure_dir(output_dir)
    chunk_window_sizes = [
        int(size)
        for size in baseline_config.get(
            "chunk_window_sizes",
            [base_chunk_config["turns_per_chunk"]],
        )
    ]
    vocab_sizes = [int(size) for size in baseline_config.get("vocab_sizes", [2048])]
    lowercase = bool(baseline_config.get("lowercase", True))
    ngram_range = tuple(int(value) for value in baseline_config.get("ngram_range", [1, 2]))
    max_iter = int(baseline_config.get("max_iter", 1000))
    class_weight = baseline_config.get("class_weight")
    random_state = int(baseline_config.get("random_state", 42))

    results: list[dict[str, Any]] = []
    for turns_per_chunk in chunk_window_sizes:
        experiment_chunk_config = _build_experiment_chunk_config(
            base_chunk_config=base_chunk_config,
            turns_per_chunk=turns_per_chunk,
            baseline_config=baseline_config,
        )
        split_records = chunk_splits(conversations, split_ids, experiment_chunk_config)

        train_texts, train_labels = _extract_labeled_examples(split_records["train"], label_field)
        val_texts, val_labels = _extract_labeled_examples(split_records["val"], label_field)
        test_texts, test_labels = _extract_labeled_examples(split_records["test"], label_field)

        known_labels = set(train_labels)
        if len(known_labels) < 2:
            raise ValueError("Baseline training split must contain at least two label classes.")

        val_texts, val_labels = _filter_known_labels(val_texts, val_labels, known_labels)
        test_texts, test_labels = _filter_known_labels(test_texts, test_labels, known_labels)

        for vocab_size in vocab_sizes:
            vectorizer = TfidfVectorizer(
                max_features=vocab_size,
                lowercase=lowercase,
                ngram_range=ngram_range,
            )
            train_matrix = vectorizer.fit_transform(train_texts)
            val_matrix = vectorizer.transform(val_texts)
            test_matrix = vectorizer.transform(test_texts)

            classifier = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                class_weight=class_weight,
            )
            classifier.fit(train_matrix, train_labels)

            val_predictions = classifier.predict(val_matrix).tolist() if val_labels else []
            test_predictions = classifier.predict(test_matrix).tolist() if test_labels else []

            results.append(
                {
                    "turns_per_chunk": turns_per_chunk,
                    "overlap": int(experiment_chunk_config["overlap"]),
                    "max_features": vocab_size,
                    "label_field": label_field,
                    "classes": sorted(known_labels),
                    "train_examples": len(train_labels),
                    "val_examples": len(val_labels),
                    "test_examples": len(test_labels),
                    "val_macro_f1": _macro_f1(val_labels, val_predictions),
                    "test_macro_f1": _macro_f1(test_labels, test_predictions),
                }
            )

    score_key = f"{scoring_split}_macro_f1"
    best_result = max(results, key=lambda result: float(result[score_key])) if results else None
    payload = {
        "label_field": label_field,
        "scoring_split": scoring_split,
        "chunk_window_sizes": chunk_window_sizes,
        "vocab_sizes": vocab_sizes,
        "results": results,
        "best_result": best_result,
    }

    write_json(output_path / "baseline_results.json", payload)
    save_baseline_plot(results, scoring_split, output_path / "baseline_results.png")
    return payload
