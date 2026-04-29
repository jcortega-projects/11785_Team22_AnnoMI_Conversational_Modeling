"""TF-IDF + Logistic Regression baseline for client_talk_type prediction.

Sweeps over (context_turns, vocab_size). For each (context_turns) value the
utterance-level examples are rebuilt so therapist context is included only
when context_turns > 0. Targets are always single client utterances.
"""

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

from annomi_pipeline.data.chunking import build_client_utterance_splits
from annomi_pipeline.data.ingestion import Conversation
from annomi_pipeline.utils.io import ensure_dir, write_json
from annomi_pipeline.utils.serialization import get_by_path, is_missing


def _extract_labeled_examples(
    records: list[dict[str, Any]],
    label_field: str,
) -> tuple[list[str], list[str]]:
    """Collect texts and labels, dropping rows with missing labels."""

    texts: list[str] = []
    labels: list[str] = []
    for record in records:
        label = get_by_path(record, label_field)
        if is_missing(label):
            continue
        texts.append(str(record["text"]))
        labels.append(str(label))
    return texts, labels


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def save_baseline_plot(
    results: list[dict[str, Any]],
    scoring_split: str,
    output_path: str | Path,
) -> None:
    """Render the baseline plot (one line per vocabulary size)."""

    if not results:
        return

    score_key = f"{scoring_split}_macro_f1"
    output = Path(output_path)
    ensure_dir(output.parent)

    plt.figure(figsize=(9, 5.5))
    for max_features in sorted({int(result["max_features"]) for result in results}):
        subset = sorted(
            [r for r in results if int(r["max_features"]) == max_features],
            key=lambda r: int(r["context_turns"]),
        )
        x_values = [int(r["context_turns"]) for r in subset]
        y_values = [float(r[score_key]) for r in subset]
        plt.plot(x_values, y_values, marker="o", linewidth=2,
                 label=f"max_features={max_features}")
        for x_value, y_value in zip(x_values, y_values):
            plt.annotate(f"{y_value:.2f}", (x_value, y_value),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=8)

    plt.xlabel("Context turns (preceding utterances included)")
    plt.ylabel(f"{scoring_split.title()} macro F1")
    plt.title("Baseline: TF-IDF + Logistic Regression on client_talk_type")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Vocabulary size")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def run_baseline_experiments(
    conversations: list[Conversation],
    split_ids: dict[str, list[int | str]],
    allowed_labels: set[str],
    label_attribute: str,
    baseline_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Sweep TF-IDF/LogReg baselines over context_turns and vocab_sizes."""

    label_field = str(baseline_config.get("label_field", "metadata.client_talk_type"))
    scoring_split = str(baseline_config.get("scoring_split", "val"))
    if scoring_split not in {"val", "test"}:
        raise ValueError("baseline.scoring_split must be 'val' or 'test'")

    output_path = ensure_dir(output_dir)
    context_sweep = [int(c) for c in baseline_config.get("context_turns_sweep", [0])]
    vocab_sizes = [int(v) for v in baseline_config.get("vocab_sizes", [2048])]
    lowercase = bool(baseline_config.get("lowercase", True))
    ngram_range = tuple(int(v) for v in baseline_config.get("ngram_range", [1, 2]))
    max_iter = int(baseline_config.get("max_iter", 2000))
    class_weight = baseline_config.get("class_weight")
    random_state = int(baseline_config.get("random_state", 42))

    results: list[dict[str, Any]] = []
    for context_turns in context_sweep:
        split_records = build_client_utterance_splits(
            conversations=conversations,
            split_ids=split_ids,
            context_turns=context_turns,
            allowed_labels=allowed_labels,
            label_attribute=label_attribute,
        )

        train_texts, train_labels = _extract_labeled_examples(split_records["train"], label_field)
        val_texts, val_labels = _extract_labeled_examples(split_records["val"], label_field)
        test_texts, test_labels = _extract_labeled_examples(split_records["test"], label_field)

        if len(set(train_labels)) < 2:
            raise ValueError("Training split must contain at least two label classes.")

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

            val_pred = classifier.predict(val_matrix).tolist() if val_labels else []
            test_pred = classifier.predict(test_matrix).tolist() if test_labels else []

            results.append({
                "context_turns": context_turns,
                "max_features": vocab_size,
                "label_field": label_field,
                "classes": sorted(set(train_labels)),
                "train_examples": len(train_labels),
                "val_examples": len(val_labels),
                "test_examples": len(test_labels),
                "val_macro_f1": _macro_f1(val_labels, val_pred),
                "test_macro_f1": _macro_f1(test_labels, test_pred),
            })

    score_key = f"{scoring_split}_macro_f1"
    best_result = max(results, key=lambda r: float(r[score_key])) if results else None
    payload = {
        "label_field": label_field,
        "scoring_split": scoring_split,
        "context_turns_sweep": context_sweep,
        "vocab_sizes": vocab_sizes,
        "results": results,
        "best_result": best_result,
    }
    write_json(output_path / "baseline_results.json", payload)
    save_baseline_plot(results, scoring_split, output_path / "baseline_results.png")
    return payload
