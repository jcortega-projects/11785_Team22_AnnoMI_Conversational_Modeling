"""Generate synthetic client utterances for the AnnoMI augmentation pipeline.

This script reads real training examples exported by
`export_train_augmentation_data.py`, calls the OpenAI API (o4-mini) to
generate diverse label-preserving paraphrases, and writes candidate rows to
a separate JSONL file.

Nothing is merged into train/val/test.  All output rows carry
  verification_status="unverified"  accepted_for_training=false
so a downstream QA pass is required before any row enters the model.

Usage
-----
    python -m annomi_pipeline.scripts.generate_synthetic_candidates_openai \\
        --input  data/outputs/augmentation/train_real_examples_for_augmentation.jsonl \\
        --output data/outputs/augmentation/synthetic_candidates.jsonl \\
        --sustain-quota 800 \\
        --change-quota  300 \\
        --model o4-mini \\
        --batch-size 5 \\
        --rng-seed 42

Environment
-----------
    OPENAI_API_KEY   may come from the existing shell environment, from an
                     explicit --env-file path, from ANNOMI_ENV_FILE /
                     OPENAI_ENV_FILE, from ~/.config/annomi-mlp/.env, or
                     from the legacy project-root .env fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from annomi_pipeline.data.synthetic_generation import (
    ALLOWED_LABELS,
    BATCH_SIZE,
    BUFFER_FACTOR,
    build_prompt,
    build_synthetic_row,
    call_openai_with_retry,
    quality_filter,
    select_seeds,
)
from annomi_pipeline.utils.openai_env import load_openai_env

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-label generation loop
# ---------------------------------------------------------------------------


def generate_for_label(
    *,
    client: OpenAI,
    model: str,
    pool: list[dict],
    label: str,
    quota: int,
    output_path: Path,
    batch_size: int,
    rng_seed: int,
    inter_call_sleep: float,
    per_label_counter_start: int = 0,
) -> dict[str, int]:
    """
    Generate `quota` synthetic examples for `label`.

    Returns a stats dict:
        requested, generated_raw, accepted, rejected, failed_calls
    """
    stats = {
        "requested": quota,
        "generated_raw": 0,
        "accepted": 0,
        "rejected": 0,
        "failed_calls": 0,
    }

    # How many API calls do we need?  Apply buffer factor to cover rejections.
    target_with_buffer = math.ceil(quota * BUFFER_FACTOR)
    n_calls = math.ceil(target_with_buffer / batch_size)

    seeds = select_seeds(pool, n_calls, rng_seed=rng_seed)

    LOGGER.info(
        "[%s] quota=%d  calls_planned=%d  seeds_pool=%d  seeds_selected=%d",
        label, quota, n_calls, len(pool), len(seeds),
    )

    counter = per_label_counter_start

    for call_idx, seed in enumerate(seeds, start=1):
        if stats["accepted"] >= quota:
            LOGGER.info("[%s] quota reached at call %d — stopping.", label, call_idx)
            break

        prompt = build_prompt(seed, n=batch_size)
        raw_candidates = call_openai_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            seed_id=seed["example_id"],
        )

        if raw_candidates is None:
            stats["failed_calls"] += 1
            LOGGER.warning("[%s] call %d/%d FAILED (seed=%s)",
                           label, call_idx, n_calls, seed["example_id"])
            continue

        stats["generated_raw"] += len(raw_candidates)

        for candidate_text in raw_candidates:
            if stats["accepted"] >= quota:
                break

            accepted, reason = quality_filter(candidate_text, seed["client_text"])

            if accepted:
                row = build_synthetic_row(
                    generated_text=candidate_text.strip(),
                    seed=seed,
                    label=label,
                    model=model,
                    counter=counter,
                )
                _append_jsonl(output_path, row.to_dict())
                stats["accepted"] += 1
                counter += 1
            else:
                stats["rejected"] += 1
                LOGGER.debug(
                    "[%s] rejected (reason=%s): %r", label, reason, candidate_text[:80]
                )

        LOGGER.info(
            "[%s] call %d/%d | accepted=%d/%d | raw_this_call=%d",
            label, call_idx, n_calls,
            stats["accepted"], quota,
            len(raw_candidates),
        )

        # Rate-limit courtesy sleep
        if call_idx < len(seeds):
            time.sleep(inter_call_sleep)

    if stats["accepted"] < quota:
        LOGGER.warning(
            "[%s] UNDER-QUOTA: accepted=%d requested=%d  "
            "(increase --buffer-factor or --max-retries)",
            label, stats["accepted"], quota,
        )

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic client utterances via OpenAI (o4-mini)."
    )
    p.add_argument(
        "--input",
        default="data/outputs/augmentation/train_real_examples_for_augmentation.jsonl",
        help="Real training examples (augmentation export).",
    )
    p.add_argument(
        "--output",
        default="data/outputs/augmentation/synthetic_candidates.jsonl",
        help="Destination for synthetic candidates (JSONL).",
    )
    p.add_argument(
        "--env-file",
        help="Optional path to a dotenv file outside the repo that defines OPENAI_API_KEY.",
    )
    p.add_argument("--sustain-quota", type=int, default=800)
    p.add_argument("--change-quota",  type=int, default=300)
    p.add_argument("--model", default="o4-mini")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Utterances requested per API call.")
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument(
        "--inter-call-sleep", type=float, default=0.5,
        help="Seconds to sleep between API calls (rate-limit safety).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="If output file exists, count already-generated rows and resume "
             "from the remaining quota.",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)

    load_openai_env(search_start=Path(__file__), explicit_env_file=args.env_file)

    # ------------------------------------------------------------------
    # Validate API key
    # ------------------------------------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error(
            "OPENAI_API_KEY not found. Export it in the shell, pass --env-file, "
            "set ANNOMI_ENV_FILE/OPENAI_ENV_FILE, or use the legacy project-root .env."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Load seed pool
    # ------------------------------------------------------------------
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        sys.exit(1)

    all_rows = _read_jsonl(input_path)
    LOGGER.info("Loaded %d rows from %s", len(all_rows), input_path)

    pools: dict[str, list[dict]] = {
        label: [r for r in all_rows if r.get("label") == label]
        for label in ALLOWED_LABELS
    }
    for label, pool in pools.items():
        LOGGER.info("  %s pool: %d real examples", label, len(pool))

    # ------------------------------------------------------------------
    # Output file setup
    # ------------------------------------------------------------------
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quotas: dict[str, int] = {
        "sustain": args.sustain_quota,
        "change":  args.change_quota,
    }

    # Resume: count already-written rows per label
    already_done: dict[str, int] = {"sustain": 0, "change": 0}
    if args.resume and output_path.exists():
        existing = _read_jsonl(output_path)
        for r in existing:
            lbl = r.get("label")
            if lbl in already_done:
                already_done[lbl] += 1
        LOGGER.info(
            "Resume mode: found sustain=%d change=%d already in output.",
            already_done["sustain"], already_done["change"],
        )
    elif output_path.exists() and not args.resume:
        LOGGER.warning(
            "Output file already exists and --resume not set.  "
            "Rows will be APPENDED.  Delete the file or pass --resume to start fresh."
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    all_stats: dict[str, dict] = {}

    for label in ("sustain", "change"):
        remaining_quota = quotas[label] - already_done[label]
        if remaining_quota <= 0:
            LOGGER.info("[%s] quota already met (%d/%d) — skipping.",
                        label, already_done[label], quotas[label])
            all_stats[label] = {
                "requested": quotas[label],
                "generated_raw": 0,
                "accepted": already_done[label],
                "rejected": 0,
                "failed_calls": 0,
            }
            continue

        stats = generate_for_label(
            client=client,
            model=args.model,
            pool=pools[label],
            label=label,
            quota=remaining_quota,
            output_path=output_path,
            batch_size=args.batch_size,
            rng_seed=args.rng_seed,
            inter_call_sleep=args.inter_call_sleep,
            per_label_counter_start=already_done[label],
        )
        all_stats[label] = stats

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYNTHETIC GENERATION SUMMARY")
    print("=" * 60)
    print(f"{'Label':<12} {'Requested':>10} {'Generated':>10} {'Accepted':>10} "
          f"{'Rejected':>10} {'Failed':>8}")
    print("-" * 60)
    for label, s in all_stats.items():
        print(
            f"{label:<12} {s['requested']:>10} {s['generated_raw']:>10} "
            f"{s['accepted']:>10} {s['rejected']:>10} {s['failed_calls']:>8}"
        )
    print("=" * 60)
    total_accepted = sum(s["accepted"] for s in all_stats.values())
    print(f"Total accepted rows written to: {output_path}")
    print(f"Total accepted:  {total_accepted}")
    print(f"verification_status on all rows: 'unverified'")
    print(f"accepted_for_training on all rows: false")
    print("=" * 60)
    print()
    print("Next step: run the QA/filtering pass before merging into training.")


if __name__ == "__main__":
    main()
