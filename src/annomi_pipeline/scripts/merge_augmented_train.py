"""Merge QA-accepted synthetic candidates into an augmented training JSONL.

Reads the original train.jsonl and the QA-verified synthetic_candidates_qa.jsonl,
keeps only rows where accepted_for_training=true from the synthetic file,
and writes train_augmented.jsonl.

Val and test splits are NEVER touched.

Usage
-----
    python -m annomi_pipeline.scripts.merge_augmented_train \\
        --train   data/processed/train.jsonl \\
        --synthetic data/outputs/augmentation/synthetic_candidates_qa.jsonl \\
        --output  data/processed/train_augmented.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge synthetic candidates into training split.")
    p.add_argument("--train",     default="data/processed/train.jsonl")
    p.add_argument("--synthetic", default="data/outputs/augmentation/synthetic_candidates_qa.jsonl")
    p.add_argument("--output",    default="data/processed/train_augmented.jsonl")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    train_path     = Path(args.train).expanduser().resolve()
    synthetic_path = Path(args.synthetic).expanduser().resolve()
    output_path    = Path(args.output).expanduser().resolve()

    real_rows = _read_jsonl(train_path)
    syn_rows  = _read_jsonl(synthetic_path)

    accepted = [r for r in syn_rows if r.get("accepted_for_training") is True]
    rejected = len(syn_rows) - len(accepted)

    LOGGER.info("Real training rows  : %d", len(real_rows))
    LOGGER.info("Synthetic total     : %d", len(syn_rows))
    LOGGER.info("Synthetic accepted  : %d", len(accepted))
    LOGGER.info("Synthetic rejected  : %d", rejected)

    merged = real_rows + accepted
    _write_jsonl(output_path, merged)

    # Summary breakdown
    real_dist = Counter(r.get("label") or r.get("metadata", {}).get("client_talk_type") for r in real_rows)
    syn_dist  = Counter(r.get("label") for r in accepted)
    merged_dist = Counter(r.get("label") or r.get("metadata", {}).get("client_talk_type") for r in merged)

    print("\n" + "=" * 55)
    print("MERGE SUMMARY")
    print("=" * 55)
    print(f"{'Label':<12} {'Real':>8} {'Synthetic':>10} {'Merged':>8}")
    print("-" * 55)
    for lbl in sorted(set(list(real_dist) + list(syn_dist))):
        print(f"{str(lbl):<12} {real_dist.get(lbl,0):>8} {syn_dist.get(lbl,0):>10} {merged_dist.get(lbl,0):>8}")
    print("=" * 55)
    print(f"Total merged rows: {len(merged)}")
    print(f"Output: {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
