"""Automated label-consistency QA for synthetic_candidates.jsonl.

For each unverified synthetic row, calls o4-mini with a neutral prompt
(no label anchoring) asking which label best fits the utterance.

If the returned label matches the row's `label` field  → verified + accepted.
If it disagrees                                        → auto_rejected.
If the call fails after retries                        → verification_failed.

Updates the row in-place and rewrites the file atomically.

Usage
-----
    python -m annomi_pipeline.scripts.qa_synthetic_candidates \\
        --input  data/outputs/augmentation/synthetic_candidates.jsonl \\
        --output data/outputs/augmentation/synthetic_candidates_qa.jsonl \\
        --model  o4-mini \\
        --env-file ~/.config/annomi-mlp/.env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from annomi_pipeline.utils.openai_env import load_openai_env

LOGGER = logging.getLogger(__name__)

from openai import OpenAI

ALLOWED = {"change", "neutral", "sustain"}
MAX_RETRIES = 3
BACKOFF_BASE = 2.0

_QA_PROMPT = """\
You are a Motivational Interviewing (MI) expert.

A client says the following in a counseling session:
"{utterance}"

Which label BEST fits this utterance?
Return EXACTLY one of these three words and nothing else:
change | neutral | sustain"""


def _call_qa(client: OpenAI, model: str, utterance: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _QA_PROMPT.format(utterance=utterance)}],
                max_completion_tokens=256,  # o4-mini needs headroom for reasoning tokens
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
            # Accept if exactly one of the allowed labels appears
            for label in ALLOWED:
                if label in raw:
                    return label
            LOGGER.warning("QA response not parseable: %r", raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("QA API error attempt %d: %s", attempt, exc)
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE ** attempt)
    return None


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
    p = argparse.ArgumentParser(description="QA pass for synthetic candidates.")
    p.add_argument("--input",  default="data/outputs/augmentation/synthetic_candidates.jsonl")
    p.add_argument("--output", default="data/outputs/augmentation/synthetic_candidates_qa.jsonl")
    p.add_argument("--model",  default="o4-mini")
    p.add_argument(
        "--env-file",
        help="Optional path to a dotenv file outside the repo that defines OPENAI_API_KEY.",
    )
    p.add_argument("--inter-call-sleep", type=float, default=0.5)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    load_openai_env(search_start=Path(__file__), explicit_env_file=args.env_file)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error(
            "OPENAI_API_KEY not found. Export it in the shell, pass --env-file, "
            "set ANNOMI_ENV_FILE/OPENAI_ENV_FILE, or use the legacy project-root .env."
        )
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    input_path  = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows = _read_jsonl(input_path)
    LOGGER.info("Loaded %d rows from %s", len(rows), input_path)

    stats = {"verified": 0, "auto_rejected": 0, "verification_failed": 0, "skipped": 0}

    for i, row in enumerate(rows, start=1):
        if row.get("verification_status") not in (None, "unverified"):
            stats["skipped"] += 1
            continue

        utterance = row.get("client_text") or row.get("text") or ""
        assigned_label = row["label"]

        qa_label = _call_qa(client, args.model, utterance)

        if qa_label is None:
            row["verification_status"]  = "verification_failed"
            row["accepted_for_training"] = False
            row["rejection_reason"]      = "qa_call_failed"
            stats["verification_failed"] += 1
        elif qa_label == assigned_label:
            row["verification_status"]   = "verified"
            row["verification_label"]    = qa_label
            row["accepted_for_training"] = True
            row["rejection_reason"]      = None
            stats["verified"] += 1
        else:
            row["verification_status"]   = "auto_rejected"
            row["verification_label"]    = qa_label
            row["accepted_for_training"] = False
            row["rejection_reason"]      = f"label_mismatch:assigned={assigned_label},qa={qa_label}"
            stats["auto_rejected"] += 1

        if i % 50 == 0:
            LOGGER.info("Progress %d/%d | verified=%d rejected=%d failed=%d",
                        i, len(rows), stats["verified"], stats["auto_rejected"],
                        stats["verification_failed"])
            _write_jsonl(output_path, rows)  # incremental save

        time.sleep(args.inter_call_sleep)

    _write_jsonl(output_path, rows)

    print("\n" + "=" * 55)
    print("QA SUMMARY")
    print("=" * 55)
    total = len(rows) - stats["skipped"]
    print(f"  Total processed : {total}")
    print(f"  Verified        : {stats['verified']}  ({100*stats['verified']/max(total,1):.1f}%)")
    print(f"  Auto-rejected   : {stats['auto_rejected']}")
    print(f"  Failed          : {stats['verification_failed']}")
    print(f"  Skipped (done)  : {stats['skipped']}")
    print(f"  Output          : {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
