"""GPT-4o-mini zero-shot & few-shot classification on AnnoMI.

Stage 3 / TA Change II: prompt-based frontier LLM as a classifier, comparison baseline
against every trained model in the report.

Usage:
    python scripts/run_gpt4o_mini_classify.py --mode zero_shot
    python scripts/run_gpt4o_mini_classify.py --mode few_shot
    python scripts/run_gpt4o_mini_classify.py --mode both     # default

Requires:
    pip install openai python-dotenv scikit-learn tqdm
    .env file at repo root with: OPENAI_API_KEY=sk-...

Cost estimate: ~$0.10 per mode on the 866-example test set.
Runtime: ~5–10 min per mode with concurrency=10.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm.asyncio import tqdm

# -------------------- Config --------------------
MODEL        = "gpt-4o-mini"
CLASS_NAMES  = ["change", "neutral", "sustain"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES  = len(CLASS_NAMES)

CONCURRENCY  = 10        # parallel API requests
TEMPERATURE  = 0.0       # deterministic
SEED         = 42
FEW_SHOT_K   = 3         # examples per class → 9 total in few-shot prompt
MAX_RETRIES  = 3         # per-request retry on transient errors

# GPT-4o-mini pricing (as of 2024-2025). Used for cost reporting only.
PRICE_IN_PER_1M  = 0.150
PRICE_OUT_PER_1M = 0.600

REPO = Path(__file__).resolve().parent.parent
TRAIN_JSONL = REPO / "data" / "processed" / "train.jsonl"
TEST_JSONL  = REPO / "data" / "processed" / "test.jsonl"

# -------------------- Canonical MI label definitions --------------------
# Matches src/annomi_pipeline/data/synthetic_generation.py. Using the same wording
# the dataset was labelled against avoids introducing a definitional mismatch.
SYSTEM_PROMPT = """\
You are an expert in Motivational Interviewing (MI) transcript annotation.

Your task: classify a single CLIENT utterance from an MI counseling session into
exactly ONE of three categories:

- "change": the client is expressing desire, ability, reason, or commitment to
  change their behavior (the target behavior being discussed in the session,
  such as reducing alcohol, quitting smoking, eating healthier, etc.).

- "sustain": the client is defending the status quo, expressing reasons NOT to
  change, or resisting the idea of changing their behavior.

- "neutral": everything else — off-topic remarks, short acknowledgements
  ("okay", "sure", "yeah"), clarifying questions, or statements that are
  neither clearly change talk nor clearly sustain talk.

Rules:
- Judge only the client's utterance itself.
- Acknowledgements, hedges, and ambivalent statements that lean neither way are
  "neutral".
- If the client states a concrete desire, ability, reason, or commitment to
  change, label "change" — even if briefly stated.
- If the client defends the current behavior or resists changing, label "sustain".

Respond with a JSON object matching the provided schema: {"label": "<one of change|neutral|sustain>"}.
"""

RESPONSE_SCHEMA = {
    "name": "mi_classification",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": CLASS_NAMES},
        },
        "required": ["label"],
        "additionalProperties": False,
    },
    "strict": True,
}

# -------------------- Data loading --------------------
def _get_nested(d: dict, dotted: str):
    cur = d
    for p in dotted.split("."):
        cur = cur[p]
    return cur

def load_split(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lab = _get_nested(rec, "metadata.client_talk_type")
            if lab not in CLASS_TO_IDX:
                continue
            rows.append({
                "text":  rec["text"],
                "topic": rec.get("topic", ""),
                "label": lab,
            })
    return rows

# -------------------- Prompt builders --------------------
def build_messages_zero_shot(text: str, topic: str) -> list[dict]:
    user_msg = (
        f"Session topic: {topic or 'unknown'}\n\n"
        f'Client utterance to classify:\n"{text}"'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

def build_messages_few_shot(text: str, topic: str, exemplars: list[dict]) -> list[dict]:
    """Inject k examples per class as prior user/assistant turns."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        messages.append({
            "role": "user",
            "content": (
                f"Session topic: {ex['topic'] or 'unknown'}\n\n"
                f'Client utterance to classify:\n"{ex["text"]}"'
            ),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({"label": ex["label"]}),
        })
    messages.append({
        "role": "user",
        "content": (
            f"Session topic: {topic or 'unknown'}\n\n"
            f'Client utterance to classify:\n"{text}"'
        ),
    })
    return messages

def sample_few_shot_pool(train_rows: list[dict], k_per_class: int, seed: int) -> list[dict]:
    """Stratified sample of k examples per class, shuffled together for in-context order.

    Same exemplars used for every test item (fixed with seed) — avoids noise from
    per-item resampling and keeps cost predictable.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in train_rows:
        by_class[r["label"]].append(r)
    pool = []
    for cls in CLASS_NAMES:
        pool.extend(rng.sample(by_class[cls], k_per_class))
    rng.shuffle(pool)  # interleave classes so the model doesn't see all of one class in a row
    return pool

# -------------------- Async classification --------------------
async def classify_one(client: AsyncOpenAI, messages: list[dict], semaphore: asyncio.Semaphore):
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                )
                content = resp.choices[0].message.content
                parsed = json.loads(content)
                return {
                    "label": parsed["label"],
                    "input_tokens":  resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                    "error": None,
                }
            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"label": "neutral", "input_tokens": 0, "output_tokens": 0,
                            "error": f"{type(e).__name__}: {e}"}
                await asyncio.sleep(2 ** attempt)  # exponential backoff

async def classify_all(test_rows: list[dict], mode: str, exemplars: list[dict] | None):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for r in test_rows:
        if mode == "zero_shot":
            msgs = build_messages_zero_shot(r["text"], r["topic"])
        else:
            msgs = build_messages_few_shot(r["text"], r["topic"], exemplars)
        tasks.append(classify_one(client, msgs, semaphore))
    results = await tqdm.gather(*tasks, desc=f"{mode} ({len(tasks)} examples)")
    return results

# -------------------- Metrics --------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    per_class = {
        str(i): {"precision": float(p[i]), "recall": float(r[i]),
                 "f1": float(f[i]), "support": int(s[i])}
        for i in range(NUM_CLASSES)
    }
    return {"accuracy": float(acc), "f1_macro": float(f1m),
            "f1_weighted": float(f1w), "per_class": per_class}

# -------------------- Leaderboard --------------------
LEADERBOARD_REFS = {
    "Baseline MiniLM (frozen+MLP)":         0.5465,
    "Para-MPNet (frozen+MLP, best frozen)": 0.5494,
    "RoBERTa fine-tune":                    0.5823,
    "DeBERTa-v3 fine-tune":                 0.5793,
    "Qwen3-Emb-8B frozen+MLP":              0.5855,
}

def print_leaderboard(this_run_name: str, this_f1: float):
    ranked = dict(LEADERBOARD_REFS)
    ranked[this_run_name] = this_f1
    print("\n=== Test macro F1 (with this run) ===")
    for name, val in sorted(ranked.items(), key=lambda kv: -kv[1]):
        marker = "  ←" if name == this_run_name else ""
        print(f"  {val:.4f}   {name}{marker}")

def verdict(mode: str, f1_macro: float, sustain_f1: float):
    print(f"\n--- Verdict ({mode}) ---")
    if f1_macro >= 0.58:
        print(f"🟢 Strong: beats frozen-encoder ceiling and matches fine-tune leaders ({f1_macro:.4f}).")
    elif f1_macro >= 0.55:
        print(f"🟡 Decent: exceeds frozen-encoder ceiling ({f1_macro:.4f}) but below fine-tunes.")
    elif f1_macro >= 0.45:
        print(f"🟠 Expected territory for prompt-only LLM on specialised classification ({f1_macro:.4f}). "
              "Narratively useful — quantifies the gap task-specific training closes.")
    else:
        print(f"🔴 Weak: {f1_macro:.4f}. Double-check prompt wording / label distribution.")
    print(f"Sustain F1: {sustain_f1:.4f}  (RoBERTa-ft sustain = 0.393 for reference)")

# -------------------- Runner --------------------
async def run_mode(mode: str, train_rows: list[dict], test_rows: list[dict]):
    print(f"\n{'='*60}\n  Running mode: {mode}\n{'='*60}")

    exemplars = None
    if mode == "few_shot":
        exemplars = sample_few_shot_pool(train_rows, FEW_SHOT_K, SEED)
        print(f"Few-shot exemplars (k={FEW_SHOT_K}/class = {len(exemplars)} total):")
        for i, ex in enumerate(exemplars, 1):
            preview = ex["text"][:70].replace("\n", " ")
            print(f"  [{i}] {ex['label']:>7}  \"{preview}...\"")

    out_dir = REPO / "data" / f"outputs_gpt4o_mini_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = await classify_all(test_rows, mode, exemplars)
    elapsed = time.time() - t0

    y_true = np.array([CLASS_TO_IDX[r["label"]]   for r in test_rows])
    y_pred = np.array([CLASS_TO_IDX[res["label"]] for res in results])

    n_errors = sum(1 for r in results if r["error"] is not None)
    tot_in   = sum(r["input_tokens"]  for r in results)
    tot_out  = sum(r["output_tokens"] for r in results)
    cost = tot_in * PRICE_IN_PER_1M / 1e6 + tot_out * PRICE_OUT_PER_1M / 1e6

    m = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    print(f"\nRun time: {elapsed:.1f}s  |  Errors (fell back to 'neutral'): {n_errors}")
    print(f"Tokens: {tot_in:,} in / {tot_out:,} out  |  Cost: ${cost:.4f}")
    print(f"\nAccuracy:    {m['accuracy']:.4f}")
    print(f"Macro F1:    {m['f1_macro']:.4f}")
    print(f"Weighted F1: {m['f1_weighted']:.4f}")
    print("\nPer-class:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print("          " + "  ".join(f"{c:>8}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:>8}  " + "  ".join(f"{cm[i, j]:>8d}" for j in range(NUM_CLASSES)))

    # Save metrics
    payload = {
        "model_name":  MODEL,
        "mode":        mode,
        "label_field": "metadata.client_talk_type",
        "class_names": CLASS_NAMES,
        "test":        m,
        "runtime_seconds": elapsed,
        "errors":      n_errors,
        "tokens":      {"input": tot_in, "output": tot_out},
        "cost_usd":    cost,
        "config": {
            "temperature": TEMPERATURE, "concurrency": CONCURRENCY,
            "seed": SEED, "few_shot_k_per_class": FEW_SHOT_K if mode == "few_shot" else None,
            "system_prompt_hash": hash(SYSTEM_PROMPT),
        },
        "few_shot_exemplars": [
            {"label": ex["label"], "topic": ex["topic"], "text": ex["text"]}
            for ex in (exemplars or [])
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(
        {"class_names": CLASS_NAMES, "matrix": cm.tolist()}, indent=2))

    # Save raw predictions for later inspection
    preds_out = []
    for row, res in zip(test_rows, results):
        preds_out.append({
            "text": row["text"], "topic": row["topic"],
            "true": row["label"], "pred": res["label"], "error": res["error"],
        })
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds_out) + "\n"
    )

    print(f"\nWrote: {out_dir}/metrics.json")
    print(f"Wrote: {out_dir}/confusion_matrix.json")
    print(f"Wrote: {out_dir}/predictions.jsonl")

    sustain_f1 = m["per_class"]["2"]["f1"]
    print_leaderboard(f"GPT-4o-mini ({mode})", m["f1_macro"])
    verdict(mode, m["f1_macro"], sustain_f1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"], default="both")
    args = parser.parse_args()

    # Load .env from repo root
    load_dotenv(REPO / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not found. Add it to .env at the repo root.")

    random.seed(SEED); np.random.seed(SEED)

    train_rows = load_split(TRAIN_JSONL)
    test_rows  = load_split(TEST_JSONL)
    print(f"Loaded train={len(train_rows)} test={len(test_rows)}")

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]
    for mode in modes:
        asyncio.run(run_mode(mode, train_rows, test_rows))

if __name__ == "__main__":
    main()
