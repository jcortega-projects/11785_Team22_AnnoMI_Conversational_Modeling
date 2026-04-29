"""Synthetic client-utterance generation for the AnnoMI augmentation pipeline.

This module handles everything between a real training example (seed) and a
candidate synthetic row ready for QA:

  1. Prompt construction (label-specific, context-aware)
  2. OpenAI API calls with retry/backoff
  3. Output parsing and quality gating
  4. Schema assembly for the output JSONL

Design decisions
----------------
text vs client_text
    On synthetic rows both `text` and `client_text` are set to the generated
    utterance.  The original `text` field is intentionally kept ambiguous in
    the seed schema (it could be a chunk or a single utterance).  For purely
    synthetic rows there is no chunk ambiguity, so the two fields should
    be identical.  Downstream code that already reads `client_text` will
    continue to work; code that reads `text` will also see the utterance.

context truncation
    The full `context` string from the seed can be very long.  We truncate to
    the last MAX_CONTEXT_LINES lines so the prompt stays focused and cheap.

Batching
    Each API call asks for BATCH_SIZE utterances from one seed.  This keeps
    diversity high (each call sees a unique seed) while reducing the number
    of round trips.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 5          # utterances requested per API call
MAX_CONTEXT_LINES: int = 6   # last N lines of context string kept in prompt
MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0    # seconds; doubles on each retry
BUFFER_FACTOR: float = 1.25  # generate this multiple of target, then trim

ALLOWED_LABELS: frozenset[str] = frozenset({"sustain", "change"})

# Patterns that indicate a therapist or artifact line slipped into output
THERAPIST_PREFIXES: tuple[str, ...] = (
    "therapist:",
    "counselor:",
    "doctor:",
    "provider:",
    "t:",
    "th:",
)
ARTIFACT_PATTERNS: re.Pattern = re.compile(
    r"^\s*(\d+[\.\)]\s|[-•]\s|client:\s|\"|\[|\{)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SHARED_RULES = """\
Rules:
- Write in first person, natural conversational speech — not clinical or formal.
- Each utterance is 1–3 sentences.
- Same topic as the source: {topic}.
- Do NOT copy the source utterance verbatim.
- Do NOT include therapist lines.
- Do NOT include numbering, bullet points, labels, explanations, or quotes.
- Vary sentence structure and vocabulary across the 5 utterances.
- Sound like spoken language from a real counseling session.
- Keep utterances plausible, not too polished, not too long."""

_SUSTAIN_INSTRUCTION = """\
Generate {n} diverse CLIENT utterances that are "sustain talk" — the client is
defending the status quo, expressing reasons NOT to change, or resisting the
idea of changing their behavior.  Stay semantically close enough to the source
that the "sustain" label is unambiguous
Avoid hedging that weakens the label strength.
Example: do not convert strong sustain talk into ambivalent talk.
"""

_CHANGE_INSTRUCTION = """\
Generate {n} diverse CLIENT utterances that are "change talk" — the client is
expressing desire, ability, reason, or commitment to change their behavior.
Stay semantically close enough to the source that the "change" label is
unambiguous.  Do NOT generate sustain talk or ambiguous statements."""

_PROMPT_TEMPLATE = """\
You are producing training data for a Motivational Interviewing (MI) research
classifier.  The task is to classify individual CLIENT utterances into one of
three categories: change talk, sustain talk, or neutral.

{label_instruction}

Context (last few turns before the source utterance):
---
{context_snippet}
---

Source utterance ({label} talk):
"{client_text}"

{rules}

Return ONLY a JSON array of exactly {n} strings — one per utterance.
No other text, no markdown, no explanation.
["utterance 1", "utterance 2", ...]"""


def build_prompt(seed: dict[str, Any], n: int = BATCH_SIZE) -> str:
    """Construct the generation prompt for a single seed example."""
    label: str = seed["label"]
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported label for generation: {label!r}")

    instruction = (
        _SUSTAIN_INSTRUCTION if label == "sustain" else _CHANGE_INSTRUCTION
    ).format(n=n)

    rules = _SHARED_RULES.format(topic=seed.get("topic") or "the topic discussed")
    context_snippet = _truncate_context(seed.get("context") or "")

    return _PROMPT_TEMPLATE.format(
        label_instruction=instruction,
        context_snippet=context_snippet or "(no prior context available)",
        label=label,
        client_text=seed["client_text"],
        rules=rules,
        n=n,
    )


def _truncate_context(context: str) -> str:
    """Keep only the last MAX_CONTEXT_LINES non-empty lines of context."""
    lines = [l for l in context.splitlines() if l.strip()]
    return "\n".join(lines[-MAX_CONTEXT_LINES:])


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


def call_openai_with_retry(
    client: Any,
    model: str,
    prompt: str,
    seed_id: str,
) -> list[str] | None:
    """
    Call the OpenAI chat completions endpoint.

    Returns a list of raw string candidates on success, None if all retries
    fail.  The caller is responsible for quality filtering.
    """
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # o4-mini is a reasoning model — do NOT pass temperature.
                # max_completion_tokens covers both reasoning and output tokens.
                max_completion_tokens=1024,
            )
            raw_text = response.choices[0].message.content or ""
            candidates = _parse_json_array(raw_text)
            if candidates is not None:
                return candidates

            LOGGER.warning(
                "seed=%s attempt=%d: JSON parse failed; raw=%r",
                seed_id, attempt, raw_text[:200],
            )

        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "seed=%s attempt=%d: API error: %s", seed_id, attempt, exc
            )

        if attempt < MAX_RETRIES:
            sleep_seconds = BACKOFF_BASE ** attempt
            LOGGER.debug("Sleeping %.1fs before retry …", sleep_seconds)
            time.sleep(sleep_seconds)

    LOGGER.error("seed=%s: all %d retries exhausted", seed_id, MAX_RETRIES)
    return None


def _parse_json_array(text: str) -> list[str] | None:
    """
    Extract a JSON array of strings from the model output.

    The model sometimes wraps the array in markdown code fences or adds a
    leading sentence.  We strip those before parsing.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    # Find the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start : end + 1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    # Keep only string items
    return [item for item in parsed if isinstance(item, str)]


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def quality_filter(
    candidate: str,
    seed_client_text: str,
) -> tuple[bool, str | None]:
    """
    Return (accepted, rejection_reason).

    Checks (in order):
      1. Empty output
      2. Exact duplicate of the source utterance
      3. Looks like a therapist line
      4. Formatting artifact (numbering, role prefix, stray bracket)
    """
    text = candidate.strip()

    if not text:
        return False, "empty_output"

    if text.lower() == seed_client_text.strip().lower():
        return False, "exact_duplicate_of_source"

    lower = text.lower()
    for prefix in THERAPIST_PREFIXES:
        if lower.startswith(prefix):
            return False, "therapist_speech_detected"

    if ARTIFACT_PATTERNS.match(text):
        return False, "formatting_artifact"

    return True, None


# ---------------------------------------------------------------------------
# Schema assembly
# ---------------------------------------------------------------------------


@dataclass
class SyntheticRow:
    """One output row in synthetic_candidates.jsonl."""

    # Provenance
    example_id: str
    original_example_id: str
    synthetic_candidate_id: str
    split: str
    source_type: str
    augmentation_method: str
    augmentation_source: str
    generator_model: str

    # Utterance identity (carried from seed)
    transcript_id: Any
    utterance_id: Any
    topic: str | None
    mi_quality: str | None

    # Text
    # Both `text` and `client_text` are set to the generated utterance.
    # For real examples `text` may differ from `client_text` when a row
    # was built from a multi-utterance chunk.  For synthetic rows there is
    # no chunk concept — the generated utterance IS the unit, so the fields
    # are identical.  Code reading `client_text` continues to work unchanged.
    text: str
    client_text: str

    # Label
    label: str
    metadata: dict[str, Any]

    # Context (carried from seed, unchanged)
    context: str | None
    prior_turns: list[dict[str, Any]]

    # QA placeholders — filled by the downstream verification pass, not here.
    #
    # Recommended QA prompt (auto-label consistency check):
    #   "A client says: '{client_text}'
    #    Which label best fits this utterance?
    #    Return exactly one of: change | neutral | sustain"
    #
    # Compare the response against `label`.  If they disagree, set
    # verification_status="auto_rejected" and rejection_reason="label_mismatch".
    # If they agree, set verification_status="verified" and
    # verification_label=<returned label>, accepted_for_training=True.
    #
    # seed_transcript_id in metadata enables grouping QA failures by transcript
    # to detect systemic generation drift on specific transcripts/topics.
    verification_status: str = "unverified"
    verification_label: Any = None
    accepted_for_training: bool = False
    rejection_reason: Any = None

    # Schema fields present in seed but not meaningful for synthetic rows
    chunk_id: Any = None
    start_utterance: Any = None
    end_utterance: Any = None
    turn_count: int = 1
    stored_context_turns: int = 0
    source_dataset: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "original_example_id": self.original_example_id,
            "synthetic_candidate_id": self.synthetic_candidate_id,
            "split": self.split,
            "source_type": self.source_type,
            "augmentation_method": self.augmentation_method,
            "augmentation_source": self.augmentation_source,
            "source_dataset": self.source_dataset,
            "generator_model": self.generator_model,
            "transcript_id": self.transcript_id,
            "utterance_id": self.utterance_id,
            "chunk_id": self.chunk_id,
            "start_utterance": self.start_utterance,
            "end_utterance": self.end_utterance,
            "text": self.text,
            "client_text": self.client_text,
            "label": self.label,
            "turn_count": self.turn_count,
            "topic": self.topic,
            "mi_quality": self.mi_quality,
            "stored_context_turns": self.stored_context_turns,
            "context": self.context,
            "prior_turns": self.prior_turns,
            "metadata": self.metadata,
            "verification_status": self.verification_status,
            "verification_label": self.verification_label,
            "accepted_for_training": self.accepted_for_training,
            "rejection_reason": self.rejection_reason,
        }


def build_synthetic_row(
    generated_text: str,
    seed: dict[str, Any],
    label: str,
    model: str,
    counter: int,
) -> SyntheticRow:
    """Assemble one output row from a generated candidate + its seed."""
    candidate_id = f"syn_{label}_{counter:04d}"
    example_id = f"synthetic_{uuid.uuid4().hex[:8]}"

    return SyntheticRow(
        example_id=example_id,
        original_example_id=seed["example_id"],
        synthetic_candidate_id=candidate_id,
        split="train",          # all candidates are train-only by design
        source_type="synthetic",
        augmentation_method="llm_label_preserving_paraphrase",
        augmentation_source=seed["example_id"],
        generator_model=model,
        transcript_id=seed.get("transcript_id"),
        utterance_id=seed.get("utterance_id"),
        topic=seed.get("topic"),
        mi_quality=seed.get("mi_quality"),
        text=generated_text,        # see docstring: text == client_text for synthetic
        client_text=generated_text,
        label=label,
        metadata={
            "client_talk_type": label,          # mirrors label; kept in sync
            "seed_example_id": seed["example_id"],
            "seed_transcript_id": seed.get("transcript_id"),  # for transcript-level QA grouping
        },
        context=seed.get("context"),
        prior_turns=seed.get("prior_turns") or [],
    )


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def select_seeds(
    pool: list[dict[str, Any]],
    n_calls: int,
    rng_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select `n_calls` distinct seeds from `pool`.

    Uses a fixed RNG seed for reproducibility.  If n_calls > len(pool),
    cycles through the pool (but logs a warning since seed reuse lowers
    diversity).
    """
    import random

    rng = random.Random(rng_seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    if n_calls <= len(shuffled):
        return shuffled[:n_calls]

    # Cycle: only if pool is smaller than demand
    LOGGER.warning(
        "n_calls=%d > pool_size=%d; some seeds will be reused.", n_calls, len(pool)
    )
    result: list[dict[str, Any]] = []
    while len(result) < n_calls:
        result.extend(shuffled)
    return result[:n_calls]
