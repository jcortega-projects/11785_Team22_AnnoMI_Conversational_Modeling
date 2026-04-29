# Stage 1 Report — Data Ingestion, Splits, Synthetic Generation, Baseline

This document is the **frozen** specification of everything that exists before
any modeling experiment runs. It is the input to `EXPERIMENTS_REPORT.md`
(Rounds 1–10, Stage 2) and a reference for the final writeup.

If something in this document changes, every Stage 2 experiment must be
re-run — which is why this is kept separate and rarely edited.

---

## 1. Task

Classify individual **client utterances** from motivational interviewing (MI)
therapy conversations into one of three talk-type labels:

| Label | Definition |
|---|---|
| `change` | Client speech moving *toward* behaviour change ("I want to quit smoking", "Yeah, that's a good idea") |
| `sustain` | Client speech resisting change ("I can't stop, it's too hard", "I don't want to get into that right now") |
| `neutral` | Off-topic or non-committal client speech ("Okay.", "I understand.") |

Therapist utterances are **not prediction targets**. They may optionally be
used as causal context (prior turns only; no future leak), but the default
configuration (`context_turns = 0`) uses the current client utterance alone.

**Primary metric**: macro F1 across the three classes (weights all classes
equally regardless of support). Secondary: accuracy, weighted F1, per-class
precision/recall/F1.

---

## 2. Dataset

**Source**: `AnnoMI/AnnoMI-simple.csv` — 9,699 utterance rows from 133
expert-annotated MI counselling sessions, mixing high-quality and low-quality
therapy. Public dataset from UCCollab.

### Columns used by the pipeline

| Column | Role |
|---|---|
| `transcript_id` | Conversation identifier — used as the splitting unit |
| `utterance_id` | Within-conversation turn order |
| `interlocutor` | `therapist` or `client` |
| `utterance_text` | Raw text of the utterance |
| `client_talk_type` | Label — `change` / `neutral` / `sustain` for client rows; NaN for therapist rows |
| `mi_quality` | Transcript-level quality label — used **only** for stratified splitting |
| `topic`, `video_title`, `video_url`, `timestamp` | Metadata carried through to processed examples |

### Raw label distribution (client utterances only)

| Label | Count | Share |
|---|---:|---:|
| neutral | 3,102 | 64.4% |
| change | 1,174 | 24.4% |
| sustain | 541 | 11.2% |
| **Total client** | **4,817** | 100% |

The 4,882 therapist rows have `client_talk_type = NaN` and are excluded from
the target set.

---

## 3. Pipeline Overview

```
AnnoMI-simple.csv
        │
        ▼
load + validate columns                       (src/annomi_pipeline/data/ingestion.py)
        │
        ▼
group by transcript_id, sort by utterance_id  → Conversation objects
        │
        ▼
transcript-level split 70/15/15               (stratified by mi_quality)
        │
        ▼
build client-utterance examples               (src/annomi_pipeline/data/chunking.py)
   ├── target = client utterances only
   ├── label  = client_talk_type ∈ {change, neutral, sustain}
   └── causal context (context_turns ∈ {0, 1, 2, 4})
        │
        ▼
write data/processed/{train,val,test}.jsonl
write data/processed/label_mapping.json
        │
        ▼
token statistics                               (src/annomi_pipeline/data/tokenizer.py)
        │
        ▼
TF-IDF + Logistic Regression baseline sweep    (src/annomi_pipeline/stage1/baseline.py)
        ▼
data/outputs/{baseline_results.json, baseline_results.png, stage1_summary.json}
```

**Driver**: `python -m annomi_pipeline.scripts.run_stage1 --config configs/data_config.yaml`

---

## 4. Splitting Methodology

### Transcript-level splitting — no leakage

The 133 transcripts are split as **whole units** into 70/15/15 train/val/test,
stratified by `mi_quality`. Every utterance from a given conversation is
assigned to exactly one split. This prevents the model from learning
conversation-specific idiosyncrasies and recognising test transcripts via
sibling training utterances.

Seed = 42 everywhere (`src/annomi_pipeline/utils/seed.py:set_global_seed`).

### Resulting splits

| Split | Transcripts | Client utterances |
|---|---:|---:|
| train | 93 | 3,153 |
| val | 20 | 798 |
| test | 20 | 866 |

### Per-split label distribution

| Split | change | neutral | sustain | sustain share |
|---|---:|---:|---:|---:|
| train | 729 (23.1%) | 2,091 (66.3%) | 333 (10.6%) | 10.6% |
| val | 227 (28.4%)\* | 440 (55.1%) | 131 (16.4%) | 16.4% |
| test | 218 (25.2%) | 571 (65.9%) | 77 (8.9%) | 8.9% |

> \* **Note on val label distribution**: the 798-example val split has a
> heavier sustain prevalence (16.4%) than either train (10.6%) or test
> (8.9%). This has downstream consequences in Stage 2 — see
> `EXPERIMENTS_REPORT.md` § 6.9 on why augmentation improved val F1 but
> regressed test F1 (the val composition rewarded over-prediction of
> sustain).

---

## 5. Example Construction

### Utterance-level prediction unit

Each example corresponds to a **single client utterance**. This matches the
labeling granularity (`client_talk_type` is a per-utterance annotation) and
avoids the information loss of majority-vote labeling across multi-turn
windows.

### Causal-only optional context

A `context_turns` knob in `configs/data_config.yaml` can prefix the target
utterance with up to N preceding turns (therapist or client). Future turns
are never included (enforced by `end_utterance == utterance_id` invariant).

Default = 0 (current client utterance only) for the cleanest signal.
Context sweeps (1, 2, 4) were tested in Stage 2 Round 3 and regressed
monotonically — see `EXPERIMENTS_REPORT.md` § 5.

### Stable label mapping

Stage 1 writes `data/processed/label_mapping.json`, reused verbatim by every
downstream stage:

```json
{
  "label_to_id": {"change": 0, "neutral": 1, "sustain": 2},
  "id_to_label": {"0": "change", "1": "neutral", "2": "sustain"}
}
```

---

## 6. Token Statistics

Whitespace-tokenized, on the `context_turns = 0` configuration:

| Split | Count | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|
| train | 3,153 | ~22 | ~17 | ~55 | 200+ |
| val | 798 | ~22 | ~17 | ~55 | — |
| test | 866 | ~22 | ~17 | ~55 | — |

(Switch `tokenizer.type: huggingface` in `configs/data_config.yaml` to get
DistilBERT-tokenizer counts; rough 1.3× inflation.)

**Implication for Stage 2**: a `max_length = 128` token window covers p99+
of the utterance corpus, leaving comfortable headroom for any context-turn
prefix up to ctx1. Longer context (ctx2/ctx4) exceeds 128 tokens in the
tail, but Stage 2 experiments confirmed longer context *hurts* on this task
(encoders built for single-sentence semantics).

---

## 7. Synthetic Data Generation

### Purpose

Tested in Stage 2 Rounds 1–2 (frozen encoders) and Round 9 (LoRA fine-tune)
as a minority-class augmentation strategy. **Spoiler: it did not help in any
regime** — see `EXPERIMENTS_REPORT.md` § 8.8. Documented here for
reproducibility.

### Method

- **Generator**: OpenAI `o4-mini` via `src/annomi_pipeline/data/synthetic_generation.py`
- **Strategy**: label-preserving paraphrase with real-training-set seeds
- **Batch size**: 5 utterances per API call, distinct seed per call
- **Labels generated**: `sustain` (volume sweep: 200 / 400 / 800) and `change` (+300 fixed)
- **QA**: acceptance rate = 1,100 / 1,100 candidates (100%) — QA gates on
  label prefix, length, and stop-word presence
- **Prompt template**: system message with the canonical MI label
  definitions (identical definitions later reused in the GPT-4o-mini
  classifier, Round 10)

### Augmented training-set compositions

All written to `data/processed/train_augmented.jsonl` and related files:

| Config | change | neutral | sustain | Total | sustain share |
|---|---:|---:|---:|---:|---:|
| Real baseline | 729 | 2,091 | 333 | 3,153 | 10.6% |
| +s200 | 1,029 | 2,091 | 533 | 3,653 | 14.6% |
| +s400 | 1,029 | 2,091 | 733 | 3,853 | 19.0% |
| +s800 | 1,029 | 2,091 | 1,133 | 4,253 | 26.6% |

> **Distributional mismatch note**: s800 inflates sustain to 26.6% of
> training — 2.5× the real prevalence and 3.0× the test prevalence.
> Stage 2 Round 9 diagnosed this as the source of the negative result:
> the LLM-paraphrase distribution does not match AnnoMI sustain style, and
> any decision boundary shift toward sustain rewards the val split
> (16.4% sustain) at the cost of the test split (8.9% sustain).

### Artefacts

```
data/outputs/augmentation/
    sustain_s200.jsonl       # 200 synthetic sustain utterances
    sustain_s400.jsonl       # 400 synthetic sustain utterances
    sustain_s800.jsonl       # 800 synthetic sustain utterances
    change_s300.jsonl        # 300 synthetic change utterances
    generation_log.jsonl     # per-call token / cost / seed log
data/processed/
    train_augmented.jsonl    # 4,253 examples (baseline + s800 + c300)
```

---

## 8. Stage 1 Baseline — TF-IDF + Logistic Regression

### Purpose

Establish the floor that any neural model must beat to justify additional
complexity.

### Method

Sweep over `context_turns_sweep: [0, 2, 4]` × `vocab_sizes: [1024, 2048, 4096]`.
For each combination:

1. Rebuild utterance examples at that context length
2. Fit `TfidfVectorizer(lowercase=True, ngram_range=(1, 2))` on training texts
3. Train `LogisticRegression(class_weight="balanced", max_iter=2000)`
4. Compute macro F1 on val (for selection) and test (for reporting)

Test is **never** inspected for selection — selection is val-only.

### Results

**Best configuration**: `context_turns = 0, max_features = 2048, ngram (1,2)`
**Best metrics**: val macro F1 = **0.4619** | test macro F1 = **0.5078**

This is the reference floor for Stage 2. Every neural experiment in
`EXPERIMENTS_REPORT.md` is compared against both this (the classical
baseline) and the frozen MiniLM + MLP baseline (the Stage 2 floor at 0.5465).

---

## 9. Stage 1 Outputs on Disk

```
data/processed/
    train.jsonl                  # 3,153 client-utterance examples
    val.jsonl                    # 798
    test.jsonl                   # 866
    train_augmented.jsonl        # 4,253 (baseline + s800 + c300)
    label_mapping.json           # change/neutral/sustain → 0/1/2

data/processed_ctx1/, processed_ctx2/, processed_ctx4/
    (same layout, with preceding turns prepended to each utterance)

data/outputs/
    baseline_results.json        # full sweep grid + best result
    baseline_results.png         # macro F1 vs context_turns, one line per vocab
    token_stats.json             # length statistics
    stage1_summary.json          # top-level manifest
    augmentation/                # synthetic generation artefacts (§ 7)
```

---

## 10. Validation Invariants

Enforced at runtime and re-checked after every full Stage 1 run:

- Target rows are **only** client utterances (`metadata.speaker == "client"`).
- Labels are exactly the set `{change, neutral, sustain}`. NaN, `n/a`, and any
  unexpected value are filtered before any example is written.
- Train, val, and test `transcript_id` sets are **pairwise disjoint**.
- Every example has `end_utterance == utterance_id` — no future context leak.
- `data/processed/label_mapping.json` exists and is consistent across all
  downstream stages (Stage 2, embedding generation, LoRA notebook,
  GPT-4o-mini classifier).
- Stage 2 embedding `.npy` row counts match the corresponding JSONL record
  counts for every split (asserted by `scripts/run_embeddings.py`).
- Unit tests pass: `pytest tests/` covers ingestion, chunking, stage-1
  baseline, dataset, tokenizer, and labeling-policy invariants
  (7 test modules in `tests/`).

---

## 11. Configuration Surface

Three YAML files fully control Stage 1. Editing them is the only way to
change task formulation, splitting, context length, or baseline sweep.

| Config | Owns |
|---|---|
| `configs/data_config.yaml` | CSV column mapping, task target/labels, context turns, train/val/test split ratios, baseline sweep grid |
| `configs/embeddings_config.yaml` | (Stage 2) Embedding backend and dim — shown here because Stage 1 writes the JSONL that Stage 2 reads |
| `configs/train_config*.yaml` | (Stage 2) MLP architecture, optimizer, scheduler, early stopping, class weighting |

Per-experiment configs (`train_config_aug_s200.yaml`,
`train_config_roberta_finetune.yaml`, etc.) inherit from the base and
override specific fields. See `EXPERIMENTS_REPORT.md` § 10 for the full
mapping of config → output directory.

---

## 12. Reproducing Stage 1 End-to-End

```bash
# from the project root
python -m annomi_pipeline.scripts.run_stage1 --config configs/data_config.yaml
```

Produces all files listed in § 9. Runs in < 2 minutes on any laptop. No GPU
needed.

To regenerate synthetic data:

```bash
python -m annomi_pipeline.scripts.run_synthetic \
    --config configs/synthetic_config.yaml \
    --api-key-env OPENAI_API_KEY
```

(Requires `OPENAI_API_KEY` environment variable; ~$1–2 total cost for the
1,100-utterance sweep.)

---

## 13. Known Limitations of Stage 1

These are the Stage-1-side caveats a reader of the final report needs to
know to interpret Stage 2 results correctly:

1. **Single-split point estimates.** With only 20 transcripts in test, split
   variance alone is ±0.05 macro F1. A transcript-grouped k-fold CV would
   turn point estimates into confidence intervals. Not implemented due to
   compute/time budget (every fold = rerun of every Stage 2 experiment).

2. **Val composition is not representative of test.** Val has 16.4% sustain;
   test has 8.9%. Any model whose decision boundary responds to sustain
   prevalence (e.g. augmented models with shifted priors) will show
   *misleading* val F1 trajectories. Mitigation: in Stage 2 we report val
   F1 for checkpoint selection only, and use the val-test gap as a
   distribution-shift diagnostic (see `EXPERIMENTS_REPORT.md` § 6.9).

3. **LLM-generated synthetic sustain has distributional drift from real
   AnnoMI sustain.** Diagnosed across three modeling regimes (frozen, loss
   ablation, LoRA) and consistently hurts test F1. Not a Stage 1 bug — the
   generation protocol is correct; the underlying LLM just does not
   reproduce MI pragmatic patterns. Synthetic data is kept on disk for
   reproducibility of the negative finding.

4. **Transcript-level stratification is only on `mi_quality`, not on topic
   or label balance.** If topic shift dominates the hard cases, this could
   bias specific splits. Stage 2 did not diagnose strong topic-shift
   effects, but a second stratification axis (e.g. topic bucket) would be
   a defensible refinement.

---

## 14. Pointer to Stage 2

All experiments that consume these splits, embeddings, and augmented data
are documented in `EXPERIMENTS_REPORT.md`, Rounds 1–10:

- Rounds 1–2: frozen MiniLM + MLP, synthetic augmentation sweep
- Round 3: context-turns sweep (ctx0 → ctx4)
- Round 4: encoder swap (MPNet, Para-MPNet, RoBERTa, DeBERTa — all frozen)
- Round 5: end-to-end fine-tuning (RoBERTa, DeBERTa-v3)
- Round 6: ModernBERT end-to-end fine-tune (rejected)
- Round 7: Qwen3-Embedding-8B frozen + MLP (TA Change I Phase 1)
- Round 8: **Qwen3-Embedding-8B + LoRA fine-tune (final #1, test F1 = 0.6131)**
- Round 9: LoRA + s800 augmentation (negative result, resolves § 7 question)
- Round 10: GPT-4o-mini zero-shot and few-shot (TA Change II)
