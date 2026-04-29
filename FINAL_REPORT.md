# Final Report — AnnoMI Client Talk-Type Classification

**Team**: <!-- TODO: your names + Andrew IDs -->
**Course**: 11-785 Intro to Deep Learning, Spring 2026
**Date**: <!-- TODO: submission date -->

> **How to use this skeleton.** Each section is a stub with (a) a one-line
> statement of what belongs there, (b) a bulleted checklist of content to
> include, and (c) pointers to where in `STAGE1_REPORT.md` or
> `EXPERIMENTS_REPORT.md` the source material lives. Fill in prose; don't
> re-derive numbers — copy them from the source docs and cite the section.
> Target length: **8–12 pages** (or whatever the course rubric specifies).

---

## Abstract

*(≤ 200 words. Read last, after the rest is written.)*

**What to include**:
- One sentence: the task (MI client talk-type classification on AnnoMI).
- One sentence: the dataset (4,817 client utterances across 133 transcripts,
  3-way imbalanced: 64% neutral / 24% change / 11% sustain).
- Two–three sentences: the experimental arc — frozen baselines →
  fine-tuning breaks a ~0.55 ceiling → an 8B retrieval-pretrained encoder
  with LoRA lifts it to **0.6131** → a frontier LLM (GPT-4o-mini) reaches
  within 0.011 F1 using no training.
- One sentence: the headline finding and its implication
  (representation-side scale + lightweight adapter fine-tune beats every
  mid-size fine-tune; prompted frontier LLM is competitive but deployment
  constraints favour the open fine-tune).

**Source**: `EXPERIMENTS_REPORT.md` § 7 (leaderboard) and § 7.5 (best model).

---

## 1. Introduction

*(~1 page. Context, motivation, contribution.)*

**What to include**:
- **Problem framing**: why MI, why talk-type classification, clinical
  relevance of `sustain` detection.
- **Why it's hard**: severe class imbalance, short utterances (median ~17
  tokens), pragmatic distinctions that surface-level semantics miss.
- **Contribution**: five bullet points — the frozen-ceiling characterisation,
  the end-to-end fine-tune breakthrough, the scale result, the LoRA result,
  the frontier-LLM comparison.
- **Paper organization**: one-sentence-per-section roadmap.

**Sources**:
- Task framing: `STAGE1_REPORT.md` § 1
- Class imbalance numbers: `STAGE1_REPORT.md` § 2, § 4
- Contribution arc: `EXPERIMENTS_REPORT.md` § 8.6 (three-tier separation)

---

## 2. Related Work

*(~0.5 page. Keep tight.)*

**What to include**:
- **Motivational Interviewing classification**: cite the original AnnoMI
  dataset paper; mention prior work on MI talk-type tagging if any.
- **Text classification with class imbalance**: class-weighted CE, focal
  loss, minority-class oversampling (acknowledge the frozen-MiniLM
  literature).
- **Representation learning at scale**: sentence-transformers (SBERT),
  retrieval-pretrained embedding models (Qwen3-Embedding, E5, BGE).
- **Parameter-efficient fine-tuning**: LoRA (Hu et al. 2021), with a
  sentence on why it's the right tool at 8B scale on 24GB GPUs.
- **LLM-as-classifier**: prior work on prompting frontier LLMs for
  classification tasks, with structured JSON outputs.

**Sources**: your own literature review; EXPERIMENTS_REPORT cites none of
these beyond model names.

---

## 3. Data & Task

*(~1 page. Condensed summary of Stage 1.)*

### 3.1 AnnoMI dataset

- 9,699 utterances, 133 transcripts, mixed MI-quality therapy sessions.
- Filter to client-only rows with valid labels → **4,817 client utterances**.
- Label distribution: 64.4% neutral / 24.4% change / 11.2% sustain.

### 3.2 Splits

One table: 93/20/20 transcripts → 3,153/798/866 utterances. Transcript-level
stratified split on `mi_quality`. **Call out** the val sustain prevalence
(16.4%) vs train (10.6%) vs test (8.9%) — it matters for results in § 5.3.

### 3.3 Evaluation protocol

- **Primary metric**: macro F1 (equal weight on all three classes).
- **Selection**: on val macro F1; test never seen during model selection.
- **Seeding**: seed=42 globally across Python/NumPy/PyTorch.

### 3.4 Synthetic data (brief)

One paragraph: we also generated 1,100 label-preserving paraphrase
utterances with o4-mini for minority-class augmentation. Tested in §§ 5.2
and 5.4; negative result in both regimes.

**Sources**:
- § 3.1: `STAGE1_REPORT.md` § 2
- § 3.2: `STAGE1_REPORT.md` § 4
- § 3.3: `STAGE1_REPORT.md` § 1
- § 3.4: `STAGE1_REPORT.md` § 7

---

## 4. Methods

*(~2 pages. Describe every paradigm represented in the leaderboard.)*

### 4.1 Baseline — TF-IDF + Logistic Regression

One paragraph. TF-IDF (ngram 1–2, vocab 2048) + class-balanced LogReg.
Reference floor: val F1 = 0.4619, test F1 = 0.5078.

**Source**: `STAGE1_REPORT.md` § 8.

### 4.2 Frozen encoder + MLP (Rounds 1–4, 7)

- Architecture diagram: encoder (frozen, no gradient) → pooled vector →
  MLP(hidden=128, layers=2, dropout=0.4) → 3-way softmax.
- Five encoders tested: MiniLM, MPNet, Para-MPNet, RoBERTa-base (MLM),
  DeBERTa-v3-base (MLM), Qwen3-Embedding-8B.
- Class-weighted CE, AdamW(lr=1e-3, wd=1e-3), ReduceLROnPlateau patience=2,
  early stopping patience=6.

**Source**: `EXPERIMENTS_REPORT.md` § 2, § 6, § 6.7, footer.

### 4.3 End-to-end fine-tune (Round 5)

- `AutoModelForSequenceClassification`, full parameter update.
- HPs: lr=2e-5, batch=16, max_length=128, epochs=4, linear warmup (10%),
  weight_decay=0.01, grad_clip=1.0, class-weighted CE.
- Two architectures: RoBERTa-base (125M), DeBERTa-v3-base (184M).

**Source**: `EXPERIMENTS_REPORT.md` § 6.5, footer.

### 4.4 LoRA fine-tune on Qwen-8B (Round 8 — best system)

- Base: Qwen3-Embedding-8B in fp16, frozen.
- LoRA adapters on attention projections only (q/k/v/o), r=16, α=32,
  dropout=0.05, task_type=FEATURE_EXTRACTION.
- Custom classifier: last-token pool on left-padded sequences → fp32
  Linear(4096, 3).
- Effective batch 16 (batch=2 × grad_accum=8), lr=2e-4, warmup=10%,
  3 epochs, best-by-val-F1 checkpointing.
- Gradient checkpointing + `enable_input_require_grads()` required to fit
  on A10G 24 GB.

**Source**: `EXPERIMENTS_REPORT.md` § 6.8, § 7.5, footer.

### 4.5 Frontier LLM prompting (Round 10)

- `gpt-4o-mini` via OpenAI API, temperature=0.0, structured JSON schema
  output (`enum = [change, neutral, sustain]`, strict=true).
- Zero-shot: system prompt = canonical MI definitions.
- Few-shot: same system + 9 deterministic exemplars (3 per class, seed=42).
- Async concurrency = 10, 866 test utterances, < $0.10 per run.

**Source**: `EXPERIMENTS_REPORT.md` § 6.10, footer.

### 4.6 Augmentation (Rounds 1, 9 — tested twice, negative both times)

Brief paragraph with pointer to § 5.4 for results.

**Source**: `STAGE1_REPORT.md` § 7; `EXPERIMENTS_REPORT.md` § 3, § 6.9, § 8.8.

---

## 5. Results

*(~2–3 pages. The core of the report.)*

### 5.1 Headline leaderboard

**Table**: top-6 configs ranked by test macro F1. Copy from
`EXPERIMENTS_REPORT.md` § 7:

| Rank | System | Training | Test F1 | Accuracy | Sustain F1 |
|---|---|---|---|---|---|
| 🥇 1 | Qwen-8B + LoRA | LoRA r=16 | **0.6131** | **73.1%** | **0.431** |
| 🥈 2 | GPT-4o-mini few-shot | Prompt (9 exemplars) | 0.6019 | 69.6% | 0.388 |
| 🥉 3 | GPT-4o-mini zero-shot | Prompt | 0.5898 | 73.0% | 0.427 |
| 4 | Qwen-8B frozen + MLP | Frozen | 0.5855 | 69.5% | 0.379 |
| 5 | RoBERTa fine-tune | End-to-end | 0.5823 | 67.6% | 0.393 |
| 6 | DeBERTa-v3 fine-tune | End-to-end | 0.5793 | 68.6% | 0.374 |

*(Full 22-row leaderboard in `EXPERIMENTS_REPORT.md` § 7.)*

### 5.2 Narrative arc

Three paragraphs, each anchoring on a leaderboard region:

1. **Frozen ceiling** (~0.55): all 13 frozen experiments cluster in 0.45–0.55.
   Pretraining objective (sentence similarity > raw MLM) matters; context
   length hurts; augmentation hurts. See `EXPERIMENTS_REPORT.md` § 8.5.
2. **Fine-tuning breakthrough** (~0.58): RoBERTa and DeBERTa end-to-end
   land within 0.003 F1 of each other at ~0.58. Sustain F1 jumps from 0.35
   (frozen) to 0.39. Reference: `EXPERIMENTS_REPORT.md` § 6.5.
3. **Scale + LoRA** (0.61): Qwen-8B frozen already beats every end-to-end
   125–184M fine-tune on majority-class metrics. LoRA on top delivers
   +0.028 test F1 and the first sustain F1 above 0.40.
   Reference: `EXPERIMENTS_REPORT.md` § 6.7, § 6.8, § 7.5.

### 5.3 Best model deep-dive

Dedicated subsection for **Qwen-8B + LoRA** (the #1 system):
- Full HP table (copy from `EXPERIMENTS_REPORT.md` § 7.5).
- Training trajectory showing 1-epoch overfit, best-epoch=1.
- Confusion matrix (needs to be downloaded from the AWS run —
  `data/outputs_qwen8b_lora/metrics.json`).
- Per-class precision/recall/F1.
- Comparison table vs the four prior leaders
  (`EXPERIMENTS_REPORT.md` § 7.5 has it).

### 5.4 Ablations & negative results

Three subsections, each a paragraph + a mini-table:

- **Context-turn sweep regresses monotonically** — § 5 of experiments.
- **Augmentation hurts in three regimes** (frozen, frozen+no-weights,
  LoRA) — § 8.8. Call out the val-test gap collapse diagnostic from § 6.9.
- **ModernBERT rejected** — § 6.6. HP-sensitive at 3k examples, never
  matches RoBERTa/DeBERTa.

### 5.5 Frontier-LLM comparison

Standalone subsection for GPT-4o-mini (the TA Change II deliverable):
- Zero-shot vs few-shot table with per-class F1.
- Cost and latency (<$0.10, ~50 s per 866 utterances).
- Narrative framing: closed frontier LLM reaches 0.6019, within 0.011 of
  the open fine-tune. Discuss deployment tradeoffs (PHI, closed weights)
  that favour the open system for this specific use case.

**Source**: `EXPERIMENTS_REPORT.md` § 6.10, § 9.4a.

---

## 6. Discussion

*(~1 page. Three findings, each a paragraph.)*

### 6.1 Three tiers, shrinking margins

The 22 experiments separate into three regimes (§ 8.6 of experiments):
Tier 1 (LoRA, 0.613), Tier 2 (ft / scale / GPT, 0.58–0.60), Tier 3
(frozen + aug / context, 0.45–0.55). The gap from Tier 2 to Tier 1 is
+0.011 — narrower than Tier 3 → Tier 2 (+0.03–0.05). Further gains
likely require more data or task reframing, not bigger models.

### 6.2 Sustain F1 is a representation problem, not a data problem

Sustain F1 > 0.40 was achieved only by (a) LoRA on a pretrained 8B base
and (b) frontier LLM zero-shot. Both carry pragmatic pretraining signal.
Mid-size retrieval/MLM encoders, even end-to-end fine-tuned, cap at 0.393
(§ 8.7 of experiments). Class-weighted CE and loss engineering do not
change this — augmentation across three modeling regimes could not break
the ceiling either.

### 6.3 Augmentation from paraphrase LLMs is not a substitute for real minority data

Three independent tests (frozen Round 1, ± class-weights Round 2, LoRA
Round 9) all regressed test F1. The LLM-generated sustain distribution is
observably different from real AnnoMI sustain — diagnosable via val-test
gap collapse (Round 9: gap fell from +0.064 to +0.017 when augmented).
The finding generalises: post-hoc synthetic oversampling should be
validated against a held-out split with realistic class prior *before*
being recommended as a minority-class fix.

**Source**: `EXPERIMENTS_REPORT.md` § 8.6–8.8, § 9.3.

---

## 7. Response to TA Feedback

*(~0.5 page. Makes grading explicit.)*

The TA feedback requested three experimental changes in priority order:

| Change | Priority | Action taken | Result |
|---|---|---|---|
| **I**: test fundamentally different encoder class | HIGH | ModernBERT (rejected) + Qwen3-Embedding-8B frozen + LoRA fine-tune | Qwen-8B + LoRA = new #1 at **0.6131** |
| **II**: prompt-based LLM classification | MEDIUM | GPT-4o-mini zero-shot + few-shot (k=3) | **0.5898 / 0.6019** — within 0.011 F1 of Tier 1 |
| **III**: MLP hyperparameter ablations | LOW | Deprioritised after Changes I–II delivered | Would move row 4–6 by <0.02 F1 |

Full details: `EXPERIMENTS_REPORT.md` § 9.4, § 9.4a, § 9.5.

---

## 8. Limitations

*(~0.5 page. Be honest; these are the things a reviewer will ask.)*

1. **Single test split.** Every macro F1 is a point estimate on one 866-
   utterance split. Split variance alone is ±0.05. No k-fold CV due to
   compute budget — 22 experiments × 5 folds was infeasible.
   (`STAGE1_REPORT.md` § 13.)
2. **Val composition is not representative of test.** Val sustain = 16.4%,
   test sustain = 8.9%. We report val-test gap alongside val F1 as a
   distribution-shift diagnostic, but some val F1 movements are noise.
3. **No topic-aware stratification.** Splits are stratified on MI quality
   only, not topic. A topic-shift-driven split could perturb results.
4. **No end-to-end fine-tune of Qwen-8B.** Full-parameter 8B fine-tune does
   not fit on 24 GB A10G. LoRA is the parameter-efficient substitute; we
   did not ablate adapter rank beyond r=16.
5. **Closed-weight comparison.** GPT-4o-mini is a moving target; numbers
   reported here (snapshot date: `2026-04-24`) may not be reproducible
   once OpenAI revises the model.

---

## 9. Future Work

*(~0.25 page. Three bullets.)*

- **Transcript-grouped k-fold CV** to replace point estimates with
  confidence intervals.
- **Contrastive or pragmatic pretraining** on MI-style corpora (e.g.
  helpline transcripts, counseling dialogue) before LoRA fine-tune — the
  Section 6.2 finding suggests pretraining objective is the remaining
  lever.
- **Ensemble of Tier-1 open fine-tune + Tier-2 frontier LLM**, since they
  make different error patterns (LoRA over-predicts neutral on short
  utterances; zero-shot GPT over-predicts neutral globally).

---

## 10. Reproducibility

*(~0.25 page. Point at the docs and code.)*

- **Stage 1 (data, splits, baseline, synthetic generation)**:
  `STAGE1_REPORT.md`
- **Stage 2 (all 22 experiments, Rounds 1–10)**: `EXPERIMENTS_REPORT.md`
- **Artefact mapping (config → output dir)**: `EXPERIMENTS_REPORT.md` § 10
- **Code**:
  - Pipeline: `src/annomi_pipeline/` (ingestion, chunking, training)
  - Fine-tune training: `src/annomi_pipeline/training/train_finetune.py`
  - LoRA notebook: `notebooks/qwen8b_lora_aws.py`
  - Frozen Qwen embeddings: `notebooks/qwen8b_embedding_aws.py`
  - GPT-4o-mini classifier: `scripts/run_gpt4o_mini_classify.py`
- **Seeding**: seed=42 globally (`src/annomi_pipeline/utils/seed.py`).
- **Hardware**: local MPS for mid-size fine-tunes; AWS `ml.g5.2xlarge`
  (A10G 24 GB) for Qwen-8B.

---

## Appendix A — Full Leaderboard (22 experiments)

*(Copy verbatim from `EXPERIMENTS_REPORT.md` § 7.)*

## Appendix B — Confusion Matrices

*(Include the five most informative: baseline MiniLM, RoBERTa ft, Qwen
frozen, Qwen+LoRA, GPT-4o-mini few-shot. All in `EXPERIMENTS_REPORT.md`
§ 2, § 6.5, § 6.7, § 6.8, § 6.10.)*

## Appendix C — Hyperparameters

*(One table per paradigm: frozen+MLP, end-to-end ft, LoRA, GPT prompt.
Copy from `EXPERIMENTS_REPORT.md` footer.)*

---

## Writing checklist (delete before submission)

- [ ] Abstract written after rest of report is final
- [ ] Every number in this doc is cross-checked against its source section
- [ ] LoRA confusion matrix downloaded from AWS run and included in § 5.3
- [ ] Figures rendered at final resolution (300 dpi for print, 150 for
      screen)
- [ ] BibTeX references for: AnnoMI, SBERT, RoBERTa, DeBERTa, ModernBERT,
      Qwen3-Embedding, LoRA, GPT-4o-mini (technical report), PEFT library
- [ ] Contribution statement filled (which team member did what)
- [ ] Page count within course rubric
- [ ] Spell-check / grammar pass
- [ ] Remove all `<!-- TODO -->` markers

---

## Pointers back to primary sources

Every claim in this report should trace back to one of:

- `STAGE1_REPORT.md` — dataset, splits, synthetic data, baseline
- `EXPERIMENTS_REPORT.md` — Rounds 1–10, leaderboard, best model,
  per-class analyses, recommendations, artefact mapping

If a claim can't be traced, it either needs a citation or it needs to be
demoted from a result to a hypothesis.
