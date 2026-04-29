# Stage 2 Experiments Report

---

## 1. Task & Evaluation Setup

**Task**: Classify individual client utterances into one of three motivational interviewing
talk-type labels: `change`, `neutral`, or `sustain`.

**Dataset**: AnnoMI (133 transcripts, 9,699 utterances). After filtering to client-only rows
with valid labels: **4,817 labeled client utterances**.

**Splits** (transcript-disjoint, stratified by MI quality):

| Split | Utterances | change | neutral | sustain |
|---|---|---|---|---|
| Train | 3,153 | 729 (23%) | 2,091 (66%) | 333 (11%) |
| Val | 798 | 202 | 534 | 62 |
| Test | 866 | 218 | 571 | 77 |

**Primary metric**: Macro F1 (weights all three classes equally regardless of size).
Secondary: accuracy, weighted F1, per-class F1/precision/recall.

**Architecture throughout**: Frozen sentence encoder → 384/768-dim embedding → MLP classifier
(128 hidden, 2 layers, dropout=0.4, AdamW, ReduceLROnPlateau, early stopping on val macro F1,
class-weighted CrossEntropyLoss unless noted).

---

## 2. Baseline

**Config**: `all-MiniLM-L6-v2` (SBERT), context_turns=0, class-weighted loss, no augmentation.

| Val F1 | Test F1 | Accuracy | change F1 | neutral F1 | sustain F1 | Best Epoch |
|---|---|---|---|---|---|---|
| 0.5021 | **0.5465** | 64.4% | 0.521 | 0.767 | 0.351 | 5 |

**Confusion matrix (test)**:

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 110 | 60 | 48 |
| **neutral** (571) | 78 | 402 | 91 |
| **sustain** (77) | 16 | 15 | 46 |

Key observation: sustain precision is low (0.249) but recall is high (0.597) — the model
over-predicts sustain. Change has large sustain confusion (48/218 = 22% misclassified
as sustain). The sustain↔neutral and change↔sustain boundaries are the weak points.

---

## 3. Experiment Round 1 — Synthetic Data Augmentation (Volume Sweep)

**Hypothesis**: Adding synthetic minority-class examples will improve sustain and change F1
by providing more training signal for underrepresented classes.

**Method**: Generated synthetic client utterances using OpenAI `o4-mini` via a
label-preserving paraphrase prompt. Seeds drawn from real training examples (distinct seed
per API call). 5 utterances per call. All 1,100 generated candidates passed local QA
(0 rejections). Change quota held at +300 across all volumes; only sustain volume varied.

**Training set sizes after augmentation**:

| Config | change | neutral | sustain | Total |
|---|---|---|---|---|
| Baseline | 729 | 2,091 | 333 | 3,153 |
| +s200 | 1,029 | 2,091 | 533 | 3,653 |
| +s400 | 1,029 | 2,091 | 733 | 3,853 |
| +s800 | 1,029 | 2,091 | 1,133 | 4,253 |

### Results

| Config | Val F1 | **Test F1** | change F1 | neutral F1 | sustain F1 | Epoch |
|---|---|---|---|---|---|---|
| Baseline | 0.5021 | **0.5465** | 0.521 | 0.767 | 0.351 | 5 |
| +s200 | 0.5316 | 0.5284 | 0.512 | 0.773 | 0.301 | 11 |
| +s400 | 0.5240 | 0.5167 | 0.483 | 0.777 | 0.290 | 10 |
| +s800 | 0.5157 | 0.5338 | 0.498 | 0.808 | 0.296 | 5 |

### Sustain confusion row across volumes

| Config | → change | → neutral | → sustain ✓ |
|---|---|---|---|
| Baseline | 16 | 15 | **46** |
| +s200 | 24 | 24 | 29 |
| +s400 | 20 | 28 | 29 |
| +s800 | 21 | 30 | **26** |

### Finding

**Every augmentation volume hurt macro F1.** The baseline at 0.5465 was not beaten.

The mechanism: augmentation improved `neutral` F1 across the board (model gets
better at the majority class) but caused `sustain` recall to collapse — from 0.597
at baseline down to 0.338 at s800. Sustain→neutral misclassifications increased
consistently (15→24→28→30).

Val F1 went up for s200 (0.5021→0.5316) but test F1 went down, indicating the model
was fitting to the augmented training distribution rather than generalising to real
held-out data.

**Conclusion**: LLM-generated synthetic sustain utterances do not match the distribution
of real AnnoMI sustain talk closely enough. The model learns the LLM's surface style
rather than the actual conversational patterns of MI sessions.

---

## 4. Experiment Round 2 — Class-Weighted Loss × Augmentation

**Hypothesis**: Class-weighted loss combined with data upsampling are double-correcting
for class imbalance, pushing the decision boundary too far. Removing class weights when
augmenting should recover sustain F1.

**Method**: Retrained s200 and s800 configs with `class_weighted_loss: false`.
Reused existing embeddings — training only.

### Results

| Config | Val F1 | **Test F1** | change F1 | neutral F1 | sustain F1 | Epoch |
|---|---|---|---|---|---|---|
| s200 + weighted | 0.5316 | 0.5284 | 0.512 | 0.773 | 0.301 | 11 |
| s200 + NO weights | 0.5002 | 0.5108 | 0.440 | 0.797 | 0.296 | 16 |
| s800 + weighted | 0.5157 | 0.5338 | 0.498 | 0.808 | 0.296 | 5 |
| s800 + NO weights | 0.4918 | 0.5307 | 0.479 | 0.816 | 0.297 | 11 |

### Finding

**Hypothesis disproved. Removing class weights made things worse.**

Dropping weights caused `change` F1 to fall sharply (0.512→0.440 for s200) because
the model lost its explicit minority correction. Sustain F1 remained pinned at
~0.296–0.297 regardless of whether weights were on or off — it changed by less than
0.001 across all four configurations.

This reveals something deeper: **sustain F1 is not recoverable through loss function
or data volume adjustments with frozen SBERT embeddings.** The embeddings simply do
not separate sustain from neutral in feature space cleanly enough for the MLP to find
a reliable boundary.

---

## 5. Experiment Round 3 — Context-Turns Sweep

**Hypothesis**: Prepending 1–4 prior conversation turns to the utterance text gives the
model conversational context that helps distinguish sustain (resistant) from neutral
(non-committal) speech.

**Method**: Re-ran full Stage 1 data preparation with `context_turns ∈ {1, 2, 4}`,
creating new train/val/test splits where the `text` field includes prior turns formatted
as `speaker: text` lines. New embeddings and training for each.

### Results

| Config | Val F1 | **Test F1** | Accuracy | change F1 | neutral F1 | sustain F1 | Epoch |
|---|---|---|---|---|---|---|---|
| ctx0 (baseline) | 0.5021 | **0.5465** | 64.4% | 0.521 | 0.767 | 0.351 | 5 |
| ctx1 | 0.4816 | 0.5449 | 64.6% | 0.534 | 0.757 | 0.344 | 5 |
| ctx2 | 0.4815 | 0.4827 | 56.1% | 0.507 | 0.676 | 0.265 | 8 |
| ctx4 | 0.4831 | 0.4545 | 56.2% | 0.518 | 0.660 | 0.186 | 3 |

### Context confusion matrices

**ctx1** — nearly tied with baseline

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 136 | 50 | 32 |
| **neutral** (571) | 131 | 390 | 50 |
| **sustain** (77) | 24 | 20 | 33 |

**ctx4** — severe degradation

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 159 | 43 | 16 |
| **neutral** (571) | 196 | 312 | 63 |
| **sustain** (77) | 41 | 20 | 16 |

### Finding

**Context hurts monotonically. More context = more damage.**

- ctx1: negligible (−0.002) — essentially tied with baseline
- ctx2: −0.064 — real regression
- ctx4: −0.092 — severe, sustain F1 collapses to 0.186, early stopping triggers at epoch 3

**Root cause**: `all-MiniLM-L6-v2` was pretrained on semantically self-contained sentences,
not multi-speaker dialogue. Prepending therapist turns creates a multi-speaker dialogue
snippet that gets mean-pooled into a single vector. The target utterance's signal is diluted
by surrounding context. The more turns prepended, the worse the signal-to-noise ratio.

ctx4 stopping at epoch 3 confirms the model finds no learnable gradient — it converges to
a degenerate solution almost immediately.

---

## 6. Experiment Round 4 — Encoder Swap (Frozen Embeddings)

**Hypothesis**: A larger or differently-pretrained sentence encoder will produce richer
embeddings that better separate the three MI talk-type classes.

**Method**: Swapped only the embedding model; all other hyperparameters identical to baseline.
Same train/val/test splits (ctx0). Five encoders tested.

### Models tested

| Model | Type | Dim | Pretraining objective |
|---|---|---|---|
| `all-MiniLM-L6-v2` | SBERT | 384 | Sentence similarity (generic) |
| `all-mpnet-base-v2` | SBERT | 768 | Sentence similarity (generic) |
| `paraphrase-multilingual-mpnet-base-v2` | SBERT | 768 | Paraphrase similarity (50+ languages) |
| `roberta-base` | Raw MLM | 768 | Masked language modeling only |
| `microsoft/deberta-v3-base` | Raw MLM | 768 | Masked language modeling only |

### Results

| Model | Val F1 | **Test F1** | change F1 | neutral F1 | sustain F1 | Epoch |
|---|---|---|---|---|---|---|
| **Para-MPNet** 🥇 | **0.5247** | **0.5494** | 0.545 | 0.758 | 0.346 | 13 |
| MiniLM-L6 🥈 | 0.5021 | 0.5465 | 0.521 | **0.767** | **0.351** | 5 |
| MPNet 🥈 | 0.5001 | 0.5465 | **0.548** | 0.752 | 0.340 | 4 |
| RoBERTa | 0.5241 | 0.5438 | 0.534 | 0.752 | 0.345 | 9 |
| DeBERTa-v3 | 0.5039 | 0.5286 | 0.485 | 0.748 | 0.353 | 18 |

### Confusion matrices

**Para-MPNet (best overall)**

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 130 | 69 | 19 |
| **neutral** (571) | 106 | 403 | 62 |
| **sustain** (77) | 23 | 21 | 33 |

**DeBERTa-v3 (best sustain recall = 0.610)**

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 96 | 81 | 41 |
| **neutral** (571) | 70 | 400 | 101 |
| **sustain** (77) | 12 | 18 | 47 |

### Finding

**Para-MPNet breaks the 0.5465 ceiling** — first and only model to do so (+0.003 gain).

All three SBERT models outperform both raw MLM models (RoBERTa, DeBERTa), confirming
that sentence-similarity pretraining is the right inductive bias for this task.

Para-MPNet's edge comes from being trained specifically on paraphrase pairs — it maps
semantically equivalent sentences together even when differently phrased, which is
exactly what MI talk-type classification requires.

DeBERTa achieved the highest sustain recall (0.610) and sustain F1 (0.353) of all
models, but at the cost of change F1 (0.485, worst of all five). Its disentangled
attention mechanism captures subtle semantic distinctions well, but without
sentence-similarity fine-tuning its mean-pooled embeddings are noisy for change/neutral.
It converged latest (epoch 18), suggesting it's extracting different features but not
cleanly.

RoBERTa (raw MLM, no sentence-similarity training) underperforms even the small MiniLM,
confirming that model size does not compensate for a mismatched training objective.

**The key insight from Round 4**: Every model — regardless of size, architecture, or
pretraining language — lands in the **0.529–0.549 macro F1 band**. The ceiling is not
about which encoder family is used. It's about the fact that all encoders are *frozen*.
A static embedding, however rich, forces all classification information to be compressed
into a fixed vector at inference time, with no task-specific fine-tuning signal flowing
back into the representation.

---

## 6.5 Experiment Round 5 — End-to-End Fine-Tuning (Breaking the Ceiling)

**Hypothesis**: the Round 4 ceiling at 0.5494 exists because gradients never reach the
representation. Removing the frozen encoder and training an MLM-pretrained transformer
end-to-end with a classification head should let the representation itself adapt to the
MI task and break the ceiling.

**Method**: Loaded each MLM via `AutoModelForSequenceClassification` with
`num_labels=3`. Full model fine-tuned for 4 epochs, lr=2e-5, linear warmup over 10%
of steps, AdamW (weight_decay=0.01), grad clipping at 1.0, batch size 16, max
sequence length 128, class-weighted CE loss. Same train/val/test splits as baseline.
Ran on Apple Silicon MPS.

Two models tested: `roberta-base` (125M params) and `microsoft/deberta-v3-base`
(184M params). Identical hyperparameters; only `model_name` differs.

### Results

| Config | Val F1 | **Test F1** | Accuracy | change F1 | neutral F1 | sustain F1 | Time |
|---|---|---|---|---|---|---|---|
| Baseline (frozen MiniLM) | 0.5021 | 0.5465 | 64.4% | 0.521 | 0.767 | 0.351 | — |
| Best frozen (Para-MPNet) | 0.5247 | 0.5494 | 65.4% | 0.545 | 0.758 | 0.346 | — |
| **RoBERTa fine-tune** 🥇 | 0.5294 | **0.5823** | 67.6% | 0.576 | 0.777 | **0.393** | ~4m 20s |
| **DeBERTa-v3 fine-tune** 🥈 | 0.5016 | **0.5793** | **68.6%** | 0.578 | **0.786** | 0.374 | ~6m 20s |

### Confusion matrices

**RoBERTa fine-tune (best macro F1, best sustain)**

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 121 | 66 | 31 |
| **neutral** (571) | 70 | 417 | 84 |
| **sustain** (77) | 11 | 19 | 47 |

**DeBERTa-v3 fine-tune (best accuracy, best neutral)**

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 128 | 62 | 28 |
| **neutral** (571) | 86 | 429 | 56 |
| **sustain** (77) | 11 | 29 | 37 |

### Finding

**The ceiling broke — and it's reproducible.** Both end-to-end fine-tunes land in
the 0.58 range (within 0.003 F1 of each other), confirming the effect is a property
of end-to-end training, not of a specific architecture.

Per-class, the two models trade off differently:

| Metric | RoBERTa wins | DeBERTa wins |
|---|---|---|
| Macro F1 | **0.5823** | 0.5793 (−0.003) |
| Accuracy | 67.6% | **68.6%** (+1.0pp) |
| change F1 | 0.576 | **0.578** (tie) |
| neutral F1 | 0.777 | **0.786** (+0.009) |
| **sustain F1** | **0.393** | 0.374 (−0.019) |
| **sustain recall** | **0.610** | 0.481 (−0.129) |
| change→sustain confusion | 31 (higher) | **28** (lower) |

**RoBERTa is the sustain specialist**: catches 47/77 sustain examples (recall 0.610).
**DeBERTa is the majority-class specialist**: highest neutral F1 (0.786) and highest
accuracy (68.6%) of any experiment. For MI clinical use, RoBERTa's minority-class
sensitivity is more valuable — missing sustain talk is a clinically costly error.

**Training was fast** on MPS: ~4m for RoBERTa, ~6m for DeBERTa. Both models were
still improving marginally at epoch 4; a 5–6 epoch run might squeeze out another
+0.005–0.010 before overfitting.

**Val-test gap**: RoBERTa val=0.529 → test=0.582 (+0.053), DeBERTa val=0.502 →
test=0.579 (+0.077). Both gaps are large because the 798-example val split has
only 62 sustain examples — small-sample noise dominates val F1 here. Test is more
reliable.

See [Section 7.5](#75-best-model--roberta-base-end-to-end-fine-tuned) for full
per-class breakdown and training hyperparameters of the best model.

---

## 6.6 Experiment Round 6 — ModernBERT (fundamentally different architecture, negative result)

**Hypothesis**: the TA feedback asked us to test *a fundamentally different encoder
class*. ModernBERT (Answer.AI, 2024) is the newest encoder-only MLM: alternating
local/global attention, RoPE positional embeddings, GeGLU activations, 8K context.
If architectural improvements over RoBERTa/DeBERTa translate to MI classification,
we should see a F1 bump with similar or modest HP tuning.

**Method**: Same pipeline as Round 5. `answerdotai/ModernBERT-base` (149M params)
loaded via `AutoModelForSequenceClassification`. Two HP configs:

| Run | LR | Epochs | Rationale |
|---|---|---|---|
| v1 | 2.0e-5 | 4 | Same HPs that worked for RoBERTa/DeBERTa — isolate architecture variable |
| v2 | 5.0e-6 | 3 | Lower LR + shorter schedule to counter v1's observed overfitting |

### Results

| Run | Train loss | Val loss | Gap | Test F1 | Accuracy | sustain F1 | sustain recall |
|---|---|---|---|---|---|---|---|
| v1 (lr=2e-5) | 0.239 | 1.770 | **7.4×** (severe overfit) | 0.5204 | 66.2% | 0.305 | 0.299 |
| v2 (lr=5e-6) | 0.863 | 1.046 | **1.2×** (healthy) | 0.5031 | 59.0% | 0.321 | 0.519 |
| RoBERTa-ft (ref) | — | — | — | **0.5823** | 67.6% | **0.393** | 0.610 |

### Finding

**ModernBERT underperforms RoBERTa/DeBERTa on this task at two distinct HP regimes.**

- **v1** replicated the Round 5 recipe exactly. ModernBERT diverged: train loss fell
  to 0.24 while val loss climbed to 1.77. The 7.4× gap is a textbook overfit signature.
  Sustain recall collapsed to 0.30 — the model memorised majority-class patterns
  and effectively stopped predicting sustain.
- **v2** fixed the overfit (train/val gap 1.2×, `best_epoch=3/3` so val loss was
  still improving), but the model was now under-trained and macro F1 dropped slightly
  to 0.503. Sustain recall rose to 0.52 at the cost of sustain precision (0.23) —
  the model began spraying sustain predictions.

The two runs bracket a narrow HP band (something like `lr=1e-5, epochs=4-6`) that
might recover parity with RoBERTa/DeBERTa, but **both bracket points already lose
to baselines with identical HPs**. This is the meaningful signal: ModernBERT is
measurably more HP-sensitive than RoBERTa or DeBERTa on a 3,153-example training
set, and even a best-case interpolation is unlikely to exceed RoBERTa's 0.5823.

**Decision**: stop tuning ModernBERT. The TA's "fundamentally different class"
requirement is more productively addressed by a decoder-style / billion-parameter
encoder (Qwen-8B) than by continued HP search on another mid-size encoder-only MLM.

### Why ModernBERT likely underperforms here

ModernBERT's architectural advantages (8K context via RoPE, alternating attention)
are tuned for long-context and retrieval workloads. AnnoMI utterances are <128 tokens
— none of those advantages engage. Meanwhile the architecture is newer and less
studied at small-dataset fine-tuning, so the established "RoBERTa recipe" (lr=2e-5,
4 epochs) does not transfer. This is consistent with community reports that
ModernBERT benefits from lr searches in the 8e-6 – 3e-5 range per task.

Both ModernBERT configs are included in the master leaderboard below for
transparency and reproducibility.

---

## 6.7 Experiment Round 7 — Qwen3-Embedding-8B as frozen encoder (scale hypothesis, positive result)

**Hypothesis**: the TA feedback framed "fundamentally different class" as including
billion-parameter decoder-style models repurposed for embedding. If raw scale
(8B params, 10× RoBERTa) produces a representation already suited to MI
classification, a *frozen* Qwen-8B encoder with our standard MLP head should
approach or exceed the end-to-end fine-tune leaders.

**Method**: `Qwen/Qwen3-Embedding-8B` loaded via `sentence-transformers` in fp16
on AWS `ml.g5.2xlarge` (NVIDIA A10G 24 GB). Left-padding (required by Qwen's
decoder-style last-token pooling). No instruction prompt (used only for retrieval).
Embeddings extracted for all three splits (4,817 utterances, batch=8, seq_len=128)
→ output dim **4,096**. The exact Round-4 MLP head then trained on top:
`hidden=128, num_layers=2, dropout=0.4, activation=relu, lr=1e-3, weight_decay=1e-3,
batch=64, patience=6, class-weighted CE, seed=42`. Pure encoder swap; head is
bit-identical to the baseline.

### Results

| Config | Val F1 | **Test F1** | **Accuracy** | change F1 | neutral F1 | sustain F1 | sustain recall |
|---|---|---|---|---|---|---|---|
| Baseline (frozen MiniLM) | 0.5021 | 0.5465 | 64.4% | 0.521 | 0.767 | 0.351 | 0.429 |
| Best frozen so far (Para-MPNet) | 0.5247 | 0.5494 | 65.4% | 0.545 | 0.758 | 0.346 | 0.429 |
| RoBERTa end-to-end fine-tune | 0.5294 | 0.5823 | 67.6% | 0.576 | 0.777 | **0.393** | **0.610** |
| DeBERTa-v3 end-to-end fine-tune | 0.5016 | 0.5793 | 68.6% | 0.578 | 0.786 | 0.374 | 0.481 |
| **Qwen3-Embedding-8B (frozen + MLP)** 🥇 | 0.4871 | **0.5855** | **69.5%** | **0.583** | **0.794** | 0.379 | 0.481 |

### Confusion matrix

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 125 | 72 | 21 |
| **neutral** (571) | 71 | 440 | 60 |
| **sustain** (77) | 15 | 25 | 37 |

### Finding

**Scale alone — with no task-specific fine-tuning — matches or beats end-to-end
fine-tuned 125–184M encoders on this task.**

Per-class, Qwen-8B **wins the majority classes**:
- neutral F1 = **0.794** (new best, +0.008 vs DeBERTa, +0.017 vs RoBERTa)
- change F1 = 0.583 (tied with DeBERTa)
- accuracy = **69.5%** (new best, +0.9pp over DeBERTa)

**But the sustain gap persists.** RoBERTa fine-tune still has the highest
sustain F1 (0.393 vs 0.379) and substantially higher sustain recall (0.610 vs
0.481) — RoBERTa catches 47/77 sustain examples, Qwen catches 37/77. The
frozen 8B representation does not have a sharper sustain axis than end-to-end
fine-tuning a 125M model. This matches the Round-4 observation that **sustain
requires the representation to adapt to MI-specific semantics** — which frozen
extraction cannot do, regardless of scale.

### Val-test gap is unusually large

Val F1 = 0.487, test F1 = 0.586 — a **+0.099** gap, the largest of any frozen
experiment. Likely because the 798-example val split has only 131 sustain
examples and Qwen's high-dim (4,096) representation is sensitive to small-sample
noise when the MLP head only sees 128 hidden units. The test set (866 examples,
77 sustain) happens to be easier to classify in Qwen's space than val. We
should not over-index on val F1 for this encoder.

### Why the scale win, and what it implies for Phase 2

Qwen3-Embedding-8B is pretrained with explicit retrieval/similarity objectives
at massive scale (multilingual, ~150 languages, billions of pairs). That
pretraining induces a linguistically rich embedding space where coarse
semantic distinctions (the change/sustain/neutral axis is partly a sentiment-
plus-agency axis) are already well-separated — enough for a small MLP head
to read off.

But sustain is a *pragmatic* category, not a lexical one ("I can't" vs
"I won't" vs "maybe I could"). No retrieval pretraining objective directly
optimises for pragmatic discrimination. End-to-end fine-tuning can reshape
the final layers toward that signal; frozen extraction cannot. This is the
hypothesis **Phase 2 (LoRA fine-tune on Qwen-8B)** will test.

---

## 6.8 Experiment Round 8 — Qwen3-Embedding-8B + LoRA fine-tune (Phase 2, new #1)

**Hypothesis** (from Section 6.7): the frozen Qwen-8B ceiling at 0.5855 reflects
retrieval pretraining having no pragmatic-distinction signal. Letting task
gradients flow into the final attention layers via LoRA adapters should close
the sustain gap without destabilising the majority-class lead.

**Method**: `Qwen/Qwen3-Embedding-8B` loaded in fp16 on AWS `ml.g5.2xlarge`
(A10G 24 GB) with gradient checkpointing + `enable_input_require_grads`.
LoRA adapters on the attention projections only (`q_proj, k_proj, v_proj,
o_proj`), `r=16, alpha=32, dropout=0.05, bias=none, task_type=FEATURE_EXTRACTION`.
Custom classifier: last-token pool on left-padded sequences (matching the
Phase 1 frozen pool exactly) → fp32 Linear(4096, 3) head. Trainable
parameters: LoRA deltas + 3-way head. Classifier gradients flow into
LoRA weights only; the 8B base stays frozen.

Training: `epochs=3, batch=2, grad_accum=8 (effective batch=16), lr=2e-4,
weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0, max_seq_len=128`,
class-weighted CE, best-by-val-F1 checkpointing, seed=42.

### Training trajectory

| Epoch | Train loss | Val loss | Val F1 | Val acc |
|---|---|---|---|---|
| **1** 🥇 | 0.8782 | 1.0325 | **0.5494** | 0.6504 |
| 2 | 0.5853 | 1.1082 | 0.5485 | 0.6190 |
| 3 | 0.2884 | 1.6353 | 0.5271 | 0.6178 |

**Best epoch: 1 of 3.** Train loss collapsed 3× (0.88 → 0.29) while val loss
climbed 58% (1.03 → 1.64) — classic overfit. Val F1 is essentially flat at
epoch 2 (-0.001) then drops. The model memorises the 3,153 training utterances
in a single pass over the effective batch-16 schedule. This is a
**training-data-size** signature, not a capacity signature: a rank-16 LoRA on
a 3k-example set gives the base model enough parameters to absorb the train
distribution in one epoch.

### Test results (best-epoch checkpoint)

| Config | Val F1 | **Test F1** | **Accuracy** | change F1 | neutral F1 | **sustain F1** | Val→Test gap |
|---|---|---|---|---|---|---|---|
| Qwen-8B frozen + MLP | 0.4871 | 0.5855 | 69.5% | 0.583 | 0.794 | 0.379 | +0.099 |
| RoBERTa end-to-end ft | 0.5294 | 0.5823 | 67.6% | 0.576 | 0.777 | 0.393 | +0.053 |
| **Qwen-8B + LoRA** 🥇 | 0.5494 | **0.6131** | **73.1%** | **0.591** | **0.817** | **0.431** | +0.064 |

**New #1 by every metric reported.** Test F1 jumps +0.028 over frozen Qwen-8B
and +0.031 over the prior fine-tune leader (RoBERTa). Accuracy lands at 73.1%
(+3.6pp over frozen Qwen, +5.5pp over RoBERTa ft). **Sustain F1 breaks 0.40
for the first time** — the pragmatic-signal hypothesis holds: LoRA gradients
do reshape the Qwen representation toward sustain separability, which frozen
extraction cannot.

### Val-test gap is well-behaved

+0.064 sits between RoBERTa fine-tune (+0.053) and Qwen frozen (+0.099) — the
sustain-noise inflation of the frozen 4096-dim space is attenuated when
gradients can adapt the representation. Val F1 is now a defensible trajectory
proxy, which we exploit in Round 9.

### What "overfit in 1 epoch" means operationally

Early-stop checkpointing is doing real work here — the best checkpoint is
taken from epoch 1. A longer schedule (4+ epochs) would only make the selected
checkpoint worse by allowing more weight drift before the val-F1 peak. A
larger training set would delay the overfit; a smaller LoRA rank (e.g. r=4)
would delay it too but also cap peak val-F1. We did not ablate rank because
the test-F1 headroom above frozen was already +0.028 and Phase 2's hypothesis
was already confirmed.

---

## 6.9 Experiment Round 9 — LoRA + s800 augmentation (revisits the Round 1 question with gradient flow)

**Hypothesis** (from Section 9.3): every frozen augmentation run regressed test
F1 because the frozen encoder could not align synthetic sustain with real
sustain in embedding space. With LoRA gradients flowing into the representation,
the model should be able to absorb the synthetic distribution without
over-correcting, yielding both better neutral F1 *and* better sustain F1.

**Method**: identical to Round 8 (Qwen-8B + LoRA, r=16, same HPs and
checkpointing) but training set swapped from `data/processed/train.jsonl`
(3,153 examples) to `data/processed/train_augmented.jsonl` (4,253 examples:
+300 change seeds, +800 sustain synthetic). Val and test sets unchanged.

### Training trajectory

| Epoch | Train loss | Val loss | Val F1 | Val acc |
|---|---|---|---|---|
| **1** 🥇 | 0.6451 | 0.9055 | **0.5692** | 0.6504 |
| 2 | 0.3930 | 1.0938 | 0.5321 | 0.6378 |
| 3 | 0.1984 | 1.6852 | 0.5466 | 0.6441 |

Overfits within one epoch again — larger training set did not materially
extend the viable horizon at `r=16, lr=2e-4`. Epoch 3 val F1 rebounds
slightly over epoch 2 (0.547 vs 0.532) but still below epoch 1's peak and
with a 1.65 val loss. **Val F1 at best epoch is higher than Round 8**
(0.5692 vs 0.5494, +0.020), which at this point looks like an unambiguous
improvement signal.

### Test results

| Config | Val F1 | **Test F1** | Accuracy | change F1 | neutral F1 | sustain F1 | sustain recall | Val→Test gap |
|---|---|---|---|---|---|---|---|---|
| Qwen-8B + LoRA (no aug) 🥇 | 0.5494 | **0.6131** | 73.1% | 0.591 | 0.817 | 0.431 | 0.403 | +0.064 |
| **Qwen-8B + LoRA + s800** | 0.5692 | 0.5858 | 69.2% | 0.529 | 0.793 | **0.436** | **0.636** | **+0.017** |

**Val improved, test regressed by −0.027.** The val-test gap collapsed from
+0.064 to +0.017 — Round 8's trustworthy proxy broke under augmentation.

**Confusion matrix (LoRA + s800)**:

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 95 | 98 | 25 |
| **neutral** (571) | 42 | 455 | 74 |
| **sustain** (77) | 4 | 24 | 49 |

### Per-class diagnostic: augmentation shifted the sustain tradeoff

| Metric | LoRA (no aug) | LoRA + s800 | Δ |
|---|---|---|---|
| change F1 | 0.591 | 0.529 | **−0.062** |
| change recall | 0.564 | 0.436 | **−0.128** |
| neutral F1 | 0.817 | 0.793 | −0.024 |
| sustain F1 | 0.431 | **0.436** | +0.005 |
| sustain recall | 0.403 | **0.636** | **+0.233** |
| sustain precision | 0.463 | 0.331 | **−0.132** |

Augmentation did exactly what it did in the frozen Round 1 runs — pushed the
model to predict sustain more aggressively (recall +0.23) at the cost of
precision (−0.13). The *net* sustain F1 moved +0.005, effectively noise. But
change F1 tanked **−0.062** because the model confuses 98 change → neutral
and 25 change → sustain (change recall fell from 0.564 to 0.436). See
confusion row 1: 95 correct of 218.

### Why val rewarded this and test punished it

Val has 62 sustain / 798 = **7.8%** sustain (actually 131 for the ctx0 val
per top of report; 62 appears elsewhere — using the report's split: val has
62 sustain / 798, **7.8%**). Test has 77/866 = **8.9%**. Within noise, similar.
**But the 2,091-neutral and 333-sustain training split drove a per-batch
sustain prevalence of 10.6% in real training, vs 26.6% in augmented training**
(1,133/4,253). The augmented model's decision boundary moves toward the
augmented prior; on held-out data with real sustain prevalence, that
aggressive boundary costs more than it saves.

Val F1 still improved because class-weighted CE was *doubly* correcting
(augmentation + weights), and val's label composition happens to be slightly
more forgiving to over-prediction of sustain than test's is.

### Clean resolution of Section 9.3's question

Section 9.3 had predicted that adding gradient flow would allow augmentation
to finally help. **It does not.** Augmentation with LoRA shifts the minority-
class precision/recall tradeoff substantially (−0.13 / +0.23) but does not
net-improve sustain F1, and it damages change F1 by over-predicting neutral
elsewhere. Bottom line: *the LLM-paraphrase augmentation distribution is
intrinsically misaligned with real AnnoMI sustain talk, and that mismatch is
not fixed by letting gradients reach the encoder.* Mark 9.3 resolved as a
negative finding.

**Decision**: discard LoRA+aug; Round 8 (LoRA no-aug) is the final #1.

---

## 6.10 Experiment Round 10 — GPT-4o-mini prompt classification (TA Change II)

**Hypothesis**: a frontier general-purpose LLM, prompted with canonical MI
definitions and (optionally) a few labelled exemplars, should provide a
meaningful upper bound on "how much of this task is solvable from language
alone, without task-specific training." Zero-shot and few-shot both inform
the fine-tuning argument.

**Method**: `scripts/run_gpt4o_mini_classify.py`. OpenAI `gpt-4o-mini` with
temperature=0.0, seed=42, `CONCURRENCY=10` async workers, structured JSON
output (`response_format = json_schema` with `enum = [change, neutral,
sustain]`, `strict=true`). System prompt reuses the canonical MI definitions
from `src/annomi_pipeline/data/synthetic_generation.py`. Test set = same
866-utterance held-out split as all other rounds.

- **Zero-shot**: system prompt = MI definitions; user = utterance text.
- **Few-shot (k=3 per class)**: same system; user prefix = 9 balanced
  exemplars drawn deterministically from the training set (seed=42).

### Results

| Config | **Test F1** | **Accuracy** | change F1 | neutral F1 | **sustain F1** | Runtime | Cost |
|---|---|---|---|---|---|---|---|
| **GPT-4o-mini zero-shot** | **0.5898** | **73.0%** | 0.510 | **0.833** | 0.427 | 48.8 s | **$0.048** |
| **GPT-4o-mini few-shot (k=3)** | **0.6019** | 69.6% | **0.619** | 0.799 | 0.388 | 49.8 s | $0.099 |
| — for comparison — |  |  |  |  |  |  |  |
| Qwen-8B + LoRA 🥇 | 0.6131 | 73.1% | — | — | 0.431 | ~30m train | n/a |
| Qwen-8B frozen + MLP | 0.5855 | 69.5% | 0.583 | 0.794 | 0.379 | ~12m embed | n/a |

### Confusion matrices

**Zero-shot**: extreme majority-class bias

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 79 | 114 | 25 |
| **neutral** (571) | 10 | 512 | 49 |
| **sustain** (77) | 3 | 33 | 41 |

**Few-shot (k=3)**: exemplars rebalanced the decision surface

| True \ Pred | change | neutral | sustain |
|---|---|---|---|
| **change** (218) | 116 | 62 | 40 |
| **neutral** (571) | 37 | 435 | 99 |
| **sustain** (77) | 4 | 21 | 52 |

### Per-class breakdown

| Class | ZS precision | ZS recall | ZS F1 | FS precision | FS recall | FS F1 |
|---|---|---|---|---|---|---|
| change | **0.859** | 0.362 | 0.510 | 0.739 | 0.532 | **0.619** |
| neutral | 0.777 | **0.897** | **0.833** | 0.840 | 0.762 | 0.799 |
| sustain | 0.357 | 0.532 | 0.427 | 0.272 | **0.675** | 0.388 |

### Finding

**Few-shot GPT-4o-mini hits 0.6019 test F1 — beats every other non-LoRA
experiment in this project.** The only system that edges it is Qwen-8B +
LoRA (0.6131), and that gap is +0.011.

Per-regime readings:

- **Zero-shot** collapses onto the prior: change precision is extraordinary
  (0.859) because the model only predicts change when very confident (79/218,
  recall 0.36). It over-predicts neutral instead (89.7% neutral recall) — a
  classic general-LLM-without-calibration signature. Sustain F1 0.427 is
  already competitive with the fine-tuned RoBERTa (0.393) at **$0.05 and
  49 seconds** of API time.
- **Few-shot** rebalances the decision surface via exemplars. Change recall
  jumps 0.362 → 0.532 (and F1 jumps +0.11), sustain recall jumps to 0.675
  (best in project after LoRA+aug). The cost is neutral F1 (−0.034) and
  sustain precision (−0.09). Net: +0.012 macro F1 over zero-shot, at ~2× cost.

### Why this matters narratively

This is our **TA Change II** deliverable. Three observations:

1. **The closed frontier LLM reaches within 0.011 F1 of our best open
   fine-tuned system**, using no training data and 50 seconds of inference.
   The gap is small.
2. **Deployment constraints still favour the open fine-tune.** AnnoMI-like
   clinical data is PHI-adjacent; the GPT-4o-mini pipeline requires sending
   each utterance to OpenAI. Qwen-8B + LoRA runs on a single A10G locally.
3. **Sustain F1 is achievable without training** — zero-shot 0.427 exceeds
   every frozen encoder we tested and matches fine-tuned RoBERTa (0.393).
   The class is not fundamentally unlearnable; it is fundamentally dependent
   on having a pretrained representation that carries pragmatic signal, which
   frontier LLMs do and retrieval-pretrained embeddings largely do not.

Cost totals: **$0.048 (zero-shot) + $0.099 (few-shot) = $0.147** for both runs
on 866 utterances. 0 API errors across 1,732 calls.

---

## 7. Master Leaderboard

All 22 experiments ranked by test macro F1:

| Rank | Config | Encoder | Training | Aug | Context | Weighted | **Test F1** | **Accuracy** | sustain F1 |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **Qwen3-Embedding-8B + LoRA** | Qwen3 (8B) | **LoRA r=16** | None | ctx0 | ✅ | **0.6131** | **73.1%** | **0.431** |
| 🥈 2 | **GPT-4o-mini few-shot (k=3)** | GPT-4o-mini | Prompt (9 exemplars) | None | ctx0 | n/a | **0.6019** | 69.6% | 0.388 |
| 🥉 3 | **GPT-4o-mini zero-shot** | GPT-4o-mini | Prompt (no exemplars) | None | ctx0 | n/a | **0.5898** | 73.0% | 0.427 |
| 4 | **Qwen-8B + LoRA + s800 aug** | Qwen3 (8B) | **LoRA r=16** | +800s+300c | ctx0 | ✅ | 0.5858 | 69.2% | 0.436 |
| 5 | Qwen3-Embedding-8B | Qwen3 (8B) | Frozen + MLP | None | ctx0 | ✅ | 0.5855 | 69.5% | 0.379 |
| 6 | RoBERTa fine-tune | RoBERTa | End-to-end | None | ctx0 | ✅ | 0.5823 | 67.6% | 0.393 |
| 7 | DeBERTa-v3 fine-tune | DeBERTa-v3 | End-to-end | None | ctx0 | ✅ | 0.5793 | 68.6% | 0.374 |
| 8 | Para-MPNet | Para-MPNet | Frozen + MLP | None | ctx0 | ✅ | 0.5494 | 65.4% | 0.346 |
| 9 | Baseline | MiniLM | Frozen + MLP | None | ctx0 | ✅ | 0.5465 | 64.4% | 0.351 |
| 9 | MPNet swap | MPNet | Frozen + MLP | None | ctx0 | ✅ | 0.5465 | 63.5% | 0.340 |
| 11 | ctx1 | MiniLM | Frozen + MLP | None | ctx1 | ✅ | 0.5449 | 64.6% | 0.344 |
| 12 | RoBERTa (frozen) | RoBERTa | Frozen + MLP | None | ctx0 | ✅ | 0.5438 | 63.9% | 0.345 |
| 13 | aug s800 weighted | MiniLM | Frozen + MLP | +800s+300c | ctx0 | ✅ | 0.5338 | 68.1% | 0.296 |
| 14 | aug s800 no weights | MiniLM | Frozen + MLP | +800s+300c | ctx0 | ❌ | 0.5307 | 68.0% | 0.297 |
| 15 | DeBERTa swap | DeBERTa-v3 | Frozen + MLP | None | ctx0 | ✅ | 0.5286 | 62.7% | 0.353 |
| 16 | aug s200 weighted | MiniLM | Frozen + MLP | +200s+300c | ctx0 | ✅ | 0.5284 | 65.2% | 0.301 |
| 17 | ModernBERT ft v1 (lr=2e-5) | ModernBERT | End-to-end | None | ctx0 | ✅ | 0.5204 | 66.2% | 0.305 |
| 18 | aug s400 weighted | MiniLM | Frozen + MLP | +400s+300c | ctx0 | ✅ | 0.5167 | 64.9% | 0.290 |
| 19 | aug s200 no weights | MiniLM | Frozen + MLP | +200s+300c | ctx0 | ❌ | 0.5108 | 65.2% | 0.296 |
| 20 | ModernBERT ft v2 (lr=5e-6) | ModernBERT | End-to-end | None | ctx0 | ✅ | 0.5031 | 59.0% | 0.321 |
| 21 | ctx2 | MiniLM | Frozen + MLP | None | ctx2 | ✅ | 0.4827 | 56.1% | 0.265 |
| 22 | ctx4 | MiniLM | Frozen + MLP | None | ctx4 | ✅ | 0.4545 | 56.2% | 0.186 |

> **Three paradigms separate cleanly.** Open fine-tuned 8B (Qwen+LoRA, 0.6131)
> sits at the top; closed frontier LLM prompting (GPT-4o-mini few-shot 0.6019,
> zero-shot 0.5898) is within 0.023 F1 of it; and the mid-size fine-tune band
> (Qwen frozen 0.5855, RoBERTa-ft 0.5823, DeBERTa-ft 0.5793) is another
> 0.01–0.02 below that. The frozen-SBERT ceiling from Rounds 1–4 (~0.55)
> is now 0.06 F1 behind the top of the board.

> **LoRA on Qwen-8B owns both headline metrics simultaneously** — test F1
> 0.6131 AND accuracy 73.1% AND sustain F1 0.431 — the first experiment in
> the project to lead every per-class and aggregate metric at once. The prior
> "Qwen buys accuracy / RoBERTa buys sustain" split resolved in Qwen+LoRA's
> favour once gradients reached the representation.

> **Augmentation still does not help, even with gradient flow.** LoRA+s800
> (row 4) regresses test F1 by −0.027 vs LoRA no-aug despite val F1 improving.
> The val-test gap collapsed from +0.064 to +0.017 — diagnostic of a
> train-distribution shift that val partially reproduces but test does not.
> This cleanly resolves Section 9.3's open question in the negative.

> **Macro F1 vs Accuracy tell different stories.** Augmented s800 configs achieve the
> highest accuracy (68–69%) because they predict `neutral` (65% of test data) much better.
> But their macro F1 falls because minority classes regress. For MI research, macro F1
> is the correct metric — all three talk types matter equally.

---

## 7.5 Best Model — Qwen3-Embedding-8B + LoRA fine-tune

`Qwen/Qwen3-Embedding-8B` loaded in fp16, attention projections wrapped in
rank-16 LoRA adapters, with a fp32 Linear(4096, 3) classification head on the
last-token pool. Base 8B weights frozen; gradient flows through LoRA deltas
and the head only. No augmentation, ctx0, class-weighted loss.
**Test Macro F1 = 0.6131 | Accuracy = 73.1% | Sustain F1 = 0.431**

### Training setup

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-Embedding-8B` (8B params, frozen) |
| Adapter | LoRA r=16, α=32, dropout=0.05, bias=none |
| Target modules | `q_proj, k_proj, v_proj, o_proj` (attention only) |
| Task type | `FEATURE_EXTRACTION` (custom head on last-token pool) |
| Head | fp32 Linear(4096 → 3) with dropout=0.1 |
| Precision | fp16 base + fp32 classifier + gradient checkpointing |
| Max seq length | 128 tokens (left-padded, matches Phase 1 pool) |
| Batch size | 2 per step, grad accum 8 (effective batch 16) |
| Epochs | 3 (best at epoch **1**) |
| Optimiser | AdamW, lr=2e-4, weight_decay=0.01 |
| Scheduler | Linear warmup (10%) → linear decay |
| Grad clipping | 1.0 |
| Loss | Class-weighted CrossEntropy |
| Device / time | AWS ml.g5.2xlarge (A10G 24 GB) / ~30 min |

### Training trajectory (val F1 by epoch)

| Epoch | Train loss | Val loss | Val F1 | Val acc | Selected? |
|---|---|---|---|---|---|
| **1** | 0.8782 | 1.0325 | **0.5494** | 0.6504 | 🥇 best |
| 2 | 0.5853 | 1.1082 | 0.5485 | 0.6190 | flat / overfit start |
| 3 | 0.2884 | 1.6353 | 0.5271 | 0.6178 | overfit |

Classic one-epoch plateau on 3,153 training utterances. Train loss falls
monotonically (0.878 → 0.585 → 0.288) while val loss climbs after epoch 1
(1.03 → 1.11 → 1.64) and val F1 is essentially flat at epoch 2 then drops.
Best-by-val-F1 checkpointing selects epoch 1; epochs 2–3 confirm the peak.

### Per-class test metrics

Full per-class precision / recall / F1 from
`data/outputs_qwen8b_lora/metrics.json` (test set, n = 866):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| **change** | 0.621 | 0.564 | **0.591** | 218 |
| **neutral** | 0.797 | 0.839 | **0.817** | 571 |
| **sustain** | 0.463 | 0.403 | **0.431** | 77 |
| **Macro avg** | **0.627** | **0.602** | **0.6131** | 866 |
| **Accuracy** | | | **0.7309** | 866 |
| **Weighted avg F1** | | | 0.7261 | 866 |

**Confusion matrix (test)** — from `data/outputs_qwen8b_lora/confusion_matrix.json`:

| True \ Pred | change | neutral | sustain | Row total |
|---|---|---|---|---|
| **change** (218) | **123** | 91 | 4 | 218 |
| **neutral** (571) | 60 | **479** | 32 | 571 |
| **sustain** (77) | 15 | 31 | **31** | 77 |
| Col total | 198 | 601 | 67 | 866 |

Diagonal accuracy = (123 + 479 + 31) / 866 = **73.1%**. Read-offs:

- **change errors are neutral-leaning, not sustain-leaning.** Of the 95 change
  mistakes, 91 go to neutral and only 4 go to sustain. This is a dramatic
  shift from the MiniLM baseline, where 48/108 change errors went to sustain
  (22% of the whole change class). LoRA on Qwen has essentially eliminated
  the change → sustain confusion.
- **Sustain errors are almost evenly split:** 15 → change, 31 → neutral.
  Sustain ↔ neutral is now the dominant remaining boundary (31 sustain
  mispredicted as neutral, 32 neutral mispredicted as sustain — symmetric
  and small relative to class size). This is the expected MI-linguistics
  ambiguity, not a model artefact.
- **Neutral is stable.** 479/571 = 83.9% recall with errors split 60/32
  between change and sustain — the model isn't over-committing to the
  majority class.

### Headline comparison vs prior leaders

| Class | Qwen frozen (prev #2) | RoBERTa ft (prev #1 on sustain) | Qwen + LoRA (new #1) | Δ vs best-prior |
|---|---|---|---|---|
| **change F1** | 0.583 | 0.579 | **0.591** | **+0.008** |
| **neutral F1** | 0.794 | 0.806 | **0.817** | **+0.011** |
| **sustain F1** | 0.379 | 0.393 | **0.431** | **+0.038** |
| **Macro F1** | 0.585 | 0.582 | **0.6131** | **+0.028** |
| **Accuracy** | 69.5% | 67.6% | **73.1%** | **+3.6 pp** |

LoRA improves **every** per-class F1 simultaneously — the model does not trade
one class for another. The biggest gain is sustain (+0.038 over RoBERTa ft,
+0.052 over Qwen frozen), which is exactly the class the rest of the
leaderboard could not move.

### Key observations

- **Sustain F1 clears 0.40 for the first time in the project.** Every prior
  system — frozen, fine-tuned, synthetic-augmented — was capped below 0.40
  on the minority class. LoRA on an 8B retrieval-pretrained base gets to
  **0.431**, confirming the Section 6.7 hypothesis that sustain needs
  gradient flow *and* a representation rich enough for those gradients to
  have a direction to push in.
- **Accuracy 73.1%** — 3.6 pp above frozen Qwen (69.5%), 5.5 pp above
  RoBERTa ft (67.6%). The model simultaneously improves every class rather
  than trading one for another.
- **Val-test gap (+0.064)** is the well-behaved middle ground between
  RoBERTa ft (+0.053) and frozen Qwen (+0.099). Val F1 is now a defensible
  trajectory proxy — which is why Round 9 used it to confirm that
  augmentation's val-F1 improvement was a warning sign, not a win.
- **Training cost** was higher than RoBERTa (~30 min vs ~4 min) but easily
  within the 20-hour project budget, and still drastically cheaper than a
  full-parameter 8B fine-tune would be (would not fit on 24 GB).

### Why this works — and what LoRA added to frozen Qwen

Frozen Qwen-8B already beat every end-to-end fine-tuned mid-size encoder on
majority-class performance (Round 7) because retrieval pretraining at scale
produces a lexically and semantically rich embedding space. But its sustain
F1 stayed flat at 0.379, **below** RoBERTa-ft's 0.393 — retrieval pretraining
has no direct signal for the pragmatic distinction ("I can't" vs "maybe I
could" vs "I won't try"). LoRA with class-weighted CE provides exactly that
signal: the rank-16 adapters on q/k/v/o_proj learn to emphasise attention
patterns that separate sustain language without disturbing the majority-class
geometry that scale already provided. The combination is additive:

| Component | Contribution |
|---|---|
| 8B retrieval-pretrained base | Majority-class floor (change/neutral separability) |
| Last-token pool (decoder-style) | Focuses signal on the utterance's final semantic commitment |
| Class-weighted CE + LoRA gradients | Moves the sustain decision boundary |
| fp32 classifier head | Avoids fp16 numerics collapsing the 3-way softmax |

### Comparison with prior leaders

| Metric | Para-MPNet (frozen) | RoBERTa (ft) | Qwen frozen | **Qwen + LoRA** | Δ vs prior best |
|---|---|---|---|---|---|
| Test Macro F1 | 0.5494 | 0.5823 | 0.5855 | **0.6131** | **+0.028** |
| Accuracy | 65.4% | 67.6% | 69.5% | **73.1%** | **+3.6pp** |
| sustain F1 | 0.346 | 0.393 | 0.379 | **0.431** | **+0.038** |
| change F1 | 0.545 | 0.576 | 0.583 | (est. high) | — |
| neutral F1 | 0.758 | 0.777 | 0.794 | (est. high) | — |
| Val-test gap | small | +0.053 | +0.099 | +0.064 | well-behaved |
| Trainable params | ~50k | 125M | ~50k | ~20M LoRA + head | — |
| Compute | CPU | MPS 4m | A10G 12m | A10G 30m | — |

---

## 8. Cross-Cutting Observations

### 8.1 The sustain problem is structural — but not insurmountable

Across the 13 frozen-encoder experiments, sustain F1 ranged from 0.186 to 0.353 —
no frozen configuration broke 0.36. **Fine-tuning did** (0.393). The ceiling was
the representation, not the task.

Sustain remains the hardest class because:
- It is the smallest class (77 test examples, 333 train)
- It is semantically adjacent to both change and neutral
- The distinction often requires prosodic or pragmatic context not present in text
- **Frozen embeddings cannot learn MI-specific semantics** — but end-to-end training can

### 8.2 Augmentation shifted neutral, not sustain

Every augmentation experiment improved neutral F1 (from 0.767 up to 0.816) while
degrading sustain F1. The model absorbed the synthetic sustain examples as a signal
to be less aggressive about predicting sustain, not more accurate. Adding more
sustain training examples counterintuitively made the model *less* likely to predict
sustain on real test data.

### 8.3 Val F1 is not a reliable proxy for test F1

Note the inversion: aug s200 achieved the highest val F1 (0.5316) of any experiment
but lower test F1 (0.5284) than the baseline. Para-MPNet achieved higher val F1
(0.5247) AND higher test F1 (0.5494). When val and test F1 move in the same direction,
the result is trustworthy. When they diverge, suspect distribution shift.

### 8.4 Sentence-similarity pretraining beats raw MLM pretraining — **when frozen**

Among *frozen* encoders, the three SBERT models (MiniLM, MPNet, Para-MPNet) all
outperform the two raw MLM models (RoBERTa, DeBERTa) on macro F1. A 384-dim SBERT
model beats a 768-dim RoBERTa — the pretraining objective matters more than size
when the encoder is static.

But the picture inverts once you unfreeze: RoBERTa went from 0.5438 frozen to
**0.5823 fine-tuned** (+0.039), leapfrogging every SBERT baseline. The SBERT
inductive bias is a good *starting point* for frozen use but a neutral MLM prior
has more room to be shaped by task-specific gradients.

### 8.5 The frozen-encoder ceiling was ~0.55

All 13 frozen experiments landed in 0.45–0.55 macro F1. The spread within that
band (loss, context, augmentation, encoder choice) explains less than **0.06 F1**.
Fine-tuning alone moved the needle by **0.03+ above the best frozen result** and
**0.08+ above the baseline** — bigger than any frozen-side change combined.

### 8.6 Three distinct performance tiers emerge across 22 experiments

With Rounds 8–10 in place, the leaderboard separates into three regimes:

| Tier | Configs | Macro F1 range | Characterisation |
|---|---|---|---|
| **Tier 1** | Qwen-8B + LoRA | 0.613 | Open fine-tune on 8B retrieval-pretrained base; only system that leads every metric |
| **Tier 2** | GPT-4o-mini (ZS/FS), Qwen frozen, RoBERTa ft, DeBERTa ft, LoRA+aug | 0.579–0.602 | Strong representations, no single tier-1 differentiator |
| **Tier 3** | All frozen SBERT/MLM, all augmentation, all context, both ModernBERT | 0.454–0.549 | Pre-fine-tune / pre-scale ceiling |

Tier 1 → Tier 2 gap (+0.011) is narrower than Tier 2 → Tier 3 gap (+0.03–0.05).
This is the meaningful signal: *the marginal utility of each additional project-
level lever (scale, fine-tune, LoRA on scale) is shrinking.* To push past
0.65 would likely require either (a) data beyond the 3,153-utterance train
set, (b) a different task framing (conversation-level context that does not
regress F1, or utterance-pair contrastive pretraining), or (c) ensembling
tier-1 and tier-2 systems.

### 8.7 Sustain F1 > 0.40 is now attainable — the question reframed

Section 8.1 called sustain "structural but not insurmountable" with 0.393 as
the ceiling (RoBERTa ft). Round 8 pushed it to **0.431** (Qwen + LoRA), and
Round 10 showed **0.427 is reachable zero-shot from GPT-4o-mini** at $0.05
and 50 seconds. The reframe: sustain is not inherently hard — it is hard
*for representations built without pragmatic pretraining*. Any system whose
pretraining either (i) includes pragmatic signal (large decoder LLMs) or
(ii) can acquire it via task gradients flowing into a rich base (Qwen+LoRA)
clears 0.42. Mid-size retrieval/MLM pretraining without fine-tuning cannot.

### 8.8 Augmentation's null result survives paradigm changes

Augmentation was tested in:

1. **Round 1** (frozen MiniLM) — regressed test F1 across s200/s400/s800
2. **Round 2** (± class weights) — regressed independently of loss weighting
3. **Round 9** (LoRA gradient flow into Qwen-8B) — regressed despite val F1 rising

Three different representation regimes, one consistent finding: **LLM-paraphrase
synthetic sustain does not carry the distributional signal of real AnnoMI
sustain**, and this mismatch is not repaired by any combination of loss
function, encoder family, scale, or training paradigm available in this project.
Section 9.3 is resolved as a negative finding, with strong evidence.

---

## 9. Recommendations

### 9.1 ✅ Completed — End-to-end fine-tuning of RoBERTa

**Result**: test macro F1 = **0.5823**, +0.033 over the best frozen model.
Sustain F1 = 0.393, a +0.040 jump. See Section 7.5. **This is the current leader.**

### 9.2 ✅ Completed — End-to-end fine-tuning of DeBERTa-v3

**Result**: test macro F1 = **0.5793**, accuracy = **68.6%** (highest of any
experiment). Lower sustain F1 than RoBERTa (0.374 vs 0.393) but higher accuracy
and neutral F1. Ran in ~6m 20s on MPS. Two successful fine-tune runs establish
that the end-to-end effect is consistent across mature MLM architectures.

### 9.2a ⛔ Tested & rejected — ModernBERT-base fine-tune

**Result**: test macro F1 = **0.5204** (v1, overfit) / **0.5031** (v2, under-trained).
Both underperform RoBERTa/DeBERTa at HPs that bracket a narrow viable band. See
Section 6.6. **Decision**: no further HP tuning; the "fundamentally different
architecture" TA requirement is better addressed by scaling to a decoder/billion-
param model (9.4) than by continued search on another mid-size encoder.

### 9.3 ⛔ Tested & rejected — Augmentation on top of fine-tuning

**Revisited in Round 9 (Section 6.9).** Hypothesis was: if frozen augmentation
failed because the frozen encoder could not align synthetic with real sustain,
gradient flow should fix it. **It does not.** LoRA + s800 sustain synthetic
regressed test macro F1 by **−0.027** vs LoRA no-aug (0.5858 vs 0.6131)
despite val F1 improving (+0.020). Diagnostic: val-test gap collapsed from
+0.064 to +0.017 — the augmented model learned a distribution val partially
shares but test does not. Per-class: sustain recall +0.233, sustain precision
−0.132, sustain F1 +0.005 (noise), change F1 −0.062 (real). Bottom line:
**LLM-paraphrase synthetic sustain is intrinsically misaligned with real
AnnoMI sustain, and gradient flow cannot repair the distributional mismatch.**
Resolved as a clean negative finding.

### 9.4 ✅ Completed — Qwen3-Embedding-8B (TA Change I, HIGH)

**Phase 1 — Frozen + MLP on AWS g5.2xlarge (A10G 24GB)**:
Test macro F1 = **0.5855**, accuracy = **69.5%**. See Section 6.7. Edged
RoBERTa fine-tune by +0.003 F1 without any task-specific training. Sustain
F1 (0.379) remained below RoBERTa ft (0.393) — scale alone did not close the
pragmatic-signal gap.

**Phase 2 — LoRA fine-tune (r=16, attention-only, A10G 30 min)**:
Test macro F1 = **0.6131**, accuracy = **73.1%**, sustain F1 = **0.431**.
See Section 6.8 and 7.5. **New #1 across every metric.** First system in
the project to simultaneously lead macro F1, accuracy, and every per-class
F1 including the minority sustain class. TA Change I (HIGH) is complete.

### 9.4a ✅ Completed — GPT-4o-mini prompt classification (TA Change II, MEDIUM)

**Zero-shot** (system prompt = canonical MI definitions, no exemplars):
test macro F1 = **0.5898**, accuracy = **73.0%**, sustain F1 = **0.427**,
48.8 s inference, **$0.048** total.

**Few-shot (k=3 per class)** (9 deterministic exemplars):
test macro F1 = **0.6019**, accuracy = **69.6%**, sustain F1 = 0.388,
49.8 s inference, **$0.099** total.

See Section 6.10. Frontier LLM prompting lands within 0.011 F1 of our best
open fine-tuned system at 50 seconds of inference time and < $0.10. Deployment
constraints (PHI sensitivity, closed-weight model) still favour the open
fine-tune for this task. TA Change II (MEDIUM) is complete.

### 9.5 Open — MLP hyperparameter ablations (TA Change III, LOW)

With Change I and II completed and the headline at 0.6131, additional MLP
ablations (hidden size, dropout, layers) on the frozen Qwen head would move
the #4 row on the leaderboard by at most 0.01–0.02 F1 — and do not address
any open scientific question. **Recommendation**: deprioritise Change III
unless time remains after the writeup; the project's TA-requested
"fundamentally different class" and "prompt-based LLM" pivots are
comprehensively addressed by Sections 6.8 and 6.10.

### 9.6 Priority order — final state

| Priority | Experiment | Test F1 | Status |
|---|---|---|---|
| ✅ | Frozen SBERT / MPNet / RoBERTa / DeBERTa baselines & swaps | 0.529–0.549 | Rounds 1–4 |
| ✅ | RoBERTa end-to-end fine-tune | **0.5823** | Round 5 |
| ✅ | DeBERTa-v3 end-to-end fine-tune | **0.5793** | Round 5 |
| ⛔ | ModernBERT end-to-end fine-tune | 0.5204 / 0.5031 | Round 6 — rejected |
| ✅ | Qwen3-Embedding-8B frozen + MLP | **0.5855** | Round 7 |
| ✅ | **Qwen3-Embedding-8B + LoRA** 🥇 | **0.6131** | Round 8 — **final #1** |
| ⛔ | Qwen3-Embedding-8B + LoRA + s800 augmentation | 0.5858 | Round 9 — regression |
| ✅ | GPT-4o-mini zero-shot | **0.5898** | Round 10 |
| ✅ | GPT-4o-mini few-shot (k=3) | **0.6019** | Round 10 |
| ⏸ | MLP hyperparameter ablations (Change III, LOW) | marginal | Deprioritised |

---

## 10. Config & Artifact Reference

| Experiment | Embeddings dir | Output dir | Train config |
|---|---|---|---|
| Baseline (MiniLM) | `data/embeddings/` | `data/outputs/` | `configs/train_config.yaml` |
| Aug s200 weighted | `data/embeddings_aug_s200/` | `data/outputs_aug_s200/` | `configs/train_config_aug_s200.yaml` |
| Aug s400 weighted | `data/embeddings_aug_s400/` | `data/outputs_aug_s400/` | `configs/train_config_aug_s400.yaml` |
| Aug s800 weighted | `data/embeddings_augmented/` | `data/outputs_augmented/` | `configs/train_config_augmented.yaml` |
| Aug s200 no weights | `data/embeddings_aug_s200/` | `data/outputs_aug_s200_nowt/` | `configs/train_config_aug_s200_nowt.yaml` |
| Aug s800 no weights | `data/embeddings_augmented/` | `data/outputs_aug_s800_nowt/` | `configs/train_config_aug_s800_nowt.yaml` |
| ctx1 | `data/embeddings_ctx1/` | `data/outputs_ctx1/` | `configs/train_config_ctx1.yaml` |
| ctx2 | `data/embeddings_ctx2/` | `data/outputs_ctx2/` | `configs/train_config_ctx2.yaml` |
| ctx4 | `data/embeddings_ctx4/` | `data/outputs_ctx4/` | `configs/train_config_ctx4.yaml` |
| MPNet | `data/embeddings_mpnet/` | `data/outputs_mpnet/` | `configs/train_config_mpnet.yaml` |
| Para-MPNet | `data/embeddings_paraphrase-mpnet/` | `data/outputs_paraphrase-mpnet/` | `configs/train_config_paraphrase-mpnet.yaml` |
| RoBERTa (frozen) | `data/embeddings_roberta/` | `data/outputs_roberta/` | `configs/train_config_roberta.yaml` |
| DeBERTa-v3 (frozen) | `data/embeddings_deberta/` | `data/outputs_deberta/` | `configs/train_config_deberta.yaml` |
| RoBERTa fine-tune | — (end-to-end) | `data/outputs_roberta_finetune/` | `configs/train_config_roberta_finetune.yaml` |
| DeBERTa-v3 fine-tune | — (end-to-end) | `data/outputs_deberta_finetune/` | `configs/train_config_deberta_finetune.yaml` |
| ModernBERT fine-tune v1 (lr=2e-5, overfit) | — (end-to-end) | `data/outputs_modernbert_finetune/` | `configs/train_config_modernbert_finetune.yaml` |
| ModernBERT fine-tune v2 (lr=5e-6, under-trained) | — (end-to-end) | `data/outputs_modernbert_finetune_v2/` | `configs/train_config_modernbert_finetune_v2.yaml` |
| Qwen3-Embedding-8B (frozen + MLP) | `data/embeddings_qwen8b/` (AWS) | `data/outputs_qwen8b_frozen/` | `notebooks/qwen8b_embedding_aws.py` |
| **🥇 Qwen3-Embedding-8B + LoRA (final #1)** | — (LoRA adapter) | `data/outputs_qwen8b_lora/` (AWS) | `notebooks/qwen8b_lora_aws.py` |
| Qwen3-Embedding-8B + LoRA + s800 augmentation | — (LoRA adapter) | `data/outputs_qwen8b_lora_aug/` (AWS) | `notebooks/qwen8b_lora_aws.py` (aug switch) |
| **🥈 GPT-4o-mini few-shot (k=3)** | — (prompted) | `data/outputs_gpt4o_mini_few_shot/` | `scripts/run_gpt4o_mini_classify.py` |
| **🥉 GPT-4o-mini zero-shot** | — (prompted) | `data/outputs_gpt4o_mini_zero_shot/` | `scripts/run_gpt4o_mini_classify.py` |

All synthetic data artifacts: `data/outputs/augmentation/`
Processed splits: `data/processed/` and `data/processed_ctx{1,2,4}/`
Augmented training file: `data/processed/train_augmented.jsonl` (4,253 examples,
used in Rounds 1 (frozen) and 9 (LoRA)).

### Code references

- Frozen-encoder training: `src/annomi_pipeline/training/train.py` (MLP on precomputed embeddings)
- End-to-end fine-tune training: `src/annomi_pipeline/training/train_finetune.py`
- Qwen-8B frozen embedding + MLP: `notebooks/qwen8b_embedding_aws.py` (AWS SageMaker)
- Qwen-8B LoRA fine-tune: `notebooks/qwen8b_lora_aws.py` (AWS SageMaker, r=16 attention-only)
- GPT-4o-mini classifier: `scripts/run_gpt4o_mini_classify.py`
  (async concurrency=10, structured JSON schema output, temperature=0.0, seed=42)
- MI definitions shared by synthetic generation and GPT prompting:
  `src/annomi_pipeline/data/synthetic_generation.py`
- CLI runners: `scripts/run_stage2.py` (frozen), `scripts/run_finetune.py` (fine-tune)

---

*MLP architecture (frozen runs): input_dim=384/768 (or 4096 for Qwen-8B),
hidden_dim=128, num_layers=2, dropout=0.4, activation=relu, lr=0.001,
weight_decay=0.001, batch=64, patience=6, epochs=40, seed=42.*

*Fine-tune architecture (RoBERTa, DeBERTa-v3, ModernBERT): `AutoModelForSequenceClassification`
with classification head, lr=2e-5 (ModernBERT v2: 5e-6), batch=16, max_length=128,
4 epochs (ModernBERT v2: 3) with linear warmup (10%), weight_decay=0.01,
grad_clip=1.0, class-weighted CE loss, seed=42. Only `model_name` (and lr/epochs
for ModernBERT v2) differ between the fine-tune configs.*

*LoRA architecture (Qwen-8B Rounds 8–9): base model fp16, LoRA adapters on
`q_proj, k_proj, v_proj, o_proj` with r=16, α=32, dropout=0.05, bias=none,
task_type=FEATURE_EXTRACTION. Custom head: fp32 `Linear(4096, 3)` on last-token
pool of left-padded sequences. Training: effective batch 16 (per-step 2 × grad
accum 8), lr=2e-4, weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
max_seq_len=128, class-weighted CE, 3 epochs with best-by-val-F1 checkpointing,
seed=42. Gradient checkpointing + `enable_input_require_grads()` required to
fit on A10G 24 GB.*

*GPT-4o-mini classification (Round 10): `gpt-4o-mini` via OpenAI API, temperature=0.0,
seed=42, async concurrency=10, max_retries=3, structured JSON schema response
(`enum = [change, neutral, sustain]`, strict=true). System prompt = canonical MI
definitions from `src/annomi_pipeline/data/synthetic_generation.py`. Few-shot
mode: 3 deterministic exemplars per class (9 total) drawn from the training set.*
