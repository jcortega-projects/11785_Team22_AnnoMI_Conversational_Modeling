"""
Qwen3-Embedding-8B + LoRA fine-tune on AnnoMI (client talk-type classification).

Phase 2 of the Qwen experiment. Picks up from Phase 1 (frozen + MLP, 0.5855 test F1).
Hypothesis: LoRA adapters on attention projections let task gradients reshape the
top layers enough to close the sustain-F1 gap with RoBERTa fine-tune (0.393 vs
our 0.379 frozen) while preserving Qwen's majority-class lead.

Hardware: AWS ml.g5.2xlarge (NVIDIA A10G 24 GB).

Prerequisite: train.jsonl, val.jsonl, test.jsonl already uploaded to the notebook
working directory during Phase 1. If you relaunched the instance, re-upload them.

Expected runtime: ~60–90 min total (fp16 base + gradient checkpointing + batch 2
with grad_accum 8, 3 epochs).
"""

# %% [markdown]
# ## Cell 1 — Install dependencies
# `peft` for LoRA, `accelerate` for clean device handling, `bitsandbytes` not needed
# (we use plain fp16, not 4-bit quant).

# %%
!pip install -q --upgrade "transformers>=4.51.0" "peft>=0.11.0" "accelerate>=0.30.0" "huggingface_hub>=0.24" scikit-learn tqdm

# %% [markdown]
# ## Cell 2 — Login to Hugging Face (skip if already cached from Phase 1)

# %%
import os
from getpass import getpass
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    hf_token = getpass("Hugging Face token: ").strip()
login(token=hf_token)

# %% [markdown]
# ## Cell 3 — Imports & configuration

# %%
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
LABEL_FIELD = "metadata.client_talk_type"
CLASS_NAMES = ["change", "neutral", "sustain"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

TRAIN_JSONL = "train.jsonl"; VAL_JSONL = "val.jsonl"; TEST_JSONL = "test.jsonl"
OUT_DIR = Path("outputs_qwen8b_lora"); OUT_DIR.mkdir(exist_ok=True)

# Encoder
MAX_SEQ_LENGTH = 128

# LoRA
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention only

# Training
EPOCHS           = 3
BATCH_SIZE       = 2            # micro-batch; A10G can't fit more with grad checkpointing off
GRAD_ACCUM_STEPS = 8            # effective batch = 16
LR               = 2e-4         # 10× full-FT lr; LoRA standard
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
MAX_GRAD_NORM    = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Cell 4 — Data loading & tokenization

# %%
def _get_nested(d, dotted):
    cur = d
    for p in dotted.split("."):
        cur = cur[p]
    return cur

def load_split(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lab = _get_nested(rec, LABEL_FIELD)
            if lab not in CLASS_TO_IDX:
                continue
            texts.append(rec["text"])
            labels.append(CLASS_TO_IDX[lab])
    return texts, np.array(labels, dtype=np.int64)

train_texts, train_y = load_split(TRAIN_JSONL)
val_texts,   val_y   = load_split(VAL_JSONL)
test_texts,  test_y  = load_split(TEST_JSONL)

def dist(y):
    return {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)}
print(f"train: {len(train_texts):>5}  {dist(train_y)}")
print(f"val:   {len(val_texts):>5}  {dist(val_y)}")
print(f"test:  {len(test_texts):>5}  {dist(test_y)}")

# Tokenizer — LEFT padding (Qwen decoder-style last-token pooling)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    print(f"Set pad_token -> eos_token ({tok.eos_token_id})")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels
        self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding=False, return_tensors=None)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": int(self.labels[i])}

def collate_fn(batch):
    # Dynamic left-padding per batch for efficiency
    input_ids = [b["input_ids"] for b in batch]
    masks     = [b["attention_mask"] for b in batch]
    labels    = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    max_len = max(len(x) for x in input_ids)
    pad_id = tok.pad_token_id
    padded_ids, padded_masks = [], []
    for ids, m in zip(input_ids, masks):
        pad = max_len - len(ids)
        padded_ids.append([pad_id]*pad + ids)
        padded_masks.append([0]*pad + m)
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        "labels": labels,
    }

train_ds = TextDataset(train_texts, train_y, tok, MAX_SEQ_LENGTH)
val_ds   = TextDataset(val_texts,   val_y,   tok, MAX_SEQ_LENGTH)
test_ds  = TextDataset(test_texts,  test_y,  tok, MAX_SEQ_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# ## Cell 5 — Load base model (fp16) + enable gradient checkpointing

# %%
t0 = time.time()
base = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
base.gradient_checkpointing_enable()
base.enable_input_require_grads()   # needed with grad ckpt + LoRA
print(f"Base model loaded in {time.time()-t0:.1f}s")
print(f"Hidden size: {base.config.hidden_size}  |  Num layers: {base.config.num_hidden_layers}")

# %% [markdown]
# ## Cell 6 — Wrap with LoRA (attention projections only)

# %%
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,  # we put our own head on top
)
base_lora = get_peft_model(base, lora_cfg)
base_lora.print_trainable_parameters()

# %% [markdown]
# ## Cell 7 — Classification head + full forward module

# %%
class QwenLoRAClassifier(nn.Module):
    """Qwen base (with LoRA) → last-token pool → Linear head."""
    def __init__(self, base_with_lora, hidden_size, num_classes):
        super().__init__()
        self.base = base_with_lora
        # Head in fp32 for numerical stability
        self.classifier = nn.Linear(hidden_size, num_classes).to(torch.float32)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, output_hidden_states=False)
        hidden = out.last_hidden_state          # (B, T, H) in fp16
        # With left-padding the last real token is always at position -1
        pooled = hidden[:, -1, :].to(torch.float32)
        logits = self.classifier(pooled)
        return logits

model = QwenLoRAClassifier(base_lora, base.config.hidden_size, NUM_CLASSES).to(DEVICE)

# Trainable params summary
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,}  |  Total: {total:,}  |  %: {100*trainable/total:.3f}")

# %% [markdown]
# ## Cell 8 — Optimizer, scheduler, loss

# %%
class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_y)
print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))

# Only pass trainable (LoRA + classifier head) params to optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
total_steps  = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"Optimizer steps/epoch: {steps_per_epoch}  |  total: {total_steps}  |  warmup: {warmup_steps}")

# %% [markdown]
# ## Cell 9 — Eval helper

# %%
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
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

@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_pred = []; all_true = []; tot_loss = 0.0; n = 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lab  = batch["labels"].to(DEVICE)
        logits = model(ids, mask)
        tot_loss += loss_fn(logits, lab).item() * lab.size(0); n += lab.size(0)
        all_pred.append(logits.argmax(-1).cpu().numpy())
        all_true.append(lab.cpu().numpy())
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    m = compute_metrics(y_true, y_pred)
    m["loss"] = tot_loss / n
    return m, y_pred, y_true

# %% [markdown]
# ## Cell 10 — Train (main loop)

# %%
best_val_f1 = -1.0; best_epoch = -1
best_lora_state = None; best_head_state = None
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0; seen = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True)

    for step, batch in pbar:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lab  = batch["labels"].to(DEVICE)

        logits = model(ids, mask)
        loss = loss_fn(logits, lab) / GRAD_ACCUM_STEPS
        loss.backward()

        running += loss.item() * GRAD_ACCUM_STEPS * lab.size(0); seen += lab.size(0)
        pbar.set_postfix(loss=f"{running/seen:.4f}")

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    train_loss = running / seen
    val_m, _, _ = evaluate(val_loader)
    print(f"epoch {epoch}: train_loss={train_loss:.4f}  "
          f"val_loss={val_m['loss']:.4f}  val_F1macro={val_m['f1_macro']:.4f}  val_acc={val_m['accuracy']:.4f}")
    history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k,v in val_m.items() if k != "per_class"}})

    if val_m["f1_macro"] > best_val_f1:
        best_val_f1 = val_m["f1_macro"]; best_epoch = epoch
        # Save LoRA adapters + head only (not the 16GB base)
        best_lora_state = {k: v.detach().cpu().clone()
                           for k, v in model.base.state_dict().items() if "lora" in k}
        best_head_state = {k: v.detach().cpu().clone()
                           for k, v in model.classifier.state_dict().items()}
        print(f"   ↑ new best val F1 ({best_val_f1:.4f}) at epoch {epoch}")

print(f"\nBest epoch: {best_epoch}  |  Best val F1 macro: {best_val_f1:.4f}")

# %% [markdown]
# ## Cell 11 — Restore best weights & evaluate on test

# %%
# Load best LoRA adapters + head back into the model
with torch.no_grad():
    for k, v in best_lora_state.items():
        model.base.state_dict()[k].copy_(v.to(model.base.state_dict()[k].device))
    model.classifier.load_state_dict(best_head_state)

test_m, test_pred, test_true = evaluate(test_loader)
val_m_best, val_pred, val_true = evaluate(val_loader)
cm = confusion_matrix(test_true, test_pred, labels=list(range(NUM_CLASSES)))

print("\n=== TEST RESULTS ===")
print(f"Accuracy:    {test_m['accuracy']:.4f}")
print(f"Macro F1:    {test_m['f1_macro']:.4f}")
print(f"Weighted F1: {test_m['f1_weighted']:.4f}")
print("\nPer-class:")
print(classification_report(test_true, test_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print("          " + "  ".join(f"{c:>8}" for c in CLASS_NAMES))
for i, c in enumerate(CLASS_NAMES):
    print(f"{c:>8}  " + "  ".join(f"{cm[i,j]:>8d}" for j in range(NUM_CLASSES)))

# %% [markdown]
# ## Cell 12 — Save artifacts (metrics JSON, LoRA adapters, head)

# %%
metrics_payload = {
    "model_name": MODEL_NAME,
    "device": DEVICE,
    "label_field": LABEL_FIELD,
    "class_names": CLASS_NAMES,
    "best_epoch": best_epoch,
    "validation": {"epoch": best_epoch, **val_m_best},
    "test": test_m,
    "train_history": history,
    "lora_config": {
        "r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGETS, "bias": "none",
    },
    "training_config": {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO, "max_grad_norm": MAX_GRAD_NORM,
        "max_seq_length": MAX_SEQ_LENGTH, "class_weighted_loss": True,
        "gradient_checkpointing": True, "dtype": "float16",
    },
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
(OUT_DIR / "confusion_matrix.json").write_text(json.dumps(
    {"class_names": CLASS_NAMES, "matrix": cm.tolist()}, indent=2))

# Save LoRA adapters via PEFT's serialization (small, few MB)
model.base.save_pretrained(OUT_DIR / "lora_adapters")
# Save classifier head separately
torch.save(model.classifier.state_dict(), OUT_DIR / "classifier_head.pt")

print(f"\nWrote: {OUT_DIR/'metrics.json'}")
print(f"Wrote: {OUT_DIR/'confusion_matrix.json'}")
print(f"Wrote: {OUT_DIR/'lora_adapters/'}")
print(f"Wrote: {OUT_DIR/'classifier_head.pt'}")

# %% [markdown]
# ## Cell 13 — Leaderboard comparison (auto-verdict)

# %%
leaders = {
    "Baseline MiniLM (frozen+MLP)":            0.5465,
    "Para-MPNet (frozen+MLP, best frozen)":    0.5494,
    "RoBERTa fine-tune":                       0.5823,
    "DeBERTa-v3 fine-tune":                    0.5793,
    "Qwen3-Emb-8B frozen+MLP (Phase 1)":       0.5855,
    "Qwen3-Emb-8B + LoRA (this run, Phase 2)": test_m["f1_macro"],
}
print("\n=== Test macro F1 ===")
for name, val in sorted(leaders.items(), key=lambda kv: -kv[1]):
    marker = "  ←" if "Phase 2" in name else ""
    print(f"  {val:.4f}   {name}{marker}")

delta_vs_p1   = test_m["f1_macro"] - 0.5855
delta_vs_roberta = test_m["f1_macro"] - 0.5823

# Sustain is the key minority-class signal
sustain_f1 = test_m["per_class"]["2"]["f1"]
sustain_vs_roberta = sustain_f1 - 0.393

print(f"\nΔ vs Phase 1 (Qwen frozen): {delta_vs_p1:+.4f}")
print(f"Δ vs RoBERTa fine-tune:     {delta_vs_roberta:+.4f}")
print(f"Sustain F1: {sustain_f1:.4f}  (Δ vs RoBERTa ft: {sustain_vs_roberta:+.4f})")

if delta_vs_p1 >= 0.02 and sustain_f1 >= 0.393:
    print("\n✅ LoRA win: beats Phase 1 by ≥0.02 F1 AND matches/exceeds RoBERTa on sustain. "
          "This is the headline result.")
elif delta_vs_p1 >= 0.01:
    print("\n🟢 Moderate LoRA win: beats Phase 1, modest gain. Include as new #1 in report.")
elif delta_vs_p1 >= 0:
    print("\n🟡 LoRA marginal: ≤0.01 F1 over Phase 1. Frozen encoding was already near-ceiling "
          "for this encoder on this task.")
else:
    print("\n⛔ LoRA regression: worse than frozen Phase 1. Possible overfitting on sustain-low "
          "val split. Consider: lower LR (1e-4), fewer epochs, or different LoRA targets.")

# %% [markdown]
# ## Cell 14 — Download back to local
# Files to download from Jupyter Files tab → `outputs_qwen8b_lora/`:
#   - `metrics.json`
#   - `confusion_matrix.json`
#   - `lora_adapters/`             (adapter_config.json + adapter_model.safetensors)
#   - `classifier_head.pt`
#
# Copy to local repo under: `data/outputs_qwen8b_lora/`
