"""
Qwen3-Embedding-8B frozen + MLP on AnnoMI (client talk-type classification).

Intended to be run cell-by-cell in a SageMaker notebook (kernel: conda_pytorch_p310).
Paste each `# %%`-delimited block into a separate Jupyter cell in order.

Hardware: AWS ml.g5.2xlarge (NVIDIA A10G 24 GB). Qwen3-Embedding-8B in fp16 is ~16 GB
of weights; activations for seq_len<=128 and batch_size=8 leave comfortable headroom.

Before starting, upload these three files into the notebook's working directory
(Jupyter Files tab → Upload):
    - train.jsonl
    - val.jsonl
    - test.jsonl
(Local paths: Code/data/processed/{train,val,test}.jsonl)

Expected runtime end-to-end: ~12 min (≈10 min embedding extraction, ≈2 min MLP).
"""

# %% [markdown]
# ## Cell 1 — Install dependencies
# `torch` is already in `conda_pytorch_p310`. We add `sentence-transformers` (latest),
# `transformers` pinned to a version that supports Qwen3 (>=4.51), and helpers.

# %%
!pip install -q --upgrade "sentence-transformers>=3.0.0" "transformers>=4.51.0" "huggingface_hub>=0.24" scikit-learn tqdm

# %% [markdown]
# ## Cell 2 — Login to Hugging Face
# Qwen3-Embedding-8B is Apache-2.0 and not gated, but authenticating is still useful
# for avoiding rate limits on the large (~16 GB) download.

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
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
LABEL_FIELD = "metadata.client_talk_type"
CLASS_NAMES = ["change", "neutral", "sustain"]           # fixed order → integer labels 0,1,2
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Paths (relative to notebook working dir)
TRAIN_JSONL = "train.jsonl"
VAL_JSONL   = "val.jsonl"
TEST_JSONL  = "test.jsonl"
OUT_DIR = Path("outputs_qwen8b_frozen"); OUT_DIR.mkdir(exist_ok=True)
EMB_DIR = Path("embeddings_qwen8b");     EMB_DIR.mkdir(exist_ok=True)

# Encoder
EMBED_BATCH_SIZE = 8           # conservative; A10G has 24 GB, increase to 16 if headroom
MAX_SEQ_LENGTH   = 128

# MLP head (mirrors Round-4 frozen baseline exactly → apples-to-apples encoder swap)
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT      = 0.4
ACTIVATION   = "relu"
LR           = 1e-3
WEIGHT_DECAY = 1e-3
MLP_BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE   = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Cell 4 — Load splits
# Reads the three JSONLs, extracts `text` and the nested `metadata.client_talk_type`
# label, maps labels to integer indices.

# %%
def _get_nested(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        cur = cur[part]
    return cur

def load_split(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            label_str = _get_nested(rec, LABEL_FIELD)
            if label_str not in CLASS_TO_IDX:
                continue  # skip any rows with unexpected labels
            texts.append(rec["text"])
            labels.append(CLASS_TO_IDX[label_str])
    return texts, np.array(labels, dtype=np.int64)

train_texts, train_y = load_split(TRAIN_JSONL)
val_texts,   val_y   = load_split(VAL_JSONL)
test_texts,  test_y  = load_split(TEST_JSONL)

def dist(y):
    return {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)}

print(f"train: {len(train_texts):>5}  {dist(train_y)}")
print(f"val:   {len(val_texts):>5}  {dist(val_y)}")
print(f"test:  {len(test_texts):>5}  {dist(test_y)}")

# %% [markdown]
# ## Cell 5 — Load Qwen3-Embedding-8B
# Loads in fp16 onto GPU. First call downloads ~16 GB of weights from HF hub
# (1–3 min on SageMaker). Subsequent loads are cached under `~/.cache/huggingface/`.
#
# Qwen3-Embedding is a `sentence-transformers`-compatible model — no manual pooling
# or instruction templating is needed for classification (instruction prompts are
# only for retrieval tasks). We just call `.encode(texts)`.

# %%
from sentence_transformers import SentenceTransformer

t0 = time.time()
model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "eager"},
    tokenizer_kwargs={"padding_side": "left"},  # Qwen3-Embedding expects left-padding
)
model.max_seq_length = MAX_SEQ_LENGTH
model.to(DEVICE)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s  |  embed dim = {model.get_sentence_embedding_dimension()}")

# Quick memory check
if DEVICE == "cuda":
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after model load: {mem:.1f} GB")

# %% [markdown]
# ## Cell 6 — Extract embeddings for all three splits
# Single pass, no gradients. Output shape per split: (N, 4096).

# %%
@torch.inference_mode()
def embed_texts(texts, desc: str):
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # normalize is for cosine retrieval; keep raw for MLP
    )
    print(f"  {desc}: {embs.shape}  |  {time.time()-t0:.1f}s")
    return embs.astype(np.float32)

train_X = embed_texts(train_texts, "train")
val_X   = embed_texts(val_texts,   "val")
test_X  = embed_texts(test_texts,  "test")

print(f"\nEmbedding dim: {train_X.shape[1]}")

# %% [markdown]
# ## Cell 7 — Save embeddings
# Persist so we can rerun MLP training without re-embedding. Also so LoRA-phase
# work can reuse the same splits if we still want frozen comparison later.

# %%
np.save(EMB_DIR / "train_X.npy", train_X); np.save(EMB_DIR / "train_y.npy", train_y)
np.save(EMB_DIR / "val_X.npy",   val_X);   np.save(EMB_DIR / "val_y.npy",   val_y)
np.save(EMB_DIR / "test_X.npy",  test_X);  np.save(EMB_DIR / "test_y.npy",  test_y)
print(f"Saved embeddings to {EMB_DIR}/")

# Free the encoder's ~16 GB before training the MLP head
del model
torch.cuda.empty_cache()
print(f"After freeing encoder: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

# %% [markdown]
# ## Cell 8 — MLP definition (matches `src/annomi_pipeline/training/mlp_model.py` exactly)

# %%
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, activation, num_classes):
        super().__init__()
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        layers = []
        cur = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(cur, hidden_dim), act_map[activation](), nn.Dropout(dropout)]
            cur = hidden_dim
        layers.append(nn.Linear(cur, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)

# %% [markdown]
# ## Cell 9 — Train the MLP head (class-weighted CE, early stopping on val macro F1)

# %%
def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    per_class = {
        str(i): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i in range(NUM_CLASSES)
    }
    return {"accuracy": float(acc), "f1_macro": float(f1m), "f1_weighted": float(f1w), "per_class": per_class}

train_loader = make_loader(train_X, train_y, MLP_BATCH_SIZE, shuffle=True)
val_loader   = make_loader(val_X,   val_y,   MLP_BATCH_SIZE, shuffle=False)
test_loader  = make_loader(test_X,  test_y,  MLP_BATCH_SIZE, shuffle=False)

head = MLPClassifier(
    input_dim=train_X.shape[1],
    hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
    dropout=DROPOUT, activation=ACTIVATION, num_classes=NUM_CLASSES,
).to(DEVICE)

class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_y)
print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
optim   = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_f1 = -1.0
best_epoch = -1
best_state = None
bad = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    head.train()
    tot = 0.0; n = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        logits = head(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        tot += loss.item() * xb.size(0); n += xb.size(0)
    train_loss = tot / n

    head.eval()
    preds = []; truths = []
    vtot = 0.0; vn = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = head(xb)
            vtot += loss_fn(logits, yb).item() * xb.size(0); vn += xb.size(0)
            preds.append(logits.argmax(-1).cpu().numpy())
            truths.append(yb.cpu().numpy())
    val_loss = vtot / vn
    val_pred = np.concatenate(preds); val_truth = np.concatenate(truths)
    val_m = compute_metrics(val_truth, val_pred)

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k,v in val_m.items() if k != "per_class"}})
    print(f"ep {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_F1macro={val_m['f1_macro']:.4f}  val_acc={val_m['accuracy']:.4f}")

    if val_m["f1_macro"] > best_val_f1:
        best_val_f1 = val_m["f1_macro"]
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nBest epoch: {best_epoch}  |  best val F1 macro: {best_val_f1:.4f}")
head.load_state_dict(best_state)

# %% [markdown]
# ## Cell 10 — Evaluate on test, write metrics & confusion matrix

# %%
head.eval()
preds = []; truths = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = head(xb)
        preds.append(logits.argmax(-1).cpu().numpy())
        truths.append(yb.numpy())
test_pred = np.concatenate(preds); test_truth = np.concatenate(truths)
test_m = compute_metrics(test_truth, test_pred)
cm = confusion_matrix(test_truth, test_pred, labels=list(range(NUM_CLASSES)))

print("\n=== TEST RESULTS ===")
print(f"Accuracy:       {test_m['accuracy']:.4f}")
print(f"Macro F1:       {test_m['f1_macro']:.4f}")
print(f"Weighted F1:    {test_m['f1_weighted']:.4f}")
print("\nPer-class:")
print(classification_report(test_truth, test_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print("          " + "  ".join(f"{c:>8}" for c in CLASS_NAMES))
for i, c in enumerate(CLASS_NAMES):
    print(f"{c:>8}  " + "  ".join(f"{cm[i,j]:>8d}" for j in range(NUM_CLASSES)))

metrics_payload = {
    "model_name": MODEL_NAME,
    "device": DEVICE,
    "label_field": LABEL_FIELD,
    "class_names": CLASS_NAMES,
    "best_epoch": best_epoch,
    "validation": {
        "epoch": best_epoch,
        **compute_metrics(val_truth, val_pred),
    },
    "test": test_m,
    "train_history": history,
    "mlp_config": {
        "input_dim": int(train_X.shape[1]),
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "activation": ACTIVATION,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "batch_size": MLP_BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "class_weighted_loss": True,
    },
    "encoder_config": {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "dtype": "float16",
    },
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

cm_payload = {
    "class_names": CLASS_NAMES,
    "matrix": cm.tolist(),
}
(OUT_DIR / "confusion_matrix.json").write_text(json.dumps(cm_payload, indent=2))

torch.save({
    "state_dict": head.state_dict(),
    "input_dim": int(train_X.shape[1]),
    "num_classes": NUM_CLASSES,
    "model_config": {
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "dropout": DROPOUT, "activation": ACTIVATION,
    },
}, OUT_DIR / "mlp_head.pt")

print(f"\nWrote: {OUT_DIR/'metrics.json'}")
print(f"Wrote: {OUT_DIR/'confusion_matrix.json'}")
print(f"Wrote: {OUT_DIR/'mlp_head.pt'}")

# %% [markdown]
# ## Cell 11 — Quick comparison to RoBERTa fine-tune leader (sanity check)

# %%
leaders = {
    "Baseline MiniLM (frozen+MLP)":    0.5465,
    "Para-MPNet (frozen+MLP, best frozen)": 0.5494,
    "RoBERTa fine-tune (best overall)": 0.5823,
    "DeBERTa-v3 fine-tune":             0.5793,
    "Qwen3-Embedding-8B (frozen+MLP, this run)": test_m["f1_macro"],
}
print("\n=== Test macro F1 comparison ===")
for name, val in sorted(leaders.items(), key=lambda kv: -kv[1]):
    marker = "  ←" if name.startswith("Qwen") else ""
    print(f"  {val:.4f}   {name}{marker}")

verdict = test_m["f1_macro"]
if verdict >= 0.60:
    print("\n✅ Phase 1 strong result. Phase 2 (LoRA fine-tune) is well-motivated.")
elif verdict >= 0.55:
    print("\n🟡 Phase 1 beats frozen-encoder ceiling but not the fine-tune leaders. Phase 2 still worth trying.")
else:
    print("\n⛔ Phase 1 below frozen-encoder ceiling. Scale did not help as a frozen encoder — "
          "reconsider whether Phase 2 (LoRA) justifies the 2h cost.")

# %% [markdown]
# ## Cell 12 — Download results back to local machine
# From the Jupyter **Files** tab, download:
#   - `outputs_qwen8b_frozen/metrics.json`
#   - `outputs_qwen8b_frozen/confusion_matrix.json`
#   - `outputs_qwen8b_frozen/mlp_head.pt`        (optional — needed for re-evaluation)
#   - `embeddings_qwen8b/*.npy`                   (optional — ~75 MB total, useful if
#                                                  we want to re-train MLP locally or for LoRA phase)
#
# Copy into the local repo under:
#   - `data/outputs_qwen8b_frozen/`
#   - `data/embeddings_qwen8b/`
#
# Then ping Claude to update `EXPERIMENTS_REPORT.md` with the new row and decide
# whether Phase 2 (LoRA fine-tune) goes forward.
