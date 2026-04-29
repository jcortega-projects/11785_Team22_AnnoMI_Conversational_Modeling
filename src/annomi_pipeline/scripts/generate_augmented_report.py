"""Generate Augmented_Report.md comparing baseline vs augmented pipeline runs.

Reads metrics.json and confusion_matrix.json from both output directories,
then writes a Markdown report to the project root.

Usage
-----
    python -m annomi_pipeline.scripts.generate_augmented_report \\
        --baseline-dir  data/outputs \\
        --augmented-dir data/outputs_augmented \\
        --output        Augmented_Report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _delta(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.4f}"


def _pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def _f(val: float) -> str:
    return f"{val:.4f}"


def _cm_table(cm: dict) -> str:
    names  = cm["class_names"]
    matrix = cm["matrix"]
    n = len(names)
    header = "| True \\ Pred |" + "".join(f" **{n}** |" for n in names)
    sep    = "|---|" + "---|" * n
    rows   = []
    for i, true_name in enumerate(names):
        cells = [f" **{true_name}** |"] + [f" {matrix[i][j]} |" for j in range(n)]
        rows.append("|" + "".join(cells))
    return "\n".join([header, sep] + rows)


def _per_class_table(metrics: dict, class_names: list[str], label: str) -> str:
    pc = metrics.get("test", metrics).get("per_class", {})
    if not pc:
        return "_Per-class data not available._"
    header = f"| Class | Precision | Recall | F1 | Support |"
    sep    = "|---|---|---|---|---|"
    rows   = []
    for i, name in enumerate(class_names):
        d = pc.get(str(i), {})
        rows.append(
            f"| **{name}** | {_f(d.get('precision',0))} | {_f(d.get('recall',0))} "
            f"| {_f(d.get('f1',0))} | {d.get('support','-')} |"
        )
    return "\n".join([header, sep] + rows)


def _delta_per_class_table(
    base_metrics: dict,
    aug_metrics: dict,
    class_names: list[str],
) -> str:
    bpc = base_metrics.get("test", base_metrics).get("per_class", {})
    apc = aug_metrics.get("test", aug_metrics).get("per_class", {})
    if not bpc or not apc:
        return "_Per-class data not available._"
    header = "| Class | F1 Baseline | F1 Augmented | Δ F1 | Recall Baseline | Recall Augmented | Δ Recall |"
    sep    = "|---|---|---|---|---|---|---|"
    rows   = []
    for i, name in enumerate(class_names):
        b = bpc.get(str(i), {})
        a = apc.get(str(i), {})
        bf1  = b.get("f1", 0);    af1  = a.get("f1", 0)
        brec = b.get("recall",0); arec = a.get("recall",0)
        rows.append(
            f"| **{name}** | {_f(bf1)} | {_f(af1)} | {_delta(bf1,af1)} "
            f"| {_f(brec)} | {_f(arec)} | {_delta(brec,arec)} |"
        )
    return "\n".join([header, sep] + rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir",  default="data/outputs")
    p.add_argument("--augmented-dir", default="data/outputs_augmented")
    p.add_argument("--output", default="Augmented_Report.md")
    p.add_argument("--baseline-train",  default="data/processed/train.jsonl")
    p.add_argument("--augmented-train", default="data/processed/train_augmented.jsonl")
    p.add_argument("--synthetic-qa",    default="data/outputs/augmentation/synthetic_candidates_qa.jsonl")
    return p.parse_args()


def _count_labels(jsonl_path: Path) -> dict[str, int]:
    counts: Counter = Counter()
    if not jsonl_path.exists():
        return {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                lbl = row.get("label") or (row.get("metadata") or {}).get("client_talk_type")
                if lbl:
                    counts[lbl] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()

    base_dir = Path(args.baseline_dir).expanduser().resolve()
    aug_dir  = Path(args.augmented_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    base_metrics = _load(base_dir / "metrics.json")
    aug_metrics  = _load(aug_dir  / "metrics.json")
    base_cm      = _load(base_dir / "confusion_matrix.json")
    aug_cm       = _load(aug_dir  / "confusion_matrix.json")

    class_names = base_metrics["class_names"]
    bt = base_metrics["test"]
    at = aug_metrics["test"]
    bv = base_metrics["validation"]
    av = aug_metrics["validation"]

    # Training set distribution
    base_train_dist = _count_labels(Path(args.baseline_train))
    aug_train_dist  = _count_labels(Path(args.augmented_train))

    # Synthetic QA stats
    syn_stats: dict[str, int] = {}
    syn_path = Path(args.synthetic_qa)
    if syn_path.exists():
        with syn_path.open() as f:
            syn_rows = [json.loads(l) for l in f if l.strip()]
        syn_stats["total"]    = len(syn_rows)
        syn_stats["accepted"] = sum(1 for r in syn_rows if r.get("accepted_for_training"))
        syn_stats["rejected"] = sum(1 for r in syn_rows if r.get("verification_status") == "auto_rejected")
        syn_stats["failed"]   = sum(1 for r in syn_rows if r.get("verification_status") == "verification_failed")
        syn_stats["sustain"]  = sum(1 for r in syn_rows if r.get("accepted_for_training") and r.get("label")=="sustain")
        syn_stats["change"]   = sum(1 for r in syn_rows if r.get("accepted_for_training") and r.get("label")=="change")

    report = f"""# Augmented Pipeline Report
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

---

## 1. Overview

This report compares the **baseline MLP + SBERT pipeline** (original training data only)
against the **augmented pipeline** (original data + o4-mini synthetic candidates for
`sustain` and `change` classes).

The target task is **client talk-type classification** on the AnnoMI dataset:
`change` | `neutral` | `sustain` — evaluated with **macro F1** as the primary metric.

---

## 2. Training Set Composition

### Baseline training set
| Label   | Count |
|---------|-------|
| change  | {base_train_dist.get('change','-')} |
| neutral | {base_train_dist.get('neutral','-')} |
| sustain | {base_train_dist.get('sustain','-')} |
| **Total** | **{sum(base_train_dist.values())}** |

### Augmented training set (after QA filtering)
| Label   | Count |
|---------|-------|
| change  | {aug_train_dist.get('change','-')} |
| neutral | {aug_train_dist.get('neutral','-')} |
| sustain | {aug_train_dist.get('sustain','-')} |
| **Total** | **{sum(aug_train_dist.values())}** |

### Synthetic generation summary
| Stat | Value |
|------|-------|
| Synthetic candidates generated | {syn_stats.get('total', 'N/A')} |
| Accepted (QA-verified) | {syn_stats.get('accepted', 'N/A')} |
| Auto-rejected (label mismatch) | {syn_stats.get('rejected', 'N/A')} |
| Failed QA calls | {syn_stats.get('failed', 'N/A')} |
| Accepted sustain | {syn_stats.get('sustain', 'N/A')} |
| Accepted change  | {syn_stats.get('change', 'N/A')} |

*Validation and test splits were NOT touched. All synthetic rows were train-only.*

---

## 3. Headline Metrics

### Test set (held-out)

| Metric | Baseline | Augmented | Δ |
|---|---|---|---|
| **Macro F1** | **{_f(bt['f1_macro'])}** | **{_f(at['f1_macro'])}** | **{_delta(bt['f1_macro'], at['f1_macro'])}** |
| Weighted F1 | {_f(bt['f1_weighted'])} | {_f(at['f1_weighted'])} | {_delta(bt['f1_weighted'], at['f1_weighted'])} |
| Accuracy    | {_pct(bt['accuracy'])} | {_pct(at['accuracy'])} | {_delta(bt['accuracy'], at['accuracy'])} |
| Macro Precision | {_f(bt['precision_macro'])} | {_f(at['precision_macro'])} | {_delta(bt['precision_macro'], at['precision_macro'])} |
| Macro Recall    | {_f(bt['recall_macro'])} | {_f(at['recall_macro'])} | {_delta(bt['recall_macro'], at['recall_macro'])} |

### Validation set (best checkpoint)

| Metric | Baseline | Augmented | Δ |
|---|---|---|---|
| **Macro F1** | **{_f(bv['f1_macro'])}** | **{_f(av['f1_macro'])}** | **{_delta(bv['f1_macro'], av['f1_macro'])}** |
| Weighted F1 | {_f(bv['f1_weighted'])} | {_f(av['f1_weighted'])} | {_delta(bv['f1_weighted'], av['f1_weighted'])} |
| Accuracy    | {_pct(bv['accuracy'])} | {_pct(av['accuracy'])} | {_delta(bv['accuracy'], av['accuracy'])} |
| Best epoch  | {base_metrics['best_epoch']} | {aug_metrics['best_epoch']} | — |

---

## 4. Per-Class Breakdown

### Baseline — per-class test metrics

{_per_class_table(base_metrics, class_names, 'Baseline')}

### Augmented — per-class test metrics

{_per_class_table(aug_metrics, class_names, 'Augmented')}

### Delta table (Augmented − Baseline)

{_delta_per_class_table(base_metrics, aug_metrics, class_names)}

---

## 5. Confusion Matrices

### Baseline

{_cm_table(base_cm)}

*Rows = true label, Columns = predicted label.*

### Augmented

{_cm_table(aug_cm)}

*Rows = true label, Columns = predicted label.*

---

## 6. Analysis

### What changed

The augmentation targeted the two minority classes.
`sustain` was the primary target (test support: {base_metrics['test']['per_class'].get('2', {}).get('support', '?')} examples),
`change` was secondary.

Key observations:

- **Macro F1 {'improved' if at['f1_macro'] > bt['f1_macro'] else 'did not improve'}**:
  baseline {_f(bt['f1_macro'])} → augmented {_f(at['f1_macro'])} ({_delta(bt['f1_macro'], at['f1_macro'])}).
- **Sustain F1**: baseline {_f((base_metrics['test']['per_class'].get('2', {}).get('f1', 0)))}
  → augmented {_f((aug_metrics['test']['per_class'].get('2', {}).get('f1', 0)))}.
- **Change F1**: baseline {_f((base_metrics['test']['per_class'].get('0', {}).get('f1', 0)))}
  → augmented {_f((aug_metrics['test']['per_class'].get('0', {}).get('f1', 0)))}.
- **Neutral F1**: baseline {_f((base_metrics['test']['per_class'].get('1', {}).get('f1', 0)))}
  → augmented {_f((aug_metrics['test']['per_class'].get('1', {}).get('f1', 0)))}.

### Confusion matrix interpretation

The key confusion pattern in MI talk-type classification is **sustain ↔ neutral**,
since both represent non-change speech and are semantically adjacent.
The confusion matrices above reveal how augmentation shifted the model's decision
boundary for the minority classes.

### Accuracy vs macro F1

Accuracy is biased toward `neutral` (majority class at ~65% of data).
The primary metric is **macro F1**, which weights all three classes equally.
A model that improves macro F1 while accuracy stays flat is improving exactly
where it matters — on the hard, under-represented classes.

---

## 7. Limitations

1. **QA filter is automated, not human-verified.** The o4-mini consistency check
   catches label drift but not semantic naturalness. A subset of accepted examples
   may still sound slightly unnatural for real MI sessions.

2. **Synthetic data topic distribution may not match test distribution.** The seeds
   span all AnnoMI topics but the synthetic paraphrases may overfit to the surface
   forms of those topics.

3. **Class-weighted loss is applied to augmented training too.** Since we increased
   sustain/change counts, the class weights shift, which interacts with augmentation
   in a non-trivial way.

4. **Single run.** Results reflect one training seed (42). Variance across seeds
   may be larger than the observed delta for small improvements.

---

## 8. Recommendations

1. **If macro F1 improved ≥ +0.02**: augmentation is confirmed useful. Next step is
   to sweep sustain augmentation size (500 / 700 / 900) to find the curve peak.

2. **If macro F1 improved < +0.02 or regressed**: augmentation noise is dominating.
   Tighten the QA filter (raise the consistency threshold) or reduce synthetic count.

3. **In both cases**: plot per-class F1 vs augmentation volume — this tells you
   whether sustain improved at the cost of change or neutral.

4. **Next experiment**: context-window sweep (context_turns = 1, 2, 4) combined with
   augmentation. Context often helps more than raw data volume for MI classification.

---

*Report generated by `generate_augmented_report.py`.*
*Baseline output dir: `{base_dir}`*
*Augmented output dir: `{aug_dir}`*
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {out_path}")
    print(f"Baseline macro F1:  {_f(bt['f1_macro'])}")
    print(f"Augmented macro F1: {_f(at['f1_macro'])}")
    print(f"Delta: {_delta(bt['f1_macro'], at['f1_macro'])}")


if __name__ == "__main__":
    main()
