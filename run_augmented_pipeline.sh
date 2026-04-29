#!/usr/bin/env bash
# run_augmented_pipeline.sh
#
# Runs every step after synthetic_candidates.jsonl is ready:
#   1. QA pass on synthetic candidates
#   2. Merge accepted rows into train_augmented.jsonl
#   3. Re-run baseline embeddings + training (refreshes per-class metrics)
#   4. Run augmented embeddings + training
#   5. Generate Augmented_Report.md
#
# Usage:
#   bash run_augmented_pipeline.sh

set -euo pipefail
PYTHON=/Users/juanortega/miniconda3/envs/annomi-mlp/bin/python
PROJECT=/Users/juanortega/Documents/CMU/Spring\ 2026/IDL/project/Code

cd "$PROJECT"

echo ""
echo "============================================================"
echo " STEP 1 — QA: verify synthetic candidates with o4-mini"
echo "============================================================"
$PYTHON -m annomi_pipeline.scripts.qa_synthetic_candidates \
    --input  data/outputs/augmentation/synthetic_candidates.jsonl \
    --output data/outputs/augmentation/synthetic_candidates_qa.jsonl \
    --model  o4-mini \
    --inter-call-sleep 0.5 \
    --log-level INFO

echo ""
echo "============================================================"
echo " STEP 2 — MERGE: build train_augmented.jsonl"
echo "============================================================"
$PYTHON -m annomi_pipeline.scripts.merge_augmented_train \
    --train     data/processed/train.jsonl \
    --synthetic data/outputs/augmentation/synthetic_candidates_qa.jsonl \
    --output    data/processed/train_augmented.jsonl

echo ""
echo "============================================================"
echo " STEP 3 — BASELINE: re-run embeddings + training"
echo "           (refreshes metrics with per-class F1 data)"
echo "============================================================"
$PYTHON -m annomi_pipeline.scripts.run_embeddings \
    --config configs/embeddings_config.yaml

$PYTHON -m annomi_pipeline.scripts.run_stage2 \
    --config configs/train_config.yaml

echo ""
echo "============================================================"
echo " STEP 4 — AUGMENTED: run embeddings + training"
echo "============================================================"
$PYTHON -m annomi_pipeline.scripts.run_embeddings \
    --config configs/embeddings_config_augmented.yaml

$PYTHON -m annomi_pipeline.scripts.run_stage2 \
    --config configs/train_config_augmented.yaml

echo ""
echo "============================================================"
echo " STEP 5 — REPORT: generate Augmented_Report.md"
echo "============================================================"
$PYTHON -m annomi_pipeline.scripts.generate_augmented_report \
    --baseline-dir  data/outputs \
    --augmented-dir data/outputs_augmented \
    --output        Augmented_Report.md \
    --baseline-train  data/processed/train.jsonl \
    --augmented-train data/processed/train_augmented.jsonl \
    --synthetic-qa    data/outputs/augmentation/synthetic_candidates_qa.jsonl

echo ""
echo "============================================================"
echo " DONE — Augmented_Report.md written to project root"
echo "============================================================"
