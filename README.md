# AnnoMI Stage 1 Pipeline

This repository is scoped to the Stage 1 system described in `Team_22.pdf`: a reproducible preprocessing pipeline for therapist-client conversations from the AnnoMI dataset, plus a simple TF-IDF + logistic regression baseline used to generate Figure 2 (`baseline_results.png`).

The default configuration targets `AnnoMI/AnnoMI-simple.csv`, which is already present in this workspace.

## What Stage 1 Produces

Running Stage 1 does the following:

1. Loads and validates the AnnoMI utterance-level CSV.
2. Reconstructs full therapist-client conversations by `transcript_id`.
3. Splits conversations into train/validation/test at the transcript level to avoid leakage.
4. Chunks each conversation with a sliding window.
5. Writes chunked JSONL datasets and tokenized JSONL datasets.
6. Computes token statistics for the generated chunks.
7. Runs a TF-IDF + logistic regression baseline sweep across chunk sizes and vocabulary sizes.
8. Saves preliminary baseline results as `data/outputs/baseline_results.json` and `data/outputs/baseline_results.png`.

## Dataset

AnnoMI is a public dataset of motivational interviewing conversations with utterance-level annotations. Each row contains:

- a `transcript_id` and `utterance_id` for reconstructing the dialogue,
- speaker identity (`interlocutor`),
- utterance text,
- transcript-level MI quality labels,
- therapist and client behavior annotations.

The repository keeps the original source file path configurable. Stage 1 can optionally copy that source CSV into `data/raw/` for a more self-contained artifact layout.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate annomi-mlp
```

If you prefer plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

## Repository Layout

```text
project_root/
  README.md
  requirements.txt
  environment.yml
  pyproject.toml
  configs/
  data/
  src/
  tests/
```

## Pipeline Usage

```bash
python -m annomi_pipeline.scripts.run_stage1 --config configs/data_config.yaml
```

## Outputs

After running Stage 1, the repository will contain:

- chunked datasets in `data/processed/`,
- tokenized datasets in `data/tokenized/`,
- reports in `data/outputs/`.

Key artifacts include:

- `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/tokenized/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/outputs/token_stats.json`
- `data/outputs/baseline_results.json`
- `data/outputs/baseline_results.png`
- `data/outputs/stage1_summary.json`

## Notes

- The default baseline target is `metadata.therapist_behavior_mode`, which aligns with the conversational behavior framing in the report.
- The baseline sweep varies dialogue chunk window size and TF-IDF vocabulary size to produce Figure 2.
- Non-stage-1 source files are ignored via `.gitignore` so they remain local and do not get pushed accidentally.
