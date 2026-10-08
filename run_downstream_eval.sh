#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/medhalt-main}"
SCRATCH="${SCRATCH_DIR:-/scratch/adityasrikar.c/medhalt}"
RESULTS_DIR="$SCRATCH/results_v4"
OUTPUT_DIR="$SCRATCH/downstream_audit"

cd "$PROJECT_DIR"

python eval_downstream_all_models.py \
  --results_dir "$RESULTS_DIR" \
  --seed 0 \
  --protocol "loto" \
  --n_medqa 200 \
  --n_pubmedqa 200 \
  --dtype "bfloat16" \
  --batch_size 64 \
  --output_dir "$OUTPUT_DIR"