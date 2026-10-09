
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/adityasrikar.c/projects/medhalt-main}"
SCRATCH="${SCRATCH_DIR:-/scratch/adityasrikar.c/medhalt}"

RESULTS_DIR="$SCRATCH/results_v4"
OUTPUT_DIR="$SCRATCH/downstream_audit_corrected"

cd "$PROJECT_DIR"

python eval_downstream_all_models.py \
    --results_dir "$RESULTS_DIR" \
    --models \
        mediphi_3.8b \
        mistral_7b \
        med42_8b \
        qwen3_8b \
    --seeds 0 1 2 \
    --protocol loto \
    --selection_targets FCT NOTA \
    --sample_seed 42 \
    --n_medqa 200 \
    --n_pubmedqa 200 \
    --bootstrap_samples 2000 \
    --harm_eps 0.10 \
    --dtype bfloat16 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR"
