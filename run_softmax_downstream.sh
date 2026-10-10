#!/bin/bash
set -euo pipefail

MODELS=(
    med42_8b
    mediphi_3.8b
    mistral_7b
    qwen3_8b
)

SEEDS=(0 1 2)

RESULTS_DIR="results/results_v4"
OUTPUT_DIR="results/softmax_downstream"

mkdir -p "$OUTPUT_DIR/logs"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        echo "=========================================="
        echo "Running Model: $MODEL | Seed: $SEED"
        echo "Started: $(date)"
        echo "=========================================="

        python eval_downstream_softmax.py \
            --results_dir "$RESULTS_DIR" \
            --models "$MODEL" \
            --seeds "$SEED" \
            --protocol loto \
            --selection_targets FCT NOTA \
            --n_medqa 200 \
            --n_pubmedqa 200 \
            --temperature 1.5 \
            --center 0.5 \
            --output_dir "$OUTPUT_DIR/${MODEL}_seed${SEED}" \
            > "$OUTPUT_DIR/logs/${MODEL}_seed${SEED}.log" 2>&1

        echo "Completed: $MODEL | Seed: $SEED"
        echo "Finished: $(date)"
        echo ""
    done
done

echo "All downstream evaluations completed successfully!"