#!/bin/bash
set -euo pipefail

MODELS=(
    med42_8b
    mediphi_3.8b
    mistral_7b
    qwen3_8b
)

SEEDS=(0 1 2)

SPLITS=(
    test
    unfiltered
)

RESULTS_DIR="results/results_v4"
OUTPUT_DIR="results/softmax_medhalt"

mkdir -p "$OUTPUT_DIR/logs"

TOTAL_RUNS=24
RUN=0

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do

            RUN=$((RUN + 1))

            echo "=========================================="
            echo "Run: $RUN / $TOTAL_RUNS"
            echo "Model: $MODEL"
            echo "Seed: $SEED"
            echo "Split: $SPLIT"
            echo "Started: $(date)"
            echo "=========================================="

            python eval_medhalt_softmax.py \
                --results_dir "$RESULTS_DIR" \
                --models "$MODEL" \
                --seeds "$SEED" \
                --protocol loto \
                --selection_targets FCT NOTA \
                --split "$SPLIT" \
                --temperature 1.5 \
                --center 0.5 \
                --output_dir "$OUTPUT_DIR/$SPLIT/${MODEL}_seed${SEED}" \
                > "$OUTPUT_DIR/logs/${MODEL}_seed${SEED}_${SPLIT}.log" 2>&1

            echo "Completed: $MODEL | Seed: $SEED | Split: $SPLIT"
            echo "Finished: $(date)"
            echo ""

        done
    done
done

echo "=========================================="
echo "All 24 evaluations completed successfully!"
echo "=========================================="