#!/usr/bin/env bash

PROJECT_DIR="/root/medhalt-main/medhalt-main"
SCRIPT="$PROJECT_DIR/run_casual_transfer.py"

HF_REPO="damonsalvatore123/medhalt"

mkdir -p /results/logs

echo "=========================================="
echo "Starting Mistral experiment"
echo "Time: $(date)"
echo "=========================================="

python "$SCRIPT" \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --n_samples 100 \
    --top_k 20 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 5000 \
    --num_random_controls 20 \
    --seed 42 \
    --output_dir /results/mistral_7b_seed42 \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /results/logs/mistral_7b_seed42.log

MISTRAL_STATUS=${PIPESTATUS[0]}

if [ "$MISTRAL_STATUS" -eq 0 ]; then
    echo "Mistral completed successfully."
else
    echo "Mistral FAILED with exit code $MISTRAL_STATUS."
    echo "Continuing to Qwen anyway."
fi

echo
echo "=========================================="
echo "Starting Qwen experiment"
echo "Time: $(date)"
echo "=========================================="

python "$SCRIPT" \
    --model_name Qwen/Qwen3-8B \
    --is_qwen \
    --n_samples 100 \
    --top_k 20 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 5000 \
    --num_random_controls 20 \
    --seed 42 \
    --output_dir /results/qwen3_8b_seed42 \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /results/logs/qwen3_8b_seed42.log

QWEN_STATUS=${PIPESTATUS[0]}

if [ "$QWEN_STATUS" -eq 0 ]; then
    echo "Qwen completed successfully."
else
    echo "Qwen FAILED with exit code $QWEN_STATUS."
fi

echo
echo "=========================================="
echo "Experiments finished"
echo "Time: $(date)"
echo "=========================================="
echo "Mistral exit code: $MISTRAL_STATUS"
echo "Qwen exit code:    $QWEN_STATUS"
echo
echo "Logs:"
echo "  /results/logs/mistral_7b_seed42.log"
echo "  /results/logs/qwen3_8b_seed42.log"