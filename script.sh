#!/usr/bin/env bash

PROJECT_DIR="/root/medhalt-main/medhalt-main"
SCRIPT="$PROJECT_DIR/run_casual_transfer.py"

HF_REPO="damonsalvatore123/medhalt"

mkdir -p /results/logs

cleanup_gpu() {
    echo "Clearing GPU memory..."
    pkill -9 -f "run_casual_transfer.py" 2>/dev/null || true
    sleep 3
}

# ==========================================
# 1. Mistral-7B-Instruct (Baseline General)
# ==========================================
#cleanup_gpu
#echo "=========================================="
#echo "Starting Mistral-7B-Instruct"
#echo "=========================================="
#
#python "$SCRIPT" \
#    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#    --n_samples 100 \
#    --top_k 20 \
#    --head_batch_size 16 \
#    --k_values 5 10 15 20 30 50 \
#    --denom_thresholds 0.0001 0.001 0.01 0.1 \
#    --bootstrap_samples 2000 \
#    --num_random_controls 20 \
#    --control_samples 50 \
#    --seed 42 \
#    --output_dir /results/mistral_7b_seed42 \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /results/logs/mistral_7b_seed42.log

# ==========================================
# 2. Qwen3-8B (General Non-Thinking)
# ==========================================
#cleanup_gpu
#echo "=========================================="
#echo "Starting Qwen3-8B"
#echo "=========================================="
#
#python "$SCRIPT" \
#    --model_name Qwen/Qwen3-8B \
#    --is_qwen \
#    --n_samples 100 \
#    --top_k 20 \
#    --head_batch_size 16 \
#    --k_values 5 10 15 20 30 50 \
#    --denom_thresholds 0.0001 0.001 0.01 0.1 \
#    --bootstrap_samples 2000 \
#    --num_random_controls 20 \
#    --control_samples 50 \
#    --seed 42 \
#    --output_dir /results/qwen3_8b_seed42 \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /results/logs/qwen3_8b_seed42.log

# ==========================================
# 3. BioMistral-7B (Medical Domain Pre-trained)
# ==========================================
cleanup_gpu
echo "=========================================="
echo "Starting BioMistral-7B"
echo "=========================================="

python "$SCRIPT" \
    --model_name BioMistral/BioMistral-7B \
    --n_samples 100 \
    --top_k 20 \
    --head_batch_size 16 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 2000 \
    --num_random_controls 20 \
    --control_samples 50 \
    --seed 42 \
    --output_dir /results/biomistral_7b_seed42 \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /results/logs/biomistral_7b_seed42.log

# ==========================================
# 4. Llama3-Med42-8B (Clinical Aligned)
# ==========================================
#cleanup_gpu
#echo "=========================================="
#echo "Starting Llama3-Med42-8B"
#echo "=========================================="
#
#python "$SCRIPT" \
#    --model_name m42-health/Llama3-Med42-8B \
#    --n_samples 100 \
#    --top_k 20 \
#    --head_batch_size 16 \
#    --k_values 5 10 15 20 30 50 \
#    --denom_thresholds 0.0001 0.001 0.01 0.1 \
#    --bootstrap_samples 2000 \
#    --num_random_controls 20 \
#    --control_samples 50 \
#    --seed 42 \
#    --output_dir /results/med42_8b_seed42 \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /results/logs/med42_8b_seed42.log

echo "=========================================="
echo "All experiments completed successfully."
echo "=========================================="