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
# 1. Llama3-Med42-8B (Clinical Aligned)
# ==========================================
cleanup_gpu
echo "=========================================="
echo "Starting Llama3-Med42-8B"
echo "Time: $(date)"
echo "=========================================="

python "$SCRIPT" \
    --model_name m42-health/Llama3-Med42-8B \
    --n_samples 100 \
    --top_k 20 \
    --head_batch_size 16 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 2000 \
    --num_random_controls 20 \
    --control_samples 50 \
    --seed 42 \
    --output_dir /results/med42_8b_seed42 \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /results/logs/med42_8b_seed42.log

STATUS_MED42=${PIPESTATUS[0]}
if [ "$STATUS_MED42" -eq 0 ]; then
    echo "[SUCCESS] Llama3-Med42-8B completed."
else
    echo "[WARNING] Llama3-Med42-8B failed with exit code $STATUS_MED42."
fi

# ==========================================
# 2. BioMistral-7B (Medical Domain Pre-trained)
# ==========================================
cleanup_gpu
echo "=========================================="
echo "Starting BioMistral-7B"
echo "Time: $(date)"
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

STATUS_BIOMISTRAL=${PIPESTATUS[0]}
if [ "$STATUS_BIOMISTRAL" -eq 0 ]; then
    echo "[SUCCESS] BioMistral-7B completed."
else
    echo "[WARNING] BioMistral-7B failed with exit code $STATUS_BIOMISTRAL. Continuing to next model..."
fi

# ==========================================
# 3. Mistral-7B-Instruct (Baseline General)
# ==========================================
cleanup_gpu
echo "=========================================="
echo "Starting Mistral-7B-Instruct"
echo "Time: $(date)"
echo "=========================================="

python "$SCRIPT" \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --n_samples 100 \
    --top_k 20 \
    --head_batch_size 16 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 2000 \
    --num_random_controls 20 \
    --control_samples 50 \
    --seed 42 \
    --output_dir /results/mistral_7b_seed42_new \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /results/mistral_7b_seed42_new/logs/mistral_7b_seed42_new.log

STATUS_MISTRAL=${PIPESTATUS[0]}
if [ "$STATUS_MISTRAL" -eq 0 ]; then
    echo "[SUCCESS] Mistral-7B completed."
else
    echo "[WARNING] Mistral-7B failed with exit code $STATUS_MISTRAL. Continuing to next model..."
fi

# ==========================================
# 4. Qwen3-8B (General Non-Thinking)
# ==========================================
cleanup_gpu
echo "=========================================="
echo "Starting Qwen3-8B"
echo "Time: $(date)"
echo "=========================================="

python "$SCRIPT" \
    --model_name Qwen/Qwen3-8B \
    --is_qwen \
    --n_samples 100 \
    --top_k 20 \
    --head_batch_size 16 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 2000 \
    --num_random_controls 20 \
    --control_samples 50 \
    --seed 42 \
    --output_dir /results/qwen3_8b_seed42_new \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /results/qwen3_8b_seed42_new/logs/qwen3_8b_seed42_new.log

STATUS_QWEN=${PIPESTATUS[0]}
if [ "$STATUS_QWEN" -eq 0 ]; then
    echo "[SUCCESS] Qwen3-8B completed."
else
    echo "[WARNING] Qwen3-8B failed with exit code $STATUS_QWEN. Continuing to next model..."
fi

cleanup_gpu

echo "=========================================="
echo "RUN SUMMARY"
echo "=========================================="
echo "Med42-8B:       Exit code $STATUS_MED42"
echo "BioMistral-7B:  Exit code $STATUS_BIOMISTRAL"
echo "Mistral-7B:     Exit code $STATUS_MISTRAL"
echo "Qwen3-8B:       Exit code $STATUS_QWEN"
echo "=========================================="