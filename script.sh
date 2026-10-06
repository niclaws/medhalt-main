#!/usr/bin/env bash

PROJECT_DIR="~/projects/medhalt-main/medhalt-main"
SCRIPT="$PROJECT_DIR/run_casual_transfer.py"

HF_REPO="damonsalvatore123/medhalt"

mkdir -p /scratch/adityasrikar.c/medhalt/logs

cleanup_gpu() {
    echo "Clearing GPU memory..."
    pkill -9 -f "run_casual_transfer.py" 2>/dev/null || true
    sleep 3
}

# ==========================================
# 1. Llama3-Med42-8B (Clinical Aligned)
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
#    --output_dir /scratch/adityasrikar.c/medhalt/results/med42_8b_seed42 \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /scratch/adityasrikar.c/medhalt/logs/med42_8b_seed42.log
#
#STATUS_MED42=${PIPESTATUS[0]}
#if [ "$STATUS_MED42" -eq 0 ]; then
#    echo "[SUCCESS] Llama3-Med42-8B completed."
#else
#    echo "[WARNING] Llama3-Med42-8B failed with exit code $STATUS_MED42."
#fi

# ==========================================
# 2. MedGemma-4B-IT (4B Medical Specialist)
# ==========================================
#cleanup_gpu
#echo "=========================================="
#echo "Starting MedGemma-4B-IT"
#echo "=========================================="
#
#python "$SCRIPT" \
#    --model_name google/medgemma-4b-it \
#    --n_samples 100 \
#    --top_k 20 \
#    --head_batch_size 16 \
#    --k_values 5 10 15 20 30 50 \
#    --denom_thresholds 0.0001 0.001 0.01 0.1 \
#    --bootstrap_samples 2000 \
#    --num_random_controls 20 \
#    --control_samples 50 \
#    --seed 42 \
#    --output_dir /scratch/adityasrikar.c/medhalt/results/medgemma_4b_seed42 \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /scratch/adityasrikar.c/medhalt/logs/medgemma_4b_seed42.log
#
#STATUS_MEDGEMMA=${PIPESTATUS[0]}
#if [ "$STATUS_MEDGEMMA" -eq 0 ]; then
#    echo "[SUCCESS] MedGemma-4B-IT completed."
#else
#    echo "[WARNING] MedGemma-4B-IT failed with exit code $STATUS_MEDGEMMA."
#fi

# ==========================================
# 3. Mistral-7B-Instruct (Baseline General)
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
#    --output_dir /scratch/adityasrikar.c/medhalt/results/mistral_7b_seed42_new \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /scratch/adityasrikar.c/medhalt/results/mistral_7b_seed42_new/logs/mistral_7b_seed42_new.log
#
#STATUS_MISTRAL=${PIPESTATUS[0]}
#if [ "$STATUS_MISTRAL" -eq 0 ]; then
#    echo "[SUCCESS] Mistral-7B completed."
#else
#    echo "[WARNING] Mistral-7B failed with exit code $STATUS_MISTRAL. Continuing to next model..."
#fi
#
## ==========================================
## 4. Qwen3-8B (General Non-Thinking)
## ==========================================
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
#    --output_dir /scratch/adityasrikar.c/medhalt/results/qwen3_8b_seed42_new \
#    --hf_repo_id "$HF_REPO" \
#    2>&1 | tee /scratch/adityasrikar.c/medhalt/results/qwen3_8b_seed42_new/logs/qwen3_8b_seed42_new.log
#
#STATUS_QWEN=${PIPESTATUS[0]}
#if [ "$STATUS_QWEN" -eq 0 ]; then
#    echo "[SUCCESS] Qwen3-8B completed."
#else
#    echo "[WARNING] Qwen3-8B failed with exit code $STATUS_QWEN. Continuing to next model..."
#fi

# ==========================================
# 5. MediPhi-Instruct (3.8B Medical Specialist)
# ==========================================
cleanup_gpu
echo "=========================================="
echo "Starting MediPhi-Instruct-3.8B"
echo "=========================================="

python "$SCRIPT" \
    --model_name microsoft/MediPhi-Instruct \
    --n_samples 100 \
    --top_k 20 \
#    In ADA, take head_batch_size as 8
    --head_batch_size 8 \
#    Otherwise in bigger compute, take head_batch_size as 16
#    --head_batch_size 16 \
    --k_values 5 10 15 20 30 50 \
    --denom_thresholds 0.0001 0.001 0.01 0.1 \
    --bootstrap_samples 2000 \
    --num_random_controls 20 \
    --control_samples 50 \
    --seed 42 \
    --output_dir /scratch/adityasrikar.c/medhalt/results/mediphi_3.8b_seed42 \
    --hf_repo_id "$HF_REPO" \
    2>&1 | tee /scratch/adityasrikar.c/medhalt/logs/mediphi_3.8b_seed42.log

STATUS_MEDIPHI=${PIPESTATUS[0]}
if [ "$STATUS_MEDIPHI" -eq 0 ]; then
    echo "[SUCCESS] MediPhi-Instruct completed."
else
    echo "[WARNING] MediPhi-Instruct failed with exit code $STATUS_MEDIPHI."
fi

cleanup_gpu

echo "=========================================="
echo "RUN SUMMARY"
echo "=========================================="
#echo "Med42-8B:       Exit code $STATUS_MED42"
#echo "MedGemma-4B-IT:       Exit code $STATUS_MEDGEMMA"
#echo "Mistral-7B:     Exit code $STATUS_MISTRAL"
#echo "Qwen3-8B:       Exit code $STATUS_QWEN"
echo "MediPhi-Instruct:       Exit code $STATUS_MEDIPHI"
echo "=========================================="