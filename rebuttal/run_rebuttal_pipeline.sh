
# nohup bash run_all_rebuttal.sh > rebuttal_all.log 2>&1 &
#!/usr/bin/env bash
set -euo pipefail

export RESULTS_DIR="results/results_v4"
export OUT_ROOT="results/rebuttal_nf4"
export DATA_DIR="medhalt/datasets"
export SEEDS="0"
export PYTHONUNBUFFERED=1

# ======================================
# SMALLER MODELS — BF16
# ======================================

export MODELS="med42_8b mediphi_3.8b mistral_7b qwen3_8b"
export BATCH_SIZE=16

for stage in audit format attenuation strength treatment probe; do
    echo "========================================"
    echo "Starting experiment: $stage"
    echo "Time: $(date)"
    echo "========================================"

    bash rebuttal/run_rebuttal_individual.sh "$stage" \
        --splits test \
        --quantization none \
        --fail-fast

    echo "Completed $stage at $(date)"
done

# ======================================
# LARGER MODELS — NF4 4-bit
# ======================================

#export MODELS="medgemma_27b med42_70b"
#
#bash rebuttal/run_rebuttal_individual.sh scale \
#    --quantization nf4 \
#    --large-mode layer \
#    --large-all-layers \
#    --large-n-samples 1000 \
#    --large-max-pairs 800 \
#    --large-discovery-limit 100 \
#    --splits test \
#    --fail-fast

echo "All experiments completed at $(date)"
