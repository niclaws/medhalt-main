#!/usr/bin/env bash
set -euo pipefail
# Run from repository root, in the environment already used by rebuttal.
OUT_ROOT="${MEDGEMMA_OUT_ROOT:-results/rebuttal_medgemma_full}"
DATA_DIR="${DATA_DIR:-medhalt/datasets}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg
mkdir -p "$OUT_ROOT"
# Stage arguments are optional: for example bash run_medgemma_full.sh --experiments strength treatment probe
python -u -m rebuttal.medgemma_rebuttal \
  --models medgemma_27b --seeds 0 \
  --quantization nf4 --dtype bfloat16 --attn-implementation sdpa \
  --large-mode layer --large-all-layers \
  --large-n-samples "${MEDGEMMA_N_SAMPLES:-1000}" \
  --large-max-pairs "${MEDGEMMA_MAX_PAIRS:-800}" \
  --large-discovery-limit "${MEDGEMMA_DISCOVERY_LIMIT:-100}" \
  --batch-size "${MEDGEMMA_BATCH_SIZE:-16}" --splits test \
  --data-dir "$DATA_DIR" --output-dir "$OUT_ROOT" --resume --fail-fast \
  "$@" 2>&1 | tee -a "$OUT_ROOT/console.log"
