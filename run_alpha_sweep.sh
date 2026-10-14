#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="results/results_v4/"
OUTPUT_DIR="results/softmax_alpha_sweep"
MODELS=(med42_8b mediphi_3.8b mistral_7b qwen3_8b)
SEEDS=(0 1 2)
ALPHA_CAPS=(0.25 0.5 1.0 2.0)

mkdir -p "$OUTPUT_DIR/logs"

# Refuse silent CPU fallback (CUDA 13 build + older driver mismatch, etc.).
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(f'CUDA unavailable (torch={torch.__version__}, torch CUDA={torch.version.cuda}).')
print('GPU:',torch.cuda.get_device_name(0))
PY

RUN=0
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN=$((RUN+1))
    echo "[$(date)] RUN $RUN/12: $MODEL seed=$SEED; alpha caps: ${ALPHA_CAPS[*]}"
    # The call is blocking: next run starts only after this one exits.
    python -u eval_softmax_alpha_sweep.py \
      --results_dir "$RESULTS_DIR" \
      --models "$MODEL" \
      --seeds "$SEED" \
      --scope both \
      --protocol loto \
      --selection_targets FCT NOTA \
      --medhalt_splits test unfiltered \
      --n_medqa 200 \
      --n_pubmedqa 200 \
      --alpha_caps "${ALPHA_CAPS[@]}" \
      --temperature 1.5 \
      --center 0.5 \
      --dtype bfloat16 \
      --output_dir "$OUTPUT_DIR" \
      > "$OUTPUT_DIR/logs/${MODEL}_seed${SEED}.log" 2>&1
    echo "[$(date)] DONE: $MODEL seed=$SEED"
  done
done

echo "All model–seed alpha sweeps completed."
echo "Summaries: $OUTPUT_DIR/*_summary.csv"
