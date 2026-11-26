
#!/usr/bin/env bash
# Run from the original repository root.
set -euo pipefail

STAGE="${1:-audit}"
if [[ $# -gt 0 ]]; then shift; fi

case "$STAGE" in
  audit|format|attenuation|strength|treatment|probe|scale) ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

RESULTS_DIR="${RESULTS_DIR:-results/results_v4}"
OUT_ROOT="${OUT_ROOT:-results/rebuttal}"
DATA_DIR="${DATA_DIR:-medhalt/datasets}"
SEEDS="${SEEDS:-0}"

if [[ "$STAGE" == "scale" ]]; then
  MODELS="${MODELS:-med42_70b medgemma_27b}"
  MED42_BATCH_SIZE="${MED42_BATCH_SIZE:-8}"
  MEDGEMMA_BATCH_SIZE="${MEDGEMMA_BATCH_SIZE:-16}"
else
  MODELS="${MODELS:-med42_8b mediphi_3.8b mistral_7b qwen3_8b}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
fi

read -r -a models <<< "$MODELS"
read -r -a seeds <<< "$SEEDS"

mkdir -p "$OUT_ROOT/$STAGE"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$STAGE" == "scale" ]]; then

  # Run larger models separately with model-specific batch sizes.
  for model in "${models[@]}"; do

    case "$model" in
      med42_70b)
        batch="$MED42_BATCH_SIZE"
        ;;
      medgemma_27b)
        batch="$MEDGEMMA_BATCH_SIZE"
        ;;
      *)
        echo "Unsupported scale model: $model" >&2
        exit 2
        ;;
    esac

    echo "[$(date -Is)] Running $model with batch size $batch" \
      | tee -a "$OUT_ROOT/$STAGE/console.log"

    python -u -m rebuttal.run --experiments "$STAGE" \
      --models "$model" --seeds "${seeds[@]}" \
      --results-dir "$RESULTS_DIR" \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT_ROOT/$STAGE" \
      --quantization nf4 \
      --batch-size "$batch" \
      --resume "$@" \
      2>&1 | tee -a "$OUT_ROOT/$STAGE/console.log"

  done

else

  # Run smaller models using the common batch size.
  python -u -m rebuttal.run --experiments "$STAGE" \
    --models "${models[@]}" --seeds "${seeds[@]}" \
    --results-dir "$RESULTS_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_ROOT/$STAGE" \
    --quantization none \
    --batch-size "$BATCH_SIZE" \
    --resume "$@" \
    2>&1 | tee -a "$OUT_ROOT/$STAGE/console.log"

fi
