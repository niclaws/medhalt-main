#!/usr/bin/env bash
# Med-HALT: safe-layer hallucination mitigation + causal transfer, 4 models x 3 seeds on one GPU.
# Per run: safe-layer selection, cross-task safe-layer ranking, risk-gated safe-layer steering,
# hallucination before/after tables, then component-level circuits and transfer.
# Each run resumes from its last finished stage (state/ in its output dir),
# so re-launching this script after a crash or preemption continues where it stopped.

set -uo pipefail   # no `set -e`: one failed model must not stop the others

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_V8_API_ENABLED=1

PROJECT_DIR="${PROJECT_DIR:-/scratch/adityasrikar.c/projects/medhalt-main}"   # $HOME expands; "~" inside quotes does not
SCRIPT="$PROJECT_DIR/run_casual_transfer.py"
SCRATCH="${SCRATCH_DIR:-/scratch/adityasrikar.c/medhalt}"
RESULTS="$SCRATCH/results_v4"   # v4 checkpoints are not compatible with earlier versions
LOGS="$SCRATCH/logs_v4"
HF_REPO="${HF_REPO:-ENETER_REPO_ID/medhalt}"
SEEDS=(${SEEDS:-0 1 2})

mkdir -p "$RESULTS" "$LOGS"
cd "$PROJECT_DIR" || { echo "PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }
[ -f "$SCRIPT" ] || { echo "Missing $SCRIPT"; exit 1; }

# ---- preflight: catch stale / mismatched copies before burning GPU hours
for f in patching_engine.py experiment_runner.py safe_layer_steering.py run_casual_transfer.py; do
  [ -f "$PROJECT_DIR/$f" ] || { echo "PREFLIGHT FAIL: missing $PROJECT_DIR/$f"; exit 1; }
done
HELP="$(python "$SCRIPT" --help 2>&1)" || { echo "PREFLIGHT FAIL: $SCRIPT --help errored:"; echo "$HELP"; exit 1; }
for flag in --layer_alphas --safety_lambda --harm_eps --layer_criterion --unfiltered_frac --router_features --gen_splits; do
  grep -q -- "$flag" <<< "$HELP" || { echo "PREFLIGHT FAIL: $SCRIPT does not accept $flag (stale copy of run_casual_transfer.py?)"; exit 1; }
done
grep -q "import safe_layer_steering" "$SCRIPT" || { echo "PREFLIGHT FAIL: $SCRIPT does not use safe_layer_steering.py"; exit 1; }
python -c "import sys; sys.path.insert(0, '$PROJECT_DIR'); import safe_layer_steering, experiment_runner as ER; \
assert hasattr(ER, 'router_features') and hasattr(ER, 'steering_outcomes'), 'stale experiment_runner.py'" \
  || { echo "PREFLIGHT FAIL: module versions do not match"; exit 1; }
echo "[preflight] OK: all four Python files present and consistent"

# tag | HF model id
MODELS=(
#  "med42_8b|m42-health/Llama3-Med42-8B"
#  "mistral_7b|mistralai/Mistral-7B-Instruct-v0.3"
  "qwen3_8b|Qwen/Qwen3-8B"
#  "mediphi_3.8b|microsoft/MediPhi-Instruct"
)

COMMON_ARGS=(
  --n_samples 1200
  --max_pairs_per_task 1000
  --top_k 20
  --k_values 5 10 15 20 30 50
  --lambdas 0 0.5 1 2 4 8
  --alphas 0.5 1 2 4
  --num_random_controls 20
  --min_denom 1.0
  --min_margin 0.5
  --batch_size 128
  --attn_implementation flash_attention_2
  --bootstrap_samples 2000
  --permutations 10000
  --gen_max_samples 40
  --layer_alphas 0.5 1 2 4 8
  --safety_lambda 2.0
  --harm_eps 0.10
  --layer_criterion worst
  --unfiltered_frac 0.2
  --router_features vocab
  --gen_splits unfiltered test
  --audit_per_cell 5
  --dtype bfloat16
  --resume
)

# SMOKE=1 bash script.sh  -> one model, one seed, small sample: checks the full
# pipeline on the GPU (and that steering changes predictions) in well under an hour.
if [ "${SMOKE:-0}" = "1" ]; then
  MODELS=("${MODELS[0]}")
  SEEDS=(0)
  RESULTS="$SCRATCH/results_v4_smoke"; LOGS="$SCRATCH/logs_v4_smoke"; mkdir -p "$RESULTS" "$LOGS"
  COMMON_ARGS+=(--n_samples 120 --max_pairs_per_task 80 --k_values 10 20 --num_random_controls 5
                --bootstrap_samples 500 --permutations 2000 --gen_max_samples 10)
  echo "[smoke] ${MODELS[0]} seed 0, reduced sample sizes -> $RESULTS"
fi

declare -A STATUS
for entry in "${MODELS[@]}"; do
  tag="${entry%%|*}"
  model="${entry#*|}"
  for seed in "${SEEDS[@]}"; do
    run="${tag}_seed${seed}"
    out="$RESULTS/$run"
    mkdir -p "$out"
    echo "=========================================="
    echo "Starting $model (seed $seed) -> $out"
    echo "=========================================="
    python "$SCRIPT" \
      --model_name "$model" \
      --seed "$seed" \
      --output_dir "$out" \
      --hf_repo_id "$HF_REPO" \
      --hf_path_in_repo "v4/${model//\//_}/seed${seed}" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "$LOGS/${run}.log"
    STATUS[$run]=${PIPESTATUS[0]}
    if [ "${STATUS[$run]}" -eq 0 ]; then
      echo "[SUCCESS] $run"
    else
      echo "[WARNING] $run failed with exit code ${STATUS[$run]} (log: $LOGS/${run}.log). Continuing."
    fi
  done

  # aggregate this model across whichever seeds finished
  dirs=()
  for seed in "${SEEDS[@]}"; do
    [ -f "$RESULTS/${tag}_seed${seed}/summary.csv" ] && dirs+=("$RESULTS/${tag}_seed${seed}")
  done
  if [ "${#dirs[@]}" -gt 0 ]; then
    python "$SCRIPT" --aggregate_dirs "${dirs[@]}" --output_dir "$RESULTS/${tag}_aggregate" \
      2>&1 | tee -a "$LOGS/${tag}_aggregate.log"
  fi
done

echo "=========================================="
echo "RUN SUMMARY"
echo "=========================================="
for run in "${!STATUS[@]}"; do
  printf "%-24s exit %s\n" "$run" "${STATUS[$run]}"
done | sort
echo "=========================================="
