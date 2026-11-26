# Med-HALT NAACL rebuttal add-on

**Additive installation.** Copy only this `rebuttal/` directory into the original `medhalt-main/` repository. No original Python, shell, data, or result file needs to be edited. The archive does not bundle model weights, the original datasets, `.git` metadata, or prior GPU results. You must point it at your original locally generated run checkpoints. Only load trusted `.pt` files; they use PyTorch pickle serialization.

## Quick start (from repository root)

```bash
unzip -o medhalt_rebuttal_addon_nf4.zip -d /path/to/medhalt-main
cd /path/to/medhalt-main
python -m rebuttal.verify_originals
python -m pip install -r rebuttal/requirements-extra.txt
python -m unittest discover -s rebuttal/tests -v
python -m rebuttal.run --help
```

Use your already working CUDA/PyTorch/Transformers environment for the original four models. The additional requirements are sklearn, matplotlib and packaging; do not upgrade Transformers unnecessarily in that environment. For the large-model path a separate environment is preferable: install cluster-compatible PyTorch and the original `requirements.txt`, followed by `rebuttal/requirements-large.txt`. The MedGemma text model requires Transformers >=4.50; the adapter is written for the Transformers 4.x API, and the supplied large requirements select >=4.56,<5. Actual real-model/GPU compatibility still requires the smoke test.

Set paths (the archive's original `script.sh` uses a scratch `results_v4` folder; use the *actual* paper run directories):

```bash
export RESULTS_DIR=/path/to/results_v4
export OUT_ROOT=/path/to/new/rebuttal_results
export DATA_DIR="$PWD/medhalt/datasets"
# Existing run directory convention:
# $RESULTS_DIR/med42_8b_seed0/
# $RESULTS_DIR/mediphi_3.8b_seed0/
# $RESULTS_DIR/mistral_7b_seed0/
# $RESULTS_DIR/qwen3_8b_seed0/
```

Required original artifacts per model–seed:

```
run_config.json
splits.json
state/pairs_FCT.pt
state/base_FCT.pt
state/pairs_NOTA.pt
state/base_NOTA.pt
state/vectors_FCT.pt         # format only
state/vectors_NOTA.pt        # format only
state/selection_FCT.pt       # format only
state/selection_NOTA.pt      # format only
safe_layer/safe_layer_selection.csv  # strength, attenuation, treatment, probe
# state/pairs_*.pt and state/base_*.pt for additional guard tasks if the
# original LOTO selection_tasks includes them (needed to reconstruct original router)
```

If you only have `summary.csv`/plots, you cannot reconstruct the exact original steering directions or circuits. Recover the `state/` files from the original scratch storage first. The runner does not invent replacements for missing checkpoints.

`--resume` skips completed conditions after checking a per-model configuration/provenance manifest. An incomplete condition is restarted and its previous partial JSONL is retained. Large-model base passes are separately checkpointed every 10 pairs, and component discovery effects every pair. Use a new output folder when changing any settings or code, and never run two jobs writing to the same model/seed/condition directory. The stage script uses `tee` to capture original stdout (including tqdm, original model logs, and OOM messages) in addition to structured Python logging.

## Priority 0 — CPU audit of Table 1 (before any new GPU work)

```bash
bash rebuttal/run_rebuttal.sh audit
```

Outputs under `$OUT_ROOT/audit/<model>_seed<seed>/audit/`:
- `funnel.csv`: requested, built, raw pool, rejected, holdout, actual pool, clean-correct, margin-qualified, contrast-qualified, retained, split sizes, holdout/pool accuracies, invalid-selected count, overlap counts.
- `pair_audit.csv`: row-by-row recalculated conditions from saved candidate logits. It reports both the contrast-specific margin actually used by the original `validate()` and the gold-vs-best-wrong margin (these are not always identical).
- `funnel_vs_original.csv` if the original `selection_funnel.csv` exists; `original_settings.json`.
- `options.csv` and `distribution_*.csv`: actual gold/predicted/contrasted option letter counts by split and clean/corrupt.

The uploaded *launcher* defaults to 300 requested / 200 retained cap, while the manuscript describes 1000 / 800. The actual per-run `run_config.json` is authoritative; the code in the archive alone does not explain the numerical discrepancy. This audit cannot reconstruct preselection/replenishment that was never recorded in checkpoints. Investigate discrepancies before claiming the issue is only a typo. If invalid selected pairs or source/test overlap are detected, rerun affected original experiments rather than only correcting Table 1.

## 1. Matched-format and paraphrased-hint controls

Quick pilot, two models/seed 0, test split only; use a separate output root for a pilot:

```bash
OUT_ROOT="$OUT_ROOT/pilot" MODELS="med42_8b mistral_7b" SEEDS="0" \
  bash rebuttal/run_rebuttal.sh format --splits test --limit 30
```

Full original 4 checkpoints × 3 seeds (test and unfiltered):

```bash
bash rebuttal/run_rebuttal.sh format
```

Runs all four source→target combinations. Use `--sources FCT --targets FCT` for a smaller source-only FCT paraphrase/matched mechanism job. The variants:

- `original`: unchanged saved question/prompt.
- `paraphrase_1`, `paraphrase_2`: same gold answer, incorrect suggested option and question, two fixed paraphrases of the original authority hint.
- `fct_position`: permute options consistently, cycle gold positions and remap the misleading hint label. Position-dependent questions/options are rejected and logged.
- `nota_identity_reworded`, `nota_position`: replace NOTA wording with “None of the other listed options” in both comparison arms, then permute positions in one arm. This avoids changing what a non-final “above” would refer to. Original NOTA is also evaluated. Correct-position balance may not be exact after rejections; use actual distributions.
- `matched_hint`, `matched_distractor`: same FCT question, substantive gold option, identical option order/correct position and original plausible distractors in the hint arm. In the distractor arm the clean prompt has unrelated distractors and the corrupt prompt restores the original distractors. Filler texts come from a different subject and exclude all original saved FCT/NOTA question IDs and normalized question strings. **These are heuristic distractors, not clinician-verified invalid answers.** Inspect the logged filler choices and manually audit a sample; if ambiguity is discovered, report the audit and rerun with a declared exclusion rule.

The code runs original harm-aware patching and original harm-aware steering, and a stricter steering arm using **the exact oracle-patch circuit**. The original `calibrate_and_select()` can choose different circuits for patching versus steering. The stricter arm fixes the patch-selected components, learns no new directions, and calibrates alpha using the original objective on source validation only. The exact selected circuit IDs and all alpha calibration scores are saved. Do not confuse original-operator comparison with the stricter identical-component comparison.

For every source/target test or unfiltered comparison, exclude source discovery/validation question IDs and exact normalized question text matches (including within-task duplicates). This catches exact duplicates but not paraphrased semantic duplicates. Save exclusions. Never fit a new direction or select an alpha using test labels. Use `matched_cohort_summary.csv` / root `matched_format_comparisons.csv` to compare the original and modified conditions on identical successfully constructed IDs and, separately, the common valid contrast subset. Normalized gain is calculated only on valid contrasts; accuracy/induction are also reported on all constructed examples. Oracle patch applies only to corrupted prompts, so its accuracy must be compared to corrupted accuracy, not combined clean/corrupt accuracy.

## 2. Attenuation-matched gating (Table 7)

```bash
# Same original evaluation settings and selection policy as Table 7.
# Default: 4 models, 3 data seeds, 200 MedQA + 200 PubMedQA, dataset seed 42.
# If original softmax prediction files are available, verify their exact sample IDs:
bash rebuttal/run_rebuttal.sh attenuation \
  --reference-soft-dir /path/to/original/softmax_downstream
# Or without old CSVs:
# bash rebuttal/run_rebuttal.sh attenuation
```

Optional capped gate controls (more forward passes):

```bash
OUT_ROOT="$OUT_ROOT/caps" bash rebuttal/run_rebuttal.sh attenuation --gate-caps 2 1 .5 .25
```

Reuses the original LOTO `safe_layer_selection.csv` and `fit_from_discovery()` reconstruction, including guard-task features. Temperature 1.5, center .5, 5 deterministic permutation reps. First pass computes the original input-specific gate weights for valid tokenizations. Second pass applies unchanged vector/layer with `baseline`, `always_on`, `soft`, `constant_mean`, and `shuffled_soft` arms. The mean is computed **separately in every condition**; shuffled weights preserve the complete distribution. No evaluation labels are used to compute these weights. This is a retrospective matched-strength diagnostic, not an independently calibrated deployment policy. `--batch-size 1` reproduces the original one-intervention-per-forward batch shape more closely if numerical drift matters. The original external evaluator may have used a different attention backend than the training run; pass its backend explicitly if known.

The four models × 3 seeds × 2 heldout-probe LOTO policies × 2 datasets give 48 possible conditions. Only originally active policies are evaluated. The manuscript reports 26 active / 48 possible. Check root `attenuation_cohort.csv`, `completed_conditions.csv`, and failure logs. The runner warns if active count differs. **Do not report 26/48 unless the saved selections and completed evaluation actually confirm it.** `--reference-soft-dir` checks accepted sample IDs and gold labels exactly against original per-question CSVs, and logs probability drift. Without it the original sample loader and seed are reused, but current HF dataset revisions/tokenization may differ. The new run freezes its prompts/IDs in `external_samples.json`; use `--external-samples-json` to reuse that manifest in another run. Exact historical cohort identity cannot be guaranteed without historical IDs or a frozen dataset revision.

Key outputs: `sample_weights.json` (includes every permutation and effective strength), `predictions.csv`, `paired_soft_comparisons.csv`, `rejections.csv`, `reference_check.json` if requested, `table7_extension.csv` (condition-level average after averaging shuffle reps), `paired_soft_macro.csv`. `prediction_change_rate`, `induced`, `rescued`, `induction_rate`, `rescue_rate`, Wilson upper bound, safety score and baseline drift are logged. Note original paper reports BF16 numerical drift, so tiny differences need caution.

## 3. Low strength / norm-constrained steering

```bash
bash rebuttal/run_rebuttal.sh strength \
  --alphas 0 .125 .25 .5 1 2 4 8 \
  --rhos .05 .1 .25 .5 --random-reps 3
```

Reconstructs source-task mean-difference vectors from cached discovery activations; uses the source-only selected layer and original selected alpha. For both source and other task it evaluates the ungated absolute strength grid. Three independent random controls are normalized to the learned vector's per-layer norm and evaluated with the same strengths. In a separate arm, `alpha_eff=min(alpha_selected, rho*||h||/||v||)` at the selected layer; the pre-rounding nominal relative edit is logged. At a single residual hook, `h` is the unedited activation. BF16 rounding means the exact realized norm after rounding need not equal the nominal ratio bit-for-bit. Alpha zero returns the actual cached unedited candidate scores. No target tuning. Abstaining original source policies are explicitly skipped rather than silently substituted. A negative result at alpha=8 alone does not establish a low-strength failure. Read the full curve.

`strength_curve.csv`, `strength_per_condition.csv` and `strength_<source>_<target>_<split>.png` report rescue versus induction with alpha annotations (aggregated descriptively). `predictions.csv` includes layer, direction, repeat, alpha, rho, effective alpha, activation norm, direction norm and nominal relative edit. The combined clean/corrupt prompt analysis clusters uncertainty by underlying question. Report random replicates separately or average within each model/seed/condition; do not count the replicates as independent models.

## 4. Treatment-effect-aware gate

```bash
bash rebuttal/run_rebuttal.sh treatment
# With external QA evaluation as well:
OUT_ROOT="$OUT_ROOT/treatment_external" bash rebuttal/run_rebuttal.sh treatment --treatment-external
# Optional stronger disjoint training design (extra inference; do this if time allows):
OUT_ROOT="$OUT_ROOT/treatment_fresh" bash rebuttal/run_rebuttal.sh treatment \
  --treatment-extra-pairs 200 --treatment-external
```

The original selected source layer, direction and alpha stay fixed. It generates paired no-edit and actual edit outcomes. A regularized multinomial classifier uses the original router's candidate-free vocabulary statistics and residual PCA feature map, predicting `{no_change, rescue, induce}`. Utility is `P(rescue)-2*P(induce)`; a threshold is chosen on source validation using the original harm-aversion objective and Wilson screening rule. No target/external labels are used for training or threshold calibration. Six arms: baseline, always-on, reconstructed original base-error gate, learned treatment gate, base-error oracle, benefit oracle. Oracles use ground truth and are diagnostic only. The benefit oracle can avoid unnecessary edits without improving the fixed-policy rescue count beyond the base-error oracle.

**Default fit caveat:** discovery is reused for direction estimation and outcome-router fitting; original validation is reused for policy selection and threshold calibration. Held-out test remains independent. Do not claim independent source training/calibration or a safety guarantee. `--treatment-extra-pairs 200` instead fits the router on fresh source data excluding both original tasks' IDs and normalized questions, saving the complete fresh cohort. This is stronger if you have enough raw pairs and GPU time. The existing validation still selects the threshold, so this is not independent post-selection safety certification. The original base-error router is reconstructed on its original discovery split even when benefit training is fresh. Compare these methods with that training distinction disclosed. A `--train-limit` is explicitly a reduced router training pilot; do not compare it as an equal-data method without disclosing the limit.

Files: `training.json`, `routers.pt`, `<fit_split>_outcomes.csv`, `validation_outcomes.csv`, `threshold_grid.csv`, `treatment_diagnostics.csv` (rescue/induction AUROC, coverage, actual event counts) and all per-input predicted probabilities, treatment utilities, actual steering correctness and decisions. Benefit gates with no safe positive validation utility abstain. The Wilson bound is a descriptive screening bound, not an across-distribution or post-selection guarantee.

## 5. ITI-inspired linear-probe direction

```bash
bash rebuttal/run_rebuttal.sh probe --probe-c 1
```

Uses original discovery labels of actual answer correctness, not simply clean versus corrupted templates. Fit L2-regularized logistic regression at the same selected source residual layer, with standardized activations. Convert the probe coefficients back into the unstandardized activation coordinates (`w_original=w_scaled/scale`), then norm-match to the mean-difference direction. The same layer is used for all arms. Compare mean difference, norm-matched random and probe with (a) the original fixed alpha; and (b) each estimator's alpha chosen independently on the same source validation grid, allowing alpha=0/abstention. No test data used for alpha selection.

This is **ITI-inspired residual linear-probe steering**, not a faithful replication of Li et al.'s head-wise ITI intervention. Do not label it “full ITI” in the rebuttal. Files: `probe.pt`, `probe_training.json` (class counts, cosine, norms and fit accuracy), `validation_metrics.csv`, `validation_predictions.csv`, `chosen_alphas.json`, test and unfiltered predictions. If source discovery contains only one label class, fail visibly instead of fitting a meaningless probe.

## 6. Larger models: Med42-70B and MedGemma-27B text

Model aliases:
- `med42_70b`: `m42-health/Llama3-Med42-70B`
- `medgemma_27b`: `google/medgemma-27b-text-it`

Both may require approved Hugging Face access. MedGemma requires license/terms acceptance. Use `hf auth login` interactively if needed; never put tokens in logs or scripts.

### NF4 single-GPU loading (new default)

`--quantization auto` loads the two larger models in BitsAndBytes NF4 with nested/double quantization, BF16 computation and `device_map={"": 0}`. The original four models retain their existing precision and loader. Explicit `--quantization nf4` selects the same mode for a fresh scale run, including an optional 8B quantization-control run. Explicit `--quantization none` restores the original unquantized scale loader. Existing original BF16 checkpoint reuse with quantization is rejected.

Install `rebuttal/requirements-large.txt` in a separate CUDA-compatible environment. Select exactly one scheduler-allocated GPU using `CUDA_VISIBLE_DEVICES` if needed. NF4 rejects CPU/disk offloading, multiple visible devices, and `--max-memory` (the explicit single-device map is used instead). It validates actual 4-bit module loading and CUDA parameter placement; no silent BF16 fallback. This is weight quantization, not 4-bit activation arithmetic. Use `--dtype bfloat16` (default), or float16 only if required. NF4 cannot guarantee that 70B fits on 48 GB; use an 80 GB or larger GPU if the loader or batch size 1 OOMs. MedGemma-27B should have substantially more headroom on L40S. Models may still require substantial download/cache disk space.

The separate `ShardedEngine` captures per-layer states to CPU and performs full-sequence forwards with `use_cache=False`. It supports head/MLP patching and steering and residual steering. The original engine remains unchanged. Quantization metadata, model load time, CUDA memory and periodic cumulative intervention throughput are logged. OOM batch reduction remains active for intervention sweeps; model loading and batch size 1 cannot be rescued by batch reduction. No KV-cache optimization is included. Use fresh output paths when switching quantization or upgrading this code (configuration and code fingerprints prevent incompatible resume).

### GPU smoke tests (run each on one allocated GPU first)

```bash
python -m pip install -r rebuttal/requirements-large.txt
export OUT_ROOT=/path/to/rebuttal_nf4_results
CUDA_VISIBLE_DEVICES=0 python -m rebuttal.gpu_smoke --model medgemma_27b \
  --quantization nf4 --output "$OUT_ROOT/medgemma_smoke.json"
CUDA_VISIBLE_DEVICES=0 python -m rebuttal.gpu_smoke --model med42_70b \
  --quantization nf4 --output "$OUT_ROOT/med42_70b_smoke.json"
```

Check `loaded_in_4bit`, `linear4bit_modules`, device map, finite scores, no-op drift and nonzero intervention effects. The real models have not been GPU-tested in the delivery environment. GPU 0 here is the logical GPU in your allocation; do not override the scheduler's GPU selection with a physical ID outside your allocation.

### Recommended deadline-scale run: residual layer discovery and test interventions

```bash
CUDA_VISIBLE_DEVICES=0 MODELS="medgemma_27b" SEEDS="0" BATCH_SIZE=2 \
  bash rebuttal/run_rebuttal.sh scale --quantization nf4 \
  --large-mode layer --large-all-layers --large-n-samples 1000 \
  --large-max-pairs 800 --large-discovery-limit 100 --splits test --fail-fast
CUDA_VISIBLE_DEVICES=0 MODELS="med42_70b" SEEDS="0" BATCH_SIZE=1 \
  bash rebuttal/run_rebuttal.sh scale --quantization nf4 \
  --large-mode layer --large-all-layers --large-n-samples 1000 \
  --large-max-pairs 800 --large-discovery-limit 100 --splits test --fail-fast
```

Start with a small separate pilot (`--large-n-samples 100 --large-max-pairs 80 --large-discovery-limit 2 --limit 2`, separate `OUT_ROOT`) to measure throughput before full runs. For all discovery pairs, use `--large-discovery-limit 0`. Original builder, filters and 75/10/15 split fractions are reused, with a 20% holdout. The layer pilot ranks source residual layers by oracle patch gain, learns a source mean-difference direction and calibrates alpha on source validation. The selected layer receives target-clean oracle replacement (up to arithmetic rounding) and source steering on test prompts. Both FCT/NOTA transfer directions and in-task conditions are included. Report this as a residual-layer scale extension, not a Table 3 k=20 head/MLP replication. Report quantization, actual sample counts, one seed, number of candidate layers and abstentions.

### More comparable but far more expensive: full component AIE mode

```bash
CUDA_VISIBLE_DEVICES=0 MODELS="med42_70b" SEEDS="0" BATCH_SIZE=1 \
  OUT_ROOT="$OUT_ROOT/component" bash rebuttal/run_rebuttal.sh scale \
  --quantization nf4 --large-mode component --large-n-samples 300 \
  --large-max-pairs 200 --large-discovery-limit 8 --large-random-controls 5 \
  --k 20 --splits test --limit 30 --fail-fast
```

This mode uses the original `compute_steering_vectors`, `discovery_sweep`, `component_scores`, and `calibrate_and_select` over ALL attention heads and MLPs. It runs per-component patch and steering discovery, source validation harm-aware selection, cross-task oracle patch and additive steering, same-circuit steering, and patch controls (top-k AIE, contiguous, non-top-k random and layer-matched). `--large-discovery-limit` caps discovery only; 0 means full discovery. `--large-random-controls 20` uses the original random-control count. For close replication of main Table 3 use 1000 requested / 800 cap, all discovery, full source validation and test; still report full-forward engine differences and actual retained counts. **Full 70B component discovery is exceptionally costly with full-sequence re-forwards.** Even 8 discovery pairs × thousands of components × three intervention regimes can exceed the deadline. Launch a tiny pilot, measure real throughput, and only run the full mode with enough GPU time. The default layer pilot is safer for a four-day rebuttal. Do not report uncompleted conditions as completed or combine layer-pilot and component-mode effects as the same metric.

Files: `new_checkpoints/run_config.json`, `state/`, `splits.json`, `data_audit/`, `model_runtime.json`, source policy JSON, discovery effects and selected directions, per-prompt outcomes and distributions. To compare within-family scale under identical reduced protocol run Med42-8B separately with `--engine sharded --quantization nf4 --large-mode layer` (and a separate BF16 control with `--quantization none`) and the same sample/seed/grid settings; conditional filtered cohorts may differ across checkpoints, so also compare unfiltered shared question IDs.

## Outputs and interpretation

Each stage writes:

```
<OUT_ROOT>/<stage>/
  console.log                         # via run_rebuttal.sh
  rebuttal.log                        # structured logs
  failures.json                       # explicit failures; nonzero exit on failure
  completed_conditions.csv            # completed conditions only
  all_condition_summaries.csv          # per model/seed/condition/method/run type
  by_model.csv                        # average over data seeds
  macro_models.csv                    # model-level descriptive summaries
  <model>_seed<seed>/
    config.json                       # original file fingerprints, code hashes, args
    environment.json                  # versions and GPU info
    model_runtime.json                # actual checkpoint/runtime configuration
    <condition>/
      metadata.json
      predictions.jsonl               # incrementally flushed raw rows
      predictions.csv                 # completed row table
      summary.csv                     # per method/arm metrics
      distribution_gold.csv
      distribution_pred_base.csv
      distribution_pred_after.csv
      DONE.json                       # written only after summary succeeds
```

JSONL records store actual candidate logits and predictions, question IDs, intervention settings and correctness. `summary.csv` uses rescue rate among initially wrong and induction rate among initially correct; accuracy change = `(rescues-induced)/N` in percentage points; safety score = `(rescues-2*induced)/N`. `n_questions` and 95% paired cluster bootstrap intervals for delta accuracy preserve clean/corrupt pairs. McNemar p-values are raw/descriptive; the combined clean/corrupt exact test is not independent, so use question-cluster uncertainty and do not claim adjusted significance. If claiming confirmatory tests in the paper, define comparison families beforehand and apply multiple-testing correction. A passing per-condition Wilson upper bound is not a post-selection or transfer guarantee. Condition averages are descriptive; models are fixed checkpoints, not independent patient populations. The code does not claim clinically verified hallucinations. Empty/abstained/unavailable cells are not imputed.

To regenerate aggregate CSVs/plots after moving completed output files, without running models:

```bash
python -m rebuttal.run --aggregate-only --output-dir "$OUT_ROOT/strength"
```

### Suggested work order given the deadline

1. Audit all existing run checkpoints and fix Table 1/repository issues. CPU work.
2. Launch uncapped attenuation-matched control on all original active external conditions; exact reviewer request.
3. Launch format control for two fixed models/seed first, then extend. Do not choose models after seeing results.
4. Strength and norm sweep. Use source-only selected policies; log abstentions. Expand to all saved seeds if feasible.
5. Treatment gate and probe; prefer a fresh source fit if time allows.
6. Large-model smoke test and reduced pilot in parallel on spare hardware only. Component mode optional and much costlier. Use the resulting evidence to narrow claims, even if the controls yield positive transfer or show a format explanation.

## Verification and limits of this delivery

- No original code files changed: `python -m rebuttal.verify_originals` checks hashes of 46 original code/config files against the uploaded archive.
- `python -m compileall -q rebuttal`, CLI `--help`/`--plan`, shell syntax, and seven CPU regression tests passed during preparation. Tests exercise actual original checkpoint format/builders with synthetic model states, original router and selectors, core calculations and the sharded adapter's hooks on a tiny CPU network. They do not establish performance or verify real Transformers/GPU behavior.
- The delivery environment had no CUDA, Transformers, or HF datasets installation; there were no original run checkpoints in the uploaded archive. No real model experiment was run, and no results are fabricated. Run GPU smoke tests and compare new baselines/selection counts against your actual paper artifacts before reporting metrics.
- `--reference-soft-dir` is optional but recommended for exact Table 7 cohort checking; the historical external data revision is not known. Keep raw data licenses and anonymity in mind before sharing logs or checkpoints.
