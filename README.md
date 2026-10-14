# When Causal Circuits Fail to Transfer

**Activation patching, cross-task steering, and intervention safety in medical language models**

We study a central question: **Do components localized by causal activation patching support effective and safe, reusable inference-time steering on other tasks?** The contribution is an **intervention audit**, not a new general-purpose steering algorithm or a clinically validated hallucination detector.

We first discover influential attention heads and MLPs using clean–corrupt contrastive inputs. We then test whether selected component locations and steering directions transfer across tasks, and whether risk-aware selection and gating reduce unintended errors. We report both positive causal restoration and negative transfer rather than assuming a successful patch implies safe deployment.

## Research questions

1. Does causal localization outperform non-top-*k* random and structured component controls?
2. Do components identified on one task remain causally influential on another?
3. Why can **oracle cross-task patching** succeed while **reusable activation steering** fails?
4. Can harm-aware selection, validation risk bounds, abstention, hard gating, softmax weighting, or smaller steering strengths avoid induced errors?
5. Do intervention outcomes generalize to external medical question-answering benchmarks?

## Models and tasks

Experiments cover four decoder-only models with three independent data seeds (typically `0 1 2`):

| Run identifier | Hugging Face model |
|---|---|
| `med42_8b` | `m42-health/Llama3-Med42-8B` |
| `mediphi_3.8b` | `microsoft/MediPhi-Instruct` |
| `mistral_7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `qwen3_8b` | `Qwen/Qwen3-8B` |

The four probes are derived from **Med-HALT** ([source dataset](https://huggingface.co/datasets/openlifescienceai/Med-HALT)):

| Task | Type | Clean / corrupted contrast |
|---|---|---|
| **FCT** — False Confidence Test | Reasoning | Original medical question versus the same question with a misleading “expert” answer hint; the true answer does not change. |
| **NOTA** — None of the Above | Reasoning | Correct answer remains “none of the above”; the corruption replaces implausible distractors with plausible but incorrect options. |
| **PMID2Title** | Counterfactual bibliographic identity | Different PubMed identifiers are associated with different correct titles. |
| **Title2URL** | Counterfactual bibliographic identity | Different article titles are associated with different PubMed link/identifier targets. |

The bibliographic identity tasks are **not** clinical hallucination benchmarks; their clean and counterfactual inputs may have different correct answers. Keep these separate from the reasoning probes in analyses.

External evaluation uses **MedQA-USMLE-4-options** (`GBaker/MedQA-USMLE-4-options`, `test` split) and **PubMedQA** (`qiaojin/PubMedQA`, `pqa_labeled` **train** split in the supplied evaluator). Do not describe the latter as an official held-out PubMedQA test set.

## Method overview

### 1. Build and validate contrastive pairs

For a clean input `x+`, corrupted input `x-`, correct answer `a`, and contrasted incorrect answer `b`, use the next-token logit difference

```text
d(x) = z_a(x) - z_b(x)
D_i  = d(x_i+) - d(x_i-)
```

Pairs are filtered for answer-token compatibility, margins, and contrast validity. The pipeline records reasons for rejection in `pair_validation.csv` and saves sample counts in `selection_funnel.csv`. Valid pairs are split into **discovery / validation / test** (default proportions `0.50 / 0.25 / 0.25`); an additional unfiltered holdout supports separate auditing. The candidate-pair count is not necessarily the number of retained valid pairs.

### 2. Causal localization and controls

Oracle activation patching replaces the corrupted activation of component `c` with that pair's **own clean activation** at the final prompt position. The normalized patching effect is

```text
E_ic   = [d(x_i-; h_c <- h_c(x_i+)) - d(x_i-)] / D_i
AIE(c) = mean_i E_ic
```

AIE is the **average indirect effect under this specific intervention**, not a guarantee of safe steering. Components include attention heads and MLPs. The experiment evaluates:

- **AIE top-*k*** components ranked by `|AIE|`.
- **Harm-aware ranking:** `R_c - lambda * H_c`, with positive and negative restoration contributions estimated on discovery data and `lambda` selected on validation.
- **Non-top-*k* random controls:** draws outside the top-*k* set, **not** unrestricted random components.
- **Layer-matched random controls** and **contiguous-*k* component windows**.
- Circuit-size sensitivity, usually `k = 5, 10, 15, 20, 30, 50`.

### 3. Cross-task patching versus reusable steering

These interventions answer **different questions**:

- **Oracle cross-task patching:** select *component locations* on a source task, then patch those locations with each target pair's *target-clean* activations. This requires access to the clean target reference at evaluation time and is a diagnostic, **not deployable correction**.
- **Reusable activation steering:** derive a source-task clean-minus-corrupt activation direction and add it to target activations without a target-clean reference. This tests actual directional transfer.

We compare normalized contrast restoration, answer-selection accuracy, rescues, **newly induced errors on initially correct examples**, layer/component overlap, direction cosine similarity, and representational diagnostics (including CKA). Similarity is descriptive and does not establish causal interchangeability.

### 4. Risk-constrained selection and gating

For a selected layer `ell` and steering strength `alpha`, validation scores use

```text
S = (N_rescue - 2 * N_induce) / N
```

where `N_rescue` counts incorrect-to-correct changes, `N_induce` counts correct-to-incorrect changes, and `N` counts evaluated prompts. An intervention is admissible only when the **one-sided 95% Wilson upper confidence bound** on `N_induce / N_correct` is at most **0.10** on each selection task. Wilson intervals reflect uncertainty from finite validation counts, including cases with zero observed induced errors. The selected policy maximizes worst-task validation score among admissible candidates; the policy **abstains** when none qualify.

Selection protocols include **source-task**, **pooled-task**, and **leave-one-task-out (LOTO)** selection. For LOTO, the target task's direction and layer are selected using the *other* probe. Passing a validation constraint is **not** a transferable safety certificate; failing one does not prove that true risk exceeds 10%.

The original hard gate uses a detector of **baseline error risk** to decide whether to steer. A later softmax variant applies a **continuous probability weight** instead of an on/off decision:

```text
alpha_effective = P_softmax(steer | input) * min(alpha_selected, alpha_cap)
h' = h + alpha_effective * steering_direction
```

This probability estimates baseline-error propensity, **not** the probability that steering will help. Alpha-cap sweeps study `0.25, 0.5, 1.0, 2.0`; a small cap can reduce harm simply by causing fewer prediction changes.

### 5. External and generation evaluation

- **External accuracy:** baseline, always-on steering, hard risk-gated steering, and probability-weighted steering on MedQA and PubMedQA.
- **Error accounting:** rescue rate, induced-error rate, accuracy changes (percentage points), and Wilson upper bounds.
- **Generation audit:** automatically labeled output-level proxies and optional human review sheets. These are **not** clinician-adjudicated medical hallucination outcomes.
- **Uncertainty and controls:** bootstrap estimates, paired tests, random controls, and across-seed summaries where supported by the scripts.


**Versioning:** use mutually compatible versions of all core Python modules. The checkpoint-based softmax runners require the original discovery outputs in `results_v4`, not just compact CSV archives. V4 `state/` checkpoints are not interchangeable with older experiment versions.

## Requirements and GPU limitations

Python 3.10+ and a CUDA-enabled PyTorch environment are recommended. Install dependencies into an isolated environment; use PyTorch builds compatible with the GPU driver/CUDA setup.

```bash
pip install -r requirements.txt
```

`flash-attn` is **optional and environment-dependent**. The main runner defaults to `--attn_implementation flash_attention_2`; if FlashAttention is not installed/supported, try `--attn_implementation sdpa` instead. Authenticate to Hugging Face if a model or dataset requires access.

**Important GPU note:** The supplied original patching engine and softmax adapters use **Hugging Face Transformers with single-device model loading**. The batch launcher processes model/seed runs sequentially; it is **not** vLLM tensor parallelism. Four GPUs with 11 GB each do **not** automatically provide one 44-GB address space. An unquantized 7B–8B BF16 model generally exceeds 11 GB of single-GPU VRAM before activations/cache; run on a GPU with adequate memory or make and verify separate model-sharding changes. Do not present the scripts as `tensor_parallel_size=4` implementations.

## How to run

Run all commands from the project directory containing the `.py` modules. Use your own output paths, and run `python <script>.py --help` if the local archive differs from the examples below.

### A. Main experiments

```bash
CUDA_VISIBLE_DEVICES=0 python run_casual_transfer.py \
  --model_name m42-health/Llama3-Med42-8B \
  --seed 0 \
  --n_samples 300 --max_pairs_per_task 200 \
  --split_fracs 0.5 0.25 0.25 \
  --top_k 20 --k_values 5 10 15 20 30 50 \
  --lambdas 0 0.5 1 2 4 8 \
  --alphas 0.5 1 2 4 \
  --layer_alphas 0.5 1 2 4 8 \
  --safety_lambda 2.0 --harm_eps 0.10 \
  --layer_criterion worst \
  --num_random_controls 20 \
  --min_denom 1.0 --min_margin 0.5 \
  --router_features vocab \
  --attn_implementation sdpa \
  --output_dir results/results_v4/med42_8b_seed0 \
  --resume
```

Example batch launch after checking the wrapper:

```bash
export PROJECT_DIR="$PWD"
export SCRATCH_DIR="$PWD/results"
export SEEDS="0 1 2"
bash script.sh                     
```

### B. Aggregate completed seeds

```bash
python run_casual_transfer.py \
  --aggregate_dirs \
    results/results_v4/med42_8b_seed0 \
    results/results_v4/med42_8b_seed1 \
    results/results_v4/med42_8b_seed2 \
  --output_dir results/results_v4/med42_8b_aggregate
```

Aggregate only completed runs of the **same model and experiment version**.

### C. Original external MedQA / PubMedQA audit

```bash
python eval_downstream_all_models.py \
  --results_dir results/results_v4 \
  --seed 0 --protocol loto \
  --n_medqa 200 --n_pubmedqa 200 \
  --dtype bfloat16 --batch_size 64 \
  --output_dir results/downstream_audit
```

The supplied original evaluator loops through all four models; it requires corresponding completed model/seed artifacts. Available selection protocols are `loto`, `pooled`, `source:FCT`, and `source:NOTA`.

### D. Softmax gating on Med-HALT held-out inputs

```bash
python eval_medhalt_softmax.py \
  --results_dir results/results_v4 \
  --models med42_8b --seeds 0 \
  --protocol loto --selection_targets FCT NOTA \
  --split test --run_types clean corrupt \
  --temperature 1.5 --center 0.5 \
  --output_dir results/softmax_medhalt
```

This uses original `.pt` discovery checkpoints and performs **new forward passes** with a continuous steering strength. Previously cached hard-gate results are not substitutes.

### E. Softmax gating on external benchmarks

```bash
python eval_downstream_softmax.py \
  --results_dir results/results_v4 \
  --models med42_8b --seeds 0 \
  --protocol loto --selection_targets FCT NOTA \
  --datasets medqa pubmedqa \
  --n_medqa 200 --n_pubmedqa 200 \
  --temperature 1.5 --center_mode fixed --center 0.5 \
  --output_dir results/softmax_downstream
```

For full coverage, use `--models med42_8b mediphi_3.8b mistral_7b qwen3_8b --seeds 0 1 2` **only after** all matching checkpoints exist and adequate GPU resources are available.

### F. Steering-strength (alpha-cap) sweep

The sequential wrapper from the alpha-sweep add-on assumes `results/results_v4/` and enumerates four models × three seeds, with caps `0.25 0.5 1.0 2.0`:

```bash
bash run_alpha_sweep_sequential.sh
```

The original **always-on** comparison uses its selected (uncapped) alpha; it is **not** a matched-strength always-on control. Avoid selecting a preferred cap on the held-out test data and then presenting the same evaluation as unbiased confirmation. Use independent calibration where possible.

