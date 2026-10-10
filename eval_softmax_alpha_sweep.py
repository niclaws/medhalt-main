#!/usr/bin/env python3
"""Evaluate multiple SOFTMAX STEERING ALPHA CAPS in one sequential model load.

Additive standalone experiment: does NOT alter original discovery, steering,
softmax files, or model weights. It reuses existing .pt checkpoints and
selection policies under results/results_v4/{model}_seed{seed}.

For each question, baseline/always-on are computed once; probability-weighted
steering is evaluated separately at each alpha cap. This saves time versus
restarting the whole model for every alpha value.

    alpha_effective = min(alpha_selected, alpha_cap) * P_softmax(steer)

P_softmax is the SAME original error-risk softmax gate, with unchanged
--temperature and --center for all caps. Always-on uses alpha_selected.
No gate labels, test answers, or downstream labels are used for tuning.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import eval_downstream_all_models as original
from experiment_runner import router_features
from patching_engine import CausalEngine, log
from softmax_gate_adapter import (
    ProbabilisticResidualSteerer,
    SoftmaxGateConfig,
    candidate_probabilities,
    gate_from_router,
)
from softmax_eval_helpers import checkpoint_split, summarize_predictions


def alpha_slug(alpha: float) -> str:
    """Stable, path-safe string: .25->0p25, 1.0->1p0."""
    text = f"{alpha:g}"
    return (text if "." in text else text + ".0").replace(".", "p")


def effective_alpha(selected_alpha: float, alpha_cap: float, p_on: float) -> float:
    """Scale the activation intervention, not the final output probabilities."""
    if not all(math.isfinite(v) for v in (selected_alpha, alpha_cap, p_on)):
        raise ValueError("Nonfinite alpha or probability")
    if selected_alpha < 0 or alpha_cap <= 0 or not 0 <= p_on <= 1:
        raise ValueError("Bad selected alpha, cap or probability")
    return float(min(selected_alpha, alpha_cap) * p_on)


def base_result_fields(
    base_logits: np.ndarray,
    always_logits: np.ndarray,
    gold: int,
    answer_labels: list[str],
) -> dict:
    """Shared per-sample candidate probabilities and baseline/always correctness."""
    b = int(np.argmax(base_logits))
    a = int(np.argmax(always_logits))
    return {
        "gold_index": gold,
        "gold_label": answer_labels[gold],
        "pred_base": answer_labels[b],
        "pred_always": answer_labels[a],
        "correct_base": b == gold,
        "correct_always": a == gold,
        "gold_prob_base": float(candidate_probabilities(base_logits)[gold]),
        "gold_prob_always": float(candidate_probabilities(always_logits)[gold]),
        "candidate_labels": json.dumps(answer_labels),
        "candidate_prob_base": json.dumps(
            candidate_probabilities(base_logits).tolist()
        ),
        "candidate_prob_always": json.dumps(
            candidate_probabilities(always_logits).tolist()
        ),
    }


def evaluate_medhalt_caps(
    engine,
    examples,
    selection: dict,
    vector,
    router,
    gate_config: SoftmaxGateConfig,
    caps: list[float],
    batch_size: int,
    run_types: list[str],
) -> pd.DataFrame:
    active = bool(selection["selected"])
    selected_alpha = float(selection["alpha"]) if active else 0.0
    steerer = (
        ProbabilisticResidualSteerer(
            engine, vector, int(selection["layer"]), selected_alpha, batch_size
        )
        if active
        else None
    )

    rows = []
    for pair, cached in tqdm(examples, desc="Med-HALT alpha sweep", leave=False):
        for run_type in run_types:
            state = engine.prepare(pair[f"{run_type}_ids"], pair["cand_ids"])
            base_logits = np.asarray(state.cand_logits, dtype=np.float64)
            always_logits = steerer.run_always(state) if active else base_logits
            gold = 0 if run_type == "clean" else int(cached["gold_corrupt"])
            labels = list(pair.get("cand_labels") or map(str, range(len(base_logits))))
            if not 0 <= gold < len(labels):
                raise ValueError("Invalid gold index in cached Med-HALT pair")
            fixed = base_result_fields(base_logits, always_logits, gold, labels)

            if active:
                feat = router_features(cached, run_type)
                prob = gate_from_router(router, feat, 1.0, gate_config)
                risk, p_on, p_off = prob.risk, prob.p_on, prob.p_off
            else:
                risk, p_on, p_off = float("nan"), 0.0, 1.0

            for cap in caps:
                alpha_eff = effective_alpha(selected_alpha, cap, p_on)
                soft_logits = steerer.run(state, alpha_eff) if active else base_logits
                soft_idx = int(np.argmax(soft_logits))
                soft_probs = candidate_probabilities(soft_logits)
                rows.append(
                    {
                        "sample_id": str(pair["sample_id"]),
                        "task": str(pair["task"]),
                        "pair_type": str(pair["pair_type"]),
                        "run_type": run_type,
                        "alpha_cap": cap,
                        "alpha_selected": selected_alpha,
                        "alpha_capped_before_gate": min(selected_alpha, cap),
                        "alpha_effective": alpha_eff,
                        "gate_on_prob": p_on,
                        "gate_off_prob": p_off,
                        "risk_score": risk,
                        "gate_temperature": gate_config.temperature,
                        "gate_center": gate_config.center,
                        **fixed,
                        "pred_soft": labels[soft_idx],
                        "correct_soft": soft_idx == gold,
                        "gold_prob_soft": float(soft_probs[gold]),
                        "candidate_prob_soft": json.dumps(soft_probs.tolist()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_downstream_caps(
    engine,
    samples: list[dict],
    selection: dict,
    vector,
    router,
    gate_config: SoftmaxGateConfig,
    caps: list[float],
    batch_size: int,
):
    active = bool(selection["selected"])
    selected_alpha = float(selection["alpha"]) if active else 0.0
    steerer = (
        ProbabilisticResidualSteerer(
            engine, vector, int(selection["layer"]), selected_alpha, batch_size
        )
        if active
        else None
    )
    rows, rejected = [], []

    for i, sample in enumerate(
        tqdm(samples, desc="Downstream alpha sweep", leave=False)
    ):
        try:
            state, prompt_ids, shared_prefix = original.prepare_choices(engine, sample)
        except ValueError as exc:
            rejected.append(
                {
                    "sample_id": i,
                    "dataset_index": sample["dataset_index"],
                    "reason": str(exc),
                }
            )
            continue

        base_logits = np.asarray(state.cand_logits, dtype=np.float64)
        always_logits = steerer.run_always(state) if active else base_logits
        answers = list(sample["answers"])
        gold = int(answers.index(sample["gold"]))
        fixed = base_result_fields(base_logits, always_logits, gold, answers)
        if active:
            if shared_prefix:
                stats, resid = engine.prompt_features(prompt_ids)
            else:
                stats, resid = state.vocab_stats, state.resid
            feat = {
                "vocab_stats": np.asarray(stats),
                "resid": resid.float().cpu(),
                "cand_logits": base_logits,
            }
            prob = gate_from_router(router, feat, 1.0, gate_config)
            risk, p_on, p_off = prob.risk, prob.p_on, prob.p_off
        else:
            risk, p_on, p_off = float("nan"), 0.0, 1.0

        for cap in caps:
            alpha_eff = effective_alpha(selected_alpha, cap, p_on)
            soft_logits = steerer.run(state, alpha_eff) if active else base_logits
            soft_idx = int(np.argmax(soft_logits))
            soft_probs = candidate_probabilities(soft_logits)
            rows.append(
                {
                    "sample_id": i,
                    "dataset_index": sample["dataset_index"],
                    "gold": sample["gold"],
                    "alpha_cap": cap,
                    "alpha_selected": selected_alpha,
                    "alpha_capped_before_gate": min(selected_alpha, cap),
                    "alpha_effective": alpha_eff,
                    "gate_on_prob": p_on,
                    "gate_off_prob": p_off,
                    "risk_score": risk,
                    "gate_temperature": gate_config.temperature,
                    "gate_center": gate_config.center,
                    **fixed,
                    "pred_soft": answers[soft_idx],
                    "correct_soft": soft_idx == gold,
                    "gold_prob_soft": float(soft_probs[gold]),
                    "candidate_prob_soft": json.dumps(soft_probs.tolist()),
                    "max_candidate_prob_soft": float(soft_probs.max()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(
        rejected, columns=["sample_id", "dataset_index", "reason"]
    )


def write_one_condition(
    frame: pd.DataFrame,
    out_root: Path,
    kind: str,
    split: str,
    model: str,
    seed: int,
    target: str,
    selection: dict,
    run_type: str | None,
    args: argparse.Namespace,
):
    """One CSV per alpha cap, plus a row for the aggregate summary."""
    all_summaries = []
    for cap, cap_df in frame.groupby("alpha_cap", sort=True):
        dirname = (
            out_root / kind / f"alpha_{alpha_slug(float(cap))}" / f"{model}_seed{seed}"
        )
        if kind == "medhalt":
            dirname = dirname / split
        dirname.mkdir(parents=True, exist_ok=True)
        if kind == "medhalt":
            label = f"{model}_seed{seed}_{args.protocol}_{target}_{split}"
        else:
            label = f"{model}_seed{seed}_{args.protocol}_{target}_{split}"
        suffix = f"_{run_type}" if run_type else ""
        cap_df.to_csv(dirname / f"{label}{suffix}_softmax_predictions.csv", index=False)
        if run_type is None and kind == "medhalt":
            for rt, part in cap_df.groupby("run_type"):
                all_summaries += write_one_condition(
                    part,
                    out_root,
                    kind,
                    split,
                    model,
                    seed,
                    target,
                    selection,
                    rt,
                    args,
                )
            continue
        metrics = summarize_predictions(
            cap_df,
            seed=args.sample_seed + seed,
            reps=args.bootstrap_samples,
            harm_eps=args.harm_eps,
        )
        metrics.update(
            {
                "model_tag": model,
                "seed": seed,
                "protocol": args.protocol,
                "selection_target": target,
                "evaluated_task": target if kind == "medhalt" else split,
                "dataset": split if kind == "downstream" else "medhalt",
                "split": split,
                "run_type": run_type if run_type is not None else "",
                "selected": bool(selection["selected"]),
                "layer": int(selection["layer"]) if selection["selected"] else -1,
                "alpha_selected": (
                    float(selection["alpha"]) if selection["selected"] else 0.0
                ),
                "alpha_cap": float(cap),
                "temperature": args.temperature,
                "center": args.center,
                "safety_harm_eps": args.harm_eps,
            }
        )
        all_summaries.append(metrics)
    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results_dir", default="results/results_v4/")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(original.MODELS),
        default=["med42_8b", "mediphi_3.8b", "mistral_7b", "qwen3_8b"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--scope", choices=["both", "medhalt", "downstream"], default="both"
    )
    parser.add_argument(
        "--medhalt_splits",
        nargs="+",
        choices=["validation", "test", "unfiltered"],
        default=["test", "unfiltered"],
    )
    parser.add_argument(
        "--selection_targets",
        nargs="+",
        choices=["FCT", "NOTA"],
        default=["FCT", "NOTA"],
    )
    parser.add_argument(
        "--protocol",
        choices=["loto", "pooled", "source:FCT", "source:NOTA"],
        default="loto",
    )
    parser.add_argument(
        "--alpha_caps", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--center", type=float, default=0.5)
    parser.add_argument("--n_medqa", type=int, default=200)
    parser.add_argument("--n_pubmedqa", type=int, default=200)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--harm_eps", type=float, default=0.10)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="results/softmax_alpha_sweep")
    args = parser.parse_args()

    caps = sorted(set(args.alpha_caps))
    if not caps or not all(math.isfinite(a) and a > 0 for a in caps):
        parser.error("All --alpha_caps must be finite positive numbers")
    if len({alpha_slug(a) for a in caps}) != len(caps):
        parser.error("Two alpha caps have colliding output directory names")
    if any(
        n <= 0
        for n in [
            args.n_medqa,
            args.n_pubmedqa,
            args.bootstrap_samples,
            args.batch_size,
        ]
    ):
        parser.error("Sample counts, bootstrap_samples and batch_size must be positive")
    if not (0 < args.harm_eps < 1):
        parser.error("--harm_eps must be in (0,1)")
    gate = SoftmaxGateConfig(temperature=args.temperature, center=args.center)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha_sweep_config.json").write_text(
        json.dumps({**vars(args), "alpha_caps": caps}, indent=2)
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refusing to run BF16/FP16 intervention sweeps on CPU."
        )
    log(
        f"[alpha sweep] device={torch.cuda.get_device_name(0)} caps={caps} scope={args.scope}"
    )

    samples_by_ds = {}
    if args.scope in ("both", "downstream"):
        for ds, n in [("medqa", args.n_medqa), ("pubmedqa", args.n_pubmedqa)]:
            samples_by_ds[ds] = original.load_samples(ds, n, args.sample_seed)

    combined_summaries = []
    for model in args.models:
        for seed in args.seeds:
            run_dir = Path(args.results_dir) / f"{model}_seed{seed}"
            if not run_dir.exists():
                log(f"[SKIP] No results directory: {run_dir}")
                continue
            selected = {}
            for task in args.selection_targets:
                try:
                    selected[task] = original.get_selection(
                        run_dir, args.protocol, task
                    )
                except (FileNotFoundError, ValueError, KeyError) as ex:
                    log(f"[SKIP] Cannot load {model} seed={seed} task={task}: {ex}")
            if not selected:
                continue
            model_config_file = root / f"{model}_seed{seed}_alpha_sweep_config.json"
            model_config_file.write_text(
                json.dumps(
                    {
                        **vars(args),
                        "alpha_caps": caps,
                        "model_for_this_run": model,
                        "seed_for_this_run": seed,
                    },
                    indent=2,
                )
            )
            log(f"[MODEL START] {model} seed={seed} (loading once for ALL caps)")
            engine = CausalEngine(
                original.MODELS[model],
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )
            try:
                for target, selection in selected.items():
                    vector, router = original.fit_from_discovery(run_dir, selection)
                    log(
                        f'[TARGET] {model} seed={seed} {target} original_alpha={selection["alpha"]} '
                        f'selected={selection["selected"]}'
                    )
                    if args.scope in ("both", "medhalt"):
                        for split in args.medhalt_splits:
                            pairs = checkpoint_split(run_dir, target, split)
                            if not pairs:
                                log(
                                    f"[SKIP] No Med-HALT {split} examples for {model}/{seed}/{target}"
                                )
                                continue
                            predictions = evaluate_medhalt_caps(
                                engine,
                                pairs,
                                selection,
                                vector,
                                router,
                                gate,
                                caps,
                                args.batch_size,
                                ["clean", "corrupt"],
                            )
                            combined_summaries += write_one_condition(
                                predictions,
                                root,
                                "medhalt",
                                split,
                                model,
                                seed,
                                target,
                                selection,
                                None,
                                args,
                            )
                    if args.scope in ("both", "downstream"):
                        for dataset, samples in samples_by_ds.items():
                            predictions, rejected = evaluate_downstream_caps(
                                engine,
                                samples,
                                selection,
                                vector,
                                router,
                                gate,
                                caps,
                                args.batch_size,
                            )
                            if predictions.empty:
                                log(
                                    f"[SKIP] All downstream questions rejected: {dataset}"
                                )
                                continue
                            combined_summaries += write_one_condition(
                                predictions,
                                root,
                                "downstream",
                                dataset,
                                model,
                                seed,
                                target,
                                selection,
                                None,
                                args,
                            )
                            for cap in caps:
                                dest = (
                                    root
                                    / "downstream"
                                    / f"alpha_{alpha_slug(cap)}"
                                    / f"{model}_seed{seed}"
                                )
                                rejected.to_csv(
                                    dest
                                    / f"{model}_seed{seed}_{args.protocol}_{target}_{dataset}_rejected.csv",
                                    index=False,
                                )
                # Save each model–seed's metrics immediately in case another run fails later
                own = [
                    r
                    for r in combined_summaries
                    if r["model_tag"] == model and r["seed"] == seed
                ]
                out_path = root / f"{model}_seed{seed}_summary.csv"
                pd.DataFrame(own).to_csv(out_path, index=False)
                log(
                    f"[MODEL DONE] {model} seed={seed}: {len(own)} summary rows -> {out_path}"
                )
            finally:
                del engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Each Bash iteration launches a NEW Python process. Build the
            # cumulative aggregate from completed per-model/seed summaries,
            # rather than overwriting it with just this process's rows.
            completed_files = sorted(root.glob("*_seed*_summary.csv"))
            combined = pd.concat(
                (pd.read_csv(p) for p in completed_files),
                ignore_index=True,
            )
            keys = [
                "model_tag",
                "seed",
                "selection_target",
                "dataset",
                "split",
                "run_type",
                "alpha_cap",
            ]
            combined = combined.drop_duplicates(keys, keep="last")
            combined.to_csv(root / "alpha_sweep_allruns_summary.csv", index=False)

    if not combined_summaries:
        raise RuntimeError(
            "No evaluations completed. Verify selections, state checkpoints, and datasets."
        )
    log(f"[DONE] {len(combined_summaries)} condition summary rows under {root}")


if __name__ == "__main__":
    main()
