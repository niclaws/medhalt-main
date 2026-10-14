from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import eval_downstream_all_models as original
from experiment_runner import router_features
from patching_engine import CausalEngine, log
from softmax_gate_adapter import (
    SoftmaxGateConfig,
    gate_from_router,
    candidate_probabilities,
    ProbabilisticResidualSteerer,
)
from softmax_eval_helpers import (
    checkpoint_split,
    summarize_predictions,
    temperature_from_validation,
)


def run_evaluation(
    engine, examples, cfg, vector, router, gate_cfg, batch_size, run_types
):
    steerer = (
        ProbabilisticResidualSteerer(
            engine, vector, int(cfg["layer"]), float(cfg["alpha"]), batch_size
        )
        if cfg["selected"]
        else None
    )
    rows = []
    for pair, b in tqdm(examples, desc="Med-HALT softmax", leave=False):
        for rt in run_types:
            st = engine.prepare(pair[f"{rt}_ids"], pair["cand_ids"])
            base_logits = np.asarray(st.cand_logits, dtype=float)
            pb = candidate_probabilities(base_logits)
            risk = float("nan")
            on = 0.0
            off = 1.0
            alpha_eff = 0.0
            soft_logits = always_logits = base_logits
            if steerer is not None:
                # Original router_features sees only the allowed stored prompt features;
                # for shared answer prefixes this uses cached candidate-free data.
                f = router_features(b, rt)
                gate = gate_from_router(router, f, steerer.alpha, gate_cfg)
                risk, on, off, alpha_eff = (
                    gate.risk,
                    gate.p_on,
                    gate.p_off,
                    gate.alpha_effective,
                )
                soft_logits = steerer.run(st, alpha_eff)
                always_logits = steerer.run_always(st)
            ps = candidate_probabilities(soft_logits)
            pa = candidate_probabilities(always_logits)
            gold = 0 if rt == "clean" else int(b["gold_corrupt"])
            if not 0 <= gold < len(base_logits):
                raise ValueError("Bad gold index in base checkpoint")
            preds = {
                "base": int(np.argmax(base_logits)),
                "always": int(np.argmax(always_logits)),
                "soft": int(np.argmax(soft_logits)),
            }
            labels = pair.get("cand_labels") or [
                str(i) for i in range(len(base_logits))
            ]
            rows.append(
                {
                    "sample_id": str(pair["sample_id"]),
                    "task": pair["task"],
                    "pair_type": pair["pair_type"],
                    "run_type": rt,
                    "gold_index": gold,
                    "gold_label": labels[gold],
                    "pred_base": labels[preds["base"]],
                    "pred_always": labels[preds["always"]],
                    "pred_soft": labels[preds["soft"]],
                    "correct_base": preds["base"] == gold,
                    "correct_always": preds["always"] == gold,
                    "correct_soft": preds["soft"] == gold,
                    "risk_score": risk,
                    "gate_off_prob": off,
                    "gate_on_prob": on,
                    "alpha_effective": alpha_eff,
                    "alpha_selected": float(cfg["alpha"]) if steerer else 0.0,
                    "gate_temperature": gate_cfg.temperature,
                    "gate_center": gate_cfg.center,
                    "gold_prob_base": float(pb[gold]),
                    "gold_prob_soft": float(ps[gold]),
                    "gold_prob_always": float(pa[gold]),
                    "candidate_labels": json.dumps(labels),
                    "candidate_prob_base": json.dumps(pb.tolist()),
                    "candidate_prob_soft": json.dumps(ps.tolist()),
                    "candidate_prob_always": json.dumps(pa.tolist()),
                }
            )
    if not rows:
        raise ValueError("No Med-HALT examples in selected split/task")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--results_dir", required=True)
    ap.add_argument(
        "--models", nargs="+", choices=sorted(original.MODELS), default=["med42_8b"]
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument(
        "--protocol",
        default="loto",
        choices=["loto", "pooled", "source:FCT", "source:NOTA"],
    )
    ap.add_argument(
        "--selection_targets",
        nargs="+",
        choices=["FCT", "NOTA", "PMID2Title", "Title2URL"],
        default=["FCT", "NOTA"],
    )
    ap.add_argument("--split", choices=["test", "unfiltered"], default="test")
    ap.add_argument(
        "--run_types",
        nargs="+",
        choices=["clean", "corrupt"],
        default=["clean", "corrupt"],
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--auto_temperature",
        action="store_true",
        help="Calibrate original risk score on Med-HALT validation split",
    )
    ap.add_argument("--center", type=float, default=0.5)
    ap.add_argument("--center_mode", choices=["fixed", "original_tau"], default="fixed")
    ap.add_argument("--harm_eps", type=float, default=0.10)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    ap.add_argument(
        "--attn_implementation",
        default=None,
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output_dir", default="./medhalt_softmax_results")
    args = ap.parse_args()
    SoftmaxGateConfig(temperature=args.temperature, center=args.center)
    if not 0 < args.harm_eps < 1:
        ap.error("--harm_eps must be between zero and one")
    if args.bootstrap_samples < 1:
        ap.error("--bootstrap_samples must be >=1")
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "softmax_config.json").write_text(json.dumps(vars(args), indent=2))
    all_results = []
    for model in args.models:
        for seed in args.seeds:
            rd = Path(args.results_dir) / f"{model}_seed{seed}"
            if not rd.exists():
                log(f"[SKIP] missing {rd}")
                continue
            configs = {}
            for target in args.selection_targets:
                try:
                    config = original.get_selection(rd, args.protocol, target)
                    examples = checkpoint_split(rd, target, args.split)
                    if examples:
                        configs[target] = (config, examples)
                    else:
                        log(f"[SKIP] no {args.split} examples for {target}")
                except (ValueError, KeyError, FileNotFoundError) as e:
                    log(f"[SKIP] {model}/{seed}/{target}: {e}")
            if not configs:
                continue
            engine = CausalEngine(
                original.MODELS[model],
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )
            try:
                for target, (cfg, examples) in configs.items():
                    vector, router = original.fit_from_discovery(rd, cfg)
                    if args.auto_temperature and cfg["selected"]:
                        features, wrong = [], []
                        for task in cfg["selection_tasks"]:
                            for _, b in checkpoint_split(rd, task, "validation"):
                                for rt in ("clean", "corrupt"):
                                    features.append(router_features(b, rt))
                                    wrong.append(not bool(b[f"correct_{rt}"]))
                        temperature = temperature_from_validation(
                            router, features, wrong
                        )
                    else:
                        temperature = args.temperature
                    if args.center_mode == "original_tau" and cfg["selected"]:
                        center = float(cfg["tau"])
                        if not (np.isfinite(center) and 0 < center < 1):
                            raise ValueError(
                                f"tau={center} cannot be used as softmax center; use --center_mode fixed"
                            )
                    else:
                        center = args.center
                    gate_cfg = SoftmaxGateConfig(temperature=temperature, center=center)
                    preds = run_evaluation(
                        engine,
                        examples,
                        cfg,
                        vector,
                        router,
                        gate_cfg,
                        args.batch_size,
                        args.run_types,
                    )
                    name = f'{model}_seed{seed}_{args.protocol.replace(":","-")}_{target}_{args.split}'
                    preds.to_csv(dest / f"{name}_softmax_predictions.csv", index=False)
                    for rt, df in preds.groupby("run_type"):
                        metrics = summarize_predictions(
                            df,
                            seed=seed + 42,
                            reps=args.bootstrap_samples,
                            harm_eps=args.harm_eps,
                        )
                        metrics.update(
                            {
                                "model_tag": model,
                                "seed": seed,
                                "protocol": args.protocol,
                                "selection_target": target,
                                "evaluated_task": target,
                                "split": args.split,
                                "run_type": rt,
                                "pair_type": str(df.pair_type.iloc[0]),
                                "selected": bool(cfg["selected"]),
                                "layer": int(cfg["layer"]) if cfg["selected"] else -1,
                                "alpha": (
                                    float(cfg["alpha"]) if cfg["selected"] else 0.0
                                ),
                                "original_hard_tau": float(cfg["tau"]),
                                "temperature": temperature,
                                "center": center,
                            }
                        )
                        all_results.append(metrics)
                        log(
                            f'[Med-HALT soft] {name}/{rt}: base={metrics["acc_base"]:.3f} '
                            f'always={metrics["acc_always"]:.3f} soft={metrics["acc_soft"]:.3f} '
                            f'P(on)={metrics["mean_p_on"]:.3f} UCB={metrics["ucb_soft"]:.3f}'
                        )
            finally:
                del engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if not all_results:
        raise RuntimeError("No Med-HALT results were produced")
    pd.DataFrame(all_results).to_csv(dest / "softmax_medhalt_summary.csv", index=False)
    log(f"Saved {len(all_results)} group summaries to {dest}")


if __name__ == "__main__":
    main()
