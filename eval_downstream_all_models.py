from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from tqdm import tqdm

from experiment_runner import (
    RiskRouter,
    default_router_layers,
    router_features,
)
from patching_engine import CausalEngine, Intervention, log
from safe_layer_steering import wilson_upper

MODELS = {
    "mediphi_3.8b": "microsoft/MediPhi-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "med42_8b": "m42-health/Llama3-Med42-8B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}

HALLUCINATION_TASKS = {"FCT", "NOTA"}

DATASETS = {
    "medqa": (
        "GBaker/MedQA-USMLE-4-options",
        None,
        "test",
        "ABCD",
    ),
    "pubmedqa": (
        "qiaojin/PubMedQA",
        "pqa_labeled",
        "train",
        ("yes", "no", "maybe"),
    ),
}


def load_samples(
    name: str,
    n: int,
    sample_seed: int,
) -> list[dict]:
    repo, config, split, answers = DATASETS[name]

    if config:
        ds = load_dataset(repo, config, split=split)
    else:
        ds = load_dataset(repo, split=split)

    ds = ds.add_column(
        "_original_index",
        list(range(len(ds))),
    )

    ds = ds.shuffle(seed=sample_seed)
    ds = ds.select(range(min(n, len(ds))))

    samples = []

    for row in ds:
        if name == "medqa":
            opts = row["options"]

            choices = "\n".join(f"{c}) {opts[c]}" for c in answers)

            prompt = (
                "Answer the following multiple-choice medical question. "
                "Reply with the letter of the correct option.\n\n"
                f"Question: {row['question'].strip()}\n"
                f"Options:\n{choices}"
            )

            gold = str(row["answer_idx"]).upper()

        else:
            context = " ".join(row["context"]["contexts"]).strip()

            prompt = (
                "Based on the following medical abstract, "
                "answer the research question with yes, no, or maybe.\n\n"
                f"Context: {context}\n\n"
                f"Question: {row['question'].strip()}"
            )

            gold = str(row["final_decision"]).strip().lower()

        if gold not in answers:
            raise ValueError(f"Unexpected gold answer {gold!r} in {name}")

        samples.append(
            {
                "dataset_index": int(row["_original_index"]),
                "prompt": prompt,
                "prefill": "Answer:",
                "answers": list(answers),
                "gold": gold,
            }
        )

    return samples


def _load_checkpoint(
    run_dir: Path,
    task: str,
) -> list[dict]:
    state = run_dir / "state"

    pairs_file = state / f"pairs_{task}.pt"
    base_file = state / f"base_{task}.pt"

    for path in (pairs_file, base_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required training checkpoint missing: {path}")

    pairs = torch.load(
        pairs_file,
        map_location="cpu",
        weights_only=False,
    )["pairs"]

    base = torch.load(
        base_file,
        map_location="cpu",
        weights_only=False,
    )

    if len(pairs) != len(base):
        raise ValueError(f"Mismatched pairs/base checkpoints for {task}")

    with (run_dir / "splits.json").open() as f:
        splits = json.load(f)

    if task not in splits or "discovery" not in splits[task]:
        raise ValueError(f"No discovery split for {task}")

    sample_to_index = {str(p["sample_id"]): i for i, p in enumerate(pairs)}

    ids = [str(s) for s in splits[task]["discovery"]]

    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate discovery IDs for {task}")

    if any(s not in sample_to_index for s in ids):
        raise ValueError(f"Unmatched discovery IDs for {task}")

    return [base[sample_to_index[s]] for s in ids]


def get_selection(
    run_dir: Path,
    protocol: str,
    target: str,
) -> dict:
    path = run_dir / "safe_layer" / "safe_layer_selection.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    matches = df[(df["protocol"] == protocol) & (df["target"] == target)]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one selection for "
            f"protocol={protocol}, target={target}; "
            f"found {len(matches)} in {path}. "
            f"No first-row fallback permitted."
        )

    row = matches.iloc[0].to_dict()

    row["selected"] = str(row["selected"]).strip().lower() == "true"

    row["selection_tasks"] = str(row["selection_tasks"]).split("+")

    if row.get("router_mode", "vocab") != "vocab":
        raise ValueError(
            "Only the candidate-free 'vocab' router "
            "is supported for external evaluation."
        )

    if row["selected"]:
        if int(row["layer"]) < 0:
            raise ValueError("Selected layer is invalid")

        if float(row["alpha"]) <= 0:
            raise ValueError("Selected steering strength is invalid")

        # -inf is a valid calibrated threshold:
        # it means always activate steering.
        if np.isnan(float(row["tau"])):
            raise ValueError("Selected configuration has no valid tau")

    return row


def fit_from_discovery(
    run_dir: Path,
    row: dict,
) -> tuple[torch.Tensor | None, RiskRouter | None]:
    """Reconstruct steering vectors and fit the original router.

    Training uses discovery examples only.
    The threshold is loaded from the validation selection CSV.
    """

    if not row["selected"]:
        return None, None

    tasks = row["selection_tasks"]
    vector_name = str(row["vec"])

    if vector_name in HALLUCINATION_TASKS:
        vector_tasks = [vector_name]

    elif vector_name == "POOLED":
        vector_tasks = [t for t in tasks if t in HALLUCINATION_TASKS]

    elif vector_name.startswith("LOTO:"):
        heldout = vector_name.split(":", 1)[1]

        if heldout in tasks:
            raise ValueError("LOTO target is present in selection tasks")

        vector_tasks = [t for t in tasks if t in HALLUCINATION_TASKS]

    else:
        raise ValueError(f"Unsupported steering vector: {vector_name}")

    if not vector_tasks or not tasks:
        raise ValueError("Empty vector or selection task list")

    # Include guard tasks in router fitting when they
    # were used by the original training protocol.
    base_by_task = {task: _load_checkpoint(run_dir, task) for task in tasks}

    vectors = []

    for task in vector_tasks:
        if task not in base_by_task:
            raise ValueError(f"Missing discovery source task: {task}")

        entries = base_by_task[task]

        vector = torch.stack(
            [b["resid_clean"].float() - b["resid_corrupt"].float() for b in entries]
        ).mean(0)

        vectors.append(vector)

    # Equal-weight average of task-level directions,
    # matching build_layer_vectors().
    steer_vector = torch.stack(vectors).mean(0).float().cpu()

    feats = []
    wrong = []

    for task in tasks:
        for b in base_by_task[task]:
            for run_type in ("clean", "corrupt"):
                feats.append(router_features(b, run_type))

                wrong.append(not bool(b[f"correct_{run_type}"]))

    router = RiskRouter(
        default_router_layers(steer_vector.shape[0]),
        mode="vocab",
    )

    router.fit(
        feats,
        np.asarray(wrong, dtype=bool),
    )

    # Preserve threshold calibrated by the original
    # Med-HALT validation experiment.
    router.tau = float(row["tau"])

    return steer_vector, router


def prepare_choices(
    engine: CausalEngine,
    sample: dict,
):
    """Score first-divergent candidate tokens.

    Reject ambiguous tokenization rather than silently
    substituting unrelated token IDs.
    """

    prompt = engine.render_prompt(
        sample["prompt"],
        sample["prefill"],
    )

    base_ids = engine.encode(prompt)

    continuations = [
        engine.continuation_ids(
            prompt,
            base_ids,
            " " + ans,
        )
        for ans in sample["answers"]
    ]

    if any(not c for c in continuations):
        raise ValueError("Candidate is not a proper prompt continuation")

    common = 0

    while (
        all(len(c) > common for c in continuations)
        and len({c[common] for c in continuations}) == 1
    ):
        common += 1

    if any(len(c) <= common for c in continuations):
        raise ValueError("Candidate is a shared-prefix-only continuation")

    cand = [c[common] for c in continuations]

    if len(set(cand)) != len(cand):
        raise ValueError("Candidate first-divergent tokens collide")

    state = engine.prepare(
        base_ids + continuations[0][:common],
        cand,
    )

    return state, base_ids, common


def evaluate(
    engine: CausalEngine,
    samples: list[dict],
    cfg: dict,
    vector: torch.Tensor | None,
    router: RiskRouter | None,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    rejected = []

    active = cfg["selected"]

    if active:
        if router is None or vector is None:
            raise ValueError("Selected intervention requires router and vector")

        if vector.shape != (
            engine.num_layers,
            engine.hidden_size,
        ):
            raise ValueError(f"Steering vector shape mismatch: {vector.shape}")

        steer_resid = vector.unsqueeze(0).to(engine.device)

        intervention = [
            Intervention(
                resid_layers=np.array(
                    [int(cfg["layer"])],
                    dtype=np.int64,
                ),
                resid_alpha=float(cfg["alpha"]),
                resid_src=0,
            )
        ]

    for i, sample in enumerate(tqdm(samples, leave=False)):
        try:
            st, ids, prefix_length = prepare_choices(
                engine,
                sample,
            )

        except ValueError as exc:
            rejected.append(
                {
                    "sample_id": i,
                    "dataset_index": sample["dataset_index"],
                    "reason": str(exc),
                }
            )
            continue

        logits = st.cand_logits.numpy()
        prediction_base = int(logits.argmax())

        prediction_always = prediction_base
        risk_score = float("nan")
        gated = False

        if active:
            steered_logits = engine.run_interventions(
                st,
                intervention,
                steer_resid=steer_resid,
                batch_size=batch_size,
            )[0]

            prediction_always = int(steered_logits.argmax())

            # Router must score the original prompt,
            # before any shared answer-prefix tokens.
            if prefix_length:
                vs, res = engine.prompt_features(ids)
            else:
                vs, res = st.vocab_stats, st.resid

            features = {
                "vocab_stats": np.asarray(vs),
                "resid": res.float().cpu(),
                "cand_logits": logits,
            }

            # Actual fitted logistic risk router.
            risk_score = float(router.predict([features])[0])

            gated = bool(risk_score > router.tau)

        prediction_gated = prediction_always if gated else prediction_base

        gold_idx = sample["answers"].index(sample["gold"])

        rows.append(
            {
                "sample_id": i,
                "dataset_index": sample["dataset_index"],
                "gold": sample["gold"],
                "pred_base": sample["answers"][prediction_base],
                "pred_always": sample["answers"][prediction_always],
                "pred_gated": sample["answers"][prediction_gated],
                "correct_base": prediction_base == gold_idx,
                "correct_always": prediction_always == gold_idx,
                "correct_gated": prediction_gated == gold_idx,
                "risk_score": risk_score,
                "gate_threshold": (float(cfg["tau"]) if active else np.nan),
                "steered_gated": gated,
            }
        )

    if not rows:
        raise ValueError("All samples failed candidate-token alignment")

    return (
        pd.DataFrame(rows),
        pd.DataFrame(rejected),
    )


def paired_bootstrap_interval(
    correct_base: np.ndarray,
    correct_after: np.ndarray,
    seed: int,
    reps: int,
) -> tuple[float, float]:

    diff = correct_after.astype(np.float64) - correct_base.astype(np.float64)

    rng = np.random.default_rng(seed)

    means = np.empty(reps, dtype=float)

    for i in range(reps):
        indices = rng.integers(
            0,
            len(diff),
            size=len(diff),
        )
        means[i] = diff[indices].mean()

    return tuple(
        float(v)
        for v in np.quantile(
            means,
            [0.025, 0.975],
        )
    )


def summarize(
    df: pd.DataFrame,
    bootstrap_seed: int,
    n_boot: int,
) -> dict:

    cb = df.correct_base.to_numpy(dtype=bool)

    result = {
        "n_samples": len(df),
        "n_correct_base": int(cb.sum()),
        "acc_base": float(cb.mean()),
        "gate_rate": float(df.steered_gated.mean()),
    }

    for method in ("always", "gated"):
        ca = df[f"correct_{method}"].to_numpy(dtype=bool)

        induced = int((cb & ~ca).sum())

        rescued = int((~cb & ca).sum())

        lo, hi = paired_bootstrap_interval(
            cb,
            ca,
            bootstrap_seed,
            n_boot,
        )

        discordant = induced + rescued

        p_mcnemar = (
            float(
                binomtest(
                    min(induced, rescued),
                    discordant,
                    0.5,
                ).pvalue
            )
            if discordant
            else 1.0
        )

        result.update(
            {
                f"acc_{method}": float(ca.mean()),
                f"delta_{method}": float(ca.mean() - cb.mean()),
                f"delta_{method}_ci_low": lo,
                f"delta_{method}_ci_high": hi,
                f"induced_{method}": induced,
                f"rescued_{method}": rescued,
                f"induction_rate_{method}": (
                    induced / cb.sum() if cb.any() else np.nan
                ),
                f"rescue_rate_{method}": (
                    rescued / (~cb).sum() if (~cb).any() else np.nan
                ),
                f"ucb_{method}": float(
                    wilson_upper(
                        induced,
                        int(cb.sum()),
                    )
                ),
                f"p_mcnemar_{method}": p_mcnemar,
            }
        )

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--results_dir",
        default="/scratch/adityasrikar.c/medhalt/results_v4",
    )

    ap.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )

    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )

    ap.add_argument(
        "--protocol",
        default="loto",
        choices=[
            "loto",
            "pooled",
            "source:FCT",
            "source:NOTA",
        ],
    )

    ap.add_argument(
        "--selection_targets",
        nargs="+",
        choices=sorted(HALLUCINATION_TASKS),
        default=["FCT", "NOTA"],
    )

    ap.add_argument(
        "--n_medqa",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--n_pubmedqa",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--sample_seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--bootstrap_samples",
        type=int,
        default=2000,
    )

    ap.add_argument(
        "--harm_eps",
        type=float,
        default=0.10,
    )

    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
    )

    ap.add_argument(
        "--attn_implementation",
        default=None,
        choices=[
            "flash_attention_2",
            "sdpa",
            "eager",
        ],
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--output_dir",
        default="./downstream_audit_corrected",
    )

    args = ap.parse_args()

    if (
        min(
            args.n_medqa,
            args.n_pubmedqa,
            args.bootstrap_samples,
        )
        <= 0
    ):
        ap.error("Sample counts and bootstrap_samples must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "medqa": load_samples(
            "medqa",
            args.n_medqa,
            args.sample_seed,
        ),
        "pubmedqa": load_samples(
            "pubmedqa",
            args.n_pubmedqa,
            args.sample_seed,
        ),
    }

    with (out_dir / "audit_config.json").open("w") as f:
        json.dump(
            vars(args),
            f,
            indent=2,
        )

    results = []

    for tag in args.models:
        model_name = MODELS[tag]

        for seed in args.seeds:
            run_dir = Path(args.results_dir) / f"{tag}_seed{seed}"

            if not run_dir.is_dir():
                log(f"[SKIP] No run directory: {run_dir}")
                continue

            selected = {}

            for target in args.selection_targets:
                try:
                    selected[target] = get_selection(
                        run_dir,
                        args.protocol,
                        target,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    log(f"[SKIP] {tag}/seed{seed}/{target}: {exc}")

            if not selected:
                log(f"[SKIP] No matching configurations in {run_dir}")
                continue

            log(f"[Model] {model_name} seed={seed}")

            engine = CausalEngine(
                model_name,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
            )

            try:
                for target, cfg in selected.items():
                    vector, router = fit_from_discovery(
                        run_dir,
                        cfg,
                    )

                    label = (
                        f"{tag}_seed{seed}_"
                        f"{args.protocol.replace(':', '-')}_"
                        f"{target}"
                    )

                    for dataset_name, samples in data.items():
                        preds, rejects = evaluate(
                            engine,
                            samples,
                            cfg,
                            vector,
                            router,
                            args.batch_size,
                        )

                        preds.to_csv(
                            out_dir / f"{label}_{dataset_name}_predictions.csv",
                            index=False,
                        )

                        rejects.to_csv(
                            out_dir / f"{label}_{dataset_name}_rejected.csv",
                            index=False,
                        )

                        metrics = summarize(
                            preds,
                            args.sample_seed + seed,
                            args.bootstrap_samples,
                        )

                        metrics.update(
                            {
                                "model_tag": tag,
                                "model_name": model_name,
                                "seed": seed,
                                "protocol": args.protocol,
                                "selection_target": target,
                                "selection_vec": cfg["vec"],
                                "selection_tasks": "+".join(cfg["selection_tasks"]),
                                "selected": cfg["selected"],
                                "layer": int(cfg["layer"]),
                                "alpha": float(cfg["alpha"]),
                                "tau": float(cfg["tau"]),
                                "dataset": dataset_name,
                                "dataset_split": DATASETS[dataset_name][2],
                                "sample_seed": args.sample_seed,
                                "n_requested": len(samples),
                                "n_rejected": len(rejects),
                                "safety_ucb_gated_le_eps": (
                                    metrics["ucb_gated"] <= args.harm_eps
                                ),
                            }
                        )

                        results.append(metrics)

                        log(
                            f"[Result] {tag}/{seed}/{target}/{dataset_name}: "
                            f"base={metrics['acc_base']:.3f}, "
                            f"always={metrics['acc_always']:.3f}, "
                            f"gated={metrics['acc_gated']:.3f}, "
                            f"route={metrics['gate_rate']:.3f}"
                        )

            finally:
                del engine
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if not results:
        raise RuntimeError(
            "No evaluations completed. Check run paths and selected models."
        )

    summary = pd.DataFrame(results)

    # Preserve original output filename for compatibility.
    path = out_dir / "downstream_zero_degradation_summary.csv"

    summary.to_csv(
        path,
        index=False,
    )

    aggregate = summary.groupby(
        [
            "model_tag",
            "model_name",
            "protocol",
            "selection_target",
            "dataset",
        ],
        as_index=False,
    ).agg(
        n_seeds=("seed", "nunique"),
        acc_base_mean=("acc_base", "mean"),
        acc_always_mean=("acc_always", "mean"),
        acc_gated_mean=("acc_gated", "mean"),
        delta_gated_mean=("delta_gated", "mean"),
        delta_gated_std=("delta_gated", "std"),
        gate_rate_mean=("gate_rate", "mean"),
        induced_gated_mean=("induced_gated", "mean"),
        rescued_gated_mean=("rescued_gated", "mean"),
    )

    aggregate.to_csv(
        out_dir / "downstream_over_seeds.csv",
        index=False,
    )

    cols = [
        "model_tag",
        "seed",
        "selection_target",
        "dataset",
        "acc_base",
        "acc_always",
        "acc_gated",
        "delta_gated",
        "gate_rate",
        "ucb_gated",
        "p_mcnemar_gated",
    ]

    print("\nDOWNSTREAM SAFETY AUDIT " "(not a zero-degradation claim)\n")

    print(summary[cols].round(4).to_string(index=False))

    log(f"Summary: {path}")


if __name__ == "__main__":
    main()
