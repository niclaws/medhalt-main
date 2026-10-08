#!/usr/bin/env python3
"""
eval_downstream_all_models.py — Downstream Zero-Degradation Evaluation.

Evaluates MedQA (USMLE 4-options) and PubMedQA (yes/no/maybe) on all 4 models:
  1. microsoft/MediPhi-Instruct
  2. mistralai/Mistral-7B-Instruct-v0.3
  3. m42-health/Llama3-Med42-8B
  4. Qwen/Qwen3-8B

Evaluates:
  - none (unmodified baseline)
  - safe_layer_always (unconditional safe-layer steering)
  - safe_layer_gated (inference-time risk-router gated steering)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from experiment_runner import RiskRouter
from patching_engine import CausalEngine, Intervention, log
from safe_layer_steering import mcnemar_exact, wilson_upper

MODELS = [
    ("mediphi_3.8b", "microsoft/MediPhi-Instruct"),
    ("mistral_7b", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("med42_8b", "m42-health/Llama3-Med42-8B"),
    ("qwen3_8b", "Qwen/Qwen3-8B"),
]

MCQ_LETTERS = ["A", "B", "C", "D"]
PUBMED_LABELS = ["yes", "no", "maybe"]


def format_medqa_prompt(question: str, options: Dict[str, str]) -> Tuple[str, str]:
    opts_str = "\n".join(f"{k}) {options[k]}" for k in MCQ_LETTERS if k in options)
    prompt = (
        "Answer the following multiple-choice medical question. "
        "Reply with the letter of the correct option.\n\n"
        f"Question: {question.strip()}\nOptions:\n{opts_str}"
    )
    return prompt, "Answer:"


def format_pubmedqa_prompt(question: str, context: Dict) -> Tuple[str, str]:
    ctx_list = context.get("contexts", [])
    ctx_text = " ".join(ctx_list).strip()
    prompt = (
        "Based on the following medical abstract, answer the research question with yes, no, or maybe.\n\n"
        f"Context: {ctx_text}\n\nQuestion: {question.strip()}"
    )
    return prompt, "Answer:"


def load_dataset_samples(dataset_name: str, n_samples: int) -> List[Dict]:
    if dataset_name == "medqa":
        log(f"[Data] Loading MedQA-USMLE-4-options test split ({n_samples} samples)...")
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        samples = []
        for row in ds.select(range(min(n_samples, len(ds)))):
            p, pf = format_medqa_prompt(row["question"], row["options"])
            samples.append(
                {
                    "prompt": p,
                    "prefill": pf,
                    "gold_str": row["answer_idx"],
                    "cand_strings": [" " + l for l in MCQ_LETTERS],
                    "cand_labels": MCQ_LETTERS,
                }
            )
        return samples
    elif dataset_name == "pubmedqa":
        log(
            f"[Data] Loading PubMedQA (pqa_labeled) train split ({n_samples} samples)..."
        )
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        samples = []
        for row in ds.select(range(min(n_samples, len(ds)))):
            p, pf = format_pubmedqa_prompt(row["question"], row["context"])
            samples.append(
                {
                    "prompt": p,
                    "prefill": pf,
                    "gold_str": row["final_decision"].strip().lower(),
                    "cand_strings": [" " + l for l in PUBMED_LABELS],
                    "cand_labels": PUBMED_LABELS,
                }
            )
        return samples
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_steering_artifacts(results_dir: str, tag: str, seed: int, protocol: str):
    """Loads optimal layer, alpha, tau, and steering direction from run checkpoints."""
    run_dir = os.path.join(results_dir, f"{tag}_seed{seed}")
    sel_path = os.path.join(run_dir, "safe_layer", "safe_layer_selection.csv")
    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"Missing {sel_path}")

    sel_df = pd.read_csv(sel_path)
    matched = sel_df[sel_df.protocol == protocol]
    if matched.empty:
        matched = sel_df.iloc[[0]]

    cfg = matched.iloc[0]
    layer = int(cfg.layer)
    alpha = float(cfg.alpha)
    tau = float(cfg.tau)
    vec_name = str(cfg.vec)

    st_dir = os.path.join(run_dir, "state")
    v_fct = torch.stack(
        [
            b["resid_clean"].float() - b["resid_corrupt"].float()
            for b in torch.load(os.path.join(st_dir, "base_FCT.pt"), weights_only=False)
        ]
    ).mean(0)
    v_nota = torch.stack(
        [
            b["resid_clean"].float() - b["resid_corrupt"].float()
            for b in torch.load(
                os.path.join(st_dir, "base_NOTA.pt"), weights_only=False
            )
        ]
    ).mean(0)

    if vec_name == "FCT" or "FCT" in vec_name:
        steer_vec = v_fct
    elif vec_name == "NOTA" or "NOTA" in vec_name:
        steer_vec = v_nota
    else:
        steer_vec = (v_fct + v_nota) / 2.0

    return layer, alpha, tau, steer_vec


def eval_model_on_dataset(
    engine: CausalEngine,
    samples: List[Dict],
    layer: int,
    alpha: float,
    tau: float,
    steer_vec: torch.Tensor,
    batch_size: int = 64,
) -> pd.DataFrame:
    steer_tensor = steer_vec.unsqueeze(0).to(engine.device)
    rows = []

    for i, s in enumerate(tqdm(samples, leave=False)):
        p_text = engine.render_prompt(s["prompt"], s["prefill"])
        p_ids = engine.encode(p_text)

        c_ids = []
        for cs in s["cand_strings"]:
            cont = engine.continuation_ids(p_text, p_ids, cs)
            c_ids.append(cont[0] if cont else engine.encode(cs.strip())[0])

        gold_idx = s["cand_labels"].index(s["gold_str"])

        st = engine.prepare(p_ids, c_ids)
        base_logits = st.cand_logits.numpy()
        pred_base = int(base_logits.argmax())

        iv = [
            Intervention(resid_layers=np.array([layer]), resid_alpha=alpha, resid_src=0)
        ]
        steered_logits = engine.run_interventions(
            st, iv, steer_resid=steer_tensor, batch_size=batch_size
        )[0]
        pred_steered = int(steered_logits.argmax())

        # Candidate-free next-token normalized entropy as deployable router proxy
        risk_score = float(st.vocab_stats[0]) / max(
            np.log(engine.tokenizer.vocab_size), 1e-6
        )
        steered = bool(risk_score > tau)
        pred_gated = pred_steered if steered else pred_base

        rows.append(
            {
                "sample_id": i,
                "gold": s["gold_str"],
                "correct_base": bool(pred_base == gold_idx),
                "correct_always": bool(pred_steered == gold_idx),
                "steered_gated": steered,
                "correct_gated": bool(pred_gated == gold_idx),
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-model downstream audit on MedQA & PubMedQA"
    )
    ap.add_argument(
        "--results_dir",
        default="/scratch/adityasrikar.c/medhalt/results_v4",
        help="Base scratch directory containing results_v4 runs",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Seed run to pull steering vectors from"
    )
    ap.add_argument(
        "--protocol",
        default="loto",
        choices=["loto", "pooled", "source:FCT", "source:NOTA"],
    )
    ap.add_argument(
        "--n_medqa", type=int, default=200, help="Number of MedQA test samples"
    )
    ap.add_argument(
        "--n_pubmedqa", type=int, default=200, help="Number of PubMedQA test samples"
    )
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output_dir", default="./downstream_audit_results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    medqa_samples = load_dataset_samples("medqa", args.n_medqa)
    pubmedqa_samples = load_dataset_samples("pubmedqa", args.n_pubmedqa)

    summary_records = []

    for tag, model_name in MODELS:
        log(f"\n{'='*70}\n[Model] Auditing {model_name} ({tag})\n{'='*70}")
        try:
            layer, alpha, tau, steer_vec = load_steering_artifacts(
                args.results_dir, tag, args.seed, args.protocol
            )
            log(f"[Artifacts] Layer: {layer}, Alpha: {alpha:.2f}, Tau: {tau:.3f}")
        except Exception as e:
            log(f"[SKIP] Could not load artifacts for {tag}: {e}")
            continue

        engine = CausalEngine(model_name, dtype=args.dtype)

        for d_name, d_samples in [
            ("MedQA", medqa_samples),
            ("PubMedQA", pubmedqa_samples),
        ]:
            log(f"[Run] Evaluating {d_name} ({len(d_samples)} items)...")
            df = eval_model_on_dataset(
                engine, d_samples, layer, alpha, tau, steer_vec, args.batch_size
            )
            csv_path = os.path.join(
                args.output_dir, f"{tag}_{d_name.lower()}_predictions.csv"
            )
            df.to_csv(csv_path, index=False)

            n = len(df)
            acc_base = float(df.correct_base.mean())
            acc_always = float(df.correct_always.mean())
            acc_gated = float(df.correct_gated.mean())

            induced_always = int((df.correct_base & ~df.correct_always).sum())
            rescued_always = int((~df.correct_base & df.correct_always).sum())
            induced_gated = int((df.correct_base & ~df.correct_gated).sum())
            rescued_gated = int((~df.correct_base & df.correct_gated).sum())

            n_corr = int(df.correct_base.sum())
            ucb_always = wilson_upper(induced_always, n_corr)
            ucb_gated = wilson_upper(induced_gated, n_corr)

            summary_records.append(
                {
                    "model_tag": tag,
                    "model_name": model_name,
                    "dataset": d_name,
                    "n_samples": n,
                    "acc_base": acc_base,
                    "acc_always": acc_always,
                    "acc_gated": acc_gated,
                    "delta_always": acc_always - acc_base,
                    "delta_gated": acc_gated - acc_base,
                    "gate_rate": float(df.steered_gated.mean()),
                    "induced_always": induced_always,
                    "rescued_always": rescued_always,
                    "induced_gated": induced_gated,
                    "rescued_gated": rescued_gated,
                    "ucb_always": ucb_always,
                    "ucb_gated": ucb_gated,
                    "p_mcnemar_gated": mcnemar_exact(rescued_gated, induced_gated),
                }
            )

        del engine
        gc.collect()
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_records)
    sum_path = os.path.join(args.output_dir, "downstream_zero_degradation_summary.csv")
    summary_df.to_csv(sum_path, index=False)

    print("\n" + "=" * 90)
    print("                DOWNSTREAM ZERO-DEGRADATION SUMMARY REPORT                 ")
    print("=" * 90)
    print(
        summary_df[
            [
                "model_tag",
                "dataset",
                "acc_base",
                "acc_gated",
                "delta_gated",
                "gate_rate",
                "ucb_gated",
                "p_mcnemar_gated",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("=" * 90)
    log(f"Wrote summary table to {sum_path}")


if __name__ == "__main__":
    main()
