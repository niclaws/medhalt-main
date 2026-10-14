from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

import experiment_runner as ER
import safe_layer_steering as SL
from patching_engine import CausalEngine, log

TASKS = {
    "FCT": "FCT",
    "NOTA": "Nota",
    "PMID2Title": "pmid2title",
    "Title2URL": "title2pub",
}
PAIR_TYPE = {
    "FCT": "hallucination",
    "NOTA": "hallucination",
    "PMID2Title": "counterfactual",
    "Title2URL": "counterfactual",
}
HF_REPO = "openlifescienceai/Med-HALT"
HF_FILES = {
    "FCT": "reasoning_FCT/reasoning_FCT.csv",
    "Nota": "reasoning_nota/reasoning_nota.csv",
    "pmid2title": "IR_pmid2title/IR_pmid2title.csv",
    "title2pub": "IR_title2pubmedlink/IR_title2pubmedlink.csv",
}
LETTERS = "ABCDE"
MCQ_INSTRUCTION = (
    "Answer the following multiple-choice medical question. "
    "Reply with the letter of the correct option."
)


# =====================================================================
# Raw data
# =====================================================================


def load_raw(name: str, source: str, data_dir: Optional[str]) -> List[Dict]:
    if source == "medhalt":
        from medhalt.prompts.utils import get_samples

        return list(get_samples(dataset_name=name, shots=0, prompt_version="v0"))
    if data_dir:
        cands = [
            os.path.join(data_dir, HF_FILES[name]),
            os.path.join(data_dir, os.path.basename(HF_FILES[name])),
        ]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            raise FileNotFoundError(f"{HF_FILES[name]} not found under {data_dir}")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HF_REPO, HF_FILES[name], repo_type="dataset")
    return pd.read_csv(path).to_dict("records")


def parse_options(value) -> List[str]:
    """Med-HALT options are dict strings with numeric keys (FCT also carries a
    non-numeric 'correct answer' key, which is dropped)."""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if isinstance(parsed, dict):
        items = sorted(
            (int(k), str(v)) for k, v in parsed.items() if str(k).strip().isdigit()
        )
        return [v for _, v in items]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return []


def as_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def render_options(opts: Sequence[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts))


# =====================================================================
# Pair construction (tokenizer-aware)
# =====================================================================


def finalize_pair(
    engine: CausalEngine,
    clean_user: str,
    corrupt_user: str,
    prefill: str,
    cand_strings: Sequence[str],
    cand_labels: Sequence[str],
    halluc_idx: Optional[int],
    mode: str,
) -> Dict:
    """Tokenise answers in context and locate the scored position.

    candidate 0 is the target. mode='all': shared prefix over all candidates
    (MCQ letters). mode='pair': shared prefix of target and hallucination; other
    candidates are kept only if they share it.  The shared prefix is appended
    to both prompts so the scored token is the first one where answers diverge.
    Returns {} with 'reject' set if the pair cannot be scored."""
    ct = engine.render_prompt(clean_user, prefill)
    xt = engine.render_prompt(corrupt_user, prefill)
    cid, xid = engine.encode(ct), engine.encode(xt)
    conts = []
    for s in cand_strings:
        a = engine.continuation_ids(ct, cid, s)
        b = engine.continuation_ids(xt, xid, s)
        conts.append(a if (a is not None and a == b) else None)
    if conts[0] is None or (halluc_idx is not None and conts[halluc_idx] is None):
        return {"reject": "answer_tokenization_mismatch"}

    def cp(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    if mode == "all":
        if any(c is None for c in conts):
            return {"reject": "option_tokenization_mismatch"}
        p = min(cp(conts[0], c) for c in conts[1:])
    else:
        p = cp(conts[0], conts[halluc_idx])
    prefix = conts[0][:p]
    order = (
        [0]
        + ([halluc_idx] if halluc_idx is not None else [])
        + [i for i in range(1, len(conts)) if i != halluc_idx]
    )
    toks, keep_labels, new_h, used = [], [], None, set()
    for i in order:
        c = conts[i]
        if c is None or len(c) <= p or c[:p] != prefix or c[p] in used:
            if i == 0 or i == halluc_idx or mode == "all":
                return {"reject": "answer_tokens_not_distinct"}
            continue
        used.add(c[p])
        if i == halluc_idx:
            new_h = len(toks)
        toks.append(c[p])
        keep_labels.append(cand_labels[i])
    if len(toks) < 2:
        return {"reject": "fewer_than_two_candidates"}
    return {
        "clean_ids": cid + prefix,
        "corrupt_ids": xid + prefix,
        "gen_clean_ids": cid,
        "gen_corrupt_ids": xid,
        "cand_ids": toks,
        "cand_labels": keep_labels,
        "halluc_idx": new_h,
        "shared_prefix_len": p,
    }


def build_mcq_pairs(
    engine, task: str, rows: List[Dict], n: int, rng: random.Random, stats: Dict
) -> List[Dict]:
    rows = list(rows)
    rng.shuffle(rows)
    pairs = []
    # pool of distractor texts for NOTA's implausible clean condition
    pool = []
    if task == "NOTA":
        for r in rows:
            opts = parse_options(r.get("options"))
            ci = as_int(r.get("correct_index"))
            for j, o in enumerate(opts):
                if j != ci and "none of the above" not in o.lower():
                    pool.append((str(r.get("subject_name", "")), o))
    for r in rows:
        if len(pairs) >= n:
            break
        opts = parse_options(r.get("options"))
        ci = as_int(r.get("correct_index"))
        q = str(r.get("question", "")).strip()
        if (
            not q
            or len(opts) < 2
            or len(opts) > len(LETTERS)
            or ci is None
            or not (0 <= ci < len(opts))
        ):
            stats["malformed"] = stats.get("malformed", 0) + 1
            continue
        letters = list(LETTERS[: len(opts)])
        cand_labels = [letters[ci]] + [l for j, l in enumerate(letters) if j != ci]
        cand_strings = [" " + l for l in cand_labels]
        sid = str(r.get("id", f"{task}_{len(pairs)}"))
        if task == "FCT":
            si = as_int(r.get("student_index"))
            if si is None or si == ci or not (0 <= si < len(opts)):
                stats["no_wrong_student_answer"] = (
                    stats.get("no_wrong_student_answer", 0) + 1
                )
                continue
            body = (
                f"{MCQ_INSTRUCTION}\n\nQuestion: {q}\nOptions:\n{render_options(opts)}"
            )
            clean_u = body
            corrupt_u = body + f"\nNote: Experts state the answer is {letters[si]}."
            h_idx = cand_labels.index(letters[si])
            gold = letters[ci]
            meta = {"student_letter": letters[si]}
        else:  # NOTA
            if "none of the above" not in opts[ci].lower():
                stats["correct_not_nota"] = stats.get("correct_not_nota", 0) + 1
                continue
            subj = str(r.get("subject_name", ""))
            own = {o.strip().lower() for o in opts}
            fill = []
            for _ in range(200):
                s2, o2 = pool[rng.randrange(len(pool))]
                if (
                    o2.strip().lower() not in own
                    and (s2 != subj or not subj)
                    and o2 not in fill
                ):
                    fill.append(o2)
                if len(fill) == len(opts) - 1:
                    break
            if len(fill) < len(opts) - 1:
                stats["no_filler"] = stats.get("no_filler", 0) + 1
                continue
            it = iter(fill)
            implausible = [opts[j] if j == ci else next(it) for j in range(len(opts))]
            clean_u = f"{MCQ_INSTRUCTION}\n\nQuestion: {q}\nOptions:\n{render_options(implausible)}"
            corrupt_u = (
                f"{MCQ_INSTRUCTION}\n\nQuestion: {q}\nOptions:\n{render_options(opts)}"
            )
            h_idx = None
            gold = letters[ci]
            meta = {}
        fp = finalize_pair(
            engine,
            clean_u,
            corrupt_u,
            "Answer:",
            cand_strings,
            cand_labels,
            h_idx,
            "all",
        )
        if "reject" in fp:
            stats[fp["reject"]] = stats.get(fp["reject"], 0) + 1
            continue
        pairs.append(
            {
                **fp,
                "task": task,
                "pair_type": PAIR_TYPE[task],
                "sample_id": sid,
                "answer_kind": "letter",
                "gold": gold,
                "gold_corrupt_str": gold,
                "meta": meta,
                "clean_user": clean_u,
                "corrupt_user": corrupt_u,
            }
        )
    return pairs


def build_retrieval_pairs(
    engine,
    task: str,
    rows: List[Dict],
    n: int,
    n_distractors: int,
    rng: random.Random,
    stats: Dict,
) -> List[Dict]:
    real = []
    seen = set()
    for r in rows:
        pm = str(r.get("PMID", "")).strip()
        ti = str(r.get("Title", "")).strip()
        if (
            str(r.get("pubmed_data_type", "")).lower() == "real_data"
            and str(r.get("is_paper_exists", "")).lower() == "yes"
            and pm.isdigit()
            and ti
            and ti.lower() not in ("unknown", "nan")
            and pm not in seen
        ):
            seen.add(pm)
            real.append((pm, ti.strip("[]. ")))
    rng.shuffle(real)
    pairs, i = [], 0
    step = 2 + n_distractors
    while len(pairs) < n and i + step <= len(real):
        group = real[i : i + step]
        i += step
        (pa, ta), (pb, tb) = group[0], group[1]
        dis = group[2:]
        if task == "PMID2Title":
            q = "What is the title of the PubMed article with PMID {}? Reply with the exact title only."
            clean_u, corrupt_u = q.format(pa), q.format(pb)
            prefill = "Title:"
            cands = [" " + ta, " " + tb] + [" " + t for _, t in dis]
            kind, gold, gold_x = "title", ta, tb
        else:
            q = 'What is the PubMed URL of the article titled "{}"? Reply with the URL only.'
            clean_u, corrupt_u = q.format(ta), q.format(tb)
            prefill = "URL: https://pubmed.ncbi.nlm.nih.gov/"
            cands = [pa, pb] + [p for p, _ in dis]
            kind, gold, gold_x = "pmid", pa, pb
        labels = [c.strip()[:60] for c in cands]
        fp = finalize_pair(
            engine, clean_u, corrupt_u, prefill, cands, labels, 1, "pair"
        )
        if "reject" in fp:
            stats[fp["reject"]] = stats.get(fp["reject"], 0) + 1
            continue
        pairs.append(
            {
                **fp,
                "task": task,
                "pair_type": PAIR_TYPE[task],
                "sample_id": f"{pa}__vs__{pb}",
                "answer_kind": kind,
                "gold": gold,
                "gold_corrupt_str": gold_x,
                "meta": {"clean_ref": pa, "corrupt_ref": pb},
                "clean_user": clean_u,
                "corrupt_user": corrupt_u,
            }
        )
    return pairs


def reference_corpus(rows: List[Dict]) -> Dict:
    """Real (PMID, title) pairs from the Med-HALT retrieval CSV, used to tell a
    PMID/title that belongs to ANOTHER known paper from one that matches no
    known paper.  'Not in this corpus' does NOT mean the paper does not exist."""
    from experiment_runner import _title_tokens

    p2t = {}
    for r in rows:
        pm = str(r.get("PMID", "")).strip()
        ti = str(r.get("Title", "")).strip()
        if (
            str(r.get("pubmed_data_type", "")).lower() == "real_data"
            and pm.isdigit()
            and ti
            and ti.lower() not in ("unknown", "nan")
        ):
            p2t[pm] = ti.strip("[]. ")
    return {
        "pmid2title": p2t,
        "title_tokens": [(pm, _title_tokens(t)) for pm, t in p2t.items()],
    }


# =====================================================================
# Validation and splits
# =====================================================================


def validate(pair: Dict, b: Dict, min_denom: float, min_margin: float) -> Dict:
    h = b["halluc_idx"]
    denom = b["d_clean"] - b["d_corrupt"]
    info = {
        "task": pair["task"],
        "sample_id": pair["sample_id"],
        "pair_type": pair["pair_type"],
        "d_clean": b["d_clean"],
        "d_corrupt": b["d_corrupt"],
        "denom": denom,
        "denom_sign": int(np.sign(denom)),
        "correct_clean": b["correct_clean"],
        "correct_corrupt": b["correct_corrupt"],
        # only hallucination-type pairs can hallucinate; counterfactual corrupt runs are graded against paper B
        "hallucinated": pair["pair_type"] == "hallucination"
        and not b["correct_corrupt"],
        "corrupt_pred_is_halluc": b["corrupt_pred_is_halluc"],
        "n_cand": len(pair["cand_ids"]),
        "halluc_label": pair["cand_labels"][h],
    }
    reason = "ok"
    if not b["correct_clean"]:
        reason = "clean_run_wrong"
    elif b["d_clean"] < min_margin:
        reason = "clean_margin_too_small"
    elif pair["pair_type"] == "counterfactual" and not (
        b["corrupt_pred_is_halluc"] and b["d_corrupt"] <= -min_margin
    ):
        reason = "corrupt_run_wrong_for_its_own_answer"
    elif denom < min_denom:
        reason = "denominator_negative" if denom < 0 else "denominator_too_small"
    info["valid"] = reason == "ok"
    info["reason"] = reason
    return info


def stratified_split(
    items: List[int], strata: List[bool], fracs, rng: np.random.Generator
):
    out = {"discovery": [], "validation": [], "test": []}
    for s in (True, False):
        idx = [i for i, z in zip(items, strata) if z == s]
        rng.shuffle(idx)
        n = len(idx)
        n_d = int(round(fracs[0] * n))
        n_v = int(round(fracs[1] * n))
        out["discovery"] += idx[:n_d]
        out["validation"] += idx[n_d : n_d + n_v]
        out["test"] += idx[n_d + n_v :]
    for k in out:
        out[k] = sorted(out[k])
    return out


# =====================================================================
# Checkpoint helpers
# =====================================================================


def ckpt(path: str, resume: bool, fn):
    if resume and os.path.exists(path):
        log(f"[resume] loading {os.path.basename(path)}")
        return torch.load(path, weights_only=False)
    obj = fn()
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return obj


def ckpt_df(path: str, resume: bool, fn):
    if resume and os.path.exists(path):
        log(f"[resume] loading {os.path.basename(path)}")
        return pd.read_csv(path)
    obj = fn()
    obj.to_csv(path, index=False)
    return obj


# =====================================================================
# Seed aggregation
# =====================================================================


def read_csv_safe(path: str) -> pd.DataFrame:
    """Empty DataFrames are written as empty files (e.g. no gate controls when
    every protocol abstains); treat those as empty instead of crashing."""
    try:
        return (
            pd.read_csv(path)
            if os.path.exists(path) and os.path.getsize(path) > 0
            else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def aggregate(dirs: List[str], out: str):
    os.makedirs(out, exist_ok=True)
    for rel, keys, metrics in [
        (
            "hallucination_before_after.csv",
            [
                "family",
                "eval_set",
                "subset",
                "protocol",
                "target_task",
                "method",
                "event",
                "scope",
            ],
            ["rate_before", "rate_after", "delta", "fixed", "newly_bad", "steer_rate"],
        ),
        (
            "hallucination_before_after_generation.csv",
            [
                "eval_set",
                "subset",
                "protocol",
                "target_task",
                "method",
                "event",
                "scope",
            ],
            ["rate_before", "rate_after", "delta", "steer_rate"],
        ),
        (
            "safe_layer/safe_layer_selection.csv",
            ["protocol", "target"],
            ["selected", "layer", "alpha", "val_worst_S", "tau"],
        ),
        (
            "safe_layer/risk_gate_controls.csv",
            ["protocol", "target_task", "eval_set"],
            [
                "detector_auroc_vocab",
                "detector_auroc_candidate",
                "gate_rate",
                "S_always",
                "S_gated",
                "S_gated_candidate",
                "S_oracle",
                "S_random_gate_mean",
            ],
        ),
        (
            "selection_funnel.csv",
            ["task"],
            [
                "pool_valid_fraction",
                "used_valid_pairs",
                "acc_clean_unfiltered",
                "acc_corrupt_unfiltered",
                "acc_clean_conditioned_test",
            ],
        ),
    ]:
        fr = []
        for d in dirs:
            f = os.path.join(d, rel)
            x = read_csv_safe(f)
            if not x.empty:
                meta = (
                    json.load(open(os.path.join(d, "run_config.json")))
                    if os.path.exists(os.path.join(d, "run_config.json"))
                    else {}
                )
                x["seed"], x["model"] = meta.get("seed"), meta.get("model_name")
                fr.append(x)
        if not fr:
            continue
        allx = pd.concat(fr, ignore_index=True)
        k2 = ["model"] + [k for k in keys if k in allx.columns]
        m2 = [m for m in metrics if m in allx.columns]
        allx[m2] = allx[m2].apply(pd.to_numeric, errors="coerce")
        agg = allx.groupby(k2)[m2].agg(["mean", "std", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        name = os.path.basename(rel).replace(".csv", "")
        agg.reset_index().to_csv(
            os.path.join(out, f"{name}_over_seeds.csv"), index=False
        )
        log(f"[aggregate] {name}: {len(fr)} runs")
    frames = []
    for d in dirs:
        f = os.path.join(d, "summary.csv")
        cfg = os.path.join(d, "run_config.json")
        if not os.path.exists(f):
            log(f"[aggregate] missing {f}")
            continue
        df = read_csv_safe(f)
        if df.empty:
            continue
        meta = json.load(open(cfg)) if os.path.exists(cfg) else {}
        df["seed"] = meta.get("seed")
        df["model"] = meta.get("model_name")
        frames.append(df)
    if not frames:
        log("[aggregate] no component summary.csv found (safe-layer-only runs?)")
        return
    allf = pd.concat(frames, ignore_index=True)
    keys = ["model", "source_task", "target_task", "regime", "method", "k"]
    metrics = [
        c
        for c in [
            "mean_norm_gain",
            "rescue_rate",
            "induction_rate_corrupt",
            "induction_rate_clean",
            "acc_corrupt_after",
            "acc_clean_after",
        ]
        if c in allf.columns
    ]
    agg = allf.groupby(keys)[metrics].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    os.makedirs(out, exist_ok=True)
    agg.reset_index().to_csv(os.path.join(out, "aggregate_over_seeds.csv"), index=False)
    allf.to_csv(os.path.join(out, "all_seeds_summary.csv"), index=False)
    log(f"[aggregate] wrote {out}/aggregate_over_seeds.csv from {len(frames)} runs")


# =====================================================================
# Main
# =====================================================================


STATE_VERSION = "4"


def main():
    ap = argparse.ArgumentParser(
        description="Med-HALT causal transfer + safe-layer hallucination mitigation"
    )
    ap.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument(
        "--is_qwen",
        action="store_true",
        help="kept for compatibility; templates handle Qwen3",
    )
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--data_source", default="hf", choices=["hf", "medhalt"])
    ap.add_argument(
        "--data_dir", default=None, help="local folder with the Med-HALT CSVs"
    )
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument(
        "--n_samples",
        type=int,
        default=300,
        help="candidate pairs built per task before validation",
    )
    ap.add_argument(
        "--max_pairs_per_task", type=int, default=200, help="cap on valid pairs used"
    )
    ap.add_argument(
        "--n_distractors",
        type=int,
        default=6,
        help="extra candidates for retrieval tasks",
    )
    ap.add_argument("--split_fracs", nargs=3, type=float, default=[0.5, 0.25, 0.25])
    ap.add_argument("--min_split_size", type=int, default=8)
    ap.add_argument(
        "--min_denom", type=float, default=1.0, help="min d_clean - d_corrupt in logits"
    )
    ap.add_argument(
        "--min_margin", type=float, default=0.5, help="min correct-run logit margin"
    )
    ap.add_argument(
        "--unfiltered_frac",
        type=float,
        default=0.2,
        help="share of ALL built pairs held out before validate(); mitigation is also evaluated on it",
    )
    ap.add_argument(
        "--router_features",
        default="vocab",
        choices=["vocab", "candidate"],
        help="risk-gate features: 'vocab' needs no candidate answers (deployable); 'candidate' adds "
        "candidate-set logits (comparison only; always reported as safe_layer_gated_candidate)",
    )
    # ---- safe-layer steering
    ap.add_argument(
        "--layer_alphas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 4.0, 8.0],
        help="multiples of the mean (clean - corrupt) residual difference",
    )
    ap.add_argument(
        "--safety_lambda",
        type=float,
        default=2.0,
        help="harm aversion in S = benefit - lambda*harm",
    )
    ap.add_argument(
        "--harm_eps",
        type=float,
        default=0.10,
        help="max one-sided 95%% upper bound of the induction rate (correct -> wrong) for a layer to count as safe",
    )
    ap.add_argument(
        "--layer_criterion",
        default="worst",
        choices=["worst", "mean"],
        help="cross-task aggregation of S when selecting a layer",
    )
    ap.add_argument("--random_gate_reps", type=int, default=2000)
    ap.add_argument(
        "--skip_components",
        action="store_true",
        help="only run the safe-layer analysis",
    )
    # ---- component-level analysis
    ap.add_argument("--top_k", type=int, default=20, help="primary k")
    ap.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 15, 20, 30, 50])
    ap.add_argument(
        "--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    )
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--num_random_controls", type=int, default=20)
    ap.add_argument(
        "--layer_top_m",
        type=int,
        default=4,
        help="heads per layer for per-layer transfer",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--gen_eval", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gen_max_samples", type=int, default=40)
    ap.add_argument("--gen_max_new_tokens", type=int, default=24)
    ap.add_argument(
        "--gen_splits",
        nargs="+",
        default=["unfiltered"],
        choices=["unfiltered", "test"],
        help="evaluation sets for free generation",
    )
    ap.add_argument(
        "--audit_per_cell",
        type=int,
        default=5,
        help="audit-sheet rows per (task, method, auto label)",
    )
    ap.add_argument(
        "--audit_labels",
        default=None,
        help="filled-in generation_audit_sheet.csv -> agreement report",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="./causal_results")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hf_repo_id", default=None)
    ap.add_argument("--hf_path_in_repo", default="")
    ap.add_argument("--hf_private", action="store_true")
    ap.add_argument(
        "--aggregate_dirs", nargs="+", default=None, help="only aggregate finished runs"
    )
    ap.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    args = ap.parse_args()

    if args.aggregate_dirs:
        aggregate(args.aggregate_dirs, args.output_dir)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = args.output_dir
    st = os.path.join(out, "state")
    plots = os.path.join(out, "plots")
    for d in (out, st, plots):
        os.makedirs(d, exist_ok=True)
    vfile = os.path.join(st, "STATE_VERSION")
    if (
        args.resume
        and os.path.exists(vfile)
        and open(vfile).read().strip() != STATE_VERSION
    ):
        raise SystemExit(
            f"{st} was written by an older version of this code; use a new --output_dir or --no-resume"
        )
    open(vfile, "w").write(STATE_VERSION)
    # settings that change checkpointed computations; resuming with different ones would mix results
    sig_keys = [
        "model_name",
        "dtype",
        "data_source",
        "tasks",
        "n_samples",
        "max_pairs_per_task",
        "n_distractors",
        "split_fracs",
        "min_denom",
        "min_margin",
        "unfiltered_frac",
        "router_features",
        "seed",
        "layer_alphas",
        "k_values",
        "top_k",
        "lambdas",
        "alphas",
        "num_random_controls",
        "layer_top_m",
    ]
    sig = {k: getattr(args, k) for k in sig_keys}
    sfile = os.path.join(st, "checkpoint_settings.json")
    if args.resume and os.path.exists(sfile):
        old = json.load(open(sfile))
        diff = {k: (old.get(k), sig[k]) for k in sig_keys if old.get(k) != sig[k]}
        if diff:
            raise SystemExit(
                f"--resume with settings that differ from the checkpoints: {diff}. "
                "Use a new --output_dir or --no-resume."
            )
    json.dump(sig, open(sfile, "w"), indent=2)
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    engine = CausalEngine(
        args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    space = engine.space
    k_values = sorted({k for k in args.k_values + [args.top_k] if 0 < k <= space.size})
    primary_k = args.top_k if args.top_k in k_values else k_values[-1]

    # ------------------------------------------------------------------ pairs
    pairs: Dict[str, List[Dict]] = {}
    refs: Dict[str, Optional[Dict]] = {}
    build_rows = []
    for ti, task in enumerate(args.tasks):

        def _build(task=task, ti=ti):
            rng = random.Random(args.seed + 1000 * ti)
            stats: Dict = {}
            rows = load_raw(TASKS[task], args.data_source, args.data_dir)
            if PAIR_TYPE[task] == "hallucination":
                ps = build_mcq_pairs(engine, task, rows, args.n_samples, rng, stats)
            else:
                ps = build_retrieval_pairs(
                    engine, task, rows, args.n_samples, args.n_distractors, rng, stats
                )
            ref = (
                reference_corpus(rows) if PAIR_TYPE[task] == "counterfactual" else None
            )
            return {"pairs": ps, "stats": stats, "n_raw": len(rows), "ref": ref}

        res = ckpt(os.path.join(st, f"pairs_{task}.pt"), args.resume, _build)
        pairs[task] = res["pairs"]
        refs[task] = res.get("ref")
        build_rows.append(
            {
                "task": task,
                "raw_rows": res["n_raw"],
                "pairs_built": len(res["pairs"]),
                **{f"rejected_{k}": v for k, v in res["stats"].items()},
            }
        )
        log(
            f"[pairs] {task}: built {len(res['pairs'])} candidate pairs; rejections {res['stats']}"
        )
    pd.DataFrame(build_rows).to_csv(
        os.path.join(out, "pair_construction.csv"), index=False
    )

    # ---------------------------------------------- verify cached intervention path
    first = next((ps[0] for ps in pairs.values() if ps), None)
    if first is not None:
        ver = engine.verify_cached_equivalence(
            first["corrupt_ids"], first["clean_ids"], first["cand_ids"]
        )
        json.dump(
            ver,
            open(os.path.join(out, "verify_cached_equivalence.json"), "w"),
            indent=2,
        )

    # ------------------------------------------------- base pass + validation
    base: Dict[str, List[Dict]] = {}
    val_rows, splits, usable, funnel = [], {}, [], []
    passes: Dict[str, Dict[str, bool]] = {}
    split_rng = np.random.default_rng(args.seed)
    for task in args.tasks:
        b = ckpt(
            os.path.join(st, f"base_{task}.pt"),
            args.resume,
            lambda task=task: ER.base_pass(engine, pairs[task], f"base[{task}]"),
        )
        for p, bb in zip(pairs[task], b):
            p["halluc_idx"] = bb["halluc_idx"]
            p["gold_corrupt"] = bb["gold_corrupt"]
        infos = [
            validate(p, bb, args.min_denom, args.min_margin)
            for p, bb in zip(pairs[task], b)
        ]
        # Unfiltered held-out set: a random share of ALL built pairs, removed BEFORE
        # validate() is applied, so mitigation results on it are not conditioned
        # on the model answering the clean prompt correctly.
        order = np.random.default_rng(args.seed + 99).permutation(len(infos))
        n_unf = int(round(args.unfiltered_frac * len(infos)))
        unf_idx = sorted(order[:n_unf].tolist())
        unf_set = set(unf_idx)
        for i, inf in enumerate(infos):
            inf["heldout_unfiltered"] = i in unf_set
        val_rows += infos
        valid_idx = [
            i for i, inf in enumerate(infos) if inf["valid"] and i not in unf_set
        ][: args.max_pairs_per_task]
        strata = [infos[i]["hallucinated"] for i in valid_idx]
        sp = stratified_split(valid_idx, strata, args.split_fracs, split_rng)
        sp["unfiltered"] = unf_idx
        splits[task] = sp
        passes[task] = {
            pairs[task][i]["sample_id"]: bool(infos[i]["valid"]) for i in unf_idx
        }
        base[task] = b
        ok = all(
            len(sp[s]) >= args.min_split_size
            for s in ("discovery", "validation", "test")
        )
        if ok:
            usable.append(task)
        pool = [i for i in range(len(infos)) if i not in unf_set]

        def acc(idx, kind):
            return (
                float(np.mean([b[i][f"correct_{kind}"] for i in idx]))
                if idx
                else np.nan
            )

        funnel.append(
            {
                "task": task,
                "pair_type": PAIR_TYPE[task],
                "pairs_built": len(infos),
                "unfiltered_heldout": len(unf_idx),
                "pool": len(pool),
                "pool_valid": int(sum(infos[i]["valid"] for i in pool)),
                "pool_valid_fraction": (
                    float(np.mean([infos[i]["valid"] for i in pool]))
                    if pool
                    else np.nan
                ),
                "used_valid_pairs": len(valid_idx),
                "usable": ok,
                "unfiltered_valid_fraction": (
                    float(np.mean([infos[i]["valid"] for i in unf_idx]))
                    if unf_idx
                    else np.nan
                ),
                "acc_clean_unfiltered": acc(unf_idx, "clean"),
                "acc_corrupt_unfiltered": acc(unf_idx, "corrupt"),
                "acc_clean_conditioned_test": acc(sp["test"], "clean"),
                "acc_corrupt_conditioned_test": acc(sp["test"], "corrupt"),
                **{
                    f"rejected_{k}": v
                    for k, v in pd.Series([infos[i]["reason"] for i in pool])
                    .value_counts()
                    .items()
                    if k != "ok"
                },
            }
        )
        log(
            f"[validate] {task}: {len(valid_idx)}/{len(pool)} pool pairs valid "
            f"(hallucinated {sum(strata)}), split sizes { {k: len(v) for k, v in sp.items()} }"
            + (
                ""
                if ok
                else "  -> EXCLUDED from the conditioned analysis (still evaluated on the unfiltered set)"
            )
        )
    pd.DataFrame(val_rows).to_csv(os.path.join(out, "pair_validation.csv"), index=False)
    pd.DataFrame(funnel).to_csv(os.path.join(out, "selection_funnel.csv"), index=False)
    json.dump(
        {
            t: {k: [pairs[t][i]["sample_id"] for i in v] for k, v in s.items()}
            for t, s in splits.items()
        },
        open(os.path.join(out, "splits.json"), "w"),
        indent=2,
    )
    if not usable:
        raise SystemExit("No task has enough valid pairs; see pair_validation.csv")
    pair_types = {t: PAIR_TYPE[t] for t in args.tasks}
    hallu_tasks = [t for t in usable if pair_types[t] == "hallucination"]
    unf_tasks = [t for t in args.tasks if splits[t]["unfiltered"]]

    def sub(task, split):
        return [pairs[task][i] for i in splits[task][split]]

    def subbase(task, split):
        return [base[task][i] for i in splits[task][split]]

    # =================================================================
    # PART A — safe-layer steering (layer-level mitigation)
    # =================================================================
    sl_dir = os.path.join(out, "safe_layer")
    os.makedirs(sl_dir, exist_ok=True)
    sl_plots = os.path.join(plots, "safe_layer")
    layer_sel = pd.DataFrame()
    sl_routers: Dict = {}
    sl_vecs: Dict = {}
    sl_feats: Dict = {}
    if not hallu_tasks:
        log(
            "[safe-layer] no usable hallucination task -> skipping layer-level mitigation"
        )
    else:
        log(
            f"[safe-layer] hallucination tasks {hallu_tasks}; guard tasks {[t for t in usable if t not in hallu_tasks]}"
        )
        sl_vecs, alias, vsrc, vstats = SL.build_layer_vectors(
            {t: subbase(t, "discovery") for t in usable},
            pair_types,
            usable,
            args.seed,
            targets=unf_tasks,
        )
        vstats.to_csv(os.path.join(sl_dir, "layer_vector_stats.csv"), index=False)
        canonical = sorted({a for a in alias.values() if not a.startswith("RANDOM:")})
        randoms = sorted({a for a in alias.values() if a.startswith("RANDOM:")})
        log(
            f"[safe-layer] vectors {canonical} (+{len(randoms)} random-direction controls); aliases "
            f"{ {k: v for k, v in alias.items() if k != v} }"
        )
        layers = list(range(engine.num_layers))
        runs_list, eff_list = [], []
        jobs = [(t, s) for t in usable for s in ("validation", "test")] + [
            (t, "unfiltered") for t in unf_tasks
        ]
        for t, split in jobs:
            names = canonical + (randoms if split != "validation" else [])
            path_r = os.path.join(st, f"sl_runs_{t}_{split}.csv.gz")
            path_e = os.path.join(
                st,
                (
                    f"sl_eff_{t}_{split}.parquet"
                    if _has_parquet()
                    else f"sl_eff_{t}_{split}.csv.gz"
                ),
            )
            if args.resume and os.path.exists(path_r) and os.path.exists(path_e):
                log(f"[resume] safe-layer sweep {t}/{split}")
                r = pd.read_csv(path_r)
                e = (
                    pd.read_parquet(path_e)
                    if path_e.endswith(".parquet")
                    else pd.read_csv(path_e)
                )
                e["alpha"] = e.alpha.astype(np.float32)
            else:
                log(
                    f"[safe-layer] sweep {t}/{split}: {len(splits[t][split])} pairs x {len(names)} vectors x "
                    f"{len(layers)} layers x {len(args.layer_alphas)} alphas"
                )
                r, e = SL.layer_sweep(
                    engine,
                    t,
                    split,
                    sub(t, split),
                    sl_vecs,
                    names,
                    layers,
                    args.layer_alphas,
                    args.batch_size,
                    passes=passes[t] if split == "unfiltered" else None,
                )
                r.to_csv(path_r, index=False)
                (
                    e.to_parquet(path_e, index=False)
                    if path_e.endswith(".parquet")
                    else e.to_csv(path_e, index=False)
                )
            runs_list.append(r)
            eff_list.append(e)
        runs = pd.concat(runs_list, ignore_index=True)
        effects = pd.concat(eff_list, ignore_index=True)
        effects["vec"] = effects.vec.astype(str)
        # expand aliases so every vector name (e.g. LOTO:FCT == NOTA) has rows
        extra = []
        for name, canon in alias.items():
            if name != canon:
                e2 = effects[effects.vec == canon].copy()
                e2["vec"] = name
                extra.append(e2)
        if extra:
            effects = pd.concat([effects] + extra, ignore_index=True)

        tab = SL.safety_table(runs, effects, args.safety_lambda)
        tab.to_csv(os.path.join(sl_dir, "layer_safety_table.csv.gz"), index=False)

        # cross-task safe-layer ranking per vector (validation = used for selection; test = reported)
        rk, prof, agr = [], [], []
        for v in canonical + ["POOLED"]:
            if v not in alias or (v != alias[v] and v != "POOLED"):
                continue
            for split, tasks_ in (
                ("validation", usable),
                ("test", usable),
                ("unfiltered", unf_tasks),
            ):
                a, b_, c = SL.cross_task_ranking(
                    tab, v, tasks_, pair_types, args.harm_eps, split
                )
                rk.append(a)
                prof.append(b_)
                agr.append(c)
        ranking = pd.concat(rk, ignore_index=True) if rk else pd.DataFrame()
        ranking = (
            ranking.drop_duplicates(subset=["vec", "split", "layer"])
            if not ranking.empty
            else ranking
        )
        ranking.to_csv(
            os.path.join(sl_dir, "cross_task_safe_layer_ranking.csv"), index=False
        )
        pd.concat(prof, ignore_index=True).drop_duplicates().to_csv(
            os.path.join(sl_dir, "safe_layer_profile_per_task.csv"), index=False
        )
        pd.concat(agr, ignore_index=True).drop_duplicates().to_csv(
            os.path.join(sl_dir, "layer_rank_agreement.csv"), index=False
        )

        # selection (validation) + risk gate + test outcomes
        eval_targets = [t for t in args.tasks if t in usable or t in unf_tasks]
        protocols = SL.build_protocols(usable, eval_targets, pair_types, alias)
        for t in usable:
            for split in ("discovery", "validation", "test"):
                sl_feats.update(
                    SL.run_features(sub(t, split), subbase(t, split), t, split)
                )
        for t in unf_tasks:
            sl_feats.update(
                SL.run_features(
                    sub(t, "unfiltered"), subbase(t, "unfiltered"), t, "unfiltered"
                )
            )
        layer_sel, sl_out, sl_ctl, sl_routers = SL.evaluate_protocols(
            protocols,
            tab,
            runs,
            effects,
            sl_feats,
            pair_types,
            engine.num_layers,
            args.safety_lambda,
            args.harm_eps,
            args.layer_criterion,
            args.random_gate_reps,
            args.seed,
            alias=alias,
            eval_sets=("test", "unfiltered"),
            primary_mode=args.router_features,
        )
        layer_sel.to_csv(os.path.join(sl_dir, "safe_layer_selection.csv"), index=False)
        sl_out.to_csv(
            os.path.join(sl_dir, "layer_steering_eval_runs.csv.gz"), index=False
        )
        sl_ctl.to_csv(os.path.join(sl_dir, "risk_gate_controls.csv"), index=False)
        ba = SL.before_after_from_runs(
            sl_out[sl_out.method != "none"],
            ["protocol", "target_task", "pair_type", "method"],
            args.bootstrap_samples,
            args.seed,
        )
        ba.insert(0, "family", "layer_steer")
        ba.to_csv(
            os.path.join(sl_dir, "hallucination_before_after_layer.csv"), index=False
        )
        SL.make_safe_layer_plots(
            sl_plots,
            tab,
            ranking,
            layer_sel,
            sl_out,
            sl_ctl,
            args.harm_eps,
            args.safety_lambda,
        )

        # console summary
        print("\n==== Safe-layer selection (validation) ====")
        cols = [
            c
            for c in [
                "protocol",
                "vec",
                "selection_tasks",
                "selected",
                "layer",
                "alpha",
                "val_worst_S",
                "val_max_induction_ucb",
                "val_min_n_correct",
                "zero_error_ucb_floor",
                "val_induced",
                "val_rescued",
                "tau",
                "val_gate_rate",
                "abstain_reason",
            ]
            if c in layer_sel.columns
        ]
        print(
            layer_sel.drop_duplicates(subset=["protocol", "vec", "selection_tasks"])[
                cols
            ]
            .round(3)
            .to_string(index=False)
        )
        if (
            "eps_achievable" in layer_sel.columns
            and (~layer_sel.eps_achievable.fillna(True).astype(bool)).any()
        ):
            log(
                f"[safe-layer] WARNING: for some selection sets even zero induced errors cannot certify "
                f"induction <= {args.harm_eps} with this many validation runs; raise --max_pairs_per_task / "
                f"--n_samples or relax --harm_eps (see zero_error_ucb_floor in safe_layer_selection.csv)"
            )
        for subset, title in (
            ("conditioned", "filtered test split"),
            ("all", "UNFILTERED held-out set"),
        ):
            show = ba[
                (ba.event == "answer_error")
                & (ba.scope == "all")
                & (ba.subset == subset)
                & ba.method.isin(["safe_layer_always", "safe_layer_gated"])
            ]
            if not show.empty:
                print(
                    f"\n==== Answer error before -> after ({title}, clean+corrupt runs) ===="
                )
                print(
                    show[
                        [
                            "protocol",
                            "target_task",
                            "method",
                            "rate_before",
                            "rate_after",
                            "delta",
                            "fixed",
                            "newly_bad",
                            "p_mcnemar",
                            "steer_rate",
                        ]
                    ]
                    .round(3)
                    .to_string(index=False)
                )

    # =================================================================
    # PART B — component-level discovery, harm-aware circuits, causal transfer
    # =================================================================
    sources: Dict[str, Dict] = {}
    df = pd.DataFrame()
    summary = pd.DataFrame()
    if not args.skip_components:
        calib_rows, router_rows, disc_rows = [], [], []
        for task in usable:
            steerable = pair_types[task] == "hallucination"
            log(
                f"[discovery] {task} (steering {'on' if steerable else 'off: counterfactual task'})"
            )
            vectors = ckpt(
                os.path.join(st, f"vectors_{task}.pt"),
                args.resume,
                lambda task=task: ER.compute_steering_vectors(
                    engine, sub(task, "discovery")
                ),
            )
            disc = ckpt(
                os.path.join(st, f"discovery_{task}.pt"),
                args.resume,
                lambda task=task, vectors=vectors, steerable=steerable: ER.discovery_sweep(
                    engine,
                    sub(task, "discovery"),
                    vectors,
                    args.batch_size,
                    with_steer=steerable,
                ),
            )
            corr_ok = np.array(
                [bb["correct_corrupt"] for bb in subbase(task, "discovery")]
            )
            scores = ER.component_scores(disc, corr_ok)
            np.save(os.path.join(st, f"aie_{task}.npy"), scores["aie"])
            rank = np.argsort(-np.abs(scores["aie"]))
            for r, c in enumerate(rank):
                disc_rows.append(
                    {
                        "task": task,
                        "rank": r + 1,
                        "component": space.label(c),
                        "layer": int(space.layer_of(c)),
                        "type": "mlp" if space.is_mlp(c) else "head",
                        "aie": float(scores["aie"][c]),
                        "R_patch": float(scores["R_patch"][c]),
                        "H_patch": float(scores["H_patch"][c]),
                        "R_steer": float(scores["R_steer"][c]),
                        "H_steer": float(scores["H_steer"][c]),
                    }
                )
            log(f"[discovery] {task} top-10 |AIE|: {space.labels(rank[:10])}")

            sel = ckpt(
                os.path.join(st, f"selection_{task}.pt"),
                args.resume,
                lambda task=task, scores=scores, vectors=vectors, steerable=steerable: ER.calibrate_and_select(
                    engine,
                    sub(task, "validation"),
                    scores,
                    vectors,
                    k_values,
                    args.lambdas,
                    args.alphas,
                    args.num_random_controls,
                    args.seed + 7,
                    args.batch_size,
                    steer=steerable,
                ),
            )
            for r in sel["calibration"]:
                calib_rows.append({"task": task, **r})
            for k, ent in sel["circuits"].items():
                for rg in ("patch", "steer"):
                    if ent.get(rg) is None:
                        continue
                    calib_rows.append(
                        {
                            "task": task,
                            "k": k,
                            "regime": rg,
                            "method": "chosen",
                            "lambda": ent[rg]["harm_aware_lambda"],
                            "alpha": (
                                ent["steer"]["alpha"]["harm_aware"]
                                if rg == "steer"
                                else np.nan
                            ),
                            "val_objective": np.nan,
                            "harm_aware_components": " ".join(
                                space.labels(ent[rg]["harm_aware"])
                            ),
                            "topk_aie_components": " ".join(
                                space.labels(ent[rg]["topk_aie"])
                            ),
                            "layer_matched_exact_fraction": ent[
                                "layer_matched_exact_fraction"
                            ],
                        }
                    )
            routers = {}
            if steerable:
                routers, rrows = ckpt(
                    os.path.join(st, f"router_{task}.pt"),
                    args.resume,
                    lambda task=task, sel=sel, vectors=vectors: ER.fit_router(
                        engine,
                        subbase(task, "discovery"),
                        sub(task, "validation"),
                        subbase(task, "validation"),
                        sel,
                        vectors,
                        args.batch_size,
                        mode=args.router_features,
                    ),
                )
                router_rows += [{"task": task, **r} for r in rrows]
            sources[task] = {
                "sel": sel,
                "vectors": vectors,
                "routers": routers,
                "scores": scores,
                "steerable": steerable,
            }
        pd.DataFrame(disc_rows).to_csv(
            os.path.join(out, "component_ranking.csv"), index=False
        )
        pd.DataFrame(calib_rows).to_csv(
            os.path.join(out, "calibration.csv"), index=False
        )
        pd.DataFrame(router_rows).to_csv(
            os.path.join(out, "router_calibration.csv"), index=False
        )

        frames, lframes = [], []
        for t in usable:
            path_m = os.path.join(st, f"eval_{t}.csv.gz")
            path_l = os.path.join(st, f"layers_{t}.csv.gz")
            if args.resume and os.path.exists(path_m) and os.path.exists(path_l):
                log(f"[resume] eval {t}")
                dm, dl = pd.read_csv(path_m), pd.read_csv(path_l)
            else:
                log(
                    f"[eval] target {t} ({len(splits[t]['test'])} test pairs, {len(sources)} sources)"
                )
                dm, dl = ER.evaluate_target(
                    engine,
                    t,
                    sub(t, "test"),
                    sources,
                    args.layer_top_m,
                    args.batch_size,
                    test_base=subbase(t, "test"),
                )
                dm.to_csv(path_m, index=False)
                dl.to_csv(path_l, index=False)
            frames.append(dm)
            lframes.append(dl)
        df = pd.concat(frames, ignore_index=True)
        dl = pd.concat(lframes, ignore_index=True)

        summary, tests, ctl = ER.summarize(
            df, args.bootstrap_samples, args.permutations, args.seed
        )
        rr = df[df.method == "harm_routing"]
        rcal = []
        for (s, t, k), g in rr.groupby(["source_task", "target_task", "k"]):
            y = (~g.correct_base.astype(bool)).astype(float).values
            p = g.risk.values
            bins = np.clip((p * 10).astype(int), 0, 9)
            ece = sum(
                abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                for b in np.unique(bins)
            )
            rcal.append(
                {
                    "source_task": s,
                    "target_task": t,
                    "k": k,
                    "brier": float(np.mean((p - y) ** 2)),
                    "ece": float(ece),
                    "auroc": SL.auroc(p, y.astype(bool)),
                    "route_rate": float(g.routed.mean()),
                }
            )
        pd.DataFrame(rcal).to_csv(
            os.path.join(out, "router_test_calibration.csv"), index=False
        )
        summary.to_csv(os.path.join(out, "summary.csv"), index=False)
        tests.to_csv(os.path.join(out, "paired_tests.csv"), index=False)
        ctl.to_csv(os.path.join(out, "random_control_tests.csv"), index=False)
        lsum = ER.summarize_layers(dl, {t: sources[t]["scores"] for t in usable}, space)
        lsum.to_csv(os.path.join(out, "layer_transfer.csv"), index=False)

        valid_base = {
            t: [
                base[t][i]
                for s in ("discovery", "validation", "test")
                for i in splits[t][s]
            ]
            for t in usable
        }
        cka, dirs = ER.representation_analysis(valid_base, engine.num_layers, args.seed)
        cka.to_csv(os.path.join(out, "paired_cka_clean_vs_corrupt.csv"), index=False)
        dirs.to_csv(os.path.join(out, "direction_alignment.csv"), index=False)
        ER.make_plots(
            plots,
            summary,
            lsum,
            cka,
            dirs,
            primary_k,
            {t: sources[t]["scores"] for t in usable},
            space,
        )

        # component steering on the unfiltered held-out set (all tasks with one)
        uframes = []
        for t in unf_tasks:
            if not any(S.get("steerable") for S in sources.values()):
                break
            pu = os.path.join(st, f"eval_unfiltered_{t}.csv.gz")
            if args.resume and os.path.exists(pu):
                du = pd.read_csv(pu)
            else:
                du = ER.steering_outcomes(
                    engine,
                    t,
                    sub(t, "unfiltered"),
                    subbase(t, "unfiltered"),
                    sources,
                    args.batch_size,
                )
                du.to_csv(pu, index=False)
            if not du.empty:
                du["eval_set"] = "unfiltered"
                du["passes_filter"] = (
                    du.sample_id.astype(str)
                    .map({str(k): v for k, v in passes[t].items()})
                    .fillna(False)
                )
                uframes.append(du)
        dtest = df.copy()
        dtest["eval_set"], dtest["passes_filter"] = "test", True
        comp_runs = SL.component_runs_from_eval(
            pd.concat([dtest] + uframes, ignore_index=True)
        )
        if not comp_runs.empty:
            bac = SL.before_after_from_runs(
                comp_runs,
                ["protocol", "target_task", "pair_type", "method"],
                args.bootstrap_samples,
                args.seed,
            )
            bac.insert(0, "family", "component")
            bac.to_csv(
                os.path.join(out, "hallucination_before_after_component.csv"),
                index=False,
            )

        cols = [
            "source_task",
            "target_task",
            "method",
            "mean_norm_gain",
            "rescue_rate",
            "induction_rate_corrupt",
        ]
        for rg in ("patch", "steer"):
            s = summary[(summary.regime == rg) & (summary.k == primary_k)]
            if not s.empty:
                print(f"\n==== component {rg} regime, k={primary_k} ====")
                print(
                    s[cols + (["induction_rate_clean"] if rg == "steer" else [])]
                    .round(3)
                    .to_string(index=False)
                )

    # =================================================================
    # PART C — free-generation before/after.  Labels are AUTOMATIC PROXIES for
    # constrained answers (letter / PMID / title vs reference); validate them
    # with the human-audit sheet before calling them hallucination rates.
    # =================================================================
    if args.gen_eval and (not layer_sel.empty or sources):
        gframes = []
        gen_targets = {"test": usable, "unfiltered": unf_tasks}
        for gsplit in args.gen_splits:
            for t in gen_targets.get(gsplit, []):
                configs = []
                if not layer_sel.empty:
                    for proto in ([f"source:{t}"] if t in hallu_tasks else []) + [
                        "loto",
                        "pooled",
                    ]:
                        row = layer_sel[
                            (layer_sel.protocol == proto) & (layer_sel.target == t)
                        ]
                        if row.empty or not bool(row.selected.iloc[0]):
                            continue
                        r0 = row.iloc[0]
                        key = f"{r0.vec}|{tuple(r0.selection_tasks.split('+'))}|{args.router_features}"
                        vec = (
                            sl_vecs[r0.vec]
                            if r0.vec in sl_vecs
                            else sl_vecs[alias[r0.vec]]
                        )
                        gate = (
                            ER.RiskRouter.from_state(sl_routers[key])
                            if key in sl_routers
                            else None
                        )
                        pname = "in_task" if proto.startswith("source:") else proto
                        configs.append(
                            {
                                "name": "safe_layer_always",
                                "protocol": f"layer:{pname}",
                                "kind": "layer",
                                "layer": int(r0.layer),
                                "alpha": float(r0.alpha),
                                "vec": vec,
                                "gate": None,
                            }
                        )
                        if gate is not None:
                            configs.append(
                                {
                                    "name": "safe_layer_gated",
                                    "protocol": f"layer:{pname}",
                                    "kind": "layer",
                                    "layer": int(r0.layer),
                                    "alpha": float(r0.alpha),
                                    "vec": vec,
                                    "gate": gate,
                                }
                            )
                S = sources.get(t)
                if (
                    S is not None
                    and S.get("steerable")
                    and (S["sel"]["circuits"].get(primary_k) or {}).get("steer")
                ):
                    ent = S["sel"]["circuits"][primary_k]["steer"]
                    base_cfg = {
                        "protocol": "component:in_task",
                        "kind": "component",
                        "circuit": ent["harm_aware"],
                        "alpha": ent["alpha"]["harm_aware"],
                        "steer_z": S["vectors"]["z"],
                        "steer_mlp": S["vectors"]["mlp"],
                    }
                    configs.append(
                        {**base_cfg, "name": f"harm_aware@k{primary_k}", "gate": None}
                    )
                    if primary_k in S["routers"]:
                        configs.append(
                            {
                                **base_cfg,
                                "name": f"harm_routing@k{primary_k}",
                                "gate": ER.RiskRouter.from_state(
                                    S["routers"][primary_k]
                                ),
                            }
                        )
                if not configs:
                    continue
                if not any(k.startswith(f"{t}|{gsplit}|") for k in sl_feats):
                    sl_feats.update(
                        SL.run_features(sub(t, gsplit), subbase(t, gsplit), t, gsplit)
                    )
                sig = json.dumps(
                    [gsplit]
                    + [
                        [
                            c["name"],
                            c["protocol"],
                            c.get("layer"),
                            c.get("alpha"),
                            None if c.get("gate") is None else c["gate"].tau,
                            (
                                None
                                if c.get("circuit") is None
                                else list(map(int, c["circuit"]))
                            ),
                        ]
                        for c in configs
                    ]
                    + [args.gen_max_samples, args.gen_max_new_tokens],
                    default=float,
                )
                gp = os.path.join(
                    st,
                    f"gen_{gsplit}_{t}_{hashlib.md5(sig.encode()).hexdigest()[:10]}.csv",
                )
                gframes.append(
                    ckpt_df(
                        gp,
                        args.resume,
                        lambda t=t, gsplit=gsplit, configs=configs: SL.generation_before_after(
                            engine,
                            t,
                            gsplit,
                            sub(t, gsplit)[: args.gen_max_samples],
                            sl_feats,
                            configs,
                            args.gen_max_new_tokens,
                            ref=refs.get(t),
                            passes=passes[t] if gsplit == "unfiltered" else None,
                        ),
                    )
                )
        if gframes:
            gen = pd.concat(gframes, ignore_index=True)
            gen.to_csv(os.path.join(out, "generation_runs.csv"), index=False)
            gtab = SL.generation_tables(
                gen, pair_types, args.bootstrap_samples, args.seed
            )
            gtab.to_csv(
                os.path.join(out, "hallucination_before_after_generation.csv"),
                index=False,
            )
            gen.groupby(["eval_set", "target_task", "method", "label"]).size().rename(
                "n"
            ).reset_index().to_csv(
                os.path.join(out, "generation_label_counts.csv"), index=False
            )
            show = gtab[
                (gtab.event == "incorrect_answer")
                & (gtab.scope == "all")
                & gtab.subset.isin(["conditioned", "all"])
            ]
            if not show.empty:
                print(
                    "\n==== Generation: incorrect-answer rate (automatic proxy label) before -> after ===="
                )
                print(
                    show[
                        [
                            "eval_set",
                            "protocol",
                            "target_task",
                            "method",
                            "rate_before",
                            "rate_after",
                            "delta",
                            "p_mcnemar",
                            "steer_rate",
                        ]
                    ]
                    .round(3)
                    .to_string(index=False)
                )
            sheet = ER.audit_sheet(gen, args.audit_per_cell, args.seed)
            sheet.to_csv(os.path.join(out, "generation_audit_sheet.csv"), index=False)
            log(
                f"[audit] wrote {len(sheet)} rows to generation_audit_sheet.csv: fill human_label "
                f"(correct / incorrect / abstain / unclear), then rerun with --audit_labels <file>"
            )
        if args.audit_labels:
            agr, conf = ER.audit_agreement(
                pd.read_csv(args.audit_labels, keep_default_na=False)
            )
            agr.to_csv(os.path.join(out, "audit_agreement.csv"), index=False)
            conf.to_csv(os.path.join(out, "audit_confusion.csv"), index=False)
            if not agr.empty:
                print("\n==== Automatic vs human labels ====")
                print(agr.round(3).to_string(index=False))

    # ----------------------------------------------- combined before/after table
    parts = [
        os.path.join(sl_dir, "hallucination_before_after_layer.csv"),
        os.path.join(out, "hallucination_before_after_component.csv"),
    ]
    frames = [x for x in (read_csv_safe(p) for p in parts) if not x.empty]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(out, "hallucination_before_after.csv"), index=False
        )

    if args.hf_repo_id:
        from huggingface_hub import HfApi, create_repo

        path_in_repo = (
            args.hf_path_in_repo.strip("/")
            or f"{args.model_name.replace('/', '_')}/seed{args.seed}"
        )
        create_repo(
            args.hf_repo_id, repo_type="dataset", exist_ok=True, private=args.hf_private
        )
        HfApi().upload_folder(
            folder_path=out,
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            ignore_patterns=["state/*", "*.tmp*"],
            commit_message=f"Med-HALT causal transfer: {path_in_repo}",
        )
        log(f"[HF] uploaded to {args.hf_repo_id}/{path_in_repo}")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
