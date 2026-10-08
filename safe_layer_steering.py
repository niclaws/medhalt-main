"""
safe_layer_steering.py — layer-level hallucination mitigation.

Implements the four pieces the component-level analysis did not cover:

1. Safe-layer selection
   Steering = add alpha * v_l to the residual stream after decoder layer l at
   the decision position (and every generated position).  v_l is the mean
   (clean - corrupt) residual on the DISCOVERY split.  No clean reference is
   needed at inference time.
   For every (vector, target task, layer, alpha) on every run (clean + corrupt):
       rescued  = run was wrong before steering, correct after
       induced  = run was correct before steering, wrong after
       benefit  B = #rescued / #runs        harm H = #induced / #runs
       safety   S = B - lambda * H          (lambda = harm aversion, default 2)
       induction UCB = one-sided 95% Wilson upper bound of #induced / #correct
   A (layer, alpha) is SAFE if the induction UCB <= eps on every guarded task,
   and it is selected if it is safe and has the largest S (> 0).  If nothing
   is safe the method abstains (no steering).  Selection uses validation only.

2. Cross-task safe-layer ranking
   Per vector, layers are ranked by worst-case S over the hallucination tasks
   (benefit) subject to the induction constraint on ALL tasks, including the
   counterfactual retrieval tasks, which have no hallucination to fix and
   therefore act as off-target damage guards.  Also reported: mean-S rank,
   per-task ranks and the rank agreement (Spearman) between tasks.
   Protocols for choosing the layer used on a target t:
       source:<s>  vector + layer chosen on task s only; applied to every t
       pooled      vector pooled over hallucination tasks; layer chosen by the
                   worst case over all tasks (t included)
       loto        leave-one-task-out: vector from the other hallucination
                   tasks, layer chosen on the other tasks only, then applied
                   to the unseen task t (the strict transfer test)

3. Risk-gated safe-layer steering
   A logistic hallucination-risk detector (inference-time features: candidate
   margin / normalized entropy, vocabulary entropy, residual PCA at 3 depths)
   is trained on DISCOVERY runs of the protocol's selection tasks.  The
   threshold tau is chosen on VALIDATION runs to maximise S under the same
   induction constraint.  Steering is applied only when risk > tau.
   Controls on test: always-steer, oracle gate (steer only wrong runs, an
   upper bound), random gate with the same steering rate (Monte Carlo), and a
   random direction with the same per-layer norm at the selected layer.

4. Hallucination change before and after
   On the held-out test split, for every method: error rate and
   hallucinated-answer rate before and after, absolute and relative change,
   #rescued / #induced, exact McNemar p-value (paired), paired bootstrap CI
   (resampling questions), Holm correction.  The same table is produced from
   free generation (unsupported-claim rate) for the steered generations.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiment_runner import (
    INCORRECT_LABELS,
    UNVERIFIABLE_LABELS,
    RiskRouter,
    classify_generation,
    default_router_layers,
    holm,
    prepare_pair,
    router_features,
    run_gold,
)
from patching_engine import CausalEngine, Intervention, log

HALLU = "hallucination"


# =====================================================================
# Statistics helpers
# =====================================================================


def wilson_upper(k, n, z: float = 1.6449):
    """One-sided 95% Wilson upper bound for a binomial rate (vectorised)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.ones_like(k)
    m = n > 0
    p = np.where(m, k / np.where(m, n, 1), 0)
    nn = np.where(m, n, 1)
    denom = 1 + z * z / nn
    centre = p + z * z / (2 * nn)
    adj = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn))
    out = np.where(m, np.minimum(1.0, (centre + adj) / denom), 1.0)
    return out if out.ndim else float(out)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test; b, c = discordant counts."""
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2 * tail))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().values
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(a[m]).rank(), pd.Series(b[m]).rank())[0, 1])


# =====================================================================
# 0. Layer steering vectors
# =====================================================================


def build_layer_vectors(
    base_disc: Dict[str, List[Dict]],
    pair_types: Dict[str, str],
    usable: Sequence[str],
    seed: int,
    targets: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], Dict[str, List[str]], pd.DataFrame]:
    """Returns (vectors, alias, sources, stats).

    vectors[name] : [L, D] fp32.  Names: one per hallucination task, POOLED
                    (mean of task vectors, equal weights), LOTO:<t> for each
                    usable task (mean over hallucination tasks != t), and
                    RANDOM:<name> controls (random direction, same per-layer norm).
    alias[name]   : canonical vector name (identical vectors are swept once).
    sources[name] : hallucination tasks the vector was estimated from."""
    vec: Dict[str, torch.Tensor] = {}
    srcs: Dict[str, List[str]] = {}
    hallu = [t for t in usable if pair_types[t] == HALLU and base_disc.get(t)]
    for t in hallu:
        vec[t] = torch.stack(
            [
                b["resid_clean"].float() - b["resid_corrupt"].float()
                for b in base_disc[t]
            ]
        ).mean(0)
        srcs[t] = [t]
    combos: Dict[Tuple[str, ...], str] = {(t,): t for t in hallu}
    alias = {t: t for t in hallu}

    def add_combo(name, members):
        members = tuple(sorted(members))
        if not members:
            return
        if members in combos:
            alias[name] = combos[members]
            return
        vec[name] = torch.stack([vec[m] for m in members]).mean(0)
        srcs[name] = list(members)
        combos[members] = name
        alias[name] = name

    add_combo("POOLED", hallu)
    for t in list(dict.fromkeys(list(usable) + list(targets or []))):
        add_combo(f"LOTO:{t}", [s for s in hallu if s != t])
    for name, members in list(srcs.items()):
        srcs.setdefault(name, members)
    for name in list(alias):
        srcs.setdefault(name, srcs[alias[name]])

    # random-direction controls, norm-matched per layer, for every canonical vector
    g = torch.Generator().manual_seed(seed + 12345)
    for name in sorted(set(alias.values())):
        v = vec[name]
        r = torch.randn(v.shape, generator=g)
        r = r / r.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
        vec[f"RANDOM:{name}"] = r
        alias[f"RANDOM:{name}"] = f"RANDOM:{name}"
        srcs[f"RANDOM:{name}"] = srcs[name]

    # stats: vector norm relative to the typical residual norm at that layer
    rows = []
    allres = [b["resid_corrupt"].float() for t in hallu for b in base_disc[t]]
    rnorm = torch.stack(allres).norm(dim=-1).mean(0) if allres else None
    for name in sorted(set(alias.values())):
        if name.startswith("RANDOM:"):
            continue
        for l in range(vec[name].shape[0]):
            n = float(vec[name][l].norm())
            rows.append(
                {
                    "vector": name,
                    "sources": "+".join(srcs[name]),
                    "layer": l,
                    "vector_norm": n,
                    "mean_resid_norm": float(rnorm[l]) if rnorm is not None else np.nan,
                    "relative_norm": (
                        n / float(rnorm[l])
                        if rnorm is not None and float(rnorm[l]) > 0
                        else np.nan
                    ),
                }
            )
    names = sorted(
        set(alias.values()) - {n for n in alias.values() if n.startswith("RANDOM:")}
    )
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for l in range(vec[a].shape[0]):
                cos = torch.nn.functional.cosine_similarity(
                    vec[a][l], vec[b][l], dim=0
                ).item()
                rows.append(
                    {
                        "vector": f"{a}~{b}",
                        "sources": "cosine",
                        "layer": l,
                        "vector_norm": cos,
                        "mean_resid_norm": np.nan,
                        "relative_norm": np.nan,
                    }
                )
    return vec, alias, srcs, pd.DataFrame(rows)


# =====================================================================
# 1a. Layer sweep (every vector x layer x alpha on every run)
# =====================================================================


def _gold_margin(logits: np.ndarray, gold: int) -> np.ndarray:
    logits = np.atleast_2d(logits)
    other = np.delete(logits, gold, axis=1).max(axis=1)
    return logits[:, gold] - other


def layer_sweep(
    engine: CausalEngine,
    task: str,
    split: str,
    pairs: List[Dict],
    vectors: Dict[str, torch.Tensor],
    vec_names: Sequence[str],
    layers: Sequence[int],
    alphas: Sequence[float],
    batch_size: int = 64,
    passes: Optional[Dict[str, bool]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (runs, effects).
    passes: sample_id -> did the pair pass validate() (unfiltered set only).
    runs    : one row per (pair, run_type) with the unsteered outcome.
    effects : one row per (run, vector, layer, alpha) with the steered outcome."""
    vec_names = list(vec_names)
    stack = torch.stack([vectors[n] for n in vec_names])  # [V, L, D]
    tags = [(vi, l, a) for vi in range(len(vec_names)) for l in layers for a in alphas]
    ivs = [
        Intervention(resid_layers=np.array([l]), resid_alpha=float(a), resid_src=vi)
        for vi, l, a in tags
    ]
    t_vec = np.array([vec_names[t[0]] for t in tags], dtype=object)
    t_layer = np.array([t[1] for t in tags], dtype=np.int16)
    t_alpha = np.array([t[2] for t in tags], dtype=np.float32)
    runs, eff = [], []
    for p in tqdm(pairs, desc=f"layer-sweep[{task}/{split}]", leave=False):
        sc, so = prepare_pair(engine, p)
        h = int(p["halluc_idx"])
        for rt, st in (("clean", so), ("corrupt", sc)):
            g = run_gold(p, rt)
            b = st.cand_logits.numpy()
            rid = f"{task}|{split}|{p['sample_id']}|{rt}"
            runs.append(
                {
                    "run_id": rid,
                    "task": task,
                    "split": split,
                    "sample_id": p["sample_id"],
                    "pair_type": p["pair_type"],
                    "run_type": rt,
                    "gold": g,
                    "halluc_idx": h,
                    "pred_base": int(b.argmax()),
                    "correct_base": bool(b.argmax() == g),
                    "halluc_pick_base": bool(b.argmax() == h),
                    "margin_base": float(_gold_margin(b, g)[0]),
                    "passes_filter": bool((passes or {}).get(p["sample_id"], True)),
                }
            )
            if not ivs:
                continue
            out = engine.run_interventions(
                st, ivs, steer_resid=stack, batch_size=batch_size
            )
            pred = out.argmax(1)
            eff.append(
                pd.DataFrame(
                    {
                        "run_id": rid,
                        "vec": t_vec,
                        "layer": t_layer,
                        "alpha": t_alpha,
                        "pred_after": pred.astype(np.int16),
                        "correct_after": pred == g,
                        "halluc_pick_after": pred == h,
                        "margin_after": _gold_margin(out, g).astype(np.float32),
                    }
                )
            )
    effects = pd.concat(eff, ignore_index=True) if eff else pd.DataFrame()
    return pd.DataFrame(runs), effects


# =====================================================================
# 1b. Safety table
# =====================================================================


def safety_table(runs: pd.DataFrame, effects: pd.DataFrame, lam: float) -> pd.DataFrame:
    """One row per (vec, task, split, layer, alpha)."""
    d = effects.merge(runs, on="run_id", how="inner")
    d["wrong0"] = ~d.correct_base
    d["rescued"] = d.wrong0 & d.correct_after
    d["induced"] = d.correct_base & ~d.correct_after
    d["err_after"] = ~d.correct_after
    d["dmargin"] = d.margin_after - d.margin_base
    d["dm_wrong"] = d.dmargin.where(d.wrong0)
    d["dm_right"] = d.dmargin.where(d.correct_base)
    hall = d.pair_type == HALLU
    # hallucinated-answer pick is only an error signal on hallucination tasks
    d["hp_b"] = d.halluc_pick_base.where(hall).astype(float)
    d["hp_a"] = d.halluc_pick_after.where(hall).astype(float)
    keys = ["vec", "task", "pair_type", "split", "layer", "alpha"]
    t = (
        d.groupby(keys, observed=True)
        .agg(
            n_runs=("run_id", "size"),
            n_wrong_base=("wrong0", "sum"),
            n_correct_base=("correct_base", "sum"),
            rescued=("rescued", "sum"),
            induced=("induced", "sum"),
            err_before=("wrong0", "mean"),
            err_after=("err_after", "mean"),
            halluc_pick_before=("hp_b", "mean"),
            halluc_pick_after=("hp_a", "mean"),
            mean_dmargin_wrong=("dm_wrong", "mean"),
            mean_dmargin_correct=("dm_right", "mean"),
        )
        .reset_index()
    )
    t["rescue_rate"] = t.rescued / t.n_wrong_base.replace(0, np.nan)
    t["induction_rate"] = t.induced / t.n_correct_base.replace(0, np.nan)
    t["induction_ucb"] = wilson_upper(t.induced.values, t.n_correct_base.values)
    t["benefit"] = t.rescued / t.n_runs
    t["harm"] = t.induced / t.n_runs
    t["S"] = t.benefit - lam * t.harm
    t["delta_err"] = t.err_after - t.err_before
    return t


# =====================================================================
# 2. Safe-layer selection and cross-task ranking
# =====================================================================


def _per_config(
    tab: pd.DataFrame,
    vec: str,
    tasks: Sequence[str],
    split: str,
    pair_types: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate over tasks for each (layer, alpha): worst/mean S over the
    hallucination tasks, max induction UCB over all tasks."""
    sub = tab[(tab.vec == vec) & (tab.task.isin(tasks)) & (tab.split == split)]
    if sub.empty:
        return pd.DataFrame()
    benefit_tasks = [t for t in tasks if pair_types[t] == HALLU]
    sb = sub[sub.task.isin(benefit_tasks)]
    g_all = sub.groupby(["layer", "alpha"]).agg(
        max_induction_ucb=("induction_ucb", "max"),
        max_induction_rate=("induction_rate", "max"),
        n_tasks=("task", "nunique"),
        min_n_correct=("n_correct_base", "min"),
        total_induced=("induced", "sum"),
        total_rescued=("rescued", "sum"),
        total_wrong=("n_wrong_base", "sum"),
    )
    if sb.empty:
        g_b = pd.DataFrame(
            index=g_all.index,
            data={
                "worst_S": np.nan,
                "mean_S": np.nan,
                "mean_rescue": np.nan,
                "n_benefit_tasks": 0,
            },
        )
    else:
        g_b = sb.groupby(["layer", "alpha"]).agg(
            worst_S=("S", "min"),
            mean_S=("S", "mean"),
            mean_rescue=("rescue_rate", "mean"),
            n_benefit_tasks=("task", "nunique"),
        )
    out = g_all.join(g_b, how="left").reset_index()
    out = out[out.n_tasks == len(set(tasks) & set(sub.task))]
    return out


def select_safe_layer(
    tab,
    vec,
    tasks,
    pair_types,
    eps: float,
    criterion: str = "worst",
    split: str = "validation",
) -> Dict:
    agg = _per_config(tab, vec, tasks, split, pair_types)
    key = "worst_S" if criterion == "worst" else "mean_S"
    res = {
        "vec": vec,
        "selection_tasks": "+".join(tasks),
        "criterion": criterion,
        "eps": eps,
        "selected": False,
        "layer": -1,
        "alpha": 0.0,
        "val_worst_S": np.nan,
        "val_mean_S": np.nan,
        "val_max_induction_ucb": np.nan,
        "n_safe_configs": 0,
        "n_configs": int(len(agg)),
        "abstain_reason": "no_validation_data",
    }
    if agg.empty:
        return res
    # Evidence check: with n correct validation runs on the smallest task, even
    # ZERO induced errors gives an upper bound of about z^2/(n+z^2).  If that
    # already exceeds eps, no layer can ever be certified safe.
    n_min = int(agg.min_n_correct.min())
    floor = float(wilson_upper(0, n_min))
    per_task = (
        tab[(tab.vec == vec) & tab.task.isin(tasks) & (tab.split == split)]
        .groupby("task")
        .n_correct_base.first()
    )
    res.update(
        {
            "val_min_n_correct": n_min,
            "zero_error_ucb_floor": floor,
            "val_n_correct_per_task": ";".join(
                f"{t}={int(n)}" for t, n in per_task.items()
            ),
            "eps_achievable": floor <= eps,
        }
    )
    agg["safe"] = agg.max_induction_ucb <= eps
    res["n_safe_configs"] = int(agg.safe.sum())
    cand = agg[agg.safe & (agg[key] > 0)]
    if cand.empty:
        res["abstain_reason"] = (
            "insufficient_evidence: zero-error bound %.3f > eps" % floor
            if floor > eps
            else (
                "no_safe_config"
                if not agg.safe.any()
                else "no_safe_config_with_positive_S"
            )
        )
        return res
    best = cand.sort_values(
        [key, "mean_S", "max_induction_ucb"], ascending=[False, False, True]
    ).iloc[0]
    res.update(
        {
            "selected": True,
            "abstain_reason": "",
            "layer": int(best.layer),
            "alpha": float(best.alpha),
            "val_worst_S": float(best.worst_S),
            "val_mean_S": float(best.mean_S),
            "val_max_induction_ucb": float(best.max_induction_ucb),
            "val_induced": int(best.total_induced),
            "val_rescued": int(best.total_rescued),
            "val_wrong_runs": int(best.total_wrong),
        }
    )
    return res


def cross_task_ranking(
    tab, vec, tasks, pair_types, eps, split
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ranking, per_task_profile, rank_agreement) for one vector."""
    agg = _per_config(tab, vec, tasks, split, pair_types)
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    agg["safe"] = agg.max_induction_ucb <= eps
    # best alpha per layer: safe first, then worst-case S
    agg = agg.sort_values(
        ["layer", "safe", "worst_S", "mean_S"], ascending=[True, False, False, False]
    )
    per_layer = agg.groupby("layer").head(1).copy()
    per_layer["rank_worst_S"] = per_layer.worst_S.rank(ascending=False, method="min")
    per_layer["rank_mean_S"] = per_layer.mean_S.rank(ascending=False, method="min")
    order = per_layer.sort_values(
        ["safe", "worst_S", "mean_S"], ascending=[False, False, False]
    ).index
    per_layer.loc[order, "safe_rank"] = np.arange(1, len(order) + 1)
    sub = tab[(tab.vec == vec) & (tab.task.isin(tasks)) & (tab.split == split)]
    prof = sub.merge(per_layer[["layer", "alpha"]], on=["layer", "alpha"], how="inner")
    prof = prof.sort_values(["task", "layer"])
    prof["rank_in_task"] = prof.groupby("task").S.rank(ascending=False, method="min")
    wide = prof.pivot_table(index="layer", columns="task", values="S")
    for c in wide.columns:
        per_layer[f"S_{c}"] = per_layer.layer.map(wide[c])
    per_layer.insert(0, "vec", vec)
    per_layer.insert(1, "split", split)
    agree = []
    ts = list(wide.columns)
    for i, a in enumerate(ts):
        for b in ts[i + 1 :]:
            agree.append(
                {
                    "vec": vec,
                    "split": split,
                    "task_a": a,
                    "task_b": b,
                    "spearman_layer_S": _spearman(wide[a].values, wide[b].values),
                }
            )
    return per_layer.sort_values("safe_rank"), prof, pd.DataFrame(agree)


def build_protocols(usable, eval_targets, pair_types, alias) -> List[Dict]:
    """Every (protocol, target) with its vector and selection task set.
    Selection only ever uses `usable` tasks (enough filtered pairs); targets
    also include tasks evaluated only on the unfiltered held-out set."""
    hallu = [t for t in usable if pair_types[t] == HALLU]
    out = []
    for s in hallu:
        for t in eval_targets:
            out.append(
                {
                    "protocol": f"source:{s}",
                    "target": t,
                    "vec": s,
                    "selection_tasks": [s],
                }
            )
    if "POOLED" in alias and len(hallu) > 1:
        for t in eval_targets:
            out.append(
                {
                    "protocol": "pooled",
                    "target": t,
                    "vec": "POOLED",
                    "selection_tasks": list(usable),
                }
            )
    for t in eval_targets:
        name = f"LOTO:{t}"
        others = [x for x in usable if x != t]
        if name in alias and any(pair_types[x] == HALLU for x in others):
            out.append(
                {
                    "protocol": "loto",
                    "target": t,
                    "vec": name,
                    "selection_tasks": others,
                }
            )
    return out


# =====================================================================
# 3. Risk-gated steering
# =====================================================================


def run_features(pairs: List[Dict], base: List[Dict], task: str, split: str):
    """Router features (inference-time, no clean reference) for every run."""
    out = {}
    for p, b in zip(pairs, base):
        for rt in ("clean", "corrupt"):
            out[f"{task}|{split}|{p['sample_id']}|{rt}"] = {
                "feat": router_features(b, rt),
                "correct": bool(b[f"correct_{rt}"]),
            }
    return out


def fit_gate_router(
    feats: Dict[str, Dict], run_ids: Sequence[str], num_layers: int, mode: str
) -> RiskRouter:
    wrong = np.array([not feats[r]["correct"] for r in run_ids])
    return RiskRouter(default_router_layers(num_layers), mode=mode).fit(
        [feats[r]["feat"] for r in run_ids], wrong
    )


def predict_risk(router: RiskRouter, feats, run_ids) -> np.ndarray:
    return router.predict([feats[r]["feat"] for r in run_ids])


def choose_tau(
    risk, correct_base, correct_after, lam, eps
) -> Tuple[float, float, float]:
    """tau maximising S = (rescued - lam*induced)/n among gates whose
    induction UCB <= eps; tau = 1 (never steer) if no gate helps."""
    risk = np.asarray(risk)
    cb = np.asarray(correct_base, bool)
    ca = np.asarray(correct_after, bool)
    n, nr = len(risk), cb.sum()
    best = (1.0, 0.0, 0.0)
    for tau in np.concatenate([[-np.inf], np.unique(risk)]):
        gate = risk > tau
        resc = int((~cb & ca & gate).sum())
        ind = int((cb & ~ca & gate).sum())
        S = (resc - lam * ind) / max(n, 1)
        if wilson_upper(ind, nr) <= eps and S > best[1] + 1e-12:
            best = (float(tau), float(S), float(gate.mean()))
    return best


def evaluate_protocols(
    protocols: List[Dict],
    tab: pd.DataFrame,
    runs: pd.DataFrame,
    effects: pd.DataFrame,
    feats: Dict[str, Dict],
    pair_types: Dict[str, str],
    num_layers: int,
    lam: float,
    eps: float,
    criterion: str,
    n_random_gate: int,
    seed: int,
    alias: Optional[Dict[str, str]] = None,
    eval_sets: Sequence[str] = ("test", "unfiltered"),
    primary_mode: str = "vocab",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Returns (selection_table, per_run_outcomes, gate_controls, routers).

    Two gates are trained per selection: the candidate-free 'vocab' router
    (deployable; used for safe_layer_gated when primary_mode='vocab') and the
    'candidate' router (comparison only, safe_layer_gated_candidate)."""
    modes = [primary_mode] + [m for m in ("vocab", "candidate") if m != primary_mode]
    rng = np.random.default_rng(seed)
    sel_rows, out_frames, ctl_rows = [], [], []
    routers: Dict[str, Dict] = {}
    sel_cache: Dict[Tuple, Dict] = {}
    eff_idx = effects.set_index(["vec", "layer", "alpha"]).sort_index()
    vec_names = set(eff_idx.index.get_level_values(0))

    def outcomes(vec, layer, alpha, run_ids):
        sub = eff_idx.loc[(vec, layer, np.float32(alpha))]
        return sub[sub.run_id.isin(run_ids)].set_index("run_id").reindex(run_ids)

    for pr in protocols:
        key = (pr["vec"], tuple(pr["selection_tasks"]))
        rkey = "|".join(map(str, key))
        if key not in sel_cache:
            s = select_safe_layer(
                tab, pr["vec"], pr["selection_tasks"], pair_types, eps, criterion
            )
            gate = {}
            if s["selected"]:
                disc_ids = [
                    r
                    for r in feats
                    if r.split("|")[1] == "discovery"
                    and r.split("|")[0] in pr["selection_tasks"]
                ]
                val = runs[
                    (runs.split == "validation") & runs.task.isin(pr["selection_tasks"])
                ]
                o = outcomes(pr["vec"], s["layer"], s["alpha"], list(val.run_id))
                for mode in modes:
                    router = fit_gate_router(feats, disc_ids, num_layers, mode)
                    risk = predict_risk(router, feats, list(val.run_id))
                    tau, sg, rate = choose_tau(
                        risk,
                        val.correct_base.values,
                        o.correct_after.values.astype(bool),
                        lam,
                        eps,
                    )
                    router.tau = tau
                    routers[f"{rkey}|{mode}"] = router.state_dict()
                    sfx = "" if mode == primary_mode else f"_{mode}"
                    gate.update(
                        {
                            f"tau{sfx}": tau,
                            f"val_S_gated{sfx}": sg,
                            f"val_gate_rate{sfx}": rate,
                        }
                    )
            else:
                for mode in modes:
                    sfx = "" if mode == primary_mode else f"_{mode}"
                    gate.update(
                        {
                            f"tau{sfx}": 1.0,
                            f"val_S_gated{sfx}": 0.0,
                            f"val_gate_rate{sfx}": 0.0,
                        }
                    )
            sel_cache[key] = {**s, **gate, "router_mode": primary_mode}
        s = sel_cache[key]
        sel_rows.append({"protocol": pr["protocol"], "target": pr["target"], **s})

        for es in eval_sets:
            test = runs[(runs.split == es) & (runs.task == pr["target"])].reset_index(
                drop=True
            )
            if test.empty:
                continue
            ids = list(test.run_id)
            cb = test.correct_base.values.astype(bool)
            hb = test.halluc_pick_base.values.astype(bool)
            n = len(test)
            meta = pd.DataFrame(
                {
                    "protocol": pr["protocol"],
                    "target_task": pr["target"],
                    "pair_type": pair_types[pr["target"]],
                    "eval_set": es,
                    "vec": pr["vec"],
                    "layer": s["layer"],
                    "alpha": s["alpha"],
                    "sample_id": test.sample_id.values,
                    "run_type": test.run_type.values,
                    "passes_filter": (
                        test.passes_filter.values if "passes_filter" in test else True
                    ),
                    "correct_base": cb,
                    "halluc_pick_base": hb,
                }
            )

            def emit(method, ca, ha, gated, risk=None):
                f = meta.copy()
                f["method"] = method
                f["correct_after"] = np.asarray(ca, bool)
                f["halluc_pick_after"] = np.asarray(ha, bool)
                f["steered"] = np.asarray(gated, bool)
                f["risk"] = risk if risk is not None else np.nan
                out_frames.append(f)

            zeros = np.zeros(n, bool)
            emit("none", cb, hb, zeros)
            if not s["selected"]:
                for m in ["safe_layer_always", "safe_layer_gated"] + [
                    f"safe_layer_gated_{x}" for x in modes[1:]
                ]:
                    emit(
                        m, cb, hb, zeros
                    )  # abstained: identical to the unmodified model
                continue
            o = outcomes(pr["vec"], s["layer"], s["alpha"], ids)
            ca = o.correct_after.values.astype(bool)
            ha = o.halluc_pick_after.values.astype(bool)
            emit("safe_layer_always", ca, ha, np.ones(n, bool))
            risks, gates = {}, {}
            for mode in modes:
                router = RiskRouter.from_state(routers[f"{rkey}|{mode}"])
                risks[mode] = predict_risk(router, feats, ids)
                gates[mode] = risks[mode] > router.tau
                name = (
                    "safe_layer_gated"
                    if mode == primary_mode
                    else f"safe_layer_gated_{mode}"
                )
                emit(
                    name,
                    np.where(gates[mode], ca, cb),
                    np.where(gates[mode], ha, hb),
                    gates[mode],
                    risks[mode],
                )
            emit("oracle_gate", np.where(~cb, ca, cb), np.where(~cb, ha, hb), ~cb)
            canon = (alias or {}).get(pr["vec"], pr["vec"])
            if f"RANDOM:{canon}" in vec_names:
                orr = outcomes(f"RANDOM:{canon}", s["layer"], s["alpha"], ids)
                if orr.correct_after.notna().all():
                    emit(
                        "random_direction",
                        orr.correct_after.values.astype(bool),
                        orr.halluc_pick_after.values.astype(bool),
                        np.ones(n, bool),
                    )

            gate = gates[primary_mode]
            nr = cb.sum()

            def S_of(g):
                return ((~cb & ca & g).sum() - lam * (cb & ~ca & g).sum()) / n

            S_g = S_of(gate)
            m = int(gate.sum())
            rand_S = (
                np.array(
                    [
                        S_of(
                            np.isin(np.arange(n), rng.choice(n, size=m, replace=False))
                        )
                        for _ in range(n_random_gate)
                    ]
                )
                if 0 < m < n
                else np.full(1, S_g)
            )
            row = {
                "protocol": pr["protocol"],
                "target_task": pr["target"],
                "pair_type": pair_types[pr["target"]],
                "eval_set": es,
                "vec": pr["vec"],
                "layer": s["layer"],
                "alpha": s["alpha"],
                "n_runs": n,
                "base_error_rate": float((~cb).mean()),
                "router_mode": primary_mode,
                "tau": s["tau"],
                "gate_rate": float(gate.mean()),
                "S_always": S_of(np.ones(n, bool)),
                "S_gated": S_g,
                "S_oracle": S_of(~cb),
                "S_random_gate_mean": float(rand_S.mean()),
                "p_gated_beats_random_gate": float(
                    (1 + (rand_S >= S_g - 1e-12).sum()) / (len(rand_S) + 1)
                ),
                "rescued_gated": int((~cb & ca & gate).sum()),
                "induced_gated": int((cb & ~ca & gate).sum()),
                "rescued_always": int((~cb & ca).sum()),
                "induced_always": int((cb & ~ca).sum()),
                "induction_ucb_gated": float(
                    wilson_upper(int((cb & ~ca & gate).sum()), nr)
                ),
            }
            for mode in modes:
                row[f"detector_auroc_{mode}"] = auroc(risks[mode], ~cb)
                row[f"gate_rate_{mode}"] = float(gates[mode].mean())
                row[f"S_gated_{mode}"] = S_of(gates[mode])
            ctl_rows.append(row)
    outs = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    return pd.DataFrame(sel_rows), outs, pd.DataFrame(ctl_rows), routers


# =====================================================================
# 4. Before / after tables
# =====================================================================


def expand_subsets(df: pd.DataFrame) -> pd.DataFrame:
    """subset column: 'conditioned' for the filtered test split; for the
    unfiltered held-out set 'all' plus 'passes_filter' / 'fails_filter', so the
    effect of the validity filter on the numbers is visible."""
    if "eval_set" not in df.columns:
        d = df.copy()
        d["eval_set"], d["subset"] = "test", "conditioned"
        return d
    parts = []
    t = df[df.eval_set == "test"].copy()
    t["subset"] = "conditioned"
    parts.append(t)
    u = df[df.eval_set == "unfiltered"]
    if not u.empty:
        a = u.copy()
        a["subset"] = "all"
        parts.append(a)
        pf = u.passes_filter.astype(bool)
        if pf.any():
            b = u[pf].copy()
            b["subset"] = "passes_filter"
            parts.append(b)
        if (~pf).any():
            c = u[~pf].copy()
            c["subset"] = "fails_filter"
            parts.append(c)
    return pd.concat(parts, ignore_index=True)


def paired_rate_change(
    df: pd.DataFrame,
    keys: Sequence[str],
    bad_before: str,
    bad_after: str,
    label: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Rate of a bad event before vs after, paired per run, for clean / corrupt /
    all runs.  CI resamples questions (sample_id), keeping both runs together."""
    rows = []
    rng = np.random.default_rng(seed)
    for gk, g in df.groupby(list(keys), observed=True):
        gk = gk if isinstance(gk, tuple) else (gk,)
        for scope in ("corrupt", "clean", "all"):
            s = g if scope == "all" else g[g.run_type == scope]
            s = s[s[bad_before].notna() & s[bad_after].notna()]
            if s.empty:
                continue
            b0 = s[bad_before].astype(bool).values
            b1 = s[bad_after].astype(bool).values
            fixed = int((b0 & ~b1).sum())
            broke = int((~b0 & b1).sum())
            n = len(s)
            per = (
                pd.DataFrame({"sid": s.sample_id.values, "b0": b0, "b1": b1})
                .groupby("sid")
                .agg(n=("b0", "size"), b0=("b0", "sum"), b1=("b1", "sum"))
            )
            lo = hi = np.nan
            if len(per) >= 2 and n_boot > 0:
                idx = rng.integers(0, len(per), size=(n_boot, len(per)))
                nn, x0, x1 = (
                    per.n.values[idx].sum(1),
                    per.b0.values[idx].sum(1),
                    per.b1.values[idx].sum(1),
                )
                dd = (x1 - x0) / nn
                lo, hi = float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))
            rb, ra = b0.mean(), b1.mean()
            rows.append(
                {
                    **dict(zip(keys, gk)),
                    "event": label,
                    "scope": scope,
                    "n_runs": n,
                    "n_questions": len(per),
                    "rate_before": rb,
                    "rate_after": ra,
                    "delta": ra - rb,
                    "relative_change": (ra - rb) / rb if rb > 0 else np.nan,
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                    "fixed": fixed,
                    "newly_bad": broke,
                    "p_mcnemar": mcnemar_exact(fixed, broke),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        fam_keys = [k for k in keys if k not in ("method",)] + ["event", "scope"]
        out["p_holm"] = np.nan
        for _, idx in out.groupby(fam_keys, observed=True).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_mcnemar"].values)
    return out


def before_after_from_runs(
    runs_df: pd.DataFrame, keys: Sequence[str], n_boot: int, seed: int
) -> pd.DataFrame:
    """Scored-answer error rate (wrong candidate wins) and picked-distractor
    rate (hallucination tasks: the hinted / plausible wrong option wins)
    before/after, per eval set and subset."""
    d = expand_subsets(runs_df)
    keys = ["eval_set", "subset"] + [k for k in keys if k not in ("eval_set", "subset")]
    d["err_b"] = ~d.correct_base.astype(bool)
    d["err_a"] = ~d.correct_after.astype(bool)
    hall = d.pair_type == HALLU
    d["hp_b"] = d.halluc_pick_base.astype(bool).where(hall)
    d["hp_a"] = d.halluc_pick_after.astype(bool).where(hall)
    a = paired_rate_change(d, keys, "err_b", "err_a", "answer_error", n_boot, seed)
    b = (
        paired_rate_change(
            d[hall], keys, "hp_b", "hp_a", "picked_distractor", n_boot, seed
        )
        if hall.any()
        else pd.DataFrame()
    )
    if "steered" in d.columns:
        rate = (
            d.groupby(list(keys), observed=True)
            .steered.mean()
            .rename("steer_rate")
            .reset_index()
        )
        a = a.merge(rate, on=list(keys), how="left")
        if not b.empty:
            b = b.merge(rate, on=list(keys), how="left")
    return pd.concat([a, b], ignore_index=True)


def component_runs_from_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run before/after rows for the component-level methods, in the same
    format as evaluate_protocols' outcomes (random replicates excluded)."""
    keep = df[
        (df.method.isin(["topk_aie", "harm_aware", "contiguous", "harm_routing"]))
        & (df.rep < 0)
    ].copy()
    if keep.empty:
        return keep
    keep["protocol"] = (
        "component_" + keep.regime.astype(str) + ":" + keep.source_task.astype(str)
    )
    keep["method"] = keep.method.astype(str) + "@k" + keep.k.astype(int).astype(str)
    # only harm_routing can decline to steer; its rows carry the router decision
    keep["steered"] = (
        keep["routed"].astype("boolean").fillna(True).astype(bool)
        if "routed" in keep.columns
        else True
    )
    for c, v in (("eval_set", "test"), ("passes_filter", True)):
        if c not in keep.columns:
            keep[c] = v
    return keep[
        [
            "protocol",
            "target_task",
            "pair_type",
            "eval_set",
            "passes_filter",
            "method",
            "sample_id",
            "run_type",
            "correct_base",
            "correct_after",
            "halluc_pick_base",
            "halluc_pick_after",
            "steered",
        ]
    ]


# =====================================================================
# 4b. Generation before / after
# =====================================================================


def generation_before_after(
    engine: CausalEngine,
    target: str,
    split: str,
    pairs: List[Dict],
    feats: Dict[str, Dict],
    configs: List[Dict],
    max_new_tokens: int,
    ref: Optional[Dict] = None,
    passes: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """configs: dicts with keys name, protocol, kind in {"layer","component"},
    gate (RiskRouter or None), and steering parameters.  A gated config whose
    gate is closed reuses the unsteered generation (identical by construction).
    Labels are automatic proxies (see experiment_runner.classify_generation)."""
    rows = []
    for p in tqdm(pairs, desc=f"gen[{target}/{split}]", leave=False):
        for rt in ("clean", "corrupt"):
            rid = f"{target}|{split}|{p['sample_id']}|{rt}"
            ids = p[f"gen_{rt}_ids"]
            common = {
                "target_task": target,
                "eval_set": split,
                "sample_id": p["sample_id"],
                "run_type": rt,
                "passes_filter": bool((passes or {}).get(p["sample_id"], True)),
                "prompt_question": p.get(f"{rt}_user", "")[-600:],
                "reference_answer": (
                    p["gold"] if rt == "clean" else p.get("gold_corrupt_str", p["gold"])
                ),
            }
            base_txt = engine.generate(ids, max_new_tokens)
            b = classify_generation(base_txt, p, rt, ref)
            rows.append(
                {
                    **common,
                    "protocol": "none",
                    "method": "none",
                    "steered": False,
                    "text": base_txt[:300],
                    "label": b["label"],
                    "label_base": b["label"],
                    "picked_distractor": b["picked_distractor"],
                    "picked_distractor_base": b["picked_distractor"],
                    "f1_gold": b["f1_gold"],
                }
            )
            for c in configs:
                steer = True
                if c.get("gate") is not None:
                    steer = bool(
                        c["gate"].predict([feats[rid]["feat"]])[0] > c["gate"].tau
                    )
                if not steer:
                    txt = base_txt
                elif c["kind"] == "layer":
                    txt = engine.generate(
                        ids,
                        max_new_tokens,
                        resid_layers=[c["layer"]],
                        resid_alpha=c["alpha"],
                        resid_vec=c["vec"],
                    )
                else:
                    txt = engine.generate(
                        ids,
                        max_new_tokens,
                        steer=c["circuit"],
                        alpha=c["alpha"],
                        steer_z=c["steer_z"],
                        steer_mlp=c["steer_mlp"],
                    )
                o = classify_generation(txt, p, rt, ref)
                rows.append(
                    {
                        **common,
                        "protocol": c["protocol"],
                        "method": c["name"],
                        "steered": steer,
                        "text": txt[:300],
                        "label": o["label"],
                        "label_base": b["label"],
                        "picked_distractor": o["picked_distractor"],
                        "picked_distractor_base": b["picked_distractor"],
                        "f1_gold": o["f1_gold"],
                    }
                )
    return pd.DataFrame(rows)


def generation_tables(
    gen: pd.DataFrame, pair_types: Dict[str, str], n_boot: int, seed: int
) -> pd.DataFrame:
    """Before/after rates of proxy events in free generation:
    incorrect_answer    label in INCORRECT_LABELS (a definite wrong answer)
    unverifiable_answer retrieval only: PMID / title not found in the reference corpus
    not_correct         anything but 'correct' (incl. abstain, unparsable, partial)
    abstain             model declined
    picked_distractor   hallucination tasks: chose the hinted / plausible wrong option
    """
    if gen.empty:
        return pd.DataFrame()
    d = expand_subsets(gen[gen.method != "none"].copy())
    d["pair_type"] = d.target_task.map(pair_types)
    for name, test in (
        ("inc", lambda s: s.isin(INCORRECT_LABELS)),
        ("unv", lambda s: s.isin(UNVERIFIABLE_LABELS)),
        ("nc", lambda s: s != "correct"),
        ("abs", lambda s: s == "abstain"),
    ):
        d[f"{name}_b"] = test(d.label_base)
        d[f"{name}_a"] = test(d.label)
    keys = ["eval_set", "subset", "protocol", "target_task", "pair_type", "method"]
    parts = [
        paired_rate_change(d, keys, "inc_b", "inc_a", "incorrect_answer", n_boot, seed),
        paired_rate_change(d, keys, "nc_b", "nc_a", "not_correct", n_boot, seed),
        paired_rate_change(d, keys, "abs_b", "abs_a", "abstain", n_boot, seed),
    ]
    retr = d[d.pair_type != HALLU]
    if not retr.empty:
        parts.append(
            paired_rate_change(
                retr, keys, "unv_b", "unv_a", "unverifiable_answer", n_boot, seed
            )
        )
    hd = d[
        (d.pair_type == HALLU)
        & d.picked_distractor_base.notna()
        & d.picked_distractor.notna()
    ].copy()
    if not hd.empty:
        hd["pd_b"] = hd.picked_distractor_base.astype(bool)
        hd["pd_a"] = hd.picked_distractor.astype(bool)
        parts.append(
            paired_rate_change(
                hd, keys, "pd_b", "pd_a", "picked_distractor", n_boot, seed
            )
        )
    ab = (
        pd.concat([p for p in parts if not p.empty], ignore_index=True)
        if any(not p.empty for p in parts)
        else pd.DataFrame()
    )
    rate = d.groupby(keys).steered.mean().rename("steer_rate").reset_index()
    return ab.merge(rate, on=keys, how="left") if not ab.empty else ab


# =====================================================================
# Plots
# =====================================================================


def make_safe_layer_plots(
    out_dir, tab, rankings, selections, outcomes, controls, eps, lam, feats_risk=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if rankings is not None and not rankings.empty:
        for (vec, split), r in rankings.groupby(["vec", "split"]):
            if vec.startswith("RANDOM:"):
                continue
            scols = [c for c in r.columns if c.startswith("S_")]
            r = r.sort_values("layer")
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for c in scols:
                axes[0].plot(r.layer, r[c], marker="o", ms=3, label=c[2:])
            axes[0].axhline(0, color="k", lw=0.8)
            if selections is not None and not selections.empty:
                chosen = selections[
                    (selections.vec == vec) & selections.selected.astype(bool)
                ]
                for _, cr in chosen.drop_duplicates(
                    subset=["layer", "alpha"]
                ).iterrows():
                    for ax_ in axes:
                        ax_.axvline(cr.layer, color="tab:green", ls=":", lw=1.5)
                    axes[0].annotate(
                        f"selected L{int(cr.layer)} α={cr.alpha:g}",
                        (cr.layer, axes[0].get_ylim()[1]),
                        fontsize=7,
                        color="tab:green",
                        ha="left",
                        va="top",
                    )
            axes[0].set_ylabel(f"safety S = B − {lam:g}·H")
            axes[0].set_title(f"{vec}: safety per layer ({split}; best α per layer)")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)
            axes[1].plot(
                r.layer,
                r.max_induction_ucb,
                marker="o",
                ms=3,
                color="tab:red",
                label="max induction UCB (all tasks)",
            )
            axes[1].axhline(eps, color="tab:red", ls="--", lw=1, label=f"ε = {eps:g}")
            unsafe = r[~r.safe.astype(bool)]
            axes[1].scatter(
                unsafe.layer,
                unsafe.max_induction_ucb,
                marker="x",
                color="k",
                zorder=3,
                label="unsafe",
            )
            axes[1].set_xlabel("layer")
            axes[1].set_ylabel("induction UCB")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    out_dir, f"safe_layer_profile_{vec.replace(':', '-')}_{split}.png"
                ),
                dpi=150,
            )
            plt.close(fig)

            if scols:
                M = r[scols].values.T
                fig, ax = plt.subplots(figsize=(11, 0.6 * len(scols) + 2))
                v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
                im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=-v, vmax=v)
                ax.set_yticks(range(len(scols)))
                ax.set_yticklabels([c[2:] for c in scols])
                ax.set_xticks(range(len(r)))
                ax.set_xticklabels(r.layer.values, fontsize=7)
                ax.set_xlabel("layer")
                ax.set_title(f"{vec}: cross-task safety S ({split})")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        out_dir,
                        f"safe_layer_heatmap_{vec.replace(':', '-')}_{split}.png",
                    ),
                    dpi=150,
                )
                plt.close(fig)

    if outcomes is not None and not outcomes.empty:
        methods = [
            "none",
            "safe_layer_always",
            "safe_layer_gated",
            "oracle_gate",
            "random_direction",
        ]
        for (proto, es), g in outcomes.groupby(["protocol", "eval_set"]):
            targets = sorted(g.target_task.unique())
            fig, ax = plt.subplots(figsize=(1.8 * len(targets) + 3, 4.5))
            w = 0.8 / len(methods)
            for i, m in enumerate(methods):
                gm = g[g.method == m]
                if gm.empty:
                    continue
                vals = [
                    1 - gm[gm.target_task == t].correct_after.mean() for t in targets
                ]
                ax.bar(np.arange(len(targets)) + i * w, vals, width=w, label=m)
            ax.set_xticks(np.arange(len(targets)) + 0.4 - w / 2)
            ax.set_xticklabels(targets)
            ax.set_ylabel(f"answer error rate ({es}, clean+corrupt runs)")
            ax.set_title(f"Before (none) vs after: {proto} [{es}]")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    out_dir, f"before_after_{proto.replace(':', '-')}_{es}.png"
                ),
                dpi=150,
            )
            plt.close(fig)

        # risk-gate trade-off curves on test
        al = outcomes[outcomes.method == "safe_layer_always"]
        gt = outcomes[outcomes.method == "safe_layer_gated"]
        for (proto, t, es), g in al.groupby(["protocol", "target_task", "eval_set"]):
            gg = gt[
                (gt.protocol == proto) & (gt.target_task == t) & (gt.eval_set == es)
            ]
            if gg.empty or gg.risk.isna().all():
                continue
            cb = g.correct_base.values.astype(bool)
            ca = g.correct_after.values.astype(bool)
            risk = gg.risk.values
            nw, nr = max((~cb).sum(), 1), max(cb.sum(), 1)
            taus = np.concatenate([[-np.inf], np.unique(risk), [np.inf]])
            pts = np.array(
                [
                    [
                        ((~cb & ca & (risk > tt)).sum()) / nw,
                        ((cb & ~ca & (risk > tt)).sum()) / nr,
                    ]
                    for tt in taus
                ]
            )
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(pts[:, 1], pts[:, 0], marker=".", label="risk gate (vary τ)")
            ax.plot(
                [0, pts[0, 1]],
                [0, pts[0, 0]],
                ls="--",
                color="grey",
                label="random gate",
            )
            c = controls[
                (controls.protocol == proto)
                & (controls.target_task == t)
                & (controls.eval_set == es)
            ]
            if not c.empty:
                tau = c.tau.iloc[0]
                mode = c.router_mode.iloc[0]
                ax.scatter(
                    [((cb & ~ca & (risk > tau)).sum()) / nr],
                    [((~cb & ca & (risk > tau)).sum()) / nw],
                    color="tab:red",
                    zorder=3,
                    label=f"chosen τ ({mode} gate, AUROC {c[f'detector_auroc_{mode}'].iloc[0]:.2f})",
                )
            ax.set_xlabel("induction rate (correct → wrong)")
            ax.set_ylabel("rescue rate (wrong → correct)")
            ax.set_title(f"{proto} → {t} [{es}]")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    out_dir,
                    f"risk_gate_tradeoff_{proto.replace(':', '-')}_{t}_{es}.png",
                ),
                dpi=150,
            )
            plt.close(fig)
