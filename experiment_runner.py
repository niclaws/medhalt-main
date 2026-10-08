"""
experiment_runner.py — discovery, circuit selection, mitigation and evaluation.

Two intervention regimes are evaluated on the SAME held-out test samples:

  patch : oracle activation patching. The selected components on the corrupt run
          receive the clean run's activations at the final position. This is the
          mechanistic / transfer analysis. It needs the clean reference, so it is
          not a deployable mitigation.
  steer : activation steering. delta[c] = mean_discovery(clean_act - corrupt_act)
          is added (scaled by alpha) to each selected component on any run. It
          needs no oracle, so it is the deployable mitigation. It is applied to
          corrupt runs (rescue) AND clean runs (induction of new errors).

Selection methods, for every k in --k_values:
  topk_aie        top-k by |AIE| (the original baseline, now computed correctly)
  random          uniform random k components (excluding top-k), R replicates
  layer_matched   random component of the same (layer, type) per top-k component
  contiguous      best contiguous window of k components (layer-major order)
  harm_aware      top-k by S(c) = R(c) - lambda * H(c), lambda (and alpha) chosen
                  on the validation split by evaluating the *joint* circuit
  harm_routing    harm_aware steering gated per input by a calibrated
                  hallucination-risk estimator (steer regime only)
  none            unmodified model
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from patching_engine import (
    CausalEngine,
    ComponentSpace,
    Intervention,
    RunState,
    cosine,
    log,
    paired_linear_cka,
)

CLIP = 2.0  # clipping for normalized effects inside calibration objectives


# =====================================================================
# Basic helpers
# =====================================================================


def logit_diff(rows: np.ndarray, halluc_idx: int) -> np.ndarray:
    rows = np.atleast_2d(rows)
    return rows[:, 0] - rows[:, halluc_idx]


def is_correct(rows: np.ndarray, gold: int = 0) -> np.ndarray:
    rows = np.atleast_2d(rows)
    return rows.argmax(axis=1) == gold


def run_gold(pair: Dict, run_type: str) -> int:
    """Index of the correct candidate for a run.  Candidate 0 is the clean
    run's answer.  For hallucination pairs the corrupt run has the same
    correct answer (0); for counterfactual pairs (PMID2Title, Title2URL) the
    corrupt run asks about a different paper, so its correct answer is the
    halluc_idx candidate (paper B)."""
    if run_type == "clean":
        return 0
    return int(pair.get("gold_corrupt", 0))


def prepare_pair(engine: CausalEngine, pair: Dict) -> Tuple[RunState, RunState]:
    sc = engine.prepare(pair["corrupt_ids"], pair["cand_ids"])
    so = engine.prepare(pair["clean_ids"], pair["cand_ids"])
    return sc, so


# =====================================================================
# Stage A: base pass (no interventions) — used for validation
# =====================================================================


def base_pass(engine: CausalEngine, pairs: List[Dict], desc: str = "base") -> List[Dict]:
    """For every candidate pair: clean/corrupt candidate logits, the
    hallucination candidate (chosen here when the task defines it via the
    model), logit differences, correctness and last-position residuals."""
    out = []
    for p in tqdm(pairs, desc=desc, leave=False):
        sc, so = prepare_pair(engine, p)
        lc = so.cand_logits.numpy()
        lx = sc.cand_logits.numpy()
        h = p.get("halluc_idx")
        if h is None:  # e.g. NOTA: the distractor the model prefers when corrupted
            h = int(np.argmax(lx[1:]) + 1)
        gold_x = 0 if p["pair_type"] == "hallucination" else int(h)
        out.append(
            {
                "halluc_idx": int(h),
                "gold_corrupt": gold_x,
                "clean_logits": lc,
                "corrupt_logits": lx,
                "d_clean": float(lc[0] - lc[h]),
                "d_corrupt": float(lx[0] - lx[h]),
                "correct_clean": bool(lc.argmax() == 0),
                "correct_corrupt": bool(lx.argmax() == gold_x),
                "corrupt_pred_is_halluc": bool(lx.argmax() == h),
                "halluc_pick_clean": bool(lc.argmax() == h),
                "entropy_clean": so.vocab_entropy,
                "entropy_corrupt": sc.vocab_entropy,
                "vocab_clean": so.vocab_stats,
                "vocab_corrupt": sc.vocab_stats,
                "resid_clean": so.resid.float().cpu().to(torch.float16),
                "resid_corrupt": sc.resid.float().cpu().to(torch.float16),
            }
        )
        # Router features at the last PROMPT token.  When the candidates share an
        # answer prefix (e.g. PMIDs starting with the same digits) the scored
        # position lies after that prefix, which depends on the candidate set;
        # the deployable router must not see it.
        for rt in ("clean", "corrupt"):
            g_ids = p.get(f"gen_{rt}_ids")
            if g_ids is not None and list(g_ids) != list(p[f"{rt}_ids"]):
                vs, rs = engine.prompt_features(g_ids)
                out[-1][f"vocab_prompt_{rt}"] = vs
                out[-1][f"resid_prompt_{rt}"] = rs.float().cpu().to(torch.float16)
    return out


# =====================================================================
# Stage B: steering vectors and per-component discovery sweeps
# =====================================================================


def compute_steering_vectors(engine: CausalEngine, pairs: List[Dict]) -> Dict[str, torch.Tensor]:
    """delta = mean over discovery pairs of (clean - corrupt) activation at the
    final position, per head-slice (o_proj input) and per MLP output."""
    sz = None
    sm = None
    for p in tqdm(pairs, desc="steering vectors", leave=False):
        sc, so = prepare_pair(engine, p)
        dz = (so.z.float() - sc.z.float()).cpu()
        dm = (so.mlp.float() - sc.mlp.float()).cpu()
        sz = dz if sz is None else sz + dz
        sm = dm if sm is None else sm + dm
    n = max(len(pairs), 1)
    return {"z": sz / n, "mlp": sm / n}


def _vec_to_device(vectors, engine):
    return vectors["z"].to(engine.device), vectors["mlp"].to(engine.device)


def discovery_sweep(
    engine: CausalEngine,
    pairs: List[Dict],
    vectors: Dict[str, torch.Tensor],
    batch_size: int = 64,
    with_steer: bool = True,
) -> Dict[str, np.ndarray]:
    """Single-component effects on every discovery pair.
    with_steer=False (counterfactual tasks) skips the steering sweeps.

    e_patch[i, c] : (d_patched - d_corrupt) / denom      (patch c, clean->corrupt)
    gain[i, c]    : (d_steered_corrupt - d_corrupt) / denom   (steer c, alpha=1)
    harm[i, c]    : (d_clean - d_steered_clean) / denom       (steer c on clean run)
    """
    C = engine.space.size
    singles = [np.array([c]) for c in range(C)]
    ivs_patch = [Intervention(patch=s) for s in singles]
    ivs_steer = [Intervention(steer=s, alpha=1.0) for s in singles]
    vz, vm = _vec_to_device(vectors, engine)
    E, G, Hm = [], [], []
    for p in tqdm(pairs, desc="discovery", leave=False):
        h = p["halluc_idx"]
        sc, so = prepare_pair(engine, p)
        d_clean = float(so.cand_logits[0] - so.cand_logits[h])
        d_corr = float(sc.cand_logits[0] - sc.cand_logits[h])
        denom = d_clean - d_corr
        lp = engine.run_interventions(sc, ivs_patch, source=so, batch_size=batch_size)
        E.append((logit_diff(lp, h) - d_corr) / denom)
        if with_steer:
            ls = engine.run_interventions(sc, ivs_steer, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
            lo = engine.run_interventions(so, ivs_steer, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
            G.append((logit_diff(ls, h) - d_corr) / denom)
            Hm.append((d_clean - logit_diff(lo, h)) / denom)
        else:
            G.append(np.zeros(C, dtype=np.float32))
            Hm.append(np.zeros(C, dtype=np.float32))
    return {
        "e_patch": np.stack(E).astype(np.float32),
        "gain": np.stack(G).astype(np.float32),
        "harm": np.stack(Hm).astype(np.float32),
    }


def component_scores(disc: Dict[str, np.ndarray], base_correct_corrupt: np.ndarray) -> Dict[str, np.ndarray]:
    """Benefit/harm decomposition used by harm-aware selection."""
    e = disc["e_patch"]
    g = disc["gain"]
    hm = disc["harm"]
    aie = e.mean(0)
    R_patch = np.clip(e, 0, None).mean(0)
    H_patch = np.clip(-e, 0, None).mean(0)
    R_steer = np.clip(g, 0, None).mean(0)
    # harm when steering: pushing correct clean runs toward the error, and
    # pushing corrupt runs that were already correct toward the error
    H_clean = np.clip(hm, 0, None).mean(0)
    if base_correct_corrupt.any():
        H_corr_ok = np.clip(-g[base_correct_corrupt], 0, None).mean(0)
    else:
        H_corr_ok = np.zeros_like(H_clean)
    return {
        "aie": aie,
        "R_patch": R_patch,
        "H_patch": H_patch,
        "R_steer": R_steer,
        "H_steer": H_clean + H_corr_ok,
        "mean_gain": g.mean(0),
        "mean_harm": hm.mean(0),
    }


# =====================================================================
# Circuit selection
# =====================================================================


def topk(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-score, kind="stable")[:k]


def contiguous_window(signal: np.ndarray, k: int) -> np.ndarray:
    """Window of k consecutive components (layer-major order) with max signal sum."""
    k = min(k, len(signal))
    cs = np.concatenate([[0.0], np.cumsum(signal)])
    sums = cs[k:] - cs[:-k]
    start = int(np.argmax(sums))
    return np.arange(start, start + k)


def random_circuit(C: int, k: int, rng: np.random.Generator, exclude: Sequence[int]) -> np.ndarray:
    pool = np.setdiff1d(np.arange(C), np.asarray(exclude))
    if len(pool) < k:
        pool = np.arange(C)
    return rng.choice(pool, size=k, replace=False)


def layer_matched_circuit(
    space: ComponentSpace, reference: Sequence[int], rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """Same (layer, type) as each reference component, never a reference
    component. MLPs have one per layer, so they are matched to the nearest
    unused MLP layer. Returns (circuit, fraction matched exactly)."""
    ref = set(int(c) for c in reference)
    used = set()
    out, exact = [], 0
    for c in reference:
        l = int(space.layer_of(c))
        if space.is_mlp(c):
            order = sorted(range(space.num_layers), key=lambda x: (abs(x - l), x))
            cands = [space.mlp(x) for x in order if space.mlp(x) not in ref and space.mlp(x) not in used]
            if cands:
                choice = cands[0]
                exact += int(space.layer_of(choice) == l)
            else:
                pool = [x for x in range(space.size) if x not in ref and x not in used]
                choice = int(rng.choice(pool))
        else:
            cands = [x for x in space.heads_in_layer(l) if x not in ref and x not in used]
            if cands:
                choice = int(rng.choice(cands))
                exact += 1
            else:
                near = sorted(range(space.num_layers), key=lambda x: (abs(x - l), x))
                choice = None
                for x in near:
                    cc = [y for y in space.heads_in_layer(x) if y not in ref and y not in used]
                    if cc:
                        choice = int(rng.choice(cc))
                        break
                if choice is None:
                    pool = [x for x in range(space.size) if x not in ref and x not in used]
                    choice = int(rng.choice(pool))
        out.append(int(choice))
        used.add(int(choice))
    return np.array(out), exact / max(len(reference), 1)


def harm_aware_scores(scores: Dict[str, np.ndarray], regime: str, lam: float) -> np.ndarray:
    if regime == "patch":
        return scores["R_patch"] - lam * scores["H_patch"]
    return scores["R_steer"] - lam * scores["H_steer"]


# =====================================================================
# Evaluation primitives on a split
# =====================================================================


def evaluate_circuits_on_pairs(
    engine: CausalEngine,
    pairs: List[Dict],
    circuits: List[np.ndarray],
    regime: str,
    alphas: Sequence[float],
    vectors=None,
    batch_size: int = 64,
):
    """Joint effect of each circuit on each pair. Returns dict of arrays
    [n_pairs, n_circuits]: gain (corrupt, normalized), harm (clean, normalized,
    steer only), corr_after_corrupt, corr_after_clean (steer only)."""
    n, m = len(pairs), len(circuits)
    gain = np.full((n, m), np.nan, dtype=np.float32)
    harm = np.full((n, m), np.nan, dtype=np.float32)
    ca_x = np.zeros((n, m), dtype=bool)
    ca_c = np.zeros((n, m), dtype=bool)
    vz = vm = None
    if regime == "steer":
        vz, vm = _vec_to_device(vectors, engine)
    for i, p in enumerate(pairs):
        h = p["halluc_idx"]
        sc, so = prepare_pair(engine, p)
        d_clean = float(so.cand_logits[0] - so.cand_logits[h])
        d_corr = float(sc.cand_logits[0] - sc.cand_logits[h])
        denom = d_clean - d_corr
        if regime == "patch":
            ivs = [Intervention(patch=c) for c in circuits]
            lx = engine.run_interventions(sc, ivs, source=so, batch_size=batch_size)
            gain[i] = (logit_diff(lx, h) - d_corr) / denom
            ca_x[i] = is_correct(lx)
        else:
            ivs = [Intervention(steer=c, alpha=a) for c, a in zip(circuits, alphas)]
            lx = engine.run_interventions(sc, ivs, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
            lo = engine.run_interventions(so, ivs, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
            gain[i] = (logit_diff(lx, h) - d_corr) / denom
            harm[i] = (d_clean - logit_diff(lo, h)) / denom
            ca_x[i] = is_correct(lx)
            ca_c[i] = is_correct(lo)
    return {"gain": gain, "harm": harm, "corr_corrupt": ca_x, "corr_clean": ca_c}


def calibration_objective(res: Dict[str, np.ndarray], regime: str) -> np.ndarray:
    g = np.clip(res["gain"], -CLIP, CLIP).mean(0)
    if regime == "patch":
        return g
    return g - np.clip(res["harm"], -CLIP, CLIP).mean(0)


# =====================================================================
# Stage C: calibration on validation (lambda, alpha) + circuits per method
# =====================================================================


def calibrate_and_select(
    engine: CausalEngine,
    val_pairs: List[Dict],
    scores: Dict[str, np.ndarray],
    vectors,
    k_values: Sequence[int],
    lambdas: Sequence[float],
    alphas: Sequence[float],
    num_random: int,
    seed: int,
    batch_size: int = 64,
    steer: bool = True,
) -> Dict:
    """steer=False (counterfactual tasks): only the patch regime is calibrated;
    entry["steer"] is None because there is no hallucination to mitigate."""
    space = engine.space
    C = space.size
    rng = np.random.default_rng(seed)
    sel: Dict = {"k_values": list(map(int, k_values)), "circuits": {}, "calibration": []}

    for k in k_values:
        top = topk(np.abs(scores["aie"]), k)
        contig = contiguous_window(scores["aie"], k)
        entry: Dict = {}

        # ---- patch regime: lambda for harm-aware
        cands = [topk(harm_aware_scores(scores, "patch", lam), k) for lam in lambdas]
        res = evaluate_circuits_on_pairs(engine, val_pairs, cands, "patch", [0.0] * len(cands), batch_size=batch_size)
        obj = calibration_objective(res, "patch")
        best = int(np.nanargmax(obj))
        for lam, o in zip(lambdas, obj):
            sel["calibration"].append({"k": k, "regime": "patch", "method": "harm_aware", "lambda": lam, "alpha": np.nan, "val_objective": float(o)})
        entry["patch"] = {
            "topk_aie": top,
            "contiguous": contig,
            "harm_aware": cands[best],
            "harm_aware_lambda": float(lambdas[best]),
        }

        # ---- steer regime: (lambda, alpha) for harm-aware; alpha for others
        if not steer:
            entry["steer"] = None
            grid = []
        else:
            grid = [(lam, a) for lam in lambdas for a in alphas]
        if steer:
            cands_s = [topk(harm_aware_scores(scores, "steer", lam), k) for lam, _ in grid]
            circ = cands_s + [top] * len(alphas) + [contig] * len(alphas)
            al = [a for _, a in grid] + list(alphas) + list(alphas)
            res = evaluate_circuits_on_pairs(engine, val_pairs, circ, "steer", al, vectors, batch_size)
            obj = calibration_objective(res, "steer")
            ng = len(grid)
            b_h = int(np.nanargmax(obj[:ng]))
            b_t = int(np.nanargmax(obj[ng : ng + len(alphas)]))
            b_c = int(np.nanargmax(obj[ng + len(alphas) :]))
            for (lam, a), o in zip(grid, obj[:ng]):
                sel["calibration"].append({"k": k, "regime": "steer", "method": "harm_aware", "lambda": lam, "alpha": a, "val_objective": float(o)})
            for a, o in zip(alphas, obj[ng : ng + len(alphas)]):
                sel["calibration"].append({"k": k, "regime": "steer", "method": "topk_aie", "lambda": np.nan, "alpha": a, "val_objective": float(o)})
            for a, o in zip(alphas, obj[ng + len(alphas) :]):
                sel["calibration"].append({"k": k, "regime": "steer", "method": "contiguous", "lambda": np.nan, "alpha": a, "val_objective": float(o)})
            entry["steer"] = {
                "topk_aie": top,
                "contiguous": contig,
                "harm_aware": cands_s[b_h],
                "harm_aware_lambda": float(grid[b_h][0]),
                "alpha": {
                    "harm_aware": float(grid[b_h][1]),
                    "topk_aie": float(alphas[b_t]),
                    "contiguous": float(alphas[b_c]),
                    "random": float(alphas[b_t]),
                    "layer_matched": float(alphas[b_t]),
                },
            }

        # ---- random controls (shared by both regimes)
        rand, lm, lm_exact = [], [], []
        for r in range(num_random):
            rand.append(random_circuit(C, k, rng, exclude=top))
            c, ex = layer_matched_circuit(space, top, rng)
            lm.append(c)
            lm_exact.append(ex)
        entry["random"] = rand
        entry["layer_matched"] = lm
        entry["layer_matched_exact_fraction"] = float(np.mean(lm_exact)) if lm_exact else np.nan
        sel["circuits"][int(k)] = entry
    return sel


# =====================================================================
# Stage C2: risk-calibrated router
# =====================================================================


def router_features(b: Dict, run_type: str) -> Dict:
    """Router inputs for one run, from its base-pass entry.

    vocab_stats / resid are taken at the LAST PROMPT TOKEN (before any shared
    answer prefix), over the full vocabulary, so they need no candidate set.
    cand_logits are kept only for the 'candidate' comparison mode."""
    return {
        "vocab_stats": np.asarray(b.get(f"vocab_prompt_{run_type}", b[f"vocab_{run_type}"]), dtype=np.float64),
        "resid": (b.get(f"resid_prompt_{run_type}") if b.get(f"resid_prompt_{run_type}") is not None
                  else b[f"resid_{run_type}"]).float(),
        "cand_logits": np.asarray(b[f"{run_type}_logits"], dtype=np.float64),
    }


class RiskRouter:
    """Logistic hallucination-risk estimator.

    mode="vocab" (default, deployable): full-vocabulary next-token entropy,
        top-1 probability and top1-top2 margin at the last prompt token, plus
        PCA projections of the residual stream at three depths.  Uses no
        candidate answers, so it applies to open-ended generation.
    mode="candidate" (comparison only): additionally uses the logits of the
        task's candidate answers (margin, normalised entropy, max prob).  For
        MCQ these are the options shown in the prompt; for the retrieval tasks
        they are built from reference answers and are NOT available at
        inference time.
    Steering is applied iff predicted risk > tau."""

    def __init__(self, layers: Sequence[int], n_pca: int = 4, l2: float = 1.0, mode: str = "vocab"):
        if mode not in ("vocab", "candidate"):
            raise ValueError(mode)
        self.layers = list(layers)
        self.n_pca = n_pca
        self.l2 = l2
        self.mode = mode
        self.tau = 1.0

    @staticmethod
    def _cand_feats(cand_logits: np.ndarray) -> np.ndarray:
        s = np.sort(cand_logits)[::-1]
        p = np.exp(cand_logits - cand_logits.max())
        p = p / p.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum()) / max(np.log(len(p)), 1e-6)
        margin = float(s[0] - s[1]) if len(s) > 1 else 0.0
        return np.array([margin, ent, float(p.max()) * len(p)], dtype=np.float64)

    def _raw(self, feats: Sequence[Dict]) -> np.ndarray:
        rows = []
        for f in feats:
            parts = [f["vocab_stats"]]
            if self.mode == "candidate":
                parts.append(self._cand_feats(f["cand_logits"]))
            for j, l in enumerate(self.layers):
                parts.append((f["resid"][l].double().numpy() - self.mu[j]) @ self.pcs[j])
            rows.append(np.concatenate(parts))
        return np.stack(rows)

    def fit(self, feats: Sequence[Dict], wrong: np.ndarray):
        self.mu, self.pcs = [], []
        for l in self.layers:
            X = torch.stack([f["resid"][l].double() for f in feats]).numpy()
            mu = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            self.mu.append(mu)
            self.pcs.append(Vt[: self.n_pca].T)
        F = self._raw(feats)
        self.fm = F.mean(0)
        self.fs = F.std(0) + 1e-6
        X = torch.tensor((F - self.fm) / self.fs)
        y = torch.tensor(np.asarray(wrong, dtype=np.float64))
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = X @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + self.l2 * (w**2).sum() / len(y)
            loss.backward()
            return loss

        if 0 < y.sum() < len(y):
            opt.step(closure)
        else:
            with torch.no_grad():
                b.fill_(10.0 if y.mean() > 0.5 else -10.0)
        self.w = w.detach().numpy()
        self.b = float(b.detach().numpy()[0])
        return self

    def predict(self, feats: Sequence[Dict]) -> np.ndarray:
        if not len(feats):
            return np.zeros(0)
        F = (self._raw(feats) - self.fm) / self.fs
        return 1.0 / (1.0 + np.exp(-(F @ self.w + self.b)))

    def choose_tau(self, risk: np.ndarray, delta_correct: np.ndarray):
        """delta_correct[r] = correct_if_steered - correct_if_not, per run."""
        best_tau, best_val = 1.0, 0.0
        for tau in np.unique(np.concatenate([risk, [0.0, 1.0]]))[::-1]:
            val = float(delta_correct[risk > tau].sum())
            if val > best_val + 1e-12:
                best_tau, best_val = float(tau), val
        self.tau = best_tau
        return best_tau

    def state_dict(self):
        return {k: getattr(self, k) for k in ["layers", "n_pca", "l2", "mode", "tau", "mu", "pcs", "fm", "fs", "w", "b"]}

    @classmethod
    def from_state(cls, d):
        r = cls(d["layers"], d["n_pca"], d["l2"], d.get("mode", "vocab"))
        for k, v in d.items():
            setattr(r, k, v)
        return r


def default_router_layers(num_layers: int) -> List[int]:
    return sorted({num_layers // 4, num_layers // 2, (3 * num_layers) // 4})


def fit_router(
    engine: CausalEngine,
    disc_base: List[Dict],
    val_pairs: List[Dict],
    val_base: List[Dict],
    sel: Dict,
    vectors,
    batch_size: int = 64,
    mode: str = "vocab",
) -> Tuple[Dict[int, Dict], List[Dict]]:
    """Router fitted on discovery runs (labels = base model wrong), tau chosen
    per k on validation runs using the harm-aware steering circuit."""
    feats, wrong = [], []
    for b in disc_base:
        for kind in ("clean", "corrupt"):
            feats.append(router_features(b, kind))
            wrong.append(not b[f"correct_{kind}"])
    base_router = RiskRouter(default_router_layers(engine.num_layers), mode=mode).fit(feats, np.array(wrong))

    # validation runs: features + steered outcome for each k's harm-aware circuit
    vz, vm = _vec_to_device(vectors, engine)
    ks = sel["k_values"]
    vfeats, before = [], []
    after = {k: [] for k in ks}
    for p, b in zip(val_pairs, val_base):
        sc, so = prepare_pair(engine, p)
        ivs = [Intervention(steer=sel["circuits"][k]["steer"]["harm_aware"], alpha=sel["circuits"][k]["steer"]["alpha"]["harm_aware"]) for k in ks]
        for st, rt in ((so, "clean"), (sc, "corrupt")):
            gold = run_gold(p, rt)
            lg = engine.run_interventions(st, ivs, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
            vfeats.append(router_features(b, rt))
            before.append(bool(st.cand_logits.numpy().argmax() == gold))
            for j, k in enumerate(ks):
                after[k].append(bool(lg[j].argmax() == gold))
    risk = base_router.predict(vfeats)
    routers, rows = {}, []
    for k in ks:
        r = RiskRouter.from_state(dict(base_router.state_dict()))
        delta = np.array(after[k], dtype=float) - np.array(before, dtype=float)
        tau = r.choose_tau(risk, delta) if len(delta) else 1.0
        routers[int(k)] = r.state_dict()
        rows.append({"k": k, "router_mode": mode, "tau": tau, "val_runs": len(delta),
                     "val_net_gain_if_always": float(delta.sum()) if len(delta) else np.nan,
                     "val_net_gain_routed": float(delta[risk > tau].sum()) if len(delta) else np.nan,
                     "val_route_rate": float((risk > tau).mean()) if len(delta) else np.nan})
    return routers, rows


# =====================================================================
# Stage D: held-out test evaluation (all methods, all k, all source tasks)
# + per-layer cross-task transfer
# =====================================================================

DET_METHODS = ["topk_aie", "harm_aware", "contiguous"]
RAND_METHODS = ["random", "layer_matched"]


def evaluate_target(
    engine: CausalEngine,
    target: str,
    test_pairs: List[Dict],
    sources: Dict[str, Dict],
    layer_top_m: int = 4,
    batch_size: int = 64,
    test_base: Optional[List[Dict]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """sources[s] = {"sel", "vectors", "routers", "scores"} for each source task.
    test_base: base-pass entries aligned with test_pairs (router features).

    Returns (method_rows, layer_rows). One row per
    (source, regime, method, k, rep, run_type, sample)."""
    space = engine.space
    L, H = engine.num_layers, engine.num_heads
    rows, lrows = [], []

    for pi, p in enumerate(tqdm(test_pairs, desc=f"test[{target}]", leave=False)):
        h = p["halluc_idx"]
        sc, so = prepare_pair(engine, p)
        base = {"corrupt": sc.cand_logits.numpy(), "clean": so.cand_logits.numpy()}
        bentry = test_base[pi] if test_base is not None else None
        d_clean = float(base["clean"][0] - base["clean"][h])
        d_corr = float(base["corrupt"][0] - base["corrupt"][h])
        denom = d_clean - d_corr
        gold = {"clean": run_gold(p, "clean"), "corrupt": run_gold(p, "corrupt")}
        common = {
            "target_task": target,
            "sample_id": p["sample_id"],
            "pair_type": p["pair_type"],
            "denom": denom,
            # a real hallucination: hallucination-type pair whose corrupt run is wrong
            "hallucinated": bool(p["pair_type"] == "hallucination" and base["corrupt"].argmax() != gold["corrupt"]),
        }

        def add(src, regime, method, k, rep, run_type, logits_row):
            b = base[run_type]
            g = gold[run_type]
            d_b = float(b[0] - b[h])
            d_a = float(logits_row[0] - logits_row[h])
            rows.append({
                **common,
                "source_task": src, "regime": regime, "method": method, "k": k, "rep": rep,
                "run_type": run_type,
                "d_base": d_b, "d_after": d_a,
                "norm_gain": (d_a - d_b) / denom,
                "correct_base": bool(b.argmax() == g),
                "correct_after": bool(np.argmax(logits_row) == g),
                "halluc_pick_base": bool(b.argmax() == h),
                "halluc_pick_after": bool(np.argmax(logits_row) == h),
                # patch regime on counterfactual pairs: did the clean run's answer transfer?
                "flip_to_clean": bool(np.argmax(logits_row) == 0),
            })

        # ---- none
        for rt in ("corrupt", "clean"):
            add("none", "none", "none", 0, -1, rt, base[rt])

        # ---- patch regime: all sources, all methods, one batched call
        p_ivs, p_tags = [], []
        for s, S in sources.items():
            for k, ent in S["sel"]["circuits"].items():
                for m in DET_METHODS:
                    p_ivs.append(Intervention(patch=ent["patch"][m])); p_tags.append((s, m, k, -1))
                for m in RAND_METHODS:
                    for r, c in enumerate(ent[m]):
                        p_ivs.append(Intervention(patch=c)); p_tags.append((s, m, k, r))
            # per-layer: source's top-m heads (by |AIE|) inside each layer
            aie = S["scores"]["aie"]
            for l in range(L):
                hs = space.heads_in_layer(l)
                pick = hs[np.argsort(-np.abs(aie[hs]))[:layer_top_m]]
                p_ivs.append(Intervention(patch=pick)); p_tags.append((s, "layer_topm", l, -2))
        # source-independent per-layer: whole attention layer / MLP
        for l in range(L):
            p_ivs.append(Intervention(patch=space.heads_in_layer(l))); p_tags.append(("*", "layer_all_heads", l, -2))
            p_ivs.append(Intervention(patch=np.array([space.mlp(l)]))); p_tags.append(("*", "layer_mlp", l, -2))
        lp = engine.run_interventions(sc, p_ivs, source=so, batch_size=batch_size)
        for (s, m, k, r), lg in zip(p_tags, lp):
            if r == -2:
                lrows.append({**common, "source_task": s, "regime": "patch", "probe": m, "layer": k, "run_type": "corrupt",
                              "norm_gain": (float(lg[0] - lg[h]) - d_corr) / denom,
                              "correct_base": bool(base["corrupt"].argmax() == gold["corrupt"]),
                              "correct_after": bool(np.argmax(lg) == gold["corrupt"]),
                              "flip_to_clean": bool(np.argmax(lg) == 0)})
            else:
                add(s, "patch", m, k, r, "corrupt", lg)

        # ---- steer regime: hallucination-type sources only (vectors differ per source)
        for s, S in sources.items():
            if not S.get("steerable", True):
                continue
            vz, vm = _vec_to_device(S["vectors"], engine)
            s_ivs, s_tags = [], []
            for k, ent in S["sel"]["circuits"].items():
                al = ent["steer"]["alpha"]
                for m in DET_METHODS:
                    s_ivs.append(Intervention(steer=ent["steer"][m], alpha=al[m])); s_tags.append((m, k, -1))
                for m in RAND_METHODS:
                    for r, c in enumerate(ent[m]):
                        s_ivs.append(Intervention(steer=c, alpha=al[m])); s_tags.append((m, k, r))
            aie = S["scores"]["aie"]
            a_layer = S["sel"]["circuits"][max(S["sel"]["circuits"])]["steer"]["alpha"]["topk_aie"]
            for l in range(L):
                hs = space.heads_in_layer(l)
                pick = hs[np.argsort(-np.abs(aie[hs]))[:layer_top_m]]
                s_ivs.append(Intervention(steer=pick, alpha=a_layer)); s_tags.append(("layer_topm", l, -2))
            for rt, st in (("corrupt", sc), ("clean", so)):
                lg_all = engine.run_interventions(st, s_ivs, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
                harm_rows = {}
                for (m, k, r), lg in zip(s_tags, lg_all):
                    if r == -2:
                        d_b = float(base[rt][0] - base[rt][h])
                        lrows.append({**common, "source_task": s, "regime": "steer", "probe": m, "layer": k, "run_type": rt,
                                      "norm_gain": (float(lg[0] - lg[h]) - d_b) / denom,
                                      "correct_base": bool(base[rt].argmax() == gold[rt]),
                                      "correct_after": bool(np.argmax(lg) == gold[rt]),
                                      "flip_to_clean": bool(np.argmax(lg) == 0)})
                        continue
                    add(s, "steer", m, k, r, rt, lg)
                    if m == "harm_aware":
                        harm_rows[k] = lg
                # ---- routing: gate harm-aware steering by the source router
                for k, lg in harm_rows.items():
                    router = RiskRouter.from_state(S["routers"][k])
                    risk = float(router.predict([router_features(bentry, rt)])[0])
                    use = risk > router.tau
                    add(s, "steer", "harm_routing", k, -1, rt, lg if use else base[rt])
                    rows[-1]["routed"] = bool(use)
                    rows[-1]["risk"] = risk
    return pd.DataFrame(rows), pd.DataFrame(lrows)


def steering_outcomes(
    engine: CausalEngine,
    target: str,
    pairs: List[Dict],
    base_entries: List[Dict],
    sources: Dict[str, Dict],
    batch_size: int = 64,
) -> pd.DataFrame:
    """Correctness before/after component-level steering (no patching, no
    normalisation), for evaluation sets where the clean run need not be
    correct (the unfiltered held-out set).  Same columns as evaluate_target."""
    rows = []
    for p, bentry in tqdm(list(zip(pairs, base_entries)), desc=f"steer-unfiltered[{target}]", leave=False):
        h = p["halluc_idx"]
        sc, so = prepare_pair(engine, p)
        for s, S in sources.items():
            if not S.get("steerable", True):
                continue
            vz, vm = _vec_to_device(S["vectors"], engine)
            ivs, tags = [], []
            for k, ent in S["sel"]["circuits"].items():
                al = ent["steer"]["alpha"]
                for m in DET_METHODS:
                    ivs.append(Intervention(steer=ent["steer"][m], alpha=al[m]))
                    tags.append((m, k))
            for rt, st in (("clean", so), ("corrupt", sc)):
                g = run_gold(p, rt)
                b = st.cand_logits.numpy()
                lg_all = engine.run_interventions(st, ivs, steer_z=vz, steer_mlp=vm, batch_size=batch_size)
                common = {"source_task": s, "target_task": target, "regime": "steer", "rep": -1, "sample_id": p["sample_id"],
                          "pair_type": p["pair_type"], "run_type": rt, "correct_base": bool(b.argmax() == g),
                          "halluc_pick_base": bool(b.argmax() == h)}
                for (m, k), lg in zip(tags, lg_all):
                    rows.append({**common, "method": m, "k": k, "correct_after": bool(lg.argmax() == g),
                                 "halluc_pick_after": bool(lg.argmax() == h)})
                    if m == "harm_aware" and k in S["routers"]:
                        router = RiskRouter.from_state(S["routers"][k])
                        risk = float(router.predict([router_features(bentry, rt)])[0])
                        use = risk > router.tau
                        rows.append({**common, "method": "harm_routing", "k": k, "routed": bool(use), "risk": risk,
                                     "correct_after": bool(lg.argmax() == g) if use else bool(b.argmax() == g),
                                     "halluc_pick_after": bool(lg.argmax() == h) if use else bool(b.argmax() == h)})
    return pd.DataFrame(rows)


# =====================================================================
# Stage G: generation-based behavioural metrics
# =====================================================================

# ---------------------------------------------------------------------
# Generation labelling.
#
# These are AUTOMATIC PROXY labels for constrained answers (an MCQ letter, a
# PMID, a paper title). They say whether the answer matches the reference,
# not whether arbitrary free text is factually supported.  Validate them
# with the human-audit sheet (audit_sheet / audit_agreement) before
# calling anything a hallucination rate.
#
#   correct               matches the run's reference answer
#   wrong_option          MCQ: another option letter
#   wrong_pmid_known      PMID that belongs to a DIFFERENT paper in the reference corpus
#   wrong_pmid_unverified PMID not in the reference corpus (existence not checked)
#   other_paper_title     title closely matches a different reference paper
#   partial_match         0.5 <= token-F1 with the reference title < strict threshold
#   unmatched_title       title matches no reference paper (token-F1 < 0.5 to all)
#   abstain / unparsable
# ---------------------------------------------------------------------

ABSTAIN_RE = r"(i'?m not sure|not sure|i don'?t know|do not know|i cannot|i can'?t|unable to|not aware|no such|insufficient information|cannot be determined)"
INCORRECT_LABELS = {"wrong_option", "wrong_pmid_known", "wrong_pmid_unverified", "other_paper_title", "unmatched_title"}
UNVERIFIABLE_LABELS = {"wrong_pmid_unverified", "unmatched_title"}


def _title_tokens(s: str) -> set:
    import re

    return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 2}


def _f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    c = len(a & b)
    if c == 0:
        return 0.0
    p, r = c / len(a), c / len(b)
    return 2 * p * r / (p + r)


def classify_generation(text: str, pair: Dict, run_type: str = "clean", ref: Optional[Dict] = None,
                        strict_f1: float = 0.8) -> Dict:
    """Returns {"label", "picked_distractor", "f1_gold", "f1_best_other", "matched_ref"}.
    ref = {"pmid2title": {pmid: title}, "title_tokens": [(pmid, tokenset), ...]}."""
    import re

    t = text.strip()
    out = {"label": "unparsable", "picked_distractor": None, "f1_gold": np.nan, "f1_best_other": np.nan, "matched_ref": None}
    kind = pair["answer_kind"]
    gold = pair["gold"] if run_type == "clean" else pair.get("gold_corrupt_str", pair["gold"])
    abstains = bool(re.search(ABSTAIN_RE, t[:80], flags=re.I))
    if kind == "letter":
        # an option letter followed by ) . : , or the end (not the article "A patient ...")
        m = (re.match(r"\s*[\(\[]?([A-E])(?:[\)\].:,]|\s*$)", t)
             or re.search(r"(?:answer|option)\s*(?:is|:)?\s*[\(\[]?([A-E])\b", t, flags=re.I)
             or re.search(r"\b([A-E])\)", t))
        letter = m.group(1) if m else None
        if letter is None:
            # the model may restate an option's text instead of its letter
            opts = re.findall(r"^([A-E])\) (.+)$", pair.get(f"{run_type}_user", ""), flags=re.M)
            low = t.lower()
            hits = [l for l, o in opts if len(o.strip()) >= 3 and low.startswith(o.strip().lower()[:40])]
            letter = hits[0] if len(hits) == 1 else None
        if letter is None:
            out["label"] = "abstain" if abstains else "unparsable"
            return out
        out["label"] = "correct" if letter == gold else "wrong_option"
        h = pair.get("halluc_idx")
        if h is not None:
            out["picked_distractor"] = bool(letter == pair["cand_labels"][h])
        return out
    if kind == "pmid":
        m = re.search(r"(\d{4,9})", t)
        if not m:
            out["label"] = "abstain" if abstains else "unparsable"
            return out
        pm = m.group(1)
        out["matched_ref"] = pm
        if pm == gold:
            out["label"] = "correct"
        elif ref is not None and pm in ref.get("pmid2title", {}):
            out["label"] = "wrong_pmid_known"
        else:
            out["label"] = "wrong_pmid_unverified"
        return out
    # title
    a = _title_tokens(t)
    if not a:
        out["label"] = "abstain" if abstains else "unparsable"
        return out
    if abstains and len(a) < 6:
        out["label"] = "abstain"
        return out
    fg = _f1(a, _title_tokens(gold))
    out["f1_gold"] = fg
    best, best_id = 0.0, None
    if ref is not None:
        gold_tok = _title_tokens(gold)
        for pm, tok in ref.get("title_tokens", []):
            if tok == gold_tok:
                continue
            f = _f1(a, tok)
            if f > best:
                best, best_id = f, pm
    out["f1_best_other"] = best
    if fg >= strict_f1 and fg >= best:
        out["label"] = "correct"
    elif best >= strict_f1 and best > fg:
        out["label"], out["matched_ref"] = "other_paper_title", best_id
    elif fg >= 0.5:
        out["label"] = "partial_match"
    elif best >= 0.5:
        out["label"], out["matched_ref"] = "other_paper_title", best_id
    else:
        out["label"] = "unmatched_title"
    return out


def audit_sheet(gen: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """Stratified sample (task x method x auto label) for manual labelling.
    Fill `human_label` with one of: correct / incorrect / abstain / unclear."""
    if gen.empty:
        return gen
    parts = []
    for _, g in gen.groupby(["target_task", "method", "label"]):
        parts.append(g.sample(min(n_per_cell, len(g)), random_state=seed))
    s = pd.concat(parts, ignore_index=True)
    s.insert(0, "audit_id", np.arange(len(s)))
    s["human_label"] = ""
    s["human_notes"] = ""
    cols = ["audit_id", "target_task", "method", "protocol", "run_type", "sample_id", "prompt_question", "reference_answer",
            "text", "label", "human_label", "human_notes"]
    return s[[c for c in cols if c in s.columns]]


def _collapse(lbl: str) -> str:
    if lbl == "correct":
        return "correct"
    if lbl in INCORRECT_LABELS or lbl == "incorrect":
        return "incorrect"
    if lbl == "abstain":
        return "abstain"
    return "unclear"


def audit_agreement(sheet: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Agreement of automatic labels with human labels (Cohen's kappa on
    correct / incorrect / abstain / unclear) and the confusion table."""
    s = sheet[sheet.human_label.astype(str).str.strip() != ""].copy()
    if s.empty:
        return pd.DataFrame(), pd.DataFrame()
    s["auto"] = s.label.map(_collapse)
    s["human"] = s.human_label.astype(str).str.strip().str.lower().map(_collapse)
    rows = []
    for key, g in [("ALL", s)] + list(s.groupby("target_task")):
        cats = sorted(set(g.auto) | set(g.human))
        po = float((g.auto == g.human).mean())
        pe = float(sum((g.auto == c).mean() * (g.human == c).mean() for c in cats))
        inc = g[g.auto == "incorrect"]
        rows.append({"target_task": key, "n": len(g), "raw_agreement": po,
                     "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
                     "precision_auto_incorrect": float((inc.human == "incorrect").mean()) if len(inc) else np.nan,
                     "recall_human_incorrect": float((g[g.human == "incorrect"].auto == "incorrect").mean())
                     if (g.human == "incorrect").any() else np.nan})
    conf = pd.crosstab(s.auto, s.human).reset_index()
    return pd.DataFrame(rows), conf


# =====================================================================
# Stage F: representation analysis (paired CKA + direction alignment)
# =====================================================================


def representation_analysis(base_by_task: Dict[str, List[Dict]], num_layers: int, seed: int = 0):
    """(a) per task, per layer: paired linear CKA between clean and corrupt
    residuals of the SAME samples (how much the corruption reshapes geometry);
    (b) across tasks, per layer: cosine between mean (clean - corrupt) residual
    directions, with split-half reliability as the within-task ceiling."""
    rng = np.random.default_rng(seed)
    cka_rows, dir_rows = [], []
    deltas = {}
    for t, base in base_by_task.items():
        if len(base) < 4:
            continue
        Xc = torch.stack([b["resid_clean"].float() for b in base])  # [n, L, D]
        Xx = torch.stack([b["resid_corrupt"].float() for b in base])
        D = Xc - Xx
        deltas[t] = D
        for l in range(num_layers):
            cka_rows.append({"task": t, "layer": l, "n": len(base),
                             "paired_cka_debiased": paired_linear_cka(Xc[:, l], Xx[:, l], True),
                             "paired_cka_biased": paired_linear_cka(Xc[:, l], Xx[:, l], False)})
    tasks = list(deltas)
    for i, a in enumerate(tasks):
        for b in tasks[i:]:
            for l in range(num_layers):
                if a == b:
                    n = deltas[a].shape[0]
                    perm = rng.permutation(n)
                    h1, h2 = perm[: n // 2], perm[n // 2 :]
                    val = cosine(deltas[a][h1, l].mean(0), deltas[a][h2, l].mean(0))
                    kind = "split_half"
                else:
                    val = cosine(deltas[a][:, l].mean(0), deltas[b][:, l].mean(0))
                    kind = "cross_task"
                dir_rows.append({"task_a": a, "task_b": b, "layer": l, "kind": kind, "direction_cosine": val})
    return pd.DataFrame(cka_rows), pd.DataFrame(dir_rows)


# =====================================================================
# Stage H: statistics
# =====================================================================


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int, conf: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or n_boot <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    a = (1 - conf) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def sign_flip_p(diff: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided paired permutation test of mean(diff) == 0."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def per_sample_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse random replicates to per-sample means; one row per
    (source, target, regime, method, k, sample, run_type)."""
    keys = ["source_task", "target_task", "regime", "method", "k", "sample_id", "run_type"]
    agg = df.groupby(keys, as_index=False).agg(
        norm_gain=("norm_gain", "mean"),
        correct_after=("correct_after", "mean"),
        correct_base=("correct_base", "first"),
        hallucinated=("hallucinated", "first"),
        pair_type=("pair_type", "first"),
        halluc_pick_base=("halluc_pick_base", "first"),
        halluc_pick_after=("halluc_pick_after", "mean"),
        flip_to_clean=("flip_to_clean", "mean"),
    )
    return agg


def summarize(df: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (summary, paired_tests, random_control_tests)."""
    ps = per_sample_table(df[df["method"] != "none"])
    summ = []
    for key, g in ps.groupby(["source_task", "target_task", "regime", "method", "k"]):
        s, t, rg, m, k = key
        gx = g[g.run_type == "corrupt"]
        gc = g[g.run_type == "clean"]
        hallu = gx[gx.hallucinated]
        ok_x = gx[gx.correct_base]
        ok_c = gc[gc.correct_base]
        lo, hi = bootstrap_ci(gx.norm_gain.values, n_boot, seed)
        row = {
            "source_task": s, "target_task": t, "regime": rg, "method": m, "k": k,
            "pair_type": gx.pair_type.iloc[0] if len(gx) else None,
            "n_test": len(gx),
            "mean_norm_gain": gx.norm_gain.mean(), "median_norm_gain": gx.norm_gain.median(),
            "gain_ci_low": lo, "gain_ci_high": hi,
            "acc_corrupt_base": gx.correct_base.mean(), "acc_corrupt_after": gx.correct_after.mean(),
            "n_hallucinated": len(hallu),
            "rescue_rate": hallu.correct_after.mean() if len(hallu) else np.nan,
            "induction_rate_corrupt": (1 - ok_x.correct_after).mean() if len(ok_x) else np.nan,
            "mean_norm_gain_hallucinated": hallu.norm_gain.mean() if len(hallu) else np.nan,
            "halluc_pick_corrupt_base": gx.halluc_pick_base.mean(), "halluc_pick_corrupt_after": gx.halluc_pick_after.mean(),
            # patch regime on counterfactual pairs this is the identity-transfer rate
            "flip_to_clean_rate_corrupt": gx.flip_to_clean.mean(),
        }
        if len(gc):
            row.update({
                "acc_clean_base": gc.correct_base.mean(), "acc_clean_after": gc.correct_after.mean(),
                "induction_rate_clean": (1 - ok_c.correct_after).mean() if len(ok_c) else np.nan,
                "mean_norm_harm_clean": -gc.norm_gain.mean(),
            })
        summ.append(row)
    summary = pd.DataFrame(summ)

    # ---- paired tests on identical samples: each method vs none and vs topk_aie
    tests = []
    for (s, t, rg, k), g in ps.groupby(["source_task", "target_task", "regime", "k"]):
        # per-sample net correctness change (corrupt + clean runs averaged)
        g = g.copy()
        g["delta_correct"] = g.correct_after - g.correct_base.astype(float)
        piv_acc = g.pivot_table(index="sample_id", columns="method", values="delta_correct", aggfunc="mean")
        piv_gain = g[g.run_type == "corrupt"].pivot_table(index="sample_id", columns="method", values="norm_gain", aggfunc="mean")
        fam = []
        for metric, piv in (("delta_correct", piv_acc), ("norm_gain", piv_gain)):
            for m in piv.columns:
                for ref in ("none", "topk_aie"):
                    if m == ref or (ref != "none" and ref not in piv.columns):
                        continue
                    a = piv[m].values
                    b = np.zeros_like(a) if ref == "none" else piv[ref].values
                    d = a - b
                    d = d[np.isfinite(d)]
                    lo, hi = bootstrap_ci(d, n_boot, seed)
                    fam.append({"source_task": s, "target_task": t, "regime": rg, "k": k, "metric": metric,
                                "method": m, "reference": ref, "n": len(d), "mean_diff": float(d.mean()) if len(d) else np.nan,
                                "ci_low": lo, "ci_high": hi, "p_signflip": sign_flip_p(d, n_perm, seed)})
        if fam:
            adj = holm([f["p_signflip"] for f in fam])
            for f, a in zip(fam, adj):
                f["p_holm"] = a
            tests.extend(fam)
    tests = pd.DataFrame(tests)

    # ---- empirical p-values of deterministic circuits vs random replicates
    ctl = []
    xr = df[(df.method.isin(RAND_METHODS)) & (df.run_type == "corrupt")]
    rep_means = xr.groupby(["source_task", "target_task", "regime", "method", "k", "rep"]).norm_gain.mean().reset_index()
    obs = summary[summary.method.isin(["topk_aie", "harm_aware"])]
    for _, o in obs.iterrows():
        for cm in RAND_METHODS:
            r = rep_means[(rep_means.source_task == o.source_task) & (rep_means.target_task == o.target_task)
                          & (rep_means.regime == o.regime) & (rep_means.method == cm) & (rep_means.k == o.k)].norm_gain.values
            if len(r) == 0:
                continue
            ctl.append({"source_task": o.source_task, "target_task": o.target_task, "regime": o.regime, "k": o.k,
                        "observed_method": o.method, "control": cm, "observed_mean": o.mean_norm_gain,
                        "control_mean": float(r.mean()), "control_sd": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
                        "p_one_sided": float((1 + (r >= o.mean_norm_gain).sum()) / (len(r) + 1)), "n_controls": len(r)})
    return summary, tests, pd.DataFrame(ctl)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_layers(lrows: pd.DataFrame, scores_by_task: Dict[str, Dict], space: ComponentSpace) -> pd.DataFrame:
    """Per-layer cross-task metrics: transfer (norm gain), rescue/induction,
    and Spearman correlation of the two tasks' head-AIE profiles in that layer."""
    out = []
    if lrows.empty:
        return pd.DataFrame()
    for key, g in lrows.groupby(["source_task", "target_task", "regime", "probe", "layer", "run_type"]):
        s, t, rg, probe, l, rt = key
        hallu = g[g.hallucinated]
        ok = g[g.correct_base]
        row = {"source_task": s, "target_task": t, "regime": rg, "probe": probe, "layer": l, "run_type": rt,
               "n": len(g), "mean_norm_gain": g.norm_gain.mean(),
               "rescue_rate": hallu.correct_after.mean() if (len(hallu) and rt == "corrupt") else np.nan,
               "induction_rate": (1 - ok.correct_after).mean() if len(ok) else np.nan}
        if s in scores_by_task and t in scores_by_task:
            hs = space.heads_in_layer(int(l))
            row["head_aie_spearman"] = _spearman(scores_by_task[s]["aie"][hs], scores_by_task[t]["aie"][hs])
        out.append(row)
    res = pd.DataFrame(out)
    # relative transfer: T_l(s->t) / T_l(t->t)
    if not res.empty:
        diag = res[res.source_task == res.target_task][["target_task", "regime", "probe", "layer", "run_type", "mean_norm_gain"]]
        diag = diag.rename(columns={"mean_norm_gain": "in_task_gain"})
        res = res.merge(diag, on=["target_task", "regime", "probe", "layer", "run_type"], how="left")
        res["relative_transfer"] = res.mean_norm_gain / res.in_task_gain.replace(0, np.nan)
    return res


# =====================================================================
# Plots
# =====================================================================


def make_plots(out_dir: str, summary: pd.DataFrame, layer_sum: pd.DataFrame, cka: pd.DataFrame, dirs: pd.DataFrame,
               primary_k: int, scores_by_task: Dict[str, Dict], space: ComponentSpace):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    tasks = sorted(set(summary.target_task)) if not summary.empty else []

    # per-task head AIE heatmaps
    for t, sc in scores_by_task.items():
        a = sc["aie"].reshape(space.num_layers, space.per_layer)
        fig, ax = plt.subplots(figsize=(10, 7))
        v = np.nanmax(np.abs(a)) or 1.0
        im = ax.imshow(a, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xlabel(f"head (last column = MLP)")
        ax.set_ylabel("layer")
        ax.set_title(f"{t}: AIE (discovery split, final-position patching)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"aie_{t}.png"), dpi=160); plt.close(fig)

    # transfer matrices at primary k
    for rg in ("patch", "steer"):
        for m in ("topk_aie", "harm_aware", "harm_routing", "random"):
            sub = summary[(summary.regime == rg) & (summary.method == m) & (summary.k == primary_k)]
            if sub.empty:
                continue
            src = sorted(set(sub.source_task))
            M = np.full((len(src), len(tasks)), np.nan)
            for _, r in sub.iterrows():
                if r.target_task in tasks:
                    M[src.index(r.source_task), tasks.index(r.target_task)] = r.mean_norm_gain
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(M, cmap="viridis")
            ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=30, ha="right")
            ax.set_yticks(range(len(src))); ax.set_yticklabels(src)
            ax.set_xlabel("target task (test split)"); ax.set_ylabel("source task (circuit)")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, "NA" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", color="w")
            ax.set_title(f"{rg} / {m}: mean normalized gain (k={primary_k})")
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"transfer_{rg}_{m}_k{primary_k}.png"), dpi=160); plt.close(fig)

    # k-sensitivity per method (in-task, averaged over tasks)
    for rg in ("patch", "steer"):
        sub = summary[(summary.regime == rg) & (summary.source_task == summary.target_task)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for m, g in sub.groupby("method"):
            gg = g.groupby("k").mean_norm_gain.mean()
            ax.plot(gg.index, gg.values, marker="o", label=m)
        ax.set_xlabel("k"); ax.set_ylabel("mean normalized gain (in-task, test)")
        ax.set_title(f"{rg}: gain vs circuit size"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"k_sensitivity_{rg}.png"), dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for m, g in sub.groupby("method"):
            gg = g.groupby("k")[["rescue_rate", "induction_rate_corrupt"]].mean()
            ax.plot(gg.index, gg.rescue_rate, marker="o", label=f"{m} rescue")
            ax.plot(gg.index, gg.induction_rate_corrupt, marker="x", ls="--", label=f"{m} induction")
        ax.set_xlabel("k"); ax.set_ylabel("rate"); ax.set_title(f"{rg}: rescue vs induction (in-task)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"rescue_induction_{rg}.png"), dpi=160); plt.close(fig)

    # per-layer cross-task transfer curves
    if layer_sum is not None and not layer_sum.empty:
        for rg in ("patch", "steer"):
            sub = layer_sum[(layer_sum.regime == rg) & (layer_sum.probe == "layer_topm") & (layer_sum.run_type == "corrupt")]
            for t, g in sub.groupby("target_task"):
                fig, ax = plt.subplots(figsize=(9, 5))
                for s, gs in g.groupby("source_task"):
                    gs = gs.sort_values("layer")
                    ax.plot(gs.layer, gs.mean_norm_gain, marker="o", ms=3, label=f"{s}→{t}")
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel("layer"); ax.set_ylabel("mean normalized gain")
                ax.set_title(f"{rg}: per-layer transfer into {t} (source top-m heads per layer)")
                ax.legend(); ax.grid(alpha=0.3)
                fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"layer_transfer_{rg}_to_{t}.png"), dpi=160); plt.close(fig)

    if cka is not None and not cka.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for t, g in cka.groupby("task"):
            ax.plot(g.layer, g.paired_cka_debiased, marker="o", ms=3, label=t)
        ax.set_xlabel("layer"); ax.set_ylabel("paired linear CKA (clean vs corrupt, debiased)")
        ax.set_title("How much the corruption reshapes representations"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "paired_cka_clean_vs_corrupt.png"), dpi=160); plt.close(fig)
    if dirs is not None and not dirs.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (a, b, kind), g in dirs.groupby(["task_a", "task_b", "kind"]):
            ax.plot(g.layer, g.direction_cosine, ls="--" if kind == "split_half" else "-",
                    label=f"{a}" + (" (split-half)" if kind == "split_half" else f" ↔ {b}"))
        ax.set_xlabel("layer"); ax.set_ylabel("cosine of mean (clean − corrupt) direction")
        ax.set_title("Cross-task alignment of hallucination directions"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "direction_alignment.png"), dpi=160); plt.close(fig)
