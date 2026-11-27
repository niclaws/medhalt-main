from __future__ import annotations
import csv, hashlib, json, logging, math, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from ._timing import LOG, timed, timed_call, progress, step

TASKS = ("FCT", "NOTA")
RTS = ("clean", "corrupt")
ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RHOS = (0.05, 0.1, 0.25, 0.5)
LARGE_MODELS = {
    "med42_70b": "m42-health/Llama3-Med42-70B",
    "medgemma_27b": "google/medgemma-27b-text-it",
}
MODELS = {
    "med42_8b": "m42-health/Llama3-Med42-8B",
    "mediphi_3.8b": "microsoft/MediPhi-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen3_8b": "Qwen/Qwen3-8B",
    **LARGE_MODELS,
}


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in x]
    if isinstance(x, torch.Tensor):
        return clean_json(x.detach().cpu().tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, (Path, torch.device, torch.dtype)):
        return str(x)
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


@timed
def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(obj), indent=2, sort_keys=True))
    os.replace(tmp, path)


def stable_seed(*parts):
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little"
    ) % (2**32)


@timed
def load_pt(path):
    # Only use trusted, locally generated original experiment checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def wilson(k, n, z=1.6448536269514722):
    if not n:
        return 1.0
    p = k / n
    return min(
        1.0,
        (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
        / (1 + z * z / n),
    )


def metrics(base, after, pred_base=None, pred_after=None):
    b = np.asarray(base, dtype=bool)
    a = np.asarray(after, dtype=bool)
    if b.shape != a.shape or not b.size:
        raise ValueError("Empty or mismatched predictions")
    i = int((b & ~a).sum())
    r = int((~b & a).sum())
    n = len(b)
    nc = int(b.sum())
    nw = n - nc
    return dict(
        n=n,
        n_correct_base=nc,
        n_wrong_base=nw,
        acc_base=float(b.mean()),
        acc_after=float(a.mean()),
        delta_pp=100 * (r - i) / n,
        induced=i,
        rescued=r,
        induction_rate=i / nc if nc else None,
        rescue_rate=r / nw if nw else None,
        induction_ucb=wilson(i, nc),
        safety_score=(r - 2 * i) / n,
        net_correct=r - i,
        mcnemar_p=float(binomtest(min(i, r), i + r, 0.5).pvalue) if i + r else 1.0,
        prediction_change_rate=(
            float(np.mean(np.asarray(pred_base) != np.asarray(pred_after)))
            if pred_base is not None
            else None
        ),
    )


@timed
def choose_threshold(score, base, after, eps=0.10):
    score = np.asarray(score, float)
    base = np.asarray(base, bool)
    after = np.asarray(after, bool)
    best = (0.0, float("inf"))
    rows = []
    for t in np.r_[np.inf, np.unique(score)[::-1], -np.inf]:
        use = score > t
        m = metrics(base, np.where(use, after, base))
        m.update(threshold=float(t), coverage=float(use.mean()))
        rows.append(m)
        if m["induction_ucb"] <= eps and m["safety_score"] > best[0] + 1e-12:
            best = (m["safety_score"], float(t))
    return best[1], rows


def cap_alpha(alpha, rho, h_norm, v_norm):
    if min(alpha, rho, h_norm, v_norm) < 0 or not np.all(
        np.isfinite([alpha, rho, h_norm, v_norm])
    ):
        raise ValueError("Invalid norm/strength")
    return min(alpha, rho * h_norm / v_norm) if v_norm else 0.0


def matched_weights(weights, reps, seed):
    w = np.asarray(weights, float)
    if not len(w) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Invalid weights")
    rng = np.random.default_rng(seed)
    yield "soft", -1, w, np.arange(len(w))
    yield "constant_mean", -1, np.full(len(w), w.mean()), np.full(len(w), -1)
    for rep in range(reps):
        perm = rng.permutation(len(w))
        yield "shuffled_soft", rep, w[perm], perm


def row_for(base, after, labels, gold, **extra):
    b = np.asarray(base, float)
    a = np.asarray(after, float)
    if (
        len(b) != len(labels)
        or b.shape != a.shape
        or not np.isfinite(b).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("Bad candidate logits")
    ib = int(b.argmax())
    ia = int(a.argmax())
    return dict(
        **extra,
        gold=labels[gold],
        pred_base=labels[ib],
        pred_after=labels[ia],
        correct_base=ib == gold,
        correct_after=ia == gold,
        base_logits=b.tolist(),
        after_logits=a.tolist(),
    )


class Store:
    """One atomic directory per condition, raw JSONL survives failures; resume skips only completed conditions."""

    @timed
    def __init__(self, root, config, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = clean_json(config)
        self.resume = resume
        f = self.root / "config.json"
        if f.exists():
            if not resume:
                raise FileExistsError(f"{f}: use --resume or a new output directory")
            if json.loads(f.read_text()) != self.config:
                raise ValueError(f"Config changed: {f}; use a new output directory")
        else:
            write_json(f, self.config)

    def condition(self, name, fn, **meta):
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "DONE.json").exists():
            LOG.info("Resume skip %s", name)
            return
        start = time.perf_counter()
        LOG.info("START %s %s", name, meta)
        if (d / "predictions.jsonl").exists():
            LOG.info("Preserving incomplete predictions for %s", name)
            old = d / f"partial_{time.time_ns()}.jsonl"
            os.replace(d / "predictions.jsonl", old)
        write_json(d / "metadata.json", meta)
        try:
            with (d / "predictions.jsonl").open("w", buffering=1) as f:

                def emit(row):
                    f.write(
                        json.dumps(clean_json({**meta, **row}), allow_nan=False) + "\n"
                    )

                with step(f"Condition execution {name}"):
                    fn(d, emit)
            summarize_dir(d)
            write_json(d / "DONE.json", {"seconds": time.perf_counter() - start})
            LOG.info("DONE %s %.1fs", name, time.perf_counter() - start)
        except BaseException:
            LOG.exception(
                "FAILED %s | elapsed=%.3fs; incomplete JSONL retained",
                name,
                time.perf_counter() - start,
            )
            raise


def clustered_delta_ci(g, reps=1000):
    # Resample question IDs, not clean/corrupt prompts separately.
    x = g.assign(delta=g.correct_after.astype(int) - g.correct_base.astype(int))
    q = x.groupby("sample_id", sort=False).delta.agg(["sum", "size"]).to_numpy(float)
    if len(q) < 2:
        return (None, None, len(q))
    seed = stable_seed(
        "bootstrap",
        *map(
            str,
            g.iloc[0][
                [
                    c
                    for c in [
                        "model",
                        "seed",
                        "source",
                        "target",
                        "split",
                        "method",
                        "variant",
                        "direction",
                        "alpha",
                        "rho",
                        "rep",
                        "run_type",
                    ]
                    if c in g
                ]
            ].values,
        ),
    )
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(q), (reps, len(q)))
    draws = q[ix]
    vals = 100 * draws[:, :, 0].sum(1) / draws[:, :, 1].sum(1)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi), len(q)


@timed
def summarize_dir(d):
    f = d / "predictions.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if not rows:
        return
    df = pd.DataFrame(rows)
    timed_call("to_csv", df.to_csv, d / "predictions.csv", index=False)
    keys = [
        k
        for k in [
            "experiment",
            "model",
            "seed",
            "source",
            "target",
            "heldout",
            "split",
            "variant",
            "method",
            "direction",
            "alpha",
            "rho",
            "rep",
            "run_type",
            "cohort",
        ]
        if k in df
    ]
    out = []
    # Per run_type and combined summaries; no pooling of unrelated conditions.
    for combined in [False, True] if "run_type" in keys else [False]:
        gkeys = [k for k in keys if not (combined and k == "run_type")]
        frame = (
            df
            if not combined
            else (
                df[~df.method.astype(str).str.startswith("oracle_")]
                if "method" in df
                else df
            )
        )
        for vals, g in frame.groupby(gkeys, dropna=False, sort=False):
            if not isinstance(vals, tuple):
                vals = (vals,)
            m = metrics(g.correct_base, g.correct_after, g.pred_base, g.pred_after)
            m.update(dict(zip(gkeys, vals)))
            lo, hi, nq = clustered_delta_ci(g)
            m.update(
                delta_ci_low_pp=lo,
                delta_ci_high_pp=hi,
                n_questions=nq,
                mcnemar_note=(
                    "descriptive_raw_p; combined clean/corrupt not independent"
                    if combined
                    else "raw_p_unadjusted"
                ),
            )
            if "routed" in g:
                m["coverage"] = g.routed.astype(float).mean()
            if "baseline_prediction_drift" in g:
                m["baseline_prediction_drift_rate"] = (
                    g.baseline_prediction_drift.astype(float).mean()
                )
            if "utility" in g:
                m["mean_predicted_utility"] = pd.to_numeric(
                    g.utility, errors="coerce"
                ).mean()
            if combined:
                m["run_type"] = "all"
            for k in ["alpha_effective", "relative_edit", "gain"]:
                if k in g:
                    m["mean_" + k] = pd.to_numeric(g[k], errors="coerce").mean()
            out.append(m)
    timed_call("to_csv", pd.DataFrame(out).to_csv, d / "summary.csv", index=False)
    distkeys = [k for k in keys if k not in ("cohort",)]
    if distkeys:
        for col in ("gold", "pred_base", "pred_after"):
            timed_call(
                "to_csv",
                df.groupby(distkeys + [col], dropna=False)
                .size()
                .rename("count")
                .reset_index()
                .to_csv,
                d / f"distribution_{col}.csv",
                index=False,
            )


@timed
def aggregate(root):
    root = Path(root)
    frames = []
    conditions = []
    for f in progress(sorted(root.rglob("metadata.json")), "aggregate condition files"):
        if not (f.parent / "DONE.json").exists():
            continue
        meta = json.loads(f.read_text())
        conditions.append({**meta, "path": str(f.parent.relative_to(root))})
        p = f.parent / "summary.csv"
        if p.exists():
            frames.append(
                timed_call("pd.read_csv", pd.read_csv, p).assign(
                    condition_path=str(f.parent.relative_to(root))
                )
            )
    if conditions:
        timed_call(
            "to_csv",
            pd.DataFrame(conditions).to_csv,
            root / "completed_conditions.csv",
            index=False,
        )
    if frames:
        df = pd.concat(frames, ignore_index=True)
        timed_call(
            "to_csv", df.to_csv, root / "all_condition_summaries.csv", index=False
        )
        paper_tables(root, df)
        # Average equally over seeds per checkpoint, then checkpoints. Do not treat seeds as independent models.
        keys = [
            k
            for k in [
                "experiment",
                "source",
                "target",
                "heldout",
                "split",
                "variant",
                "method",
                "direction",
                "alpha",
                "rho",
                "rep",
                "run_type",
                "cohort",
            ]
            if k in df
        ]
        vals = [
            k
            for k in [
                "delta_pp",
                "acc_base",
                "acc_after",
                "induction_rate",
                "rescue_rate",
                "safety_score",
                "mean_gain",
                "mean_alpha_effective",
                "mean_relative_edit",
                "prediction_change_rate",
            ]
            if k in df
        ]
        if "model" in df and keys:
            bymodel = (
                df.groupby(["model"] + keys, dropna=False)[vals].mean().reset_index()
            )
            timed_call("to_csv", bymodel.to_csv, root / "by_model.csv", index=False)
            timed_call(
                "to_csv",
                bymodel.groupby(keys, dropna=False)[vals]
                .agg(["mean", "std", "count"])
                .to_csv,
                root / "macro_models.csv",
            )
        # Optional plots, condition-level means (diagnostic, no independent-run inference).
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            s = (
                df[
                    (df.experiment == "strength")
                    & (df.method == "alpha_sweep")
                    & (df.run_type == "all")
                ]
                if {"experiment", "method", "run_type"} <= set(df)
                else pd.DataFrame()
            )
            if len(s):
                for (source, target, split), g in s.groupby(
                    ["source", "target", "split"]
                ):
                    fig, ax = plt.subplots()
                    for direction, h in g.groupby("direction"):
                        p = (
                            h.groupby("alpha")[["rescue_rate", "induction_rate"]]
                            .mean()
                            .sort_index()
                        )
                        ax.plot(p.induction_rate, p.rescue_rate, "o-", label=direction)
                        for a, r in p.iterrows():
                            ax.annotate(
                                f"{a:g}", (r.induction_rate, r.rescue_rate), fontsize=8
                            )
                    ax.set(
                        xlabel="Induction rate among initially correct",
                        ylabel="Rescue rate among initially wrong",
                        title=f"{source} → {target} ({split})",
                    )
                    ax.legend()
                    fig.tight_layout()
                    timed_call(
                        "savefig",
                        fig.savefig,
                        root / f"strength_{source}_{target}_{split}.png",
                        dpi=160,
                    )
                    plt.close(fig)
        except Exception:
            LOG.exception("Optional plot failed")


@timed
def paper_tables(root, df):
    # For paper-style tables, average random/shuffled repetitions within a condition first.
    cols = [
        c
        for c in [
            "delta_pp",
            "acc_base",
            "acc_after",
            "induction_rate",
            "rescue_rate",
            "safety_score",
            "prediction_change_rate",
            "mean_alpha_effective",
            "mean_relative_edit",
            "mean_gain",
            "coverage",
            "induction_ucb",
        ]
        if c in df
    ]
    att = df[df.experiment == "attenuation"]
    if len(att):
        keys = ["model", "seed", "heldout", "target", "variant", "method"]
        a = att.groupby(keys, dropna=False)[cols].mean().reset_index()
        timed_call(
            "to_csv", a.to_csv, root / "attenuation_per_condition.csv", index=False
        )
        timed_call(
            "to_csv",
            a.groupby(["variant", "method", "target"], dropna=False)[cols]
            .agg(["mean", "std", "count"])
            .to_csv,
            root / "table7_extension.csv",
        )
    strength = df[df.experiment == "strength"]
    if len(strength):
        a = strength.copy()
        a.loc[a.method == "relative_cap", "alpha"] = (
            np.nan
        )  # alpha varies by selected source policy; compare by rho
        keys = [
            "model",
            "seed",
            "source",
            "target",
            "split",
            "method",
            "direction",
            "alpha",
            "rho",
            "run_type",
        ]
        a = a.groupby(keys, dropna=False)[cols].mean().reset_index()
        timed_call("to_csv", a.to_csv, root / "strength_per_condition.csv", index=False)
        group = keys[2:]
        timed_call(
            "to_csv",
            a.groupby(group, dropna=False)[cols].agg(["mean", "std", "count"]).to_csv,
            root / "strength_curve.csv",
        )
    matches = []
    paired = []
    for f in root.rglob("matched_cohort_summary.csv"):
        if not (f.parent / "DONE.json").exists():
            continue
        m = timed_call("pd.read_csv", pd.read_csv, f)
        meta = json.loads((f.parent / "metadata.json").read_text())
        matches.append(m.assign(**meta))
    for f in root.rglob("paired_soft_comparisons.csv"):
        if not (f.parent / "DONE.json").exists():
            continue
        m = timed_call("pd.read_csv", pd.read_csv, f)
        meta = json.loads((f.parent / "metadata.json").read_text())
        paired.append(m.assign(**meta))
    if matches:
        timed_call(
            "to_csv",
            pd.concat(matches, ignore_index=True).to_csv,
            root / "matched_format_comparisons.csv",
            index=False,
        )
    if paired:
        x = pd.concat(paired, ignore_index=True)
        timed_call(
            "to_csv", x.to_csv, root / "paired_soft_comparisons.csv", index=False
        )
        keys = ["model", "seed", "heldout", "target", "variant", "comparison"]
        x = x.groupby(keys, dropna=False).delta_pp.mean().reset_index()
        timed_call(
            "to_csv",
            x.groupby(["target", "variant", "comparison"])
            .delta_pp.agg(["mean", "std", "count"])
            .to_csv,
            root / "paired_soft_macro.csv",
        )
