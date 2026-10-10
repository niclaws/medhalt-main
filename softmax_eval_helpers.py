"""Shared stand-alone metrics and cached-split helpers for softmax steering.

Requires ORIGINAL local state checkpoints (pairs_*.pt and base_*.pt); the
research-paper ZIP built with --skip-checkpoints does not contain them.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from softmax_gate_adapter import SoftmaxGateConfig, two_class_softmax


def checkpoint_split(run_dir: Path, task: str, split: str) -> list[tuple[dict, dict]]:
    """Return pairs and cached base records in the ORIGINAL sample-ID order."""
    run_dir = Path(run_dir)
    if split not in ("discovery", "validation", "test", "unfiltered"):
        raise ValueError("split must be one of discovery/validation/test/unfiltered")
    with (run_dir / "splits.json").open(encoding="utf8") as f:
        split_map = json.load(f)
    if task not in split_map:
        raise KeyError(f"Unknown task {task} in {run_dir}/splits.json")
    ids = list(map(str, split_map[task].get(split, [])))
    if not ids:
        return []
    state = run_dir / "state"
    pairs_path, base_path = state / f"pairs_{task}.pt", state / f"base_{task}.pt"
    for path in (pairs_path, base_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing original checkpoint {path}; archived results CSVs alone cannot rerun interventions"
            )
    # Deliberately matches the ORIGINAL loader. Only load your own trusted .pt files.
    pairs_obj = torch.load(pairs_path, map_location="cpu", weights_only=False)
    pairs = pairs_obj["pairs"] if isinstance(pairs_obj, dict) else pairs_obj
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if len(pairs) != len(base):
        raise ValueError(f"Pairs and base rows are mismatched for {task}")
    by_id = {str(pair["sample_id"]): i for i, pair in enumerate(pairs)}
    if len(by_id) != len(pairs) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate pair or split IDs")
    if any(k not in by_id for k in ids):
        raise ValueError("Split ID absent from pairs checkpoint")
    return [(pairs[by_id[k]], base[by_id[k]]) for k in ids]


def temperature_from_validation(
    router,
    validation_features: list[dict],
    wrong: list[bool],
    candidates: tuple[float, ...] = (0.35, 0.50, 0.70, 1.0, 1.4, 2.0, 2.8, 4.0),
) -> float:
    """Optional temperature calibration on ORIGINAL Med-HALT validation labels.

    This optimizes base-error log-loss, NOT treatment effect or harm. Do not
    mistake it for a new safety certificate. External target labels are not used.
    """
    if len(validation_features) != len(wrong) or len(wrong) < 10:
        raise ValueError(
            "Need at least 10 original validation observations to tune temperature"
        )
    risk = np.asarray(router.predict(validation_features), dtype=float)
    y = np.asarray(wrong, dtype=float)
    best = (np.inf, 1.0)
    for t in candidates:
        pp = two_class_softmax(risk, SoftmaxGateConfig(temperature=t))[:, 1]
        ll = float(-(y * np.log(pp + 1e-12) + (1 - y) * np.log(1 - pp + 1e-12)).mean())
        if ll < best[0]:
            best = (ll, float(t))
    return best[1]


def interval_bootstrap(
    base: np.ndarray, after: np.ndarray, seed: int, reps: int
) -> tuple[float, float]:
    if reps <= 0:
        raise ValueError("reps must be positive")
    a = np.asarray(after, dtype=float) - np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for i in range(reps):
        draws[i] = a[rng.integers(0, len(a), size=len(a))].mean()
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def wilson_upper(num_bad: int, n_correct: int, z: float = 1.6448536269514722) -> float:
    """One-sided 95% Wilson UCB; 1 when denominator is zero (uncertifiable)."""
    if n_correct == 0:
        return 1.0
    p = num_bad / n_correct
    return float(
        min(
            1.0,
            (
                p
                + z * z / (2 * n_correct)
                + z
                * np.sqrt(p * (1 - p) / n_correct + z * z / (4 * n_correct * n_correct))
            )
            / (1 + z * z / n_correct),
        )
    )


def summarize_predictions(
    df: pd.DataFrame, seed: int, reps: int, harm_eps: float
) -> dict:
    if df.empty:
        raise ValueError("No evaluated samples")
    b = df["correct_base"].to_numpy(dtype=bool)
    result = {
        "n_samples": len(df),
        "n_correct_base": int(b.sum()),
        "acc_base": float(b.mean()),
        "mean_risk": (
            float(df["risk_score"].mean())
            if df["risk_score"].notna().any()
            else float("nan")
        ),
        "mean_p_on": float(df["gate_on_prob"].mean()),
        "min_p_on": float(df["gate_on_prob"].min()),
        "max_p_on": float(df["gate_on_prob"].max()),
        "mean_alpha_effective": float(df["alpha_effective"].mean()),
        "mean_gold_candidate_prob_base": float(df["gold_prob_base"].mean()),
        "mean_gold_candidate_prob_soft": float(df["gold_prob_soft"].mean()),
    }
    for method in ("always", "soft"):
        a = df[f"correct_{method}"].to_numpy(dtype=bool)
        bad = int((b & ~a).sum())
        good = int((~b & a).sum())
        lo, hi = interval_bootstrap(b, a, seed, reps)
        disc = bad + good
        out = {
            f"acc_{method}": float(a.mean()),
            f"delta_{method}": float(a.mean() - b.mean()),
            f"delta_{method}_ci_low": lo,
            f"delta_{method}_ci_high": hi,
            f"induced_{method}": bad,
            f"rescued_{method}": good,
            f"induction_rate_{method}": bad / int(b.sum()) if b.any() else float("nan"),
            f"rescue_rate_{method}": (
                good / int((~b).sum()) if (~b).any() else float("nan")
            ),
            f"ucb_{method}": wilson_upper(bad, int(b.sum())),
            f"p_mcnemar_{method}": (
                float(binomtest(min(bad, good), disc, p=0.5).pvalue) if disc else 1.0
            ),
        }
        result.update(out)
    result["safety_ucb_soft_le_eps"] = bool(result["ucb_soft"] <= harm_eps)
    result["n_prediction_changed_soft_vs_always"] = int(
        (df.pred_soft != df.pred_always).sum()
    )
    return result
