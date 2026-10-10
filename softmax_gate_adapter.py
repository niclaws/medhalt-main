"""Drop-in *additive* probabilistic steering adapter for the existing CausalEngine.

No original file needs to be changed.  The fitted RiskRouter.predict() returns an
estimated P(base prediction incorrect).  Convert that to a *two-class softmax*
P(steer) and continuously scale the intervention strength:

    P(off), P(on) = softmax([0, (logit(risk)-logit(center))/temperature])
    residual' = residual + (alpha * P(on)) * steering_vector

Default center=.5, temperature=1 means P(on)=risk (up to numeric clipping).
The old hard threshold tau is deliberately NOT used by default: selected
policies often had tau=-inf, making a tau-centered gate always-on.

P(steer) is a *routing weight*, not a calibrated probability that steering
will improve the answer.  Fit/check calibration and held-out harm separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch


@dataclass(frozen=True)
class SoftmaxGateConfig:
    temperature: float = 1.0
    center: float = 0.5
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and strictly positive")
        if not math.isfinite(self.center) or not 0 < self.center < 1:
            raise ValueError("center must be strictly between 0 and 1")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")


@dataclass(frozen=True)
class SoftmaxGateResult:
    risk: float
    p_off: float
    p_on: float
    alpha_base: float
    alpha_effective: float


def two_class_softmax(risks: Any, config: SoftmaxGateConfig) -> np.ndarray:
    """Return an [N,2] array (P(off), P(on)), NEVER Boolean decisions.

    Scores from the original logistic router are error risks in [0,1].
    Torch's actual 2-class softmax is used, not a hard threshold or softmax
    over a single number (which would always equal 1).
    """
    risk_np = np.asarray(risks, dtype=np.float64)
    flat = np.atleast_1d(risk_np).reshape(-1)
    if np.any(~np.isfinite(flat)) or np.any((flat < 0) | (flat > 1)):
        raise ValueError("router probabilities must be finite values in [0,1]")
    t = torch.as_tensor(
        np.clip(flat, config.epsilon, 1 - config.epsilon), dtype=torch.float64
    )
    centered_log_odds = (
        torch.logit(t) - math.log(config.center / (1 - config.center))
    ) / config.temperature
    logits = torch.stack(
        (torch.zeros_like(centered_log_odds), centered_log_odds), dim=-1
    )
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    if not np.all(np.isfinite(probs)) or not np.allclose(
        probs.sum(axis=-1), 1.0, atol=1e-10
    ):
        raise FloatingPointError("invalid gate probabilities")
    return probs


def gate_from_risk(
    risk: float, alpha: float, config: SoftmaxGateConfig
) -> SoftmaxGateResult:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    off, on = map(float, two_class_softmax([risk], config)[0])
    return SoftmaxGateResult(
        risk=float(risk),
        p_off=off,
        p_on=on,
        alpha_base=float(alpha),
        alpha_effective=float(alpha * on),
    )


def gate_from_router(
    router: Any, features: dict, alpha: float, config: SoftmaxGateConfig
) -> SoftmaxGateResult:
    """Adapter to existing RiskRouter.predict([features]) -> numpy risk score."""
    if router is None:
        raise ValueError("Selected intervention requires a fitted router")
    risks = np.asarray(router.predict([features]), dtype=np.float64)
    if risks.shape != (1,):
        raise ValueError(
            f"Expected one router probability, received shape {risks.shape}"
        )
    return gate_from_risk(float(risks[0]), alpha, config)


def candidate_probabilities(logits: Any) -> np.ndarray:
    """Softmax over *candidate logits*, not over the gate.

    Restricted-candidate next-token probabilities, NOT probabilities of
    clinical factuality.  Candidate scores are the same first-divergent-token
    scores as the original downstream evaluator.
    """
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Expected at least two finite candidate logits")
    return torch.softmax(torch.as_tensor(a, dtype=torch.float64), dim=-1).cpu().numpy()


class ProbabilisticResidualSteerer:
    """Compatibility wrapper around CausalEngine.run_interventions.

    Steers ONLY the chosen residual layer. Each input gets its OWN alpha_eff.
    Does not change engine, RiskRouter, selection or original evaluation files.
    """

    def __init__(
        self,
        engine: Any,
        vector: torch.Tensor,
        layer: int,
        alpha: float,
        batch_size: int = 64,
    ):
        from patching_engine import Intervention  # loaded only in actual model runs

        self.Intervention = Intervention
        if vector is None or vector.ndim != 2:
            raise ValueError("vector must have shape [num_layers, hidden_size]")
        if tuple(vector.shape) != (engine.num_layers, engine.hidden_size):
            raise ValueError(
                f"vector shape {tuple(vector.shape)} does not match engine {(engine.num_layers, engine.hidden_size)}"
            )
        if not 0 <= int(layer) < engine.num_layers:
            raise ValueError(f"Invalid residual layer: {layer}")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Selected alpha must be finite and positive")
        self.engine = engine
        self.vector = vector.to(device=engine.device, dtype=torch.float32).unsqueeze(0)
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def run(self, state: Any, alpha_effective: float) -> np.ndarray:
        """Actual continuous internal intervention, not post-hoc output blending."""
        if (
            not math.isfinite(alpha_effective)
            or alpha_effective < 0
            or alpha_effective > self.alpha * (1 + 1e-9)
        ):
            raise ValueError("alpha_effective must be in [0, original alpha]")
        if alpha_effective == 0.0:
            return np.asarray(state.cand_logits, dtype=np.float32).copy()
        iv = self.Intervention(
            resid_layers=np.array([self.layer], dtype=np.int64),
            resid_alpha=float(alpha_effective),
            resid_src=0,
        )
        scores = self.engine.run_interventions(
            state,
            [iv],
            steer_resid=self.vector,
            batch_size=self.batch_size,
        )
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[0] != 1:
            raise RuntimeError(f"Unexpected intervention output shape: {scores.shape}")
        return scores[0]

    def run_always(self, state: Any) -> np.ndarray:
        return self.run(state, self.alpha)
