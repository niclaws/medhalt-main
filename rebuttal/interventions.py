"""Thin adapters over the original last-position scoring API."""

import numpy as np
import torch
from patching_engine import Intervention
from .core import cap_alpha, row_for, stable_seed, RTS
from experiment_runner import prepare_pair, run_gold, router_features
from ._timing import timed, timed_call, progress


@timed
def residual(engine, state, layer, vectors, specs, batch_size):
    # specs: (vector index, alpha); one engine call for all active strengths.
    ivs = [
        Intervention(
            resid_layers=np.array([layer], np.int64),
            resid_src=int(i),
            resid_alpha=float(a),
        )
        for i, a in specs
        if a != 0
    ]
    after = (
        iter(
            timed_call(
                "engine.run_interventions",
                engine.run_interventions,
                state,
                ivs,
                steer_resid=torch.stack(vectors),
                batch_size=batch_size,
            )
        )
        if ivs
        else iter(())
    )
    return [
        next(after) if a != 0 else state.cand_logits.numpy().copy() for i, a in specs
    ]


@timed
def random_vector(v, seed):
    v_cpu = v.detach().to(device="cpu", dtype=torch.float32)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = torch.randn(
        v_cpu.shape,
        generator=g,
        dtype=torch.float32,
        device="cpu",
    )

    # Match per-layer L2 norm.
    v_norm = v_cpu.norm(dim=-1, keepdim=True)
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return (x * (v_norm / x_norm)).to(v.device)


@timed
def state_features(engine, state, p, rt):
    ids = p.get(f"gen_{rt}_ids", p[f"{rt}_ids"])
    if list(ids) != list(p[f"{rt}_ids"]):
        vs, res = timed_call("engine.prompt_features", engine.prompt_features, ids)
    else:
        vs, res = state.vocab_stats, state.resid
    return dict(
        vocab_stats=np.asarray(vs),
        resid=res.float().cpu(),
        cand_logits=state.cand_logits.numpy(),
    )


@timed
def pair_states(engine, p):
    sc, so = timed_call("prepare_pair", prepare_pair, engine, p)
    return {"clean": so, "corrupt": sc}


def emit_pair(emit, p, rt, state, after, **kw):
    emit(
        row_for(
            state.cand_logits.numpy(),
            after,
            p["cand_labels"],
            run_gold(p, rt),
            sample_id=p["sample_id"],
            run_type=rt,
            **kw,
        )
    )


@timed
def sweep_pair(
    engine, p, layer, v, randoms, alphas, rhos, selected_alpha, batch_size, emit
):
    for rt, st in progress(
        pair_states(engine, p).items(), "strength clean/corrupt arms"
    ):
        h_norm = float(st.resid[layer].float().norm())
        vectors = [v, *randoms]
        specs = []
        meta = []
        for j, w in enumerate(vectors):
            vn = float(w[layer].float().norm())
            direction = "learned" if j == 0 else "random"
            rep = -1 if j == 0 else j - 1
            for a in alphas:
                specs.append((j, a))
                meta.append(
                    dict(
                        method="alpha_sweep",
                        direction=direction,
                        rep=rep,
                        alpha=a,
                        rho=None,
                        alpha_effective=a,
                        h_norm=h_norm,
                        v_norm=vn,
                        relative_edit=a * vn / h_norm if h_norm else None,
                    )
                )
            for rho in rhos:
                a = cap_alpha(selected_alpha, rho, h_norm, vn)
                specs.append((j, a))
                meta.append(
                    dict(
                        method="relative_cap",
                        direction=direction,
                        rep=rep,
                        alpha=selected_alpha,
                        rho=rho,
                        alpha_effective=a,
                        h_norm=h_norm,
                        v_norm=vn,
                        relative_edit=a * vn / h_norm if h_norm else None,
                    )
                )
        for info, after in zip(
            meta, residual(engine, st, layer, vectors, specs, batch_size)
        ):
            emit_pair(emit, p, rt, st, after, layer=layer, **info)
