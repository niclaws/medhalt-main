"""Scale extension: fresh data/splits at new checkpoints; two explicitly separate modes.
Layer pilot tests matched residual locations. Component mode repeats AIE-based k=20
comparison (very expensive on 27/70B; use a declared discovery subset if needed).
"""

from __future__ import annotations
import random, math
import numpy as np
import pandas as pd
import torch
from patching_engine import Intervention
import experiment_runner as ER
from run_casual_transfer import (
    build_mcq_pairs,
    load_raw,
    TASKS as RAW_TASKS,
    validate,
    stratified_split,
)
from .core import LOG, TASKS, metrics, write_json, load_pt, stable_seed
from .interventions import pair_states, emit_pair, residual, random_vector
from .checkpoints import Run
from ._timing import timed, timed_call, progress


@timed
def prepare_new_run(engine, args, root, model):
    root.mkdir(parents=True, exist_ok=True)
    state = root / "state"
    state.mkdir(exist_ok=True)
    cfg = dict(
        model_name=model,
        seed=args.seed,
        n_samples=args.large_n_samples,
        max_pairs_per_task=args.large_max_pairs,
        unfiltered_frac=0.2,
        split_fracs=[0.75, 0.10, 0.15],
        min_margin=0.5,
        min_denom=1.0,
        alphas=[0.5, 1.0, 2.0, 4.0],
        dtype=str(engine.dtype).split(".")[-1],
        model_commit=getattr(engine.model.config, "_commit_hash", None),
        data_dir=args.data_dir,
        engine=type(engine).__name__,
        quantization=getattr(engine, "quantization", "none"),
    )
    from .core import clean_json
    import json

    if (root / "run_config.json").exists() and json.loads(
        (root / "run_config.json").read_text()
    ) != clean_json(cfg):
        raise ValueError("Large checkpoint settings changed")
    write_json(root / "run_config.json", cfg)
    splits = {}
    rng = np.random.default_rng(args.seed)
    for ti, t in enumerate(progress(TASKS, "large dataset tasks")):
        pfile = state / f"pairs_{t}.pt"
        bfile = state / f"base_{t}.pt"
        if pfile.exists():
            obj = load_pt(pfile)
        else:
            raw = timed_call("load_raw", load_raw, RAW_TASKS[t], "hf", args.data_dir)
            stats = {}
            pairs = timed_call(
                "build_mcq_pairs",
                build_mcq_pairs,
                engine,
                t,
                raw,
                cfg["n_samples"],
                random.Random(args.seed + 1000 * ti),
                stats,
            )
            obj = dict(pairs=pairs, stats=stats, n_raw=len(raw))
            tmp = pfile.with_suffix(".tmp")
            timed_call("torch.save", torch.save, obj, tmp)
            tmp.replace(pfile)
        pairs = obj["pairs"]
        LOG.info("Large %s built %s, rejected=%s", t, len(pairs), obj["stats"])
        if bfile.exists():
            base = load_pt(bfile)
        else:
            # Per-pair checkpointing in batches prevents losing all base forwards.
            chunks = state / f"base_chunks_{t}"
            chunks.mkdir(exist_ok=True)
            base = []
            for i in progress(range(0, len(pairs), 10), "base checkpoint chunks"):
                f = chunks / f"{i:06d}.pt"
                if f.exists():
                    b = load_pt(f)
                else:
                    b = timed_call(
                        "ER.base_pass",
                        ER.base_pass,
                        engine,
                        pairs[i : i + 10],
                        f"{t} base {i}",
                    )
                    tmp = f.with_suffix(".tmp")
                    timed_call("torch.save", torch.save, b, tmp)
                    tmp.replace(f)
                if len(b) != len(pairs[i : i + 10]):
                    raise ValueError("Large base chunk size mismatch")
                base.extend(b)
                LOG.info("Large %s base %s/%s", t, len(base), len(pairs))
            tmp = bfile.with_suffix(".tmp")
            timed_call("torch.save", torch.save, base, tmp)
            tmp.replace(bfile)
        if len(base) != len(pairs):
            raise ValueError("Large pairs/base length mismatch")
        infos = [validate(p, b, 0.5, 1.0) for p, b in zip(pairs, base)]
        hold = sorted(
            np.random.default_rng(args.seed + 99)
            .permutation(len(pairs))[: round(0.2 * len(pairs))]
            .tolist()
        )
        hs = set(hold)
        eligible = [i for i, r in enumerate(infos) if r["valid"] and i not in hs][
            : cfg["max_pairs_per_task"]
        ]
        sp = timed_call(
            "stratified_split",
            stratified_split,
            eligible,
            [infos[i]["hallucinated"] for i in eligible],
            cfg["split_fracs"],
            rng,
        )
        sp["unfiltered"] = hold
        splits[t] = {
            k: [str(pairs[i]["sample_id"]) for i in ix] for k, ix in sp.items()
        }
        LOG.info("Large %s splits=%s", t, {k: len(v) for k, v in sp.items()})
    write_json(root / "splits.json", splits)
    return timed_call("Run", Run, root)


@timed
def locate_layers(engine, examples, layers, args, d):
    rows = []
    for p, b in progress(examples, "layer discovery/calibration samples"):
        st = pair_states(engine, p)
        so, sc = st["clean"], st["corrupt"]
        h = p["halluc_idx"]
        lc = so.cand_logits.numpy()
        lx = sc.cand_logits.numpy()
        D = float(lc[0] - lc[h] - lx[0] + lx[h])
        if D < 1e-6:
            continue
        # At ONE layer, adding target clean minus target corrupt residual equals
        # oracle replacement at that location, up to arithmetic rounding.
        delta = so.resid.float() - sc.resid.float()
        ivs = [
            Intervention(resid_layers=np.array([l]), resid_alpha=1.0) for l in layers
        ]
        out = timed_call(
            "engine.run_interventions",
            engine.run_interventions,
            sc,
            ivs,
            steer_resid=delta,
            batch_size=args.batch_size,
        )
        for l, a in zip(layers, out):
            rows.append(
                dict(
                    sample_id=p["sample_id"],
                    layer=l,
                    gain=float((a[0] - a[h] - lx[0] + lx[h]) / D),
                    correct_after=int(a.argmax()) == 0,
                )
            )
    df = pd.DataFrame(rows)
    timed_call("to_csv", df.to_csv, d / "discovery_layer_patch.csv", index=False)
    if df.empty:
        raise ValueError("No valid discovery contrasts")
    scores = df.groupby("layer").gain.mean()
    return int(scores.idxmax()), scores.to_dict()


@timed
def calibrate_layer(engine, examples, layer, v, args, d):
    rows = []
    alphas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    for p, b in progress(examples, "layer discovery/calibration samples"):
        for rt, st in pair_states(engine, p).items():
            for a, y in zip(
                alphas,
                residual(
                    engine, st, layer, [v], [(0, a) for a in alphas], args.batch_size
                ),
            ):
                rows.append(
                    dict(
                        sample_id=p["sample_id"],
                        run_type=rt,
                        alpha=a,
                        correct_base=st.cand_logits.argmax().item() == 0,
                        correct_after=y.argmax() == 0,
                    )
                )
    df = pd.DataFrame(rows)
    timed_call("to_csv", df.to_csv, d / "validation_predictions.csv", index=False)
    if df.empty:
        raise ValueError("Empty source validation")
    m = pd.DataFrame(
        [
            dict(alpha=a, **metrics(g.correct_base, g.correct_after))
            for a, g in df.groupby("alpha")
        ]
    )
    timed_call("to_csv", m.to_csv, d / "validation_metrics.csv", index=False)
    eligible = m[(m.induction_ucb <= args.harm_eps) & (m.safety_score > 0)].sort_values(
        ["safety_score", "alpha"], ascending=[False, True]
    )
    selected = float(eligible.iloc[0].alpha) if len(eligible) else 0.0
    # Diagnostic strength chosen without certification, clearly separate from safe policy.
    diagnostic = float(
        m[m.alpha > 0]
        .sort_values(["safety_score", "alpha"], ascending=[False, True])
        .iloc[0]
        .alpha
    )
    return selected, diagnostic


@timed
def run_scale_layer(engine, run, source, args, d, emit):
    disc = timed_call(
        "run.split", run.split, source, "discovery", args.large_discovery_limit or None
    )
    val = timed_call("run.split", run.split, source, "validation")
    if not disc or not val:
        raise ValueError("Insufficient large-model source splits")
    layers = (
        list(range(engine.num_layers))
        if args.large_all_layers
        else sorted(
            {
                min(engine.num_layers - 1, round(f * (engine.num_layers - 1)))
                for f in (0.25, 0.5, 0.66, 0.75, 0.9, 1.0)
            }
        )
    )
    layer, loc = locate_layers(engine, disc, layers, args, d)
    v = torch.stack(
        [b["resid_clean"].float() - b["resid_corrupt"].float() for _, b in disc]
    ).mean(0)
    rand = random_vector(v, stable_seed(args.seed, source, "large-random"))
    selected, diagnostic = calibrate_layer(engine, val, layer, v, args, d)
    write_json(
        d / "policy.json",
        dict(
            mode="layer_pilot",
            layer=layer,
            layer_candidates=layers,
            layer_gains=loc,
            selected_alpha=selected,
            diagnostic_alpha=diagnostic,
            discovery_n=len(disc),
            validation_n=len(val),
            discovery_ids=[p["sample_id"] for p, _ in disc],
        ),
    )
    timed_call("torch.save", torch.save, v, d / "direction.pt")
    for target in progress(TASKS, "scale evaluation targets"):
        for split in progress(args.splits, "scale evaluation splits"):
            ex = timed_call("run.split", run.split, target, split, args.limit)
            exclude = timed_call(
                "run.cross_task_overlap", run.cross_task_overlap, source, target, split
            )
            write_json(d / f"excluded_{target}_{split}.json", sorted(exclude))
            for p, b in progress(ex, "scale layer evaluation samples"):
                if p["sample_id"] in exclude:
                    continue
                states = pair_states(engine, p)
                so, sc = states["clean"], states["corrupt"]
                h = p["halluc_idx"]
                lc = so.cand_logits.numpy()
                lx = sc.cand_logits.numpy()
                D = float(lc[0] - lc[h] - lx[0] + lx[h])
                ok = bool(lc.argmax() == 0 and lc[0] - lc[h] >= 0.5 and D >= 1.0)
                for rt, st in states.items():
                    base = st.cand_logits.numpy()
                    specs = [(0, 0.0), (0, selected), (0, diagnostic), (1, diagnostic)]
                    after = residual(
                        engine, st, layer, [v, rand], specs, args.batch_size
                    )
                    methods = [
                        "baseline",
                        "safe_policy",
                        "diagnostic_learned",
                        "diagnostic_random",
                    ]
                    if rt == "corrupt":
                        patch = timed_call(
                            "engine.run_interventions",
                            engine.run_interventions,
                            sc,
                            [
                                Intervention(
                                    resid_layers=np.array([layer]), resid_alpha=1.0
                                )
                            ],
                            steer_resid=so.resid.float() - sc.resid.float(),
                            batch_size=args.batch_size,
                        )[0]
                        after.append(patch)
                        methods.append("oracle_residual_patch")
                        specs.append((0, 1.0))
                    for name, (_, a), y in zip(methods, specs, after):
                        emit_pair(
                            emit,
                            p,
                            rt,
                            st,
                            y,
                            target=target,
                            split=split,
                            method=name,
                            alpha=a,
                            layer=layer,
                            valid=ok,
                            gain=(
                                float((y[0] - y[h] - base[0] + base[h]) / D)
                                if ok and rt == "corrupt"
                                else None
                            ),
                            variant="layer_pilot",
                        )


@timed
def run_scale_component(engine, run, source, args, d, emit):
    disc = timed_call(
        "run.split", run.split, source, "discovery", args.large_discovery_limit or None
    )
    val = timed_call("run.split", run.split, source, "validation")
    if not disc or not val:
        raise ValueError("Insufficient component source splits")
    ps = [p for p, _ in disc]
    vs = [p for p, _ in val]
    k = args.k
    v = timed_call(
        "ER.compute_steering_vectors", ER.compute_steering_vectors, engine, ps
    )
    timed_call("torch.save", torch.save, v, d / "component_vectors.pt")
    # Exact per-component AIE on the declared discovery cohort; no restricted layer shortcut.
    # Write per-pair discovery effects so progress remains if interrupted.
    parts = d / "discovery_chunks"
    parts.mkdir(exist_ok=True)
    chunks = []
    for i, p in enumerate(progress(ps, "component discovery samples")):
        f = parts / f"{i:06d}.pt"
        if f.exists():
            x = load_pt(f)
        else:
            x = timed_call(
                "ER.discovery_sweep",
                ER.discovery_sweep,
                engine,
                [p],
                v,
                args.batch_size,
                with_steer=True,
            )
            tmp = f.with_suffix(".tmp")
            timed_call("torch.save", torch.save, x, tmp)
            tmp.replace(f)
        chunks.append(x)
        LOG.info("Large component discovery %s %s/%s", source, i + 1, len(ps))
    effects = {k: np.concatenate([x[k] for x in chunks], axis=0) for k in chunks[0]}
    scores = timed_call(
        "ER.component_scores",
        ER.component_scores,
        effects,
        np.array([b["correct_corrupt"] for _, b in disc]),
    )
    timed_call("np.savez", np.savez, d / "component_scores.npz", **scores)
    ent = timed_call(
        "ER.calibrate_and_select",
        ER.calibrate_and_select,
        engine,
        vs,
        scores,
        v,
        [k],
        [0, 0.5, 1, 2, 4, 8],
        [0.5, 1.0, 2.0, 4.0],
        args.large_random_controls,
        args.seed + 7,
        args.batch_size,
    )["circuits"][k]
    timed_call("torch.save", torch.save, ent, d / "component_selection.pt")
    from .format_controls import select_matched_alpha, evaluate_variant

    a, cal = select_matched_alpha(
        engine,
        run,
        source,
        v,
        np.asarray(ent["patch"]["harm_aware"]),
        [0.5, 1.0, 2.0, 4.0],
        args.batch_size,
        None,
    )
    write_json(
        d / "policy.json",
        dict(
            mode="component",
            discovery_n=len(ps),
            discovery_ids=[p["sample_id"] for p in ps],
            validation_n=len(vs),
            matched_alpha=a,
            calibration=cal,
            selection=ent,
        ),
    )
    for target in progress(TASKS, "scale evaluation targets"):
        for split in progress(args.splits, "scale evaluation splits"):
            exclude = timed_call(
                "run.cross_task_overlap", run.cross_task_overlap, source, target, split
            )
            write_json(d / f"excluded_{target}_{split}.json", sorted(exclude))
            for p, b in progress(
                timed_call("run.split", run.split, target, split, args.limit),
                "component evaluation samples",
            ):
                if p["sample_id"] in exclude:
                    continue

                def out(r):
                    emit({**r, "target": target, "split": split})

                evaluate_variant(
                    engine,
                    p,
                    "component_scale",
                    v,
                    ent,
                    a,
                    args.batch_size,
                    out,
                    0.5,
                    1.0,
                    extra_controls=True,
                )
