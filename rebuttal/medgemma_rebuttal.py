#!/usr/bin/env python3
"""MedGemma-only residual-layer scale and rebuttal extension. Run at repo root.
Uses existing rebuttal modules; no on-disk changes to their source or small-model results.
"""

from __future__ import annotations
import copy, hashlib, json, logging, os, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import rebuttal.run as runner
from rebuttal.core import (
    LOG,
    TASKS,
    Store,
    aggregate,
    choose_threshold,
    load_pt,
    stable_seed,
    write_json,
)

TAG = "medgemma_27b"
STAGES = ("audit", "scale", "format", "attenuation", "strength", "treatment", "probe")


def random_cpu(v, seed):
    """Deterministic CPU Gaussian; per-layer norm matches the learned vector."""
    cpu = v.detach().to(device="cpu", dtype=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(cpu.shape, generator=g, dtype=torch.float32, device="cpu")
    return (
        x
        * (
            cpu.norm(dim=-1, keepdim=True)
            / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        )
    ).to(v.device)


def atomic_pt(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def locate_checkpointed(engine, disc, args, d):
    """Exact same all-layer oracle patch statistic as large.locate_layers; 10-pair resume."""
    from patching_engine import Intervention
    from rebuttal.interventions import pair_states

    layers = list(range(engine.num_layers))
    chunks = d / "discovery_chunks"
    chunks.mkdir(exist_ok=True)
    frames = []
    for start in range(0, len(disc), 10):
        part = disc[start : start + 10]
        ids = [str(p["sample_id"]) for p, b in part]
        path = chunks / f"{start:06d}.json"
        if path.exists():
            obj = json.loads(path.read_text())
            if obj["ids"] != ids or obj["layers"] != layers:
                raise ValueError("Discovery checkpoint cohort changed")
            rows = obj["rows"]
        else:
            rows = []
            for p, b in part:
                states = pair_states(engine, p)
                so, sc = states["clean"], states["corrupt"]
                h = int(p["halluc_idx"])
                lc = so.cand_logits.numpy()
                lx = sc.cand_logits.numpy()
                den = float(lc[0] - lc[h] - lx[0] + lx[h])
                if den < 1e-6:
                    continue
                ivs = [
                    Intervention(resid_layers=np.array([l]), resid_alpha=1.0)
                    for l in layers
                ]
                out = engine.run_interventions(
                    sc,
                    ivs,
                    steer_resid=so.resid.float() - sc.resid.float(),
                    batch_size=args.batch_size,
                )
                for l, y in zip(layers, out):
                    rows.append(
                        dict(
                            sample_id=str(p["sample_id"]),
                            layer=l,
                            gain=float((y[0] - y[h] - lx[0] + lx[h]) / den),
                            correct_after=int(y.argmax()) == 0,
                        )
                    )
            write_json(path, dict(ids=ids, layers=layers, rows=rows))
        frames.extend(rows)
        LOG.info("MedGemma localization %s/%s pairs", start + len(part), len(disc))
    df = pd.DataFrame(frames)
    if df.empty:
        raise ValueError("No valid discovery contrasts")
    scores = df.groupby("layer").gain.mean()
    df.to_csv(d / "discovery_layer_patch.csv", index=False)
    return int(scores.idxmax()), scores.to_dict()


def fit_risk(run, source, engine):
    from experiment_runner import RiskRouter, default_router_layers, router_features

    feats = []
    wrong = []
    for p, b in run.split(source, "discovery"):
        for rt in ("clean", "corrupt"):
            feats.append(router_features(b, rt))
            wrong.append(not bool(b[f"correct_{rt}"]))
    if not feats:
        raise ValueError("Empty risk-router discovery set")
    return RiskRouter(default_router_layers(engine.num_layers), mode="vocab").fit(
        feats, np.asarray(wrong, bool)
    )


def make_policy(engine, run, source, args, d):
    from rebuttal.large import calibrate_layer
    from experiment_runner import router_features

    disc = run.split(source, "discovery", args.large_discovery_limit or None)
    val = run.split(source, "validation")
    if not disc or not val:
        raise ValueError("Insufficient source discovery or validation")
    layer, scores = locate_checkpointed(engine, disc, args, d)
    v = (
        torch.stack(
            [b["resid_clean"].float() - b["resid_corrupt"].float() for p, b in disc]
        )
        .mean(0)
        .cpu()
    )
    if not torch.isfinite(v).all() or float(v[layer].norm()) <= 0:
        raise ValueError("Invalid reference vector")
    atomic_pt(d / "direction.pt", v)
    safe, diagnostic = calibrate_layer(engine, val, layer, v, args, d)
    alpha = safe if safe > 0 else diagnostic
    # The diagnostic fallback is NOT certified. Do not silently skip all rebuttal controls.
    risk = fit_risk(run, source, engine)
    feats = {
        (str(p["sample_id"]), rt): router_features(b, rt)
        for p, b in val
        for rt in ("clean", "corrupt")
    }
    rows = pd.read_csv(d / "validation_predictions.csv", dtype={"sample_id": str})
    rows = rows[np.isclose(rows.alpha, alpha)].copy()
    if len(rows) != 2 * len(val):
        raise ValueError("Validation prediction cohort mismatch")
    risk_scores = risk.predict(
        [feats[(s, rt)] for s, rt in zip(rows.sample_id, rows.run_type)]
    )

    def bool_col(name):
        col = rows[name]
        if pd.api.types.is_bool_dtype(col):
            return col.to_numpy(bool)
        vals = (
            col.astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if vals.isna().any():
            raise ValueError(f"Invalid validation boolean column: {name}")
        return vals.to_numpy(bool)

    tau, grid = choose_threshold(
        risk_scores, bool_col("correct_base"), bool_col("correct_after"), args.harm_eps
    )
    pd.DataFrame(grid).to_csv(d / "base_error_threshold_grid.csv", index=False)
    policy = dict(
        mode="layer",
        source=source,
        layer=layer,
        layer_gains=scores,
        layer_candidates=list(range(engine.num_layers)),
        selected_alpha=safe,
        diagnostic_alpha=diagnostic,
        alpha=alpha,
        selected=True,
        safety_certified_on_validation=bool(safe > 0),
        policy_role="validation_safe" if safe > 0 else "diagnostic_not_certified",
        tau=float(tau),
        selection_tasks=[source],
        vec=source,
        target=source,
        discovery_ids=[str(p["sample_id"]) for p, b in disc],
        discovery_n=len(disc),
        router_discovery_n=len(run.split(source, "discovery")),
        validation_n=len(val),
    )
    write_json(d / "policy.json", policy)
    LOG.info(
        "MedGemma %s layer=%d safe=%g diagnostic=%g control alpha=%g (%s)",
        source,
        layer,
        safe,
        diagnostic,
        alpha,
        policy["policy_role"],
    )


class PolicyRun:
    """Delegate checkpoint access, but use the exact saved scale direction in every control."""

    def __init__(self, run, vectors):
        self.run = run
        self.vectors = vectors

    def __getattr__(self, name):
        return getattr(self.run, name)

    def layers(self, t):
        return self.vectors[t]


def evaluate_layer_variant(engine, p, variant, v, cfg, args, emit):
    from rebuttal.interventions import pair_states, residual, emit_pair
    from patching_engine import Intervention

    states = pair_states(engine, p)
    so, sc = states["clean"], states["corrupt"]
    lc = so.cand_logits.numpy()
    lx = sc.cand_logits.numpy()
    h = p.get("halluc_idx")
    h = int(lx[1:].argmax() + 1) if h is None else int(h)
    p = {**p, "halluc_idx": h}
    den = float(lc[0] - lc[h] - lx[0] + lx[h])
    valid = bool(lc.argmax() == 0 and lc[0] - lc[h] >= 0.5 and den >= 1.0)
    layer = int(cfg["layer"])
    specs = [
        (0, 0.0),
        (0, float(cfg["selected_alpha"])),
        (0, float(cfg["diagnostic_alpha"])),
    ]
    methods = ["baseline", "safe_policy", "diagnostic_learned"]
    for rt, st in states.items():
        out = residual(engine, st, layer, [v], specs, args.batch_size)
        names = list(methods)
        aa = [a for _, a in specs]
        if rt == "corrupt":
            patch = engine.run_interventions(
                sc,
                [Intervention(resid_layers=np.array([layer]), resid_alpha=1.0)],
                steer_resid=so.resid.float() - sc.resid.float(),
                batch_size=args.batch_size,
            )[0]
            out.append(patch)
            names.append("oracle_patch")
            aa.append(1.0)
        for name, a, y in zip(names, aa, out):
            base = st.cand_logits.numpy()
            gain = (
                float((y[0] - y[h] - base[0] + base[h]) / den)
                if valid and rt == "corrupt"
                else None
            )
            emit_pair(
                emit,
                p,
                rt,
                st,
                y,
                variant=variant,
                method=name,
                alpha=a,
                valid=valid,
                contrast=den,
                contrast_label=p["cand_labels"][h],
                gain=gain,
                layer=layer,
                cohort="all_constructed",
            )
    return valid


def run_layer_format(engine, run, source, target, split, cfg, args, d, emit):
    from rebuttal.format_controls import (
        HINTS,
        rebuild,
        permute,
        distractor,
        filler_pool,
        matched_summary,
    )

    v = run.layers(source)
    write_json(d / "layer_calibration.json", cfg)
    rows = run.split(target, split, args.limit)
    overlap = run.cross_task_overlap(source, target, split)
    write_json(d / "excluded_overlap.json", sorted(overlap))
    rows = [(p, b) for p, b in rows if p["sample_id"] not in overlap]
    if target == "FCT":
        pairs = [p for t in TASKS for p, _ in run.load(t).values()]
        excluded = {p["sample_id"] for p in pairs}
        qs = {
            re.sub(
                r"\s+",
                " ",
                p["clean_user"].split("Question: ", 1)[-1].split("\nOptions:", 1)[0],
            )
            .strip()
            .casefold()
            for p in pairs
        }
        subjects, pools = filler_pool(args.data_dir, excluded, qs)
    reject = []
    manifest = []
    counts = defaultdict(int)
    for i, (p, _) in enumerate(rows):
        n = len(p["cand_labels"])
        pos = counts[n] % n
        counts[n] += 1
        seed = stable_seed(args.seed, target, p["sample_id"])
        variants = [("original", p, None)]

        def add(name, fn):
            try:
                q, info = fn()
                variants.append((name, q, info))
            except ValueError as ex:
                reject.append(
                    dict(sample_id=p["sample_id"], variant=name, reason=str(ex))
                )

        if target == "FCT":
            for j, h in enumerate(HINTS):

                def paraphrase(h=h):
                    hint = p["meta"]["student_letter"]
                    clean = p["clean_user"]
                    return (
                        rebuild(
                            engine,
                            p,
                            clean,
                            clean + "\n" + h.format(letter=hint),
                            p["gold"],
                            hint,
                            "paraphrase",
                        ),
                        None,
                    )

                add(f"paraphrase_{j+1}", paraphrase)
            add("fct_position", lambda: permute(engine, p, pos, seed))
            try:
                q, r, fill, order = distractor(engine, p, pos, seed, subjects, pools)
                variants.extend(
                    [
                        ("matched_hint", q, order),
                        ("matched_distractor", r, dict(order=order, fillers=fill)),
                    ]
                )
            except ValueError as ex:
                reject.append(
                    dict(
                        sample_id=p["sample_id"],
                        variant="matched_hint/distractor",
                        reason=str(ex),
                    )
                )
        else:
            add(
                "nota_identity_reworded",
                lambda: permute(engine, p, pos, seed, nota=True, identity=True),
            )
            add("nota_position", lambda: permute(engine, p, pos, seed, nota=True))
        for name, q, info in variants:
            valid = evaluate_layer_variant(engine, q, name, v, cfg, args, emit)
            manifest.append(
                dict(
                    sample_id=q["sample_id"],
                    variant=name,
                    clean=q["clean_user"],
                    corrupt=q["corrupt_user"],
                    gold=q["gold"],
                    valid=valid,
                    construction=info,
                )
            )
        if (i + 1) % 10 == 0:
            LOG.info("Format %s→%s %s/%s", source, target, i + 1, len(rows))
    if not manifest:
        raise ValueError("No eligible format-control pairs after overlap/rejections")
    write_json(d / "prompt_manifest.json", manifest)
    pd.DataFrame(reject, columns=["sample_id", "variant", "reason"]).to_csv(
        d / "rejections.csv", index=False
    )
    matched_summary(d)


def scale_test(engine, run, source, target, split, cfg, args, d, emit):
    from rebuttal.interventions import pair_states, residual, emit_pair
    from patching_engine import Intervention
    from rebuttal.layer_controls import subset

    v = run.layers(source)
    layer = int(cfg["layer"])
    safe = float(cfg["selected_alpha"])
    diagnostic = float(cfg["diagnostic_alpha"])
    rand = random_cpu(v, stable_seed(args.seed, source, "large-random"))
    rows = subset(run, source, target, split, args.limit, d)
    if not rows:
        raise ValueError(f"Empty held-out condition: {source}/{target}/{split}")
    for i, (p, b) in enumerate(rows):
        states = pair_states(engine, p)
        so, sc = states["clean"], states["corrupt"]
        h = int(p["halluc_idx"])
        lc = so.cand_logits.numpy()
        lx = sc.cand_logits.numpy()
        den = float(lc[0] - lc[h] - lx[0] + lx[h])
        ok = bool(lc.argmax() == 0 and lc[0] - lc[h] >= 0.5 and den >= 1.0)
        for rt, st in states.items():
            specs = [(0, 0.0), (0, safe), (0, diagnostic), (1, diagnostic)]
            names = [
                "baseline",
                "safe_policy",
                "diagnostic_learned",
                "diagnostic_random",
            ]
            out = residual(engine, st, layer, [v, rand], specs, args.batch_size)
            if rt == "corrupt":
                out.append(
                    engine.run_interventions(
                        sc,
                        [Intervention(resid_layers=np.array([layer]), resid_alpha=1.0)],
                        steer_resid=so.resid.float() - sc.resid.float(),
                        batch_size=args.batch_size,
                    )[0]
                )
                names.append("oracle_residual_patch")
                specs.append((0, 1.0))
            for name, (_, a), y in zip(names, specs, out):
                base = st.cand_logits.numpy()
                gain = (
                    float((y[0] - y[h] - base[0] + base[h]) / den)
                    if ok and rt == "corrupt"
                    else None
                )
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
                    gain=gain,
                    variant="layer_pilot",
                )
        if (i + 1) % 10 == 0:
            LOG.info("Scale %s→%s %s/%s", source, target, i + 1, len(rows))


def main(argv=None):
    p = runner.parser()
    p.set_defaults(
        models=[TAG],
        seeds=[0],
        experiments=list(STAGES),
        output_dir="results/rebuttal_medgemma_full",
        quantization="nf4",
        attn_implementation="sdpa",
        large_all_layers=True,
        large_discovery_limit=100,
        splits=["test"],
        batch_size=16,
    )
    args = p.parse_args(argv)
    root = Path(args.output_dir).resolve()
    if args.aggregate_only:
        aggregate(root)
        return 0
    if (
        args.models != [TAG]
        or args.seeds != [0]
        or args.quantization != "nf4"
        or args.large_mode != "layer"
        or not args.large_all_layers
    ):
        p.error(
            "This extension is MedGemma-27B only, NF4, seed 0, all-layer residual discovery."
        )
    if set(args.sources) != set(TASKS) or set(args.targets) != set(TASKS):
        p.error("Use both FCT and NOTA as sources and targets.")
    if args.reference_soft_dir:
        p.error(
            "No original MedGemma soft-gate cohort exists; use --external-samples-json instead."
        )
    if (
        min(
            args.batch_size,
            args.large_n_samples,
            args.large_max_pairs,
            args.random_reps,
            args.shuffle_reps,
            args.n_medqa,
            args.n_pubmedqa,
        )
        <= 0
        or args.large_discovery_limit < 0
    ):
        p.error("Invalid count")
    if (
        args.limit is not None
        and args.limit <= 0
        or args.train_limit is not None
        and args.train_limit <= 0
    ):
        p.error("Limits must be positive")
    if (
        0 not in args.alphas
        or any(a < 0 or not np.isfinite(a) for a in args.alphas)
        or any(a <= 0 or not np.isfinite(a) for a in args.rhos + args.gate_caps)
        or not 0 < args.harm_eps < 1
    ):
        p.error("Invalid strength or safety grid")
    args.alphas = sorted(set(args.alphas))
    args.rhos = sorted(set(args.rhos))
    args.gate_caps = sorted(set(args.gate_caps))
    args.data_dir = str(Path(args.data_dir).resolve())
    args.results_dir = str(Path(args.results_dir).resolve())
    args.seed = 0
    wanted = set(args.experiments)
    if args.plan:
        print("MedGemma-27B NF4 / seed 0 / residual layer suite")
        print("Output:", root)
        print("Stages:", [s for s in STAGES if s in wanted])
        print(
            "Requested per task:",
            args.large_n_samples,
            "discovery cap:",
            args.large_discovery_limit,
            "batch:",
            args.batch_size,
        )
        return 0
    root.mkdir(parents=True, exist_ok=True)
    # Own this isolated process's logger; close any previous handlers on re-entry.
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    LOG.propagate = False
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(root / "rebuttal.log", encoding="utf8"),
    ):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)
    dest = root / f"{TAG}_seed0"
    fingerprint = runner.code_fingerprint()
    fingerprint[Path(__file__).name] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    store = Store(
        dest,
        dict(
            **runner.settings(args),
            model=TAG,
            code=fingerprint,
            design="medgemma_residual_full_v1",
        ),
        args.resume,
    )
    # Always use the same frozen model revision when explicitly supplied. Record runtime model commit.
    write_json(dest / "environment.json", runner.versions())
    LOG.info("Arguments %s", vars(args))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for fresh MedGemma checkpoint creation and inference"
        )
    torch.set_default_device("cpu")
    engine = runner.engine_for(TAG, args, None)
    runtime = dict(
        model_name=engine.model_name,
        model_commit=getattr(engine.model.config, "_commit_hash", None),
        device_map=getattr(engine.model, "hf_device_map", None),
        layers=engine.num_layers,
        heads=engine.num_heads,
        hidden_size=engine.hidden_size,
        dtype=str(engine.dtype),
        quantization=getattr(engine, "quantization_metadata", {}),
    )
    old = dest / "model_runtime.json"
    if (
        old.exists()
        and json.loads(old.read_text()).get("model_commit") != runtime["model_commit"]
    ):
        raise ValueError(
            "Model revision changed; use the original --revision or a new output root"
        )
    write_json(old, runtime)
    from rebuttal.large import prepare_new_run
    from rebuttal import layer_controls as lc
    from rebuttal.interventions import residual

    run = prepare_new_run(engine, args, dest / "new_checkpoints", runner.MODELS[TAG])
    meta = dict(model=TAG, seed=0)
    store.condition("audit", lambda d, e: run.audit(d), experiment="audit", **meta)
    if wanted == {"audit"}:
        aggregate(root)
        return 0
    # Generate each policy exactly once, independently of downstream experiments.
    policies = {}
    vectors = {}
    for source in TASKS:
        store.condition(
            f"policy/{source}",
            lambda d, e: make_policy(engine, run, source, args, d),
            experiment="policy_layer",
            source=source,
            **meta,
        )
        d = dest / "policy" / source
        policies[source] = json.loads((d / "policy.json").read_text())
        vectors[source] = load_pt(d / "direction.pt").cpu()
    view = PolicyRun(run, vectors)
    pd.DataFrame(
        [
            dict(
                source=s,
                layer=c["layer"],
                selected_alpha=c["selected_alpha"],
                diagnostic_alpha=c["diagnostic_alpha"],
                control_alpha=c["alpha"],
                policy_role=c["policy_role"],
                safety_certified_on_validation=c["safety_certified_on_validation"],
                router_discovery_n=c["router_discovery_n"],
                validation_n=c["validation_n"],
            )
            for s, c in policies.items()
        ]
    ).to_csv(dest / "policy_status.csv", index=False)
    # Scoped process-local adapters: use fresh same-model policies instead of legacy CSVs.
    # Reuse the existing strength, attenuation, treatment and probe implementations unchanged on disk.
    risks = {}

    def fit_adapter(path, cfg):
        if Path(path).resolve() != run.path.resolve():
            raise ValueError("Unexpected router checkpoint path")
        source = cfg["vec"]
        if cfg["selection_tasks"] != [source]:
            raise ValueError("Expected source-only router")
        if source not in risks:
            risks[source] = fit_risk(run, source, engine)
        risk = risks[source]
        risk.tau = float(cfg["tau"])
        return vectors[source], risk

    lc.random_vector = random_cpu
    lc.fit_from_discovery = fit_adapter
    samples = None
    for stage in STAGES[1:]:
        if stage not in wanted:
            continue
        LOG.info("START STAGE %s", stage)
        t0 = time.monotonic()
        if stage == "attenuation" or stage == "treatment" and args.treatment_external:
            if samples is None:
                external = copy.copy(args)
                if (
                    root / "external_samples.json"
                ).exists() and not args.external_samples_json:
                    external.external_samples_json = str(root / "external_samples.json")
                samples = runner.load_external(external, root)
        for source in TASKS:
            cfg = policies[source]
            base_meta = dict(source=source, policy_role=cfg["policy_role"], **meta)
            if stage in ("scale", "format", "strength"):
                fn = {
                    "scale": scale_test,
                    "format": run_layer_format,
                    "strength": lc.run_strength,
                }[stage]
                for target in TASKS:
                    for split in args.splits:
                        store.condition(
                            f"{stage}/{source}_{target}_{split}",
                            lambda d, e: fn(
                                engine, view, source, target, split, cfg, args, d, e
                            ),
                            experiment="scale_layer" if stage == "scale" else stage,
                            target=target,
                            split=split,
                            **base_meta,
                        )
            elif stage == "attenuation":
                heldout = next(t for t in TASKS if t != source)
                ext_cfg = {
                    **cfg,
                    "target": heldout,
                    "protocol": "source_only_loto_no_guard_tasks",
                }
                for ds, ss in samples.items():
                    store.condition(
                        f"attenuation/loto_{heldout}_{ds}",
                        lambda d, e: lc.run_attenuation(
                            engine, view, ext_cfg, ss, args, d, e
                        ),
                        experiment="attenuation",
                        target=ds,
                        heldout=heldout,
                        split="external",
                        **base_meta,
                    )
            elif stage == "treatment":

                def treatment(d, emit):
                    fit = lc.fit_benefit(engine, view, source, cfg, args, d)
                    for target in TASKS:
                        for split in args.splits:
                            sd = d / f"{target}_{split}"
                            sd.mkdir(exist_ok=True)
                            lc.run_benefit(
                                engine,
                                view,
                                source,
                                target,
                                split,
                                cfg,
                                fit,
                                args,
                                sd,
                                lambda r: emit({**r, "target": target, "split": split}),
                            )
                    if args.treatment_external:
                        for ds, ss in samples.items():
                            sd = d / ds
                            sd.mkdir(exist_ok=True)
                            lc.run_benefit_external(
                                engine,
                                cfg,
                                fit,
                                ss,
                                args,
                                sd,
                                lambda r: emit(
                                    {**r, "target": ds, "split": "external"}
                                ),
                            )
                    lc.treatment_diagnostics(d)

                store.condition(
                    f"treatment/{source}",
                    treatment,
                    experiment="treatment",
                    **base_meta,
                )
            elif stage == "probe":

                def probe(d, emit):
                    fit = lc.calibrate_probe(engine, view, source, cfg, args, d)
                    for target in TASKS:
                        for split in args.splits:
                            sd = d / f"{target}_{split}"
                            sd.mkdir(exist_ok=True)
                            lc.run_probe(
                                engine,
                                view,
                                source,
                                target,
                                split,
                                cfg,
                                fit,
                                args,
                                sd,
                                lambda r: emit({**r, "target": target, "split": split}),
                            )

                store.condition(
                    f"probe/{source}", probe, experiment="probe", **base_meta
                )
        aggregate(root)
        LOG.info("DONE STAGE %s %.1fs", stage, time.monotonic() - t0)
    aggregate(root)
    write_json(root / "failures.json", [])
    LOG.info("All requested MedGemma stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
