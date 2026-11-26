"""Residual-strength, external attenuation, benefit gating, and linear probe controls."""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from experiment_runner import (
    RiskRouter,
    default_router_layers,
    router_features,
    run_gold,
)
from eval_downstream_all_models import prepare_choices, fit_from_discovery
from softmax_gate_adapter import SoftmaxGateConfig, two_class_softmax
from .core import (
    LOG,
    ALPHAS,
    RHOS,
    stable_seed,
    row_for,
    metrics,
    matched_weights,
    choose_threshold,
    write_json,
)
from .interventions import (
    residual,
    random_vector,
    sweep_pair,
    pair_states,
    emit_pair,
    state_features,
)
from ._timing import timed, timed_call, progress


@timed
def subset(run, source, target, split, limit, d):
    excluded = timed_call(
        "run.cross_task_overlap", run.cross_task_overlap, source, target, split
    )
    write_json(d / "excluded_overlap.json", sorted(excluded))
    return [
        (p, b)
        for p, b in timed_call("run.split", run.split, target, split, limit)
        if p["sample_id"] not in excluded
    ]


@timed
def run_strength(engine, run, source, target, split, cfg, args, d, emit):
    layer = int(cfg["layer"])
    a = float(cfg["alpha"])
    v = timed_call("run.layers", run.layers, source)
    seeds = [
        stable_seed(args.seed, source, "random-direction", i)
        for i in range(args.random_reps)
    ]
    rand = [random_vector(v, s) for s in seeds]
    write_json(
        d / "vectors.json",
        dict(
            selection=cfg,
            random_seeds=seeds,
            layer=layer,
            vector_norm=float(v[layer].norm()),
            alphas=args.alphas,
            rhos=args.rhos,
        ),
    )
    rows = subset(run, source, target, split, args.limit, d)
    for i, (p, b) in enumerate(progress(rows, "strength samples")):
        sweep_pair(
            engine, p, layer, v, rand, args.alphas, args.rhos, a, args.batch_size, emit
        )
        if (i + 1) % 10 == 0:
            LOG.info("Strength %s→%s %s/%s", source, target, i + 1, len(rows))


@timed
def external_features(engine, state, ids, prefix):
    if prefix:
        vs, res = timed_call("engine.prompt_features", engine.prompt_features, ids)
    else:
        vs, res = state.vocab_stats, state.resid
    return dict(
        vocab_stats=np.asarray(vs),
        resid=res.float().cpu(),
        cand_logits=state.cand_logits.numpy(),
    )


@timed
def run_attenuation(engine, run, cfg, samples, args, d, emit):
    """Two passes: calculate label-free per-condition weights on tokenizable samples;
    re-score same prompts with constant/shuffled original weights and soft/always baselines.
    No all-condition global mean; rejected rows excluded before matching.
    """
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    v, router = timed_call("fit_from_discovery", fit_from_discovery, run.path, cfg)
    gate = SoftmaxGateConfig(temperature=args.temperature, center=args.center)
    accepted = []
    rejected = []
    weights = []
    risks = []
    bases = []
    for i, s in enumerate(progress(samples, "attenuation weight pass")):
        try:
            st, ids, prefix = timed_call("prepare_choices", prepare_choices, engine, s)
        except ValueError as ex:
            rejected.append(dict(dataset_index=s["dataset_index"], reason=str(ex)))
            continue
        r = float(router.predict([external_features(engine, st, ids, prefix)])[0])
        w = float(two_class_softmax([r], gate)[0, 1])
        accepted.append(s)
        risks.append(r)
        weights.append(w)
        bases.append(st.cand_logits.numpy().copy())
        if (i + 1) % 25 == 0:
            LOG.info("Attenuation weight pass %s/%s", i + 1, len(samples))
    timed_call(
        "to_csv",
        pd.DataFrame(rejected, columns=["dataset_index", "reason"]).to_csv,
        d / "rejections.csv",
        index=False,
    )
    if not accepted:
        raise ValueError("No tokenizable external samples")
    if args.reference_soft_dir:
        from pathlib import Path

        dataset = d.name.rsplit("_", 1)[-1]
        tag = run.path.name
        pattern = f'{tag}_loto_{cfg["target"]}_{dataset}_softmax_predictions.csv'
        files = list(Path(args.reference_soft_dir).rglob(pattern))
        if len(files) != 1:
            raise ValueError(
                f"Expected one original softmax prediction CSV {pattern}; found {len(files)}"
            )
        ref = timed_call("pd.read_csv", pd.read_csv, files[0])
        ids = [s["dataset_index"] for s in accepted]
        if ref.dataset_index.duplicated().any() or set(ref.dataset_index) != set(ids):
            raise ValueError(f"Accepted external IDs differ from original: {pattern}")
        ref = ref.set_index("dataset_index").loc[ids]
        if list(ref.gold.astype(str)) != [str(s["gold"]) for s in accepted]:
            raise ValueError("Gold labels differ from original external cohort")
        write_json(
            d / "reference_check.json",
            dict(
                file=str(files[0]),
                same_ids=True,
                n=len(ids),
                mean_abs_gate_probability_difference=float(
                    np.mean(np.abs(ref.gate_on_prob.to_numpy() - weights))
                ),
                max_abs_gate_probability_difference=float(
                    np.max(np.abs(ref.gate_on_prob.to_numpy() - weights))
                ),
            ),
        )
    else:
        ref = None
    seed = stable_seed(args.seed, cfg["target"], "attenuation", args.sample_seed)
    caps = [("uncapped", alpha)] + [
        (f"cap_{c:g}", min(alpha, c)) for c in args.gate_caps
    ]
    strategies = []
    for name, cap in caps:
        ww = np.asarray(weights) * cap
        for method, rep, w, perm in matched_weights(ww, args.shuffle_reps, seed):
            strategies.append((name, method, rep, w, perm))
    write_json(
        d / "sample_weights.json",
        dict(
            selection=cfg,
            temperature=args.temperature,
            center=args.center,
            seed=seed,
            requested_ids=[s["dataset_index"] for s in samples],
            accepted_ids=[s["dataset_index"] for s in accepted],
            weights=weights,
            risks=risks,
            strategies=[
                dict(
                    cap=c,
                    method=m,
                    rep=r,
                    mean=float(w.mean()),
                    weights=w,
                    permutation=perm,
                )
                for c, m, r, w, perm in strategies
            ],
        ),
    )
    for i, s in enumerate(progress(accepted, "attenuation intervention pass")):
        st, ids, prefix = timed_call("prepare_choices", prepare_choices, engine, s)
        b = st.cand_logits.numpy()
        gold = s["answers"].index(s["gold"])
        hnorm = float(st.resid[layer].float().norm())
        vn = float(v[layer].norm())
        specs = [(0, 0.0), (0, alpha)] + [
            (0, float(w[i])) for _, _, _, w, _ in strategies
        ]
        logits = residual(engine, st, layer, [v], specs, args.batch_size)
        base_drift = float(np.max(np.abs(b - bases[i])))
        changed = bool(b.argmax() != bases[i].argmax())
        common = dict(
            sample_id=str(s["dataset_index"]),
            dataset_index=s["dataset_index"],
            layer=layer,
            gate_risk=risks[i],
            p_on=weights[i],
            selected_alpha=alpha,
            baseline_logit_drift=base_drift,
            baseline_prediction_drift=changed,
            h_norm=hnorm,
            v_norm=vn,
        )
        if ref is not None:
            common.update(
                original_pred_base=str(ref.iloc[i].pred_base),
                original_pred_soft=str(ref.iloc[i].pred_soft),
                original_pred_always=str(ref.iloc[i].pred_always),
                original_gate_probability=float(ref.iloc[i].gate_on_prob),
            )
        for method, a, after in zip(
            ("baseline", "always_on"), (0.0, alpha), logits[:2]
        ):
            emit(
                row_for(
                    b,
                    after,
                    s["answers"],
                    gold,
                    **common,
                    method=method,
                    variant="uncapped",
                    rep=-1,
                    alpha_effective=a,
                    relative_edit=a * vn / hnorm if hnorm else None,
                )
            )
        for (cap, method, rep, w, perm), after in zip(strategies, logits[2:]):
            a = float(w[i])
            emit(
                row_for(
                    b,
                    after,
                    s["answers"],
                    gold,
                    **common,
                    method=method,
                    variant=cap,
                    rep=rep,
                    alpha_effective=a,
                    relative_edit=a * vn / hnorm if hnorm else None,
                    weight_source_id=(
                        accepted[int(perm[i])]["dataset_index"]
                        if perm[i] >= 0
                        else None
                    ),
                )
            )
        if (i + 1) % 25 == 0:
            LOG.info("Attenuation intervention pass %s/%s", i + 1, len(accepted))
    # Paired per-question difference soft vs constant/shuffled, not independent group tests.
    df = timed_call("pd.read_json", pd.read_json, d / "predictions.jsonl", lines=True)
    results = []
    for cap, g in df.groupby("variant"):
        soft = g[g.method == "soft"].set_index("dataset_index")
        for (method, rep), h in g[
            g.method.isin(["constant_mean", "shuffled_soft", "always_on"])
        ].groupby(["method", "rep"]):
            h = h.set_index("dataset_index").loc[soft.index]
            m = metrics(
                h.correct_after, soft.correct_after, h.pred_after, soft.pred_after
            )
            m.update(variant=cap, comparison=f"soft_vs_{method}", rep=rep)
            results.append(m)
    timed_call(
        "to_csv",
        pd.DataFrame(results).to_csv,
        d / "paired_soft_comparisons.csv",
        index=False,
    )


@timed
def collect_fixed(engine, examples, layer, v, alpha, batch_size, limit=None):
    records = []
    for i, (p, b) in enumerate(
        progress(examples[:limit] if limit else examples, "fixed-policy samples")
    ):
        for rt, st in pair_states(engine, p).items():
            out = residual(engine, st, layer, [v], [(0, alpha)], batch_size)[0]
            gold = run_gold(p, rt)
            logits = st.cand_logits.numpy()
            # Match original candidate-free prompt features; recomputed baseline labels for same forward path.
            feat = state_features(engine, st, p, rt)
            records.append(
                dict(
                    p=p,
                    rt=rt,
                    feat=feat,
                    base=logits,
                    after=out,
                    correct_base=logits.argmax() == gold,
                    correct_after=out.argmax() == gold,
                )
            )
        if (i + 1) % 25 == 0:
            LOG.info(
                "Fixed-policy outcome labels %s/%s",
                i + 1,
                len(examples[:limit] if limit else examples),
            )
    return records


class BenefitRouter:
    """Use the original PCA/vocabulary feature map, predict rescue, induction, no change."""

    @timed
    def fit(self, feats, y, n_layers):
        self.feature = RiskRouter(default_router_layers(n_layers), mode="vocab").fit(
            feats, np.asarray(y) == 1
        )
        self.classes = np.unique(y)
        X = (self.feature._raw(feats) - self.feature.fm) / self.feature.fs
        self.clf = (
            LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(X, y)
            if len(self.classes) > 1
            else None
        )
        return self

    def predict(self, feats):
        X = (self.feature._raw(feats) - self.feature.fm) / self.feature.fs
        p = np.zeros((len(feats), 3))
        if self.clf is None:
            p[:, self.classes[0]] = 1.0
        else:
            p[:, self.clf.classes_.astype(int)] = self.clf.predict_proba(X)
        return p

    def state(self):
        return dict(
            feature=self.feature.state_dict(),
            classes=self.classes,
            coef=self.clf.coef_ if self.clf else None,
            intercept=self.clf.intercept_ if self.clf else None,
        )


def treatment_label(b, a):
    return 1 if not b and a else 2 if b and not a else 0


@timed
def fit_benefit(engine, run, source, cfg, args, d):
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    v = timed_call("run.layers", run.layers, source)
    # Source discovery used for vector AND outcome fitting; original validation already selects
    # policy, and is reused for gate threshold. Independent test/unfiltered untouched.
    # No target calibration. Clearly a source-only exploratory comparison, not independent policy certification.
    extra_n = getattr(args, "treatment_extra_pairs", 0)
    train = timed_call("run.split", run.split, source, "discovery")
    fit_split = "discovery"
    if extra_n:
        # Optional stricter fit set: fresh source questions disjoint from both original tasks.
        import random, re
        from run_casual_transfer import load_raw, TASKS, build_mcq_pairs

        originals = [
            p
            for t in ("FCT", "NOTA")
            for p, _ in timed_call("run.load", run.load, t).values()
        ]
        ids = {p["sample_id"] for p in originals}
        norm = lambda q: re.sub(r"\s+", " ", str(q)).strip().casefold()
        qs = {
            norm(p["clean_user"].split("Question: ", 1)[-1].split("\nOptions:", 1)[0])
            for p in originals
        }
        raw = timed_call("load_raw", load_raw, TASKS[source], "hf", str(args.data_dir))
        raw = [
            r
            for r in raw
            if str(r.get("id")) not in ids and norm(r.get("question")) not in qs
        ]
        stats = {}
        fresh = timed_call(
            "build_mcq_pairs",
            build_mcq_pairs,
            engine,
            source,
            raw,
            extra_n,
            random.Random(stable_seed(args.seed, source, "fresh-benefit")),
            stats,
        )
        if len(fresh) != extra_n:
            raise ValueError(
                f"Requested {extra_n} disjoint source pairs; built {len(fresh)}"
            )
        # NOTA contrast label is not required by residual outcome labelling; correctness gold index is 0.
        train = [(p, None) for p in fresh]
        fit_split = "fresh_source_disjoint"
        write_json(
            d / "fresh_source_manifest.json",
            dict(
                pairs=[
                    dict(
                        sample_id=p["sample_id"],
                        clean=p["clean_user"],
                        corrupt=p["corrupt_user"],
                        gold=p["gold"],
                    )
                    for p in fresh
                ],
                construction_stats=stats,
            ),
        )
    disc = collect_fixed(
        engine, train, layer, v, alpha, args.batch_size, args.train_limit
    )
    val = collect_fixed(
        engine,
        timed_call("run.split", run.split, source, "validation"),
        layer,
        v,
        alpha,
        args.batch_size,
    )
    if not disc or not val:
        raise ValueError("Missing source discovery/validation examples")
    feats = [r["feat"] for r in disc]
    y = np.array([treatment_label(r["correct_base"], r["correct_after"]) for r in disc])
    model = BenefitRouter().fit(feats, y, engine.num_layers)
    _, risk = timed_call(
        "fit_from_discovery", fit_from_discovery, run.path, cfg
    )  # reconstruct the original full-discovery base-error router exactly
    prob = model.predict([r["feat"] for r in val])
    u = prob[:, 1] - 2 * prob[:, 2]
    b = np.array([r["correct_base"] for r in val])
    a = np.array([r["correct_after"] for r in val])
    tau, grid = choose_threshold(u, b, a, args.harm_eps)
    timed_call(
        "torch.save",
        torch.save,
        dict(
            benefit=model.state(),
            base_error=risk.state_dict(),
            threshold=tau,
            layer=layer,
            alpha=alpha,
        ),
        d / "routers.pt",
    )
    timed_call(
        "to_csv", pd.DataFrame(grid).to_csv, d / "threshold_grid.csv", index=False
    )
    for name, records in [(fit_split, disc), ("validation", val)]:
        pp = model.predict([r["feat"] for r in records])
        timed_call(
            "to_csv",
            pd.DataFrame(
                [
                    dict(
                        sample_id=r["p"]["sample_id"],
                        run_type=r["rt"],
                        base_correct=r["correct_base"],
                        after_correct=r["correct_after"],
                        treatment_label=treatment_label(
                            r["correct_base"], r["correct_after"]
                        ),
                        p_nochange=x[0],
                        p_rescue=x[1],
                        p_induction=x[2],
                        utility=x[1] - 2 * x[2],
                    )
                    for r, x in zip(records, pp)
                ]
            ).to_csv,
            d / f"{name}_outcomes.csv",
            index=False,
        )
    write_json(
        d / "training.json",
        dict(
            source=source,
            fit_split=fit_split,
            threshold_split="validation",
            threshold=tau,
            label_counts=np.bincount(y, minlength=3),
            policy=cfg,
            warning=(
                "Fresh source outcome fit disjoint from original pair IDs/questions; validation reused for policy selection and threshold."
                if extra_n
                else "Discovery reused for source vector; validation reused for original policy selection and gate threshold; held-out evaluation independent."
            ),
        ),
    )
    return v, model, risk, tau


def emit_treatment(emit, b, a, labels, gold, prob, risk, tau, old_tau, **kw):
    base_ok = b.argmax() == gold
    after_ok = a.argmax() == gold
    u = float(prob[1] - 2 * prob[2])
    methods = {
        "baseline": False,
        "always_on": True,
        "base_error_gate": risk > old_tau,
        "treatment_gate": u > tau,
        "base_error_oracle": not base_ok,
        "benefit_oracle": not base_ok and after_ok,
    }
    for name, use in methods.items():
        emit(
            row_for(
                b,
                a if use else b,
                labels,
                gold,
                **kw,
                method=name,
                routed=bool(use),
                p_nochange=float(prob[0]),
                p_rescue=float(prob[1]),
                p_induction=float(prob[2]),
                utility=u,
                base_error_risk=float(risk),
                threshold=tau,
                always_correct=bool(after_ok),
            )
        )


@timed
def run_benefit(engine, run, source, target, split, cfg, fit, args, d, emit):
    v, model, risk, tau = fit
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    rows = subset(run, source, target, split, args.limit, d)
    for p, b in progress(rows, "treatment samples"):
        for rt, st in pair_states(engine, p).items():
            feat = state_features(engine, st, p, rt)
            a = residual(engine, st, layer, [v], [(0, alpha)], args.batch_size)[0]
            emit_treatment(
                emit,
                st.cand_logits.numpy(),
                a,
                p["cand_labels"],
                run_gold(p, rt),
                model.predict([feat])[0],
                risk.predict([feat])[0],
                tau,
                risk.tau,
                sample_id=p["sample_id"],
                run_type=rt,
                layer=layer,
                alpha=alpha,
            )


@timed
def treatment_diagnostics(d):
    df = timed_call("pd.read_json", pd.read_json, d / "predictions.jsonl", lines=True)
    df = df[df.method == "treatment_gate"]
    out = []
    for (target, split), g in df.groupby(["target", "split"]):
        rescue = (~g.correct_base) & g.always_correct
        induce = g.correct_base & (~g.always_correct)
        auc = lambda y, p: float(roc_auc_score(y, p)) if y.nunique() > 1 else None
        out.append(
            dict(
                target=target,
                split=split,
                n=len(g),
                rescue_auroc=auc(rescue, g.p_rescue),
                induction_auroc=auc(induce, g.p_induction),
                coverage=g.routed.mean(),
                true_rescue=rescue.sum(),
                true_induction=induce.sum(),
            )
        )
    timed_call(
        "to_csv", pd.DataFrame(out).to_csv, d / "treatment_diagnostics.csv", index=False
    )


@timed
def run_benefit_external(engine, cfg, fit, samples, args, d, emit):
    v, model, risk, tau = fit
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    rejected = []
    for s in progress(samples, "external treatment samples"):
        try:
            st, ids, pre = timed_call("prepare_choices", prepare_choices, engine, s)
        except ValueError as ex:
            rejected.append(dict(dataset_index=s["dataset_index"], reason=str(ex)))
            continue
        feat = external_features(engine, st, ids, pre)
        a = residual(engine, st, layer, [v], [(0, alpha)], args.batch_size)[0]
        emit_treatment(
            emit,
            st.cand_logits.numpy(),
            a,
            s["answers"],
            s["answers"].index(s["gold"]),
            model.predict([feat])[0],
            risk.predict([feat])[0],
            tau,
            risk.tau,
            sample_id=str(s["dataset_index"]),
            layer=layer,
            alpha=alpha,
        )
    timed_call(
        "to_csv",
        pd.DataFrame(rejected, columns=["dataset_index", "reason"]).to_csv,
        d / "rejections.csv",
        index=False,
    )


@timed
def probe_direction(run, source, layer, v, args, d):
    data = timed_call("run.split", run.split, source, "discovery")
    X = []
    y = []
    for p, b in data:
        for rt in ("clean", "corrupt"):
            X.append(b[f"resid_{rt}"][layer].float().numpy())
            y.append(int(b[f"correct_{rt}"]))
    X = np.stack(X)
    y = np.array(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Probe requires both correctness classes in source discovery")
    # ITI-inspired linear direction; same selected residual layer rather than head-wise ITI.
    scale = StandardScaler().fit(X)
    clf = LogisticRegression(C=args.probe_c, max_iter=1500, random_state=args.seed).fit(
        scale.transform(X), y
    )
    raw = clf.coef_[0] / scale.scale_
    w = torch.from_numpy(raw).float()
    norm = float(w.norm())
    vn = float(v[layer].norm())
    if norm <= 0 or vn <= 0:
        raise ValueError("Zero norm probe or reference direction")
    probe = torch.zeros_like(v)
    probe[layer] = w * (vn / norm)
    cosine = float(torch.nn.functional.cosine_similarity(v[layer], probe[layer], dim=0))
    write_json(
        d / "probe_training.json",
        dict(
            n=len(y),
            positive=int(y.sum()),
            negative=int(len(y) - y.sum()),
            layer=layer,
            C=args.probe_c,
            train_accuracy=float(clf.score(scale.transform(X), y)),
            cosine_to_mean=cosine,
            vector_norm=vn,
            probe_norm=float(probe[layer].norm()),
        ),
    )
    timed_call(
        "torch.save",
        torch.save,
        dict(
            direction=probe,
            mean=scale.mean_,
            scale=scale.scale_,
            coef_standardized=clf.coef_,
            coef_activation=raw,
            intercept=clf.intercept_,
        ),
        d / "probe.pt",
    )
    return probe


@timed
def calibrate_probe(engine, run, source, cfg, args, d):
    v = timed_call("run.layers", run.layers, source)
    layer = int(cfg["layer"])
    probe = probe_direction(run, source, layer, v, args, d)
    rand = random_vector(v, stable_seed(args.seed, source, "probe_random"))
    vectors = [v, probe, rand]
    names = ["mean_difference", "linear_probe", "random"]
    alphas = args.alphas
    specs = [(i, a) for i in range(3) for a in alphas]
    rows = []
    for p, b in progress(
        timed_call("run.split", run.split, source, "validation"),
        "probe validation samples",
    ):
        for rt, st in pair_states(engine, p).items():
            g = run_gold(p, rt)
            base = st.cand_logits.numpy()
            after = residual(engine, st, layer, vectors, specs, args.batch_size)
            for (i, a), x in zip(specs, after):
                rows.append(
                    dict(
                        method=names[i],
                        alpha=a,
                        sample_id=p["sample_id"],
                        run_type=rt,
                        correct_base=base.argmax() == g,
                        correct_after=x.argmax() == g,
                    )
                )
    df = pd.DataFrame(rows)
    timed_call("to_csv", df.to_csv, d / "validation_predictions.csv", index=False)
    vals = []
    chosen = {}
    for (m, a), g in df.groupby(["method", "alpha"]):
        vals.append(dict(method=m, alpha=a, **metrics(g.correct_base, g.correct_after)))
    s = pd.DataFrame(vals)
    timed_call("to_csv", s.to_csv, d / "validation_metrics.csv", index=False)
    for name in names:
        cand = s[
            (s.method == name)
            & (s.induction_ucb <= args.harm_eps)
            & (s.safety_score > 0)
        ].sort_values(["safety_score", "alpha"], ascending=[False, True])
        chosen[name] = float(cand.iloc[0].alpha) if len(cand) else 0.0
    write_json(d / "chosen_alphas.json", chosen)
    return vectors, names, chosen


@timed
def run_probe(engine, run, source, target, split, cfg, fit, args, d, emit):
    vectors, names, chosen = fit
    layer = int(cfg["layer"])
    fixed = float(cfg["alpha"])
    specs = [(i, fixed) for i in range(3)] + [
        (i, chosen[m]) for i, m in enumerate(names)
    ]
    for p, b in progress(
        subset(run, source, target, split, args.limit, d), "probe evaluation samples"
    ):
        for rt, st in pair_states(engine, p).items():
            for j, ((i, a), out) in enumerate(
                zip(specs, residual(engine, st, layer, vectors, specs, args.batch_size))
            ):
                hn = float(st.resid[layer].float().norm())
                vn = float(vectors[i][layer].norm())
                emit_pair(
                    emit,
                    p,
                    rt,
                    st,
                    out,
                    method=names[i],
                    variant="same_alpha" if j < 3 else "source_calibrated",
                    alpha=a,
                    layer=layer,
                    h_norm=hn,
                    v_norm=vn,
                    relative_edit=a * vn / hn if hn else None,
                )
