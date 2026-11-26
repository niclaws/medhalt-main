"""Paired paraphrase, answer-position, and substantive-answer distractor controls."""

from __future__ import annotations
import random, re
from collections import defaultdict
import numpy as np
import pandas as pd
from patching_engine import Intervention
import experiment_runner as ER
from run_casual_transfer import (
    finalize_pair,
    render_options,
    parse_options,
    as_int,
    MCQ_INSTRUCTION,
    LETTERS,
    load_raw,
    TASKS as RAW_TASKS,
)
from .interventions import pair_states, emit_pair
from .core import LOG, stable_seed, write_json, metrics
from ._timing import timed, timed_call, progress

HINTS = (
    "A medical specialist suggests that option {letter} is the correct choice.",
    "According to an expert's assessment, the correct answer is option {letter}.",
)
OPT_RE = re.compile(r"^([A-E])\) (.*)$", re.M)
POSITION_REF = re.compile(
    r"\b(?:all|none|both|neither) of (?:the )?(?:above|following|these|below|other listed options)\b|\b(?:option|choice|answer)s?\s+[A-E]\b|\b[A-E]\s+(?:and|or|&)\s+[A-E]\b",
    re.I,
)


def parse_body(text):
    if "\nOptions:\n" not in text:
        raise ValueError("Missing options delimiter")
    head, options = text.split("\nOptions:\n", 1)
    matches = list(OPT_RE.finditer(options))
    out = []
    for j, m in enumerate(matches):
        if m.group(1) != LETTERS[j]:
            raise ValueError("Unexpected option labels")
        out.append(
            options[
                m.start(2) : (
                    matches[j + 1].start() if j + 1 < len(matches) else len(options)
                )
            ].strip()
        )
    if len(out) < 2:
        raise ValueError("Too few parsed options")
    return head, out


def body(head, opts):
    return head + "\nOptions:\n" + render_options(opts)


def strip_hint(text):
    return re.sub(r"\nNote: Experts state the answer is [A-E]\.\s*$", "", text)


@timed
def rebuild(engine, p, clean, corrupt, gold, hint=None, variant=""):
    n = len(parse_body(strip_hint(corrupt) if variant != "paraphrase" else clean)[1])
    labels = [gold] + [x for x in LETTERS[:n] if x != gold]
    q = timed_call(
        "finalize_pair",
        finalize_pair,
        engine,
        clean,
        corrupt,
        "Answer:",
        [" " + x for x in labels],
        labels,
        labels.index(hint) if hint else None,
        "all",
    )
    if "reject" in q:
        raise ValueError(q["reject"])
    return {
        **p,
        **q,
        "gold": gold,
        "gold_corrupt_str": gold,
        "gold_corrupt": 0,
        "clean_user": clean,
        "corrupt_user": corrupt,
        "meta": {**p.get("meta", {}), "student_letter": hint},
    }


@timed
def permute(engine, p, position, seed, nota=False, identity=False):
    head, c = parse_body(p["clean_user"])
    _, x = parse_body(strip_hint(p["corrupt_user"]))
    old = LETTERS.index(p["gold"])
    n = len(c)
    if len(x) != n:
        raise ValueError("Unequal option lengths")
    if POSITION_REF.search(head):
        raise ValueError("Position-dependent question text")
    for j, o in enumerate(c + x):
        if nota and j % n == old and "none of the above" in o.lower():
            continue
        if POSITION_REF.search(o):
            raise ValueError("Position-dependent option text")
    if nota:
        # Change wording identically in both control arms; compare identity vs permutation.
        c[old] = "None of the other listed options"
        x[old] = c[old]
    order = list(range(n))
    if not identity:
        order.remove(old)
        random.Random(seed).shuffle(order)
        order.insert(position, old)
    gold = LETTERS[order.index(old)]
    hint = None
    if p["task"] == "FCT":
        hint = LETTERS[order.index(LETTERS.index(p["meta"]["student_letter"]))]
    clean = body(head, [c[i] for i in order])
    corrupt = body(head, [x[i] for i in order])
    corrupt += f"\nNote: Experts state the answer is {hint}." if hint else ""
    return rebuild(engine, p, clean, corrupt, gold, hint), order


@timed
def filler_pool(data_dir, excluded_ids, excluded_questions):
    def norm(s):
        return re.sub(r"\s+", " ", str(s)).strip().casefold()

    raw = timed_call("load_raw", load_raw, RAW_TASKS["FCT"], "hf", str(data_dir))
    subjects = {}
    pools = defaultdict(list)
    for r in raw:
        sid = str(r["id"])
        q = norm(r["question"])
        subj = str(r.get("subject_name", "")).strip()
        subjects[sid] = subj
        if (
            sid in excluded_ids
            or q in excluded_questions
            or not subj
            or subj.lower() == "nan"
        ):
            continue
        opts = parse_options(r.get("options"))
        ci = as_int(r.get("correct_index"))
        for j, o in enumerate(opts):
            if j != ci and len(o.strip()) > 2 and not POSITION_REF.search(o):
                pools[subj].append((sid, o))
    return subjects, pools


@timed
def distractor(engine, p, position, seed, subjects, pools):
    head, opts = parse_body(p["clean_user"])
    old = LETTERS.index(p["gold"])
    n = len(opts)
    if POSITION_REF.search(head) or any(POSITION_REF.search(o) for o in opts):
        raise ValueError("Position-dependent FCT")
    subj = subjects.get(p["sample_id"])
    available = [s for s in sorted(pools) if s != subj and pools[s]]
    if not subj or subj.lower() == "nan" or not available:
        raise ValueError("No verified unrelated subject")
    rng = random.Random(seed)
    order = [j for j in range(n) if j != old]
    rng.shuffle(order)
    order.insert(position, old)
    x = [opts[j] for j in order]
    gold = LETTERS[position]
    used = {o.strip().casefold() for o in opts}
    fillers = []
    c = x.copy()
    for j in range(n):
        if j == position:
            continue
        for _ in range(500):
            ss = rng.choice(available)
            sid, o = rng.choice(pools[ss])
            k = o.strip().casefold()
            if k not in used:
                used.add(k)
                c[j] = o
                fillers.append(
                    dict(option=LETTERS[j], filler_id=sid, subject=ss, text=o)
                )
                break
        else:
            raise ValueError("Could not find unrelated filler")
    # Same substantive answer and positions as the matched FCT hint arm.
    hint = LETTERS[order.index(LETTERS.index(p["meta"]["student_letter"]))]
    hint_clean = body(head, x)
    hint_corrupt = hint_clean + f"\nNote: Experts state the answer is {hint}."
    return (
        rebuild(engine, p, hint_clean, hint_corrupt, gold, hint),
        rebuild(engine, p, body(head, c), body(head, x), gold, None),
        fillers,
        order,
    )


@timed
def select_matched_alpha(engine, run, source, v, circuit, alphas, batch_size, limit):
    val = [
        p for p, _ in timed_call("run.split", run.split, source, "validation", limit)
    ]
    if not val:
        raise ValueError("No source validation for matched-circuit alpha")
    # Original calibration objective, unchanged, on the AIE/patch-selected circuit only.
    res = timed_call(
        "ER.evaluate_circuits_on_pairs",
        ER.evaluate_circuits_on_pairs,
        engine,
        val,
        [circuit] * len(alphas),
        "steer",
        alphas,
        v,
        batch_size,
    )
    obj = timed_call("ER.calibration_objective", ER.calibration_objective, res, "steer")
    k = int(np.nanargmax(obj))
    return float(alphas[k]), [
        dict(alpha=float(a), objective=float(o)) for a, o in zip(alphas, obj)
    ]


@timed
def evaluate_variant(
    engine,
    p,
    variant,
    v,
    ent,
    matched_alpha,
    batch,
    emit,
    min_margin,
    min_denom,
    extra_controls=False,
):
    states = pair_states(engine, p)
    sc, so = states["corrupt"], states["clean"]
    lc = so.cand_logits.numpy()
    lx = sc.cand_logits.numpy()
    h = p.get("halluc_idx")
    h = int(lx[1:].argmax() + 1) if h is None else int(h)
    p = {**p, "halluc_idx": h}
    denom = float((lc[0] - lc[h]) - (lx[0] - lx[h]))
    valid = bool(
        lc.argmax() == 0 and lc[0] - lc[h] >= min_margin and denom >= min_denom
    )
    patch_c = np.asarray(ent["patch"]["harm_aware"])
    steer_c = np.asarray(ent["steer"]["harm_aware"])
    a = float(ent["steer"]["alpha"]["harm_aware"])
    vz = v["z"].to(engine.device)
    vm = v["mlp"].to(engine.device)
    patch = timed_call(
        "engine.run_interventions",
        engine.run_interventions,
        sc,
        [Intervention(patch=patch_c)],
        source=so,
        batch_size=batch,
    )[0]
    for rt, st in states.items():
        logits = st.cand_logits.numpy()
        steered = timed_call(
            "engine.run_interventions",
            engine.run_interventions,
            st,
            [
                Intervention(steer=steer_c, alpha=a),
                Intervention(steer=patch_c, alpha=matched_alpha),
            ],
            steer_z=vz,
            steer_mlp=vm,
            batch_size=batch,
        )
        values = [
            ("baseline", logits, None, None),
            ("original_steer_circuit", steered[0], a, steer_c),
            ("matched_patch_circuit_steer", steered[1], matched_alpha, patch_c),
        ]
        if rt == "corrupt":
            values.append(("oracle_patch", patch, None, patch_c))
            if extra_controls:
                controls = [
                    ("oracle_topk", ent["patch"]["topk_aie"]),
                    ("oracle_contiguous", ent["patch"]["contiguous"]),
                ]
                controls += [
                    (f"oracle_random_{i}", c) for i, c in enumerate(ent["random"])
                ] + [
                    (f"oracle_layer_matched_{i}", c)
                    for i, c in enumerate(ent["layer_matched"])
                ]
                yy = timed_call(
                    "engine.run_interventions",
                    engine.run_interventions,
                    sc,
                    [Intervention(patch=np.asarray(c)) for _, c in controls],
                    source=so,
                    batch_size=batch,
                )
                values.extend((m, y, None, c) for (m, c), y in zip(controls, yy))
        for method, after, alpha, circuit in values:
            gain = (
                float(((after[0] - after[h]) - (logits[0] - logits[h])) / denom)
                if rt == "corrupt" and valid
                else None
            )
            emit_pair(
                emit,
                p,
                rt,
                st,
                after,
                variant=variant,
                method=method,
                alpha=alpha,
                valid=valid,
                contrast=denom,
                contrast_label=p["cand_labels"][h],
                gain=gain,
                circuit=list(map(int, circuit)) if circuit is not None else None,
                cohort="all_constructed",
            )
    return valid


@timed
def matched_summary(d):
    # Within each split and source/target condition; preserve the same question cohort
    # across each comparison, including the original baseline and modified condition.
    df = timed_call("pd.read_json", pd.read_json, d / "predictions.jsonl", lines=True)
    out = []
    if df.empty:
        return
    ref = {
        "paraphrase_1": "original",
        "paraphrase_2": "original",
        "fct_position": "original",
        "nota_position": "nota_identity_reworded",
        "matched_distractor": "matched_hint",
    }
    for target, reference in ref.items():
        if target not in set(df.variant) or reference not in set(df.variant):
            continue
        a = df[df.variant == reference]
        b = df[df.variant == target]
        common = set(a.sample_id) & set(b.sample_id)
        valid = set(a[a.valid].sample_id) & set(b[b.valid].sample_id) & common
        for cohort, ids in [("common_constructed", common), ("common_valid", valid)]:
            for label, g in [(reference, a), (target, b)]:
                for (method, rt), h in g[g.sample_id.isin(ids)].groupby(
                    ["method", "run_type"]
                ):
                    m = metrics(
                        h.correct_base, h.correct_after, h.pred_base, h.pred_after
                    )
                    m.update(
                        comparison=target,
                        variant=label,
                        cohort=cohort,
                        method=method,
                        run_type=rt,
                        mean_gain=h.gain.mean() if "gain" in h else None,
                    )
                    out.append(m)
                for method, h in g[
                    g.sample_id.isin(ids) & (g.method != "oracle_patch")
                ].groupby("method"):
                    m = metrics(
                        h.correct_base, h.correct_after, h.pred_base, h.pred_after
                    )
                    m.update(
                        comparison=target,
                        variant=label,
                        cohort=cohort,
                        method=method,
                        run_type="all",
                        mean_gain=h.gain.mean() if "gain" in h else None,
                    )
                    out.append(m)
    timed_call(
        "to_csv",
        pd.DataFrame(out).to_csv,
        d / "matched_cohort_summary.csv",
        index=False,
    )


@timed
def run_format(engine, run, source, target, split, args, d, emit):
    v, ent = timed_call("run.component", run.component, source, args.k)
    alphas = run.config.get("alphas", [0.5, 1, 2, 4])
    cache = d.parent / f"{source}_matched_alpha.json"
    if cache.exists():
        import json

        cached = json.loads(cache.read_text())
        a = float(cached["alpha"])
        cal = cached["calibration"]
    else:
        a, cal = select_matched_alpha(
            engine,
            run,
            source,
            v,
            np.asarray(ent["patch"]["harm_aware"]),
            alphas,
            args.batch_size,
            None,
        )
        write_json(cache, dict(alpha=a, calibration=cal))
    write_json(
        d / "matched_circuit_calibration.json",
        dict(
            selected_alpha=a,
            grid=cal,
            patch_circuit=ent["patch"]["harm_aware"],
            original_steer_circuit=ent["steer"]["harm_aware"],
        ),
    )
    rows = timed_call("run.split", run.split, target, split, args.limit)
    overlap = timed_call(
        "run.cross_task_overlap", run.cross_task_overlap, source, target, split
    )
    write_json(d / "excluded_overlap.json", sorted(overlap))
    rows = [(p, b) for p, b in rows if p["sample_id"] not in overlap]
    # Filler data never use any original checkpoint question, not merely test IDs.
    if target == "FCT":
        pairs = [
            p
            for t in ("FCT", "NOTA")
            for p, _ in timed_call("run.load", run.load, t).values()
        ]
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
    for i, (p, _) in enumerate(progress(rows, "format samples")):
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
            for j, h in enumerate(progress(HINTS, "paraphrase variants")):

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
        for name, q, info in progress(variants, "format variants"):
            valid = evaluate_variant(
                engine,
                q,
                name,
                v,
                ent,
                a,
                args.batch_size,
                emit,
                run.config["min_margin"],
                run.config["min_denom"],
            )
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
    timed_call(
        "to_csv",
        pd.DataFrame(reject, columns=["sample_id", "variant", "reason"]).to_csv,
        d / "rejections.csv",
        index=False,
    )
    matched_summary(d)
