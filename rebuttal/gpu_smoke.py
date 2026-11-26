"""Load one real checkpoint and check prompt scoring and hooks before expensive jobs."""

import argparse, json, random, logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .core import MODELS, write_json
from .run import engine_for, versions
from ._timing import LOG, timed_call, step, configure_logging, LOG_LEVELS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument("--engine", default="auto", choices=["auto", "legacy", "sharded"])
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--allow-cpu-offload", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--revision", default=None)
    p.add_argument("--data-dir", default="medhalt/datasets")
    p.add_argument("--output", default="results/rebuttal_smoke.json")
    p.add_argument("--quantization", choices=["auto", "none", "nf4"], default="auto")
    p.add_argument("--log-level", choices=LOG_LEVELS, type=str.upper, default="DEBUG")
    a = p.parse_args()
    output = Path(a.output)
    log_file = (
        output.with_suffix(".log")
        if output.suffix != ".log"
        else output.with_suffix(".log.log")
    )
    configure_logging(log_file, a.log_level)
    try:
        with step(f"GPU smoke model={a.model}", level=logging.INFO):
            return _run(a)
    except BaseException:
        LOG.exception("GPU smoke failed")
        raise


def _run(a):
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model smoke requires CUDA")
    from patching_engine import Intervention
    from run_casual_transfer import build_mcq_pairs, load_raw, TASKS
    from experiment_runner import prepare_pair

    e = engine_for(a.model, a, None)
    stats = {}
    ps = timed_call(
        "build_mcq_pairs",
        build_mcq_pairs,
        e,
        "FCT",
        timed_call("load_raw", load_raw, TASKS["FCT"], "hf", a.data_dir),
        1,
        random.Random(0),
        stats,
    )
    if not ps:
        raise ValueError("Could not build one tokenizable FCT pair")
    p0 = ps[0]
    sc, so = timed_call("prepare_pair", prepare_pair, e, p0)
    l = e.num_layers // 2
    delta = so.resid.float() - sc.resid.float()
    ivs = [
        Intervention(),
        Intervention(resid_layers=np.array([l]), resid_alpha=1.0),
        Intervention(patch=np.array([e.space.head(l, 0), e.space.mlp(l)])),
    ]
    base = sc.cand_logits.numpy()
    out = timed_call(
        "e.run_interventions",
        e.run_interventions,
        sc,
        ivs,
        source=so,
        steer_resid=delta,
        batch_size=1,
    )
    result = dict(
        model=a.model,
        model_commit=getattr(e.model.config, "_commit_hash", None),
        engine=type(e).__name__,
        num_layers=e.num_layers,
        num_heads=e.num_heads,
        head_dim=e.head_dim,
        device_map="auto",
        max_memory={
            0: "6GiB",
            1: "6GiB",
            2: "6GiB",
            3: "6GiB",
        },
        environment=versions(),
        quantization=getattr(e, "quantization_metadata", {"mode": "none"}),
        peak_vram_bytes=torch.cuda.max_memory_allocated(0),
        sample_id=p0["sample_id"],
        cand_labels=p0["cand_labels"],
        base=base.tolist(),
        oracle_clean=so.cand_logits.numpy().tolist(),
        no_op=out[0].tolist(),
        resid_steer=out[1].tolist(),
        component_patch=out[2].tolist(),
        max_noop_drift=float(np.max(np.abs(base - out[0]))),
        resid_max_change=float(np.max(np.abs(base - out[1]))),
        patch_max_change=float(np.max(np.abs(base - out[2]))),
    )
    write_json(a.output, result)
    print(json.dumps(result, indent=2, default=str))
    if not np.isfinite(out).all() or result["max_noop_drift"] > 1.0:
        raise RuntimeError(
            "Smoke failed: nonfinite outputs or >1 logit no-op difference"
        )
    if result["resid_max_change"] == 0 and result["patch_max_change"] == 0:
        LOG.warning(
            "Both edits had zero visible logit effect; inspect hooks before large runs"
        )
    LOG.info(
        "Smoke passed finite-output/no-op checks. This is not a scientific result."
    )


if __name__ == "__main__":
    main()
