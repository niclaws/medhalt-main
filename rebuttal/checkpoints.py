"""Inspect saved runs without a GPU; restore NOTA halluc_idx from base checkpoint."""

from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from .core import TASKS, RTS, load_pt, write_json, LOG


class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.state = self.path / "state"
        self.config = json.loads((self.path / "run_config.json").read_text())
        self.splits = json.loads((self.path / "splits.json").read_text())
        self.data = {}
        self.raw = {}

    def load(self, t):
        if t in self.data:
            return self.data[t]
        obj = load_pt(self.state / f"pairs_{t}.pt")
        pairs = obj["pairs"]
        base = load_pt(self.state / f"base_{t}.pt")
        if len(pairs) != len(base):
            raise ValueError(f"{t}: pairs/base mismatch")
        d = {}
        for p, b in zip(pairs, base):
            p = dict(p)
            sid = str(p["sample_id"])
            p["sample_id"] = sid
            p["halluc_idx"] = int(b["halluc_idx"])
            p["gold_corrupt"] = int(b["gold_corrupt"])
            if sid in d:
                raise ValueError(f"{t}: duplicate sample ID {sid}")
            d[sid] = (p, b)
        seen = set()
        for split, ids in self.splits[t].items():
            ids = list(map(str, ids))
            ss = set(ids)
            if len(ss) != len(ids) or not ss <= d.keys() or seen & ss:
                raise ValueError(f"{t}: invalid/overlapping split {split}")
            seen |= ss
        self.raw[t] = obj
        self.data[t] = d
        return d

    def split(self, t, s, limit=None):
        d = self.load(t)
        ids = list(map(str, self.splits[t].get(s, [])))
        if limit:
            ids = ids[:limit]
        return [d[i] for i in ids]

    def selection(self, protocol, target):
        from eval_downstream_all_models import get_selection

        return get_selection(self.path, protocol, target)

    def layers(self, t):
        b = [b for _, b in self.split(t, "discovery")]
        if not b:
            raise ValueError("Empty discovery split")
        # Identical accumulation to the original evaluator (FP32 mean of cached FP16 activations).
        return torch.stack(
            [x["resid_clean"].float() - x["resid_corrupt"].float() for x in b]
        ).mean(0)

    def component(self, t, k=20):
        v = load_pt(self.state / f"vectors_{t}.pt")
        ent = load_pt(self.state / f"selection_{t}.pt")["circuits"][k]
        return v, ent

    def cross_task_overlap(self, source, target, split):
        def key(p):
            return (
                re.sub(
                    r"\s+",
                    " ",
                    p["clean_user"]
                    .split("Question: ", 1)[-1]
                    .split("\nOptions:", 1)[0],
                )
                .strip()
                .casefold()
            )

        # Exclude discovery/validation by ID OR normalized question, including same-task duplicates.
        training = self.split(source, "discovery") + self.split(source, "validation")
        ids = {p["sample_id"] for p, _ in training}
        qs = {key(p) for p, _ in training}
        return {
            p["sample_id"]
            for p, _ in self.split(target, split)
            if p["sample_id"] in ids or key(p) in qs
        }

    def audit(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        details = []
        opt = []
        for t in TASKS:
            d = self.load(t)
            obj = self.raw[t]
            c = self.config
            ids = {s: set(map(str, v)) for s, v in self.splits[t].items()}
            pool = set(d) - ids["unfiltered"]
            selected = set().union(
                *(ids.get(s, set()) for s in ("discovery", "validation", "test"))
            )
            for sid, (p, b) in d.items():
                lc = np.asarray(b["clean_logits"])
                lx = np.asarray(b["corrupt_logits"])
                h = p["halluc_idx"]
                margin = float(lc[0] - lc[h])
                top_margin = float(lc[0] - lc[1:].max())
                denom = margin - float(lx[0] - lx[h])
                correct = int(lc.argmax()) == 0
                m = correct and margin >= c["min_margin"]
                valid = m and denom >= c["min_denom"]
                split = next((s for s, v in ids.items() if sid in v), "unused")
                details.append(
                    dict(
                        task=t,
                        sample_id=sid,
                        split=split,
                        clean_correct=correct,
                        cached_correct=bool(b["correct_clean"]),
                        margin_vs_contrast=margin,
                        margin_vs_best_wrong=top_margin,
                        contrast=denom,
                        margin_qualified=m,
                        valid=valid,
                        selected=sid in selected,
                        invalid_selected=sid in selected and not valid,
                        cached_correct_mismatch=correct != bool(b["correct_clean"]),
                    )
                )
                for rt, l in [("clean", lc), ("corrupt", lx)]:
                    opt.append(
                        dict(
                            task=t,
                            sample_id=sid,
                            split=split,
                            run_type=rt,
                            gold=p["gold"],
                            pred=p["cand_labels"][int(l.argmax())],
                            contrast_label=p["cand_labels"][h],
                        )
                    )
            df = pd.DataFrame([x for x in details if x["task"] == t])
            pp = df[df.sample_id.isin(pool)]
            hh = df[df.split == "unfiltered"]
            row = dict(
                task=t,
                requested=c.get("n_samples"),
                raw_rows=obj.get("n_raw"),
                built=len(d),
                construction_rejected=sum(obj.get("stats", {}).values()),
                construction_attempted=len(d) + sum(obj.get("stats", {}).values()),
                heldout=len(hh),
                pool=len(pp),
                clean_correct=int(pp.clean_correct.sum()),
                margin_qualified=int(pp.margin_qualified.sum()),
                contrast_qualified=int(pp.valid.sum()),
                retained=len(selected),
                discovery=len(ids.get("discovery", [])),
                validation=len(ids.get("validation", [])),
                test=len(ids.get("test", [])),
                clean_acc_holdout=hh.clean_correct.mean(),
                clean_acc_pool=pp.clean_correct.mean(),
                invalid_selected=int(df.invalid_selected.sum()),
                cached_correct_mismatch=int(df.cached_correct_mismatch.sum()),
                max_pairs=c.get("max_pairs_per_task"),
                unfiltered_frac=c.get("unfiltered_frac"),
                overlap_source_other_test=len(
                    self.cross_task_overlap(t, "NOTA" if t == "FCT" else "FCT", "test")
                ),
                overlap_same_task_test=len(self.cross_task_overlap(t, t, "test")),
                overlap_same_task_unfiltered=len(
                    self.cross_task_overlap(t, t, "unfiltered")
                ),
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "funnel.csv", index=False)
        pd.DataFrame(details).to_csv(out / "pair_audit.csv", index=False)
        df = pd.DataFrame(opt)
        df.to_csv(out / "options.csv", index=False)
        for k in ("gold", "pred", "contrast_label"):
            df.groupby(["task", "split", "run_type", k]).size().rename(
                "n"
            ).reset_index().to_csv(out / f"distribution_{k}.csv", index=False)

        if (self.path / "selection_funnel.csv").exists():
            original = pd.read_csv(self.path / "selection_funnel.csv")
            original.to_csv(out / "original_selection_funnel.csv", index=False)
            compare = pd.DataFrame(rows).merge(
                original, on="task", suffixes=("_recomputed", "_original")
            )
            compare.to_csv(out / "funnel_vs_original.csv", index=False)
        write_json(out / "original_settings.json", self.config)
        LOG.info("Audit:\n%s", pd.DataFrame(rows).to_string(index=False))
        return rows
