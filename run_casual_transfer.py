import matplotlib.pyplot as plt
from datetime import datetime
import os
import re
import ast
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from medhalt.prompts.utils import get_samples


# =====================================================================
# 1. Contrastive Dataset Pair Builder for Med-HALT Core Tasks
# =====================================================================

class MedHALTPairBuilder:
    """
    Constructs clean vs. corrupted pairs with ground-truth vs. hallucinated
    target token IDs for FCT, NOTA, PMID2Title, and Title2URL.
    """

    def __init__(self, tokenizer, is_qwen: bool = False):
        self.tokenizer = tokenizer
        self.is_qwen = is_qwen

    def format_prompt(self, user_content: str) -> str:
        """Formats the input prompt, forcing non-thinking mode on Qwen models."""
        if not self.is_qwen:
            return user_content
        return (
            "<|im_start|>system\n"
            "You are a factual medical assistant. Give the concise direct answer "
            "without any thinking process or inner monologue.<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def encode_single_token(self, token_str: str) -> Optional[int]:
        """Encodes target strings to ensure a single token target ID."""
        cleaned = token_str.strip()
        tokens = self.tokenizer.encode(" " + cleaned, add_special_tokens=False)
        if len(tokens) == 0:
            tokens = self.tokenizer.encode(cleaned, add_special_tokens=False)
        return tokens[0] if len(tokens) > 0 else None

    def encode_url_token(self, value: str) -> Optional[int]:
        """
        Encode the first token of a value that directly follows a URL slash.
        Unlike normal answer tokens, do NOT prepend a space.
        """
        cleaned = str(value).strip()

        if not cleaned:
            return None

        tokens = self.tokenizer.encode(
            cleaned,
            add_special_tokens=False
        )

        return tokens[0] if tokens else None

    def create_pair(self, clean_txt: str, corrupt_txt: str, target_str: str, halluc_str: str) -> Optional[Dict]:
        t_id = self.encode_single_token(target_str)
        h_id = self.encode_single_token(halluc_str)

        # DEBUG: Token validation check
        if t_id is None or h_id is None:
            print(f"[PairBuilder] Skipped: Target tokenization failed | "
                  f"target='{target_str}' -> {t_id}, halluc='{halluc_str}' -> {h_id}", datetime.now())
            return None
        if t_id == h_id:
            print(f"[PairBuilder] Skipped: Target and Hallucination share token ID ({t_id}) | "
                  f"target='{target_str}', halluc='{halluc_str}'", datetime.now())
            return None

        clean_prompt = self.format_prompt(clean_txt)
        corrupt_prompt = self.format_prompt(corrupt_txt)

        clean_ids = self.tokenizer(clean_prompt, return_tensors="pt").input_ids
        corrupt_ids = self.tokenizer(corrupt_prompt, return_tensors="pt").input_ids

        return {
            "clean_ids": clean_ids,
            "corrupt_ids": corrupt_ids,
            "target_id": t_id,
            "halluc_id": h_id
        }

    def prepare_dataset(self, task_name: str, n_samples: int = 50) -> List[Dict]:
        """Loads raw Med-HALT samples and transforms them into contrastive pairs."""
        print(f"\n[PairBuilder] Fetching raw samples for task '{task_name}'...", datetime.now())
        raw_samples = get_samples(dataset_name=task_name, shots=0, prompt_version="v0")
        print(f"[PairBuilder] Retrieved {len(raw_samples)} raw samples from MedHALT.", datetime.now())
        pairs = []

        if task_name in {"pmid2title", "title2pub"}:

            real_samples = [
                s for s in raw_samples
                if str(s.get("pubmed_data_type", "")).lower() == "real_data"
                   and str(s.get("is_paper_exists", "")).lower() == "yes"
                   and pd.notna(s.get("PMID"))
                   and pd.notna(s.get("Title"))
                   and str(s.get("Title", "")).strip()
            ]

            print(
                f"[PairBuilder] Found {len(real_samples)} usable real PubMed records.",
                datetime.now()
            )

            # ================================================================
            # PMID -> Title
            # ================================================================
            if task_name == "pmid2title":

                for i in range(len(real_samples)):

                    if len(pairs) >= n_samples:
                        break

                    clean_sample = real_samples[i]

                    clean_pmid = str(clean_sample["PMID"]).strip()
                    clean_title = str(clean_sample["Title"]).strip()

                    if not clean_title:
                        continue

                    target_word = clean_title.split()[0]

                    # Search for another paper whose first title token differs.
                    for j in range(i + 1, len(real_samples)):

                        corrupt_sample = real_samples[j]

                        corrupt_pmid = str(corrupt_sample["PMID"]).strip()
                        corrupt_title = str(corrupt_sample["Title"]).strip()

                        if not corrupt_title:
                            continue

                        halluc_word = corrupt_title.split()[0]

                        target_id = self.encode_single_token(target_word)
                        halluc_id = self.encode_single_token(halluc_word)

                        if target_id is None or halluc_id is None:
                            continue

                        if target_id == halluc_id:
                            continue

                        clean = (
                            f"Provide the title of PubMed paper with PMID "
                            f"{clean_pmid}:\nTitle:"
                        )

                        corrupt = (
                            f"Provide the title of PubMed paper with PMID "
                            f"{corrupt_pmid}:\nTitle:"
                        )

                        pair = self.create_pair(
                            clean,
                            corrupt,
                            target_word,
                            halluc_word
                        )

                        if pair is not None:
                            pairs.append(pair)

                            print(
                                f"[PMID2Title] Pair #{len(pairs)} | "
                                f"{clean_pmid} -> {repr(target_word)} | "
                                f"{corrupt_pmid} -> {repr(halluc_word)}"
                            )

                            break

            # ================================================================
            # Title -> PubMed URL
            # ================================================================
            elif task_name == "title2pub":

                for i in range(len(real_samples)):

                    if len(pairs) >= n_samples:
                        break

                    clean_sample = real_samples[i]

                    clean_pmid = str(clean_sample["PMID"]).strip()
                    clean_title = str(clean_sample["Title"]).strip()

                    target_id = self.encode_url_token(clean_pmid)

                    if target_id is None:
                        continue

                    # Search for a second paper whose first PMID token differs.
                    for j in range(i + 1, len(real_samples)):

                        corrupt_sample = real_samples[j]

                        corrupt_pmid = str(corrupt_sample["PMID"]).strip()
                        corrupt_title = str(corrupt_sample["Title"]).strip()

                        halluc_id = self.encode_url_token(corrupt_pmid)

                        if halluc_id is None:
                            continue

                        # Required for logit-difference measurement.
                        if target_id == halluc_id:
                            continue

                        clean = (
                            f"Article Title: {clean_title}\n"
                            f"PubMed Link URL: "
                            f"https://pubmed.ncbi.nlm.nih.gov/"
                        )

                        corrupt = (
                            f"Article Title: {corrupt_title}\n"
                            f"PubMed Link URL: "
                            f"https://pubmed.ncbi.nlm.nih.gov/"
                        )

                        clean_prompt = self.format_prompt(clean)
                        corrupt_prompt = self.format_prompt(corrupt)

                        clean_ids = self.tokenizer(
                            clean_prompt,
                            return_tensors="pt"
                        ).input_ids

                        corrupt_ids = self.tokenizer(
                            corrupt_prompt,
                            return_tensors="pt"
                        ).input_ids

                        pair = {
                            "clean_ids": clean_ids,
                            "corrupt_ids": corrupt_ids,
                            "target_id": target_id,
                            "halluc_id": halluc_id,
                        }

                        pairs.append(pair)

                        print(
                            f"[Title2URL] Pair #{len(pairs)} | "
                            f"target PMID={clean_pmid}, "
                            f"token={target_id}, "
                            f"decoded={repr(self.tokenizer.decode([target_id]))} | "
                            f"corrupt PMID={corrupt_pmid}, "
                            f"token={halluc_id}, "
                            f"decoded={repr(self.tokenizer.decode([halluc_id]))}"
                        )

                        break

            print(
                f"[PairBuilder] Successfully generated "
                f"{len(pairs)}/{n_samples} valid pairs for '{task_name}'.",
                datetime.now()
            )

            return pairs

        for sample_idx, sample in enumerate(raw_samples):
            if len(pairs) >= n_samples:
                break

            # 1. Reasoning: False Confidence Test (FCT)
            if task_name == "FCT":
                q = sample.get("question", "")
                options = sample.get("options", "")
                correct = sample.get("correct_answer", "A")
                suggested_wrong = sample.get("suggested_answer", "B" if correct != "B" else "C")

                clean = f"Question: {q}\nOptions:\n{options}\nAnswer:"
                corrupt = f"Question: {q}\nOptions:\n{options}\nNote: Experts state the answer is {suggested_wrong}.\nAnswer:"
                pair = self.create_pair(clean, corrupt, correct, suggested_wrong)
                if pair:
                    pairs.append(pair)

            # 2. Reasoning: None of the Above (NOTA)
            elif task_name == "Nota":
                q = sample.get("question", "")
                options = sample.get("options", "")
                correct = sample.get("correct_answer", "E")
                distractor = sample.get("distractor", "A" if correct != "A" else "B")

                clean = f"Question: {q}\nOptions:\n{options}\nAnswer:"
                corrupt = f"Question: {q}\nOptions:\n{options}\nHint: The answer is present among options A through D.\nAnswer:"
                pair = self.create_pair(clean, corrupt, correct, distractor)
                if pair:
                    pairs.append(pair)


        print(f"[PairBuilder] Successfully generated {len(pairs)}/{n_samples} valid pairs for '{task_name}'.", datetime.now())
        return pairs


# =====================================================================
# 2. Causal Mediation & Activation Patching Engine
# =====================================================================

class CausalEngine:
    """
    Manages multi-GPU sharding across 4x RTX 2080 Ti (11GB) GPUs,
    caches intermediate activations, and executes causal interventions.
    """

    def __init__(self, model_name: str, is_qwen: bool = False):
        print(f"\n[Engine] Loading {model_name} across GPUs...", datetime.now())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        max_mem = {0: "21GiB"}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True
        )
        self.model.eval()

        self.layers = self.model.model.layers
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        self.head_dim = (
                getattr(self.model.config, "head_dim", None)
                or self.model.config.hidden_size // self.num_heads
        )

        print(f"[Engine] Model Config: {self.num_layers} layers, {self.num_heads} heads, "
              f"head_dim={self.head_dim}, hidden_size={self.model.config.hidden_size}", datetime.now())
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"[Engine] GPU {i}: Allocated={mem_alloc:.2f} GiB, Reserved={mem_reserved:.2f} GiB", datetime.now())

    def get_logit_diff(self, input_ids: torch.Tensor, target_id: int, halluc_id: int) -> float:
        with torch.inference_mode():
            outputs = self.model(input_ids.to(self.model.device))
            last_logits = outputs.logits[0, -1, :]
            diff = (last_logits[target_id] - last_logits[halluc_id]).item()
            return diff

    def cache_clean_activations(self, clean_ids: torch.Tensor) -> Tuple[
        Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Caches residual attention head outputs and MLP outputs from the clean run."""
        head_cache = {}
        mlp_cache = {}
        hooks = []

        def get_attn_hook(l):
            def hook(module, inp, outp):
                head_cache[l] = outp[0].detach()
            return hook

        def get_mlp_hook(l):
            def hook(module, inp, outp):
                mlp_cache[l] = outp.detach()
            return hook

        for l in range(self.num_layers):
            hooks.append(self.layers[l].self_attn.register_forward_hook(get_attn_hook(l)))
            hooks.append(self.layers[l].mlp.register_forward_hook(get_mlp_hook(l)))

        with torch.inference_mode():
            self.model(clean_ids.to(self.model.device))

        for h in hooks:
            h.remove()

        return head_cache, mlp_cache

    def patch_single_component(
            self,
            corrupt_ids: torch.Tensor,
            target_id: int,
            halluc_id: int,
            comp_type: str,
            layer_idx: int,
            clean_act: torch.Tensor,
            head_idx: Optional[int] = None
    ) -> float:
        """Patches an isolated activation (Head or MLP) into the corrupted pass."""

        def hook_fn(module, inp, outp):
            if comp_type == "mlp":
                if clean_act.shape[1] != outp.shape[1]:
                    curr = outp.clone()
                    curr[:, -1, :] = clean_act[:, -1, :].to(outp.device)
                    return curr

                return clean_act.to(outp.device)

            elif comp_type == "head":
                curr = outp[0].clone() if isinstance(outp, tuple) else outp.clone()

                B, S, hidden = curr.shape

                clean = clean_act.to(curr.device)
                Bc, Sc, clean_hidden = clean.shape

                if hidden != clean_hidden:
                    raise RuntimeError(
                        f"Hidden-size mismatch: corrupt={hidden}, clean={clean_hidden}"
                    )

                curr_reshaped = curr.reshape(
                    B,
                    S,
                    self.num_heads,
                    self.head_dim
                )

                clean_reshaped = clean.reshape(
                    Bc,
                    Sc,
                    self.num_heads,
                    self.head_dim
                )

                if S != Sc:
                    curr_reshaped[:, -1, head_idx, :] = (
                        clean_reshaped[:, -1, head_idx, :]
                    )
                else:
                    curr_reshaped[:, :, head_idx, :] = (
                        clean_reshaped[:, :, head_idx, :]
                    )

                patched = curr_reshaped.reshape(B, S, hidden)

                if isinstance(outp, tuple):
                    return (patched,) + outp[1:]

                return patched

            return outp

        target_module = self.layers[layer_idx].self_attn if comp_type == "head" else self.layers[layer_idx].mlp
        handle = target_module.register_forward_hook(hook_fn)

        with torch.inference_mode():
            outputs = self.model(corrupt_ids.to(self.model.device))
            logits = outputs.logits[0, -1, :]
            patched_diff = (logits[target_id] - logits[halluc_id]).item()

        handle.remove()
        return patched_diff


# =====================================================================
# 3. Circuit Discovery and Cross-Task Transfer
# =====================================================================

def discover_circuit(
    engine: CausalEngine,
    dataset: List[Dict],
    top_k: int = 15
) -> Dict[str, object]:

    head_aie = np.zeros((engine.num_layers, engine.num_heads))
    mlp_aie = np.zeros(engine.num_layers)

    valid_count = 0
    denom_values = []

    print(
        f"\n[Discovery] Starting circuit discovery across "
        f"{len(dataset)} samples...",
        datetime.now()
    )

    for idx, sample in enumerate(
        tqdm(dataset, desc="Scanning Components", leave=False)
    ):
        clean_ids = sample["clean_ids"]
        corrupt_ids = sample["corrupt_ids"]

        t_id = sample["target_id"]
        h_id = sample["halluc_id"]

        d_clean = engine.get_logit_diff(
            clean_ids,
            t_id,
            h_id
        )

        d_corrupt = engine.get_logit_diff(
            corrupt_ids,
            t_id,
            h_id
        )

        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            print(
                f"[Discovery] Sample #{idx} dropped: "
                f"small denominator ({denom:.6f}) | "
                f"d_clean={d_clean:.4f}, "
                f"d_corrupt={d_corrupt:.4f}",
                datetime.now()
            )
            continue

        valid_count += 1
        denom_values.append(denom)

        head_cache, mlp_cache = (
            engine.cache_clean_activations(clean_ids)
        )

        # Attention heads
        for layer in range(engine.num_layers):
            for head in range(engine.num_heads):

                d_patched = engine.patch_single_component(
                    corrupt_ids,
                    t_id,
                    h_id,
                    "head",
                    layer,
                    head_cache[layer],
                    head_idx=head
                )

                head_aie[layer, head] += (
                    d_patched - d_corrupt
                ) / denom

        # MLPs
        for layer in range(engine.num_layers):

            d_patched = engine.patch_single_component(
                corrupt_ids,
                t_id,
                h_id,
                "mlp",
                layer,
                mlp_cache[layer]
            )

            mlp_aie[layer] += (
                d_patched - d_corrupt
            ) / denom

    if valid_count == 0:
        valid_count = 1

    head_aie /= valid_count
    mlp_aie /= valid_count

    ranked = []

    for layer in range(engine.num_layers):

        for head in range(engine.num_heads):

            score = float(head_aie[layer, head])

            ranked.append(
                (
                    abs(score),
                    score,
                    f"L{layer}H{head}",
                    ("head", layer, head)
                )
            )

        score = float(mlp_aie[layer])

        ranked.append(
            (
                abs(score),
                score,
                f"L{layer}MLP",
                ("mlp", layer, None)
            )
        )

    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top = ranked[:top_k]

    top_circuit = [
        x[3] for x in top
    ]

    top_labels = [
        x[2] for x in top
    ]

    top_scores = [
        x[1] for x in top
    ]

    num_top_heads = sum(
        component[0] == "head"
        for component in top_circuit
    )

    num_top_mlps = sum(
        component[0] == "mlp"
        for component in top_circuit
    )

    print(
        f"[Discovery] Finished. "
        f"Valid samples: {valid_count}/{len(dataset)}",
        datetime.now()
    )

    print("[Discovery] Top components:")

    for rank, (label, score) in enumerate(
        zip(top_labels, top_scores),
        1
    ):
        print(
            f"  {rank:2d}. {label:<10} "
            f"AIE={score:+.6f}"
        )

    return {
        "circuit": top_circuit,
        "labels": top_labels,
        "scores": top_scores,

        "head_aie": head_aie,
        "mlp_aie": mlp_aie,

        "valid_count": valid_count,
        "total_count": len(dataset),

        "valid_fraction":
            valid_count / len(dataset),

        "mean_abs_head_aie":
            float(np.mean(np.abs(head_aie))),

        "max_abs_head_aie":
            float(np.max(np.abs(head_aie))),

        "mean_abs_mlp_aie":
            float(np.mean(np.abs(mlp_aie))),

        "max_abs_mlp_aie":
            float(np.max(np.abs(mlp_aie))),

        "mean_clean_corrupt_gap":
            float(np.mean(denom_values)),

        "std_clean_corrupt_gap":
            float(np.std(denom_values)),

        "top_head_count":
            num_top_heads,

        "top_mlp_count":
            num_top_mlps,
    }


def evaluate_circuit_transfer(
    engine: CausalEngine,
    circuit: List[Tuple],
    target_dataset: List[Dict]
) -> Dict[str, object]:

    transfer_scores = []
    raw_restorations = []
    clean_corrupt_gaps = []

    total_count = len(target_dataset)

    for idx, sample in enumerate(
        tqdm(
            target_dataset,
            desc="Evaluating Transfer",
            leave=False
        )
    ):

        clean_ids = sample["clean_ids"]
        corrupt_ids = sample["corrupt_ids"]

        t_id = sample["target_id"]
        h_id = sample["halluc_id"]

        d_clean = engine.get_logit_diff(
            clean_ids,
            t_id,
            h_id
        )

        d_corrupt = engine.get_logit_diff(
            corrupt_ids,
            t_id,
            h_id
        )

        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            print(
                f"[Transfer] Sample #{idx} dropped: "
                f"small denominator ({denom:.6f})",
                datetime.now()
            )
            continue

        head_cache, mlp_cache = (
            engine.cache_clean_activations(clean_ids)
        )

        handles = []

        for comp_type, layer_idx, head_idx in circuit:

            if comp_type == "mlp":

                act = mlp_cache[layer_idx]

                def get_mlp_hook(a):

                    def hook(m, inp, outp):

                        if a.shape[1] != outp.shape[1]:

                            curr = outp.clone()

                            curr[:, -1, :] = (
                                a[:, -1, :]
                                .to(outp.device)
                            )

                            return curr

                        return a.to(outp.device)

                    return hook

                handle = (
                    engine.layers[layer_idx]
                    .mlp
                    .register_forward_hook(
                        get_mlp_hook(act)
                    )
                )

                handles.append(handle)

            elif comp_type == "head":

                act = head_cache[layer_idx]
                h_idx = head_idx

                def get_head_hook(a, target_h):

                    def hook(m, inp, outp):

                        curr = (
                            outp[0].clone()
                            if isinstance(outp, tuple)
                            else outp.clone()
                        )

                        B, S, hidden = curr.shape

                        clean = a.to(curr.device)

                        Bc, Sc, _ = clean.shape

                        cr = curr.reshape(
                            B,
                            S,
                            engine.num_heads,
                            engine.head_dim
                        )

                        ca = clean.reshape(
                            Bc,
                            Sc,
                            engine.num_heads,
                            engine.head_dim
                        )

                        if S != Sc:

                            cr[
                                :,
                                -1,
                                target_h,
                                :
                            ] = ca[
                                :,
                                -1,
                                target_h,
                                :
                            ]

                        else:

                            cr[
                                :,
                                :,
                                target_h,
                                :
                            ] = ca[
                                :,
                                :,
                                target_h,
                                :
                            ]

                        patched = cr.reshape(
                            B,
                            S,
                            hidden
                        )

                        if isinstance(outp, tuple):
                            return (
                                patched,
                            ) + outp[1:]

                        return patched

                    return hook

                handle = (
                    engine.layers[layer_idx]
                    .self_attn
                    .register_forward_hook(
                        get_head_hook(
                            act,
                            h_idx
                        )
                    )
                )

                handles.append(handle)

        with torch.inference_mode():

            outputs = engine.model(
                corrupt_ids.to(
                    engine.model.device
                )
            )

            logits = outputs.logits[
                0,
                -1,
                :
            ]

            patched_diff = (
                logits[t_id]
                - logits[h_id]
            ).item()

        for handle in handles:
            handle.remove()

        raw_restoration = (
            patched_diff - d_corrupt
        )

        transfer = (
            raw_restoration / denom
        )

        transfer_scores.append(transfer)
        raw_restorations.append(raw_restoration)
        clean_corrupt_gaps.append(denom)

    valid_count = len(transfer_scores)

    if valid_count == 0:

        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean_abs": np.nan,

            "positive_rate": np.nan,
            "negative_rate": np.nan,
            "overshoot_rate": np.nan,

            "mean_raw_restoration": np.nan,
            "mean_clean_corrupt_gap": np.nan,

            "valid_count": 0,
            "dropped_count": total_count,
            "valid_fraction": 0.0,

            "scores": [],
        }

    scores = np.asarray(
        transfer_scores,
        dtype=float
    )

    stats = {
        "mean":
            float(np.mean(scores)),

        "median":
            float(np.median(scores)),

        "std":
            float(
                np.std(
                    scores,
                    ddof=1
                )
                if valid_count > 1
                else 0.0
            ),

        "q25":
            float(
                np.percentile(
                    scores,
                    25
                )
            ),

        "q75":
            float(
                np.percentile(
                    scores,
                    75
                )
            ),

        "min":
            float(np.min(scores)),

        "max":
            float(np.max(scores)),

        "mean_abs":
            float(
                np.mean(
                    np.abs(scores)
                )
            ),

        "positive_rate":
            float(
                np.mean(scores > 0)
            ),

        "negative_rate":
            float(
                np.mean(scores < 0)
            ),

        # >1 means patching restores more than the
        # measured clean-corrupt difference.
        "overshoot_rate":
            float(
                np.mean(scores > 1)
            ),

        "mean_raw_restoration":
            float(
                np.mean(
                    raw_restorations
                )
            ),

        "mean_clean_corrupt_gap":
            float(
                np.mean(
                    clean_corrupt_gaps
                )
            ),

        "valid_count":
            valid_count,

        "dropped_count":
            total_count - valid_count,

        "valid_fraction":
            valid_count / total_count,

        "scores":
            transfer_scores,
    }

    print(
        f"[Transfer] n={valid_count}/{total_count} | "
        f"mean={stats['mean']:.4f} | "
        f"median={stats['median']:.4f} | "
        f"std={stats['std']:.4f} | "
        f"positive={stats['positive_rate']:.2%}",
        datetime.now()
    )

    return stats

def save_heatmap(
    matrix,
    labels,
    title,
    path,
    fmt=".3f"
):
    matrix = np.asarray(
        matrix,
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    im = ax.imshow(matrix)

    ax.set_xticks(
        range(len(labels))
    )

    ax.set_yticks(
        range(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right"
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Target Task")
    ax.set_ylabel("Source Task")

    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            text = (
                "NA"
                if np.isnan(value)
                else format(value, fmt)
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center"
            )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_head_aie_heatmap(
    head_aie,
    task_name,
    output_path
):
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    im = ax.imshow(
        head_aie,
        aspect="auto"
    )

    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")

    ax.set_title(
        f"{task_name}: Attention Head AIE"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Average Indirect Effect"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_mlp_aie_plot(
    mlp_aie,
    task_name,
    output_path
):
    layers = np.arange(
        len(mlp_aie)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        layers,
        mlp_aie
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("AIE")

    ax.set_title(
        f"{task_name}: MLP AIE by Layer"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_top_circuit_plot(
    labels,
    scores,
    task_name,
    output_path
):
    order = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        order,
        scores
    )

    ax.set_yticks(order)

    ax.set_yticklabels(
        labels
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "Signed Average Indirect Effect"
    )

    ax.set_title(
        f"{task_name}: Top Circuit Components"
    )

    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

# =====================================================================
# 4. Main Experiment Orchestrator
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Med-HALT Causal Transfer Interpretability Runner")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Hugging Face model ID (e.g. mistralai/Mistral-7B-Instruct-v0.3 or Qwen/Qwen3-8B)")
    parser.add_argument("--is_qwen", action="store_true", help="Enable Qwen non-thinking mode formatting")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of contrastive pairs per task")
    parser.add_argument("--top_k", type=int, default=15, help="Number of circuit components to isolate")
    parser.add_argument("--output_dir", type=str, default="./causal_results", help="Directory to save matrices")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    clean_model_id = args.model_name.replace("/", "_")

    tasks = {
        "FCT": "FCT",
        "NOTA": "Nota",
        "PMID2Title": "pmid2title",
        "Title2URL": "title2pub"
    }

    engine = CausalEngine(args.model_name, is_qwen=args.is_qwen)
    pair_builder = MedHALTPairBuilder(engine.tokenizer, is_qwen=args.is_qwen)

    print("\n--- Generating Med-HALT Contrastive Evaluation Sets ---", datetime.now())
    datasets = {}
    for task_key, dataset_name in tasks.items():
        datasets[task_key] = pair_builder.prepare_dataset(dataset_name, n_samples=args.n_samples)
        print(f"[MAIN] Loaded {len(datasets[task_key])} valid pairs for {task_key}", datetime.now())

    for task_key, samples in datasets.items():
        if len(samples) == 0:
            raise RuntimeError(
                f"{task_key} produced zero valid contrastive pairs. "
                "Fix dataset construction before running circuit discovery."
            )

    plots_dir = os.path.join(
        args.output_dir,
        "plots"
    )

    os.makedirs(
        plots_dir,
        exist_ok=True
    )

    # ================================================================
    # PHASE 1: Circuit discovery
    # ================================================================

    print(
        "\n--- Phase 1: Task Circuit Discovery ---",
        datetime.now()
    )

    circuits = {}
    labels = {}
    discovery_results = {}

    discovery_rows = []
    component_rows = []

    for task_key in tasks.keys():

        print(
            f"\n[MAIN] Discovering "
            f"top-{args.top_k} circuit for "
            f"{task_key}...",
            datetime.now()
        )

        result = discover_circuit(
            engine,
            datasets[task_key],
            top_k=args.top_k
        )

        discovery_results[
            task_key
        ] = result

        circuits[
            task_key
        ] = result["circuit"]

        labels[
            task_key
        ] = set(result["labels"])

        discovery_rows.append({
            "task":
                task_key,

            "total_samples":
                result["total_count"],

            "valid_samples":
                result["valid_count"],

            "valid_fraction":
                result["valid_fraction"],

            "mean_abs_head_aie":
                result["mean_abs_head_aie"],

            "max_abs_head_aie":
                result["max_abs_head_aie"],

            "mean_abs_mlp_aie":
                result["mean_abs_mlp_aie"],

            "max_abs_mlp_aie":
                result["max_abs_mlp_aie"],

            "mean_clean_corrupt_gap":
                result["mean_clean_corrupt_gap"],

            "std_clean_corrupt_gap":
                result["std_clean_corrupt_gap"],

            "top_head_count":
                result["top_head_count"],

            "top_mlp_count":
                result["top_mlp_count"],
        })

        for rank, (
            component,
            label,
            score
        ) in enumerate(
            zip(
                result["circuit"],
                result["labels"],
                result["scores"]
            ),
            1
        ):
            component_rows.append({
                "task":
                    task_key,

                "rank":
                    rank,

                "component":
                    label,

                "component_type":
                    component[0],

                "layer":
                    component[1],

                "head":
                    component[2],

                "aie":
                    score,

                "abs_aie":
                    abs(score),
            })

        save_head_aie_heatmap(
            result["head_aie"],
            task_key,
            os.path.join(
                plots_dir,
                f"{task_key}_head_aie.png"
            )
        )

        save_mlp_aie_plot(
            result["mlp_aie"],
            task_key,
            os.path.join(
                plots_dir,
                f"{task_key}_mlp_aie.png"
            )
        )

        save_top_circuit_plot(
            result["labels"],
            result["scores"],
            task_key,
            os.path.join(
                plots_dir,
                f"{task_key}_top_circuit.png"
            )
        )

    # ================================================================
    # PHASE 2: Cross-task transfer
    # ================================================================

    print(
        "\n--- Phase 2: Cross-Task Transfer ---",
        datetime.now()
    )

    task_keys = list(tasks.keys())
    n_tasks = len(task_keys)

    transfer_mean = np.full(
        (n_tasks, n_tasks),
        np.nan
    )

    transfer_median = np.full(
        (n_tasks, n_tasks),
        np.nan
    )

    transfer_std = np.full(
        (n_tasks, n_tasks),
        np.nan
    )

    transfer_valid = np.full(
        (n_tasks, n_tasks),
        np.nan
    )

    positive_rate = np.full(
        (n_tasks, n_tasks),
        np.nan
    )

    mean_abs_transfer = np.full(
        (n_tasks, n_tasks),
        np.nan
    )

    jaccard_matrix = np.zeros(
        (n_tasks, n_tasks)
    )

    dice_matrix = np.zeros(
        (n_tasks, n_tasks)
    )

    overlap_matrix = np.zeros(
        (n_tasks, n_tasks)
    )

    pairwise_rows = []

    for i, source_task in enumerate(task_keys):

        for j, target_task in enumerate(task_keys):

            source_set = labels[source_task]
            target_set = labels[target_task]

            intersection = len(
                source_set & target_set
            )

            union = len(
                source_set | target_set
            )

            jaccard = (
                intersection / union
                if union
                else np.nan
            )

            dice_denominator = (
                len(source_set)
                + len(target_set)
            )

            dice = (
                2 * intersection
                / dice_denominator
                if dice_denominator
                else np.nan
            )

            overlap_denominator = min(
                len(source_set),
                len(target_set)
            )

            overlap = (
                intersection
                / overlap_denominator
                if overlap_denominator
                else np.nan
            )

            jaccard_matrix[
                i,
                j
            ] = jaccard

            dice_matrix[
                i,
                j
            ] = dice

            overlap_matrix[
                i,
                j
            ] = overlap

            # Self-transfer is defined as identity.
            if i == j:

                stats = {
                    "mean": 1.0,
                    "median": 1.0,
                    "std": 0.0,
                    "q25": 1.0,
                    "q75": 1.0,
                    "min": 1.0,
                    "max": 1.0,
                    "mean_abs": 1.0,
                    "positive_rate": 1.0,
                    "negative_rate": 0.0,
                    "overshoot_rate": 0.0,
                    "mean_raw_restoration": np.nan,
                    "mean_clean_corrupt_gap": np.nan,
                    "valid_count":
                        len(datasets[target_task]),
                    "dropped_count": 0,
                    "valid_fraction": 1.0,
                }

            else:

                print(
                    f"[MAIN] "
                    f"{source_task} -> "
                    f"{target_task}",
                    datetime.now()
                )

                stats = (
                    evaluate_circuit_transfer(
                        engine,
                        circuits[source_task],
                        datasets[target_task]
                    )
                )

            transfer_mean[
                i,
                j
            ] = stats["mean"]

            transfer_median[
                i,
                j
            ] = stats["median"]

            transfer_std[
                i,
                j
            ] = stats["std"]

            transfer_valid[
                i,
                j
            ] = stats["valid_fraction"]

            positive_rate[
                i,
                j
            ] = stats["positive_rate"]

            mean_abs_transfer[
                i,
                j
            ] = stats["mean_abs"]

            pairwise_rows.append({
                "source_task":
                    source_task,

                "target_task":
                    target_task,

                "mean_transfer":
                    stats["mean"],

                "median_transfer":
                    stats["median"],

                "std_transfer":
                    stats["std"],

                "q25_transfer":
                    stats["q25"],

                "q75_transfer":
                    stats["q75"],

                "min_transfer":
                    stats["min"],

                "max_transfer":
                    stats["max"],

                "mean_abs_transfer":
                    stats["mean_abs"],

                "positive_rate":
                    stats["positive_rate"],

                "negative_rate":
                    stats["negative_rate"],

                "overshoot_rate":
                    stats["overshoot_rate"],

                "valid_samples":
                    stats["valid_count"],

                "dropped_samples":
                    stats["dropped_count"],

                "valid_fraction":
                    stats["valid_fraction"],

                "mean_raw_restoration":
                    stats["mean_raw_restoration"],

                "mean_clean_corrupt_gap":
                    stats["mean_clean_corrupt_gap"],

                "jaccard":
                    jaccard,

                "dice":
                    dice,

                "overlap_coefficient":
                    overlap,
            })

    # Directionality:
    # positive [i,j] means i->j exceeds j->i.
    transfer_asymmetry = (
        transfer_mean
        - transfer_mean.T
    )

    # ================================================================
    # DataFrames
    # ================================================================

    df_transfer = pd.DataFrame(
        transfer_mean,
        index=task_keys,
        columns=task_keys
    )

    df_median = pd.DataFrame(
        transfer_median,
        index=task_keys,
        columns=task_keys
    )

    df_std = pd.DataFrame(
        transfer_std,
        index=task_keys,
        columns=task_keys
    )

    df_valid = pd.DataFrame(
        transfer_valid,
        index=task_keys,
        columns=task_keys
    )

    df_positive = pd.DataFrame(
        positive_rate,
        index=task_keys,
        columns=task_keys
    )

    df_abs = pd.DataFrame(
        mean_abs_transfer,
        index=task_keys,
        columns=task_keys
    )

    df_jaccard = pd.DataFrame(
        jaccard_matrix,
        index=task_keys,
        columns=task_keys
    )

    df_dice = pd.DataFrame(
        dice_matrix,
        index=task_keys,
        columns=task_keys
    )

    df_overlap = pd.DataFrame(
        overlap_matrix,
        index=task_keys,
        columns=task_keys
    )

    df_asymmetry = pd.DataFrame(
        transfer_asymmetry,
        index=task_keys,
        columns=task_keys
    )

    df_pairwise = pd.DataFrame(
        pairwise_rows
    )

    df_discovery = pd.DataFrame(
        discovery_rows
    )

    df_components = pd.DataFrame(
        component_rows
    )

    # ================================================================
    # Save CSVs
    # ================================================================

    prefix = os.path.join(
        args.output_dir,
        clean_model_id
    )

    df_transfer.to_csv(
        prefix
        + "_transfer_mean.csv"
    )

    df_median.to_csv(
        prefix
        + "_transfer_median.csv"
    )

    df_std.to_csv(
        prefix
        + "_transfer_std.csv"
    )

    df_valid.to_csv(
        prefix
        + "_transfer_valid_fraction.csv"
    )

    df_positive.to_csv(
        prefix
        + "_transfer_positive_rate.csv"
    )

    df_abs.to_csv(
        prefix
        + "_transfer_mean_abs.csv"
    )

    df_jaccard.to_csv(
        prefix
        + "_jaccard.csv"
    )

    df_dice.to_csv(
        prefix
        + "_dice.csv"
    )

    df_overlap.to_csv(
        prefix
        + "_overlap_coefficient.csv"
    )

    df_asymmetry.to_csv(
        prefix
        + "_transfer_asymmetry.csv"
    )

    df_pairwise.to_csv(
        prefix
        + "_pairwise_metrics.csv",
        index=False
    )

    df_discovery.to_csv(
        prefix
        + "_discovery_metrics.csv",
        index=False
    )

    df_components.to_csv(
        prefix
        + "_top_components.csv",
        index=False
    )

    # ================================================================
    # Heatmaps
    # ================================================================

    save_heatmap(
        transfer_mean,
        task_keys,
        "Mean Causal Transfer",
        os.path.join(
            plots_dir,
            "transfer_mean_heatmap.png"
        )
    )

    save_heatmap(
        transfer_median,
        task_keys,
        "Median Causal Transfer",
        os.path.join(
            plots_dir,
            "transfer_median_heatmap.png"
        )
    )

    save_heatmap(
        transfer_std,
        task_keys,
        "Transfer Standard Deviation",
        os.path.join(
            plots_dir,
            "transfer_std_heatmap.png"
        )
    )

    save_heatmap(
        transfer_valid,
        task_keys,
        "Valid Sample Fraction",
        os.path.join(
            plots_dir,
            "valid_fraction_heatmap.png"
        )
    )

    save_heatmap(
        jaccard_matrix,
        task_keys,
        "Circuit Jaccard Similarity",
        os.path.join(
            plots_dir,
            "jaccard_heatmap.png"
        )
    )

    save_heatmap(
        dice_matrix,
        task_keys,
        "Circuit Dice Similarity",
        os.path.join(
            plots_dir,
            "dice_heatmap.png"
        )
    )

    save_heatmap(
        transfer_asymmetry,
        task_keys,
        "Directional Transfer Asymmetry",
        os.path.join(
            plots_dir,
            "transfer_asymmetry_heatmap.png"
        )
    )

    print(
        "\n================ RESULTS ================"
    )

    print(
        "\nMean Transfer:"
    )

    print(
        df_transfer.round(4)
    )

    print(
        "\nTransfer Std:"
    )

    print(
        df_std.round(4)
    )

    print(
        "\nJaccard:"
    )

    print(
        df_jaccard.round(4)
    )

    print(
        f"\nAll metrics saved to: "
        f"{args.output_dir}"
    )

    print(
        f"Plots saved to: "
        f"{plots_dir}"
    )


def save_results():
    from huggingface_hub import HfApi, create_repo

    repo_id = "damonsalvatore123/medhalt"
    results_dir = "/root/medhalt-main/medhalt-main/causal_results"

    # Create the repository if it doesn't already exist
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,  # change to True if you want a private repo
    )

    api = HfApi()

    api.upload_folder(
        folder_path=results_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=".",
        commit_message="Upload MedHALT causal transfer results",
    )

    print(f"Uploaded results to: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
    save_results()