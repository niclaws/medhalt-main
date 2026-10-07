import matplotlib.pyplot as plt
from datetime import datetime
import os
import re
import ast
import json
import argparse
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from medhalt.prompts.utils import get_samples
from patching_engine import CenteredKernelAlignment

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now():
    return datetime.now(IST)


# =====================================================================
# 1. Contrastive Dataset Pair Builder
# =====================================================================


class MedHALTPairBuilder:
    VALID_OPTION_LABELS = {"A", "B", "C", "D", "E"}

    def __init__(self, tokenizer, model_type: str = "default", seed: int = 42):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.seed = seed

    def format_prompt(self, user_content: str, model_type: str = "default") -> str:
        if model_type == "qwen":
            return (
                "<|im_start|>system\n"
                "You are a factual medical assistant. Give the concise direct answer "
                "without any thinking process or inner monologue.<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif model_type == "med42":
            # Llama-3 / Med42-v2 chat template
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful, respectful, and honest clinical assistant. "
                "Provide the direct and concise answer.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif model_type == "gemma":
            # Strictly end on model turn without trailing newline
            return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
        elif model_type in {"phi", "mediphi"}:
            # Native Phi-3 / MediPhi instruction format
            return f"<|user|>\n{user_content}<|end|>\n" "<|assistant|>\n"
        # Default for Mistral-7B (standard instruction format)
        return user_content

    @classmethod
    def normalize_option_label(cls, value) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip().upper()
        match = re.fullmatch(r"([A-E])(?:[\).:]?)", cleaned)
        if not match:
            return None
        label = match.group(1)
        return label if label in cls.VALID_OPTION_LABELS else None

    @staticmethod
    def option_index_to_label(value) -> Optional[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if 0 <= index < 5:
            return chr(ord("A") + index)
        return None

    @staticmethod
    def parse_numeric_options(options_value) -> List[str]:
        parsed = options_value
        if isinstance(options_value, str):
            try:
                parsed = ast.literal_eval(options_value)
            except (ValueError, SyntaxError):
                return []

        if isinstance(parsed, dict):
            numeric_items = []
            for key, value in parsed.items():
                key_str = str(key).strip()
                if key_str.isdigit():
                    numeric_items.append((int(key_str), str(value)))
            numeric_items.sort(key=lambda x: x[0])
            return [value for _, value in numeric_items]

        if isinstance(parsed, (list, tuple)):
            return [str(value) for value in parsed]

        return []

    @classmethod
    def render_lettered_options(cls, options_value) -> Tuple[str, List[str]]:
        values = cls.parse_numeric_options(options_value)
        if not values:
            return str(options_value), []
        lines = [f"{chr(ord('A') + i)}) {text}" for i, text in enumerate(values)]
        return "\n".join(lines), values

    def encode_single_token(self, token_str: str) -> Optional[int]:
        cleaned = str(token_str).strip()
        if not cleaned:
            return None

        # For Gemma/Llama models, test single-token continuation directly
        for candidate in [cleaned, " " + cleaned]:
            tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
            # Remove leading BOS if present
            if tokens and tokens[0] == self.tokenizer.bos_token_id:
                tokens = tokens[1:]
            if len(tokens) == 1:
                return tokens[0]

        tokens = self.tokenizer.encode(cleaned, add_special_tokens=False)
        if tokens and tokens[0] == self.tokenizer.bos_token_id:
            tokens = tokens[1:]
        return tokens[0] if tokens else None

    def get_first_distinct_url_token_pair(self, clean_pmid: str, corrupt_pmid: str):
        clean_tokens = self.tokenizer.encode(
            str(clean_pmid).strip(), add_special_tokens=False
        )
        corrupt_tokens = self.tokenizer.encode(
            str(corrupt_pmid).strip(), add_special_tokens=False
        )
        if not clean_tokens or not corrupt_tokens:
            return None

        for position in range(min(len(clean_tokens), len(corrupt_tokens))):
            if clean_tokens[position] != corrupt_tokens[position]:
                return {
                    "common_prefix_ids": clean_tokens[:position],
                    "target_id": clean_tokens[position],
                    "halluc_id": corrupt_tokens[position],
                    "position": position,
                    "clean_tokens": clean_tokens,
                    "corrupt_tokens": corrupt_tokens,
                }
        return None

    def create_pair(
        self,
        clean_txt: str,
        corrupt_txt: str,
        target_str: str,
        halluc_str: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict]:
        t_id = self.encode_single_token(target_str)
        h_id = self.encode_single_token(halluc_str)
        if t_id is None or h_id is None or t_id == h_id:
            return None

        clean_prompt = self.format_prompt(clean_txt, self.model_type)
        corrupt_prompt = self.format_prompt(corrupt_txt, self.model_type)
        pair = {
            "clean_ids": self.tokenizer(clean_prompt, return_tensors="pt").input_ids,
            "corrupt_ids": self.tokenizer(
                corrupt_prompt, return_tensors="pt"
            ).input_ids,
            "target_id": t_id,
            "halluc_id": h_id,
            "target_text": str(target_str),
            "halluc_text": str(halluc_str),
        }
        if metadata:
            pair.update(metadata)
        return pair

    def prepare_dataset(
        self,
        task_name: str,
        n_samples: int = 50,
        task_seed: Optional[int] = None,
    ) -> List[Dict]:
        raw_samples = list(
            get_samples(dataset_name=task_name, shots=0, prompt_version="v0")
        )
        seed = self.seed if task_seed is None else task_seed
        rng = random.Random(seed)
        rng.shuffle(raw_samples)
        pairs: List[Dict] = []

        if task_name in {"pmid2title", "title2pub"}:
            real_samples = [
                s
                for s in raw_samples
                if str(s.get("pubmed_data_type", "")).lower() == "real_data"
                and str(s.get("is_paper_exists", "")).lower() == "yes"
                and pd.notna(s.get("PMID"))
                and pd.notna(s.get("Title"))
                and str(s.get("PMID", "")).strip()
                and str(s.get("Title", "")).strip()
            ]
            rng.shuffle(real_samples)
            used_pmids = set()

            for i, clean_sample in enumerate(real_samples):
                if len(pairs) >= n_samples:
                    break

                clean_pmid = str(clean_sample["PMID"]).strip()
                clean_title = str(clean_sample["Title"]).strip()
                if clean_pmid in used_pmids:
                    continue

                for j, corrupt_sample in enumerate(real_samples):
                    if i == j:
                        continue

                    corrupt_pmid = str(corrupt_sample["PMID"]).strip()
                    corrupt_title = str(corrupt_sample["Title"]).strip()
                    if corrupt_pmid in used_pmids or corrupt_pmid == clean_pmid:
                        continue

                    if task_name == "pmid2title":
                        target_word = clean_title.split()[0]
                        halluc_word = corrupt_title.split()[0]
                        target_id = self.encode_single_token(target_word)
                        halluc_id = self.encode_single_token(halluc_word)
                        if (
                            target_id is None
                            or halluc_id is None
                            or target_id == halluc_id
                        ):
                            continue

                        clean = f"Provide the title of PubMed paper with PMID {clean_pmid}:\nTitle:"
                        corrupt = f"Provide the title of PubMed paper with PMID {corrupt_pmid}:\nTitle:"
                        pair = self.create_pair(
                            clean,
                            corrupt,
                            target_word,
                            halluc_word,
                            metadata={
                                "sample_id": f"{clean_pmid}__vs__{corrupt_pmid}",
                                "clean_ref": clean_pmid,
                                "corrupt_ref": corrupt_pmid,
                            },
                        )
                        if pair is None:
                            continue

                    else:
                        divergence = self.get_first_distinct_url_token_pair(
                            clean_pmid, corrupt_pmid
                        )
                        if divergence is None:
                            continue

                        common_prefix_ids = divergence["common_prefix_ids"]
                        target_id = divergence["target_id"]
                        halluc_id = divergence["halluc_id"]

                        clean = (
                            f"Article Title: {clean_title}\n"
                            "PubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
                        )
                        corrupt = (
                            f"Article Title: {corrupt_title}\n"
                            "PubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
                        )

                        clean_ids = self.tokenizer(
                            self.format_prompt(clean, self.model_type),
                            return_tensors="pt",
                        ).input_ids
                        corrupt_ids = self.tokenizer(
                            self.format_prompt(corrupt, self.model_type),
                            return_tensors="pt",
                        ).input_ids

                        if common_prefix_ids:
                            shared_prefix = torch.tensor(
                                [common_prefix_ids], dtype=clean_ids.dtype
                            )
                            clean_ids = torch.cat([clean_ids, shared_prefix], dim=1)
                            corrupt_ids = torch.cat([corrupt_ids, shared_prefix], dim=1)

                        pair = {
                            "clean_ids": clean_ids,
                            "corrupt_ids": corrupt_ids,
                            "target_id": target_id,
                            "halluc_id": halluc_id,
                            "target_text": clean_pmid,
                            "halluc_text": corrupt_pmid,
                            "sample_id": f"{clean_pmid}__vs__{corrupt_pmid}",
                            "clean_ref": clean_pmid,
                            "corrupt_ref": corrupt_pmid,
                            "pmid_token_position": divergence["position"],
                            "shared_prefix_length": len(common_prefix_ids),
                        }

                    pairs.append(pair)
                    used_pmids.add(clean_pmid)
                    used_pmids.add(corrupt_pmid)
                    break

            return pairs

        for sample_idx, sample in enumerate(raw_samples):
            if len(pairs) >= n_samples:
                break

            sample_id = str(sample.get("id", f"{task_name}_{sample_idx}"))
            q = sample.get("question", "")
            rendered_options, option_values = self.render_lettered_options(
                sample.get("options", "")
            )

            if not option_values:
                continue

            if task_name == "FCT":
                correct = self.option_index_to_label(sample.get("correct_index"))
                suggested_wrong = self.option_index_to_label(
                    sample.get("student_index")
                )

                if (
                    correct is None
                    or suggested_wrong is None
                    or correct == suggested_wrong
                    or int(sample.get("correct_index")) >= len(option_values)
                    or int(sample.get("student_index")) >= len(option_values)
                ):
                    continue

                clean = f"Question: {q}\nOptions:\n{rendered_options}\nAnswer:"
                corrupt = (
                    f"Question: {q}\n"
                    f"Options:\n{rendered_options}\n"
                    f"Note: Experts state the answer is {suggested_wrong}.\n"
                    "Answer:"
                )

                pair = self.create_pair(
                    clean,
                    corrupt,
                    correct,
                    suggested_wrong,
                    metadata={
                        "sample_id": sample_id,
                        "correct_index": int(sample.get("correct_index")),
                        "contrast_index": int(sample.get("student_index")),
                        "correct_answer_text": str(sample.get("correct_answer", "")),
                        "contrast_answer_text": str(sample.get("student_answer", "")),
                    },
                )

            elif task_name == "Nota":
                correct_index_raw = sample.get("correct_index")
                correct = self.option_index_to_label(correct_index_raw)

                if correct is None:
                    continue

                try:
                    correct_index = int(correct_index_raw)
                except (TypeError, ValueError):
                    continue

                if not (0 <= correct_index < len(option_values)):
                    continue

                alternate_index = None
                for field in ("student_index", "distractor_index"):
                    value = sample.get(field)
                    if value is not None and not (
                        isinstance(value, float) and np.isnan(value)
                    ):
                        try:
                            candidate = int(value)
                        except (TypeError, ValueError):
                            continue
                        if (
                            0 <= candidate < len(option_values)
                            and candidate != correct_index
                        ):
                            alternate_index = candidate
                            break

                if alternate_index is None:
                    for candidate in range(len(option_values)):
                        if candidate != correct_index:
                            alternate_index = candidate
                            break

                distractor = self.option_index_to_label(alternate_index)
                if distractor is None or distractor == correct:
                    continue

                clean = f"Question: {q}\nOptions:\n{rendered_options}\nAnswer:"
                corrupt = (
                    f"Question: {q}\n"
                    f"Options:\n{rendered_options}\n"
                    f"Hint: The answer is {distractor}.\n"
                    "Answer:"
                )

                pair = self.create_pair(
                    clean,
                    corrupt,
                    correct,
                    distractor,
                    metadata={
                        "sample_id": sample_id,
                        "correct_index": correct_index,
                        "contrast_index": alternate_index,
                        "correct_answer_text": str(sample.get("correct_answer", "")),
                        "contrast_answer_text": option_values[alternate_index],
                    },
                )
            else:
                raise ValueError(f"Unsupported task_name={task_name!r}")

            if pair is not None:
                pairs.append(pair)

        return pairs


# =====================================================================
# 2. Optimized Causal Engine with Extraction & Intervention Hooks
# =====================================================================


class CausalEngine:
    def __init__(self, model_name: str, is_qwen: bool = False):
        print(f"\n[Engine] Loading {model_name}...", now())
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Allocate explicit VRAM headroom per 11GB GPU (reserving ~1.5 GB for activations & hooks)
        # num_gpus = torch.cuda.device_count()
        # if num_gpus > 1:
        #     max_memory = {i: "9.5GiB" for i in range(num_gpus)}
        # else:
        #     max_memory = {0: "40GiB"}

        dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            # max_memory=max_memory,
            trust_remote_code=True,
        )
        self.model.eval()

        # 1. Direct path lookup prioritizing the Language Model text backbone
        layers_module = None
        for attr_path in [
            (
                "model",
                "language_model",
                "model",
                "layers",
            ),  # Gemma-3 multimodal wrapper
            ("language_model", "model", "layers"),  # Gemma-3 text backbone
            ("model", "text_model", "layers"),  # Gemma-3 variant
            ("model", "layers"),  # Standard LLaMA / Mistral / Qwen / Baichuan
            ("layers",),
        ]:
            curr = self.model
            for step in attr_path:
                curr = getattr(curr, step, None)
                if curr is None:
                    break
            if (
                curr is not None
                and isinstance(curr, (torch.nn.ModuleList, list))
                and len(curr) > 0
            ):
                layers_module = curr
                print(
                    f"[Engine] Found language decoder layers via direct path: {'.'.join(attr_path)} (Length: {len(curr)})",
                    now(),
                )
                break

        # 2. Resilient traversal: exclude vision tower and pick the largest decoder ModuleList
        if layers_module is None:
            candidate_lists = []
            for name, module in self.model.named_modules():
                # Explicitly ignore vision encoder modules
                if (
                    "vision" in name.lower()
                    or "image" in name.lower()
                    or "projector" in name.lower()
                ):
                    continue
                if (
                    name.endswith("layers")
                    and isinstance(module, torch.nn.ModuleList)
                    and len(module) > 0
                ):
                    sample_block = module[0]
                    if hasattr(sample_block, "self_attn") or hasattr(
                        sample_block, "mlp"
                    ):
                        candidate_lists.append((len(module), name, module))

            if candidate_lists:
                candidate_lists.sort(key=lambda x: x[0], reverse=True)
                layers_module = candidate_lists[0][2]
                print(
                    f"[Engine] Selected language layers: '{candidate_lists[0][1]}' (Length: {len(layers_module)})",
                    now(),
                )

        if layers_module is None:
            raise AttributeError(
                f"Could not locate text decoder layers for '{model_name}'."
            )

        self.layers = layers_module

        # Resolve config parameters accurately for standard and multimodal models
        cfg = getattr(self.model.config, "text_config", self.model.config)
        self.num_layers = len(self.layers)

        # 1. Resolve head_dim
        self.head_dim = getattr(cfg, "head_dim", None)
        hidden_sz = getattr(cfg, "hidden_size", None)

        # 2. Resolve attention head count strictly matching self_attn projection dimension
        # Check explicit query head counts first
        num_heads = getattr(cfg, "num_attention_heads", None)

        # In Gemma-3 / GQA models, ensure num_heads * head_dim == hidden_size
        if hidden_sz is not None and self.head_dim is not None:
            derived_heads = hidden_sz // self.head_dim
            if num_heads is None or (num_heads * self.head_dim != hidden_sz):
                print(
                    f"[Engine Fix] Correcting num_heads from {num_heads} to {derived_heads} "
                    f"(hidden_size={hidden_sz}, head_dim={self.head_dim})",
                    now(),
                )
                num_heads = derived_heads

        if self.head_dim is None and hidden_sz is not None and num_heads is not None:
            self.head_dim = hidden_sz // num_heads

        self.num_heads = num_heads

        print(
            f"[Engine] Model Loaded. Verified Layers: {self.num_layers}, "
            f"Query Heads: {self.num_heads}, Head Dim: {self.head_dim}, Hidden Dim: {hidden_sz}",
            now(),
        )

    def get_logit_diff(
        self, input_ids: torch.Tensor, target_id: int, halluc_id: int
    ) -> float:
        with torch.inference_mode():
            outputs = self.model(input_ids.to(self.model.device))
            last_logits = outputs.logits[0, -1, :]
            return (last_logits[target_id] - last_logits[halluc_id]).item()

    def cache_clean_activations(
        self, clean_ids: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        head_cache = {}
        mlp_cache = {}
        hooks = []

        def get_attn_hook(l):
            def hook(module, inp, outp):
                act = outp[0] if isinstance(outp, tuple) else outp
                head_cache[l] = (
                    act.detach().cpu()
                )  # Offload to CPU cache to protect 11GB VRAM

            return hook

        def get_mlp_hook(l):
            def hook(module, inp, outp):
                act = outp[0] if isinstance(outp, tuple) else outp
                mlp_cache[l] = act.detach().cpu()  # Offload to CPU cache

            return hook

        for l in range(self.num_layers):
            hooks.append(
                self.layers[l].self_attn.register_forward_hook(get_attn_hook(l))
            )
            hooks.append(self.layers[l].mlp.register_forward_hook(get_mlp_hook(l)))

        with torch.inference_mode():
            self.model(clean_ids.to(self.model.device))

        for h in hooks:
            h.remove()

        return head_cache, mlp_cache

    def extract_task_representations(
        self, samples: List[Dict], max_samples: int = 50
    ) -> Dict[int, torch.Tensor]:
        """Extracts token decision representations across all layers for CKA."""
        layer_reps = {l: [] for l in range(self.num_layers)}
        eval_samples = samples[:max_samples]
        device = self.model.device

        for sample in eval_samples:
            clean_ids = sample["clean_ids"].to(device)
            sample_cache = {}
            hooks = []

            def make_hook(layer_i):
                def hook_fn(module, inp, outp):
                    # Handle Gemma-3 tuple unpack safely
                    if isinstance(outp, tuple):
                        act = outp[0]
                    else:
                        act = outp

                    # Ensure act is a 3D tensor [B, S, D]
                    if hasattr(act, "last_hidden_state"):
                        act = act.last_hidden_state

                    sample_cache[layer_i] = (
                        act[:, -1, :].detach().to(torch.float32).cpu()
                    )

                return hook_fn

            for l in range(self.num_layers):
                hooks.append(self.layers[l].register_forward_hook(make_hook(l)))

            with torch.inference_mode():
                self.model(clean_ids)

            for h in hooks:
                h.remove()

            for l in range(self.num_layers):
                layer_reps[l].append(sample_cache[l])

        return {l: torch.cat(layer_reps[l], dim=0) for l in range(self.num_layers)}

    def patch_heads_layer_batched(
        self,
        corrupt_ids: torch.Tensor,
        target_id: int,
        halluc_id: int,
        layer_idx: int,
        clean_act: torch.Tensor,
        chunk_size: int = 32,
    ) -> np.ndarray:
        target_diffs = np.zeros(self.num_heads, dtype=np.float32)
        total_heads = self.num_heads
        first_device = self.model.device
        layer_device = next(self.layers[layer_idx].parameters()).device

        for start_idx in range(0, total_heads, chunk_size):
            end_idx = min(start_idx + chunk_size, total_heads)
            active_heads = list(range(start_idx, end_idx))
            batch_k = len(active_heads)

            batched_corrupt = corrupt_ids.repeat(batch_k, 1).to(first_device)

            def batched_patch_hook(module, inp, outp):
                curr = outp[0].clone() if isinstance(outp, tuple) else outp.clone()
                B, S, hidden = curr.shape
                clean = clean_act.to(curr.device)
                Bc, Sc, _ = clean.shape

                curr_reshaped = curr.reshape(B, S, self.num_heads, self.head_dim)
                clean_reshaped = clean.reshape(Bc, Sc, self.num_heads, self.head_dim)

                for b_idx, h_idx in enumerate(active_heads):
                    if S != Sc:
                        curr_reshaped[b_idx, -1, h_idx, :] = clean_reshaped[
                            0, -1, h_idx, :
                        ]
                    else:
                        curr_reshaped[b_idx, :, h_idx, :] = clean_reshaped[
                            0, :, h_idx, :
                        ]

                patched = curr_reshaped.reshape(B, S, hidden)
                return (patched,) + outp[1:] if isinstance(outp, tuple) else patched

            handle = self.layers[layer_idx].self_attn.register_forward_hook(
                batched_patch_hook
            )
            with torch.inference_mode():
                outputs = self.model(batched_corrupt)
                logits = outputs.logits[:, -1, :]
                diffs = (
                    (logits[:, target_id] - logits[:, halluc_id]).float().cpu().numpy()
                )

            handle.remove()
            target_diffs[start_idx:end_idx] = diffs

        return target_diffs

    def patch_single_component(
        self,
        corrupt_ids: torch.Tensor,
        target_id: int,
        halluc_id: int,
        comp_type: str,
        layer_idx: int,
        clean_act: torch.Tensor,
    ) -> float:
        def hook_fn(module, inp, outp):
            if clean_act.shape[1] != outp.shape[1]:
                curr = outp.clone()
                curr[:, -1, :] = clean_act[:, -1, :].to(outp.device)
                return curr
            return clean_act.to(outp.device)

        handle = self.layers[layer_idx].mlp.register_forward_hook(hook_fn)
        with torch.inference_mode():
            outputs = self.model(corrupt_ids.to(self.model.device))
            logits = outputs.logits[0, -1, :]
            patched_diff = (logits[target_id] - logits[halluc_id]).item()

        handle.remove()
        return patched_diff


# =====================================================================
# 3. Discovery & Evaluation Logic
# =====================================================================


def bootstrap_mean_ci(values, n_bootstrap=2000, confidence=0.95, seed=42):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    if n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = arr[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def rank_components(head_aie: np.ndarray, mlp_aie: np.ndarray) -> List[Dict]:
    ranked = []
    num_layers, num_heads = head_aie.shape
    for layer in range(num_layers):
        for head in range(num_heads):
            score = float(head_aie[layer, head])
            ranked.append(
                {
                    "component": ("head", layer, head),
                    "label": f"L{layer}H{head}",
                    "component_type": "head",
                    "layer": layer,
                    "head": head,
                    "aie": score,
                    "abs_aie": abs(score),
                }
            )
        score = float(mlp_aie[layer])
        ranked.append(
            {
                "component": ("mlp", layer, None),
                "label": f"L{layer}MLP",
                "component_type": "mlp",
                "layer": layer,
                "head": None,
                "aie": score,
                "abs_aie": abs(score),
            }
        )

    ranked.sort(key=lambda row: row["abs_aie"], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def circuit_from_ranking(ranking: List[Dict], k: int):
    selected = ranking[:k]
    return (
        [row["component"] for row in selected],
        [row["label"] for row in selected],
        [float(row["aie"]) for row in selected],
    )


def circuit_similarity(labels_a: List[str], labels_b: List[str]) -> Dict[str, float]:
    a, b = set(labels_a), set(labels_b)
    inter = len(a & b)
    union = len(a | b)
    denom = len(a) + len(b)
    min_size = min(len(a), len(b))
    return {
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else np.nan,
        "dice": (2 * inter / denom) if denom else np.nan,
        "overlap_coefficient": inter / min_size if min_size else np.nan,
    }


def summarize_transfer_rows(
    rows: List[Dict],
    threshold: float,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, object]:
    usable = [
        row
        for row in rows
        if row.get("patched", False)
        and abs(float(row["denominator"])) >= threshold
        and np.isfinite(float(row["normalized_transfer"]))
    ]

    total_count = len(rows)
    if not usable:
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
            "bootstrap_mean_ci_low": np.nan,
            "bootstrap_mean_ci_high": np.nan,
            "valid_count": 0,
            "dropped_count": total_count,
            "valid_fraction": 0.0,
        }

    scores = np.asarray([row["normalized_transfer"] for row in usable], dtype=float)
    raw = np.asarray([row["raw_restoration"] for row in usable], dtype=float)
    gaps = np.asarray([row["denominator"] for row in usable], dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(
        scores,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "q25": float(np.quantile(scores, 0.25)),
        "q75": float(np.quantile(scores, 0.75)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean_abs": float(np.mean(np.abs(scores))),
        "positive_rate": float(np.mean(scores > 0)),
        "negative_rate": float(np.mean(scores < 0)),
        "overshoot_rate": float(np.mean(scores > 1)),
        "mean_raw_restoration": float(np.mean(raw)),
        "mean_clean_corrupt_gap": float(np.mean(gaps)),
        "bootstrap_mean_ci_low": ci_low,
        "bootstrap_mean_ci_high": ci_high,
        "valid_count": int(len(scores)),
        "dropped_count": int(total_count - len(scores)),
        "valid_fraction": float(len(scores) / max(1, total_count)),
    }


def discover_circuit(
    engine: CausalEngine,
    dataset: List[Dict],
    min_denom: float = 1e-4,
    head_batch_size: int = 32,
) -> Dict[str, object]:
    sample_rows = []
    per_sample_head = []
    per_sample_mlp = []
    valid_denoms = []

    print(
        f"\n[Discovery] Starting vectorized discovery on {len(dataset)} samples...",
        now(),
    )

    for idx, sample in enumerate(
        tqdm(dataset, desc="Scanning Components", leave=False)
    ):
        clean_ids = sample["clean_ids"]
        corrupt_ids = sample["corrupt_ids"]
        t_id = sample["target_id"]
        h_id = sample["halluc_id"]

        d_clean = engine.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = engine.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        base = {
            "sample_index": idx,
            "sample_id": sample.get("sample_id", str(idx)),
            "target_text": sample.get("target_text", ""),
            "halluc_text": sample.get("halluc_text", ""),
            "target_id": t_id,
            "halluc_id": h_id,
            "d_clean": d_clean,
            "d_corrupt": d_corrupt,
            "denominator": denom,
            "abs_denominator": abs(denom),
            "used_at_min_threshold": abs(denom) >= min_denom,
        }
        sample_rows.append(base)

        if abs(denom) < min_denom:
            continue

        head_cache, mlp_cache = engine.cache_clean_activations(clean_ids)
        head_effect = np.zeros((engine.num_layers, engine.num_heads), dtype=np.float32)
        mlp_effect = np.zeros(engine.num_layers, dtype=np.float32)

        for layer in range(engine.num_layers):
            layer_diffs = engine.patch_heads_layer_batched(
                corrupt_ids,
                t_id,
                h_id,
                layer,
                head_cache[layer],
                chunk_size=head_batch_size,
            )
            head_effect[layer, :] = (layer_diffs - d_corrupt) / denom

        for layer in range(engine.num_layers):
            d_patched = engine.patch_single_component(
                corrupt_ids, t_id, h_id, "mlp", layer, mlp_cache[layer]
            )
            mlp_effect[layer] = (d_patched - d_corrupt) / denom

        per_sample_head.append(head_effect)
        per_sample_mlp.append(mlp_effect)
        valid_denoms.append(denom)
        del head_cache, mlp_cache

    valid_count = len(valid_denoms)
    if valid_count == 0:
        raise RuntimeError("Circuit discovery produced zero valid samples.")

    head_stack = np.stack(per_sample_head, axis=0)
    mlp_stack = np.stack(per_sample_mlp, axis=0)
    denom_array = np.asarray(valid_denoms, dtype=float)
    head_aie = head_stack.mean(axis=0)
    mlp_aie = mlp_stack.mean(axis=0)
    ranking = rank_components(head_aie, mlp_aie)

    return {
        "head_aie": head_aie,
        "mlp_aie": mlp_aie,
        "ranking": ranking,
        "valid_count": valid_count,
        "total_count": len(dataset),
        "valid_fraction": valid_count / max(1, len(dataset)),
        "sample_rows": sample_rows,
        "valid_denominators": denom_array,
        "per_sample_head_aie": head_stack,
        "per_sample_mlp_aie": mlp_stack,
        "mean_abs_head_aie": float(np.mean(np.abs(head_aie))),
        "max_abs_head_aie": float(np.max(np.abs(head_aie))),
        "mean_abs_mlp_aie": float(np.mean(np.abs(mlp_aie))),
        "max_abs_mlp_aie": float(np.max(np.abs(mlp_aie))),
        "mean_clean_corrupt_gap": float(np.mean(denom_array)),
        "std_clean_corrupt_gap": (
            float(np.std(denom_array, ddof=1)) if valid_count > 1 else 0.0
        ),
    }


def ranking_at_threshold(
    discovery_result: Dict, threshold: float
) -> Optional[List[Dict]]:
    denoms = discovery_result["valid_denominators"]
    mask = np.abs(denoms) >= threshold
    if not np.any(mask):
        return None
    head = discovery_result["per_sample_head_aie"][mask].mean(axis=0)
    mlp = discovery_result["per_sample_mlp_aie"][mask].mean(axis=0)
    return rank_components(head, mlp)


def evaluate_circuit_transfer(
    engine: CausalEngine,
    circuit: List[Tuple],
    target_dataset: List[Dict],
    min_denom: float = 1e-4,
) -> Dict[str, object]:
    per_sample_rows = []

    for idx, sample in enumerate(
        tqdm(target_dataset, desc="Evaluating Transfer", leave=False)
    ):
        clean_ids = sample["clean_ids"]
        corrupt_ids = sample["corrupt_ids"]
        t_id = sample["target_id"]
        h_id = sample["halluc_id"]

        d_clean = engine.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = engine.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        base = {
            "sample_index": idx,
            "sample_id": sample.get("sample_id", str(idx)),
            "target_text": sample.get("target_text", ""),
            "halluc_text": sample.get("halluc_text", ""),
            "target_id": t_id,
            "halluc_id": h_id,
            "d_clean": d_clean,
            "d_corrupt": d_corrupt,
            "denominator": denom,
            "abs_denominator": abs(denom),
        }

        if abs(denom) < min_denom:
            per_sample_rows.append(
                {
                    **base,
                    "patched": False,
                    "d_patched": np.nan,
                    "raw_restoration": np.nan,
                    "normalized_transfer": np.nan,
                }
            )
            continue

        head_cache, mlp_cache = engine.cache_clean_activations(clean_ids)
        handles = []

        for comp_type, layer_idx, head_idx in circuit:
            if comp_type == "mlp":
                act = mlp_cache[layer_idx]

                def get_mlp_hook(a):
                    def hook(m, inp, outp):
                        clean = a.to(outp.device)
                        if clean.shape[1] != outp.shape[1]:
                            curr = outp.clone()
                            curr[:, -1, :] = clean[:, -1, :]
                            return curr
                        return clean

                    return hook

                handles.append(
                    engine.layers[layer_idx].mlp.register_forward_hook(
                        get_mlp_hook(act)
                    )
                )

            elif comp_type == "head":
                act = head_cache[layer_idx]

                def get_head_hook(a, target_h):
                    def hook(m, inp, outp):
                        curr = (
                            outp[0].clone() if isinstance(outp, tuple) else outp.clone()
                        )
                        B, S, hidden = curr.shape
                        clean = a.to(curr.device)
                        Bc, Sc, clean_hidden = clean.shape
                        cr = curr.reshape(B, S, engine.num_heads, engine.head_dim)
                        ca = clean.reshape(Bc, Sc, engine.num_heads, engine.head_dim)
                        if S != Sc:
                            cr[:, -1, target_h, :] = ca[:, -1, target_h, :]
                        else:
                            cr[:, :, target_h, :] = ca[:, :, target_h, :]
                        patched = cr.reshape(B, S, hidden)
                        return (
                            (patched,) + outp[1:]
                            if isinstance(outp, tuple)
                            else patched
                        )

                    return hook

                handles.append(
                    engine.layers[layer_idx].self_attn.register_forward_hook(
                        get_head_hook(act, head_idx)
                    )
                )

        try:
            with torch.inference_mode():
                outputs = engine.model(corrupt_ids.to(engine.model.device))
                logits = outputs.logits[0, -1, :]
                d_patched = float((logits[t_id] - logits[h_id]).item())
        finally:
            for handle in handles:
                handle.remove()

        raw_restoration = d_patched - d_corrupt
        normalized_transfer = raw_restoration / denom
        per_sample_rows.append(
            {
                **base,
                "patched": True,
                "d_patched": d_patched,
                "raw_restoration": raw_restoration,
                "normalized_transfer": normalized_transfer,
            }
        )
        del head_cache, mlp_cache

    return {"per_sample": per_sample_rows}


def sample_uniform_random_circuit(universe, k, rng, exclude=None):
    exclude = set(exclude or [])
    candidates = [component for component in universe if component not in exclude]
    if len(candidates) < k:
        candidates = list(universe)
    return rng.sample(candidates, k)


def sample_layer_matched_random_circuit(universe, reference, rng):
    reference_set = set(reference)
    selected, selected_set = [], set()
    by_layer_type = {}
    by_layer = {}
    for component in universe:
        comp_type, layer, _ = component
        by_layer_type.setdefault((layer, comp_type), []).append(component)
        by_layer.setdefault(layer, []).append(component)

    for ref in reference:
        comp_type, layer, _ = ref
        candidates = [
            c
            for c in by_layer_type.get((layer, comp_type), [])
            if c not in reference_set and c not in selected_set
        ]
        if not candidates:
            candidates = [
                c
                for c in by_layer.get(layer, [])
                if c not in reference_set and c not in selected_set
            ]
        if not candidates:
            candidates = [c for c in universe if c not in selected_set]
        choice = rng.choice(candidates)
        selected.append(choice)
        selected_set.add(choice)
    return selected


def empirical_p_values(observed, random_values):
    arr = np.asarray([v for v in random_values if np.isfinite(v)], dtype=float)
    if not np.isfinite(observed) or len(arr) == 0:
        return np.nan, np.nan
    one_sided = (1 + np.sum(arr >= observed)) / (len(arr) + 1)
    two_sided = (1 + np.sum(np.abs(arr) >= abs(observed))) / (len(arr) + 1)
    return float(one_sided), float(two_sided)


def save_heatmap(matrix, labels, title, path, fmt=".3f"):
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target Task")
    ax.set_ylabel("Source Task")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "NA" if np.isnan(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cka_layer_matrix_plot(
    cka_matrix: np.ndarray, task_a: str, task_b: str, output_path: str
):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cka_matrix, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel(f"Layers ({task_b})")
    ax.set_ylabel(f"Layers ({task_a})")
    ax.set_title(f"Layer-to-Layer Linear CKA Alignment: {task_a} vs {task_b}")
    fig.colorbar(im, ax=ax, label="Linear CKA Similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_head_aie_heatmap(head_aie, task_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(head_aie, aspect="auto")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{task_name}: Attention Head AIE")
    fig.colorbar(im, ax=ax, label="Average Indirect Effect")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mlp_aie_plot(mlp_aie, task_name, output_path):
    layers = np.arange(len(mlp_aie))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layers, mlp_aie)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AIE")
    ax.set_title(f"{task_name}: MLP AIE by Layer")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_circuit_plot(labels, scores, task_name, output_path):
    order = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(order, scores)
    ax.set_yticks(order)
    ax.set_yticklabels(labels)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Signed Average Indirect Effect")
    ax.set_title(f"{task_name}: Top Circuit Components")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_denominator_histogram(values, task_name, output_path):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.abs(values), bins=30)
    ax.set_xlabel("|d_clean - d_corrupt|")
    ax.set_ylabel("Count")
    ax.set_title(f"{task_name}: Clean-Corrupt Denominator Magnitudes")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_k_sensitivity_plot(df, output_path):
    subset = df[df["source_task"] != df["target_task"]].copy()
    if subset.empty:
        return
    grouped = (
        subset.groupby("k", as_index=False)["mean_transfer"].mean().sort_values("k")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["k"], grouped["mean_transfer"], marker="o")
    ax.set_xlabel("Circuit size k")
    ax.set_ylabel("Mean off-diagonal causal transfer")
    ax.set_title("Causal Transfer vs Circuit Size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sensitivity_plot(df, primary_k, output_path):
    subset = df[
        (df["k"] == primary_k) & (df["source_task"] != df["target_task"])
    ].copy()
    if subset.empty:
        return
    grouped = (
        subset.groupby("denom_threshold", as_index=False)["mean_transfer"]
        .mean()
        .sort_values("denom_threshold")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped["denom_threshold"], grouped["mean_transfer"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Minimum |clean-corrupt denominator|")
    ax.set_ylabel("Mean off-diagonal causal transfer")
    ax.set_title(f"Denominator-Threshold Sensitivity (k={primary_k})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_random_control_plot(df, control_type, output_path):
    subset = df[df["control_type"] == control_type].copy()
    if subset.empty:
        return
    subset["pair"] = subset["source_task"] + "→" + subset["target_task"]
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(subset)), 6))
    ax.errorbar(
        x,
        subset["random_mean"],
        yerr=subset["random_std"].fillna(0.0),
        fmt="o",
        capsize=3,
        label="Random control mean ± SD",
    )
    ax.scatter(x, subset["observed_mean"], marker="x", s=70, label="Discovered circuit")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["pair"], rotation=45, ha="right")
    ax.set_ylabel("Mean causal transfer")
    ax.set_title(f"Observed vs {control_type.replace('_', ' ').title()} Controls")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 4. Main Experiment Orchestrator
# =====================================================================


def upload_results_to_huggingface(output_dir, repo_id, path_in_repo, private=False):
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=f"Upload MedHALT causal transfer results: {path_in_repo}",
    )
    print(
        f"[HF] Uploaded to https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Med-HALT paper-grade causal transfer runner"
    )
    parser.add_argument(
        "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--is_qwen", action="store_true")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--top_k", type=int, default=20, help="Primary/headline circuit size"
    )
    parser.add_argument(
        "--k_values", nargs="+", type=int, default=[5, 10, 15, 20, 30, 50]
    )
    parser.add_argument(
        "--head_batch_size",
        type=int,
        default=32,
        help="Chunk size for batched head sweep",
    )
    parser.add_argument(
        "--denom_thresholds",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1],
        help="Sensitivity thresholds; all must be >= --min_denom",
    )
    parser.add_argument("--min_denom", type=float, default=1e-4)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--bootstrap_confidence", type=float, default=0.95)
    parser.add_argument("--num_random_controls", type=int, default=20)
    parser.add_argument(
        "--control_samples",
        type=int,
        default=50,
        help="Sample subset size for random controls",
    )
    parser.add_argument(
        "--cka_samples",
        type=int,
        default=50,
        help="Sample count per task for CKA analysis",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./causal_results")
    parser.add_argument("--hf_repo_id", type=str, default="damonsalvatore123/medhalt")
    parser.add_argument("--hf_path_in_repo", type=str, default="")
    parser.add_argument("--hf_private", action="store_true")
    parser.add_argument("--skip_hf_upload", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    clean_model_id = args.model_name.replace("/", "_")
    plots_dir = os.path.join(args.output_dir, "plots", clean_model_id)
    arrays_dir = os.path.join(args.output_dir, "aie_arrays", clean_model_id)
    cka_dir = os.path.join(args.output_dir, "cka_analysis", clean_model_id)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)
    os.makedirs(cka_dir, exist_ok=True)

    thresholds = sorted(set([args.min_denom] + list(args.denom_thresholds)))
    tasks = {
        "FCT": "FCT",
        "NOTA": "Nota",
        "PMID2Title": "pmid2title",
        "Title2URL": "title2pub",
    }

    model_name_lower = args.model_name.lower()
    if args.is_qwen or "qwen" in model_name_lower:
        model_type = "qwen"
    elif "med42" in model_name_lower:
        model_type = "med42"
    elif "gemma" in model_name_lower:
        model_type = "gemma"
    elif "phi" in model_name_lower:
        model_type = "phi"
    else:
        model_type = "default"  # Works identically for Mistral

    print(f"[INIT] Model family resolved to: '{model_type}'")

    engine = CausalEngine(args.model_name, is_qwen=(model_type == "qwen"))
    pair_builder = MedHALTPairBuilder(
        engine.tokenizer, model_type=model_type, seed=args.seed
    )
    total_components = engine.num_layers * engine.num_heads + engine.num_layers
    k_values = sorted(
        {k for k in list(args.k_values) + [args.top_k] if 0 < k <= total_components}
    )

    print("\n--- Generating Med-HALT Contrastive Evaluation Sets ---", now())
    datasets = {}
    for task_index, (task_key, dataset_name) in enumerate(tasks.items()):
        datasets[task_key] = pair_builder.prepare_dataset(
            dataset_name,
            n_samples=args.n_samples,
            task_seed=args.seed + 1000 * task_index,
        )

    # ------------------------------------------------------------------
    # Phase 1: Vectorized Full Ranking per Task
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Vectorized Circuit Discovery ---", now())
    discovery_results = {}
    discovery_summary_rows = []
    discovery_sample_rows = []
    ranking_rows = []
    discovery_threshold_rows = []

    for task_key in tasks:
        print(f"\n[MAIN] Discovering ranking for {task_key}...", now())
        result = discover_circuit(
            engine,
            datasets[task_key],
            min_denom=args.min_denom,
            head_batch_size=args.head_batch_size,
        )
        discovery_results[task_key] = result

        discovery_summary_rows.append(
            {
                "task": task_key,
                "total_samples": result["total_count"],
                "valid_samples": result["valid_count"],
                "valid_fraction": result["valid_fraction"],
                "mean_abs_head_aie": result["mean_abs_head_aie"],
                "max_abs_head_aie": result["max_abs_head_aie"],
                "mean_abs_mlp_aie": result["mean_abs_mlp_aie"],
                "max_abs_mlp_aie": result["max_abs_mlp_aie"],
                "mean_clean_corrupt_gap": result["mean_clean_corrupt_gap"],
                "std_clean_corrupt_gap": result["std_clean_corrupt_gap"],
            }
        )

        for row in result["sample_rows"]:
            discovery_sample_rows.append({"task": task_key, **row})

        for row in result["ranking"]:
            ranking_rows.append(
                {"task": task_key, **{k: v for k, v in row.items() if k != "component"}}
            )

        np.save(
            os.path.join(arrays_dir, f"{task_key}_head_aie.npy"), result["head_aie"]
        )
        np.save(os.path.join(arrays_dir, f"{task_key}_mlp_aie.npy"), result["mlp_aie"])

        save_head_aie_heatmap(
            result["head_aie"],
            task_key,
            os.path.join(plots_dir, f"{task_key}_head_aie.png"),
        )
        save_mlp_aie_plot(
            result["mlp_aie"],
            task_key,
            os.path.join(plots_dir, f"{task_key}_mlp_aie.png"),
        )
        save_denominator_histogram(
            [r["denominator"] for r in result["sample_rows"]],
            task_key,
            os.path.join(plots_dir, f"{task_key}_denominator_histogram.png"),
        )

        _, primary_labels, primary_scores = circuit_from_ranking(
            result["ranking"], args.top_k
        )
        save_top_circuit_plot(
            primary_labels,
            primary_scores,
            task_key,
            os.path.join(plots_dir, f"{task_key}_top_{args.top_k}_circuit.png"),
        )

        base_labels = primary_labels
        for threshold in thresholds:
            alt_ranking = ranking_at_threshold(result, threshold)
            if alt_ranking is None:
                continue
            _, alt_labels, _ = circuit_from_ranking(alt_ranking, args.top_k)
            sim = circuit_similarity(base_labels, alt_labels)
            n_valid = int(np.sum(np.abs(result["valid_denominators"]) >= threshold))
            discovery_threshold_rows.append(
                {
                    "task": task_key,
                    "denom_threshold": threshold,
                    "valid_samples": n_valid,
                    "k": args.top_k,
                    "jaccard_to_base": sim["jaccard"],
                    "dice_to_base": sim["dice"],
                    "overlap_to_base": sim["overlap_coefficient"],
                }
            )

    pd.DataFrame(discovery_summary_rows).to_csv(
        os.path.join(args.output_dir, f"{clean_model_id}_discovery_metrics.csv"),
        index=False,
    )
    pd.DataFrame(discovery_sample_rows).to_csv(
        os.path.join(args.output_dir, f"{clean_model_id}_discovery_sample_metrics.csv"),
        index=False,
    )
    pd.DataFrame(ranking_rows).to_csv(
        os.path.join(args.output_dir, f"{clean_model_id}_full_component_ranking.csv"),
        index=False,
    )
    pd.DataFrame(discovery_threshold_rows).to_csv(
        os.path.join(
            args.output_dir, f"{clean_model_id}_discovery_threshold_sensitivity.csv"
        ),
        index=False,
    )

    # ------------------------------------------------------------------
    # Phase 2: Empirical 4x4 Transfer
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Empirical Multi-k Transfer ---", now())
    task_keys = list(tasks.keys())
    aggregate_rows = []
    threshold_rows = []
    per_sample_rows = []
    primary_stats = {}

    for k in k_values:
        circuits = {}
        labels_by_task = {}
        for task_key in task_keys:
            circuit, labels, _ = circuit_from_ranking(
                discovery_results[task_key]["ranking"], k
            )
            circuits[task_key] = circuit
            labels_by_task[task_key] = labels

        for i, source_task in enumerate(task_keys):
            for j, target_task in enumerate(task_keys):
                print(f"[MAIN] Transfer k={k}: {source_task} -> {target_task}", now())
                evaluation = evaluate_circuit_transfer(
                    engine,
                    circuits[source_task],
                    datasets[target_task],
                    min_denom=args.min_denom,
                )
                similarity = circuit_similarity(
                    labels_by_task[source_task], labels_by_task[target_task]
                )

                for row in evaluation["per_sample"]:
                    per_sample_rows.append(
                        {
                            "model": args.model_name,
                            "source_task": source_task,
                            "target_task": target_task,
                            "k": k,
                            **row,
                        }
                    )

                stats = summarize_transfer_rows(
                    evaluation["per_sample"],
                    threshold=args.min_denom,
                    n_bootstrap=args.bootstrap_samples,
                    confidence=args.bootstrap_confidence,
                    seed=args.seed + 100000 * k + 100 * i + j,
                )
                aggregate_rows.append(
                    {
                        "model": args.model_name,
                        "source_task": source_task,
                        "target_task": target_task,
                        "k": k,
                        "denom_threshold": args.min_denom,
                        "mean_transfer": stats["mean"],
                        "median_transfer": stats["median"],
                        "std_transfer": stats["std"],
                        "bootstrap_mean_ci_low": stats["bootstrap_mean_ci_low"],
                        "bootstrap_mean_ci_high": stats["bootstrap_mean_ci_high"],
                        "valid_samples": stats["valid_count"],
                        "valid_fraction": stats["valid_fraction"],
                        **similarity,
                    }
                )

                if k == args.top_k:
                    primary_stats[(source_task, target_task)] = {**stats, **similarity}

                for threshold in thresholds:
                    tstats = summarize_transfer_rows(
                        evaluation["per_sample"],
                        threshold=threshold,
                        n_bootstrap=args.bootstrap_samples,
                        confidence=args.bootstrap_confidence,
                        seed=args.seed + 700000 + 100000 * k + 100 * i + j,
                    )
                    threshold_rows.append(
                        {
                            "model": args.model_name,
                            "source_task": source_task,
                            "target_task": target_task,
                            "k": k,
                            "denom_threshold": threshold,
                            "mean_transfer": tstats["mean"],
                            "median_transfer": tstats["median"],
                            "std_transfer": tstats["std"],
                            "valid_samples": tstats["valid_count"],
                            "valid_fraction": tstats["valid_fraction"],
                        }
                    )

    df_aggregate = pd.DataFrame(aggregate_rows)
    df_threshold = pd.DataFrame(threshold_rows)
    df_samples = pd.DataFrame(per_sample_rows)
    df_aggregate.to_csv(
        os.path.join(args.output_dir, f"{clean_model_id}_k_sensitivity_metrics.csv"),
        index=False,
    )
    df_threshold.to_csv(
        os.path.join(
            args.output_dir, f"{clean_model_id}_denominator_threshold_sensitivity.csv"
        ),
        index=False,
    )
    df_samples.to_csv(
        os.path.join(args.output_dir, f"{clean_model_id}_per_sample_transfer.csv"),
        index=False,
    )

    save_k_sensitivity_plot(df_aggregate, os.path.join(plots_dir, "k_sensitivity.png"))
    save_threshold_sensitivity_plot(
        df_threshold,
        args.top_k,
        os.path.join(plots_dir, "denominator_threshold_sensitivity.png"),
    )

    matrix_names = ["mean_transfer", "median_transfer", "jaccard"]
    matrices = {
        name: np.full((len(task_keys), len(task_keys)), np.nan) for name in matrix_names
    }

    for i, source_task in enumerate(task_keys):
        for j, target_task in enumerate(task_keys):
            stats = primary_stats[(source_task, target_task)]
            matrices["mean_transfer"][i, j] = stats["mean"]
            matrices["median_transfer"][i, j] = stats["median"]
            matrices["jaccard"][i, j] = stats["jaccard"]

    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=task_keys, columns=task_keys).to_csv(
            os.path.join(args.output_dir, f"{clean_model_id}_{name}.csv")
        )
        save_heatmap(
            matrix,
            task_keys,
            f"{name.replace('_', ' ').title()} (k={args.top_k})",
            os.path.join(plots_dir, f"{name}_heatmap.png"),
        )

    # ------------------------------------------------------------------
    # Phase 3: Random Circuit Controls
    # ------------------------------------------------------------------
    control_replicate_rows = []
    control_summary_rows = []

    if args.num_random_controls > 0:
        print("\n--- Phase 3: Random Circuit Controls ---", now())
        for source_idx, source_task in enumerate(task_keys):
            ranking = discovery_results[source_task]["ranking"]
            universe = [row["component"] for row in ranking]
            observed_circuit, _, _ = circuit_from_ranking(ranking, args.top_k)

            for target_idx, target_task in enumerate(task_keys):
                observed = primary_stats[(source_task, target_task)]["mean"]
                eval_subset = datasets[target_task][
                    : min(len(datasets[target_task]), args.control_samples)
                ]

                for control_type in ["uniform", "layer_matched"]:
                    random_means = []
                    for replicate in range(args.num_random_controls):
                        rng = random.Random(
                            args.seed
                            + 1000000 * source_idx
                            + 10000 * target_idx
                            + 100 * replicate
                            + (1 if control_type == "layer_matched" else 0)
                        )
                        if control_type == "uniform":
                            circuit = sample_uniform_random_circuit(
                                universe, args.top_k, rng, exclude=observed_circuit
                            )
                        else:
                            circuit = sample_layer_matched_random_circuit(
                                universe, observed_circuit, rng
                            )

                        evaluation = evaluate_circuit_transfer(
                            engine,
                            circuit,
                            eval_subset,
                            min_denom=args.min_denom,
                        )
                        cstats = summarize_transfer_rows(
                            evaluation["per_sample"],
                            threshold=args.min_denom,
                            n_bootstrap=0,
                            confidence=args.bootstrap_confidence,
                            seed=args.seed,
                        )
                        random_means.append(cstats["mean"])
                        control_replicate_rows.append(
                            {
                                "source_task": source_task,
                                "target_task": target_task,
                                "control_type": control_type,
                                "replicate": replicate,
                                "k": args.top_k,
                                "mean_transfer": cstats["mean"],
                            }
                        )

                    finite = np.asarray(
                        [x for x in random_means if np.isfinite(x)], dtype=float
                    )
                    p_one, p_two = empirical_p_values(observed, random_means)
                    random_mean = float(np.mean(finite)) if len(finite) else np.nan
                    random_std = (
                        float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
                    )
                    control_summary_rows.append(
                        {
                            "source_task": source_task,
                            "target_task": target_task,
                            "control_type": control_type,
                            "k": args.top_k,
                            "observed_mean": observed,
                            "random_mean": random_mean,
                            "random_std": random_std,
                            "empirical_p_one_sided": p_one,
                            "empirical_p_two_sided_abs": p_two,
                            "num_controls": int(len(finite)),
                        }
                    )

    df_control_reps = pd.DataFrame(control_replicate_rows)
    df_control_summary = pd.DataFrame(control_summary_rows)
    df_control_reps.to_csv(
        os.path.join(
            args.output_dir, f"{clean_model_id}_random_control_replicates.csv"
        ),
        index=False,
    )
    df_control_summary.to_csv(
        os.path.join(args.output_dir, f"{clean_model_id}_random_control_summary.csv"),
        index=False,
    )

    # ------------------------------------------------------------------
    # Phase 4: In-Depth CKA (Centered Kernel Alignment) Analysis
    # ------------------------------------------------------------------
    print("\n--- Phase 4: In-Depth CKA Representation Analysis ---", now())
    task_reps = {}
    n_cka = args.cka_samples

    for task_key in task_keys:
        print(
            f"[CKA] Extracting layer activations for {task_key} ({n_cka} samples)...",
            now(),
        )
        task_reps[task_key] = engine.extract_task_representations(
            datasets[task_key], max_samples=n_cka
        )

    # 1. Macro Task-to-Task CKA Matrix (Layer-averaged and Final-layer)
    num_t = len(task_keys)
    macro_linear_cka = np.zeros((num_t, num_t))
    macro_rbf_cka = np.zeros((num_t, num_t))
    final_layer_linear_cka = np.zeros((num_t, num_t))
    final_l = engine.num_layers - 1

    for i, s_task in enumerate(task_keys):
        for j, t_task in enumerate(task_keys):
            # Compute average CKA across all layers
            layer_lin_sims = []
            layer_rbf_sims = []
            for l in range(engine.num_layers):
                x_rep = task_reps[s_task][l]
                y_rep = task_reps[t_task][l]
                layer_lin_sims.append(CenteredKernelAlignment.linear_cka(x_rep, y_rep))
                layer_rbf_sims.append(CenteredKernelAlignment.rbf_cka(x_rep, y_rep))

            macro_linear_cka[i, j] = float(np.mean(layer_lin_sims))
            macro_rbf_cka[i, j] = float(np.mean(layer_rbf_sims))
            final_layer_linear_cka[i, j] = CenteredKernelAlignment.linear_cka(
                task_reps[s_task][final_l], task_reps[t_task][final_l]
            )

    df_macro_lin_cka = pd.DataFrame(
        macro_linear_cka, index=task_keys, columns=task_keys
    ).round(4)
    df_macro_rbf_cka = pd.DataFrame(
        macro_rbf_cka, index=task_keys, columns=task_keys
    ).round(4)
    df_final_lin_cka = pd.DataFrame(
        final_layer_linear_cka, index=task_keys, columns=task_keys
    ).round(4)

    df_macro_lin_cka.to_csv(
        os.path.join(cka_dir, f"{clean_model_id}_mean_linear_cka.csv")
    )
    df_macro_rbf_cka.to_csv(os.path.join(cka_dir, f"{clean_model_id}_mean_rbf_cka.csv"))
    df_final_lin_cka.to_csv(
        os.path.join(cka_dir, f"{clean_model_id}_final_layer_linear_cka.csv")
    )

    save_heatmap(
        macro_linear_cka,
        task_keys,
        "Macro Mean Linear CKA",
        os.path.join(plots_dir, "macro_mean_linear_cka_heatmap.png"),
    )
    save_heatmap(
        final_layer_linear_cka,
        task_keys,
        f"Final Layer (L{final_l}) Linear CKA",
        os.path.join(plots_dir, "final_layer_linear_cka_heatmap.png"),
    )

    # 2. Granular Layer-by-Layer CKA Matrices (Cross-Task Layer Dynamics)
    cka_layer_profile_rows = []
    print("[CKA] Computing fine-grained layer-by-layer CKA dynamics...", now())

    for i, s_task in enumerate(task_keys):
        for j, t_task in enumerate(task_keys):
            if i > j:
                continue  # Symmetric cross-task inspection

            full_layer_matrix = np.zeros(
                (engine.num_layers, engine.num_layers), dtype=np.float32
            )
            for l_s in range(engine.num_layers):
                for l_t in range(engine.num_layers):
                    rep_s = task_reps[s_task][l_s]
                    rep_t = task_reps[t_task][l_t]
                    sim = CenteredKernelAlignment.linear_cka(rep_s, rep_t)
                    full_layer_matrix[l_s, l_t] = sim
                    if l_s == l_t:
                        cka_layer_profile_rows.append(
                            {
                                "source_task": s_task,
                                "target_task": t_task,
                                "layer": l_s,
                                "linear_cka": sim,
                            }
                        )

            np.save(
                os.path.join(cka_dir, f"{s_task}_vs_{t_task}_layer_cka.npy"),
                full_layer_matrix,
            )
            save_cka_layer_matrix_plot(
                full_layer_matrix,
                s_task,
                t_task,
                os.path.join(plots_dir, f"layer_cka_{s_task}_vs_{t_task}.png"),
            )

    df_layer_profiles = pd.DataFrame(cka_layer_profile_rows)
    df_layer_profiles.to_csv(
        os.path.join(cka_dir, f"{clean_model_id}_diagonal_layer_cka.csv"), index=False
    )

    # 3. Layer CKA Trajectory Plot (Reasoning vs Retrieval Cross-Alignment)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (s_t, t_t), group in df_layer_profiles.groupby(["source_task", "target_task"]):
        if s_t != t_t:
            ax.plot(
                group["layer"],
                group["linear_cka"],
                label=f"{s_t} ↔ {t_t}",
                marker="o",
                markersize=3,
            )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Linear CKA (Direct Diagonal Alignment)")
    ax.set_title("Cross-Task Representation Convergence Across Transformer Depth")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "cross_task_layer_cka_trajectories.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n================ FINAL RESULTS SUMMARY ================")
    print("\nEmpirical Mean Causal Transfer (Phase 2):")
    print(
        pd.DataFrame(
            matrices["mean_transfer"], index=task_keys, columns=task_keys
        ).round(4)
    )
    print("\nMacro Mean Linear CKA (Phase 4):")
    print(df_macro_lin_cka)

    if not args.skip_hf_upload:
        path_in_repo = args.hf_path_in_repo.strip("/") or (
            f"{clean_model_id}/n{args.n_samples}_k{args.top_k}_seed{args.seed}"
        )
        upload_results_to_huggingface(
            args.output_dir,
            args.hf_repo_id,
            path_in_repo,
            private=args.hf_private,
        )


if __name__ == "__main__":
    main()
