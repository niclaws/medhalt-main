from datetime import datetime
import os
import re
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def now():
    return datetime.now(IST)


# =====================================================================
# Centered Kernel Alignment (CKA) Engine
# =====================================================================

class CenteredKernelAlignment:
    """
    Computes Linear and RBF Centered Kernel Alignment (CKA) with
    unbiased Hilbert-Schmidt Independence Criterion (HSIC).
    """

    @staticmethod
    def centering_matrix(n: int, device: torch.device) -> torch.Tensor:
        unit = torch.ones((n, n), device=device)
        identity = torch.eye(n, device=device)
        return identity - unit / n

    @classmethod
    def linear_hsic(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes unbiased HSIC for linear kernel: X, Y shaped [N, D]."""
        n = x.size(0)
        h = cls.centering_matrix(n, x.device)
        k = torch.matmul(x, x.t())
        l = torch.matmul(y, y.t())
        k_c = torch.matmul(torch.matmul(h, k), h)
        l_c = torch.matmul(torch.matmul(h, l), h)
        return torch.sum(k_c * l_c) / ((n - 1) ** 2)

    @classmethod
    def rbf_hsic(cls, x: torch.Tensor, y: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """Computes HSIC with Radial Basis Function (RBF) kernel."""
        n = x.size(0)
        dist_x = torch.cdist(x, x, p=2) ** 2
        dist_y = torch.cdist(y, y, p=2) ** 2

        if sigma is None:
            # Median heuristic for RBF bandwidth
            sigma_x = torch.median(dist_x[dist_x > 0]).sqrt() if (dist_x > 0).any() else 1.0
            sigma_y = torch.median(dist_y[dist_y > 0]).sqrt() if (dist_y > 0).any() else 1.0
        else:
            sigma_x = sigma_y = sigma

        k = torch.exp(-dist_x / (2 * (sigma_x ** 2) + 1e-8))
        l = torch.exp(-dist_y / (2 * (sigma_y ** 2) + 1e-8))

        h = cls.centering_matrix(n, x.device)
        k_c = torch.matmul(torch.matmul(h, k), h)
        l_c = torch.matmul(torch.matmul(h, l), h)
        return torch.sum(k_c * l_c) / ((n - 1) ** 2)

    @classmethod
    def linear_cka(cls, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculates Linear CKA:
        CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        """
        x_flat = x.reshape(x.size(0), -1).float()
        y_flat = y.reshape(y.size(0), -1).float()

        hsic_xy = cls.linear_hsic(x_flat, y_flat)
        hsic_xx = cls.linear_hsic(x_flat, x_flat)
        hsic_yy = cls.linear_hsic(y_flat, y_flat)

        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom.item() <= 1e-12:
            return 0.0
        return float((hsic_xy / denom).clamp(0.0, 1.0).item())

    @classmethod
    def rbf_cka(cls, x: torch.Tensor, y: torch.Tensor, sigma: Optional[float] = None) -> float:
        """Calculates RBF Kernel CKA."""
        x_flat = x.reshape(x.size(0), -1).float()
        y_flat = y.reshape(y.size(0), -1).float()

        hsic_xy = cls.rbf_hsic(x_flat, y_flat, sigma=sigma)
        hsic_xx = cls.rbf_hsic(x_flat, x_flat, sigma=sigma)
        hsic_yy = cls.rbf_hsic(y_flat, y_flat, sigma=sigma)

        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom.item() <= 1e-12:
            return 0.0
        return float((hsic_xy / denom).clamp(0.0, 1.0).item())


class CausalMediationHarness:
    """
    Manages model loading, registers PyTorch forward hooks,
    extracts representations for CKA, and performs activation patching.
    """

    def __init__(self, model_name_or_path: str, is_qwen: bool = False):
        self.model_name = model_name_or_path
        self.is_qwen = is_qwen

        print(f"\n[Harness Init] Loading tokenizer for: {model_name_or_path}", now())
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[Harness Init] pad_token set to eos_token (ID: {self.tokenizer.pad_token_id})", now())

        dtype = torch.float16
        print(f"[Harness Init] Loading model with dtype={dtype}...", now())
        # max_mem = {0: "21GiB"} if torch.cuda.device_count() == 1 else "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            # max_memory="auto",
            trust_remote_code=True
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        self.head_dim = (
            getattr(self.model.config, "head_dim", None)
            or self.model.config.hidden_size // self.num_heads
        )

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        else:
            raise AttributeError("Unrecognized transformer layer layout.")

        print(f"[Harness Init] Model loaded successfully.", now())
        print(f"[Harness Init] Layers: {self.num_layers}, Attention Heads: {self.num_heads}, "
              f"Head Dim: {self.head_dim}, Hidden Dim: {self.model.config.hidden_size}", now())

        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                print(f"[GPU {gpu_id}] Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB", now())

    def _clean_qwen_prompt(self, text: str) -> str:
        """Enforces non-thinking mode for Qwen models."""
        if not self.is_qwen:
            return text
        system_prefix = (
            "<|im_start|>system\nYou are a precise medical QA assistant. "
            "Output the factual answer directly without any thinking process.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        return system_prefix + text + "<|im_end|>\n<|im_start|>assistant\n"

    def get_logit_diff(self, input_ids: torch.Tensor, target_id: int, halluc_id: int) -> float:
        input_device = self.model.device
        input_ids = input_ids.to(input_device)

        with torch.inference_mode():
            outputs = self.model(input_ids)
            last_logits = outputs.logits[0, -1, :]
            target_logit = last_logits[target_id].item()
            halluc_logit = last_logits[halluc_id].item()
            diff = target_logit - halluc_logit

        return diff

    def cache_activations(self, input_ids: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Extracts head outputs and MLP outputs from clean forward pass."""
        head_cache = {}
        mlp_cache = {}
        hooks = []

        def get_attn_hook(l):
            def hook(module, inp, outp):
                act = outp[0] if isinstance(outp, tuple) else outp
                head_cache[l] = act.detach()
            return hook

        def get_mlp_hook(l):
            def hook(module, inp, outp):
                act = outp[0] if isinstance(outp, tuple) else outp
                mlp_cache[l] = act.detach()
            return hook

        for l in range(self.num_layers):
            hooks.append(self.layers[l].self_attn.register_forward_hook(get_attn_hook(l)))
            hooks.append(self.layers[l].mlp.register_forward_hook(get_mlp_hook(l)))

        input_device = self.model.device
        with torch.inference_mode():
            self.model(input_ids.to(input_device))

        for h in hooks:
            h.remove()

        return head_cache, mlp_cache

    def extract_representations(
        self,
        samples: List[Dict],
        target_token_only: bool = True,
        max_samples: int = 50
    ) -> Dict[int, torch.Tensor]:
        """
        Extracts layer hidden representations across samples for in-depth CKA analysis.
        Returns: {layer_idx: Tensor of shape [N, hidden_dim]}
        """
        layer_reps = {l: [] for l in range(self.num_layers)}
        eval_samples = samples[:max_samples]
        device = self.model.device

        for sample in eval_samples:
            clean_ids = sample["clean_ids"].to(device)
            sample_cache = {}
            hooks = []

            def make_hook(layer_i):
                def hook_fn(module, inp, outp):
                    act = outp[0] if isinstance(outp, tuple) else outp
                    if target_token_only:
                        # Extract final decision token representation
                        sample_cache[layer_i] = act[:, -1, :].detach().cpu()
                    else:
                        # Mean pool across sequence length
                        sample_cache[layer_i] = act.mean(dim=1).detach().cpu()
                return hook_fn

            for l in range(self.num_layers):
                hooks.append(self.layers[l].register_forward_hook(make_hook(l)))

            with torch.inference_mode():
                self.model(clean_ids)

            for h in hooks:
                h.remove()

            for l in range(self.num_layers):
                layer_reps[l].append(sample_cache[l])

        # Stack into [N, hidden_dim] per layer
        stacked_reps = {
            l: torch.cat(layer_reps[l], dim=0) for l in range(self.num_layers)
        }
        return stacked_reps

    def patch_heads_layer_batched(
        self,
        corrupt_ids: torch.Tensor,
        target_id: int,
        halluc_id: int,
        layer_idx: int,
        clean_activation: torch.Tensor,
        chunk_size: int = 32
    ) -> np.ndarray:
        """Patches all attention heads in a layer simultaneously using batched corrupt passes."""
        target_diffs = np.zeros(self.num_heads, dtype=np.float32)
        total_heads = self.num_heads
        input_device = self.model.device

        for start_idx in range(0, total_heads, chunk_size):
            end_idx = min(start_idx + chunk_size, total_heads)
            active_heads = list(range(start_idx, end_idx))
            batch_k = len(active_heads)

            batched_corrupt = corrupt_ids.repeat(batch_k, 1).to(input_device)

            def batched_patch_hook(module, inp, outp):
                curr = outp[0].clone() if isinstance(outp, tuple) else outp.clone()
                B, S, hidden = curr.shape
                clean = clean_activation.to(curr.device)
                Bc, Sc, clean_hidden = clean.shape

                curr_reshaped = curr.view(B, S, self.num_heads, self.head_dim)
                clean_reshaped = clean.view(Bc, Sc, self.num_heads, self.head_dim)

                for b_idx, h_idx in enumerate(active_heads):
                    if S != Sc:
                        curr_reshaped[b_idx, -1, h_idx, :] = clean_reshaped[0, -1, h_idx, :]
                    else:
                        curr_reshaped[b_idx, :, h_idx, :] = clean_reshaped[0, :, h_idx, :]

                patched = curr_reshaped.view(B, S, hidden)
                return (patched,) + outp[1:] if isinstance(outp, tuple) else patched

            handle = self.layers[layer_idx].self_attn.register_forward_hook(batched_patch_hook)
            with torch.inference_mode():
                outputs = self.model(batched_corrupt)
                logits = outputs.logits[:, -1, :]
                diffs = (logits[:, target_id] - logits[:, halluc_id]).float().cpu().numpy()

            handle.remove()
            target_diffs[start_idx:end_idx] = diffs

        return target_diffs

    def patch_component(
        self,
        corrupt_ids: torch.Tensor,
        target_id: int,
        halluc_id: int,
        component_type: str,
        layer_idx: int,
        clean_activation: torch.Tensor,
        head_idx: Optional[int] = None
    ) -> float:
        """Executes counterfactual forward pass patching clean activation into corrupt run."""
        def patch_hook(module, inp, outp):
            if component_type == "mlp":
                clean_act = clean_activation.to(outp.device)
                if clean_act.shape[1] != outp.shape[1]:
                    curr = outp.clone()
                    curr[:, -1, :] = clean_act[:, -1, :]
                    return curr
                return clean_act

            elif component_type == "head":
                curr = outp[0].clone() if isinstance(outp, tuple) else outp.clone()
                B, S, hidden = curr.shape
                clean = clean_activation.to(curr.device)
                Bc, Sc, _ = clean.shape

                curr_reshaped = curr.view(B, S, self.num_heads, self.head_dim)
                clean_reshaped = clean.view(Bc, Sc, self.num_heads, self.head_dim)

                if Sc != S:
                    curr_reshaped[:, -1, head_idx, :] = clean_reshaped[:, -1, head_idx, :]
                else:
                    curr_reshaped[:, :, head_idx, :] = clean_reshaped[:, :, head_idx, :]

                patched = curr_reshaped.view(B, S, hidden)
                return (patched,) + outp[1:] if isinstance(outp, tuple) else patched
            return outp

        if component_type == "mlp":
            handle = self.layers[layer_idx].mlp.register_forward_hook(patch_hook)
        elif component_type == "head":
            handle = self.layers[layer_idx].self_attn.register_forward_hook(patch_hook)
        else:
            raise ValueError(f"Unknown component {component_type}")

        input_device = self.model.device
        with torch.inference_mode():
            outputs = self.model(corrupt_ids.to(input_device))
            logits = outputs.logits[0, -1, :]
            patched_diff = (logits[target_id] - logits[halluc_id]).item()

        handle.remove()
        return patched_diff


class ContrastiveMedHALTDataset:
    """Constructs matched (Clean, Corrupted, Target_Token, Hallucinated_Token) pairs."""

    def __init__(self, tokenizer, is_qwen: bool = False):
        self.tokenizer = tokenizer
        self.is_qwen = is_qwen

    def build_pair(self, clean_prompt: str, corrupt_prompt: str,
                   target_ans: str, halluc_ans: str) -> Optional[Dict]:
        clean_ids = self.tokenizer(clean_prompt, return_tensors="pt").input_ids
        corrupt_ids = self.tokenizer(corrupt_prompt, return_tensors="pt").input_ids

        t_id = self.tokenizer.encode(" " + target_ans.strip(), add_special_tokens=False)
        h_id = self.tokenizer.encode(" " + halluc_ans.strip(), add_special_tokens=False)

        if len(t_id) == 0:
            t_id = self.tokenizer.encode(target_ans.strip(), add_special_tokens=False)
        if len(h_id) == 0:
            h_id = self.tokenizer.encode(halluc_ans.strip(), add_special_tokens=False)

        if len(t_id) == 0 or len(h_id) == 0 or t_id[0] == h_id[0]:
            return None

        return {
            "clean_ids": clean_ids,
            "corrupt_ids": corrupt_ids,
            "target_id": t_id[0],
            "halluc_id": h_id[0]
        }

    def generate_fct_pair(self, raw_sample: Dict) -> Optional[Dict]:
        q = raw_sample["question"]
        op_correct = raw_sample["correct_answer"]
        op_distractor = raw_sample["suggested_answer"]

        clean = f"Question: {q}\nOptions:\n{raw_sample['options']}\nAnswer:"
        corrupt = f"Question: {q}\nOptions:\n{raw_sample['options']}\nNote: Experts confirm that the correct option is definitely {op_distractor}.\nAnswer:"
        return self.build_pair(clean, corrupt, op_correct, op_distractor)

    def generate_nota_pair(self, raw_sample: Dict) -> Optional[Dict]:
        q = raw_sample["question"]
        clean = f"Question: {q}\nOptions:\n{raw_sample['options_with_nota']}\nAnswer:"
        corrupt = f"Question: {q}\nOptions:\n{raw_sample['options_without_nota_with_plausible_false']}\nAnswer:"
        return self.build_pair(clean, corrupt, "None", raw_sample["plausible_token"])

    def generate_pmid2title_pair(self, raw_sample: Dict) -> Optional[Dict]:
        pmid = str(raw_sample["pmid"])
        corrupt_pmid = str(raw_sample["corrupted_pmid"])
        target_token = raw_sample["target_first_word"]
        halluc_token = raw_sample["hallucinated_first_word"]

        clean = f"Provide the title of PubMed paper with PMID {pmid}:\nTitle:"
        corrupt = f"Provide the title of PubMed paper with PMID {corrupt_pmid}:\nTitle:"
        return self.build_pair(clean, corrupt, target_token, halluc_token)

    def generate_title2url_pair(self, raw_sample: Dict) -> Optional[Dict]:
        clean_title = raw_sample["title"]
        corrupt_title = raw_sample["corrupted_title"]
        target_stub = raw_sample["target_pmid_token"]
        halluc_stub = raw_sample["hallucinated_pmid_token"]

        clean = f"Article Title: {clean_title}\nPubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
        corrupt = f"Article Title: {corrupt_title}\nPubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
        return self.build_pair(clean, corrupt, target_stub, halluc_stub)