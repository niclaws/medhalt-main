from datetime import datetime
import os
import re
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class CausalMediationHarness:
    """
    Manages model partitioning across 4x RTX 2080 Ti (11GB) GPUs,
    registers PyTorch forward hooks, and performs activation patching.
    """

    def __init__(self, model_name_or_path: str, is_qwen: bool = False):
        self.model_name = model_name_or_path
        self.is_qwen = is_qwen

        print(f"\n[Harness Init] Loading tokenizer for: {model_name_or_path}", datetime.now())
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[Harness Init] pad_token set to eos_token (ID: {self.tokenizer.pad_token_id})", datetime.now())

        # Optimal layer mapping across 4 x 11GB GPUs
        max_mem = {0: "21GiB"}

        print(f"[Harness Init] Loading model with device_map='auto' and max_memory={max_memory}...", datetime.now())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True
        )
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        self.head_dim = (
                getattr(self.model.config, "head_dim", None)
                or self.model.config.hidden_size // self.num_heads
        )

        # Locate transformer layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        else:
            raise AttributeError("Unrecognized transformer layer layout.")

        print(f"[Harness Init] Model loaded successfully.", datetime.now())
        print(f"[Harness Init] Layers: {self.num_layers}, Attention Heads: {self.num_heads}, "
              f"Head Dim: {self.head_dim}, Hidden Dim: {self.model.config.hidden_size}", datetime.now())

        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                print(f"[GPU {gpu_id}] Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB", datetime.now())

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map:
            print(f"[Harness Init] Model device map preview (first 4 layers): "
                  f"{ {k: hf_device_map[k] for k in list(hf_device_map.keys())[:4]} }", datetime.now())

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

        print(f"[get_logit_diff] SeqLen: {input_ids.shape[1]} | "
              f"Target({target_id}): {target_logit:.4f} | "
              f"Halluc({halluc_id}): {halluc_logit:.4f} | Diff: {diff:.4f}", datetime.now())
        return diff

    def cache_activations(self, input_ids: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Extracts head outputs and MLP outputs from clean forward pass.
        Returns tensors offloaded to CPU or relevant device.
        """
        print(f"\n[cache_activations] Running clean forward pass with input shape {tuple(input_ids.shape)}...", datetime.now())
        head_cache = {}
        mlp_cache = {}
        hooks = []

        def get_attn_hook(l):
            def hook(module, inp, outp):
                act = outp[0] if isinstance(outp, tuple) else outp
                head_cache[l] = act.detach().cpu()
            return hook

        def get_mlp_hook(l):
            def hook(module, inp, outp):
                act = outp[0] if isinstance(outp, tuple) else outp
                mlp_cache[l] = act.detach().cpu()
            return hook

        for l in range(self.num_layers):
            hooks.append(self.layers[l].self_attn.register_forward_hook(get_attn_hook(l)))
            hooks.append(self.layers[l].mlp.register_forward_hook(get_mlp_hook(l)))

        input_device = self.model.device
        with torch.inference_mode():
            self.model(input_ids.to(input_device))

        for h in hooks:
            h.remove()

        print(f"[cache_activations] Caching complete. Cached {len(head_cache)} attention and "
              f"{len(mlp_cache)} MLP residual representations.", datetime.now())
        if 0 in head_cache:
            print(f"[cache_activations] Layer 0 Attn shape: {tuple(head_cache[0].shape)} | "
                  f"MLP shape: {tuple(mlp_cache[0].shape)}", datetime.now())

        return head_cache, mlp_cache

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
        """
        Executes counterfactual forward pass patching clean activation into corrupt run.
        """
        head_info = f"Head={head_idx}" if head_idx is not None else ""
        print(f"[patch_component] Patching Type='{component_type}' | Layer={layer_idx} {head_info}", datetime.now())

        def patch_hook(module, inp, outp):
            if component_type == "mlp":
                clean_act = clean_activation.to(outp.device)
                if clean_act.shape[1] != outp.shape[1]:
                    print(f"  [WARN patch_hook:MLP] Sequence length mismatch! Clean: {clean_act.shape[1]} vs Corrupt: {outp.shape[1]}. "
                          f"Falling back to patching the last token position (-1).", datetime.now())
                    curr = outp.clone()
                    curr[:, -1, :] = clean_act[:, -1, :]
                    return curr
                return clean_act

            elif component_type == "head":
                curr = outp[0].clone()
                B, S, _ = curr.shape
                curr_reshaped = curr.view(B, S, self.num_heads, self.head_dim)
                clean_reshaped = clean_activation.to(curr.device).view(
                    clean_activation.shape[0], clean_activation.shape[1], self.num_heads, self.head_dim
                )

                if clean_reshaped.shape[1] != S:
                    print(f"  [WARN patch_hook:Head] Sequence length mismatch! Clean: {clean_reshaped.shape[1]} vs Corrupt: {S}. "
                          f"Patching head {head_idx} at the last token position (-1).", datetime.now())
                    curr_reshaped[:, -1, head_idx, :] = clean_reshaped[:, -1, head_idx, :]
                else:
                    curr_reshaped[:, :, head_idx, :] = clean_reshaped[:, :, head_idx, :]

                return (curr_reshaped.view_as(curr),) + outp[1:]
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
        print(f"  [patch_component] Patched Logit Diff: {patched_diff:.4f}", datetime.now())
        return patched_diff


class ContrastiveMedHALTDataset:
    """
    Constructs matched (Clean, Corrupted, Target_Token, Hallucinated_Token) pairs
    for the 4 primary Med-HALT evaluation tasks.
    """

    def __init__(self, tokenizer, is_qwen: bool = False):
        self.tokenizer = tokenizer
        self.is_qwen = is_qwen

    def build_pair(self, clean_prompt: str, corrupt_prompt: str,
                   target_ans: str, halluc_ans: str) -> Optional[Dict]:
        print(f"\n[build_pair] Building contrastive pair...", datetime.now())
        clean_ids = self.tokenizer(clean_prompt, return_tensors="pt").input_ids
        corrupt_ids = self.tokenizer(corrupt_prompt, return_tensors="pt").input_ids

        t_id = self.tokenizer.encode(" " + target_ans.strip(), add_special_tokens=False)
        h_id = self.tokenizer.encode(" " + halluc_ans.strip(), add_special_tokens=False)

        # Fallback if leading space encoding yields empty tensor
        if len(t_id) == 0:
            t_id = self.tokenizer.encode(target_ans.strip(), add_special_tokens=False)
        if len(h_id) == 0:
            h_id = self.tokenizer.encode(halluc_ans.strip(), add_special_tokens=False)

        print(f"  Target String: '{target_ans.strip()}' -> Token IDs: {t_id}", datetime.now())
        print(f"  Halluc String: '{halluc_ans.strip()}' -> Token IDs: {h_id}", datetime.now())

        if len(t_id) == 0 or len(h_id) == 0:
            print("  [ERROR build_pair] Empty tokenization encountered. Dropping pair.", datetime.now())
            return None

        if len(t_id) > 1 or len(h_id) > 1:
            print(f"  [WARN build_pair] Target/Hallucination spans multiple tokens. "
                  f"Trimming to first token ID for patching (T: {t_id[0]}, H: {h_id[0]}).", datetime.now())

        if t_id[0] == h_id[0]:
            print(f"  [ERROR build_pair] Identical Target and Hallucination token IDs ({t_id[0]}). Dropping pair.", datetime.now())
            return None

        print(f"  Pair created successfully. Clean seq: {clean_ids.shape[1]} tokens | Corrupt seq: {corrupt_ids.shape[1]} tokens", datetime.now())
        return {
            "clean_ids": clean_ids,
            "corrupt_ids": corrupt_ids,
            "target_id": t_id[0],
            "halluc_id": h_id[0]
        }

    def generate_fct_pair(self, raw_sample: Dict) -> Optional[Dict]:
        """
        FCT (False Confidence Test):
        Clean: Question presented neutrally with correct ground truth choice.
        Corrupt: Question presented with misleading confidence assertion favoring wrong choice.
        """
        print("[generate_fct_pair] Processing FCT sample...", datetime.now())
        q = raw_sample["question"]
        op_correct = raw_sample["correct_answer"]
        op_distractor = raw_sample["suggested_answer"]

        clean = f"Question: {q}\nOptions:\n{raw_sample['options']}\nAnswer:"
        corrupt = f"Question: {q}\nOptions:\n{raw_sample['options']}\nNote: Experts confirm that the correct option is definitely {op_distractor}.\nAnswer:"
        return self.build_pair(clean, corrupt, op_correct, op_distractor)

    def generate_nota_pair(self, raw_sample: Dict) -> Optional[Dict]:
        """
        NOTA (None of the Above):
        Clean: Correct option is 'None of the above'.
        Corrupt: Distractor plausible medical option inserted to elicit false recall.
        """
        print("[generate_nota_pair] Processing NOTA sample...", datetime.now())
        q = raw_sample["question"]
        clean = f"Question: {q}\nOptions:\n{raw_sample['options_with_nota']}\nAnswer:"
        corrupt = f"Question: {q}\nOptions:\n{raw_sample['options_without_nota_with_plausible_false']}\nAnswer:"
        return self.build_pair(clean, corrupt, "None", raw_sample["plausible_token"])

    def generate_pmid2title_pair(self, raw_sample: Dict) -> Optional[Dict]:
        """
        PMID -> Title:
        Clean: Valid PMID mapping to true first title word.
        Corrupt: Altered/Shuffled PMID mapping to hallucinated title word.
        """
        print("[generate_pmid2title_pair] Processing PMID2Title sample...", datetime.now())
        pmid = str(raw_sample["pmid"])
        corrupt_pmid = str(raw_sample["corrupted_pmid"])
        target_token = raw_sample["target_first_word"]
        halluc_token = raw_sample["hallucinated_first_word"]

        clean = f"Provide the title of PubMed paper with PMID {pmid}:\nTitle:"
        corrupt = f"Provide the title of PubMed paper with PMID {corrupt_pmid}:\nTitle:"
        return self.build_pair(clean, corrupt, target_token, halluc_token)

    def generate_title2url_pair(self, raw_sample: Dict) -> Optional[Dict]:
        """
        Title -> PubMed URL:
        Clean: True title mapped to correct PMID URL stub.
        Corrupt: Slightly altered disease entity in title mapped to wrong URL stub.
        """
        print("[generate_title2url_pair] Processing Title2URL sample...", datetime.now())
        clean_title = raw_sample["title"]
        corrupt_title = raw_sample["corrupted_title"]
        target_stub = raw_sample["target_pmid_token"]
        halluc_stub = raw_sample["hallucinated_pmid_token"]

        clean = f"Article Title: {clean_title}\nPubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
        corrupt = f"Article Title: {corrupt_title}\nPubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
        return self.build_pair(clean, corrupt, target_stub, halluc_stub)


# =====================================================================
# Verification Block
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MED-HALT CAUSAL MEDIATION HARNESS — TEST BENCH", datetime.now())
    print("=" * 60)

    # Mock test inputs to verify tokenization and pair generation
    test_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if test_tokenizer.pad_token is None:
        test_tokenizer.pad_token = test_tokenizer.eos_token

    dataset_builder = ContrastiveMedHALTDataset(test_tokenizer)

    mock_fct = {
        "question": "What is the primary etiology of type 1 diabetes mellitus?",
        "options": "A) Insulin resistance\nB) Autoimmune beta-cell destruction",
        "correct_answer": "B",
        "suggested_answer": "A"
    }

    print("\n--- Running Dry-Run FCT Pair Generation ---", datetime.now())
    pair = dataset_builder.generate_fct_pair(mock_fct)
    if pair:
        print("[TEST SUCCEEDED] Pair keys:", list(pair.keys()), datetime.now())
        print("[TEST SUCCEEDED] Target Token ID:", pair["target_id"],
              "Decoded:", repr(test_tokenizer.decode([pair["target_id"]])), datetime.now())
        print("[TEST SUCCEEDED] Halluc Token ID:", pair["halluc_id"],
              "Decoded:", repr(test_tokenizer.decode([pair["halluc_id"]])), datetime.now())
    else:
        print("[TEST FAILED] Pair creation failed.")