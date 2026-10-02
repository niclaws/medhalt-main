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

            # 3. Memory: PMID -> Title
            elif task_name == "pmid2title":
                pmid = str(sample.get("pmid", ""))
                title = sample.get("Title", sample.get("paper_title", "")).strip()
                fake_title = sample.get("fake_title", "Unrelated Clinical Evaluation")
                first_target_word = title.split()[0] if title else ""
                first_halluc_word = fake_title.split()[0]

                clean = f"Provide the title of PubMed paper with PMID {pmid}:\nTitle:"
                corrupt = f"Provide the title of PubMed paper with PMID {pmid[::-1]}:\nTitle:"
                pair = self.create_pair(clean, corrupt, first_target_word, first_halluc_word)
                if pair:
                    pairs.append(pair)

            # 4. Memory: Title -> PubMed URL
            elif task_name == "title2pub":
                title = sample.get("Title", sample.get("paper_title", "")).strip()
                pmid = str(sample.get("pmid", "12345678"))
                fake_pmid = str(sample.get("fake_pmid", "87654321"))

                clean = f"Article Title: {title}\nPubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
                corrupt = f"Article Title: {title} (Modified Retraction Review)\nPubMed Link URL: https://pubmed.ncbi.nlm.nih.gov/"
                pair = self.create_pair(clean, corrupt, pmid[:4], fake_pmid[:4])
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

        max_mem = {0: "9.5GiB", 1: "9.5GiB", 2: "9.5GiB", 3: "9.5GiB"}
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
        self.head_dim = getattr(self.model.config, "head_dim",
                                self.model.config.hidden_size // self.num_heads)

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
                    # Patching last-token slice if sequence lengths diverge
                    curr = outp.clone()
                    curr[:, -1, :] = clean_act[:, -1, :].to(outp.device)
                    return curr
                return clean_act.to(outp.device)
            elif comp_type == "head":
                curr = outp[0].clone()
                B, S, _ = curr.shape
                curr_reshaped = curr.view(B, S, self.num_heads, self.head_dim)
                clean_reshaped = clean_act.to(curr.device).view_as(curr_reshaped)
                if curr_reshaped.shape[1] != clean_reshaped.shape[1]:
                    curr_reshaped[:, -1, head_idx, :] = clean_reshaped[:, -1, head_idx, :]
                else:
                    curr_reshaped[:, :, head_idx, :] = clean_reshaped[:, :, head_idx, :]
                return (curr_reshaped.view(B, S, -1),) + outp[1:]
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

def discover_circuit(engine: CausalEngine, dataset: List[Dict], top_k: int = 15) -> Tuple[List[Tuple], List[str]]:
    """
    Sweeps every attention head and MLP layer to compute Average Indirect Effect (AIE).
    Returns the top-k components constituting the task circuit.
    """
    head_aie = np.zeros((engine.num_layers, engine.num_heads))
    mlp_aie = np.zeros(engine.num_layers)
    valid_count = 0

    print(f"\n[Discovery] Starting circuit discovery across {len(dataset)} samples...", datetime.now())

    for idx, sample in enumerate(tqdm(dataset, desc="Scanning Components", leave=False)):
        clean_ids = sample["clean_ids"]
        corrupt_ids = sample["corrupt_ids"]
        t_id, h_id = sample["target_id"], sample["halluc_id"]

        d_clean = engine.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = engine.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            print(f"[Discovery] Sample #{idx} dropped: small denominator ({denom:.6f}) | "
                  f"d_clean={d_clean:.4f}, d_corrupt={d_corrupt:.4f}", datetime.now())
            continue
        valid_count += 1

        head_cache, mlp_cache = engine.cache_clean_activations(clean_ids)

        # 1. Sweep Heads
        for l in range(engine.num_layers):
            for h in range(engine.num_heads):
                d_patched = engine.patch_single_component(
                    corrupt_ids, t_id, h_id, "head", l, head_cache[l], head_idx=h
                )
                head_aie[l, h] += (d_patched - d_corrupt) / denom

        # 2. Sweep MLPs
        for l in range(engine.num_layers):
            d_patched = engine.patch_single_component(
                corrupt_ids, t_id, h_id, "mlp", l, mlp_cache[l]
            )
            mlp_aie[l] += (d_patched - d_corrupt) / denom

    head_aie /= max(valid_count, 1)
    mlp_aie /= max(valid_count, 1)

    print(f"[Discovery] Finished scanning. Valid samples: {valid_count}/{len(dataset)}", datetime.now())
    print(f"[Discovery] Head AIE stats -> min: {head_aie.min():.4f}, max: {head_aie.max():.4f}, mean: {head_aie.mean():.4f}", datetime.now())
    print(f"[Discovery] MLP AIE stats  -> min: {mlp_aie.min():.4f}, max: {mlp_aie.max():.4f}, mean: {mlp_aie.mean():.4f}", datetime.now())

    # Rank and collect top-k
    ranked = []
    for l in range(engine.num_layers):
        for h in range(engine.num_heads):
            ranked.append((abs(head_aie[l, h]), f"L{l}H{h}", ("head", l, h)))
        ranked.append((abs(mlp_aie[l]), f"L{l}MLP", ("mlp", l, None)))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top_circuit = [item[2] for item in ranked[:top_k]]
    top_labels = [item[1] for item in ranked[:top_k]]

    print(f"[Discovery] Top {min(5, top_k)} Components: {ranked[:5]}", datetime.now())
    return top_circuit, top_labels


def evaluate_circuit_transfer(
        engine: CausalEngine,
        circuit: List[Tuple],
        target_dataset: List[Dict]
) -> float:
    """
    Patches all components of a source circuit simultaneously into target task inputs
    to measure causal transfer T_{s -> t}.
    """
    total_transfer = 0.0
    valid_count = 0

    for idx, sample in enumerate(tqdm(target_dataset, desc="Evaluating Transfer", leave=False)):
        clean_ids = sample["clean_ids"]
        corrupt_ids = sample["corrupt_ids"]
        t_id, h_id = sample["target_id"], sample["halluc_id"]

        d_clean = engine.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = engine.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            print(f"[Transfer] Sample #{idx} dropped: small denominator ({denom:.6f})", datetime.now())
            continue
        valid_count += 1

        head_cache, mlp_cache = engine.cache_clean_activations(clean_ids)
        handles = []

        # Hook all components of the circuit simultaneously
        for comp_type, layer_idx, head_idx in circuit:
            if comp_type == "mlp":
                act = mlp_cache[layer_idx]

                def get_mlp_hook(a):
                    def hook(m, inp, outp):
                        if a.shape[1] != outp.shape[1]:
                            curr = outp.clone()
                            curr[:, -1, :] = a[:, -1, :].to(outp.device)
                            return curr
                        return a.to(outp.device)
                    return hook

                h = engine.layers[layer_idx].mlp.register_forward_hook(get_mlp_hook(act))
                handles.append(h)
            elif comp_type == "head":
                act = head_cache[layer_idx]
                h_idx = head_idx

                def get_head_hook(a, target_h):
                    def hook(m, inp, outp):
                        curr = outp[0].clone()
                        B, S, _ = curr.shape
                        cr = curr.view(B, S, engine.num_heads, engine.head_dim)
                        ca = a.to(curr.device).view_as(cr)
                        if cr.shape[1] != ca.shape[1]:
                            cr[:, -1, target_h, :] = ca[:, -1, target_h, :]
                        else:
                            cr[:, :, target_h, :] = ca[:, :, target_h, :]
                        return (cr.view(B, S, -1),) + outp[1:]

                    return hook

                h = engine.layers[layer_idx].self_attn.register_forward_hook(get_head_hook(act, h_idx))
                handles.append(h)

        with torch.inference_mode():
            outputs = engine.model(corrupt_ids.to(engine.model.device))
            logits = outputs.logits[0, -1, :]
            patched_diff = (logits[t_id] - logits[h_id]).item()

        for h in handles:
            h.remove()

        step_transfer = (patched_diff - d_corrupt) / denom
        total_transfer += step_transfer

    avg_transfer = float(total_transfer / max(valid_count, 1))
    print(f"[Transfer] Completed evaluation across {valid_count} valid samples. Avg Transfer: {avg_transfer:.4f}", datetime.now())
    return avg_transfer


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

    print("\n--- Phase 1: Task Circuit Discovery (AIE Sweeps) ---", datetime.now())
    circuits = {}
    labels = {}
    for task_key in tasks.keys():
        print(f"\n[MAIN] Discovering top-{args.top_k} circuit for: {task_key}...", datetime.now())
        circuit, comp_labels = discover_circuit(engine, datasets[task_key], top_k=args.top_k)
        circuits[task_key] = circuit
        labels[task_key] = set(comp_labels)
        print(f"[{task_key}] Discovered Components: {comp_labels}", datetime.now())

    print("\n--- Phase 2: Cross-Task Causal Transfer Matrix Computation ---", datetime.now())
    task_keys = list(tasks.keys())
    transfer_matrix = np.zeros((4, 4))
    jaccard_matrix = np.zeros((4, 4))

    for i, s_task in enumerate(task_keys):
        for j, t_task in enumerate(task_keys):
            if i == j:
                transfer_matrix[i, j] = 1.0
                print(f"[MAIN] Transfer {s_task} -> {t_task}: 1.0000 (Self)", datetime.now())
            else:
                print(f"[MAIN] Evaluating Transfer {s_task} -> {t_task}...", datetime.now())
                score = evaluate_circuit_transfer(engine, circuits[s_task], datasets[t_task])
                transfer_matrix[i, j] = score
                print(f"[MAIN] Transfer {s_task} -> {t_task}: {score:.4f}", datetime.now())

            # Jaccard circuit similarity
            inter = len(labels[s_task] & labels[t_task])
            union = len(labels[s_task] | labels[t_task])
            jaccard_matrix[i, j] = inter / max(union, 1)

    df_transfer = pd.DataFrame(transfer_matrix, index=task_keys, columns=task_keys).round(4)
    df_jaccard = pd.DataFrame(jaccard_matrix, index=task_keys, columns=task_keys).round(4)

    # Save results
    transfer_path = os.path.join(args.output_dir, f"{clean_model_id}_causal_transfer_matrix.csv")
    jaccard_path = os.path.join(args.output_dir, f"{clean_model_id}_jaccard_overlap_matrix.csv")

    df_transfer.to_csv(transfer_path)
    df_jaccard.to_csv(jaccard_path)

    print("\n=================== RESULTS ===================")
    print("CROSS-TASK CAUSAL TRANSFER MATRIX (T_{s -> t}):", datetime.now())
    print(df_transfer, datetime.now())
    print("\nCIRCUIT JACCARD OVERLAP MATRIX:", datetime.now())
    print(df_jaccard, datetime.now())
    print(f"\nSaved CSVs to: {args.output_dir}", datetime.now())


if __name__ == "__main__":
    main()