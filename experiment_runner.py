import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from patching_engine import CausalMediationHarness, ContrastiveMedHALTDataset
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now():
    return datetime.now(IST)


# =====================================================================
# 1. Circuit Discovery (Vectorized AIE Sweeps)
# =====================================================================


def compute_task_circuit(
    harness: CausalMediationHarness,
    dataset_samples: List[Dict],
    top_k: int = 10,
    head_batch_size: int = 32,
) -> Dict[str, object]:
    """
    Calculates Average Indirect Effect (AIE) across all Attention Heads
    and MLP Blocks for a single task using vectorized per-layer batching.
    """
    num_layers = harness.num_layers
    num_heads = harness.num_heads

    head_aie = np.zeros((num_layers, num_heads), dtype=np.float32)
    mlp_aie = np.zeros(num_layers, dtype=np.float32)

    valid_samples = 0
    total_samples = len(dataset_samples)
    device = harness.model.device

    print(
        f"\n[compute_task_circuit] Starting Vectorized AIE sweep on {total_samples} samples.",
        now(),
    )
    print(
        f"[compute_task_circuit] Dimensions: {num_layers} layers x {num_heads} heads + {num_layers} MLPs "
        f"-> {num_layers + num_layers} batched passes per sample.",
        now(),
    )

    sweep_start = time.time()

    for idx, sample in enumerate(dataset_samples):
        clean_ids = sample["clean_ids"].to(device)
        corrupt_ids = sample["corrupt_ids"].to(device)
        t_id, h_id = sample["target_id"], sample["halluc_id"]

        d_clean = harness.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = harness.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            continue

        valid_samples += 1
        sample_start = time.time()

        # Cache residual states for the clean run
        head_cache, mlp_cache = harness.cache_activations(clean_ids)

        # 1. Batched Sweep: Attention Heads (1 pass per layer)
        for l in range(num_layers):
            layer_head_diffs = harness.patch_heads_layer_batched(
                corrupt_ids, t_id, h_id, l, head_cache[l], chunk_size=head_batch_size
            )
            head_aie[l, :] += (layer_head_diffs - d_corrupt) / denom

        # 2. Sweep: MLPs
        for l in range(num_layers):
            d_patched = harness.patch_component(
                corrupt_ids, t_id, h_id, "mlp", l, mlp_cache[l]
            )
            mlp_aie[l] += (d_patched - d_corrupt) / denom

        elapsed = time.time() - sample_start
        if (idx + 1) % 5 == 0 or (idx + 1) == total_samples:
            print(
                f"[compute_task_circuit] Sample #{idx+1}/{total_samples} complete in {elapsed:.2f}s | "
                f"Cumulative Valid: {valid_samples}",
                now(),
            )

    print(
        f"\n[compute_task_circuit] Sweep finished in {time.time() - sweep_start:.2f}s.",
        now(),
    )

    if valid_samples == 0:
        raise RuntimeError("Circuit discovery produced zero valid samples.")

    head_aie /= valid_samples
    mlp_aie /= valid_samples

    # Rank all components by absolute impact
    components = []
    for l in range(num_layers):
        for h in range(num_heads):
            components.append((head_aie[l, h], f"L{l}H{h}", ("head", l, h)))
        components.append((mlp_aie[l], f"L{l}MLP", ("mlp", l, None)))

    components.sort(key=lambda x: abs(x[0]), reverse=True)
    top_circuit = [item[2] for item in components[:top_k]]
    circuit_labels = [item[1] for item in components[:top_k]]
    circuit_scores = [round(float(item[0]), 4) for item in components[:top_k]]

    print(f"[compute_task_circuit] Top-{top_k} Discovered Circuit:", now())
    for rank, (label, score) in enumerate(zip(circuit_labels, circuit_scores), 1):
        print(f"  Rank {rank:2d}: {label:<8} | AIE: {score:.4f}", now())

    return {
        "circuit": top_circuit,
        "labels": circuit_labels,
        "scores": circuit_scores,
        "head_aie": head_aie,
        "mlp_aie": mlp_aie,
    }


# =====================================================================
# 2. Cross-Task Transfer Evaluation
# =====================================================================


def evaluate_transfer(
    harness: CausalMediationHarness, circuit: List[Tuple], target_samples: List[Dict]
) -> float:
    total_transfer = 0.0
    valid_count = 0
    total_samples = len(target_samples)
    device = harness.model.device

    for idx, sample in enumerate(target_samples):
        clean_ids = sample["clean_ids"].to(device)
        corrupt_ids = sample["corrupt_ids"].to(device)
        t_id, h_id = sample["target_id"], sample["halluc_id"]

        d_clean = harness.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = harness.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            continue

        head_cache, mlp_cache = harness.cache_activations(clean_ids)

        handles = []
        for comp_type, layer_idx, head_idx in circuit:
            if comp_type == "mlp":
                clean_act = mlp_cache[layer_idx]

                def make_mlp_hook(act):
                    def hook_fn(m, inp, outp):
                        a = act.to(outp.device)
                        if a.shape[1] != outp.shape[1]:
                            curr = outp.clone()
                            curr[:, -1, :] = a[:, -1, :]
                            return curr
                        return a

                    return hook_fn

                h = harness.layers[layer_idx].mlp.register_forward_hook(
                    make_mlp_hook(clean_act)
                )
                handles.append(h)

            elif comp_type == "head":
                clean_act = head_cache[layer_idx]
                h_idx = head_idx

                def make_head_hook(act, target_head):
                    def hook_fn(m, inp, outp):
                        curr = (
                            outp[0].clone() if isinstance(outp, tuple) else outp.clone()
                        )
                        B, S, hidden = curr.shape
                        cr = curr.view(B, S, harness.num_heads, harness.head_dim)
                        ca = act.to(curr.device).view(
                            act.shape[0],
                            act.shape[1],
                            harness.num_heads,
                            harness.head_dim,
                        )
                        if cr.shape[1] != ca.shape[1]:
                            cr[:, -1, target_head, :] = ca[:, -1, target_head, :]
                        else:
                            cr[:, :, target_head, :] = ca[:, :, target_head, :]
                        return (
                            (cr.view_as(curr),) + outp[1:]
                            if isinstance(outp, tuple)
                            else cr.view_as(curr)
                        )

                    return hook_fn

                h = harness.layers[layer_idx].self_attn.register_forward_hook(
                    make_head_hook(clean_act, h_idx)
                )
                handles.append(h)

        with torch.inference_mode():
            outputs = harness.model(corrupt_ids)
            logits = outputs.logits[0, -1, :]
            patched_diff = (logits[t_id] - logits[h_id]).item()

        for handle in handles:
            handle.remove()

        step_transfer = (patched_diff - d_corrupt) / denom
        total_transfer += step_transfer
        valid_count += 1

    final_transfer = float(total_transfer / max(valid_count, 1))
    return final_transfer


# =====================================================================
# 3. Main Experiment Orchestrator
# =====================================================================


def run_cross_task_experiment(
    model_name: str,
    tasks_data: Dict[str, List[Dict]],
    is_qwen: bool = False,
    top_k: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print(
        f"RUNNING EXPERIMENT: Model={model_name} | QwenMode={is_qwen} | TopK={top_k}",
        now(),
    )
    print("=" * 70)

    harness = CausalMediationHarness(model_name, is_qwen=is_qwen)
    task_names = ["FCT", "NOTA", "PMID2Title", "Title2URL"]

    circuits = {}
    circuit_label_sets = {}

    print(f"\n=======================================================")
    print(f"PHASE 1: VECTORIZED CIRCUIT DISCOVERY (Top-{top_k})", now())
    print(f"=======================================================")
    for task in task_names:
        print(f"\n>>> Running Discovery for Task: {task}", now())
        res = compute_task_circuit(harness, tasks_data[task], top_k=top_k)
        circuits[task] = res["circuit"]
        circuit_label_sets[task] = set(res["labels"])

    print(f"\n=======================================================")
    print(f"PHASE 2: CROSS-TASK CAUSAL TRANSFER MATRIX", now())
    print(f"=======================================================")
    num_tasks = len(task_names)
    transfer_matrix = np.zeros((num_tasks, num_tasks))
    jaccard_matrix = np.zeros((num_tasks, num_tasks))

    for i, s_task in enumerate(task_names):
        for j, t_task in enumerate(task_names):
            transfer_score = evaluate_transfer(
                harness, circuits[s_task], tasks_data[t_task]
            )
            transfer_matrix[i, j] = transfer_score

            inter = len(circuit_label_sets[s_task] & circuit_label_sets[t_task])
            union = len(circuit_label_sets[s_task] | circuit_label_sets[t_task])
            jaccard_matrix[i, j] = inter / max(union, 1)

    df_transfer = pd.DataFrame(
        transfer_matrix, index=task_names, columns=task_names
    ).round(4)
    df_jaccard = pd.DataFrame(
        jaccard_matrix, index=task_names, columns=task_names
    ).round(4)

    return df_transfer, df_jaccard
