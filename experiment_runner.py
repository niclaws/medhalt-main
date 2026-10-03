import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from patching_engine import CausalMediationHarness, ContrastiveMedHALTDataset
from datetime import datetime

# =====================================================================
# 1. Circuit Discovery (AIE Sweeps)
# =====================================================================

def compute_task_circuit(
        harness: CausalMediationHarness,
        dataset_samples: List[Dict],
        top_k: int = 10
) -> Dict[str, object]:
    """
    Calculates Average Indirect Effect (AIE) across all Attention Heads
    and MLP Blocks for a single task. Returns top-k components.
    """
    num_layers = harness.num_layers
    num_heads = harness.num_heads

    head_aie = np.zeros((num_layers, num_heads))
    mlp_aie = np.zeros(num_layers)

    valid_samples = 0
    total_samples = len(dataset_samples)
    device = harness.model.device

    print(f"\n[compute_task_circuit] Starting AIE sweep on {total_samples} samples.", datetime.now())
    print(f"[compute_task_circuit] Grid search dimensions: {num_layers} layers x {num_heads} heads + {num_layers} MLPs "
          f"= {num_layers * num_heads + num_layers} interventions per sample.", datetime.now())

    sweep_start = time.time()

    for idx, sample in enumerate(dataset_samples):
        clean_ids = sample["clean_ids"].to(device)
        corrupt_ids = sample["corrupt_ids"].to(device)
        t_id, h_id = sample["target_id"], sample["halluc_id"]

        d_clean = harness.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = harness.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        t_decoded = harness.tokenizer.decode([t_id])
        h_decoded = harness.tokenizer.decode([h_id])

        if abs(denom) < 1e-4:
            print(f"[WARN compute_task_circuit] Sample #{idx+1}/{total_samples} dropped: near-zero denominator ({denom:.6f}) | "
                  f"Target: '{t_decoded}'({t_id}) | Halluc: '{h_decoded}'({h_id}) | "
                  f"d_clean={d_clean:.4f}, d_corrupt={d_corrupt:.4f}", datetime.now())
            continue

        valid_samples += 1
        sample_start = time.time()

        # Cache residual states for the clean run
        head_cache, mlp_cache = harness.cache_activations(clean_ids)

        # 1. Sweep Attention Heads
        for l in range(num_layers):
            for h in range(num_heads):
                d_patched = harness.patch_component(
                    corrupt_ids, t_id, h_id, "head", l, head_cache[l], head_idx=h
                )
                head_aie[l, h] += (d_patched - d_corrupt) / denom

        # 2. Sweep MLPs
        for l in range(num_layers):
            d_patched = harness.patch_component(
                corrupt_ids, t_id, h_id, "mlp", l, mlp_cache[l]
            )
            mlp_aie[l] += (d_patched - d_corrupt) / denom

        elapsed = time.time() - sample_start
        print(f"[compute_task_circuit] Sample #{idx+1}/{total_samples} complete in {elapsed:.2f}s | "
              f"Denom: {denom:.4f} | Cumulative Valid: {valid_samples}", datetime.now())

    print(f"\n[compute_task_circuit] Sweep finished in {time.time() - sweep_start:.2f}s.", datetime.now())
    print(f"[compute_task_circuit] Valid samples utilized: {valid_samples}/{total_samples}", datetime.now())

    if valid_samples == 0:
        raise RuntimeError("Circuit discovery produced zero valid samples; refusing to fabricate an all-zero circuit.")

    head_aie /= valid_samples
    mlp_aie /= valid_samples

    print(f"[compute_task_circuit] Head AIE -> min: {head_aie.min():.4f}, max: {head_aie.max():.4f}, mean: {head_aie.mean():.4f}", datetime.now())
    print(f"[compute_task_circuit] MLP AIE  -> min: {mlp_aie.min():.4f}, max: {mlp_aie.max():.4f}, mean: {mlp_aie.mean():.4f}", datetime.now())

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

    print(f"[compute_task_circuit] Top-{top_k} Discovered Circuit:", datetime.now())
    for rank, (label, score) in enumerate(zip(circuit_labels, circuit_scores), 1):
        print(f"  Rank {rank:2d}: {label:<8} | AIE: {score:.4f}", datetime.now())

    return {
        "circuit": top_circuit,
        "labels": circuit_labels,
        "scores": circuit_scores,
        "head_aie": head_aie,
        "mlp_aie": mlp_aie
    }


# =====================================================================
# 2. Cross-Task Transfer Evaluation
# =====================================================================

def evaluate_transfer(
        harness: CausalMediationHarness,
        circuit: List[Tuple],
        target_samples: List[Dict]
) -> float:
    """
    Patches all components of a source circuit simultaneously into target task
    forward passes. Returns normalized transfer score T_{s -> t}.
    """
    total_transfer = 0.0
    valid_count = 0
    total_samples = len(target_samples)
    device = harness.model.device

    print(f"\n[evaluate_transfer] Testing transfer with circuit size: {len(circuit)} components.", datetime.now())

    for idx, sample in enumerate(target_samples):
        clean_ids = sample["clean_ids"].to(device)
        corrupt_ids = sample["corrupt_ids"].to(device)
        t_id, h_id = sample["target_id"], sample["halluc_id"]

        d_clean = harness.get_logit_diff(clean_ids, t_id, h_id)
        d_corrupt = harness.get_logit_diff(corrupt_ids, t_id, h_id)
        denom = d_clean - d_corrupt

        if abs(denom) < 1e-4:
            print(f"[WARN evaluate_transfer] Target sample #{idx+1}/{total_samples} dropped: small denom ({denom:.6f})", datetime.now())
            continue

        head_cache, mlp_cache = harness.cache_activations(clean_ids)

        handles = []
        # Register simultaneous hooks for the full circuit
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

                h = harness.layers[layer_idx].mlp.register_forward_hook(make_mlp_hook(clean_act))
                handles.append(h)

            elif comp_type == "head":
                clean_act = head_cache[layer_idx]
                h_idx = head_idx

                def make_head_hook(act, target_head):
                    def hook_fn(m, inp, outp):
                        curr = outp[0].clone()
                        B, S, _ = curr.shape
                        cr = curr.view(B, S, harness.num_heads, harness.head_dim)
                        ca = act.to(curr.device).view(
                            act.shape[0], act.shape[1], harness.num_heads, harness.head_dim
                        )
                        if cr.shape[1] != ca.shape[1]:
                            cr[:, -1, target_head, :] = ca[:, -1, target_head, :]
                        else:
                            cr[:, :, target_head, :] = ca[:, :, target_head, :]
                        return (cr.view_as(curr),) + outp[1:]
                    return hook_fn

                h = harness.layers[layer_idx].self_attn.register_forward_hook(make_head_hook(clean_act, h_idx))
                handles.append(h)

        with torch.inference_mode():
            outputs = harness.model(corrupt_ids)
            logits = outputs.logits[0, -1, :]
            patched_diff = (logits[t_id] - logits[h_id]).item()

        # Detach all simultaneous hooks
        for handle in handles:
            handle.remove()

        step_transfer = (patched_diff - d_corrupt) / denom
        total_transfer += step_transfer
        valid_count += 1

        print(f"[evaluate_transfer] Sample #{idx+1}/{total_samples} | "
              f"d_corrupt: {d_corrupt:.4f}, d_patched: {patched_diff:.4f}, denom: {denom:.4f} -> Transfer: {step_transfer:.4f}", datetime.now())

    final_transfer = float(total_transfer / max(valid_count, 1))
    print(f"[evaluate_transfer] Average Transfer over {valid_count}/{total_samples} samples: {final_transfer:.4f}", datetime.now())
    return final_transfer


# =====================================================================
# 3. Main Experiment Orchestrator
# =====================================================================

def run_cross_task_experiment(
        model_name: str,
        tasks_data: Dict[str, List[Dict]],
        is_qwen: bool = False,
        top_k: int = 15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes discovery for all 4 tasks and computes the 4x4 Causal Transfer Matrix
    and the Jaccard Circuit Overlap Matrix.
    """
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: Model={model_name} | QwenMode={is_qwen} | TopK={top_k}", datetime.now())
    print("=" * 70)

    harness = CausalMediationHarness(model_name, is_qwen=is_qwen)
    task_names = ["FCT", "NOTA", "PMID2Title", "Title2URL"]

    # Verify dataset readiness
    for task in task_names:
        count = len(tasks_data.get(task, []))
        print(f"[Dataset Check] Task '{task}': {count} pairs ready.", datetime.now())
        if count == 0:
            print(f"[ERROR] Task '{task}' has no input pairs! Exiting.", datetime.now())
            sys.exit(1)

    circuits = {}
    circuit_label_sets = {}

    print(f"\n=======================================================")
    print(f"PHASE 1: CIRCUIT DISCOVERY (Top-{top_k})", datetime.now())
    print(f"=======================================================")
    for task in task_names:
        print(f"\n>>> Running Discovery for Task: {task}", datetime.now())
        res = compute_task_circuit(harness, tasks_data[task], top_k=top_k)
        circuits[task] = res["circuit"]
        circuit_label_sets[task] = set(res["labels"])
        print(f"[COMPLETED] {task} Circuit: {res['labels']}", datetime.now())

    print(f"\n=======================================================")
    print(f"PHASE 2: CROSS-TASK CAUSAL TRANSFER MATRIX (T_{{s -> t}})", datetime.now())
    print(f"=======================================================")
    num_tasks = len(task_names)
    transfer_matrix = np.zeros((num_tasks, num_tasks))
    jaccard_matrix = np.zeros((num_tasks, num_tasks))

    for i, s_task in enumerate(task_names):
        for j, t_task in enumerate(task_names):
            print(f"\n--- Matrix Cell [{i},{j}]: Source='{s_task}' -> Target='{t_task}' ---", datetime.now())
            # Measure every cell empirically, including the diagonal.
            transfer_score = evaluate_transfer(harness, circuits[s_task], tasks_data[t_task])
            transfer_matrix[i, j] = transfer_score
            print(f"[RESULT] {s_task} -> {t_task} Transfer Score: {transfer_score:.4f}", datetime.now())

            # Jaccard overlap: |A & B| / |A | B|
            inter = len(circuit_label_sets[s_task] & circuit_label_sets[t_task])
            union = len(circuit_label_sets[s_task] | circuit_label_sets[t_task])
            j_score = inter / max(union, 1)
            jaccard_matrix[i, j] = j_score
            print(f"[RESULT] Circuit Overlap (Jaccard): {inter}/{union} = {j_score:.4f}", datetime.now())

    df_transfer = pd.DataFrame(transfer_matrix, index=task_names, columns=task_names).round(4)
    df_jaccard = pd.DataFrame(jaccard_matrix, index=task_names, columns=task_names).round(4)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY MATRICES", datetime.now())
    print("=" * 70)
    print("\nCAUSAL TRANSFER MATRIX (T_{s -> t}):", datetime.now())
    print(df_transfer.to_string())
    print("\nJACCARD CIRCUIT OVERLAP MATRIX:", datetime.now())
    print(df_jaccard.to_string())
    print("=" * 70 + "\n")

    return df_transfer, df_jaccard


# =====================================================================
# Standalone Smoke Test Block
# =====================================================================
if __name__ == "__main__":
    print("[INIT] Verifying patching runner definitions...")
    print("[INIT] Loaded compute_task_circuit, evaluate_transfer, run_cross_task_experiment successfully.")