"""
How to run this code file ?

python calculate_hallucination_metrics.py \
    --input results_new/m42-health_Llama3-Med42-8B/n100_k20_seed42/m42-health_Llama3-Med42-8B_per_sample_transfer.csv \
    --output_dir results_new/m42-health_Llama3-Med42-8B/n100_k20_seed42/hallucination_analysis \
    --min_denom 0.0001 \
    --primary_k 20
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Interpretation
# ============================================================
#
# d = logit(target) - logit(hallucinated/incorrect token)
#
# d > 0:
#     correct target is preferred
#
# d < 0:
#     hallucinated / incorrect token is preferred
#
# Therefore:
#
# Hallucination before transfer:
#     d_corrupt < 0
#
# Hallucination after transfer:
#     d_patched < 0
#
# Transfer-induced hallucination:
#     d_corrupt >= 0 AND d_patched < 0
#
# Hallucination rescue:
#     d_corrupt < 0 AND d_patched >= 0
#
# ============================================================


def safe_div(num, den):
    if den == 0:
        return np.nan
    return num / den


def calculate_group_metrics(group):
    """
    Calculate hallucination metrics for one:
        source_task -> target_task, k
    """

    n = len(group)

    before_halluc = group["d_corrupt"] < 0
    before_correct = group["d_corrupt"] >= 0

    after_halluc = group["d_patched"] < 0
    after_correct = group["d_patched"] >= 0

    # --------------------------------------------------------
    # Basic hallucination rates
    # --------------------------------------------------------

    pre_halluc_count = int(before_halluc.sum())
    post_halluc_count = int(after_halluc.sum())

    pre_halluc_rate = safe_div(pre_halluc_count, n)
    post_halluc_rate = safe_div(post_halluc_count, n)

    # --------------------------------------------------------
    # Transfer-induced hallucination
    #
    # Model was correct before patching,
    # but hallucinated token becomes preferred after patching.
    # --------------------------------------------------------

    induced_mask = before_correct & after_halluc

    induced_count = int(induced_mask.sum())
    initially_correct_count = int(before_correct.sum())

    hallucination_induction_rate = safe_div(
        induced_count,
        initially_correct_count
    )

    # Fraction over ALL samples, useful as a secondary metric.
    hallucination_induction_rate_all = safe_div(
        induced_count,
        n
    )

    # --------------------------------------------------------
    # Hallucination rescue
    #
    # Model preferred hallucinated token before patching,
    # but correct token becomes preferred after patching.
    # --------------------------------------------------------

    rescue_mask = before_halluc & after_correct

    rescue_count = int(rescue_mask.sum())

    hallucination_rescue_rate = safe_div(
        rescue_count,
        pre_halluc_count
    )

    hallucination_rescue_rate_all = safe_div(
        rescue_count,
        n
    )

    # --------------------------------------------------------
    # Hallucination persistence
    #
    # Hallucinating before AND after transfer.
    # --------------------------------------------------------

    persistence_mask = before_halluc & after_halluc

    persistence_count = int(persistence_mask.sum())

    hallucination_persistence_rate = safe_div(
        persistence_count,
        pre_halluc_count
    )

    # --------------------------------------------------------
    # Correct preference retention
    #
    # Correct before AND still correct after transfer.
    # --------------------------------------------------------

    correct_retention_mask = before_correct & after_correct

    correct_retention_count = int(correct_retention_mask.sum())

    correct_retention_rate = safe_div(
        correct_retention_count,
        initially_correct_count
    )

    # --------------------------------------------------------
    # Change in hallucination probability
    #
    # > 0  : transfer increases hallucination
    # < 0  : transfer reduces hallucination
    # --------------------------------------------------------

    hallucination_rate_change = (
        post_halluc_rate - pre_halluc_rate
        if np.isfinite(post_halluc_rate)
        and np.isfinite(pre_halluc_rate)
        else np.nan
    )

    # Percentage-point version
    hallucination_rate_change_pp = (
        100 * hallucination_rate_change
        if np.isfinite(hallucination_rate_change)
        else np.nan
    )

    # --------------------------------------------------------
    # Logit-gap changes
    # --------------------------------------------------------

    logit_shift = group["d_patched"] - group["d_corrupt"]

    mean_logit_shift = logit_shift.mean()
    median_logit_shift = logit_shift.median()

    # --------------------------------------------------------
    # Negative transfer
    #
    # IMPORTANT:
    # normalized_transfer < 0 is NOT automatically hallucination.
    # It only means patching moved in the wrong direction.
    # --------------------------------------------------------

    if "normalized_transfer" in group.columns:
        negative_transfer_rate = (
            group["normalized_transfer"] < 0
        ).mean()

        mean_transfer = group["normalized_transfer"].mean()
        median_transfer = group["normalized_transfer"].median()

        overshoot_rate = (
            group["normalized_transfer"] > 1
        ).mean()
    else:
        negative_transfer_rate = np.nan
        mean_transfer = np.nan
        median_transfer = np.nan
        overshoot_rate = np.nan

    return pd.Series({
        "valid_samples": n,

        "pre_hallucination_count": pre_halluc_count,
        "post_hallucination_count": post_halluc_count,

        "pre_hallucination_rate": pre_halluc_rate,
        "post_hallucination_rate": post_halluc_rate,

        "initially_correct_count": initially_correct_count,

        "transfer_induced_hallucination_count": induced_count,
        "hallucination_induction_rate": hallucination_induction_rate,
        "hallucination_induction_rate_all": hallucination_induction_rate_all,

        "hallucination_rescue_count": rescue_count,
        "hallucination_rescue_rate": hallucination_rescue_rate,
        "hallucination_rescue_rate_all": hallucination_rescue_rate_all,

        "hallucination_persistence_count": persistence_count,
        "hallucination_persistence_rate": hallucination_persistence_rate,

        "correct_retention_count": correct_retention_count,
        "correct_retention_rate": correct_retention_rate,

        "hallucination_rate_change": hallucination_rate_change,
        "hallucination_rate_change_pp": hallucination_rate_change_pp,

        "mean_logit_shift": mean_logit_shift,
        "median_logit_shift": median_logit_shift,

        "negative_transfer_rate": negative_transfer_rate,
        "overshoot_rate": overshoot_rate,

        "mean_normalized_transfer": mean_transfer,
        "median_normalized_transfer": median_transfer,
    })


def save_heatmap(
    summary,
    metric,
    k,
    output_path,
    cross_task_only=True,
):
    subset = summary[summary["k"] == k].copy()

    matrix = subset.pivot(
        index="source_task",
        columns="target_task",
        values=metric
    )

    # Keep a consistent ordering
    tasks = sorted(
        set(matrix.index).union(matrix.columns)
    )

    matrix = matrix.reindex(
        index=tasks,
        columns=tasks
    )

    # If we only care about cross-task effects,
    # remove source == target diagonal.
    if cross_task_only:
        for task in tasks:
            if task in matrix.index and task in matrix.columns:
                matrix.loc[task, task] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto"
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(
        matrix.columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Target Task")
    ax.set_ylabel("Source Circuit")

    readable_name = metric.replace("_", " ").title()

    ax.set_title(
        f"{readable_name} (k={k})"
    )

    values = matrix.to_numpy(dtype=float)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center"
                )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_k_curve(summary, output_path):
    """
    Average hallucination behavior across OFF-DIAGONAL
    source -> target task pairs for each k.
    """

    cross = summary[
        summary["source_task"]
        != summary["target_task"]
    ].copy()

    metrics = [
        "hallucination_induction_rate",
        "hallucination_rescue_rate",
        "post_hallucination_rate",
    ]

    grouped = (
        cross
        .groupby("k")[metrics]
        .mean()
        .reset_index()
        .sort_values("k")
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for metric in metrics:
        ax.plot(
            grouped["k"],
            grouped[metric],
            marker="o",
            label=metric.replace("_", " ").title()
        )

    ax.set_xlabel("Circuit Size k")
    ax.set_ylabel("Mean Cross-Task Rate")

    ax.set_title(
        "Cross-Task Hallucination Metrics vs Circuit Size"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return grouped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to *_per_sample_transfer.csv"
    )

    parser.add_argument(
        "--output_dir",
        default="hallucination_analysis"
    )

    parser.add_argument(
        "--min_denom",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--primary_k",
        type=int,
        default=20
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ========================================================
    # Load results
    # ========================================================

    df = pd.read_csv(args.input)

    required_columns = [
        "source_task",
        "target_task",
        "k",
        "d_corrupt",
        "d_patched",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"Loaded {len(df):,} rows"
    )

    print(
        "k values:",
        sorted(df["k"].unique())
    )

    # ========================================================
    # Filter unusable rows
    # ========================================================

    # The original experiment stores patched=False for samples
    # whose clean-corrupt denominator is too small.
    if "patched" in df.columns:

        if df["patched"].dtype == object:
            df["patched"] = (
                df["patched"]
                .astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False
                })
            )

        df = df[df["patched"] == True].copy()

    df = df[
        np.isfinite(df["d_corrupt"])
        & np.isfinite(df["d_patched"])
    ].copy()

    # Apply same denominator protection as original experiment.
    if "abs_denominator" in df.columns:
        df = df[
            df["abs_denominator"]
            >= args.min_denom
        ].copy()

    elif "denominator" in df.columns:
        df = df[
            df["denominator"].abs()
            >= args.min_denom
        ].copy()

    print(
        f"Valid rows after filtering: {len(df):,}"
    )

    # ========================================================
    # Add sample-level transition labels
    # ========================================================

    df["pre_hallucination"] = (
        df["d_corrupt"] < 0
    )

    df["post_hallucination"] = (
        df["d_patched"] < 0
    )

    df["transfer_induced_hallucination"] = (
        (df["d_corrupt"] >= 0)
        & (df["d_patched"] < 0)
    )

    df["hallucination_rescued"] = (
        (df["d_corrupt"] < 0)
        & (df["d_patched"] >= 0)
    )

    df["hallucination_persisted"] = (
        (df["d_corrupt"] < 0)
        & (df["d_patched"] < 0)
    )

    df["correct_preference_retained"] = (
        (df["d_corrupt"] >= 0)
        & (df["d_patched"] >= 0)
    )

    df["logit_shift"] = (
        df["d_patched"]
        - df["d_corrupt"]
    )

    # Save enriched sample-level CSV
    enriched_path = os.path.join(
        args.output_dir,
        "per_sample_transfer_with_hallucination_labels.csv"
    )

    df.to_csv(
        enriched_path,
        index=False
    )

    # ========================================================
    # Aggregate source -> target -> k
    # ========================================================

    group_columns = [
        "source_task",
        "target_task",
        "k",
    ]

    if "model" in df.columns:
        group_columns = [
            "model",
            *group_columns
        ]

    summary = (
        df
        .groupby(
            group_columns,
            dropna=False
        )
        .apply(
            calculate_group_metrics,
            include_groups=False
        )
        .reset_index()
    )

    summary_path = os.path.join(
        args.output_dir,
        "cross_task_hallucination_metrics.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        f"\nSaved:\n{summary_path}"
    )

    # ========================================================
    # Overall cross-task averages for every k
    # ========================================================

    cross = summary[
        summary["source_task"]
        != summary["target_task"]
    ].copy()

    important_metrics = [
        "pre_hallucination_rate",
        "post_hallucination_rate",
        "hallucination_induction_rate",
        "hallucination_rescue_rate",
        "hallucination_persistence_rate",
        "hallucination_rate_change",
        "negative_transfer_rate",
        "mean_normalized_transfer",
    ]

    k_summary = (
        cross
        .groupby("k")[important_metrics]
        .mean()
        .reset_index()
        .sort_values("k")
    )

    k_summary_path = os.path.join(
        args.output_dir,
        "cross_task_hallucination_by_k.csv"
    )

    k_summary.to_csv(
        k_summary_path,
        index=False
    )

    print(
        f"Saved:\n{k_summary_path}"
    )

    print(
        "\nCross-task hallucination summary by k:"
    )

    print(
        k_summary.to_string(index=False)
    )

    # ========================================================
    # Heatmaps for every k
    # ========================================================

    heatmap_metrics = [
        "post_hallucination_rate",
        "hallucination_induction_rate",
        "hallucination_rescue_rate",
        "hallucination_rate_change",
    ]

    for k in sorted(summary["k"].unique()):

        for metric in heatmap_metrics:

            output_path = os.path.join(
                args.output_dir,
                f"{metric}_k{k}.png"
            )

            save_heatmap(
                summary,
                metric,
                k,
                output_path,
                cross_task_only=True,
            )

    # ========================================================
    # k trajectory plot
    # ========================================================

    grouped = save_k_curve(
        summary,
        os.path.join(
            args.output_dir,
            "cross_task_hallucination_vs_k.png"
        )
    )

    grouped.to_csv(
        os.path.join(
            args.output_dir,
            "cross_task_hallucination_plot_values.csv"
        ),
        index=False
    )

    # ========================================================
    # Print primary k
    # ========================================================

    primary = cross[
        cross["k"] == args.primary_k
    ].copy()

    print(
        f"\n======================================"
    )

    print(
        f"CROSS-TASK RESULTS AT k={args.primary_k}"
    )

    print(
        f"======================================\n"
    )

    columns_to_show = [
        "source_task",
        "target_task",
        "pre_hallucination_rate",
        "post_hallucination_rate",
        "hallucination_induction_rate",
        "hallucination_rescue_rate",
        "hallucination_rate_change",
        "mean_normalized_transfer",
    ]

    print(
        primary[columns_to_show]
        .sort_values(
            ["source_task", "target_task"]
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

"""
How to run this code file ?

python calculate_hallucination_metrics.py \
    --input results_new/m42-health_Llama3-Med42-8B/n100_k20_seed42/m42-health_Llama3-Med42-8B_per_sample_transfer.csv \
    --output_dir results_new/m42-health_Llama3-Med42-8B/n100_k20_seed42/hallucination_analysis \
    --min_denom 0.0001 \
    --primary_k 20
"""