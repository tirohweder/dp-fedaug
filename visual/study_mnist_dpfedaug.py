"""
DP-FedAug MNIST Study Analysis

Comprehensive analysis of 809 experimental runs examining the interplay between:
- Differential Privacy (DP) budget (epsilon)
- Synthetic data augmentation (synthetic_count)
- Data heterogeneity (partitioning: extreme vs dirichlet, alpha)
- Dataset scale (total_n)

Generates publication-quality figures suitable for academic venues (NeurIPS, ICML, etc.)
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import wandb

# Use shared thesis style
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual.data import get_cfg_value, load_or_fetch_dataframe, to_float
from visual.thesis_style import (
    apply_thesis_style,
    COLORS as THESIS_COLORS,
    EPSILON_COLORS,
    SYNTH_CMAP,
)


PROJECT_NAME = "DP-FedAug-MNIST-Study"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "mnist_dpfedaug_study"

# Extended palette for this study (builds on thesis_style)
COLORS = {
    **THESIS_COLORS,
    "extreme": "#E63946",
    "dirichlet_low": "#F4A261",
    "dirichlet_mid": "#2A9D8F",
    "iid": "#264653",
    "dp_strong": "#7B2CBF",
    "dp_medium": "#9D4EDD",
    "dp_weak": "#C77DFF",
    "no_dp": "#E0AAFF",
}

# Noise multiplier colors (higher noise = stronger privacy)
NOISE_COLORS = {
    "none": "#48CAE4",
    "2.0": "#E0AAFF",
    "5.0": "#C77DFF",
    "10": "#9D4EDD",
    "15": "#7B2CBF",
}

HETEROGENEITY_MARKERS = {
    "extreme": "s",
    "dirichlet_0.1": "^",
    "dirichlet_1.0": "o",
    "iid": "D",
}


def _to_float(value):
    return to_float(value)


def _get_cfg_value(cfg: dict, *keys):
    return get_cfg_value(cfg, *keys)


def fetch_wandb_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    """Fetch all runs from W&B project."""
    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)
    rows = []

    for run in runs:
        cfg = run.config
        summary = run.summary

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "project": run.project,
            "entity": getattr(run, "entity", None),
            "state": run.state,
            "created_at": run.created_at,
            "dataset": _get_cfg_value(cfg, "dataset"),
            "total_n": _get_cfg_value(cfg, "total-n", "total_n"),
            "synthetic_count": _get_cfg_value(cfg, "synthetic-count", "synthetic_count"),
            "alpha": _get_cfg_value(cfg, "non-iid-alpha", "non_iid_alpha", "alpha"),
            "partitioning": _get_cfg_value(cfg, "partitioning"),
            "balancing": _get_cfg_value(cfg, "balancing"),
            "seed": _get_cfg_value(cfg, "seed"),
            "target_epsilon": _get_cfg_value(cfg, "target-epsilon", "target_epsilon", "epsilon"),
            "noise_multiplier": _get_cfg_value(cfg, "noise-multiplier", "noise_multiplier"),
            "num_clients": _get_cfg_value(cfg, "num-clients", "num_clients"),
            "rounds": _get_cfg_value(cfg, "rounds", "num_rounds"),
        }

        for key in [
            "Global Test/accuracy",
            "Global Test/accuracy (Max)",
            "Global Test/Accuracy",
            "Global Test/Accuracy (Max)",
            "Global Test/auc",
            "Global Test/AUC",
            "Global Test/loss",
            "Global Test/f1",
            "Global Test/precision",
            "Global Test/recall",
            "Global Test/ap",
            "Communication Round",
            "Avg Client Epsilon",
        ]:
            if key in summary:
                row[key] = summary.get(key)

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean the dataframe."""
    df = df.copy()

    df["total_n"] = pd.to_numeric(df["total_n"], errors="coerce")
    df["synthetic_count"] = pd.to_numeric(df["synthetic_count"], errors="coerce")
    df["alpha"] = df["alpha"].apply(_to_float)
    df["target_epsilon_num"] = df["target_epsilon"].apply(_to_float)
    df["noise_multiplier"] = pd.to_numeric(df["noise_multiplier"], errors="coerce")

    # Extract accuracy with fallbacks
    if "Global Test/accuracy (Max)" in df.columns:
        df["accuracy"] = df["Global Test/accuracy (Max)"]
    elif "Global Test/Accuracy (Max)" in df.columns:
        df["accuracy"] = df["Global Test/Accuracy (Max)"]
    else:
        df["accuracy"] = np.nan
    if "Global Test/accuracy" in df.columns:
        df["accuracy"] = df["accuracy"].combine_first(df["Global Test/accuracy"])
    if "Global Test/Accuracy" in df.columns:
        df["accuracy"] = df["accuracy"].combine_first(df["Global Test/Accuracy"])
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    # Extract AUC
    auc = df.get("Global Test/auc")
    if auc is None:
        auc = df.get("Global Test/AUC")
    df["auc"] = pd.to_numeric(auc, errors="coerce")

    # Achieved epsilon
    if "Avg Client Epsilon" in df.columns:
        df["epsilon_achieved"] = df["Avg Client Epsilon"]
    else:
        df["epsilon_achieved"] = np.nan
    df["epsilon_achieved"] = df["epsilon_achieved"].combine_first(df["target_epsilon_num"])
    df["epsilon_achieved"] = pd.to_numeric(df["epsilon_achieved"], errors="coerce")

    # Create categorical privacy label based on noise_multiplier
    # Higher noise = stronger privacy, 0 = no DP
    def privacy_label(nm):
        if nm is None or (isinstance(nm, float) and np.isnan(nm)):
            return "none"
        if nm == 0 or nm == 0.0:
            return "none"
        # Round to reasonable precision for labeling
        return f"{nm:.1f}" if nm < 10 else f"{nm:.0f}"

    df["privacy_label"] = df["noise_multiplier"].apply(privacy_label)
    # Keep epsilon_label for backward compatibility but base on noise_multiplier
    df["epsilon_label"] = df["privacy_label"]

    # Create heterogeneity label
    def heterogeneity_label(row):
        part = row.get("partitioning", "")
        alpha = row.get("alpha")
        if part == "extreme":
            return "Pathological (1-class/client)"
        elif isinstance(alpha, float) and math.isinf(alpha):
            return "IID"
        elif alpha is not None:
            return f"Dirichlet α={alpha}"
        return "Unknown"

    df["heterogeneity"] = df.apply(heterogeneity_label, axis=1)

    # Heterogeneity order for sorting (most heterogeneous first)
    def heterogeneity_order(row):
        part = row.get("partitioning", "")
        alpha = row.get("alpha")
        if part == "extreme":
            return (0, 0)
        elif isinstance(alpha, float) and math.isinf(alpha):
            return (2, float('inf'))
        elif alpha is not None:
            return (1, alpha)
        return (3, 0)

    df["het_order"] = df.apply(heterogeneity_order, axis=1)

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics grouped by key experimental factors."""
    group_cols = ["total_n", "synthetic_count", "alpha", "epsilon_label",
                  "partitioning", "balancing", "heterogeneity"]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            auc_mean=("auc", "mean"),
            epsilon_mean=("epsilon_achieved", "mean"),
        )
        .reset_index()
    )
    # Compute 95% CI
    grouped["accuracy_sem"] = grouped["accuracy_std"] / np.sqrt(grouped["runs"])
    grouped["accuracy_ci95"] = grouped.apply(
        lambda row: stats.t.ppf(0.975, row["runs"] - 1) * row["accuracy_sem"]
        if row["runs"] > 1 else 0,
        axis=1
    )
    return grouped


def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval using t-distribution."""
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h



# =============================================================================
# FIGURE 1: Main Results - Privacy-Utility-Augmentation Tradeoff
# =============================================================================
def plot_main_results_figure(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Main thesis figure: 3-panel showing key findings.
    (a) Accuracy vs Epsilon for different synthetic counts
    (b) Accuracy vs Synthetic count at different privacy levels
    (c) Heatmap of accuracy across epsilon x synthetic_count
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Use the most common total_n for main figure
    total_n_counts = df["total_n"].value_counts()
    main_total_n = total_n_counts.idxmax() if not total_n_counts.empty else df["total_n"].dropna().iloc[0]
    df_main = df[df["total_n"] == main_total_n].copy()
    summary_main = summary_df[summary_df["total_n"] == main_total_n].copy()

    # Dynamic privacy order: none first, then sorted by noise (ascending = weaker to stronger)
    privacy_labels = df_main["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    # Sort numerically (lower noise = weaker privacy first)
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)

    synth_values = sorted(df_main["synthetic_count"].dropna().unique())

    # Panel A: Accuracy vs Noise Multiplier by synthetic count
    ax = axes[0]
    for i, synth in enumerate(synth_values):
        synth_df = df_main[df_main["synthetic_count"] == synth]
        means = []
        ci_lows = []
        ci_highs = []
        x_positions = []

        for j, priv in enumerate(privacy_order):
            priv_data = synth_df[synth_df["epsilon_label"] == priv]["accuracy"].dropna()
            if len(priv_data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(priv_data)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                x_positions.append(j)

        if means:
            color = plt.cm.viridis(i / max(1, len(synth_values) - 1))
            ax.errorbar(
                x_positions, means,
                yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
                marker="o", capsize=3, label=f"Synth={int(synth)}", color=color,
                linewidth=1.5, markersize=5
            )

    ax.set_xticks(range(len(privacy_order)))
    xlabels = ["No DP" if p == "none" else f"σ={p}" for p in privacy_order]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Noise Multiplier (σ)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Privacy-Utility Tradeoff by Augmentation")
    ax.legend(title="Synthetic Data", loc="lower left", fontsize=7)
    ax.annotate("Stronger Privacy →", xy=(0.75, 0.02), xycoords="axes fraction",
               fontsize=8, color="#666666", ha="right")

    # Panel B: Accuracy vs Synthetic count at different privacy levels
    ax = axes[1]
    for priv in privacy_order:
        priv_df = df_main[df_main["epsilon_label"] == priv]
        means = []
        ci_lows = []
        ci_highs = []
        x_vals = []

        for synth in synth_values:
            synth_data = priv_df[priv_df["synthetic_count"] == synth]["accuracy"].dropna()
            if len(synth_data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(synth_data)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                x_vals.append(synth)

        if means:
            color = NOISE_COLORS.get(priv, COLORS["neutral"])
            label = "No DP" if priv == "none" else f"σ={priv}"
            linestyle = "--" if priv == "none" else "-"
            ax.errorbar(
                x_vals, means,
                yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
                marker="o", capsize=3, label=label, color=color,
                linewidth=1.5, markersize=5, linestyle=linestyle
            )

    ax.set_xlabel("Synthetic Data Count")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(b) Augmentation Effectiveness")
    ax.legend(title="Noise (σ)", loc="lower right", fontsize=7)

    # Panel C: Heatmap
    ax = axes[2]
    pivot = summary_main.pivot_table(
        index="synthetic_count",
        columns="epsilon_label",
        values="accuracy_mean",
        aggfunc="mean"
    )
    # Reorder columns
    pivot = pivot.reindex(columns=[c for c in privacy_order if c in pivot.columns])
    # Rename columns for display
    pivot.columns = ["No DP" if c == "none" else f"σ={c}" for c in pivot.columns]

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        center=pivot.values.mean(), ax=ax,
        cbar_kws={"label": "Accuracy", "shrink": 0.8},
        linewidths=0.5, linecolor="white"
    )
    ax.set_xlabel("Noise Multiplier (σ)")
    ax.set_ylabel("Synthetic Count")
    ax.set_title(f"(c) Accuracy Heatmap (N={int(main_total_n)})")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_main_results.pdf")
    fig.savefig(output_dir / "fig1_main_results.png")
    plt.close(fig)
    print("  Created: fig1_main_results.pdf/png")


# =============================================================================
# FIGURE 2: Heterogeneity Impact Under DP
# =============================================================================
def plot_heterogeneity_impact(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Figure showing how data heterogeneity interacts with DP.
    (a) Accuracy by heterogeneity level at different privacy budgets
    (b) Delta from non-DP baseline by heterogeneity
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    df_valid = df.dropna(subset=["partitioning", "alpha", "epsilon_label"])

    # Get unique heterogeneity levels sorted by het_order
    het_order_map = df_valid.groupby("heterogeneity")["het_order"].first().to_dict()
    het_labels = sorted(het_order_map.keys(), key=lambda x: het_order_map[x])

    # Dynamic privacy order
    privacy_labels = df_valid["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)
    het_colors = {
        "Pathological (1-class/client)": COLORS["extreme"],
        "IID": COLORS["iid"],
    }
    # Add colors for Dirichlet alphas
    dirichlet_hets = [h for h in het_labels if h.startswith("Dirichlet")]
    dirichlet_cmap = plt.cm.autumn
    for i, h in enumerate(dirichlet_hets):
        het_colors[h] = dirichlet_cmap(i / max(1, len(dirichlet_hets)))

    # Panel A: Grouped bar chart - accuracy by heterogeneity at each privacy level
    ax = axes[0]
    x = np.arange(len(privacy_order))
    width = 0.8 / len(het_labels)

    for i, het in enumerate(het_labels):
        means = []
        stds = []
        for priv in privacy_order:
            data = df_valid[(df_valid["heterogeneity"] == het) &
                           (df_valid["epsilon_label"] == priv)]["accuracy"].dropna()
            means.append(data.mean() if len(data) > 0 else np.nan)
            stds.append(data.std() if len(data) > 1 else 0)

        offset = (i - len(het_labels)/2 + 0.5) * width
        color = het_colors.get(het, COLORS["neutral"])
        short_label = het.replace("Dirichlet ", "Dir ").replace("(1-class/client)", "")
        ax.bar(x + offset, means, width, label=short_label, color=color,
               yerr=stds, capsize=2, alpha=0.8)

    ax.set_xticks(x)
    xlabels = ["No DP" if p == "none" else f"σ={p}" for p in privacy_order]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Noise Multiplier (σ)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Accuracy by Heterogeneity Level")
    ax.legend(title="Heterogeneity", loc="upper right", fontsize=7, ncol=2)

    # Panel B: Accuracy drop from non-DP baseline
    ax = axes[1]

    # Calculate non-DP baseline for each heterogeneity
    baselines = {}
    for het in het_labels:
        baseline_data = df_valid[(df_valid["heterogeneity"] == het) &
                                 (df_valid["epsilon_label"] == "none")]["accuracy"].dropna()
        baselines[het] = baseline_data.mean() if len(baseline_data) > 0 else np.nan

    # DP levels (exclude "none")
    dp_levels = [p for p in privacy_order if p != "none"]
    x = np.arange(len(dp_levels))
    width = 0.8 / max(len(het_labels), 1)

    for i, het in enumerate(het_labels):
        deltas = []
        for priv in dp_levels:
            data = df_valid[(df_valid["heterogeneity"] == het) &
                           (df_valid["epsilon_label"] == priv)]["accuracy"].dropna()
            dp_mean = data.mean() if len(data) > 0 else np.nan
            delta = (dp_mean - baselines.get(het, np.nan)) * 100  # percentage points
            deltas.append(delta)

        offset = (i - len(het_labels)/2 + 0.5) * width
        color = het_colors.get(het, COLORS["neutral"])
        short_label = het.replace("Dirichlet ", "Dir ").replace("(1-class/client)", "")
        ax.bar(x + offset, deltas, width, label=short_label, color=color, alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(x)
    xlabels = [f"σ={p}" for p in dp_levels]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Noise Multiplier (σ)")
    ax.set_ylabel("Accuracy Change (pp)")
    ax.set_title("(b) Privacy Cost by Heterogeneity")
    ax.legend(title="Heterogeneity", loc="lower left", fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_heterogeneity_impact.pdf")
    fig.savefig(output_dir / "fig2_heterogeneity_impact.png")
    plt.close(fig)
    print("  Created: fig2_heterogeneity_impact.pdf/png")


# =============================================================================
# FIGURE 3: Synthetic Augmentation Recovery Analysis
# =============================================================================
def plot_augmentation_recovery(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Figure showing how synthetic augmentation recovers utility lost to DP.
    Key insight: Does augmentation help more under stricter privacy?
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    df_all = df.copy()

    # Dynamic privacy order
    privacy_labels = df_all["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)
    dp_levels = [p for p in privacy_order if p != "none"]

    # Focus on runs with DP enabled
    df_dp = df[df["epsilon_label"].isin(dp_levels)].copy()

    synth_values = sorted(df["synthetic_count"].dropna().unique())

    # Panel A: Recovery rate - how much of the DP loss is recovered by augmentation
    ax = axes[0]

    # Calculate baseline (no DP, no synth) accuracy
    baseline_data = df_all[(df_all["epsilon_label"] == "none") &
                           (df_all["synthetic_count"] == 0)]["accuracy"].dropna()
    baseline_acc = baseline_data.mean() if len(baseline_data) > 0 else np.nan

    for priv in dp_levels:
        priv_df = df_all[df_all["epsilon_label"] == priv]

        # Baseline with DP but no synth
        dp_baseline_data = priv_df[priv_df["synthetic_count"] == 0]["accuracy"].dropna()
        dp_baseline = dp_baseline_data.mean() if len(dp_baseline_data) > 0 else np.nan

        # Loss due to DP
        dp_loss = baseline_acc - dp_baseline

        recovery_rates = []
        synth_x = []
        for synth in synth_values:
            if synth == 0:
                continue
            synth_data = priv_df[priv_df["synthetic_count"] == synth]["accuracy"].dropna()
            if len(synth_data) > 0:
                synth_acc = synth_data.mean()
                # Recovery = (synth_acc - dp_baseline) / dp_loss
                if dp_loss > 0:
                    recovery = ((synth_acc - dp_baseline) / dp_loss) * 100
                else:
                    recovery = 0
                recovery_rates.append(recovery)
                synth_x.append(synth)

        if recovery_rates:
            color = NOISE_COLORS.get(priv, COLORS["neutral"])
            ax.plot(synth_x, recovery_rates, marker="o", label=f"σ={priv}",
                   color=color, linewidth=2, markersize=6)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Synthetic Data Count")
    ax.set_ylabel("Utility Recovery (%)")
    ax.set_title("(a) Recovery of DP-Induced Loss")
    ax.legend(title="Privacy", loc="lower right")
    ax.text(0.02, 0.98, "100% = Full recovery", transform=ax.transAxes,
           fontsize=8, color="#666", va="top")

    # Panel B: Absolute improvement from augmentation at each privacy level
    ax = axes[1]

    for priv in privacy_order:
        priv_df = df_all[df_all["epsilon_label"] == priv]
        base_data = priv_df[priv_df["synthetic_count"] == 0]["accuracy"].dropna()
        base_mean = base_data.mean() if len(base_data) > 0 else np.nan

        improvements = []
        synth_x = []
        for synth in synth_values:
            if synth == 0:
                continue
            synth_data = priv_df[priv_df["synthetic_count"] == synth]["accuracy"].dropna()
            if len(synth_data) > 0:
                improvement = (synth_data.mean() - base_mean) * 100
                improvements.append(improvement)
                synth_x.append(synth)

        if improvements:
            color = NOISE_COLORS.get(priv, COLORS["neutral"])
            label = "No DP" if priv == "none" else f"σ={priv}"
            linestyle = "--" if priv == "none" else "-"
            ax.plot(synth_x, improvements, marker="o", label=label,
                   color=color, linewidth=2, markersize=6, linestyle=linestyle)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Synthetic Data Count")
    ax.set_ylabel("Accuracy Improvement (pp)")
    ax.set_title("(b) Augmentation Benefit by Privacy Level")
    ax.legend(title="Privacy", loc="best")

    # Panel C: Marginal benefit of augmentation
    ax = axes[2]

    # Heatmap showing augmentation benefit (synth>0 vs synth=0) at each privacy x heterogeneity
    df_valid = df_all.dropna(subset=["partitioning", "alpha", "epsilon_label"])
    het_labels = df_valid["heterogeneity"].unique()

    benefit_data = []
    for het in het_labels:
        het_df = df_valid[df_valid["heterogeneity"] == het]
        row_data = {"heterogeneity": het}
        for priv in privacy_order:
            priv_df = het_df[het_df["epsilon_label"] == priv]
            no_synth = priv_df[priv_df["synthetic_count"] == 0]["accuracy"].dropna()
            with_synth = priv_df[priv_df["synthetic_count"] > 0]["accuracy"].dropna()

            if len(no_synth) > 0 and len(with_synth) > 0:
                benefit = (with_synth.mean() - no_synth.mean()) * 100
            else:
                benefit = np.nan
            row_data[priv] = benefit
        benefit_data.append(row_data)

    benefit_df = pd.DataFrame(benefit_data)
    benefit_df = benefit_df.set_index("heterogeneity")
    benefit_df = benefit_df[[p for p in privacy_order if p in benefit_df.columns]]
    benefit_df.columns = ["No DP" if c == "none" else f"σ={c}" for c in benefit_df.columns]

    if not benefit_df.empty and benefit_df.notna().any().any():
        sns.heatmap(
            benefit_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            ax=ax, cbar_kws={"label": "Benefit (pp)", "shrink": 0.8},
            linewidths=0.5, linecolor="white"
        )
        ax.set_xlabel("Privacy Budget")
        ax.set_ylabel("Heterogeneity")
        ax.set_title("(c) Augmentation Benefit Heatmap")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
               transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_augmentation_recovery.pdf")
    fig.savefig(output_dir / "fig3_augmentation_recovery.png")
    plt.close(fig)
    print("  Created: fig3_augmentation_recovery.pdf/png")


# =============================================================================
# FIGURE 4: Pareto Frontier Analysis
# =============================================================================
def plot_pareto_frontier(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Pareto frontier showing privacy-utility tradeoffs.
    Highlights configurations that achieve best accuracy for given privacy.
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    df_valid = df.dropna(subset=["epsilon_achieved", "accuracy"]).copy()

    # Remove infinite epsilon for Pareto analysis (or assign a high value)
    df_valid = df_valid[df_valid["epsilon_achieved"] < 100]

    # Panel A: Scatter with Pareto frontier
    ax = axes[0]

    synth_values = sorted(df_valid["synthetic_count"].dropna().unique())

    for synth in synth_values:
        synth_df = df_valid[df_valid["synthetic_count"] == synth]
        color = plt.cm.viridis(synth / max(synth_values) if max(synth_values) > 0 else 0)
        ax.scatter(synth_df["epsilon_achieved"], synth_df["accuracy"],
                  alpha=0.5, s=30, c=[color], label=f"Synth={int(synth)}")

    # Compute and plot Pareto frontier
    # Group by epsilon and take max accuracy
    pareto_df = df_valid.groupby(["epsilon_label", "synthetic_count"]).agg(
        epsilon=("epsilon_achieved", "mean"),
        accuracy=("accuracy", "max")
    ).reset_index()

    # Sort by epsilon ascending, then find Pareto optimal points
    pareto_df = pareto_df.sort_values("epsilon")
    pareto_points = []
    best_acc = -np.inf
    for _, row in pareto_df.iterrows():
        if row["accuracy"] > best_acc:
            pareto_points.append((row["epsilon"], row["accuracy"]))
            best_acc = row["accuracy"]

    if pareto_points:
        pareto_eps, pareto_acc = zip(*pareto_points)
        ax.plot(pareto_eps, pareto_acc, "r--", linewidth=2, label="Pareto Frontier", zorder=10)
        ax.scatter(pareto_eps, pareto_acc, c="red", s=80, marker="*", zorder=11)

    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Privacy-Utility Tradeoff Space")
    ax.legend(loc="lower right", fontsize=7)
    ax.annotate("Better Privacy →", xy=(0.02, 0.02), xycoords="axes fraction",
               fontsize=8, color="#666")

    # Panel B: Pareto-optimal configurations table as heatmap
    ax = axes[1]

    # Get DP levels (exclude "none")
    privacy_labels = df_valid["epsilon_label"].dropna().unique().tolist()
    dp_levels = [p for p in privacy_labels if p != "none"]
    dp_levels.sort(key=lambda x: float(x))

    best_configs = []

    for priv in dp_levels:
        priv_df = df_valid[df_valid["epsilon_label"] == priv]
        if priv_df.empty:
            continue

        # Find best synthetic count for this privacy level
        synth_perf = priv_df.groupby("synthetic_count")["accuracy"].mean()
        best_synth = synth_perf.idxmax() if not synth_perf.empty else 0
        best_acc = synth_perf.max() if not synth_perf.empty else np.nan

        best_configs.append({
            "privacy": f"σ={priv}",
            "privacy_key": priv,
            "best_synth": int(best_synth),
            "accuracy": best_acc,
        })

    if best_configs:
        config_df = pd.DataFrame(best_configs)

        # Create bar chart instead of heatmap for better readability
        x = np.arange(len(config_df))
        bars = ax.bar(x, config_df["accuracy"], color=[NOISE_COLORS.get(row["privacy_key"], COLORS["neutral"])
                      for _, row in config_df.iterrows()], alpha=0.8)

        # Add labels on bars
        for bar, row in zip(bars, config_df.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"synth={row.best_synth}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(config_df["privacy"])
        ax.set_ylabel("Best Accuracy Achieved")
        ax.set_xlabel("Noise Multiplier (σ)")
        ax.set_title("(b) Optimal Configurations")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_pareto_frontier.pdf")
    fig.savefig(output_dir / "fig4_pareto_frontier.png")
    plt.close(fig)
    print("  Created: fig4_pareto_frontier.pdf/png")


# =============================================================================
# FIGURE 5: Scale Effects (Total N Analysis)
# =============================================================================
def plot_scale_effects(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Figure showing how dataset scale interacts with DP and augmentation.
    """
    apply_thesis_style()

    total_n_values = sorted(df["total_n"].dropna().unique())
    if len(total_n_values) < 2:
        print("  Skipping scale effects figure (only one total_n value)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Dynamic privacy order
    privacy_labels = df["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)
    dp_levels = [p for p in privacy_order if p != "none"]

    # Panel A: Accuracy vs Total N at different privacy levels
    ax = axes[0]

    for priv in privacy_order:
        priv_df = df[df["epsilon_label"] == priv]
        means = []
        ci_lows = []
        ci_highs = []
        x_vals = []

        for n in total_n_values:
            n_data = priv_df[priv_df["total_n"] == n]["accuracy"].dropna()
            if len(n_data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(n_data)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                x_vals.append(n)

        if means:
            color = NOISE_COLORS.get(priv, COLORS["neutral"])
            label = "No DP" if priv == "none" else f"σ={priv}"
            linestyle = "--" if priv == "none" else "-"
            ax.errorbar(x_vals, means,
                       yerr=[np.array(means) - np.array(ci_lows),
                             np.array(ci_highs) - np.array(means)],
                       marker="o", capsize=3, label=label, color=color,
                       linewidth=1.5, linestyle=linestyle)

    ax.set_xlabel("Total Dataset Size (N)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Scale vs Privacy Tradeoff")
    ax.legend(title="Noise (σ)", loc="lower right")

    # Panel B: DP cost (accuracy drop) at different scales
    ax = axes[1]

    dp_costs = {priv: [] for priv in dp_levels}

    for n in total_n_values:
        n_df = df[df["total_n"] == n]
        baseline_data = n_df[n_df["epsilon_label"] == "none"]["accuracy"].dropna()
        baseline = baseline_data.mean() if len(baseline_data) > 0 else np.nan

        for priv in dp_levels:
            priv_data = n_df[n_df["epsilon_label"] == priv]["accuracy"].dropna()
            if len(priv_data) > 0 and not np.isnan(baseline):
                cost = (baseline - priv_data.mean()) * 100
                dp_costs[priv].append((n, cost))

    for priv in dp_levels:
        if dp_costs[priv]:
            x_vals, costs = zip(*dp_costs[priv])
            color = NOISE_COLORS.get(priv, COLORS["neutral"])
            ax.plot(x_vals, costs, marker="o", label=f"σ={priv}", color=color, linewidth=2)

    ax.set_xlabel("Total Dataset Size (N)")
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_title("(b) DP Cost at Different Scales")
    ax.legend(title="Noise (σ)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)

    # Panel C: Augmentation benefit at different scales
    ax = axes[2]

    for n in total_n_values:
        n_df = df[df["total_n"] == n]

        # Focus on DP runs only
        dp_df = n_df[n_df["epsilon_label"].isin(dp_levels)]
        base_data = dp_df[dp_df["synthetic_count"] == 0]["accuracy"].dropna()
        base_mean = base_data.mean() if len(base_data) > 0 else np.nan

        synth_values = sorted(dp_df["synthetic_count"].dropna().unique())
        benefits = []
        synth_x = []

        for synth in synth_values:
            if synth == 0:
                continue
            synth_data = dp_df[dp_df["synthetic_count"] == synth]["accuracy"].dropna()
            if len(synth_data) > 0 and not np.isnan(base_mean):
                benefit = (synth_data.mean() - base_mean) * 100
                benefits.append(benefit)
                synth_x.append(synth)

        if benefits:
            color = plt.cm.plasma(n / max(total_n_values))
            ax.plot(synth_x, benefits, marker="o", label=f"N={int(n)}",
                   color=color, linewidth=1.5)

    ax.set_xlabel("Synthetic Data Count")
    ax.set_ylabel("Accuracy Improvement (pp)")
    ax.set_title("(c) Augmentation Benefit by Scale")
    ax.legend(title="Dataset Size", loc="best")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_scale_effects.pdf")
    fig.savefig(output_dir / "fig5_scale_effects.png")
    plt.close(fig)
    print("  Created: fig5_scale_effects.pdf/png")


# =============================================================================
# FIGURE 6: Comprehensive Multi-Factor Heatmaps
# =============================================================================
def plot_comprehensive_heatmaps(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Multi-panel heatmap showing all factor interactions.
    """
    apply_thesis_style()

    total_n_values = sorted(df["total_n"].dropna().unique())

    # Dynamic privacy order
    privacy_labels = df["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)

    # Create one figure per total_n value
    for total_n in total_n_values:
        df_n = df[df["total_n"] == total_n]
        summary_n = summary_df[summary_df["total_n"] == total_n]

        het_labels = sorted(df_n["heterogeneity"].dropna().unique(),
                           key=lambda x: df_n[df_n["heterogeneity"] == x]["het_order"].iloc[0]
                           if len(df_n[df_n["heterogeneity"] == x]) > 0 else (9, 0))

        n_hets = len(het_labels)
        if n_hets == 0:
            continue

        fig, axes = plt.subplots(1, n_hets, figsize=(4*n_hets + 1, 4), sharey=True)
        if n_hets == 1:
            axes = [axes]

        vmin, vmax = summary_n["accuracy_mean"].min(), summary_n["accuracy_mean"].max()

        for i, het in enumerate(het_labels):
            ax = axes[i]
            het_df = summary_n[summary_n["heterogeneity"] == het]

            pivot = het_df.pivot_table(
                index="synthetic_count",
                columns="epsilon_label",
                values="accuracy_mean",
                aggfunc="mean"
            )
            pivot = pivot.reindex(columns=[c for c in privacy_order if c in pivot.columns])
            # Rename columns for display
            pivot.columns = ["No DP" if c == "none" else f"σ={c}" for c in pivot.columns]

            if not pivot.empty:
                sns.heatmap(
                    pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=vmin, vmax=vmax, ax=ax,
                    cbar=(i == n_hets - 1),
                    cbar_kws={"label": "Accuracy", "shrink": 0.8} if i == n_hets - 1 else {},
                    linewidths=0.5, linecolor="white"
                )

            short_het = het.replace("(1-class/client)", "").replace("Dirichlet ", "α=")
            ax.set_title(short_het, fontsize=10)
            ax.set_xlabel("Noise (σ)")
            if i == 0:
                ax.set_ylabel("Synthetic Count")
            else:
                ax.set_ylabel("")

        fig.suptitle(f"Accuracy by Configuration (N={int(total_n)})", y=1.02, fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / f"fig6_heatmap_N{int(total_n)}.pdf")
        fig.savefig(output_dir / f"fig6_heatmap_N{int(total_n)}.png")
        plt.close(fig)
        print(f"  Created: fig6_heatmap_N{int(total_n)}.pdf/png")


# =============================================================================
# FIGURE 7: Delta Analysis (Improvement vs Baseline)
# =============================================================================
def plot_delta_analysis(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Show accuracy improvements/degradations relative to non-DP, no-synth baseline.
    """
    apply_thesis_style()

    total_n_values = sorted(df["total_n"].dropna().unique())

    # Dynamic privacy order
    privacy_labels = df["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)

    for total_n in total_n_values:
        df_n = df[df["total_n"] == total_n]
        summary_n = summary_df[summary_df["total_n"] == total_n]

        # Baseline: non-DP, synthetic_count=0
        baseline_df = summary_n[(summary_n["epsilon_label"] == "none") &
                                (summary_n["synthetic_count"] == 0)]

        # Get baseline per heterogeneity
        baseline_map = {}
        for _, row in baseline_df.iterrows():
            baseline_map[row["heterogeneity"]] = row["accuracy_mean"]

        if not baseline_map:
            continue

        # Compute delta for all configurations
        summary_n = summary_n.copy()
        summary_n["baseline"] = summary_n["heterogeneity"].map(baseline_map)
        summary_n["delta_pp"] = (summary_n["accuracy_mean"] - summary_n["baseline"]) * 100

        het_labels = sorted(summary_n["heterogeneity"].dropna().unique(),
                           key=lambda x: df_n[df_n["heterogeneity"] == x]["het_order"].iloc[0]
                           if len(df_n[df_n["heterogeneity"] == x]) > 0 else (9, 0))

        n_hets = len(het_labels)
        if n_hets == 0:
            continue

        fig, axes = plt.subplots(1, n_hets, figsize=(4*n_hets + 1, 4), sharey=True)
        if n_hets == 1:
            axes = [axes]

        # Common scale for all subplots
        vmin = summary_n["delta_pp"].min()
        vmax = summary_n["delta_pp"].max()
        vabs = max(abs(vmin), abs(vmax))

        for i, het in enumerate(het_labels):
            ax = axes[i]
            het_df = summary_n[summary_n["heterogeneity"] == het]

            pivot = het_df.pivot_table(
                index="synthetic_count",
                columns="epsilon_label",
                values="delta_pp",
                aggfunc="mean"
            )
            pivot = pivot.reindex(columns=[c for c in privacy_order if c in pivot.columns])
            # Rename columns for display
            pivot.columns = ["No DP" if c == "none" else f"σ={c}" for c in pivot.columns]

            if not pivot.empty:
                sns.heatmap(
                    pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                    center=0, vmin=-vabs, vmax=vabs, ax=ax,
                    cbar=(i == n_hets - 1),
                    cbar_kws={"label": "Δ Accuracy (pp)", "shrink": 0.8} if i == n_hets - 1 else {},
                    linewidths=0.5, linecolor="white"
                )

            short_het = het.replace("(1-class/client)", "").replace("Dirichlet ", "α=")
            ax.set_title(short_het, fontsize=10)
            ax.set_xlabel("Noise (σ)")
            if i == 0:
                ax.set_ylabel("Synthetic Count")
            else:
                ax.set_ylabel("")

        fig.suptitle(f"Accuracy Δ vs Baseline (N={int(total_n)}, baseline: no DP, no synth)",
                    y=1.02, fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / f"fig7_delta_N{int(total_n)}.pdf")
        fig.savefig(output_dir / f"fig7_delta_N{int(total_n)}.png")
        plt.close(fig)
        print(f"  Created: fig7_delta_N{int(total_n)}.pdf/png")


# =============================================================================
# FIGURE 8: Pathological vs Dirichlet Comparison
# =============================================================================
def plot_partitioning_comparison(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Direct comparison between extreme and Dirichlet partitioning.
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    df_valid = df.dropna(subset=["partitioning", "epsilon_label"]).copy()

    # Dynamic privacy order
    privacy_labels = df_valid["epsilon_label"].dropna().unique().tolist()
    privacy_order = ["none"] if "none" in privacy_labels else []
    numeric_labels = [p for p in privacy_labels if p != "none"]
    numeric_labels.sort(key=lambda x: float(x))
    privacy_order.extend(numeric_labels)
    dp_levels = [p for p in privacy_order if p != "none"]

    synth_values = sorted(df_valid["synthetic_count"].dropna().unique())

    # Panel A: Accuracy by partitioning scheme across privacy levels
    ax = axes[0]

    for part in ["extreme", "dirichlet"]:
        part_df = df_valid[df_valid["partitioning"] == part]
        means = []
        ci_lows = []
        ci_highs = []
        x_positions = []

        for j, priv in enumerate(privacy_order):
            priv_data = part_df[part_df["epsilon_label"] == priv]["accuracy"].dropna()
            if len(priv_data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(priv_data)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                x_positions.append(j)

        if means:
            color = COLORS["extreme"] if part == "extreme" else COLORS["primary"]
            marker = "s" if part == "extreme" else "o"
            linestyle = "--" if part == "extreme" else "-"
            label = "Pathological (1-class)" if part == "extreme" else "Dirichlet"
            ax.errorbar(x_positions, means,
                       yerr=[np.array(means) - np.array(ci_lows),
                             np.array(ci_highs) - np.array(means)],
                       marker=marker, capsize=3, label=label, color=color,
                       linewidth=2, markersize=7, linestyle=linestyle)

    ax.set_xticks(range(len(privacy_order)))
    xlabels = ["No DP" if p == "none" else f"σ={p}" for p in privacy_order]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Noise Multiplier (σ)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Partitioning Effect on Privacy-Utility")
    ax.legend(loc="lower left")

    # Panel B: Augmentation benefit by partitioning
    ax = axes[1]

    for part in ["extreme", "dirichlet"]:
        part_df = df_valid[df_valid["partitioning"] == part]

        # Use DP runs only (aggregated across noise levels)
        dp_df = part_df[part_df["epsilon_label"].isin(dp_levels)]
        base_data = dp_df[dp_df["synthetic_count"] == 0]["accuracy"].dropna()
        base_mean = base_data.mean() if len(base_data) > 0 else np.nan

        benefits = []
        synth_x = []
        for synth in synth_values:
            if synth == 0:
                continue
            synth_data = dp_df[dp_df["synthetic_count"] == synth]["accuracy"].dropna()
            if len(synth_data) > 0 and not np.isnan(base_mean):
                benefit = (synth_data.mean() - base_mean) * 100
                benefits.append(benefit)
                synth_x.append(synth)

        if benefits:
            color = COLORS["extreme"] if part == "extreme" else COLORS["primary"]
            marker = "s" if part == "extreme" else "o"
            linestyle = "--" if part == "extreme" else "-"
            label = "Pathological" if part == "extreme" else "Dirichlet"
            ax.plot(synth_x, benefits, marker=marker, label=label, color=color,
                   linewidth=2, markersize=7, linestyle=linestyle)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Synthetic Data Count")
    ax.set_ylabel("Accuracy Improvement (pp)")
    ax.set_title("(b) Augmentation Benefit (DP runs only)")
    ax.legend(loc="best")

    # Panel C: Box plots comparing distributions
    ax = axes[2]

    extreme_data = df_valid[df_valid["partitioning"] == "extreme"]["accuracy"].dropna()
    dirichlet_data = df_valid[df_valid["partitioning"] == "dirichlet"]["accuracy"].dropna()

    if len(extreme_data) > 0 and len(dirichlet_data) > 0:
        bp = ax.boxplot(
            [extreme_data, dirichlet_data],
            tick_labels=["Pathological\n(1-class/client)", "Dirichlet\n(α-controlled)"],
            patch_artist=True, widths=0.5,
            showfliers=True, flierprops={"marker": "o", "markersize": 4, "alpha": 0.5}
        )
        bp["boxes"][0].set_facecolor(COLORS["extreme"])
        bp["boxes"][1].set_facecolor(COLORS["primary"])
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # Statistical test
        if len(extreme_data) > 1 and len(dirichlet_data) > 1:
            stat, pval = stats.mannwhitneyu(extreme_data, dirichlet_data, alternative="two-sided")
            sig_text = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
            if pval < 0.001:
                sig_text += " ***"
            elif pval < 0.01:
                sig_text += " **"
            elif pval < 0.05:
                sig_text += " *"
            y_max = max(extreme_data.max(), dirichlet_data.max())
            ax.annotate(sig_text, xy=(1.5, y_max + 0.02), ha="center", fontsize=9)

        # Individual points
        for i, data in enumerate([extreme_data, dirichlet_data], 1):
            x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.3, s=12, color="black", zorder=3)

    ax.set_ylabel("Test Accuracy")
    ax.set_title("(c) Overall Distribution")

    plt.tight_layout()
    fig.savefig(output_dir / "fig8_partitioning_comparison.pdf")
    fig.savefig(output_dir / "fig8_partitioning_comparison.png")
    plt.close(fig)
    print("  Created: fig8_partitioning_comparison.pdf/png")


# =============================================================================
# Statistical Analysis and Report
# =============================================================================
def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """Perform comprehensive statistical tests."""
    results = {}
    df_valid = df.dropna(subset=["accuracy"])

    # Test 1: Effect of DP (Kruskal-Wallis across epsilon levels)
    eps_groups = [g["accuracy"].dropna().values
                  for _, g in df_valid.groupby("epsilon_label") if len(g) > 1]
    if len(eps_groups) >= 2:
        stat, pval = stats.kruskal(*eps_groups)
        results["epsilon_effect"] = {
            "test": "Kruskal-Wallis H",
            "statistic": stat,
            "p_value": pval,
        }

    # Test 2: Effect of synthetic augmentation
    synth_groups = [g["accuracy"].dropna().values
                    for _, g in df_valid.groupby("synthetic_count") if len(g) > 1]
    if len(synth_groups) >= 2:
        stat, pval = stats.kruskal(*synth_groups)
        results["synth_effect"] = {
            "test": "Kruskal-Wallis H",
            "statistic": stat,
            "p_value": pval,
        }

    # Test 3: DP vs non-DP
    no_dp = df_valid[df_valid["epsilon_label"] == "none"]["accuracy"].dropna()
    with_dp = df_valid[df_valid["epsilon_label"].isin(["1", "3", "8"])]["accuracy"].dropna()
    if len(no_dp) > 1 and len(with_dp) > 1:
        stat, pval = stats.mannwhitneyu(no_dp, with_dp, alternative="two-sided")
        results["dp_vs_no_dp"] = {
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": pval,
            "no_dp_mean": no_dp.mean(),
            "with_dp_mean": with_dp.mean(),
            "difference": no_dp.mean() - with_dp.mean(),
        }

    # Test 4: Synth vs no synth (within DP runs)
    dp_df = df_valid[df_valid["epsilon_label"].isin(["1", "3", "8"])]
    no_synth = dp_df[dp_df["synthetic_count"] == 0]["accuracy"].dropna()
    with_synth = dp_df[dp_df["synthetic_count"] > 0]["accuracy"].dropna()
    if len(no_synth) > 1 and len(with_synth) > 1:
        stat, pval = stats.mannwhitneyu(no_synth, with_synth, alternative="two-sided")
        results["synth_benefit_under_dp"] = {
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": pval,
            "no_synth_mean": no_synth.mean(),
            "with_synth_mean": with_synth.mean(),
            "improvement": with_synth.mean() - no_synth.mean(),
        }

    # Test 5: Partitioning effect
    if "partitioning" in df_valid.columns:
        extreme = df_valid[df_valid["partitioning"] == "extreme"]["accuracy"].dropna()
        dirichlet = df_valid[df_valid["partitioning"] == "dirichlet"]["accuracy"].dropna()
        if len(extreme) > 1 and len(dirichlet) > 1:
            stat, pval = stats.mannwhitneyu(extreme, dirichlet, alternative="two-sided")
            results["partitioning_effect"] = {
                "test": "Mann-Whitney U",
                "statistic": stat,
                "p_value": pval,
                "extreme_mean": extreme.mean(),
                "dirichlet_mean": dirichlet.mean(),
                "difference": dirichlet.mean() - extreme.mean(),
            }

    # Test 6: Interaction - Does synth help more under strict privacy?
    # Get DP levels dynamically
    privacy_labels = df_valid["epsilon_label"].dropna().unique().tolist()
    dp_levels = [p for p in privacy_labels if p != "none"]

    for priv in dp_levels:
        priv_df = df_valid[df_valid["epsilon_label"] == priv]
        no_synth = priv_df[priv_df["synthetic_count"] == 0]["accuracy"].dropna()
        with_synth = priv_df[priv_df["synthetic_count"] > 0]["accuracy"].dropna()
        if len(no_synth) > 1 and len(with_synth) > 1:
            stat, pval = stats.mannwhitneyu(no_synth, with_synth, alternative="two-sided")
            results[f"synth_benefit_noise_{priv}"] = {
                "test": "Mann-Whitney U",
                "statistic": stat,
                "p_value": pval,
                "no_synth_mean": no_synth.mean(),
                "with_synth_mean": with_synth.mean(),
                "improvement": with_synth.mean() - no_synth.mean(),
                "noise_multiplier": priv,
            }

    return results


def write_comprehensive_report(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive academic report."""
    report_path = output_dir / "statistical_analysis_report.txt"
    stat_results = perform_statistical_tests(df)
    df_valid = df.dropna(subset=["accuracy"])

    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("DP-FEDAUG MNIST STUDY: COMPREHENSIVE STATISTICAL ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # 1. Dataset Overview
        f.write("1. EXPERIMENTAL OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total experimental runs: {len(df_valid)}\n")
        f.write(f"Completed runs: {len(df_valid[df_valid['state'] == 'finished'])}\n")
        f.write(f"Unique seeds: {df_valid['seed'].nunique()}\n\n")

        f.write("Experimental Factors:\n")
        f.write(f"  Noise multipliers (σ): {sorted(df_valid['epsilon_label'].unique())}\n")
        f.write(f"  Synthetic counts: {sorted(df_valid['synthetic_count'].dropna().unique())}\n")
        f.write(f"  Dataset sizes (N): {sorted(df_valid['total_n'].dropna().unique())}\n")
        f.write(f"  Partitioning schemes: {df_valid['partitioning'].unique().tolist()}\n")
        if "alpha" in df_valid.columns:
            alphas = df_valid["alpha"].dropna().unique()
            alpha_strs = ["IID" if (isinstance(a, float) and math.isinf(a)) else str(a) for a in alphas]
            f.write(f"  Dirichlet α values: {sorted(set(alpha_strs))}\n")
        f.write("\n")

        # 2. Key Research Questions
        f.write("2. KEY STATISTICAL FINDINGS\n")
        f.write("-" * 40 + "\n\n")

        if "epsilon_effect" in stat_results:
            res = stat_results["epsilon_effect"]
            f.write("RQ1: Does noise multiplier (σ) significantly affect accuracy?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  H-statistic: {res['statistic']:.4f}\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'} (α=0.05)\n\n")

        if "dp_vs_no_dp" in stat_results:
            res = stat_results["dp_vs_no_dp"]
            f.write("RQ2: Does DP significantly reduce accuracy compared to non-DP?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  Non-DP mean: {res['no_dp_mean']:.4f}\n")
            f.write(f"  With-DP mean: {res['with_dp_mean']:.4f}\n")
            f.write(f"  Accuracy drop: {res['difference']:.4f} ({res['difference']*100:.2f} pp)\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'}\n\n")

        if "synth_effect" in stat_results:
            res = stat_results["synth_effect"]
            f.write("RQ3: Does synthetic augmentation significantly affect accuracy?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  H-statistic: {res['statistic']:.4f}\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'}\n\n")

        if "synth_benefit_under_dp" in stat_results:
            res = stat_results["synth_benefit_under_dp"]
            f.write("RQ4: Does augmentation help recover accuracy lost to DP?\n")
            f.write(f"  Test: {res['test']} (DP runs only)\n")
            f.write(f"  Without synth: {res['no_synth_mean']:.4f}\n")
            f.write(f"  With synth: {res['with_synth_mean']:.4f}\n")
            f.write(f"  Improvement: {res['improvement']:.4f} ({res['improvement']*100:.2f} pp)\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'}\n\n")

        # Synth benefit at each noise level
        f.write("RQ5: Does augmentation benefit vary with noise level?\n")
        for key, res in stat_results.items():
            if key.startswith("synth_benefit_noise_"):
                f.write(f"  σ={res['noise_multiplier']}: improvement = {res['improvement']*100:.2f} pp ")
                f.write(f"(p = {res['p_value']:.4e})\n")
        f.write("\n")

        if "partitioning_effect" in stat_results:
            res = stat_results["partitioning_effect"]
            f.write("RQ6: Does partitioning scheme affect performance?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  Pathological mean: {res['extreme_mean']:.4f}\n")
            f.write(f"  Dirichlet mean: {res['dirichlet_mean']:.4f}\n")
            f.write(f"  Difference: {res['difference']:.4f} ({res['difference']*100:.2f} pp)\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'}\n\n")

        # 3. Descriptive Statistics
        f.write("3. DESCRIPTIVE STATISTICS\n")
        f.write("-" * 40 + "\n\n")

        f.write("Accuracy by Noise Multiplier (σ):\n")
        eps_stats = df_valid.groupby("epsilon_label")["accuracy"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(4)
        f.write(eps_stats.to_string() + "\n\n")

        f.write("Accuracy by Synthetic Count:\n")
        synth_stats = df_valid.groupby("synthetic_count")["accuracy"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(4)
        f.write(synth_stats.to_string() + "\n\n")

        if "total_n" in df_valid.columns:
            f.write("Accuracy by Dataset Size:\n")
            n_stats = df_valid.groupby("total_n")["accuracy"].agg(
                ["count", "mean", "std", "min", "max"]
            ).round(4)
            f.write(n_stats.to_string() + "\n\n")

        if "heterogeneity" in df_valid.columns:
            f.write("Accuracy by Heterogeneity Level:\n")
            het_stats = df_valid.groupby("heterogeneity")["accuracy"].agg(
                ["count", "mean", "std", "min", "max"]
            ).round(4)
            f.write(het_stats.to_string() + "\n\n")

        # 4. Best Configurations
        f.write("4. OPTIMAL CONFIGURATIONS\n")
        f.write("-" * 40 + "\n\n")

        best_overall = summary_df.sort_values("accuracy_mean", ascending=False).head(10)
        f.write("Top 10 configurations by mean accuracy:\n")
        for _, row in best_overall.iterrows():
            priv_str = "No DP" if row["epsilon_label"] == "none" else f"σ={row['epsilon_label']}"
            f.write(f"  {priv_str}, synth={int(row['synthetic_count'])}, ")
            f.write(f"N={int(row['total_n']) if pd.notna(row['total_n']) else '?'}: ")
            f.write(f"{row['accuracy_mean']:.4f} (±{row['accuracy_std']:.4f})\n")
        f.write("\n")

        # Best for each privacy level
        f.write("Best configuration per noise level:\n")
        # Get all privacy levels dynamically
        privacy_labels = df_valid["epsilon_label"].dropna().unique().tolist()
        privacy_order = ["none"] if "none" in privacy_labels else []
        numeric_labels = [p for p in privacy_labels if p != "none"]
        numeric_labels.sort(key=lambda x: float(x))
        privacy_order.extend(numeric_labels)

        for priv in privacy_order:
            priv_df = summary_df[summary_df["epsilon_label"] == priv]
            if not priv_df.empty:
                best = priv_df.loc[priv_df["accuracy_mean"].idxmax()]
                priv_str = "No DP" if priv == "none" else f"σ={priv}"
                f.write(f"  {priv_str}: synth={int(best['synthetic_count'])}, ")
                f.write(f"acc={best['accuracy_mean']:.4f}\n")
        f.write("\n")

        # 5. Effect Sizes
        f.write("5. EFFECT SIZES\n")
        f.write("-" * 40 + "\n\n")

        # Cohen's d for DP vs no-DP
        no_dp = df_valid[df_valid["epsilon_label"] == "none"]["accuracy"].dropna()
        with_dp = df_valid[df_valid["epsilon_label"].isin(["1", "3", "8"])]["accuracy"].dropna()
        if len(no_dp) > 1 and len(with_dp) > 1:
            pooled_std = np.sqrt(((len(no_dp)-1)*no_dp.std()**2 +
                                  (len(with_dp)-1)*with_dp.std()**2) /
                                 (len(no_dp) + len(with_dp) - 2))
            if pooled_std > 0:
                cohens_d = (no_dp.mean() - with_dp.mean()) / pooled_std
                f.write(f"Cohen's d (DP effect): {cohens_d:.3f}\n")
                if abs(cohens_d) < 0.2:
                    interp = "negligible"
                elif abs(cohens_d) < 0.5:
                    interp = "small"
                elif abs(cohens_d) < 0.8:
                    interp = "medium"
                else:
                    interp = "large"
                f.write(f"  Interpretation: {interp} effect\n\n")

        # 6. LaTeX Tables
        f.write("6. LATEX TABLES FOR PUBLICATION\n")
        f.write("-" * 40 + "\n\n")

        f.write("% Table 1: Main Results\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{DP-FedAug MNIST: Accuracy by Noise Multiplier and Augmentation}\n")

        # Build header dynamically
        header_cols = ["Synthetic", "No DP"]
        for priv in numeric_labels:
            header_cols.append(f"$\\sigma={priv}$")
        f.write(f"\\begin{{tabular}}{{l{'c' * len(header_cols)}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header_cols) + " \\\\\n")
        f.write("\\midrule\n")

        for synth in sorted(df_valid["synthetic_count"].dropna().unique()):
            synth_df = summary_df[summary_df["synthetic_count"] == synth]
            row_vals = [f"{int(synth)}"]
            for priv in privacy_order:
                priv_df = synth_df[synth_df["epsilon_label"] == priv]
                if not priv_df.empty:
                    mean = priv_df["accuracy_mean"].mean()
                    std = priv_df["accuracy_std"].mean()
                    row_vals.append(f"{mean:.3f} $\\pm$ {std:.3f}")
                else:
                    row_vals.append("-")
            f.write(" & ".join(row_vals) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    # Save summary CSV
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    print("  Created: statistical_analysis_report.txt, summary_statistics.csv")


# =============================================================================
# Main Entry Point
# =============================================================================
def load_study_data(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 120,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = output_dir / "dpfedaug_study_raw.csv"
    df = load_or_fetch_dataframe(
        raw_path,
        refresh=refresh,
        fetch_fn=lambda: fetch_wandb_runs(PROJECT_NAME, entity=entity, timeout=timeout),
        normalize_fn=normalize_dataframe,
    )
    df.to_csv(raw_path, index=False)
    df = df[df["state"] == "finished"].copy()
    summary_df = summarize(df)
    summary_df.to_csv(output_dir / "dpfedaug_study_summary.csv", index=False)
    return df, summary_df


def export_study(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 120,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, summary_df = load_study_data(
        refresh=refresh,
        entity=entity,
        timeout=timeout,
        output_dir=output_dir,
    )

    if df.empty:
        print("No finished runs found.")
        return None

    print(f"\n{'='*60}")
    print("DP-FEDAUG MNIST STUDY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total finished runs: {len(df)}")
    print(f"Unique configurations: {len(summary_df)}")
    print(f"Noise multipliers (σ): {sorted(df['epsilon_label'].unique())}")
    print(f"Synthetic counts: {sorted(df['synthetic_count'].dropna().unique())}")
    print(f"Dataset sizes: {sorted(df['total_n'].dropna().unique())}")
    print(f"Partitioning schemes: {df['partitioning'].dropna().unique().tolist()}")
    print(f"{'='*60}\n")

    print("Generating publication-quality figures...")
    plot_main_results_figure(df, summary_df, output_dir)
    plot_heterogeneity_impact(df, summary_df, output_dir)
    plot_augmentation_recovery(df, summary_df, output_dir)
    plot_pareto_frontier(df, summary_df, output_dir)
    plot_scale_effects(df, summary_df, output_dir)
    plot_comprehensive_heatmaps(df, summary_df, output_dir)
    plot_delta_analysis(df, summary_df, output_dir)
    plot_partitioning_comparison(df, summary_df, output_dir)
    write_comprehensive_report(df, summary_df, output_dir)

    print(f"\n{'='*60}")
    print(f"OUTPUT SAVED TO: {output_dir}")
    print(f"{'='*60}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for DP-FedAug MNIST study."
    )
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/org.")
    parser.add_argument("--timeout", type=int, default=120, help="W&B API timeout.")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from W&B.")
    args = parser.parse_args()

    export_study(refresh=args.refresh, entity=args.entity, timeout=args.timeout)


if __name__ == "__main__":
    main()
