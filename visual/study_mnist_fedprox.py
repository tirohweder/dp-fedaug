import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Use shared thesis style
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual.data import (
    fetch_fedprox_runs,
    get_cfg_value,
    load_or_fetch_dataframe,
    normalize_fedprox_dataframe,
    summarize_fedprox_dataframe,
    to_float,
)
from visual.thesis_style import (
    apply_thesis_style,
    COLORS,
    MU_COLORS,
)


PROJECT_NAME = "FedProx-MNIST-Baseline"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "mnist_fedprox"


def _to_float(value):
    return to_float(value, zero_is_infinity=False)


def _get_cfg_value(cfg: dict, *keys):
    return get_cfg_value(cfg, *keys)


def fetch_wandb_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    return fetch_fedprox_runs(project, entity=entity, timeout=timeout)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_fedprox_dataframe(df)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_fedprox_dataframe(df)


def get_heterogeneity_label(partitioning: str, alpha: float) -> str:
    """Create a meaningful label for data heterogeneity based on partitioning scheme."""
    if partitioning == "extreme":
        return "Pathological (1-class/client)"
    elif isinstance(alpha, float) and math.isinf(alpha):
        return "IID"
    else:
        return f"Dirichlet α={alpha}"


def get_heterogeneity_order(partitioning: str, alpha: float) -> tuple:
    """Return a sort key: (partitioning_order, alpha_order) for consistent ordering.

    Order: Pathological (most heterogeneous) -> Dirichlet low α -> Dirichlet high α -> IID
    """
    if partitioning == "extreme":
        return (0, 0)  # Most heterogeneous first
    elif isinstance(alpha, float) and math.isinf(alpha):
        return (2, float('inf'))  # IID last
    else:
        return (1, alpha)  # Dirichlet in middle, sorted by alpha



def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for the mean."""
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """Perform key statistical tests for the thesis."""
    results = {}

    df_valid = df.dropna(subset=["partitioning", "alpha"])

    # Test 1: Effect of proximal_mu (Kruskal-Wallis)
    mu_groups = [g["accuracy"].dropna().values for _, g in df_valid.groupby("proximal_mu") if len(g) > 1]
    if len(mu_groups) >= 2 and all(len(g) > 0 for g in mu_groups):
        stat, pval = stats.kruskal(*mu_groups)
        results["mu_effect"] = {
            "test": "Kruskal-Wallis H",
            "statistic": stat,
            "p_value": pval,
        }

    # Test 2: FedProx vs FedAvg (mu=0 vs mu>0)
    fedavg = df_valid[df_valid["proximal_mu"] == 0]["accuracy"].dropna()
    fedprox = df_valid[df_valid["proximal_mu"] > 0]["accuracy"].dropna()
    if len(fedavg) > 1 and len(fedprox) > 1:
        stat, pval = stats.mannwhitneyu(fedavg, fedprox, alternative="two-sided")
        results["fedprox_vs_fedavg"] = {
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": pval,
            "fedavg_mean": fedavg.mean(),
            "fedprox_mean": fedprox.mean(),
            "difference": fedprox.mean() - fedavg.mean(),
        }

    # Test 3: Effect of partitioning scheme (Pathological vs Dirichlet)
    extreme_data = df_valid[df_valid["partitioning"] == "extreme"]["accuracy"].dropna()
    dirichlet_data = df_valid[df_valid["partitioning"] == "dirichlet"]["accuracy"].dropna()
    if len(extreme_data) > 1 and len(dirichlet_data) > 1:
        stat, pval = stats.mannwhitneyu(extreme_data, dirichlet_data, alternative="two-sided")
        results["partitioning_effect"] = {
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": pval,
            "extreme_mean": extreme_data.mean(),
            "dirichlet_mean": dirichlet_data.mean(),
            "difference": dirichlet_data.mean() - extreme_data.mean(),
        }

    # Test 4: Effect of alpha within Dirichlet partitioning only
    dirichlet_df = df_valid[df_valid["partitioning"] == "dirichlet"]
    alpha_groups = [g["accuracy"].dropna().values for _, g in dirichlet_df.groupby("alpha") if len(g) > 1]
    if len(alpha_groups) >= 2 and all(len(g) > 0 for g in alpha_groups):
        stat, pval = stats.kruskal(*alpha_groups)
        results["alpha_effect_dirichlet"] = {
            "test": "Kruskal-Wallis H",
            "statistic": stat,
            "p_value": pval,
            "note": "Effect of α within Dirichlet partitioning only",
        }

    # Test 5: Does FedProx help under extreme partitioning?
    extreme_fedavg = df_valid[(df_valid["partitioning"] == "extreme") & (df_valid["proximal_mu"] == 0)]["accuracy"].dropna()
    extreme_fedprox = df_valid[(df_valid["partitioning"] == "extreme") & (df_valid["proximal_mu"] > 0)]["accuracy"].dropna()
    if len(extreme_fedavg) > 1 and len(extreme_fedprox) > 1:
        stat, pval = stats.mannwhitneyu(extreme_fedavg, extreme_fedprox, alternative="two-sided")
        results["fedprox_extreme"] = {
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": pval,
            "fedavg_mean": extreme_fedavg.mean(),
            "fedprox_mean": extreme_fedprox.mean(),
            "improvement": extreme_fedprox.mean() - extreme_fedavg.mean(),
            "note": "FedProx benefit under extreme (1-class/client) partitioning",
        }

    # Test 6: FedProx benefit comparison: Pathological vs Dirichlet
    dirichlet_fedavg = df_valid[(df_valid["partitioning"] == "dirichlet") & (df_valid["proximal_mu"] == 0)]["accuracy"].dropna()
    dirichlet_fedprox = df_valid[(df_valid["partitioning"] == "dirichlet") & (df_valid["proximal_mu"] > 0)]["accuracy"].dropna()
    if all(len(x) > 0 for x in [extreme_fedavg, extreme_fedprox, dirichlet_fedavg, dirichlet_fedprox]):
        extreme_improvement = extreme_fedprox.mean() - extreme_fedavg.mean()
        dirichlet_improvement = dirichlet_fedprox.mean() - dirichlet_fedavg.mean()
        results["partitioning_benefit_comparison"] = {
            "extreme_improvement": extreme_improvement,
            "dirichlet_improvement": dirichlet_improvement,
            "fedprox_helps_more_extreme": extreme_improvement > dirichlet_improvement,
            "note": "Comparison of FedProx benefit by partitioning scheme",
        }

    return results


def plot_main_results_figure(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Main thesis figure: FedProx performance across different settings.
    3-panel figure showing key results with partitioning-aware labeling.
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    mu_values = sorted(df["proximal_mu"].dropna().unique())

    # Build heterogeneity groups: (partitioning, alpha) pairs
    df_valid = df.dropna(subset=["partitioning", "alpha"])
    het_groups = df_valid.groupby(["partitioning", "alpha"]).size().reset_index()[["partitioning", "alpha"]]
    # Sort by heterogeneity order
    het_groups["sort_key"] = het_groups.apply(
        lambda r: get_heterogeneity_order(r["partitioning"], r["alpha"]), axis=1
    )
    het_groups = het_groups.sort_values("sort_key").reset_index(drop=True)

    # Panel A: Accuracy vs μ across different heterogeneity levels
    ax = axes[0]
    for i, (_, row) in enumerate(het_groups.iterrows()):
        part, alpha = row["partitioning"], row["alpha"]
        subset = df_valid[(df_valid["partitioning"] == part) & (df_valid["alpha"] == alpha)]
        means = []
        ci_lows = []
        ci_highs = []
        x_vals = []

        for mu in mu_values:
            mu_data = subset[subset["proximal_mu"] == mu]["accuracy"].dropna()
            if len(mu_data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(mu_data)
                means.append(mean)
                ci_lows.append(ci_low)
                ci_highs.append(ci_high)
                x_vals.append(mu)

        if means:
            label = get_heterogeneity_label(part, alpha)
            color = MU_COLORS[i % len(MU_COLORS)]
            linestyle = "--" if part == "extreme" else "-"
            marker = "s" if part == "extreme" else "o"
            ax.errorbar(
                x_vals, means,
                yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
                marker=marker, capsize=3, capthick=1, label=label, color=color,
                linewidth=1.5, markersize=5, linestyle=linestyle
            )

    ax.set_xlabel("Proximal Term (μ)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Effect of Proximal Term")
    ax.legend(title="Data Distribution", loc="best", fontsize=7)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Panel B: Box plot by partitioning scheme (Dirichlet vs Pathological)
    ax = axes[1]
    dirichlet_data = df_valid[df_valid["partitioning"] == "dirichlet"]["accuracy"].dropna()
    extreme_data = df_valid[df_valid["partitioning"] == "extreme"]["accuracy"].dropna()

    if len(dirichlet_data) > 0 and len(extreme_data) > 0:
        bp = ax.boxplot(
            [extreme_data, dirichlet_data],
            tick_labels=["Pathological\n(1-class/client)", "Dirichlet\n(α-controlled)"],
            patch_artist=True, widths=0.5,
            showfliers=True, flierprops={"marker": "o", "markersize": 4, "alpha": 0.5}
        )
        bp["boxes"][0].set_facecolor(COLORS["quaternary"])
        bp["boxes"][1].set_facecolor(COLORS["primary"])
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # Add statistical annotation
        if len(extreme_data) > 1 and len(dirichlet_data) > 1:
            stat, pval = stats.mannwhitneyu(extreme_data, dirichlet_data, alternative="two-sided")
            sig_text = f"p = {pval:.3f}" if pval >= 0.001 else "p < 0.001"
            if pval < 0.05:
                sig_text += " *"
            if pval < 0.01:
                sig_text = sig_text[:-1] + "**"
            if pval < 0.001:
                sig_text = sig_text[:-2] + "***"
            y_max = max(extreme_data.max(), dirichlet_data.max())
            ax.annotate(sig_text, xy=(1.5, y_max + 0.02), ha="center", fontsize=9)

        # Add individual points
        for i, data in enumerate([extreme_data, dirichlet_data], 1):
            x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=15, color="black", zorder=3)

    ax.set_ylabel("Test Accuracy")
    ax.set_title("(b) Partitioning Scheme Impact")

    # Panel C: Heatmap with partitioning-aware labels
    ax = axes[2]
    # Build pivot with heterogeneity labels
    summary_valid = summary_df.dropna(subset=["partitioning", "alpha"])
    summary_valid = summary_valid.copy()
    summary_valid["het_label"] = summary_valid.apply(
        lambda r: get_heterogeneity_label(r["partitioning"], r["alpha"]), axis=1
    )
    summary_valid["sort_key"] = summary_valid.apply(
        lambda r: get_heterogeneity_order(r["partitioning"], r["alpha"]), axis=1
    )

    pivot_data = summary_valid.groupby(["het_label", "sort_key", "proximal_mu"]).agg(
        accuracy=("accuracy_mean", "mean")
    ).reset_index()

    pivot = pivot_data.pivot(index=["sort_key", "het_label"], columns="proximal_mu", values="accuracy")
    pivot = pivot.sort_index(ascending=True)
    # Drop sort_key from index for display
    pivot.index = pivot.index.droplevel(0)

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=pivot.values.mean(),
        ax=ax, cbar_kws={"label": "Accuracy", "shrink": 0.8},
        linewidths=0.5, linecolor="white"
    )
    ax.set_xlabel("Proximal Term (μ)")
    ax.set_ylabel("Data Distribution")
    ax.set_title("(c) Accuracy by Configuration")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_main_results.pdf")
    fig.savefig(output_dir / "fig_main_results.png")
    plt.close(fig)


def plot_heterogeneity_benefit(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Figure showing how FedProx benefit varies with data heterogeneity.
    Key research question: Does FedProx help more under non-IID conditions?
    Separates extreme partitioning from Dirichlet partitioning.
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    df_valid = df.dropna(subset=["partitioning", "alpha"])
    mu_values = sorted([m for m in df_valid["proximal_mu"].dropna().unique() if m > 0])

    # Build heterogeneity groups
    het_groups = df_valid.groupby(["partitioning", "alpha"]).size().reset_index()[["partitioning", "alpha"]]
    het_groups["sort_key"] = het_groups.apply(
        lambda r: get_heterogeneity_order(r["partitioning"], r["alpha"]), axis=1
    )
    het_groups = het_groups.sort_values("sort_key").reset_index(drop=True)
    het_groups["label"] = het_groups.apply(
        lambda r: get_heterogeneity_label(r["partitioning"], r["alpha"]), axis=1
    )

    # Panel A: Improvement over FedAvg at each heterogeneity level
    ax = axes[0]

    improvements = []
    for _, row in het_groups.iterrows():
        part, alpha, label = row["partitioning"], row["alpha"], row["label"]
        subset = df_valid[(df_valid["partitioning"] == part) & (df_valid["alpha"] == alpha)]
        fedavg = subset[subset["proximal_mu"] == 0]["accuracy"].dropna()
        if fedavg.empty:
            continue
        fedavg_mean = fedavg.mean()

        for mu in mu_values:
            fedprox = subset[subset["proximal_mu"] == mu]["accuracy"].dropna()
            if not fedprox.empty:
                improvement = (fedprox.mean() - fedavg_mean) * 100
                # Bootstrap CI
                if len(fedavg) > 1 and len(fedprox) > 1:
                    boot_diffs = []
                    for _ in range(1000):
                        fa_sample = np.random.choice(fedavg, size=len(fedavg), replace=True)
                        fp_sample = np.random.choice(fedprox, size=len(fedprox), replace=True)
                        boot_diffs.append((fp_sample.mean() - fa_sample.mean()) * 100)
                    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
                else:
                    ci_low, ci_high = improvement, improvement

                improvements.append({
                    "partitioning": part,
                    "alpha": alpha,
                    "label": label,
                    "sort_key": row["sort_key"],
                    "mu": mu,
                    "improvement": improvement,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

    if improvements:
        imp_df = pd.DataFrame(improvements)
        imp_df = imp_df.sort_values("sort_key")
        labels_order = imp_df.drop_duplicates("label")["label"].tolist()

        for i, mu in enumerate(mu_values):
            mu_df = imp_df[imp_df["mu"] == mu].sort_values("sort_key")
            if mu_df.empty:
                continue
            color = MU_COLORS[i % len(MU_COLORS)]
            # Use different marker for extreme
            markers = ["s" if p == "extreme" else "o" for p in mu_df["partitioning"]]
            ax.errorbar(
                range(len(mu_df)), mu_df["improvement"],
                yerr=[mu_df["improvement"] - mu_df["ci_low"], mu_df["ci_high"] - mu_df["improvement"]],
                marker="o", capsize=3, label=f"μ={mu}", color=color, linewidth=1.5
            )
            # Mark extreme points with squares
            extreme_idx = mu_df[mu_df["partitioning"] == "extreme"].index
            for idx in extreme_idx:
                pos = mu_df.index.get_loc(idx)
                ax.scatter([pos], [mu_df.loc[idx, "improvement"]], marker="s", s=60,
                          color=color, edgecolors="black", linewidth=1, zorder=5)

        ax.set_xticks(range(len(labels_order)))
        ax.set_xticklabels(labels_order, rotation=15, ha="right", fontsize=8)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("Data Distribution")
        ax.set_ylabel("Improvement over FedAvg (pp)")
        ax.set_title("(a) FedProx Benefit by Heterogeneity")
        ax.legend(title="Proximal μ", loc="best", fontsize=8)
        ax.text(0.02, 0.98, "← More heterogeneous",
               transform=ax.transAxes, fontsize=8, alpha=0.6, ha="left", va="top")

    # Panel B: Violin plot comparing partitioning schemes
    ax = axes[1]
    extreme_fedprox = df_valid[(df_valid["partitioning"] == "extreme") & (df_valid["proximal_mu"] > 0)]["accuracy"].dropna()
    dirichlet_low = df_valid[(df_valid["partitioning"] == "dirichlet") & (df_valid["alpha"] <= 0.5) & (df_valid["proximal_mu"] > 0)]["accuracy"].dropna()
    dirichlet_high = df_valid[(df_valid["partitioning"] == "dirichlet") & (df_valid["alpha"] > 0.5) & (df_valid["proximal_mu"] > 0)]["accuracy"].dropna()

    plot_data = []
    plot_labels = []
    plot_colors = []

    if len(extreme_fedprox) > 0:
        plot_data.append(extreme_fedprox)
        plot_labels.append("Pathological\n(1-class)")
        plot_colors.append(COLORS["quaternary"])
    if len(dirichlet_low) > 0:
        plot_data.append(dirichlet_low)
        plot_labels.append("Dirichlet\n(α≤0.5)")
        plot_colors.append(COLORS["tertiary"])
    if len(dirichlet_high) > 0:
        plot_data.append(dirichlet_high)
        plot_labels.append("Dirichlet\n(α>0.5)")
        plot_colors.append(COLORS["success"])

    if len(plot_data) >= 2:
        positions = list(range(len(plot_data)))
        parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels)

        # Add points
        for i, data in enumerate(plot_data):
            x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.4, s=15, color="black", zorder=3)

        ax.set_ylabel("Test Accuracy (FedProx)")
        ax.set_title("(b) Accuracy by Partitioning Scheme")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_heterogeneity_benefit.pdf")
    fig.savefig(output_dir / "fig_heterogeneity_benefit.png")
    plt.close(fig)


def plot_mu_sensitivity(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Figure analyzing sensitivity to the proximal term μ.
    """
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    mu_values = sorted(df["proximal_mu"].dropna().unique())

    # Aggregate across all settings
    means = []
    ci_lows = []
    ci_highs = []

    for mu in mu_values:
        mu_data = df[df["proximal_mu"] == mu]["accuracy"].dropna()
        if len(mu_data) >= 2:
            mean, ci_low, ci_high = compute_confidence_interval(mu_data)
            means.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
        else:
            means.append(mu_data.mean() if len(mu_data) > 0 else np.nan)
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)

    valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
    x = [mu_values[i] for i in valid_idx]
    y = [means[i] for i in valid_idx]
    yerr_low = [y[j] - ci_lows[valid_idx[j]] for j in range(len(valid_idx))]
    yerr_high = [ci_highs[valid_idx[j]] - y[j] for j in range(len(valid_idx))]

    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="s", capsize=4,
               color=COLORS["primary"], linewidth=2, markersize=8, capthick=1.5)

    # Highlight FedAvg baseline
    if 0 in mu_values:
        fedavg_idx = mu_values.index(0)
        if fedavg_idx in valid_idx:
            ax.axhline(y=means[fedavg_idx], color=COLORS["neutral"], linestyle="--",
                      linewidth=1, alpha=0.7, label="FedAvg baseline")

    # Find optimal mu
    if valid_idx:
        best_idx = np.argmax(y)
        best_mu = x[best_idx]
        best_acc = y[best_idx]
        ax.scatter([best_mu], [best_acc], s=150, facecolors="none", edgecolors=COLORS["success"],
                  linewidth=2, zorder=5, label=f"Best: μ={best_mu}")

    ax.set_xlabel("Proximal Term (μ)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Sensitivity to Proximal Term μ\n(aggregated across all settings)")
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_mu_sensitivity.pdf")
    fig.savefig(output_dir / "fig_mu_sensitivity.png")
    plt.close(fig)


def plot_comprehensive_heatmap(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Comprehensive heatmap showing all experimental results.
    Separates extreme from Dirichlet partitioning.
    """
    apply_thesis_style()

    summary_valid = summary_df.dropna(subset=["partitioning", "alpha"])
    mu_values = sorted(summary_valid["proximal_mu"].dropna().unique())

    # Build heterogeneity groups
    summary_valid = summary_valid.copy()
    summary_valid["het_label"] = summary_valid.apply(
        lambda r: get_heterogeneity_label(r["partitioning"], r["alpha"]), axis=1
    )
    summary_valid["sort_key"] = summary_valid.apply(
        lambda r: get_heterogeneity_order(r["partitioning"], r["alpha"]), axis=1
    )

    # Build pivot data
    pivot_data = []
    for _, row in summary_valid.iterrows():
        pivot_data.append({
            "het_label": row["het_label"],
            "sort_key": row["sort_key"],
            "mu": row["proximal_mu"],
            "accuracy": row["accuracy_mean"],
            "std": row["accuracy_std"],
            "n": row["runs"],
            "partitioning": row["partitioning"],
        })

    pivot_df = pd.DataFrame(pivot_data)

    # Aggregate if multiple rows per (het_label, mu)
    pivot_agg = pivot_df.groupby(["het_label", "sort_key", "mu", "partitioning"]).agg(
        accuracy=("accuracy", "mean"),
        std=("std", "mean"),
        n=("n", "sum"),
    ).reset_index()

    pivot = pivot_agg.pivot(index=["sort_key", "het_label"], columns="mu", values="accuracy")
    pivot = pivot.sort_index(ascending=True)
    pivot.index = pivot.index.droplevel(0)

    # Get number of unique heterogeneity levels
    n_levels = len(pivot.index)

    fig, ax = plt.subplots(figsize=(max(7, len(mu_values) * 1.3), max(4, n_levels * 0.9)))

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=pivot.values.mean(),
        ax=ax, cbar_kws={"label": "Test Accuracy", "shrink": 0.8},
        linewidths=0.5, linecolor="white"
    )
    ax.set_xlabel("Proximal Term (μ)")
    ax.set_ylabel("Data Distribution")
    ax.set_title("FedProx Performance: Complete Results\n(Pathological = 1-class/client, Dirichlet = α-controlled)")

    plt.tight_layout()
    fig.savefig(output_dir / "fig_comprehensive_heatmap.pdf")
    fig.savefig(output_dir / "fig_comprehensive_heatmap.png")
    plt.close(fig)


def write_academic_report(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive academic-style report with statistical analysis."""
    report_path = output_dir / "statistical_analysis_report.txt"
    stat_results = perform_statistical_tests(df)
    df_valid = df.dropna(subset=["partitioning", "alpha"])

    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("FEDPROX MNIST BASELINE: STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # 1. Experimental Overview
        f.write("1. EXPERIMENTAL OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total experimental runs: {len(df_valid)}\n")
        f.write(f"Unique seeds: {df_valid['seed'].nunique()}\n")
        f.write(f"Proximal μ values: {sorted(df_valid['proximal_mu'].dropna().unique())}\n")

        # Partitioning schemes
        partitionings = df_valid['partitioning'].unique().tolist()
        f.write(f"Partitioning schemes: {partitionings}\n")

        # Alpha values (only relevant for Dirichlet)
        dirichlet_df = df_valid[df_valid["partitioning"] == "dirichlet"]
        if not dirichlet_df.empty:
            alphas = sorted(dirichlet_df['alpha'].dropna().unique())
            alpha_strs = ["IID" if (isinstance(a, float) and math.isinf(a)) else str(a) for a in alphas]
            f.write(f"Dirichlet α values: {alpha_strs}\n")

        f.write(f"Dataset sizes (N): {sorted(df_valid['total_n'].dropna().unique())}\n")
        f.write("\nNote: 'Pathological' partitioning assigns each client exactly one class (most heterogeneous).\n")
        f.write("      'Dirichlet' partitioning uses α to control heterogeneity (lower α = more heterogeneous).\n\n")

        # 2. Key Research Questions
        f.write("2. KEY FINDINGS\n")
        f.write("-" * 40 + "\n\n")

        # RQ1: Does μ affect performance?
        if "mu_effect" in stat_results:
            res = stat_results["mu_effect"]
            f.write("RQ1: Does the proximal term μ significantly affect model accuracy?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  Test statistic: {res['statistic']:.4f}\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'} (α=0.05)\n\n")

        # RQ2: FedProx vs FedAvg
        if "fedprox_vs_fedavg" in stat_results:
            res = stat_results["fedprox_vs_fedavg"]
            f.write("RQ2: Does FedProx (μ>0) outperform FedAvg (μ=0)?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  FedAvg mean accuracy: {res['fedavg_mean']:.4f}\n")
            f.write(f"  FedProx mean accuracy: {res['fedprox_mean']:.4f}\n")
            f.write(f"  Difference: {res['difference']:.4f} ({res['difference']*100:.2f} pp)\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'} (α=0.05)\n\n")

        # RQ3: Effect of partitioning scheme
        if "partitioning_effect" in stat_results:
            res = stat_results["partitioning_effect"]
            f.write("RQ3: Does partitioning scheme (Pathological vs Dirichlet) affect performance?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  Pathological (1-class/client) mean accuracy: {res['extreme_mean']:.4f}\n")
            f.write(f"  Dirichlet (α-controlled) mean accuracy: {res['dirichlet_mean']:.4f}\n")
            f.write(f"  Difference: {res['difference']:.4f} ({res['difference']*100:.2f} pp)\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'} (α=0.05)\n\n")

        # RQ4: Effect of α within Dirichlet
        if "alpha_effect_dirichlet" in stat_results:
            res = stat_results["alpha_effect_dirichlet"]
            f.write("RQ4: Does α affect performance within Dirichlet partitioning?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  Test statistic: {res['statistic']:.4f}\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'} (α=0.05)\n")
            f.write(f"  Note: {res['note']}\n\n")

        # RQ5: Does FedProx help under extreme partitioning?
        if "fedprox_extreme" in stat_results:
            res = stat_results["fedprox_extreme"]
            f.write("RQ5: Does FedProx help under extreme (1-class/client) partitioning?\n")
            f.write(f"  Test: {res['test']}\n")
            f.write(f"  FedAvg mean accuracy: {res['fedavg_mean']:.4f}\n")
            f.write(f"  FedProx mean accuracy: {res['fedprox_mean']:.4f}\n")
            f.write(f"  Improvement: {res['improvement']:.4f} ({res['improvement']*100:.2f} pp)\n")
            f.write(f"  p-value: {res['p_value']:.4e}\n")
            f.write(f"  Significant: {'Yes' if res['p_value'] < 0.05 else 'No'} (α=0.05)\n\n")

        # RQ6: Where does FedProx help more?
        if "partitioning_benefit_comparison" in stat_results:
            res = stat_results["partitioning_benefit_comparison"]
            f.write("RQ6: Does FedProx help more under pathological or Dirichlet partitioning?\n")
            f.write(f"  Pathological improvement (FedProx - FedAvg): {res['extreme_improvement']*100:.2f} pp\n")
            f.write(f"  Dirichlet improvement (FedProx - FedAvg): {res['dirichlet_improvement']*100:.2f} pp\n")
            f.write(f"  FedProx helps more under extreme: {res['fedprox_helps_more_extreme']}\n\n")

        # 3. Descriptive Statistics
        f.write("3. DESCRIPTIVE STATISTICS\n")
        f.write("-" * 40 + "\n\n")

        f.write("Accuracy by Proximal μ:\n")
        mu_stats = df_valid.groupby("proximal_mu")["accuracy"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(4)
        f.write(mu_stats.to_string() + "\n\n")

        f.write("Accuracy by Partitioning Scheme:\n")
        part_stats = df_valid.groupby("partitioning")["accuracy"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(4)
        f.write(part_stats.to_string() + "\n\n")

        f.write("Accuracy by Partitioning and α (Dirichlet only):\n")
        dirichlet_alpha_stats = dirichlet_df.groupby("alpha")["accuracy"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(4)
        f.write(dirichlet_alpha_stats.to_string() + "\n\n")

        f.write("Accuracy by Full Configuration (Partitioning + α):\n")
        df_valid_copy = df_valid.copy()
        df_valid_copy["het_label"] = df_valid_copy.apply(
            lambda r: get_heterogeneity_label(r["partitioning"], r["alpha"]), axis=1
        )
        het_stats = df_valid_copy.groupby("het_label")["accuracy"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(4)
        f.write(het_stats.to_string() + "\n\n")

        # 4. Confidence Intervals
        f.write("4. 95% CONFIDENCE INTERVALS\n")
        f.write("-" * 40 + "\n\n")

        f.write("By Proximal μ:\n")
        for mu in sorted(df_valid["proximal_mu"].dropna().unique()):
            data = df_valid[df_valid["proximal_mu"] == mu]["accuracy"].dropna()
            if len(data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(data)
                mu_label = "FedAvg" if mu == 0 else f"μ={mu}"
                f.write(f"  {mu_label}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]\n")
        f.write("\n")

        f.write("By Partitioning Scheme:\n")
        for part in ["extreme", "dirichlet"]:
            data = df_valid[df_valid["partitioning"] == part]["accuracy"].dropna()
            if len(data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(data)
                part_label = "Pathological (1-class)" if part == "extreme" else "Dirichlet"
                f.write(f"  {part_label}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]\n")
        f.write("\n")

        # 5. Effect Sizes
        f.write("5. EFFECT SIZES\n")
        f.write("-" * 40 + "\n\n")

        # FedProx vs FedAvg overall
        fedavg = df_valid[df_valid["proximal_mu"] == 0]["accuracy"].dropna()
        fedprox = df_valid[df_valid["proximal_mu"] > 0]["accuracy"].dropna()
        if len(fedavg) > 1 and len(fedprox) > 1:
            pooled_std = np.sqrt(((len(fedavg)-1)*fedavg.std()**2 + (len(fedprox)-1)*fedprox.std()**2) /
                                (len(fedavg) + len(fedprox) - 2))
            cohens_d = (fedprox.mean() - fedavg.mean()) / pooled_std if pooled_std > 0 else 0
            f.write(f"Cohen's d (FedProx vs FedAvg, overall): {cohens_d:.3f}\n")
            if abs(cohens_d) < 0.2:
                effect_interp = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interp = "small"
            elif abs(cohens_d) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            f.write(f"  Interpretation: {effect_interp} effect\n\n")

        # Pathological vs Dirichlet
        extreme = df_valid[df_valid["partitioning"] == "extreme"]["accuracy"].dropna()
        dirichlet = df_valid[df_valid["partitioning"] == "dirichlet"]["accuracy"].dropna()
        if len(extreme) > 1 and len(dirichlet) > 1:
            pooled_std = np.sqrt(((len(extreme)-1)*extreme.std()**2 + (len(dirichlet)-1)*dirichlet.std()**2) /
                                (len(extreme) + len(dirichlet) - 2))
            cohens_d = (dirichlet.mean() - extreme.mean()) / pooled_std if pooled_std > 0 else 0
            f.write(f"Cohen's d (Dirichlet vs Pathological): {cohens_d:.3f}\n")
            if abs(cohens_d) < 0.2:
                effect_interp = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interp = "small"
            elif abs(cohens_d) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
            f.write(f"  Interpretation: {effect_interp} effect\n\n")

        # 6. Best Configurations
        f.write("6. TOP CONFIGURATIONS\n")
        f.write("-" * 40 + "\n\n")

        summary_valid = summary_df.dropna(subset=["partitioning", "alpha"])
        summary_valid = summary_valid.copy()
        summary_valid["het_label"] = summary_valid.apply(
            lambda r: get_heterogeneity_label(r["partitioning"], r["alpha"]), axis=1
        )

        best = summary_valid.sort_values("accuracy_mean", ascending=False).head(10)
        if not best.empty and not best["accuracy_mean"].isna().all():
            f.write("Top 10 configurations by mean accuracy:\n")
            for i, row in best.iterrows():
                f.write(f"  μ={row['proximal_mu']}, {row['het_label']}, N={int(row['total_n']) if pd.notna(row['total_n']) else '?'}: "
                       f"{row['accuracy_mean']:.4f} (±{row['accuracy_std']:.4f}, n={int(row['runs'])})\n")
        f.write("\n")

        # 7. LaTeX Table
        f.write("7. SUMMARY TABLE (LaTeX format)\n")
        f.write("-" * 40 + "\n\n")

        f.write("% Table 1: Results by Proximal Term\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{FedProx MNIST Results by Proximal Term}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Configuration & Accuracy & Std & n \\\\\n")
        f.write("\\midrule\n")

        for mu in sorted(df_valid["proximal_mu"].dropna().unique()):
            mu_data = df_valid[df_valid["proximal_mu"] == mu]["accuracy"].dropna()
            if not mu_data.empty:
                mu_label = "FedAvg ($\\mu=0$)" if mu == 0 else f"FedProx ($\\mu={mu}$)"
                f.write(f"{mu_label} & {mu_data.mean():.3f} & {mu_data.std():.3f} & {len(mu_data)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        f.write("% Table 2: Results by Partitioning Scheme\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{FedProx MNIST Results by Partitioning Scheme}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Partitioning & Accuracy & Std & n \\\\\n")
        f.write("\\midrule\n")

        for part in ["extreme", "dirichlet"]:
            part_data = df_valid[df_valid["partitioning"] == part]["accuracy"].dropna()
            if not part_data.empty:
                part_label = "Pathological (1-class/client)" if part == "extreme" else "Dirichlet ($\\alpha$-controlled)"
                f.write(f"{part_label} & {part_data.mean():.3f} & {part_data.std():.3f} & {len(part_data)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    # Save summary CSV with partitioning-aware grouping
    summary_stats = df_valid.groupby(["partitioning", "alpha", "proximal_mu"]).agg(
        n=("accuracy", "count"),
        mean=("accuracy", "mean"),
        std=("accuracy", "std"),
        min=("accuracy", "min"),
        max=("accuracy", "max"),
    ).reset_index()
    summary_stats["het_label"] = summary_stats.apply(
        lambda r: get_heterogeneity_label(r["partitioning"], r["alpha"]), axis=1
    )
    summary_stats.to_csv(output_dir / "summary_statistics.csv", index=False)


def plot_partitioning_comparison(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    """
    Dedicated figure comparing Pathological vs Dirichlet partitioning schemes.
    This highlights the dramatic difference between partitioning strategies.
    """
    apply_thesis_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    df_valid = df.dropna(subset=["partitioning", "alpha"])

    # Panel A: Bar chart comparing partitioning schemes
    ax = axes[0]
    partitionings = ["extreme", "dirichlet"]
    means = []
    stds = []
    colors = [COLORS["quaternary"], COLORS["primary"]]

    for part in partitionings:
        data = df_valid[df_valid["partitioning"] == part]["accuracy"].dropna()
        means.append(data.mean())
        stds.append(data.std())

    x = np.arange(len(partitionings))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["Pathological\n(1-class/client)", "Dirichlet\n(α-controlled)"])
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Mean Accuracy by Partitioning")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
               f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel B: Accuracy across μ values for each partitioning
    ax = axes[1]
    mu_values = sorted(df_valid["proximal_mu"].dropna().unique())

    for i, part in enumerate(["extreme", "dirichlet"]):
        part_df = df_valid[df_valid["partitioning"] == part]
        means = []
        ci_lows = []
        ci_highs = []

        for mu in mu_values:
            data = part_df[part_df["proximal_mu"] == mu]["accuracy"].dropna()
            if len(data) >= 2:
                mean, ci_low, ci_high = compute_confidence_interval(data)
            else:
                mean = data.mean() if len(data) > 0 else np.nan
                ci_low, ci_high = mean, mean
            means.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)

        label = "Pathological" if part == "extreme" else "Dirichlet"
        color = COLORS["quaternary"] if part == "extreme" else COLORS["primary"]
        linestyle = "--" if part == "extreme" else "-"
        marker = "s" if part == "extreme" else "o"

        ax.errorbar(
            mu_values, means,
            yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
            marker=marker, capsize=3, label=label, color=color,
            linewidth=1.5, markersize=6, linestyle=linestyle
        )

    ax.set_xlabel("Proximal Term (μ)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(b) Accuracy vs μ by Partitioning")
    ax.legend(title="Scheme", loc="best")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Panel C: FedProx improvement for each partitioning
    ax = axes[2]
    improvements = []

    for part in ["extreme", "dirichlet"]:
        part_df = df_valid[df_valid["partitioning"] == part]
        fedavg = part_df[part_df["proximal_mu"] == 0]["accuracy"].dropna()
        fedprox = part_df[part_df["proximal_mu"] > 0]["accuracy"].dropna()

        if len(fedavg) > 0 and len(fedprox) > 0:
            improvement = (fedprox.mean() - fedavg.mean()) * 100
            # Bootstrap CI
            if len(fedavg) > 1 and len(fedprox) > 1:
                boot_diffs = []
                for _ in range(1000):
                    fa_sample = np.random.choice(fedavg, size=len(fedavg), replace=True)
                    fp_sample = np.random.choice(fedprox, size=len(fedprox), replace=True)
                    boot_diffs.append((fp_sample.mean() - fa_sample.mean()) * 100)
                ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
            else:
                ci_low, ci_high = improvement, improvement

            improvements.append({
                "partitioning": part,
                "improvement": improvement,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    if improvements:
        imp_df = pd.DataFrame(improvements)
        x = np.arange(len(imp_df))
        colors_bar = [COLORS["quaternary"] if p == "extreme" else COLORS["primary"] for p in imp_df["partitioning"]]

        bars = ax.bar(x, imp_df["improvement"],
                     yerr=[imp_df["improvement"] - imp_df["ci_low"], imp_df["ci_high"] - imp_df["improvement"]],
                     capsize=5, color=colors_bar, alpha=0.7, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(["Pathological", "Dirichlet"])
        ax.set_ylabel("Improvement over FedAvg (pp)")
        ax.set_title("(c) FedProx Benefit by Partitioning")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        # Add value labels
        for bar, row in zip(bars, imp_df.itertuples()):
            y_pos = bar.get_height() + (row.ci_high - row.improvement) + 0.5 if bar.get_height() > 0 else bar.get_height() - 0.5
            va = "bottom" if bar.get_height() > 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f"{row.improvement:.1f}pp", ha="center", va=va, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_partitioning_comparison.pdf")
    fig.savefig(output_dir / "fig_partitioning_comparison.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export MNIST FedProx runs from W&B and generate thesis-ready figures.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/org (optional).")
    parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout in seconds.")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from W&B even if cached.")
    args = parser.parse_args()

    export_study(refresh=args.refresh, entity=args.entity, timeout=args.timeout)


def load_study_data(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 60,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = output_dir / "mnist_fedprox_raw.csv"
    df = load_or_fetch_dataframe(
        raw_path,
        refresh=refresh,
        fetch_fn=lambda: fetch_wandb_runs(PROJECT_NAME, entity=entity, timeout=timeout),
        normalize_fn=normalize_dataframe,
    )
    summary_df = summarize(df)
    summary_df.to_csv(output_dir / "mnist_fedprox_summary.csv", index=False)
    return df, summary_df


def export_study(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 60,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading MNIST FedProx study data...")
    df, summary_df = load_study_data(
        refresh=refresh,
        entity=entity,
        timeout=timeout,
        output_dir=output_dir,
    )

    if summary_df["accuracy_mean"].notna().sum() == 0:
        print("No accuracy values found in summaries. Skipping plots/report.")
        return None

    print("Generating publication-quality figures...")
    print(f"  - Total runs: {len(df)}")
    print(f"  - Unique configurations: {len(summary_df)}")

    print("  Creating main results figure...")
    plot_main_results_figure(df, summary_df, output_dir)

    print("  Creating heterogeneity benefit figure...")
    plot_heterogeneity_benefit(df, summary_df, output_dir)

    print("  Creating μ sensitivity figure...")
    plot_mu_sensitivity(df, summary_df, output_dir)

    print("  Creating comprehensive heatmap...")
    plot_comprehensive_heatmap(df, summary_df, output_dir)

    print("  Creating partitioning comparison figure...")
    plot_partitioning_comparison(df, summary_df, output_dir)

    print("  Generating statistical analysis report...")
    write_academic_report(df, summary_df, output_dir)

    print(f"\nOutput saved to {output_dir}/")
    return output_dir


if __name__ == "__main__":
    main()
