"""
Master thesis figure export script.

Regenerates ALL figures for the Results chapter (Chapter 5) and Appendix A
with unified styling from thesis_style.py.

Usage:
    python -m visual.export_all              # uses cached CSVs
    python -m visual.export_all --refresh    # re-fetch from W&B

Output goes to: visual/outputs/thesis/
"""

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
import wandb

# ---------------------------------------------------------------------------
# Resolve project root and add to path so imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visual.thesis_style import (
    apply_thesis_style,
    save_thesis_fig,
    get_fig,
    get_wide_fig,
    format_accuracy_axis,
    add_privacy_arrow,
    ALPHA_COLORS,
    ALPHA_LABELS,
    ALPHA_ORDER,
    EPSILON_COLORS,
    EPS_ORDER,
    EPS_LABELS,
    DROPOUT_COLORS,
    DROPOUT_MARKERS,
    COLORS,
    MU_COLORS,
    HEATMAP_CMAP,
    HEATMAP_ANNOT_COLOR,
    FULL_WIDTH,
    alpha_sort_key,
    synth_count_colors,
)
from visual.data import (
    fetch_exploration_runs,
    fetch_fedprox_runs,
    fetch_seeded_runs,
    load_or_fetch_dataframe,
    normalize_fedprox_dataframe,
    normalize_seeded_dataframe,
    parse_exploration_metadata,
    summarize_seeded_dataframe,
)

OUTPUT_DIR = PROJECT_ROOT / "visual" / "outputs" / "thesis"

# ---------------------------------------------------------------------------
# Cached data paths
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "visual" / "outputs"
MNIST_EXPLORATION_CACHE = CACHE_DIR / "mnist_exploration" / "mnist_exploration_raw.csv"
MNIST_SEEDED_CACHE = CACHE_DIR / "mnist_dpfedaug_seeded" / "mnist_dpfedaug_seeded_raw.csv"
CIFAR_SEEDED_CACHE = CACHE_DIR / "cifar_dpfedaug_seeded" / "cifar_dpfedaug_seeded_raw.csv"
FEDPROX_CACHE = CACHE_DIR / "mnist_fedprox" / "mnist_fedprox_raw.csv"


# ============================================================================
# DATA LOADING
# ============================================================================

# -- MNIST Exploration -------------------------------------------------------

def load_mnist_exploration(refresh=False, entity: str | None = None):
    df = load_or_fetch_dataframe(
        MNIST_EXPLORATION_CACHE,
        refresh=refresh,
        fetch_fn=lambda: fetch_exploration_runs("MNIST DP Exploration", entity=entity, timeout=60),
        normalize_fn=parse_exploration_metadata,
    )
    return df[df["state"] == "finished"].copy()


def _ci95(group):
    n = len(group)
    if n < 2:
        return 0
    return stats.t.ppf(0.975, n - 1) * group.std() / np.sqrt(n)


# -- Seeded DPFedAug (MNIST / CIFAR) ----------------------------------------

def load_seeded(dataset="mnist", refresh=False, entity: str | None = None):
    cache = MNIST_SEEDED_CACHE if dataset == "mnist" else CIFAR_SEEDED_CACHE
    project = ("DP-FedAug-MNIST-Study-seeded" if dataset == "mnist"
               else "DP-FedAug-CIFAR10-Study-seeded")
    return load_or_fetch_dataframe(
        cache,
        refresh=refresh,
        fetch_fn=lambda: fetch_seeded_runs(project, entity=entity, timeout=600),
        normalize_fn=normalize_seeded_dataframe,
    )


def summarize_seeded(df):
    return summarize_seeded_dataframe(df)


def _filter_standard(df):
    return df[df["updates_dp_enabled"] == False].copy()


# -- FedProx -----------------------------------------------------------------

def load_fedprox(refresh=False, entity: str | None = None):
    return load_or_fetch_dataframe(
        FEDPROX_CACHE,
        refresh=refresh,
        fetch_fn=lambda: fetch_fedprox_runs("FedProx-MNIST-Baseline", entity=entity, timeout=60),
        normalize_fn=normalize_fedprox_dataframe,
    )


def get_het_label(part, alpha):
    if part == "extreme":
        return "Pathological (1-class/client)"
    elif isinstance(alpha, float) and math.isinf(alpha):
        return "IID"
    else:
        return r"Dirichlet $\alpha=" + f"{alpha}" + r"$"


def get_het_order(part, alpha):
    if part == "extreme":
        return (0, 0)
    elif isinstance(alpha, float) and math.isinf(alpha):
        return (2, float("inf"))
    else:
        return (1, alpha)


def _ci(data, confidence=0.95):
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


# ============================================================================
# FIGURE GENERATORS
# ============================================================================

def fig_5_1(df, out):
    """Privacy-Utility Trade-off: ε vs accuracy with/without dropout (MNIST)."""
    eps_df = df[df["experiment_type"] == "epsilon"].copy()
    if eps_df.empty:
        return

    summary = (eps_df.groupby(["target_epsilon", "has_dropout"], dropna=False)
               .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), acc_n=("accuracy", "count"))
               .reset_index())
    summary["ci95"] = summary.apply(
        lambda r: stats.t.ppf(0.975, r["acc_n"] - 1) * r["acc_std"] / np.sqrt(r["acc_n"]) if r["acc_n"] > 1 else 0,
        axis=1)

    non_dp = summary[summary["target_epsilon"].isna()]
    dp = summary[summary["target_epsilon"].notna()].sort_values("target_epsilon")

    fig, ax = get_fig("single", aspect=0.7)

    for has_do, label, color, marker in [
        (True, "With Dropout", DROPOUT_COLORS["with"], DROPOUT_MARKERS["with"]),
        (False, "Without Dropout", DROPOUT_COLORS["without"], DROPOUT_MARKERS["without"]),
    ]:
        sub = dp[dp["has_dropout"] == has_do]
        if sub.empty: continue

        ax.errorbar(sub["target_epsilon"], sub["acc_mean"], yerr=sub["ci95"],
                    label=label, color=color, marker=marker,
                    markerfacecolor=color, markeredgecolor="white")

        bl = non_dp[non_dp["has_dropout"] == has_do]
        if not bl.empty:
            bl_acc = bl["acc_mean"].values[0]
            ax.axhline(bl_acc, color=color, ls=":", lw=1.2, alpha=0.7)

    ax.text(0.98, 0.98, "Dotted lines: non-DP baselines", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", style="italic", color=COLORS["neutral"])

    ax.set_xlabel(r"Privacy Budget ($\varepsilon$)")
    ax.set_ylabel("MNIST Test Accuracy")
    eps_vals = sorted(dp["target_epsilon"].dropna().unique())
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([str(int(e)) for e in eps_vals])
    format_accuracy_axis(ax)
    ax.legend(loc="upper left")
    add_privacy_arrow(ax)

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_1_privacy_utility_tradeoff", out)
    print("  fig_5_1_privacy_utility_tradeoff")


def fig_5_2(df, out):
    """Dropout on Convergence: epochs vs accuracy under fixed ε (MNIST)."""
    ep_df = df[df["experiment_type"] == "epochs"].copy()
    if ep_df.empty: return

    summary = (ep_df.groupby(["num_epochs", "has_dropout"], dropna=False)
               .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), acc_n=("accuracy", "count"))
               .reset_index()).sort_values("num_epochs")
    summary["ci95"] = summary.apply(
        lambda r: stats.t.ppf(0.975, r["acc_n"] - 1) * r["acc_std"] / np.sqrt(r["acc_n"]) if r["acc_n"] > 1 else 0,
        axis=1)

    fig, ax = get_fig("single", aspect=0.7)

    for has_do, label, color, marker in [
        (True, "With Dropout", DROPOUT_COLORS["with"], DROPOUT_MARKERS["with"]),
        (False, "Without Dropout", DROPOUT_COLORS["without"], DROPOUT_MARKERS["without"]),
    ]:
        sub = summary[summary["has_dropout"] == has_do]
        if sub.empty: continue

        ax.errorbar(sub["num_epochs"], sub["acc_mean"], yerr=sub["ci95"],
                    label=label, color=color, marker=marker,
                    markerfacecolor=color, markeredgecolor="white")

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("MNIST Test Accuracy")
    ax.set_xticks(sorted(summary["num_epochs"].dropna().unique()))
    format_accuracy_axis(ax)
    ax.legend(loc="lower right")

    eps_vals = ep_df["target_epsilon"].dropna().unique()
    eps_text = r"$\varepsilon = " + (f"{int(eps_vals[0])}" if len(eps_vals) == 1 else "1") + r"$"

    ax.text(0.02, 0.98, eps_text, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.8))

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_2_dropout_convergence", out)
    print("  fig_5_2_dropout_convergence")


def fig_5_3(out):
    """Distributional fidelity metrics for MNIST DP-VAE synthetic data."""
    conditions = [r"No DP ($\varepsilon=\infty$)", r"$\varepsilon=8$", r"$\varepsilon=1$"]
    alpha_precision = [0.456, 0.0, 0.0]
    beta_recall = [0.756, 0.0, 0.0]
    authenticity = [0.98, 0.95, 0.92]

    x = np.arange(len(conditions))
    w = 0.25

    fig, ax = get_fig("single", aspect=0.6)

    ax.bar(x - w, alpha_precision, w, label=r"$\alpha$-Precision", color=COLORS["primary"])
    ax.bar(x, beta_recall, w, label=r"$\beta$-Recall", color=COLORS["secondary"])
    ax.bar(x + w, authenticity, w, label="Authenticity", color=COLORS["tertiary"])

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_xlabel("Privacy Setting")
    ax.set_ylabel("Distributional Fidelity Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_3_distributional_fidelity", out)
    print("  fig_5_3_distributional_fidelity")


def fig_5_4(out):
    """CIFAR-10 enhanced pixel-space DP-VAE synthetic-only utility vs privacy."""
    csv_path = PROJECT_ROOT / "results" / "cifar_dpvae_pp_utility_agg.csv"
    if not csv_path.exists():
        print(f"  [skip] fig_5_4_cifar_dpvae_plusplus_utility (missing {csv_path})")
        return

    df = pd.read_csv(csv_path)
    sort_order = {"No DP": 0, "eps=8": 1, "eps=4": 2, "eps=2": 3, "eps=1": 4}
    df["sort_order"] = df["eps_label"].map(sort_order)
    df = df.sort_values("sort_order")

    fig, ax = get_fig("single", aspect=0.6)

    x_labels = df["eps_label"].tolist()
    accuracies = df["acc_mean"].tolist()
    stds = df["acc_std"].fillna(0.0).tolist()

    display_map = {
        "No DP": "No DP",
        "eps=8": r"$\varepsilon=8$",
        "eps=4": r"$\varepsilon=4$",
        "eps=2": r"$\varepsilon=2$",
        "eps=1": r"$\varepsilon=1$",
    }
    color_map = {
        "No DP": EPSILON_COLORS["none"],
        "eps=8": EPSILON_COLORS["8"],
        "eps=4": COLORS["secondary"],
        "eps=2": COLORS["tertiary"],
        "eps=1": EPSILON_COLORS["1"],
    }
    x_display = [display_map.get(label, label) for label in x_labels]
    bar_colors = [color_map.get(label, COLORS["neutral"]) for label in x_labels]

    bars = ax.bar(
        x_display,
        accuracies,
        yerr=stds,
        color=bar_colors,
        width=0.6,
        capsize=4,
        error_kw={"linewidth": 1.2, "color": "#333333"},
    )

    for bar, acc, std in zip(bars, accuracies, stds):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + std + 0.008, f"{acc:.1%}",
                ha="center", va="bottom", fontsize=10, color="#333333", fontweight="bold")

    ax.axhline(0.1, color=COLORS["quaternary"], ls="--", lw=1.2, alpha=0.8)
    ax.text(len(x_labels)-0.5, 0.115, "Random guess (10%)", color=COLORS["quaternary"], ha="right", fontsize=9)

    ax.set_ylabel("Synthetic-Only CIFAR-10 Test Accuracy")
    ax.set_xlabel("Privacy Budget")
    format_accuracy_axis(ax)
    ax.set_ylim(0, max(accuracies) + max(stds) + 0.08)

    sns.despine(ax=ax)
    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_4_cifar_dpvae_plusplus_utility", out)
    print("  fig_5_4_cifar_dpvae_plusplus_utility")


def fig_5_5(out):
    """Pathological heterogeneity: single-class MNIST client augmented with DP synthetic data."""
    labels = ["Local Only\n(single class)", "No DP\nsynthetic", r"$\varepsilon=8$" + "\nsynthetic",
              r"$\varepsilon=1$" + "\nsynthetic"]
    accuracies = [0.098, 0.928, 0.751, 0.136]
    bar_colors = [COLORS["neutral"], EPSILON_COLORS["none"], EPSILON_COLORS["8"], EPSILON_COLORS["1"]]

    fig, ax = get_fig("single", aspect=0.6)

    bars = ax.bar(labels, accuracies, color=bar_colors, width=0.5)

    for bar, acc in zip(bars, accuracies):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.02, f"{acc:.1%}",
                ha="center", va="bottom", fontsize=9, color="#333333")

    ax.axhline(0.1, color=COLORS["quaternary"], ls="--", lw=1.2, alpha=0.8)
    ax.text(3.4, 0.115, "Random guess (10%)", color=COLORS["quaternary"], ha="right", fontsize=9)

    ax.set_ylabel("MNIST Test Accuracy (10 classes)")
    ax.set_xlabel("Augmentation Strategy")
    format_accuracy_axis(ax)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_5_pathological_heterogeneity", out)
    print("  fig_5_5_pathological_heterogeneity")


def fig_5_6(summary_df, out):
    """MNIST DP-FedAug: accuracy across heterogeneity levels by synthetic count (no DP)."""
    df = _filter_standard(summary_df)
    df = df[df["target_epsilon_label"] == "none"]
    if df.empty: return

    total_ns = sorted(df["total_n"].unique())
    best_n = max(total_ns, key=lambda tn: df[df["total_n"] == tn]["alpha"].nunique()) if total_ns else None
    if best_n is None: return
    n_df = df[df["total_n"] == best_n]

    synth_counts = sorted(n_df["synthetic_count"].unique())
    sc_colors = synth_count_colors(synth_counts)

    alpha_x = {a: i for i, a in enumerate(ALPHA_ORDER)}
    bar_width = 0.75 / max(len(synth_counts), 1)

    fig, ax = get_fig("single", aspect=0.65)

    for j, sc in enumerate(synth_counts):
        sc_df = n_df[n_df["synthetic_count"] == sc]
        xs, ys, errs = [], [], []
        for a in ALPHA_ORDER:
            row = sc_df[sc_df["alpha"] == a]
            if row.empty: continue
            xs.append(alpha_x[a] + (j - len(synth_counts) / 2) * bar_width + bar_width / 2)
            ys.append(row["accuracy_mean"].values[0])
            errs.append(row["accuracy_std"].values[0] if not np.isnan(row["accuracy_std"].values[0]) else 0)

        if not xs: continue
        label = f"Synthetic = {int(sc)}" if sc > 0 else "Baseline (no augmentation)"

        ax.bar(xs, ys, width=bar_width, yerr=errs, label=label, color=sc_colors[sc], alpha=0.9)

    ax.set_xticks(range(len(ALPHA_ORDER)))
    ax.set_xticklabels([ALPHA_LABELS.get(a, a) for a in ALPHA_ORDER])
    ax.set_ylabel("MNIST Test Accuracy")
    ax.set_xlabel("Data Heterogeneity")
    format_accuracy_axis(ax)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_6_partitioning_augmentation", out)
    print("  fig_5_6_partitioning_augmentation")


RUNS_ALPHA01 = {
    "FedAvg": {
        "project": "FedProx-MNIST-Baseline",
        "ids": ["thjb5s0a", "558qwcyt", "6pp2drzm"],
    },
    "Best FedProx": {
        "project": "FedProx-MNIST-Baseline",
        "ids": ["gg38q1bu", "fam92ds8", "sm26jczw"],
    },
    "FedAug (no DP)": {
        "project": "DP-FedAug-MNIST-Study-seeded",
        "ids": ["ijq3et8y", "jmdgcveq", "69yevift"],
    },
    r"DP-FedAug $\varepsilon=8$": {
        "project": "DP-FedAug-MNIST-Study-seeded",
        "ids": ["bc751ws0", "dcvdf418", "lmenhsdm"],
    },
    r"DP-FedAug $\varepsilon=1$": {
        "project": "DP-FedAug-MNIST-Study-seeded",
        "ids": ["uctbx3mg", "eoanz7xo", "5wmhyi5l"],
    },
}

RUNS_EXTREME = {
    "FedAvg": {
        "project": "FedProx-MNIST-Baseline",
        "ids": ["okw0jev8", "573uwua2", "zcjkfqde"],
    },
    "Best FedProx": {
        "project": "FedProx-MNIST-Baseline",
        "ids": ["eh5tt7iq", "64rlphh6", "skym0q3z"],
    },
    "FedAug (no DP)": {
        "project": "DP-FedAug-MNIST-Study-seeded",
        "ids": ["56686942", "1hfarx28", "ci1n113n"],
    },
    r"DP-FedAug $\varepsilon=8$": {
        "project": "DP-FedAug-MNIST-Study-seeded",
        "ids": ["i0xlaces", "3vqzi0f5", "yfpxesis"],
    },
    r"DP-FedAug $\varepsilon=1$": {
        "project": "DP-FedAug-MNIST-Study-seeded",
        "ids": ["p2svzia1", "k10rb2lp", "lp6p35fp"],
    },
}

ALGORITHM_COLORS = {
    "FedAvg": "#999999",
    "Best FedProx": "#56B4E9",
    "FedAug (no DP)": "#009E73",
    r"DP-FedAug $\varepsilon=8$": "#CC6677",
    r"DP-FedAug $\varepsilon=1$": "#332288",
}

ALGORITHM_LINESTYLES = {
    "FedAvg": "-",
    "Best FedProx": "--",
    "FedAug (no DP)": "-",
    r"DP-FedAug $\varepsilon=8$": "-.",
    r"DP-FedAug $\varepsilon=1$": ":",
}


def _fedprox_alpha_label(value) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "Infinity"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _build_algorithm_table(mnist_seeded: pd.DataFrame, fedprox_df: pd.DataFrame, total_n: int):
    partitions = ["extreme", "0.1", "1", "Infinity"]
    algorithms = [
        "FedAvg",
        "Best FedProx",
        "FedAug (no DP)",
        r"DP-FedAug $\varepsilon=8$",
        r"DP-FedAug $\varepsilon=1$",
    ]
    means = np.full((len(algorithms), len(partitions)), np.nan)
    stds = np.full((len(algorithms), len(partitions)), np.nan)

    fedprox_df = fedprox_df.copy()
    if "state" in fedprox_df.columns:
        fedprox_df = fedprox_df[fedprox_df["state"] == "finished"].copy()
    fedprox_df["alpha_label"] = fedprox_df["alpha"].apply(_fedprox_alpha_label)
    fedprox_df.loc[fedprox_df["partitioning"] == "extreme", "alpha_label"] = "extreme"

    dpfedaug_df = mnist_seeded.copy()
    if "state" in dpfedaug_df.columns:
        dpfedaug_df = dpfedaug_df[dpfedaug_df["state"] == "finished"].copy()
    dpfedaug_df = dpfedaug_df[dpfedaug_df["updates_dp_enabled"] == False]

    def fedavg_stats(partition: str):
        if partition == "extreme":
            subset = fedprox_df[
                (fedprox_df["total_n"] == total_n)
                & (fedprox_df["partitioning"] == "extreme")
                & (fedprox_df["proximal_mu"] == 0.0)
            ]
        else:
            subset = fedprox_df[
                (fedprox_df["total_n"] == total_n)
                & (fedprox_df["alpha_label"] == partition)
                & (fedprox_df["proximal_mu"] == 0.0)
            ]
        return subset["accuracy"].mean(), subset["accuracy"].std()

    def best_fedprox_stats(partition: str):
        if partition == "extreme":
            subset = fedprox_df[
                (fedprox_df["total_n"] == total_n)
                & (fedprox_df["partitioning"] == "extreme")
                & (fedprox_df["proximal_mu"] > 0.0)
            ]
        else:
            subset = fedprox_df[
                (fedprox_df["total_n"] == total_n)
                & (fedprox_df["alpha_label"] == partition)
                & (fedprox_df["proximal_mu"] > 0.0)
            ]
        if subset.empty:
            return np.nan, np.nan
        best_mu = subset.groupby("proximal_mu")["accuracy"].mean().idxmax()
        best_subset = subset[subset["proximal_mu"] == best_mu]
        return best_subset["accuracy"].mean(), best_subset["accuracy"].std()

    def dpfedaug_stats(partition: str, epsilon_label: str):
        if partition == "extreme":
            subset = dpfedaug_df[
                (dpfedaug_df["total_n"] == total_n)
                & (dpfedaug_df["alpha"] == "extreme")
                & (dpfedaug_df["target_epsilon_label"] == epsilon_label)
                & (dpfedaug_df["synthetic_count"] > 0)
            ]
        else:
            subset = dpfedaug_df[
                (dpfedaug_df["total_n"] == total_n)
                & (dpfedaug_df["alpha"] == partition)
                & (dpfedaug_df["target_epsilon_label"] == epsilon_label)
                & (dpfedaug_df["synthetic_count"] > 0)
            ]
        if subset.empty:
            return np.nan, np.nan
        best_synth = subset.groupby("synthetic_count")["accuracy"].mean().idxmax()
        best_subset = subset[subset["synthetic_count"] == best_synth]
        return best_subset["accuracy"].mean(), best_subset["accuracy"].std()

    for index, partition in enumerate(partitions):
        means[0, index], stds[0, index] = fedavg_stats(partition)
        means[1, index], stds[1, index] = best_fedprox_stats(partition)
        means[2, index], stds[2, index] = dpfedaug_stats(partition, "none")
        means[3, index], stds[3, index] = dpfedaug_stats(partition, "8")
        means[4, index], stds[4, index] = dpfedaug_stats(partition, "1")

    return partitions, algorithms, means, stds


def fig_5_2_3_algorithm_comparison(mnist_seeded: pd.DataFrame, fedprox_df: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.8), sharey=False)
    partition_labels = [ALPHA_LABELS[p] for p in ["extreme", "0.1", "1", "Infinity"]]

    for panel_index, (total_n, ax) in enumerate(zip([600, 1000], axes)):
        partitions, algorithms, means, stds = _build_algorithm_table(mnist_seeded, fedprox_df, total_n)
        x = np.arange(len(partitions))
        bar_width = 0.15
        offsets = np.arange(len(algorithms)) - (len(algorithms) - 1) / 2

        for algo_index, algorithm in enumerate(algorithms):
            ax.bar(
                x + offsets[algo_index] * bar_width,
                means[algo_index],
                bar_width,
                yerr=stds[algo_index],
                label=algorithm if panel_index == 0 else None,
                color=ALGORITHM_COLORS[algorithm],
                edgecolor="white",
                linewidth=0.3,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

        ax.set_title(f"$N = {total_n}$", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(partition_labels, fontsize=9)
        ax.set_xlabel("Partitioning Regime")
        format_accuracy_axis(ax)
        ax.set_ylim(0, 1.05)
        if panel_index == 0:
            ax.set_ylabel("Test Accuracy")

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=5,
        fontsize=8,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_thesis_fig(fig, "fig_5_2_3_algorithm_comparison", out)
    print("  fig_5_2_3_algorithm_comparison")


def _fetch_histories(run_config, api: wandb.Api, entity: str | None):
    all_series = []
    for run_id in run_config["ids"]:
        run_path = f"{entity}/{run_config['project']}/{run_id}" if entity else f"{run_config['project']}/{run_id}"
        run = api.run(run_path)
        history = run.history(samples=500)
        if "Global Test/Accuracy" in history.columns:
            accuracy_column = "Global Test/Accuracy"
        elif "Global Test/accuracy" in history.columns:
            accuracy_column = "Global Test/accuracy"
        else:
            raise ValueError(f"No accuracy column found in run {run_id}.")

        if "Communication Round" in history.columns:
            series = history.set_index("Communication Round")[accuracy_column].dropna()
        else:
            series = history[accuracy_column].dropna()
            series.index = range(1, len(series) + 1)
        series.name = run_id
        all_series.append(series)

    df = pd.concat(all_series, axis=1)
    df.index.name = "round"
    return df


def fig_5_2_3b_convergence(out, entity: str | None = None):
    api = wandb.Api(timeout=60)
    panels = {
        "alpha01": {
            "title": r"$N=600,\ \alpha=0.1,\ S=50$",
            "runs": RUNS_ALPHA01,
        },
        "extreme": {
            "title": r"$N=600,\ \mathrm{Pathological},\ S=50$",
            "runs": RUNS_EXTREME,
        },
    }
    series_by_panel = {}
    for panel_key, panel in panels.items():
        series_by_panel[panel_key] = {}
        for algorithm, run_config in panel["runs"].items():
            history_df = _fetch_histories(run_config, api, entity)
            series_by_panel[panel_key][algorithm] = {
                "mean": history_df.mean(axis=1),
                "std": history_df.std(axis=1),
            }

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.2), sharey=True)
    for ax, (panel_key, panel) in zip(axes, panels.items()):
        for algorithm in ALGORITHM_COLORS:
            stats_dict = series_by_panel[panel_key][algorithm]
            rounds = stats_dict["mean"].index
            ax.plot(
                rounds,
                stats_dict["mean"],
                label=algorithm,
                color=ALGORITHM_COLORS[algorithm],
                linestyle=ALGORITHM_LINESTYLES[algorithm],
                linewidth=1.5,
            )
            ax.fill_between(
                rounds,
                stats_dict["mean"] - stats_dict["std"],
                stats_dict["mean"] + stats_dict["std"],
                alpha=0.15,
                color=ALGORITHM_COLORS[algorithm],
            )
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel("Communication Round")
        format_accuracy_axis(ax)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Test Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_thesis_fig(fig, "fig_5_2_3b_convergence", out)
    print("  fig_5_2_3b_convergence")


def fig_5_3_4_updates_dp(out):
    data = {
        "Dirichlet ($\\alpha\\!=\\!0.1$)": {
            "Baseline\n(FedAvg)": {"mean": 0.935967, "std": 0.012553},
            "FedAvg +\nUpdates DP": {"mean": 0.763100, "std": 0.039761},
            "DP-FedAug\n(VAE DP, $\\varepsilon\\!=\\!8$)": {"mean": 0.954567, "std": 0.002285},
            "Full Pipeline\n(VAE + Updates DP)": {"mean": 0.786140, "std": 0.006906},
        },
        "Pathological\n(1 class/client)": {
            "Baseline\n(FedAvg)": {"mean": 0.456767, "std": 0.036869},
            "FedAvg +\nUpdates DP": {"mean": 0.362867, "std": 0.105338},
            "DP-FedAug\n(VAE DP, $\\varepsilon\\!=\\!8$)": {"mean": 0.770700, "std": 0.016408},
            "Full Pipeline\n(VAE + Updates DP)": {"mean": 0.641175, "std": 0.011858},
        },
    }
    conditions = list(next(iter(data.values())).keys())
    regimes = list(data.keys())
    bar_colors = ["#0072B2", "#56B4E9", "#009E73", "#D55E00"]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.52))
    x = np.arange(len(regimes))
    bar_width = 0.18

    for index, condition in enumerate(conditions):
        means = [data[regime][condition]["mean"] for regime in regimes]
        stds = [data[regime][condition]["std"] for regime in regimes]
        offset = (index - (len(conditions) - 1) / 2) * (bar_width + 0.02)
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            color=bar_colors[index],
            label=condition.replace("\n", " "),
            capsize=3,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.015,
                f"{mean:.1%}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("Global Test Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_3_4_updates_dp_ablation", out)
    print("  fig_5_3_4_updates_dp_ablation")


def fig_5_7(df, out):
    """
    FedProx MNIST Ablation: Clean line plot showing the effect of mu on accuracy.
    Highlights the optimal mu setting.
    """
    df_valid = df.dropna(subset=["partitioning", "alpha", "proximal_mu", "accuracy"]).copy()
    if df_valid.empty: return

    # Create readable labels for the legend
    df_valid["het_label"] = df_valid.apply(lambda r: get_het_label(r["partitioning"], r["alpha"]), axis=1)

    # Sort by mu so the x-axis is ordered
    df_valid = df_valid.sort_values("proximal_mu")
    # Convert to string for categorical x-axis
    df_valid["proximal_mu_str"] = df_valid["proximal_mu"].apply(lambda x: f"{x:g}")

    # 1. Setup a clean, single-panel figure
    fig, axes = get_wide_fig(1, panel_width=6.0, height=4.5, constrain_width=False)
    ax = axes[0]

    # 2. Plot lines with clean shaded confidence intervals (replaces manual loop)
    sns.lineplot(
        data=df_valid,
        x="proximal_mu_str",
        y="accuracy",
        hue="het_label",
        style="het_label",
        markers=True,
        dashes=False,
        linewidth=2,
        markersize=8,
        ax=ax,
        palette="deep",
        errorbar=('ci', 95)  # 95% Confidence Interval shading
    )

    # 3. Find and annotate the global optimal mu (excluding mu=0 / FedAvg)
    fedprox_only = df_valid[df_valid["proximal_mu"] > 0]
    if not fedprox_only.empty:
        best_idx = fedprox_only["accuracy"].idxmax()
        best_mu = fedprox_only.loc[best_idx, "proximal_mu"]
        best_acc = fedprox_only.loc[best_idx, "accuracy"]
        best_mu_str = f"{best_mu:g}"

        # Find the x-coordinate for the categorical axis
        mu_strs = df_valid["proximal_mu_str"].unique().tolist()
        if best_mu_str in mu_strs:
            best_x_pos = mu_strs.index(best_mu_str)

            # Draw a subtle vertical line at the best mu
            ax.axvline(x=best_x_pos, color="gray", linestyle="--", alpha=0.5, zorder=0)
            ax.annotate(
                f"Optimal $\\mu$ = {best_mu}\n(Peak Acc: {best_acc:.3f})",
                xy=(best_x_pos, best_acc),
                xytext=(best_x_pos + 0.05, best_acc + 0.02),
                arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.2"),
                fontsize=10,
                fontweight='bold'
            )

    # 4. Clean up axes and legend for academic presentation
    ax.set_xlabel(r"Proximal Term ($\mu$)", fontsize=12, fontweight='bold')
    ax.set_ylabel("MNIST Test Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("FedProx Hyperparameter Ablation", pad=15, fontsize=14, fontweight='bold')

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Data Distribution", frameon=False, loc="lower right")

    sns.despine(ax=ax)  # Removes top and right borders for a cleaner look
    if "format_accuracy_axis" in globals(): format_accuracy_axis(ax)

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_7_fedprox_ablation", out)
    print("  fig_5_7_fedprox_ablation generated")


def fig_5_8(df, out):
    """
    FedProx vs FedAvg Benefit: Grouped bar chart showing the direct
    accuracy improvement of the optimal FedProx over standard FedAvg.
    """
    df_valid = df.dropna(subset=["partitioning", "alpha", "proximal_mu", "accuracy"]).copy()
    if df_valid.empty: return

    df_valid["het_label"] = df_valid.apply(lambda r: get_het_label(r["partitioning"], r["alpha"]), axis=1)
    df_valid["sort_key"] = df_valid.apply(lambda r: get_het_order(r["partitioning"], r["alpha"]), axis=1)

    # 1. Aggregate data: Compare mu=0 (FedAvg) vs Best mu (FedProx)
    summary = []
    for label, group in df_valid.groupby("het_label"):
        sort_key = group["sort_key"].iloc[0]

        # Get FedAvg (mu=0)
        fedavg_data = group[group["proximal_mu"] == 0]["accuracy"]
        fedavg_acc = fedavg_data.mean() if not fedavg_data.empty else np.nan

        # Get FedProx (best mu > 0)
        fp_data = group[group["proximal_mu"] > 0]
        best_fp_acc = np.nan
        if not fp_data.empty:
            # Group by mu to find the one with the highest mean accuracy
            best_mu = fp_data.groupby("proximal_mu")["accuracy"].mean().idxmax()
            best_fp_acc = fp_data[fp_data["proximal_mu"] == best_mu]["accuracy"].mean()

        if pd.notna(fedavg_acc) and pd.notna(best_fp_acc):
            summary.append(
                {"Distribution": label, "Algorithm": "FedAvg ($\mu=0$)", "Accuracy": fedavg_acc, "sort_key": sort_key})
            summary.append({"Distribution": label, "Algorithm": f"FedProx (Optimal)", "Accuracy": best_fp_acc,
                            "sort_key": sort_key})

    plot_df = pd.DataFrame(summary).sort_values("sort_key")

    # 2. Setup a single-panel figure
    fig, axes = get_wide_fig(1, panel_width=6.0, height=4.5, constrain_width=False)
    ax = axes[0]

    # 3. Draw a clean grouped bar chart
    sns.barplot(
        data=plot_df,
        x="Distribution",
        y="Accuracy",
        hue="Algorithm",
        ax=ax,
        palette=["#cccccc", "#2b8cbe"],  # Gray for baseline, strong blue for proposed
        edgecolor="black",
        linewidth=1
    )

    # 4. Add text annotations on top of the bars to show the exact gap
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=9)

    # 5. Formatting
    ax.set_xlabel("Data Heterogeneity", fontsize=12, fontweight='bold')
    ax.set_ylabel("MNIST Test Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Performance Gain: Optimal FedProx vs. FedAvg", pad=15, fontsize=14, fontweight='bold')

    ax.legend(title="", frameon=False, loc="upper left")
    sns.despine(ax=ax)

    # Slightly rotate x-axis labels if they are long
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Zoom in the Y-axis to highlight differences
    min_acc = plot_df["Accuracy"].min()
    ax.set_ylim(bottom=max(0, min_acc - 0.05))

    fig.tight_layout()
    save_thesis_fig(fig, "fig_5_8_fedprox_benefit", out)
    print("  fig_5_8_fedprox_benefit generated")


def fig_a1_heatmaps(summary_df, dataset_label, out, name_suffix):
    """DP-FedAug accuracy heatmaps: rows=synth_count, cols=epsilon, panels=alpha."""
    df = _filter_standard(summary_df)
    if df.empty: return

    alpha_cols = [a for a in ALPHA_ORDER if a in df["alpha"].values]

    for total_n, n_df in df.groupby("total_n"):
        alphas_present = [a for a in alpha_cols if a in n_df["alpha"].values]
        if not alphas_present: continue

        synth_rows = sorted(n_df["synthetic_count"].unique())
        eps_cols = [e for e in EPS_ORDER if e in n_df["target_epsilon_label"].values]
        n_panels = len(alphas_present)

        # Global min and max for a unified colorbar
        global_vmin = n_df["accuracy_mean"].min()
        global_vmax = n_df["accuracy_mean"].max()

        if n_panels == 4:
            fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.8), squeeze=False)
            axes_flat = axes.flatten()
            annot_size = 7.5
        else:
            # Adjust panel width for single-panel figures (like CIFAR-10) so they aren't squished
            panel_w = 4.5 if n_panels == 1 else 2.5
            fig, axes_flat = get_wide_fig(n_panels, panel_width=panel_w, height=3.5, constrain_width=False)

            # Slightly smaller font to comfortably fit two lines of text
            annot_size = 8.5 if n_panels == 1 else (7 if n_panels <= 2 else 6.5)

        for pi, (ax, alpha) in enumerate(zip(axes_flat, alphas_present)):
            a_df = n_df[n_df["alpha"] == alpha]
            matrix = pd.DataFrame(np.nan, index=synth_rows, columns=eps_cols, dtype=float)
            annot = pd.DataFrame("", index=synth_rows, columns=eps_cols, dtype=object)

            for sc in synth_rows:
                for eps in eps_cols:
                    row = a_df[(a_df["synthetic_count"] == sc) & (a_df["target_epsilon_label"] == eps)]
                    if not row.empty:
                        mean = row["accuracy_mean"].values[0]
                        std = row["accuracy_std"].values[0]
                        matrix.loc[sc, eps] = mean

                        # Add variance/std dev below the mean if it exists and is > 0
                        if pd.notna(std) and std > 0:
                            annot.loc[sc, eps] = f"{mean:.3f}\n(±{std:.3f})"
                        else:
                            annot.loc[sc, eps] = f"{mean:.3f}"

            show_cbar = pi == n_panels - 1

            sns.heatmap(matrix, ax=ax, annot=annot, fmt="",
                        annot_kws={"size": annot_size},
                        cmap=HEATMAP_CMAP, vmin=global_vmin, vmax=global_vmax,
                        linewidths=1.5, linecolor="white",
                        mask=matrix.isna(), cbar=show_cbar,
                        cbar_kws={"label": "Accuracy", "shrink": 0.85} if show_cbar else {})

            if show_cbar and len(ax.collections) > 0 and ax.collections[0].colorbar is not None:
                ax.collections[0].colorbar.outline.set_visible(False)

            ax.set_title(ALPHA_LABELS.get(alpha, alpha), pad=10)
            ax.set_xlabel(r"Privacy Budget ($\varepsilon$)")
            ax.set_xticklabels([EPS_LABELS.get(e, e) for e in eps_cols], rotation=0)

            if n_panels == 4:
                row_i, col_i = divmod(pi, 2)
                if col_i == 0:
                    ax.set_ylabel("Synthetic Samples per Client")
                    ax.set_yticklabels([str(int(s)) for s in synth_rows], rotation=0)
                else:
                    ax.set_yticks([])
                    ax.set_ylabel("")

                if row_i == 1:
                    ax.set_xlabel(r"Privacy Budget ($\varepsilon$)")
                else:
                    ax.set_xlabel("")
            elif pi == 0:
                ax.set_ylabel("Synthetic Samples per Client")
                ax.set_yticklabels([str(int(s)) for s in synth_rows], rotation=0)
            else:
                ax.set_yticks([])
                ax.set_ylabel("")

        fig.suptitle(f"{dataset_label} DP-FedAug Accuracy (N={int(total_n):,})", y=1.05)
        if n_panels == 4:
            fig.tight_layout(h_pad=1.1, w_pad=0.6)
        else:
            fig.tight_layout(w_pad=0.3)
        fname = f"fig_a1_{name_suffix}_heatmap_N{int(total_n)}"
        save_thesis_fig(fig, fname, out)
        print(f"  {fname}")

# ============================================================================
# MAIN
# ============================================================================

def export_all_figures(*, refresh: bool = False, entity: str | None = None) -> Path:
    apply_thesis_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting thesis figures to: {OUTPUT_DIR}\n")

    print("Section 5.1: Centralized Privacy-Utility Benchmarks")
    try:
        exploration_df = load_mnist_exploration(refresh=refresh, entity=entity)
        fig_5_1(exploration_df, OUTPUT_DIR)
        fig_5_2(exploration_df, OUTPUT_DIR)
    except Exception as e:
        print(f"  [error] {e}")

    print("\nSection 5.2: DP Synthetic Data Quality")
    fig_5_3(OUTPUT_DIR)
    fig_5_4(OUTPUT_DIR)
    fig_5_5(OUTPUT_DIR)

    print("\nSection 5.3: DP-FedAug System Performance")
    mnist_summary = None
    mnist_seeded = None
    try:
        mnist_seeded = load_seeded("mnist", refresh=refresh, entity=entity)
        mnist_summary = summarize_seeded(mnist_seeded)
        fig_5_6(mnist_summary, OUTPUT_DIR)
    except Exception as e:
        print(f"  [error] fig_5_6: {e}")

    print("\nSection 5.3.2: Cross-Method Comparison")
    fedprox_df = None
    try:
        fedprox_df = load_fedprox(refresh=refresh, entity=entity)
        if mnist_seeded is None:
            mnist_seeded = load_seeded("mnist", refresh=refresh, entity=entity)
        fig_5_2_3_algorithm_comparison(mnist_seeded, fedprox_df, OUTPUT_DIR)
        fig_5_2_3b_convergence(OUTPUT_DIR, entity=entity)
        fig_5_3_4_updates_dp(OUTPUT_DIR)
    except Exception as e:
        print(f"  [error] comparison figures: {e}")

    print("\nSection 5.3.3: FedProx Baseline")
    try:
        if fedprox_df is None:
            fedprox_df = load_fedprox(refresh=refresh, entity=entity)
        fig_5_7(fedprox_df, OUTPUT_DIR)
        fig_5_8(fedprox_df, OUTPUT_DIR)
    except Exception as e:
        print(f"  [error] fig_5_7/5_8: {e}")

    print("\nAppendix A: Heatmaps")
    try:
        if mnist_summary is None:
            mnist_seeded = load_seeded("mnist", refresh=refresh, entity=entity)
            mnist_summary = summarize_seeded(mnist_seeded)
        fig_a1_heatmaps(mnist_summary, "MNIST", OUTPUT_DIR, "mnist")
    except Exception as e:
        print(f"  [error] MNIST heatmaps: {e}")

    try:
        cifar_seeded = load_seeded("cifar", refresh=refresh, entity=entity)
        cifar_summary = summarize_seeded(cifar_seeded)
        fig_a1_heatmaps(cifar_summary, "CIFAR-10", OUTPUT_DIR, "cifar")
    except Exception as e:
        print(f"  [error] CIFAR heatmaps: {e}")

    print(f"\nDone. All figures saved to: {OUTPUT_DIR}")
    return OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Export all thesis figures.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data from W&B.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/org.")
    args = parser.parse_args()
    export_all_figures(refresh=args.refresh, entity=args.entity)


if __name__ == "__main__":
    main()
