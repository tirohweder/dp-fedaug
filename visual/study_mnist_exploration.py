"""
MNIST Differential Privacy Exploration Analysis

This script produces two publication-quality figures analyzing the effects of:
1. Privacy budget (epsilon) on model accuracy with/without dropout regularization
2. Training epochs on model convergence with/without dropout under fixed privacy constraints

Run naming convention:
- epsilon_*: Varying target epsilon experiments
- epochs_*: Varying training epochs experiments
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Use shared thesis style
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual.data import (
    compute_group_summary,
    fetch_exploration_runs,
    load_or_fetch_dataframe,
    parse_exploration_metadata,
)
from visual.thesis_style import (
    apply_thesis_style,
    save_thesis_fig,
    DROPOUT_COLORS as COLORS_MAP,
    DROPOUT_MARKERS as MARKERS_MAP,
)


PROJECT_NAME = "MNIST DP Exploration"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "mnist_exploration"

# Map to legacy names for backward compatibility
COLORS = {
    "dropout": COLORS_MAP["with"],
    "no_dropout": COLORS_MAP["without"],
    "non_dp": "#7D7D7D",
}
MARKERS = {
    "dropout": MARKERS_MAP["with"],
    "no_dropout": MARKERS_MAP["without"],
}


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float, str, bool, np.number)) or value is None


def _flatten_value(value):
    if _is_scalar(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 10:
        return json.dumps(value)
    if isinstance(value, dict) and len(value) <= 10:
        return json.dumps(value, sort_keys=True)
    return None



def fetch_wandb_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    return fetch_exploration_runs(project, entity=entity, timeout=timeout)


def parse_run_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return parse_exploration_metadata(df)


def compute_summary_statistics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return compute_group_summary(df, group_cols)


def plot_epsilon_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Figure 1: Privacy-Utility Trade-off

    Shows validation accuracy vs target epsilon (privacy budget) comparing
    models trained with and without dropout regularization.
    """
    apply_thesis_style()

    # Filter to epsilon experiment runs only
    eps_df = df[df["experiment_type"] == "epsilon"].copy()
    if eps_df.empty:
        print("No epsilon experiment runs found.")
        return

    # Compute summary statistics
    summary = compute_summary_statistics(eps_df, ["target_epsilon", "has_dropout"])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Separate non-DP baseline (epsilon=None)
    non_dp_data = summary[summary["target_epsilon"].isna()]
    dp_data = summary[summary["target_epsilon"].notna()].copy()

    # Sort by epsilon for proper line plotting
    dp_data = dp_data.sort_values("target_epsilon")

    # Plot DP results for each dropout condition
    for has_dropout, group_label, color, marker in [
        (True, "With Dropout", COLORS["dropout"], MARKERS["dropout"]),
        (False, "Without Dropout", COLORS["no_dropout"], MARKERS["no_dropout"]),
    ]:
        subset = dp_data[dp_data["has_dropout"] == has_dropout]
        if subset.empty:
            continue

        ax.errorbar(
            subset["target_epsilon"],
            subset["accuracy_mean"],
            yerr=subset["accuracy_ci95"],
            label=group_label,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            linewidth=2,
        )

        # Add non-DP baseline as horizontal reference
        baseline = non_dp_data[non_dp_data["has_dropout"] == has_dropout]
        if not baseline.empty:
            baseline_acc = baseline["accuracy_mean"].values[0]
            baseline_ci = baseline["accuracy_ci95"].values[0]
            ax.axhline(
                y=baseline_acc,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
            )
            # Add shaded region for baseline CI
            ax.axhspan(
                baseline_acc - baseline_ci,
                baseline_acc + baseline_ci,
                color=color,
                alpha=0.1,
            )

    # Add text annotation for baselines
    ax.text(
        0.98, 0.98,
        "Dashed lines: Non-DP baselines",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        style="italic",
        color="#666666",
    )

    ax.set_xlabel("Target Privacy Budget (ε)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Privacy-Utility Trade-off: Effect of Dropout on DP-SGD Training")

    # Set x-axis to show epsilon values clearly
    epsilon_vals = sorted(dp_data["target_epsilon"].dropna().unique())
    ax.set_xticks(epsilon_vals)
    ax.set_xticklabels([str(int(e)) for e in epsilon_vals])
    ax.set_xlim(min(epsilon_vals) - 0.5, max(epsilon_vals) + 0.5)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Set y-axis limits with padding
    y_min = summary["accuracy_mean"].min() - 0.05
    y_max = min(1.0, summary["accuracy_mean"].max() + 0.03)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="upper left", framealpha=0.95)

    # Add annotation about privacy interpretation
    ax.annotate(
        "← Stronger Privacy",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="#666666",
    )
    ax.annotate(
        "Weaker Privacy →",
        xy=(0.78, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="#666666",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_privacy_utility_tradeoff.png")
    plt.savefig(output_dir / "fig1_privacy_utility_tradeoff.pdf")
    plt.close()

    print("Generated: fig1_privacy_utility_tradeoff.png/pdf")


def plot_epochs_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Figure 2: Training Dynamics Under Differential Privacy

    Shows validation accuracy vs number of training epochs comparing
    models trained with and without dropout under fixed privacy constraints.
    """
    apply_thesis_style()

    # Filter to epochs experiment runs only
    epochs_df = df[df["experiment_type"] == "epochs"].copy()
    if epochs_df.empty:
        print("No epochs experiment runs found.")
        return

    # Compute summary statistics
    summary = compute_summary_statistics(epochs_df, ["num_epochs", "has_dropout"])
    summary = summary.sort_values("num_epochs")

    fig, ax = plt.subplots(figsize=(7, 5))

    for has_dropout, group_label, color, marker in [
        (True, "With Dropout", COLORS["dropout"], MARKERS["dropout"]),
        (False, "Without Dropout", COLORS["no_dropout"], MARKERS["no_dropout"]),
    ]:
        subset = summary[summary["has_dropout"] == has_dropout]
        if subset.empty:
            continue

        ax.errorbar(
            subset["num_epochs"],
            subset["accuracy_mean"],
            yerr=subset["accuracy_ci95"],
            label=group_label,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            linewidth=2,
        )

    ax.set_xlabel("Number of Training Epochs")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Training Dynamics: Effect of Dropout on Convergence with DP-SGD")

    # Set x-axis ticks
    epoch_vals = sorted(summary["num_epochs"].dropna().unique())
    ax.set_xticks(epoch_vals)
    ax.set_xticklabels([str(int(e)) for e in epoch_vals])

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Set y-axis limits with padding
    y_min = summary["accuracy_mean"].min() - 0.02
    y_max = min(1.0, summary["accuracy_mean"].max() + 0.02)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="lower right", framealpha=0.95)

    # Add annotation about fixed epsilon
    # Extract epsilon from the data if available
    eps_info = epochs_df["target_epsilon"].dropna().unique()
    if len(eps_info) == 1:
        eps_text = f"Fixed ε = {int(eps_info[0])}"
    else:
        eps_text = "Fixed privacy budget ε= 1"

    ax.text(
        0.02, 0.98,
        eps_text,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_training_dynamics.png")
    plt.savefig(output_dir / "fig2_training_dynamics.pdf")
    plt.close()

    print("Generated: fig2_training_dynamics.png/pdf")


def write_statistics_report(df: pd.DataFrame, output_dir: Path):
    """Write detailed statistics for both analyses."""
    report_path = output_dir / "analysis_statistics.txt"

    with report_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("MNIST Differential Privacy Exploration - Statistical Summary\n")
        f.write("=" * 70 + "\n\n")

        # Epsilon analysis statistics
        eps_df = df[df["experiment_type"] == "epsilon"]
        if not eps_df.empty:
            f.write("ANALYSIS 1: Privacy-Utility Trade-off (Epsilon Experiment)\n")
            f.write("-" * 70 + "\n\n")

            eps_summary = compute_summary_statistics(eps_df, ["target_epsilon", "has_dropout"])
            eps_summary = eps_summary.sort_values(["has_dropout", "target_epsilon"])

            for has_dropout in [True, False]:
                condition = "With Dropout" if has_dropout else "Without Dropout"
                f.write(f"{condition}:\n")
                subset = eps_summary[eps_summary["has_dropout"] == has_dropout]
                for _, row in subset.iterrows():
                    eps_label = "Non-DP" if pd.isna(row["target_epsilon"]) else f"ε={int(row['target_epsilon'])}"
                    f.write(f"  {eps_label:>8}: {row['accuracy_mean']:.4f} ± {row['accuracy_ci95']:.4f} "
                            f"(n={int(row['accuracy_count'])})\n")
                f.write("\n")

            # Compute dropout benefit
            f.write("Dropout Effect (accuracy difference: with - without):\n")
            for eps in eps_summary["target_epsilon"].unique():
                with_do = eps_summary[(eps_summary["target_epsilon"].eq(eps) | (pd.isna(eps_summary["target_epsilon"]) & pd.isna(eps))) & (eps_summary["has_dropout"] == True)]
                without_do = eps_summary[(eps_summary["target_epsilon"].eq(eps) | (pd.isna(eps_summary["target_epsilon"]) & pd.isna(eps))) & (eps_summary["has_dropout"] == False)]
                if not with_do.empty and not without_do.empty:
                    diff = with_do["accuracy_mean"].values[0] - without_do["accuracy_mean"].values[0]
                    eps_label = "Non-DP" if pd.isna(eps) else f"ε={int(eps)}"
                    sign = "+" if diff >= 0 else ""
                    f.write(f"  {eps_label:>8}: {sign}{diff:.4f} ({sign}{diff*100:.2f}%)\n")
            f.write("\n")

        # Epochs analysis statistics
        epochs_df = df[df["experiment_type"] == "epochs"]
        if not epochs_df.empty:
            f.write("\nANALYSIS 2: Training Dynamics (Epochs Experiment)\n")
            f.write("-" * 70 + "\n\n")

            epochs_summary = compute_summary_statistics(epochs_df, ["num_epochs", "has_dropout"])
            epochs_summary = epochs_summary.sort_values(["has_dropout", "num_epochs"])

            for has_dropout in [True, False]:
                condition = "With Dropout" if has_dropout else "Without Dropout"
                f.write(f"{condition}:\n")
                subset = epochs_summary[epochs_summary["has_dropout"] == has_dropout]
                for _, row in subset.iterrows():
                    f.write(f"  {int(row['num_epochs']):>3} epochs: {row['accuracy_mean']:.4f} ± {row['accuracy_ci95']:.4f} "
                            f"(n={int(row['accuracy_count'])})\n")
                f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("Notes:\n")
        f.write("- Uncertainty bounds represent 95% confidence intervals (t-distribution)\n")
        f.write("- All experiments use DP-SGD with Opacus\n")
        f.write("=" * 70 + "\n")

    print(f"Generated: {report_path.name}")


def load_study_data(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 60,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    raw_path = output_dir / "mnist_exploration_raw.csv"
    df = load_or_fetch_dataframe(
        raw_path,
        refresh=refresh,
        fetch_fn=lambda: fetch_wandb_runs(PROJECT_NAME, entity=entity, timeout=timeout),
        normalize_fn=parse_run_metadata,
    )
    return df[df["state"] == "finished"].copy()


def export_study(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 60,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_study_data(
        refresh=refresh,
        entity=entity,
        timeout=timeout,
        output_dir=output_dir,
    )
    if df.empty:
        print("No finished runs found.")
        return None

    print(f"\nAnalyzing {len(df)} completed runs...")
    print(f"  - Epsilon experiments: {len(df[df['experiment_type'] == 'epsilon'])}")
    print(f"  - Epochs experiments: {len(df[df['experiment_type'] == 'epochs'])}")

    plot_epsilon_analysis(df, output_dir)
    plot_epochs_analysis(df, output_dir)
    write_statistics_report(df, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for MNIST DP exploration."
    )
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/org.")
    parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout.")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from W&B.")
    args = parser.parse_args()

    export_study(refresh=args.refresh, entity=args.entity, timeout=args.timeout)


if __name__ == "__main__":
    main()
