import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use shared thesis style
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual.data import (
    fetch_seeded_runs,
    get_cfg_value,
    load_or_fetch_dataframe,
    normalize_seeded_dataframe,
    summarize_seeded_dataframe,
    to_float,
)
from visual.thesis_style import (
    apply_thesis_style,
    ALPHA_COLORS,
    ALPHA_ORDER,
    ALPHA_LABELS,
    EPS_ORDER as EPS_COLS,
    EPS_LABELS,
    HEATMAP_CMAP,
    HEATMAP_ANNOT_COLOR,
    alpha_sort_key as _alpha_sort_key,
)

PROJECT_NAME = "DP-FedAug-CIFAR10-Study-seeded"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "cifar_dpfedaug_seeded"


# --- UTILS ---

def _to_float(value):
    return to_float(value)


def _get_cfg_value(cfg: dict, *keys):
    return get_cfg_value(cfg, *keys)



# --- DATA FETCHING & PROCESSING ---

def fetch_wandb_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    return fetch_seeded_runs(project, entity=entity, timeout=timeout)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_seeded_dataframe(df)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_seeded_dataframe(df)


# --- PLOTTING HELPERS ---

# Ordered from most non-IID to most IID (imported from thesis_style)
# ALPHA_ORDER, ALPHA_LABELS, ALPHA_COLORS are imported above

# Epsilon columns (imported from thesis_style as EPS_COLS, EPS_LABELS)


# _alpha_sort_key is imported from thesis_style

def _filter_standard_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only non-DP-updates runs (standard FedAug, not updates_dp_enabled)."""
    return df[df["updates_dp_enabled"] == False].copy()


# --- PLOT 1: Accuracy vs Synthetic Count by Alpha (per total_n, per epsilon) ---

def plot_accuracy_vs_synth_by_alpha(summary_df: pd.DataFrame, output_dir: Path):
    apply_thesis_style()
    df = _filter_standard_runs(summary_df)
    if df.empty:
        return

    alphas_present = sorted(df["alpha"].unique(), key=_alpha_sort_key)

    for total_n, n_df in df.groupby("total_n"):
        epsilons = sorted(n_df["target_epsilon_label"].unique(),
                          key=lambda x: EPS_COLS.index(x) if x in EPS_COLS else 999)
        if not epsilons:
            continue

        fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 4.5), sharey=True, squeeze=False)
        axes = axes[0]

        for ax, eps in zip(axes, epsilons):
            eps_df = n_df[n_df["target_epsilon_label"] == eps]
            for alpha in alphas_present:
                a_df = eps_df[eps_df["alpha"] == alpha].sort_values("synthetic_count")
                if a_df.empty:
                    continue
                label = ALPHA_LABELS.get(alpha, alpha)
                color = ALPHA_COLORS.get(alpha, None)
                ax.errorbar(
                    a_df["synthetic_count"], a_df["accuracy_mean"],
                    yerr=a_df["accuracy_std"], marker="o", capsize=3,
                    label=label, color=color, linewidth=1.8, markersize=5,
                )
            eps_title = "No DP" if eps == "none" else f"$\\varepsilon$={eps}"
            ax.set_title(eps_title, fontsize=11)
            ax.set_xlabel("Synthetic samples per client", fontsize=10)
            ax.set_xticks(sorted(eps_df["synthetic_count"].unique()))
            if ax == axes[0]:
                ax.set_ylabel("Accuracy", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")

        fig.suptitle(f"CIFAR-10: Effect of Synthetic Augmentation (N={int(total_n)} total samples)", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / f"synth_benefit_by_alpha_N{int(total_n)}.png", bbox_inches="tight")
        plt.close(fig)


# --- PLOT 2: Accuracy Gain over Baseline (synth=0) ---

def plot_accuracy_delta_vs_synth(summary_df: pd.DataFrame, output_dir: Path):
    apply_thesis_style()
    df = _filter_standard_runs(summary_df)
    df = df[df["target_epsilon_label"] == "none"]
    if df.empty:
        return

    alphas_present = sorted(df["alpha"].unique(), key=_alpha_sort_key)
    total_ns = sorted(df["total_n"].unique())

    fig, axes = plt.subplots(1, len(total_ns), figsize=(5 * len(total_ns), 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, total_n in zip(axes, total_ns):
        n_df = df[df["total_n"] == total_n]
        for alpha in alphas_present:
            a_df = n_df[n_df["alpha"] == alpha].sort_values("synthetic_count")
            if a_df.empty:
                continue
            baseline_row = a_df[a_df["synthetic_count"] == 0]
            if baseline_row.empty:
                continue
            baseline_acc = baseline_row["accuracy_mean"].values[0]
            a_df = a_df.copy()
            a_df["delta"] = a_df["accuracy_mean"] - baseline_acc
            label = ALPHA_LABELS.get(alpha, alpha)
            color = ALPHA_COLORS.get(alpha, None)
            ax.plot(
                a_df["synthetic_count"], a_df["delta"],
                marker="o", label=label, color=color, linewidth=1.8, markersize=5,
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"N={int(total_n)}", fontsize=11)
        ax.set_xlabel("Synthetic samples per client", fontsize=10)
        ax.set_xticks(sorted(n_df["synthetic_count"].unique()))
        if ax == axes[0]:
            ax.set_ylabel("Accuracy change vs. baseline (synth=0)", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("CIFAR-10: Accuracy Gain from Synthetic Augmentation (No DP)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "synth_delta_by_alpha.png", bbox_inches="tight")
    plt.close(fig)


# --- PLOT 3: Combined overview — Accuracy vs Alpha for different synth counts ---

def plot_accuracy_vs_alpha_by_synth(summary_df: pd.DataFrame, output_dir: Path):
    apply_thesis_style()
    df = _filter_standard_runs(summary_df)
    df = df[df["target_epsilon_label"] == "none"]
    if df.empty:
        return

    total_ns = sorted(df["total_n"].unique())
    synth_counts = sorted(df["synthetic_count"].unique())
    cmap = plt.cm.viridis
    synth_colors = {s: cmap(i / max(len(synth_counts) - 1, 1)) for i, s in enumerate(synth_counts)}

    alpha_x = {a: i for i, a in enumerate(ALPHA_ORDER)}
    bar_width = 0.8 / max(len(synth_counts), 1)

    fig, axes = plt.subplots(1, len(total_ns), figsize=(5 * len(total_ns), 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, total_n in zip(axes, total_ns):
        n_df = df[df["total_n"] == total_n]

        for j, sc in enumerate(synth_counts):
            sc_df = n_df[n_df["synthetic_count"] == sc]
            xs, ys, errs = [], [], []
            for a in ALPHA_ORDER:
                row = sc_df[sc_df["alpha"] == a]
                if row.empty:
                    continue
                xs.append(alpha_x[a] + (j - len(synth_counts) / 2) * bar_width)
                ys.append(row["accuracy_mean"].values[0])
                errs.append(row["accuracy_std"].values[0])
            if not xs:
                continue
            label_str = f"synth={int(sc)}" if sc > 0 else "baseline (0)"
            ax.bar(xs, ys, width=bar_width, yerr=errs, capsize=2,
                   label=label_str, color=synth_colors[sc], edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(len(ALPHA_ORDER)))
        ax.set_xticklabels([ALPHA_LABELS.get(a, a) for a in ALPHA_ORDER], fontsize=8, rotation=15)
        ax.set_title(f"N={int(total_n)}", fontsize=11)
        ax.set_xlabel("Non-IID Setting", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("CIFAR-10: Accuracy by Distribution Setting and Synthetic Count (No DP)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_alpha_by_synth.png", bbox_inches="tight")
    plt.close(fig)


# --- PLOT 4: Heatmap — rows=synth_count, cols=epsilon, panels per alpha ---

def plot_heatmap_synth_alpha_dp(summary_df: pd.DataFrame, output_dir: Path):
    df = _filter_standard_runs(summary_df)
    if df.empty:
        return

    apply_thesis_style()

    alpha_cols = [a for a in ALPHA_ORDER if a in df["alpha"].values]

    for total_n, n_df in df.groupby("total_n"):
        alphas_present = [a for a in alpha_cols if a in n_df["alpha"].values]
        if not alphas_present:
            continue

        synth_rows = sorted(n_df["synthetic_count"].unique())
        eps_cols   = [e for e in EPS_COLS if e in n_df["target_epsilon_label"].values]
        n_panels   = len(alphas_present)

        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(7 * n_panels + 1, 4.5),
            squeeze=False,
        )
        axes = axes[0]

        for panel_idx, (ax, alpha) in enumerate(zip(axes, alphas_present)):
            a_df = n_df[n_df["alpha"] == alpha]

            matrix = pd.DataFrame(np.nan, index=synth_rows, columns=eps_cols, dtype=float)
            annot  = pd.DataFrame("",     index=synth_rows, columns=eps_cols, dtype=object)

            for sc in synth_rows:
                for eps in eps_cols:
                    row = a_df[(a_df["synthetic_count"] == sc) & (a_df["target_epsilon_label"] == eps)]
                    if not row.empty:
                        mean = row["accuracy_mean"].values[0]
                        matrix.loc[sc, eps] = mean
                        annot.loc[sc, eps]  = f"{mean:.3f}"

            vmin = np.nanmin(matrix.values)
            vmax = np.nanmax(matrix.values)

            show_cbar = panel_idx == n_panels - 1
            sns.heatmap(
                matrix,
                ax=ax,
                annot=annot,
                fmt="",
                annot_kws={"size": 11, "color": "white", "weight": "bold"},
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                linewidths=0.8,
                linecolor="lightgray",
                mask=matrix.isna(),
                cbar=show_cbar,
                cbar_kws={"label": "Accuracy", "shrink": 0.85, "aspect": 20} if show_cbar else {},
            )

            for text in ax.texts:
                text.set_color("white")
                text.set_fontsize(11)

            ax.set_title(ALPHA_LABELS.get(alpha, alpha), fontsize=11, pad=6)
            ax.set_xlabel("Target epsilon", labelpad=5)
            ax.set_xticklabels([EPS_LABELS.get(e, e) for e in eps_cols], rotation=0)

            if panel_idx == 0:
                ax.set_ylabel("Synthetic count", labelpad=6)
                ax.set_yticklabels([str(int(s)) for s in synth_rows], rotation=0)
            else:
                ax.set_yticks([])
                ax.set_ylabel("")

        fig.suptitle(
            f"CIFAR-10: Mean Accuracy Heatmap  |  N = {int(total_n)}",
            fontsize=13, fontweight="bold", y=1.03,
        )
        fig.tight_layout(w_pad=0.5)
        fig.savefig(
            output_dir / f"heatmap_synth_alpha_dp_N{int(total_n)}.png",
            bbox_inches="tight", dpi=180,
        )
        plt.close(fig)


# --- PLOT 5: DP Impact on Augmented FL ---

def plot_dp_impact(summary_df: pd.DataFrame, output_dir: Path):
    apply_thesis_style()
    df = _filter_standard_runs(summary_df)
    df = df[df["synthetic_count"] == 10]
    if df.empty:
        return

    alphas_present = sorted(df["alpha"].unique(), key=_alpha_sort_key)

    eps_order = ["none", "8", "3", "1"]
    eps_labels = {"none": "No DP", "8": "$\\varepsilon$=8", "3": "$\\varepsilon$=3", "1": "$\\varepsilon$=1"}

    total_ns = sorted(df["total_n"].unique())

    fig, axes = plt.subplots(1, len(total_ns), figsize=(5 * len(total_ns), 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, total_n in zip(axes, total_ns):
        n_df = df[df["total_n"] == total_n]
        for alpha in alphas_present:
            a_df = n_df[n_df["alpha"] == alpha]
            if a_df.empty:
                continue
            eps_present = [e for e in eps_order if e in a_df["target_epsilon_label"].values]
            xs = list(range(len(eps_present)))
            ys = [a_df[a_df["target_epsilon_label"] == e]["accuracy_mean"].values[0] for e in eps_present]
            errs = [a_df[a_df["target_epsilon_label"] == e]["accuracy_std"].values[0] for e in eps_present]
            label = ALPHA_LABELS.get(alpha, alpha)
            color = ALPHA_COLORS.get(alpha, None)
            ax.errorbar(xs, ys, yerr=errs, marker="s", capsize=3,
                        label=label, color=color, linewidth=1.8, markersize=5)
            ax.set_xticks(range(len(eps_present)))
            ax.set_xticklabels([eps_labels.get(e, e) for e in eps_present], fontsize=9)

        ax.set_title(f"N={int(total_n)}", fontsize=11)
        ax.set_xlabel("Privacy Budget", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("CIFAR-10: DP Impact on Augmented FL (synth=10 per client)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "dp_impact_synth10.png", bbox_inches="tight")
    plt.close(fig)


# --- PLOT 6: Updates DP Comparison ---

def plot_updates_dp_comparison(summary_df: pd.DataFrame, output_dir: Path):
    apply_thesis_style()

    std = summary_df[
        (summary_df["updates_dp_enabled"] == False)
        & (summary_df["synthetic_count"] == 0)
        & (summary_df["target_epsilon_label"] == "none")
    ].copy()

    dpup = summary_df[summary_df["updates_dp_enabled"] == True].copy()

    if dpup.empty:
        return

    upd_eps_values = sorted(
        [e for e in dpup["updates_dp_epsilon_label"].unique() if e != "none"],
        key=lambda x: -float(x) if x.replace(".", "").isdigit() else 0,
    )

    total_ns = sorted(dpup["total_n"].unique())
    alphas_present = sorted(dpup["alpha"].unique(), key=_alpha_sort_key)

    n_bars = 1 + len(upd_eps_values)
    bar_width = 0.8 / n_bars

    dpup_colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(upd_eps_values)))

    fig, axes = plt.subplots(
        1, len(total_ns),
        figsize=(5 * len(total_ns), 4.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, total_n in zip(axes, total_ns):
        std_n = std[std["total_n"] == total_n]
        dpup_n = dpup[dpup["total_n"] == total_n]

        x_pos = np.arange(len(alphas_present))

        std_vals, std_errs = [], []
        for alpha in alphas_present:
            row = std_n[std_n["alpha"] == alpha]
            std_vals.append(row["accuracy_mean"].values[0] if not row.empty else 0)
            std_errs.append(row["accuracy_std"].values[0] if not row.empty else 0)

        offset = -(n_bars - 1) / 2 * bar_width
        ax.bar(
            x_pos + offset, std_vals, bar_width,
            yerr=std_errs, capsize=3,
            label="No DP on Updates", color="#2ca02c", edgecolor="white",
        )

        for i, upd_eps in enumerate(upd_eps_values):
            vals, errs = [], []
            for alpha in alphas_present:
                row = dpup_n[
                    (dpup_n["alpha"] == alpha)
                    & (dpup_n["updates_dp_epsilon_label"] == upd_eps)
                ]
                vals.append(row["accuracy_mean"].values[0] if not row.empty else 0)
                errs.append(row["accuracy_std"].values[0] if not row.empty else 0)

            ax.bar(
                x_pos + offset + (i + 1) * bar_width, vals, bar_width,
                yerr=errs, capsize=3,
                label=f"Updates $\\varepsilon$={upd_eps}", color=dpup_colors[i],
                edgecolor="white",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [ALPHA_LABELS.get(a, a) for a in alphas_present],
            fontsize=9, rotation=15,
        )
        ax.set_title(f"N={int(total_n)}", fontsize=11)
        ax.set_xlabel("Non-IID Setting", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        "CIFAR-10: Impact of DP on Model Updates (synth=0, no VAE DP)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "updates_dp_comparison.png", bbox_inches="tight")
    plt.close(fig)


def cleanup_obsolete_plots(output_dir: Path):
    obsolete_patterns = [
        "academic_comparison_*.png",
        "academic_tradeoff_*.png",
        "accuracy_vs_epsilon_*.png",
        "heatmap_accuracy_*.png",
        "accuracy_vs_alpha_by_synth_dp_*.png",
    ]
    for pattern in obsolete_patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)


def load_study_data(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 600,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = output_dir / "cifar_dpfedaug_seeded_raw.csv"
    df = load_or_fetch_dataframe(
        raw_path,
        refresh=refresh,
        fetch_fn=lambda: fetch_wandb_runs(PROJECT_NAME, entity=entity, timeout=timeout),
        normalize_fn=normalize_dataframe,
    )
    summary_df = summarize(df)
    summary_df.to_csv(output_dir / "cifar_dpfedaug_seeded_summary.csv", index=False)
    return df, summary_df


def export_study(
    *,
    refresh: bool = False,
    entity: str | None = None,
    timeout: int = 600,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, summary_df = load_study_data(
        refresh=refresh,
        entity=entity,
        timeout=timeout,
        output_dir=output_dir,
    )

    cleanup_obsolete_plots(output_dir)
    plot_accuracy_vs_synth_by_alpha(summary_df, output_dir)
    plot_accuracy_delta_vs_synth(summary_df, output_dir)
    plot_accuracy_vs_alpha_by_synth(summary_df, output_dir)
    plot_heatmap_synth_alpha_dp(summary_df, output_dir)
    plot_dp_impact(summary_df, output_dir)
    plot_updates_dp_comparison(summary_df, output_dir)

    print(f"CIFAR-10 seeded plots generated in: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    export_study(refresh=args.refresh, entity=args.entity, timeout=args.timeout)


if __name__ == "__main__":
    main()
