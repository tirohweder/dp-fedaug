"""
Unified thesis figure styling for:
"(Provably) Private Federated Augmentation"
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------------------------
# Figure geometry (inches) – tuned for A4 thesis with 1-inch margins
# ---------------------------------------------------------------------------
PAGE_WIDTH = 6.3  # usable text width in inches
FULL_WIDTH = PAGE_WIDTH
HALF_WIDTH = PAGE_WIDTH / 2 - 0.15  # two-column with gap
DEFAULT_ASPECT = 0.65  # height / width

# ---------------------------------------------------------------------------
# Color palettes (Okabe-Ito / Colorblind Friendly Academic Palette)
# ---------------------------------------------------------------------------

# Non-IID heterogeneity levels (Dirichlet alpha)
ALPHA_COLORS = {
    "extreme": "#D55E00",   # Vermillion
    "0.1": "#E69F00",       # Orange
    "1": "#009E73",         # Bluish Green
    "Infinity": "#0072B2",  # Blue
}
ALPHA_ORDER = ["extreme", "0.1", "1", "Infinity"]
ALPHA_LABELS = {
    "extreme": "Pathological",
    "0.1": r"$\alpha=0.1$",
    "1": r"$\alpha=1$",
    "Infinity": r"$\alpha=\infty$",
}

# Privacy budget (epsilon) - Gradient from dark to light
EPSILON_COLORS = {
    "1": "#332288",    # Dark Indigo
    "3": "#882255",    # Wine
    "8": "#CC6677",    # Rose
    "none": "#88CCEE", # Cyan
}
EPS_ORDER = ["none", "8", "3", "1"]
EPS_LABELS = {
    "none": "No DP",
    "8": r"$\varepsilon=8$",
    "3": r"$\varepsilon=3$",
    "1": r"$\varepsilon=1$",
}

# Dropout comparison
DROPOUT_COLORS = {
    "with": "#0072B2",     # Blue
    "without": "#D55E00",  # Vermillion
}
DROPOUT_MARKERS = {
    "with": "o",
    "without": "s",
}

# General-purpose academic palette
COLORS = {
    "primary": "#0072B2",
    "secondary": "#009E73",
    "tertiary": "#E69F00",
    "quaternary": "#D55E00",
    "success": "#009E73",
    "neutral": "#999999",
}

# FedProx mu values
MU_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]

# Synthetic count gradient
SYNTH_CMAP = "mako" # Seaborn's mako or rocket are much more modern than viridis

# Heatmap settings
HEATMAP_CMAP = "Blues" # Cleaner than viridis for academic papers
HEATMAP_ANNOT_COLOR = "#111111"

# ---------------------------------------------------------------------------
# rcParams — call apply_thesis_style() once at script start
# ---------------------------------------------------------------------------

def apply_thesis_style():
    """Set matplotlib rcParams for publication-quality thesis figures."""
    plt.rcParams.update({
        # Display / export
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,

        # Font - Using Serif to match LaTeX body text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "text.usetex": False, # Set to True if you have a local LaTeX installation for perfect math rendering

        # Axes
        "axes.titlesize": 11,
        "axes.titleweight": "normal", # Bold titles look a bit dated; normal is cleaner
        "axes.labelsize": 10,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.grid": True,
        "axes.grid.axis": "y", # Only horizontal gridlines
        "axes.axisbelow": True,

        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grid
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "grid.color": "#dddddd",
        "grid.linestyle": "--",

        # Legend
        "legend.fontsize": 9,
        "legend.frameon": False, # Frameless legends are standard in top-tier papers
        "legend.borderpad": 0.4,

        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "errorbar.capsize": 0, # Removing caps makes error bars look much more modern

        # Hatch / patch
        "patch.linewidth": 0.5,
        "patch.edgecolor": "none", # Remove heavy black borders from bars/boxes
    })

# ... [Keep get_fig, get_wide_fig, save_thesis_fig, format_accuracy_axis, add_privacy_arrow, remove_all_spines exactly as they were] ...

def synth_count_colors(counts):
    """Return a dict mapping each synthetic count to a modern colormap."""
    # Updated to use modern matplotlib colormap API
    cmap = mpl.colormaps[SYNTH_CMAP]
    n = max(len(counts) - 1, 1)
    return {c: cmap(0.2 + 0.6 * (i / n)) for i, c in enumerate(sorted(counts))} # Pad edges so it's not too dark or too light

def alpha_sort_key(a: str) -> int:
    return ALPHA_ORDER.index(a) if a in ALPHA_ORDER else 999


def get_fig(width="single", aspect=DEFAULT_ASPECT, nrows=1, ncols=1, **kwargs):
    """Create a figure with thesis-appropriate dimensions.

    Parameters
    ----------
    width : str or float
        "single" (~6.3"), "half" (~3"), "full" (alias for single),
        or a numeric width in inches.
    aspect : float
        height = width * aspect  (per subplot column).
    nrows, ncols : int
        Subplot grid dimensions.
    **kwargs : dict
        Extra arguments forwarded to plt.subplots.

    Returns
    -------
    fig, axes
    """
    if isinstance(width, str):
        w = {"single": FULL_WIDTH, "full": FULL_WIDTH, "half": HALF_WIDTH}[width]
    else:
        w = float(width)

    h = w * aspect * (nrows / ncols) if ncols > 1 else w * aspect
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


def get_wide_fig(n_panels, panel_width=3.0, height=3.5, constrain_width=True, **kwargs):
    """Create a wide multi-panel figure (e.g. faceted by epsilon or alpha).

    Parameters
    ----------
    n_panels : int
        Number of side-by-side panels.
    panel_width : float
        Width per panel in inches.
    height : float
        Total figure height in inches.
    constrain_width : bool
        If True (default), cap total width at PAGE_WIDTH.
        Set False for figures that need more room.

    Returns
    -------
    fig, axes (1-D array)
    """
    w = panel_width * n_panels
    if constrain_width:
        w = min(w, PAGE_WIDTH)
    fig, axes = plt.subplots(1, n_panels, figsize=(w, height),
                             squeeze=False, **kwargs)
    return fig, axes[0]


def save_thesis_fig(fig, name: str, output_dir: str | Path):
    """Save figure as both PDF and PNG with thesis-quality settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename stem (without extension), e.g. "fig_5_1_privacy_utility".
    output_dir : str or Path
        Directory to save into (created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{name}.{ext}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis formatting helpers
# ---------------------------------------------------------------------------

def format_accuracy_axis(ax, decimals=0):
    """Format y-axis as percentage (assumes values in [0, 1])."""
    ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1.0, decimals=decimals)
    )


def add_privacy_arrow(ax, y_frac=0.02):
    """Add '← Stronger Privacy / Weaker Privacy →' annotation."""
    ax.annotate(
        r"$\leftarrow$ Stronger Privacy",
        xy=(0.02, y_frac), xycoords="axes fraction",
        fontsize=8, color="#888888",
    )
    ax.annotate(
        r"Weaker Privacy $\rightarrow$",
        xy=(0.72, y_frac), xycoords="axes fraction",
        fontsize=8, color="#888888", ha="left",
    )


def remove_all_spines(ax):
    """Remove all four spines (useful for image grids)."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


