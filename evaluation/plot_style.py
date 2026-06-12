"""
Shared matplotlib styling for evaluation / paper figures.

Used by ipa_analysis.py, generate_rq6_mu_scaling.py, and related scripts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Typography (paper figures)
FS_LABEL = 16
FS_TITLE = 15
FS_TICK = 15
FS_LEGEND = 15
FS_LEGEND_COMPACT = 10
FS_ANNOT = 10.5
FS_SUPTITLE = 12
FS_SUBPLOT_TITLE = 12

# Layout
FIG_MODEL_SCALING = (9, 5.5)
FIG_MU_SCALING = (9, 5.5)
FIG_AUC_GAMMA_EPOCHS = (12, 5)
FIG_STANDARD = (8, 5)
FIG_CROSS_MODEL = (11, 5.5)

# Pythia model-size axis
PARAM_TICKS = [70, 410, 1400, 2800]
PYTHIA_SIZE_LABELS = ["Pythia-70M", "Pythia-410M", "Pythia-1.4B", "Pythia-2.8B"]
PARAM_LABELS_SHORT = ["70M", "410M", "1.4B", "2.8B"]

# Series colors (members / non-members / gap)
COLOR_MEMBER = "#c0392b"
COLOR_NONMEMBER = "#27ae60"
COLOR_GAP_FILL = "#7f8c8d"
COLOR_ANNOT = "#555555"

# Grid & lines
GRID_ALPHA = 0.3
GRID_LW = 0.6
LINE_WIDTH = 2.2
MARKER_SIZE = 8
LEGEND_FRAME_ALPHA = 0.9

# Print-friendly bar styling (grayscale + hatch)
BAR_EDGE = "black"
BAR_MEMBER_STYLE = {"facecolor": "#2b2b2b", "hatch": ""}
BAR_NONMEMBER_STYLE = {"facecolor": "#9a9a9a", "hatch": "///"}
# Two-set Venn diagrams (PEARL vs one MIA)
VENN_PEARL_STYLE = {"facecolor": "#d9d9d9", "edgecolor": "#1a1a1a", "hatch": "", "lw": 2.0}
VENN_MIA_STYLE = {"facecolor": "#f0f0f0", "edgecolor": "#4d4d4d", "hatch": "///", "lw": 2.0}
# Four-set unified Venn (PEARL + three MIAs) — color fills for the large combined figure
VENN_FOUR_SET_STYLES = [
    {"facecolor": "#1f77b4", "edgecolor": "#1f77b4", "hatch": "", "label": "PEARL", "alpha": 0.38},
    {"facecolor": "#e377c2", "edgecolor": "#e377c2", "hatch": "", "label": "MIA-loss", "alpha": 0.38},
    {"facecolor": "#17becf", "edgecolor": "#17becf", "hatch": "", "label": "MIA-min-K", "alpha": 0.38},
    {"facecolor": "#bcbd22", "edgecolor": "#bcbd22", "hatch": "", "label": "MIA-neighborhood", "alpha": 0.38},
]
# PEARL + MIA methods in cross-model grouped bar charts
CROSS_MODEL_BAR_STYLES = [
    {"facecolor": "#1a1a1a", "hatch": ""},
    {"facecolor": "#4d4d4d", "hatch": "///"},
    {"facecolor": "#808080", "hatch": "\\\\"},
    {"facecolor": "#b3b3b3", "hatch": "xxx"},
]

# Model-size x-axis tick styling
X_TICK_ROTATION = 15


def apply_paper_rcparams() -> None:
    """Set matplotlib rcParams so implicit font sizes match paper style."""
    plt.rcParams.update(
        {
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_TITLE,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
        }
    )


def style_grid(ax: plt.Axes, *, axis: str = "both") -> None:
    ax.grid(True, alpha=GRID_ALPHA, lw=GRID_LW, axis=axis)


def configure_model_size_xaxis(
    ax: plt.Axes,
    *,
    xlabel: str = "Model size",
    short_labels: bool = False,
) -> None:
    """Log-scale x-axis with consistent tick labels and orientation."""
    ax.set_xscale("log")
    ax.set_xticks(PARAM_TICKS)
    labels = PARAM_LABELS_SHORT if short_labels else PYTHIA_SIZE_LABELS
    ax.set_xticklabels(
        labels,
        rotation=X_TICK_ROTATION,
        ha="right",
        fontsize=FS_TICK,
    )
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    style_grid(ax)


def set_axis_labels(
    ax: plt.Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FS_TITLE)


def cross_model_bar_kwargs(series_index: int) -> dict:
    """Grayscale fill + hatch for grouped cross-model bars."""
    style = CROSS_MODEL_BAR_STYLES[series_index % len(CROSS_MODEL_BAR_STYLES)]
    return {**style, "edgecolor": BAR_EDGE, "linewidth": 0.8}


def legend(
    ax: plt.Axes,
    *,
    compact: bool = False,
    **kwargs,
) -> None:
    defaults = {
        "fontsize": FS_LEGEND_COMPACT if compact else FS_LEGEND,
        "framealpha": LEGEND_FRAME_ALPHA,
    }
    defaults.update(kwargs)
    ax.legend(**defaults)
