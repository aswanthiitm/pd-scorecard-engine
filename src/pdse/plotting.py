"""Shared matplotlib styling so every figure in the report reads as one deck."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#1b2733"
MUTED = "#6b7a89"
GRID = "#dfe5ea"
ACCENT = "#0b6e99"
ACCENT2 = "#c8553d"
ACCENT3 = "#3f8f5b"
ACCENT4 = "#8a6bbf"
SERIES = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#b8860b", MUTED]


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 190,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "lines.linewidth": 1.7,
    })


def annotate(ax, text: str, *, loc: str = "upper left") -> None:
    """A small caption inside the axes stating what the reader should take away."""
    xy = {"upper left": (0.015, 0.965), "lower left": (0.015, 0.04),
          "lower right": (0.985, 0.04), "upper right": (0.985, 0.965)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "bottom" if "lower" in loc else "top"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.2,
            color=MUTED, linespacing=1.35)
