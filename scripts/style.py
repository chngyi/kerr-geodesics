"""Shared plotting style for the validation figures.

The categorical palette below was checked with a colour-vision-deficiency
validator: every adjacent pair separates by dE >= 9 under deuteranopia and
>= 25 under normal vision, all five sit inside a common lightness band, and all
clear 3:1 contrast against the figure surface.  Series are also distinguished
by dash pattern and direct labels, so identity never rests on colour alone.

Sequential quantities (impact parameter, step size) use a single-hue ramp
light-to-dark rather than the categorical set -- magnitude is an ordered
quantity and deserves an ordered encoding.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Categorical: fixed order, never cycled.
CAT = ["#2b6fd6", "#d4640a", "#009e60", "#b5379a", "#8a6a1c"]

# Secondary encoding, so series identity never rests on colour alone.
# Matplotlib linestyle tuples, in the same fixed order as CAT.
SOLID = "-"
DASH = (0, (5, 2))
DOT = (0, (1.5, 1.5))
DASHDOT = (0, (7, 2, 1.5, 2))
FINEDOT = (0, (2, 2))
STYLES = [SOLID, DASH, DOT, DASHDOT, FINEDOT]

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6f76"
GRID = "#e3e3e0"
HOLE = "#2a2a28"          # the black hole itself, in trajectory plots


def sequential(n, lo=0.20, hi=0.95):
    """n steps of a single-hue blue ramp, light to dark."""
    base = mpl.colors.to_rgb("#0a2f66")
    return [mpl.colors.to_hex(tuple(1 - t * (1 - c) for c in base))
            for t in np.linspace(lo, hi, n)]


def use():
    """Apply the house style."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",

        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 9,
        "axes.labelcolor": INK,
        "text.color": INK,

        # Recessive chrome: the data should be the darkest thing on the page.
        "axes.edgecolor": "#c9c9c5",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,

        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.0,

        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.handlelength": 2.4,
        "legend.borderpad": 0.0,
        "legend.labelspacing": 0.45,
    })


def label_line(ax, x, y, text, color, **kw):
    """Direct label on a line, in the line's colour."""
    ax.annotate(text, (x, y), color=color, fontsize=8.5,
                fontweight="medium", **kw)


def draw_hole(ax, r_horizon, r_photon=None, r_ergo=None):
    """The horizon as a filled disc, plus optional photon sphere / ergosphere.

    A 2px surface-coloured ring separates the filled disc from any trajectory
    that grazes it, so overlapping marks stay legible.
    """
    import matplotlib.patches as mpatches

    ax.add_patch(mpatches.Circle((0, 0), r_horizon, facecolor=HOLE,
                                 edgecolor=SURFACE, linewidth=2.0, zorder=5))
    if r_photon is not None:
        ax.add_patch(mpatches.Circle((0, 0), r_photon, fill=False,
                                     edgecolor=INK_MUTED, linewidth=1.1,
                                     linestyle=(0, (4, 3)), zorder=4))
    if r_ergo is not None:
        ax.add_patch(mpatches.Circle((0, 0), r_ergo, fill=False,
                                     edgecolor=CAT[1], linewidth=1.1,
                                     linestyle=(0, (2, 2)), zorder=4))

