"""All drawing/solving tools shared across every model.

Bundles two responsibilities:
  1. Visual constants: figure size, DPI, line width, output directory,
     compartment colour palette, peak-marker style.
  2. Numerical + plotting helpers: ``solve`` (RK45 wrapper around
     scipy.integrate.solve_ivp), ``plot_lines`` (filled curves with palette),
     ``style_axes`` (shared axis formatting), ``mark_peak`` (red dot on the
     peak of an infected curve), ``dominant_infected_curve`` (peak picker
     for models with multiple infected compartments) and ``save_figure``.

Every per-model file under ``models/`` imports from this module.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

FIGURE_DPI     = 150
FIGURE_SIZE    = (10, 5)
LINE_WIDTH     = 2.2
FIGS_DIR       = "figs"

PEAK_DOT_COLOR = "#D32F2F"   # vivid red for the peak marker
PEAK_DOT_SIZE  = 80          # scatter marker area (points²)

COLORS = {
    "S": "#2196F3",  # blue
    "E": "#FF9800",  # orange
    "I": "#F44336",  # red
    "R": "#4CAF50",  # green
    "P": "#9C27B0",  # purple – Positively infected
    "N": "#E91E63",  # pink   – Negatively infected
    "D": "#795548",  # brown  – Doubtful
}


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_figs_dir() -> None:
    """Create the output figures directory if it does not exist."""
    os.makedirs(FIGS_DIR, exist_ok=True)


def save_figure(fig: plt.Figure, model_name: str) -> None:
    """Save *fig* to FIGS_DIR/<model_name>_model_ex.png."""
    ensure_figs_dir()
    path = os.path.join(FIGS_DIR, f"{model_name}_model_ex.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------

def solve(
    fun: Any,
    y0: list[float],
    t_end: float,
    t_steps: int,
    args: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Run solve_ivp (RK45) and return (time, states) as plain ndarrays."""
    t_eval: np.ndarray = np.linspace(0.0, t_end, t_steps)
    result: Any = solve_ivp(
        fun=fun, t_span=(0.0, t_end), y0=y0,
        args=args, t_eval=t_eval, method="RK45",
    )
    return np.asarray(result.t), np.asarray(result.y)


# ---------------------------------------------------------------------------
# Axes and curve helpers
# ---------------------------------------------------------------------------

def style_axes(ax: plt.Axes, title: str, t_end: float) -> None:
    """Apply shared axes formatting. Only the compartment legend is displayed."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Number of individuals", fontsize=11)
    ax.set_xlim(0, t_end)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mark_peak(
    ax: plt.Axes,
    t: np.ndarray,
    curve: np.ndarray,
) -> tuple[float, float]:
    """Place a red dot at the peak of *curve* and return (peak_t, peak_v).

    The dot coordinates are taken directly from the *curve* array that was
    passed to ax.plot(), so the marker always sits exactly on the line.
    """
    idx    = int(np.argmax(curve))
    peak_t = float(t[idx])
    peak_v = float(curve[idx])
    ax.scatter(peak_t, peak_v, color=PEAK_DOT_COLOR, s=PEAK_DOT_SIZE, zorder=5)
    return peak_t, peak_v


def dominant_infected_curve(
    named_curves: dict[str, np.ndarray],
    infected_keys: list[str],
) -> tuple[str, np.ndarray]:
    """Return the name and array of the infected compartment with the highest peak.

    Used by sentiment-aware models (SEPNS, SEDPNR) that have more than one
    "infected" compartment.  Whichever curve in *infected_keys* reaches the
    largest peak value is returned so the peak dot is placed on that curve,
    guaranteeing it lies exactly on a plotted line.
    """
    best_label  = infected_keys[0]
    best_peak   = float(np.max(named_curves[best_label]))
    for key in infected_keys[1:]:
        candidate = float(np.max(named_curves[key]))
        if candidate > best_peak:
            best_peak  = candidate
            best_label = key
    return best_label, named_curves[best_label]


def plot_lines(
    ax: plt.Axes,
    t: np.ndarray,
    compartments: dict[str, np.ndarray],
) -> None:
    """Draw one filled, labelled line per compartment.

    The colour for each line is looked up from ``COLORS`` using the first
    letter of the compartment label (S, E, I, R, P, N, D).
    """
    for label, values in compartments.items():
        color = COLORS[label[0]]
        ax.fill_between(t, values, alpha=0.08, color=color)
        ax.plot(t, values, label=label, color=color, linewidth=LINE_WIDTH)
