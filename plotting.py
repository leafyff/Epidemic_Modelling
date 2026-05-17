"""Shared solver and plotting helpers used by all model_* functions."""

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import solve_ivp

from constants import (
    COLORS,
    FIGS_DIR,
    FIGURE_DPI,
    LINE_WIDTH,
    PEAK_DOT_COLOR,
    PEAK_DOT_SIZE,
)


def ensure_figs_dir() -> None:
    """Create the output figures directory if it does not exist."""
    os.makedirs(FIGS_DIR, exist_ok=True)


def save_figure(fig: plt.Figure, model_name: str) -> None:
    """Save *fig* to FIGS_DIR/<model_name>_model_ex.png."""
    ensure_figs_dir()
    path = os.path.join(FIGS_DIR, f"{model_name}_model_ex.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")


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
    t: np.ndarray,
    named_curves: dict[str, np.ndarray],
    infected_keys: list[str],
) -> tuple[str, np.ndarray]:
    """Return the name and array of the infected compartment with the highest peak.

    Parameters
    ----------
    named_curves    : mapping of compartment label -> values array (same order as plotted)
    infected_keys   : subset of keys that represent infected compartments

    The function finds which of the *infected_keys* compartments reaches the
    highest value, and returns that (label, array) pair.  The peak dot is then
    placed on that curve, guaranteeing it lies exactly on a plotted line.
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
    """Draw one filled, labelled line per compartment."""
    for label, values in compartments.items():
        color = COLORS[label[0]]   # first letter maps to COLORS key (S/E/I/R/P/N/D)
        ax.fill_between(t, values, alpha=0.08, color=color)
        ax.plot(t, values, label=label, color=color, linewidth=LINE_WIDTH)
