"""Render a previously saved JSON sample as a matplotlib figure.

Reads a JSON sample produced by ``sampling.create_sample`` and plots
its compartments versus time, without re-running the ODE or comparing
the trajectory against the model it came from. The figure is saved to
``figs/<sample-stem>_sample.png`` via the shared ``drawing.save_figure``
helper and returned to the caller.

The JSON layout consumed here is the one written by
``sampling.create_sample``:

    {
        "model"       : "<MODEL NAME>",
        "params"      : { ... },
        "n_points"    : 1000,
        "time"        : [t0, t1, ...],
        "compartments": { "S": [...], "I": [...], ... }
    }
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from drawing import FIGURE_DPI, FIGURE_SIZE, plot_lines, save_figure, style_axes
from estimation import load_sample


def plot_sample(path: str) -> plt.Figure:
    """Load a JSON sample at *path* and draw it; return the matplotlib Figure.

    The compartments are drawn with the project palette (blue S, orange E,
    red I, etc.). No model is re-run and no comparison is performed — the
    plot reflects exactly the arrays stored in the JSON file.
    """
    sample = load_sample(path)
    model = sample["model"]
    t = np.asarray(sample["time"], dtype=float)
    compartments = {
        name: np.asarray(values, dtype=float)
        for name, values in sample["compartments"].items()
    }
    n_points = sample["n_points"]
    population = float(sample["params"]["population"])
    t_end = float(t[-1])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    style_axes(
        ax,
        f"{model} sample  |  N={population:,.0f}  |  {n_points} points",
        t_end,
    )
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = save_figure(fig, stem, suffix="_sample")

    print(f"\n--- Sample plot: {model} ---")
    print(f"  Source file        : {path}")
    print(f"  Compartments       : {', '.join(compartments)}")
    print(f"  Points             : {len(t)}")
    print(f"  Time range         : 0 -> {t_end}")
    print(f"  Saved figure       : {out_path}")
    return fig
