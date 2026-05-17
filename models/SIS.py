"""
================================================================================
SIS MODEL — Susceptible / Infected / Susceptible
================================================================================

The SIS model extends SI with *recovery without immunity*: infected
individuals heal and return directly to the Susceptible pool, where they may
be re-infected.  This makes SIS the canonical model for diseases (or
behaviours) that confer no lasting protection, such as the common cold,
many sexually transmitted infections, and recurrent adoption phenomena.

Compartments:

    S – Susceptible : healthy individuals capable of being (re)infected.
    I – Infected    : currently infectious; recovers to S at rate gamma.

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

    dS/dt = -beta * S * I / N + gamma * I
    dI/dt = +beta * S * I / N - gamma * I

Conservation:  S(t) + I(t) = N for all t.

Parameters
    beta  : transmission rate.   Units: 1 / time.
    gamma : recovery rate.       Mean infectious period = 1 / gamma.

--------------------------------------------------------------------------------
Basic reproduction number and endemic equilibrium
--------------------------------------------------------------------------------

    R0 = beta / gamma

Two regimes separated by R0:

    R0 < 1 : the disease-free equilibrium I* = 0 is globally stable; every
             epidemic eventually dies out.
    R0 > 1 : the disease-free equilibrium is unstable, and the system
             converges to the endemic equilibrium

                 I* / N = 1 - 1/R0,     S* / N = 1/R0.

When R0 > 1 the plot shows a dotted horizontal line at the endemic level
I*, computed at plot time.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Common cold, gonorrhoea and other recurrent infections.
* Information / rumour propagation where individuals can lose interest and
  later be re-exposed.
* Any system where the long-term persistence of an infection matters more
  than the shape of a single outbreak peak.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

from drawing import (
    COLORS,
    FIGURE_DPI,
    FIGURE_SIZE,
    mark_peak,
    plot_lines,
    save_figure,
    solve,
    style_axes,
)


@dataclass
class SISParams:
    """Parameters for the SIS (Susceptible-Infected-Susceptible) model."""
    population      : int   = 10_000  # total number of individuals in the network
    initial_infected: int   = 10      # number of infected individuals at t=0
    beta            : float = 0.30    # S -> I   transmission rate
    gamma           : float = 0.10    # I -> S   recovery rate (no permanent immunity)
    t_end           : float = 120.0   # simulation end time (days)
    t_steps         : int   = 1_000   # number of equally-spaced time points


def sis_ode(
    _t: float, y: list[float],
    beta: float, gamma: float, N: float,
) -> list[float]:
    """SIS model ODEs. Recovered individuals re-enter Susceptible (no immunity).

    dS/dt = -beta * S * I / N  +  gamma * I
    dI/dt = +beta * S * I / N  -  gamma * I
    """
    S, I = y
    new_infected = beta * S * I / N
    recoveries   = gamma * I
    return [-new_infected + recoveries, +new_infected - recoveries]


def model_sis(params: SISParams) -> plt.Figure:
    """Simulate and plot the SIS model. Recovered individuals re-enter the Susceptible pool."""
    print("\n--- SIS Model ---")
    N  = float(params.population)
    I0 = float(params.initial_infected)
    S0 = N - I0
    r0 = params.beta / params.gamma

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial infected             : {I0:,.0f}")
    print(f"  beta  (transmission rate)    : {params.beta}")
    print(f"  gamma (recovery rate)        : {params.gamma}")
    print(f"  R0 = beta / gamma            : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        sis_ode, [S0, I0], params.t_end, params.t_steps,
        (params.beta, params.gamma, N),
    )
    S, I = y

    compartments = {"Susceptible": S, "Infected": I}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    peak_t, peak_v = mark_peak(ax, t, compartments["Infected"])

    if r0 > 1.0:
        endemic_I = N * (1.0 - 1.0 / r0)
        ax.axhline(
            endemic_I, color=COLORS["I"], linestyle=":", linewidth=1.2, alpha=0.7,
            label=f"Endemic I* = {endemic_I:,.0f}",
        )
        ax.legend(fontsize=10, framealpha=0.9)

    style_axes(
        ax,
        fr"SIS Model  |  N={N:,.0f}  |  $\beta$={params.beta}  |  $\gamma$={params.gamma}",
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SIS")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SIS model simulation complete.")
    return fig
