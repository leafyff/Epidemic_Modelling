"""
================================================================================
SIR MODEL — Susceptible / Infected / Recovered
================================================================================

The SIR model is the classical Kermack–McKendrick (1927) compartmental
model. It adds a third compartment to SI in which recovered individuals
acquire *permanent immunity* and therefore never return to the susceptible
pool. SIR is the standard starting point for modelling acute infectious
diseases that confer lasting immunity (measles, mumps, rubella, etc.).

Compartments:

    S – Susceptible : healthy individuals who can be infected.
    I – Infected   : currently infectious individuals.
    R – Recovered  : individuals who have recovered AND are permanently immune.
                     (R also commonly stands for "removed", which includes
                     deaths in some formulations.)

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N - gamma * I
    dR/dt = +gamma * I

Conservation:  S(t) + I(t) + R(t) = N for all t.

Parameters
    beta  : transmission rate.   Units: 1 / time.
    gamma : recovery rate.       Mean infectious period = 1 / gamma.

--------------------------------------------------------------------------------
Basic reproduction number and epidemic threshold
--------------------------------------------------------------------------------

    R0 = beta / gamma

* R0 < 1 : no epidemic. I(t) decreases monotonically from I(0).
* R0 > 1 : I(t) initially grows, reaches a peak, then decays as S falls
           below the threshold S_threshold = N / R0.

Final size of the epidemic, s_inf = S(∞)/N, satisfies the transcendental
equation

    s_inf = exp(-R0 * (1 - s_inf))

so a fraction 1 - s_inf of the population is eventually infected.  The
herd-immunity threshold (the susceptible fraction required to prevent
sustained transmission) is 1/R0.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Childhood diseases conferring lifelong immunity (measles, mumps, rubella).
* First-pass modelling of acute viral outbreaks (early COVID-19 work,
  influenza pandemics) when latency and exposure can be neglected.
* Estimating epidemic final size and herd-immunity thresholds.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

from drawing import (
    FIGURE_DPI,
    FIGURE_SIZE,
    mark_peak,
    plot_lines,
    save_figure,
    solve,
    style_axes,
)


@dataclass
class SIRParams:
    """Parameters for the SIR (Susceptible-Infected-Recovered) model."""
    population        : int   = 10_000  # total number of individuals in the network
    initial_infected  : int   = 10      # number of infected individuals at t=0
    initial_recovered : int   = 0       # number of recovered (immune) individuals at t=0
    beta              : float = 0.30    # S -> I   transmission rate
    gamma             : float = 0.10    # I -> R   recovery rate (permanent immunity)
    t_end             : float = 160.0   # simulation end time (days)
    t_steps           : int   = 1_000   # number of equally-spaced time points


def sir_ode(
    _t: float, y: list[float],
    beta: float, gamma: float, N: float,
) -> list[float]:
    """SIR model ODEs. Recovered individuals gain permanent immunity.

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N  -  gamma * I
    dR/dt = +gamma * I

    Epidemic threshold: R0 = beta / gamma > 1.
    """
    S, I, _R = y
    new_infected = beta * S * I / N
    recoveries   = gamma * I
    return [-new_infected, +new_infected - recoveries, +recoveries]


def model_sir(params: SIRParams) -> plt.Figure:
    """Simulate and plot the SIR model"""
    print("\n--- SIR Model ---")
    N      = float(params.population)
    I0     = float(params.initial_infected)
    R0_val = float(params.initial_recovered)
    S0     = N - I0 - R0_val
    r0     = params.beta / params.gamma

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial infected             : {I0:,.0f}")
    print(f"  Initial recovered            : {R0_val:,.0f}")
    print(f"  beta  (transmission rate)    : {params.beta}")
    print(f"  gamma (recovery rate)        : {params.gamma}")
    print(f"  R0 = beta / gamma            : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        sir_ode, [S0, I0, R0_val], params.t_end, params.t_steps,
        (params.beta, params.gamma, N),
    )
    S, I, R = y

    compartments = {"Susceptible": S, "Infected": I, "Recovered": R}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    peak_t, peak_v = mark_peak(ax, t, compartments["Infected"])

    style_axes(
        ax,
        fr"SIR Model  |  N={N:,.0f}  |  $\beta$={params.beta}  |  $\gamma$={params.gamma}",
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SIR")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SIR model simulation complete.")
    return fig
