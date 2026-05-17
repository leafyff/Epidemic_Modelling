"""
================================================================================
SI MODEL — Susceptible / Infected
================================================================================

The SI model is the simplest possible compartmental epidemic model. It
divides a closed population of N individuals into two compartments:

    S  – Susceptible : currently healthy individuals capable of being infected.
    I  – Infected    : currently infectious individuals. Once infected, an
                       individual remains infected forever — there is no
                       recovery and no immunity in this model.

Because there is no exit from the I compartment, infection is *irreversible*:
the number of infected can only grow, and provided at least one infected
individual exists at t = 0, every susceptible eventually becomes infected.

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

The classical mean-field SI dynamics are:

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N

with the conservation law S(t) + I(t) = N for all t.

Parameters
    beta : transmission rate, i.e. effective contact rate × probability of
           transmission per contact.  Units: 1 / time.

--------------------------------------------------------------------------------
Analytical behaviour
--------------------------------------------------------------------------------

Let i(t) = I(t)/N be the infected fraction.  Then i obeys the logistic
equation di/dt = beta * i * (1 - i), with closed-form solution

    i(t) = i0 / (i0 + (1 - i0) * exp(-beta * t))

i.e. infection grows exponentially when i is small and saturates at 1 as
t → ∞.  There is no basic reproduction number threshold (R0) because the
disease never dies out: every outbreak with i0 > 0 reaches i = 1.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Diseases or behaviours where no meaningful recovery exists on the
  timescale of interest (e.g. lifelong infections, persistent adoption).
* A first-order qualitative sanity check before fitting richer models.
* Pedagogical baseline against which SIR / SIS / SEIR can be compared.
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
class SIParams:
    """Parameters for the SI (Susceptible-Infected) model."""
    population      : int   = 10_000  # total number of individuals in the network
    initial_infected: int   = 10      # number of infected individuals at t=0
    beta            : float = 0.30    # S -> I   transmission rate
    t_end           : float = 60.0    # simulation end time (days)
    t_steps         : int   = 1_000   # number of equally-spaced time points


def si_ode(_t: float, y: list[float], beta: float, N: float) -> list[float]:
    """SI model ODEs.

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N
    """
    S, I = y
    new_infected = beta * S * I / N
    return [-new_infected, +new_infected]


def model_si(params: SIParams) -> plt.Figure:
    """Simulate and plot the SI model. Nobody recovers; all eventually become infected."""
    print("\n--- SI Model ---")
    N  = float(params.population)
    I0 = float(params.initial_infected)
    S0 = N - I0

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial infected             : {I0:,.0f}")
    print(f"  beta  (transmission rate)    : {params.beta}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(si_ode, [S0, I0], params.t_end, params.t_steps, (params.beta, N))
    S, I = y

    compartments = {"Susceptible": S, "Infected": I}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    # SI infection rises monotonically; the peak is the final value.
    peak_t, peak_v = mark_peak(ax, t, compartments["Infected"])

    style_axes(
        ax,
        fr"SI Model  |  N={N:,.0f}  |  $\beta$={params.beta}",
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SI")
    print(f"  Final infected               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SI model simulation complete.")
    return fig
