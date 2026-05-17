"""
================================================================================
SEIR MODEL — Susceptible / Exposed / Infected / Recovered
================================================================================

The SEIR model refines SIR by inserting an *Exposed* (latent) compartment
between Susceptible and Infected. Exposed individuals have been infected
but are not yet infectious — they are in the incubation period. This makes
SEIR the appropriate model for diseases with a non-negligible latency,
such as influenza, smallpox, Ebola and COVID-19.

Compartments:

    S – Susceptible : healthy individuals who can be infected.
    E – Exposed    : infected but NOT YET infectious (latent / incubating).
    I – Infected   : currently infectious individuals.
    R – Recovered  : recovered, permanently immune.

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

    dS/dt = -beta  * S * I / N
    dE/dt = +beta  * S * I / N  -  sigma * E
    dI/dt = +sigma * E          -  gamma * I
    dR/dt = +gamma * I

Conservation:  S(t) + E(t) + I(t) + R(t) = N for all t.

Parameters
    beta  : transmission rate. Units: 1 / time.
    sigma : rate of progression from Exposed to Infected.
            Mean incubation period = 1 / sigma  (reported by the simulator).
    gamma : recovery rate.
            Mean infectious period = 1 / gamma.

--------------------------------------------------------------------------------
Basic reproduction number
--------------------------------------------------------------------------------

For the SEIR model, R0 has the same form as SIR:

    R0 = beta / gamma

(The presence of the Exposed compartment changes the *timing* of the
outbreak — pushing the peak later and making the early-growth phase look
sub-exponential — but does not change the eventual epidemic threshold.)

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Influenza and influenza-like illnesses with a 1–3-day incubation period.
* Ebola, smallpox and other diseases with several-day latency.
* COVID-19 modelling at the population level.
* Any disease where ignoring incubation would over-predict the speed of
  the epidemic.
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
class SEIRParams:
    """Parameters for the SEIR (Susceptible-Exposed-Infected-Recovered) model."""
    population      : int   = 10_000  # total number of individuals in the network
    initial_exposed : int   = 0       # number of exposed (latent) individuals at t=0
    initial_infected: int   = 10      # number of infected individuals at t=0
    beta            : float = 0.30    # S -> E   transmission / contact rate
    sigma           : float = 0.20    # E -> I   incubation rate  (1/sigma = mean incubation days)
    gamma           : float = 0.10    # I -> R   recovery rate (permanent immunity)
    t_end           : float = 200.0   # simulation end time (days)
    t_steps         : int   = 1_000   # number of equally-spaced time points


def seir_ode(
    _t: float, y: list[float],
    beta: float, sigma: float, gamma: float, N: float,
) -> list[float]:
    """SEIR model ODEs. Adds a latent Exposed compartment before Infected.

    dS/dt = -beta * S * I / N
    dE/dt = +beta * S * I / N  -  sigma * E
    dI/dt = +sigma * E          -  gamma * I
    dR/dt = +gamma * I

    Mean incubation period = 1 / sigma days.
    """
    S, E, I, _R = y
    new_exposed  = beta * S * I / N
    new_infected = sigma * E
    recoveries   = gamma * I
    return [
        -new_exposed,
        +new_exposed  - new_infected,
        +new_infected - recoveries,
        +recoveries,
    ]


def model_seir(params: SEIRParams) -> plt.Figure:
    """Simulate and plot the SEIR model (SIR + latent Exposed compartment)."""
    print("\n--- SEIR Model ---")
    N               = float(params.population)
    E0              = float(params.initial_exposed)
    I0              = float(params.initial_infected)
    S0              = N - E0 - I0
    r0              = params.beta / params.gamma
    mean_incubation = 1.0 / params.sigma

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial exposed              : {E0:,.0f}")
    print(f"  Initial infected             : {I0:,.0f}")
    print(f"  beta  (transmission rate)    : {params.beta}")
    print(f"  sigma (incubation rate)      : {params.sigma}  ->  mean incubation = {mean_incubation:.1f} days")
    print(f"  gamma (recovery rate)        : {params.gamma}")
    print(f"  R0 = beta / gamma            : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        seir_ode, [S0, E0, I0, 0.0], params.t_end, params.t_steps,
        (params.beta, params.sigma, params.gamma, N),
    )
    S, E, I, R = y

    compartments = {"Susceptible": S, "Exposed": E, "Infected": I, "Recovered": R}

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    peak_t, peak_v = mark_peak(ax, t, compartments["Infected"])

    style_axes(
        ax,
        (fr"SEIR Model  |  N={N:,.0f}  |  $\beta$={params.beta}"
         fr"  |  $\sigma$={params.sigma}  |  $\gamma$={params.gamma}"),
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SEIR")
    print(f"  Mean incubation period       : {mean_incubation:.1f} days")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SEIR model simulation complete.")
    return fig
