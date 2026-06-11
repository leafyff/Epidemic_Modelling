"""
================================================================================
SEDIS MODEL — Susceptible / Exposed / Doubtful / Infected / Susceptible
================================================================================

The SEDIS model is a rumour-propagation model that inserts a *Doubtful*
(D) compartment between Exposed and Infected. The Doubtful state captures
the human fact-checking step: after being exposed to a rumour, many
individuals do not immediately accept or reject it but enter a period of
uncertainty during which they evaluate plausibility, cross-check sources,
or simply hesitate.

Because rumours on social media confer no permanent immunity, every
non-susceptible compartment can decay back to S — exposed individuals can
reject the rumour outright, doubtful individuals can decide it is false
and disengage, and spreaders eventually lose interest.

Compartments:

    S – Susceptible : not currently aware of or spreading the rumour.
    E – Exposed    : has just encountered the rumour.
    D – Doubtful   : uncertain; weighing evidence (fact-checking, hesitating).
    I – Infected   : has accepted the rumour and is actively spreading it.

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

Let lambda = alpha * S be the exposure rate (per-S "leakage" form used in
Govindankutty & Gopalan 2024; the rate at which a susceptible becomes
exposed does not depend on the current number of spreaders):

    dS/dt = -lambda + mu1*E + mu2*D + mu3*I
    dE/dt = +lambda - (beta1 + beta2 + mu1) * E
    dD/dt = +beta1 * E - (gamma + mu2) * D
    dI/dt = +beta2 * E + gamma * D - mu3 * I

Conservation:  S(t) + E(t) + D(t) + I(t) = total population for all t.

Parameters
    alpha  : S -> E   per-susceptible exposure rate (independent of I).
    beta1  : E -> D   exposed becomes sceptical / doubtful.
    beta2  : E -> I   exposed directly accepts and spreads.
    gamma  : D -> I   doubtful individual is eventually convinced.
    mu1    : E -> S   exposed rejects the rumour outright.
    mu2    : D -> S   doubtful rejects after verification.
    mu3    : I -> S   spreader loses interest / rumour grows stale.

--------------------------------------------------------------------------------
Approximate basic reproduction number
--------------------------------------------------------------------------------

Because the exposure term ``alpha * S`` does not depend on the number of
spreaders, the classical next-generation R0 (which measures the average
number of secondary infections per primary infection) is not well-defined
for this model. Instead the simulator prints a steady-state ratio that
characterises the relative pull of the spreader pool against the leakage
of E back to S:

    ratio = alpha * beta2 / (mu3 * (beta1 + beta2 + mu1))

It is a qualitative indicator only; values above ~1 suggest a substantial
spreader population at equilibrium.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Misinformation campaigns where a sizeable fraction of recipients pause
  to verify before resharing.
* Modelling the impact of fact-checking interventions: increasing mu2
  (doubtful rejection) shrinks the long-run spreader pool.
* Comparison baseline against SEDPNR, which further splits spreaders by
  sentiment and adds a terminal "restrained" compartment.
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
class SEDISParams:
    """Parameters for the SEDIS (Susceptible-Exposed-Doubtful-Infected-Susceptible) model."""
    population       : int   = 10_000  # total number of individuals in the network
    initial_exposed  : int   = 0       # number of exposed individuals at t=0
    initial_doubtful : int   = 0       # number of doubtful individuals at t=0
    initial_infected : int   = 10      # number of infected (spreading) individuals at t=0
    alpha            : float = 0.20    # S  -> E   per-S exposure rate (independent of I)
    beta1            : float = 0.10    # E  -> D   exposed becomes sceptical / doubtful
    beta2            : float = 0.15    # E  -> I   exposed directly accepts and spreads
    gamma            : float = 0.08    # D  -> I   doubtful is eventually convinced, starts spreading
    mu1              : float = 0.04    # E  -> S   exposed rejects information early
    mu2              : float = 0.05    # D  -> S   doubtful rejects after rechecking
    mu3              : float = 0.05    # I  -> S   spreader loses interest / information becomes stale
    t_end            : float = 200.0   # simulation end time (days)
    t_steps          : int   = 1_000   # number of equally-spaced time points


def sedis_ode(
    _t: float, y: list[float],
    alpha: float, beta1: float, beta2: float, gamma: float,
    mu1: float, mu2: float, mu3: float,
) -> list[float]:
    """SEDIS model ODEs for rumour propagation with a Doubtful compartment.

    Compartments: S, E, D, I

    Exposure is the per-S leakage form ``alpha * S`` (no I factor): every
    susceptible transitions to E at constant per-capita rate alpha,
    independent of how many spreaders are currently active. This is the
    form used in Govindankutty & Gopalan (2024); modif_SEDIS replaces it
    with the mass-action term ``alpha * S * I``.

    dS/dt = -alpha * S            +  mu1*E  +  mu2*D  +  mu3*I
    dE/dt = +alpha * S            -  (beta1 + beta2 + mu1) * E
    dD/dt = +beta1 * E            -  (gamma + mu2) * D
    dI/dt = +beta2 * E + gamma * D  -  mu3 * I
    """
    S, E, D, I = y
    exposure = alpha * S
    e_out    = (beta1 + beta2 + mu1) * E
    d_out    = (gamma + mu2) * D
    return [
        -exposure + mu1 * E + mu2 * D + mu3 * I,
        +exposure - e_out,
        +beta1 * E - d_out,
        +beta2 * E + gamma * D - mu3 * I,
    ]


def model_sedis(params: SEDISParams) -> plt.Figure:
    """Simulate and plot the SEDIS model for misinformation with a Doubtful compartment."""
    print("\n--- SEDIS Model ---")
    N  = float(params.population)
    E0 = float(params.initial_exposed)
    D0 = float(params.initial_doubtful)
    I0 = float(params.initial_infected)
    S0 = N - E0 - D0 - I0
    r0 = (params.alpha * params.beta2
          / (params.mu3 * (params.beta1 + params.beta2 + params.mu1)))

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial exposed              : {E0:,.0f}")
    print(f"  Initial doubtful             : {D0:,.0f}")
    print(f"  Initial infected             : {I0:,.0f}")
    print(f"  alpha  (S -> E, per S)       : {params.alpha}")
    print(f"  beta1  (E -> D)              : {params.beta1}")
    print(f"  beta2  (E -> I)              : {params.beta2}")
    print(f"  gamma  (D -> I)              : {params.gamma}")
    print(f"  mu1    (E -> S)              : {params.mu1}")
    print(f"  mu2    (D -> S)              : {params.mu2}")
    print(f"  mu3    (I -> S)              : {params.mu3}")
    print(f"  Equilibrium ratio (qual.)    : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        sedis_ode,
        [S0, E0, D0, I0],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2, params.gamma,
         params.mu1, params.mu2, params.mu3),
    )
    S, E, D, I = y

    compartments = {
        "Susceptible": S,
        "Exposed"    : E,
        "Doubtful"   : D,
        "Infected"   : I,
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    peak_t, peak_v = mark_peak(ax, t, compartments["Infected"])

    style_axes(
        ax,
        (fr"SEDIS Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\gamma$={params.gamma}"),
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SEDIS")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SEDIS model simulation complete.")
    return fig
