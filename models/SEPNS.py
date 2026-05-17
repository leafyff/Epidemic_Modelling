"""
================================================================================
SEPNS MODEL — Susceptible / Exposed / Pos. Infected / Neg. Infected / Susceptible
================================================================================

The SEPNS model is a *sentiment-aware* rumour-propagation model. It refines
SEIS for social-network misinformation by splitting the Infected compartment
into two emotionally distinct spreader classes: those who share a rumour with
a *positive* tone (P) and those who share it with a *negative* tone (N).
Both spreader types eventually lose interest and return to the Susceptible
pool — there is no permanent immunity for social rumours, since the same
individual can be re-exposed to the same or related misinformation later.

Compartments:

    S       – Susceptible            : not currently aware of or spreading the rumour.
    E       – Exposed                : has encountered the rumour, deciding what to do.
    P       – Positively Infected    : sharing the rumour with positive sentiment
                                      (endorsement, approval, excitement).
    N       – Negatively Infected    : sharing the rumour with negative sentiment
                                      (outrage, criticism, fear).

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

Let lambda = alpha * S * (P + N) / N be the per-susceptible exposure rate.
Then

    dS/dt = -lambda + mu1 * P + mu2 * N + mu_e * E
    dE/dt = +lambda - (beta1 + beta2 + mu_e) * E
    dP/dt = +beta1 * E - mu1 * P
    dN/dt = +beta2 * E - mu2 * N

Conservation:  S(t) + E(t) + P(t) + N(t) = total population for all t.

Parameters
    alpha  : S -> E exposure / contact rate (both spreader types are contagious).
    beta1  : E -> P rate at which exposed adopts the positive-sentiment voice.
    beta2  : E -> N rate at which exposed adopts the negative-sentiment voice.
    mu1    : P -> S rate at which positive spreaders disengage.
    mu2    : N -> S rate at which negative spreaders disengage.
    mu_e   : E -> S rate at which exposed individuals reject the rumour outright.

--------------------------------------------------------------------------------
Sentiment asymmetry and effective reproduction
--------------------------------------------------------------------------------

In typical social-media settings beta2 > beta1 (negative-sentiment posts
recruit faster) and mu1 ≈ mu2 (decay rates are similar), so N tends to
outgrow P even when both start equal. The simulator prints an approximate
reproduction number

    R0_approx = 0.5 * (beta1 + beta2) / (mu1 + mu2)

useful for quick sanity checks. The plot's peak marker is placed on
whichever of P or N reaches the higher value.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Modelling viral misinformation on social networks where the same rumour
  produces both approving and critical reshares.
* Studying how sentiment skew (e.g. moderation that suppresses one class
  more than the other) shifts long-run propagation.
* A baseline for comparison with SEDIS and SEDPNR, which add a "doubtful"
  fact-checking compartment.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

from drawing import (
    FIGURE_DPI,
    FIGURE_SIZE,
    dominant_infected_curve,
    mark_peak,
    plot_lines,
    save_figure,
    solve,
    style_axes,
)


@dataclass
class SEPNSParams:
    """Parameters for the SEPNS (Susceptible-Exposed-PosInfected-NegInfected-Susceptible) model."""
    population          : int   = 10_000  # total number of individuals in the network
    initial_exposed     : int   = 0       # number of exposed individuals at t=0
    initial_pos_infected: int   = 5       # number of positive-sentiment spreaders at t=0
    initial_neg_infected: int   = 5       # number of negative-sentiment spreaders at t=0
    alpha               : float = 0.20    # S  -> E   exposure / contact rate
    beta1               : float = 0.15    # E  -> P   exposed adopts positive-sentiment spreading
    beta2               : float = 0.20    # E  -> N   exposed adopts negative-sentiment spreading (faster)
    mu1                 : float = 0.05    # P  -> S   positive spreader loses interest, returns to susceptible
    mu2                 : float = 0.05    # N  -> S   negative spreader loses interest, returns to susceptible
    mu_e                : float = 0.03    # E  -> S   exposed individual rejects information early
    t_end               : float = 200.0   # simulation end time (days)
    t_steps             : int   = 1_000   # number of equally-spaced time points


def sepns_ode(
    _t: float, y: list[float],
    alpha: float, beta1: float, beta2: float,
    mu1: float, mu2: float, mu_e: float, N: float,
) -> list[float]:
    """SEPNS model ODEs for social-network rumour propagation.

    Compartments: S, E, P (pos. infected), N_comp (neg. infected)

    dS/dt = -(alpha * S * (P + N_comp) / N)  +  mu1*P  +  mu2*N_comp  +  mu_e*E
    dE/dt = +(alpha * S * (P + N_comp) / N)  -  (beta1 + beta2 + mu_e) * E
    dP/dt = +beta1 * E  -  mu1 * P
    dN/dt = +beta2 * E  -  mu2 * N_comp
    """
    S, E, P, N_comp = y
    total_spreaders = P + N_comp
    exposure        = alpha * S * total_spreaders / N
    e_out           = (beta1 + beta2 + mu_e) * E
    return [
        -exposure + mu1 * P + mu2 * N_comp + mu_e * E,
        +exposure - e_out,
        +beta1 * E - mu1 * P,
        +beta2 * E - mu2 * N_comp,
    ]


def model_sepns(params: SEPNSParams) -> plt.Figure:
    """Simulate and plot the SEPNS model for social-network misinformation spread."""
    print("\n--- SEPNS Model ---")
    N  = float(params.population)
    E0 = float(params.initial_exposed)
    P0 = float(params.initial_pos_infected)
    N0 = float(params.initial_neg_infected)
    S0 = N - E0 - P0 - N0
    r0 = (params.beta1 + params.beta2) / (params.mu1 + params.mu2) * 0.5

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial exposed              : {E0:,.0f}")
    print(f"  Initial pos. infected (P)    : {P0:,.0f}")
    print(f"  Initial neg. infected (N)    : {N0:,.0f}")
    print(f"  alpha  (S -> E)              : {params.alpha}")
    print(f"  beta1  (E -> P)              : {params.beta1}")
    print(f"  beta2  (E -> N)              : {params.beta2}")
    print(f"  mu1    (P -> S)              : {params.mu1}")
    print(f"  mu2    (N -> S)              : {params.mu2}")
    print(f"  mu_e   (E -> S rejection)    : {params.mu_e}")
    print(f"  Approx. R0                   : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        sepns_ode,
        [S0, E0, P0, N0],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2,
         params.mu1, params.mu2, params.mu_e, N),
    )
    S, E, P, N_comp = y

    compartments = {
        "Susceptible"       : S,
        "Exposed"           : E,
        "Pos. Infected (P)" : P,
        "Neg. Infected (N)" : N_comp,
    }

    # Mark the peak on whichever infected curve (P or N) reaches its highest point.
    _, peak_curve = dominant_infected_curve(
        t, compartments,
        infected_keys=["Pos. Infected (P)", "Neg. Infected (N)"],
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    peak_t, peak_v = mark_peak(ax, t, peak_curve)

    style_axes(
        ax,
        (fr"SEPNS Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\beta_2$={params.beta2}"),
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SEPNS")
    print(f"  Peak spreader compartment    : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SEPNS model simulation complete.")
    return fig
