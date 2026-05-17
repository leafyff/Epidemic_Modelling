"""
================================================================================
SEDPNR MODEL — Susceptible / Exposed / Doubtful /
                Positively Infected / Negatively Infected / Restrained
================================================================================

The SEDPNR model is the most complete misinformation-spread model in this
suite. It is the formulation proposed by Govindankutty & Gopalan (2024) and
combines the two main extensions developed elsewhere in the package:

  * the *sentiment split* of the Infected compartment (P / N) from SEPNS, and
  * the *Doubtful* fact-checking compartment from SEDIS,

then adds a terminal *Restrained* compartment for individuals who have
permanently disengaged from the topic and will not spread again. Together
these six compartments capture the lifecycle of a viral social-media rumour:
encounter → hesitation / fact-check → emotional uptake → eventual silence.

Compartments:

    S – Susceptible            : not currently aware of or spreading the rumour.
    E – Exposed                : has encountered the rumour.
    D – Doubtful               : sceptical; verifying or hesitating.
    P – Positively Infected    : spreading with a positive / approving voice.
    N – Negatively Infected    : spreading with a negative / critical voice.
    R – Restrained             : has permanently stopped spreading (terminal).

--------------------------------------------------------------------------------
Governing equations  (paper equations 5–10)
--------------------------------------------------------------------------------

    dS/dt = mu1*E + mu2*D - alpha*S
    dE/dt = alpha*S - (beta1 + beta2 + gamma + mu1)*E
    dD/dt = gamma*E - (beta3 + beta4 + mu2)*D
    dP/dt = beta1*E + beta3*D - lambda1*P
    dN/dt = beta2*E + beta4*D - lambda2*N
    dR/dt = lambda1*P + lambda2*N

Note: alpha acts as a *per-capita* rate on S, matching the homogeneous /
mean-field formulation in the original paper.

Parameters
    alpha   : S -> E   contact / exposure rate.
    beta1   : E -> P   exposed adopts positive-sentiment spreading.
    beta2   : E -> N   exposed adopts negative-sentiment spreading.
    beta3   : D -> P   doubtful converted to positive spreader.
    beta4   : D -> N   doubtful converted to negative spreader.
    gamma   : E -> D   exposed becomes doubtful.
    lambda1 : P -> R   positive spreader becomes permanently restrained.
    lambda2 : N -> R   negative spreader becomes permanently restrained.
    mu1     : E -> S   exposed rejects information.
    mu2     : D -> S   doubtful rejects after verification.

--------------------------------------------------------------------------------
Basic reproduction number
--------------------------------------------------------------------------------

For sentiment-aware spread, an individual rumour "lineage" is dominated by
whichever sentiment recruits faster relative to its decay into Restrained:

    R0 = max( beta1 / lambda1,  beta2 / lambda2 )

R0 > 1 indicates that at least one sentiment voice can sustain itself.

--------------------------------------------------------------------------------
Long-run behaviour
--------------------------------------------------------------------------------

Because the Restrained compartment is absorbing (only inflow, no outflow),
S(t), E(t), D(t), P(t), N(t) all tend to zero as t → ∞, with R(t) → N - S(∞).
The model therefore produces a clean *epidemic curve* for each sentiment and
a monotone build-up of the silenced population.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Modelling viral misinformation on social networks with realistic
  sentiment polarisation and fact-checking dynamics.
* Studying the impact of moderation strategies that increase lambda1 / lambda2
  (e.g. content removal pushing spreaders into the Restrained state).
* Evaluating policies that increase mu2 (e.g. labels and warnings that help
  doubtful users disengage rather than be converted).
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
class SEDPNRParams:
    """Parameters for the SEDPNR model (Govindankutty & Gopalan, 2024)."""
    population          : int   = 10_000  # total number of individuals in the network
    initial_exposed     : int   = 10      # number of exposed individuals at t=0
    initial_doubtful    : int   = 10      # number of doubtful individuals at t=0
    initial_pos_infected: int   = 5       # number of positive-sentiment spreaders at t=0
    initial_neg_infected: int   = 5       # number of negative-sentiment spreaders at t=0
    initial_restrained  : int   = 0       # number of restrained (silent) individuals at t=0
    alpha               : float = 0.20    # S  -> E   contact / exposure rate
    beta1               : float = 0.15    # E  -> P   exposed adopts positive-sentiment spreading
    beta2               : float = 0.20    # E  -> N   exposed adopts negative-sentiment spreading
    beta3               : float = 0.10    # D  -> P   doubtful converted to positive spreader
    beta4               : float = 0.12    # D  -> N   doubtful converted to negative spreader
    gamma               : float = 0.10    # E  -> D   exposed becomes doubtful
    lambda1             : float = 0.05    # P  -> R   positive spreader becomes restrained
    lambda2             : float = 0.05    # N  -> R   negative spreader becomes restrained
    mu1                 : float = 0.03    # E  -> S   exposed rejects information, returns to susceptible
    mu2                 : float = 0.04    # D  -> S   doubtful rejects after verification
    t_end               : float = 100.0   # simulation end time (days)
    t_steps             : int   = 1_000   # number of equally-spaced time points


def sedpnr_ode(
    _t: float, y: list[float],
    alpha: float,
    beta1: float, beta2: float, beta3: float, beta4: float,
    gamma: float, lambda1: float, lambda2: float,
    mu1: float, mu2: float,
) -> list[float]:
    """SEDPNR model ODEs (Govindankutty & Gopalan 2024, equations 5–10).

    Compartments: S, E, D, P, N_comp, R

    dS/dt = mu1*E + mu2*D - alpha*S
    dE/dt = alpha*S - (beta1 + beta2 + gamma + mu1)*E
    dD/dt = gamma*E - (beta3 + beta4 + mu2)*D
    dP/dt = beta1*E + beta3*D - lambda1*P
    dN/dt = beta2*E + beta4*D - lambda2*N_comp
    dR/dt = lambda1*P + lambda2*N_comp
    """
    S, E, D, P, N_comp, _R = y
    e_out = (beta1 + beta2 + gamma + mu1) * E
    d_out = (beta3 + beta4 + mu2) * D
    return [
        mu1 * E + mu2 * D - alpha * S,
        alpha * S - e_out,
        gamma * E - d_out,
        beta1 * E + beta3 * D - lambda1 * P,
        beta2 * E + beta4 * D - lambda2 * N_comp,
        lambda1 * P + lambda2 * N_comp,
    ]


def model_sedpnr(params: SEDPNRParams) -> plt.Figure:
    """Simulate and plot the full SEDPNR misinformation model."""
    print("\n--- SEDPNR Model ---")
    N    = float(params.population)
    E0   = float(params.initial_exposed)
    D0   = float(params.initial_doubtful)
    P0   = float(params.initial_pos_infected)
    N0   = float(params.initial_neg_infected)
    R0_v = float(params.initial_restrained)
    S0   = N - E0 - D0 - P0 - N0 - R0_v
    r0   = max(params.beta1 / params.lambda1, params.beta2 / params.lambda2)

    print(f"  Population                           : {N:,.0f}")
    print(f"  Initial exposed                      : {E0:,.0f}")
    print(f"  Initial doubtful                     : {D0:,.0f}")
    print(f"  Initial pos. infected (P)            : {P0:,.0f}")
    print(f"  Initial neg. infected (N)            : {N0:,.0f}")
    print(f"  Initial restrained                   : {R0_v:,.0f}")
    print(f"  alpha   (S -> E)                     : {params.alpha}")
    print(f"  beta1   (E -> P)                     : {params.beta1}")
    print(f"  beta2   (E -> N)                     : {params.beta2}")
    print(f"  beta3   (D -> P)                     : {params.beta3}")
    print(f"  beta4   (D -> N)                     : {params.beta4}")
    print(f"  gamma   (E -> D)                     : {params.gamma}")
    print(f"  lambda1 (P -> R)                     : {params.lambda1}")
    print(f"  lambda2 (N -> R)                     : {params.lambda2}")
    print(f"  mu1     (E -> S)                     : {params.mu1}")
    print(f"  mu2     (D -> S)                     : {params.mu2}")
    print(f"  R0 = max(beta1/lambda1, beta2/lambda2) : {r0:.3f}")
    print(f"  Simulation period                    : {params.t_end} days")

    t, y = solve(
        sedpnr_ode,
        [S0, E0, D0, P0, N0, R0_v],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2, params.beta3, params.beta4,
         params.gamma, params.lambda1, params.lambda2,
         params.mu1, params.mu2),
    )
    S, E, D, P, N_comp, R = y

    compartments = {
        "Susceptible"       : S,
        "Exposed"           : E,
        "Doubtful"          : D,
        "Pos. Infected (P)" : P,
        "Neg. Infected (N)" : N_comp,
        "Restrained"        : R,
    }

    # Mark the peak on whichever infected curve (P or N) reaches its highest point.
    _, peak_curve = dominant_infected_curve(
        compartments,
        infected_keys=["Pos. Infected (P)", "Neg. Infected (N)"],
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    plot_lines(ax, t, compartments)
    peak_t, peak_v = mark_peak(ax, t, peak_curve)

    style_axes(
        ax,
        (fr"SEDPNR Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\beta_2$={params.beta2}"),
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "SEDPNR")
    print(f"  Peak spreader compartment            : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SEDPNR model simulation complete.")
    return fig
