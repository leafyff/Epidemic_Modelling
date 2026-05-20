"""
================================================================================
modif_SEDIS MODEL — Modified Susceptible / Exposed / Doubtful / Infected / Susceptible
================================================================================

A variant of the SEDIS rumour-propagation model. The compartments and
every other transition are identical to SEDIS; the *only* difference is
that the exposure term carries an explicit factor of I, so susceptibles
become exposed at the rate

    exposure = alpha * S * I       (depends on the number of spreaders)

instead of the SEDIS rate ``alpha * S``. This makes the rate at which
new individuals are exposed proportional to the current number of active
spreaders rather than just to the susceptible pool.

Because the exposure term now scales with I, the numerical value of
``alpha`` here is much smaller than in SEDIS — it is a per-pair
interaction rate rather than a per-S rate.

Compartments:

    S – Susceptible : not currently aware of or spreading the rumour.
    E – Exposed    : has just encountered the rumour.
    D – Doubtful   : uncertain; weighing evidence (fact-checking, hesitating).
    I – Infected   : has accepted the rumour and is actively spreading it.

--------------------------------------------------------------------------------
Governing equations
--------------------------------------------------------------------------------

    dS/dt = mu1*E + mu2*D + mu3*I - alpha*S*I
    dE/dt = alpha*S*I             - (beta1 + beta2 + mu1) * E
    dD/dt = beta1 * E             - (gamma + mu2) * D
    dI/dt = gamma * D + beta2 * E - mu3 * I

Conservation:  S(t) + E(t) + D(t) + I(t) = total population for all t.

Parameters
    alpha  : S -> E   per-pair exposure rate (per S, per I, per unit time).
    beta1  : E -> D   exposed becomes sceptical / doubtful.
    beta2  : E -> I   exposed directly accepts and spreads.
    gamma  : D -> I   doubtful individual is eventually convinced.
    mu1    : E -> S   exposed rejects the rumour outright.
    mu2    : D -> S   doubtful rejects after verification.
    mu3    : I -> S   spreader loses interest / rumour grows stale.

--------------------------------------------------------------------------------
Approximate basic reproduction number
--------------------------------------------------------------------------------

The same direct E -> I -> reinfection loop used for SEDIS, but scaled by
the population because the contact term now depends on the spreader pool:

    R0_approx = alpha * N * beta2 / (mu3 * (beta1 + beta2 + mu1))

This underestimates the D -> I contribution; treat it as qualitative.

--------------------------------------------------------------------------------
Typical use cases
--------------------------------------------------------------------------------

* Misinformation spread where the exposure pressure on a susceptible
  individual grows with the absolute number of active spreaders, rather
  than being independent of how many are currently spreading.
* Sensitivity analysis comparing the SEDIS exposure term ``alpha * S``
  with the I-dependent ``alpha * S * I`` form.
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
class ModifSEDISParams:
    """Parameters for the modif_SEDIS (mass-action SEDIS) model."""
    population       : int   = 10_000   # total number of individuals in the network
    initial_exposed  : int   = 0        # number of exposed individuals at t=0
    initial_doubtful : int   = 0        # number of doubtful individuals at t=0
    initial_infected : int   = 10       # number of infected (spreading) individuals at t=0
    alpha            : float = 2.0e-5   # S  -> E   per-pair exposure rate (per S, per I)
    beta1            : float = 0.10     # E  -> D   exposed becomes sceptical / doubtful
    beta2            : float = 0.15     # E  -> I   exposed directly accepts and spreads
    gamma            : float = 0.08     # D  -> I   doubtful is eventually convinced, starts spreading
    mu1              : float = 0.04     # E  -> S   exposed rejects information early
    mu2              : float = 0.05     # D  -> S   doubtful rejects after rechecking
    mu3              : float = 0.05     # I  -> S   spreader loses interest / information becomes stale
    t_end            : float = 200.0    # simulation end time (days)
    t_steps          : int   = 1_000    # number of equally-spaced time points


def modif_sedis_ode(
    _t: float, y: list[float],
    alpha: float, beta1: float, beta2: float, gamma: float,
    mu1: float, mu2: float, mu3: float,
) -> list[float]:
    """modif_SEDIS model ODEs (exposure term carries an explicit I factor).

    Compartments: S, E, D, I

    dS/dt = mu1*E + mu2*D + mu3*I - alpha*S*I
    dE/dt = alpha*S*I             - (beta1 + beta2 + mu1) * E
    dD/dt = beta1 * E             - (gamma + mu2) * D
    dI/dt = gamma * D + beta2 * E - mu3 * I
    """
    S, E, D, I = y
    exposure = alpha * S * I
    e_out    = (beta1 + beta2 + mu1) * E
    d_out    = (gamma + mu2) * D
    return [
        mu1 * E + mu2 * D + mu3 * I - exposure,
        exposure - e_out,
        beta1 * E - d_out,
        gamma * D + beta2 * E - mu3 * I,
    ]


def model_modif_sedis(params: ModifSEDISParams) -> plt.Figure:
    """Simulate and plot the modif_SEDIS model (SEDIS with alpha*S*I exposure)."""
    print("\n--- modif_SEDIS Model ---")
    N  = float(params.population)
    E0 = float(params.initial_exposed)
    D0 = float(params.initial_doubtful)
    I0 = float(params.initial_infected)
    S0 = N - E0 - D0 - I0
    r0 = (params.alpha * N * params.beta2
          / (params.mu3 * (params.beta1 + params.beta2 + params.mu1)))

    print(f"  Population                   : {N:,.0f}")
    print(f"  Initial exposed              : {E0:,.0f}")
    print(f"  Initial doubtful             : {D0:,.0f}")
    print(f"  Initial infected             : {I0:,.0f}")
    print(f"  alpha  (S -> E, per S*I)     : {params.alpha}")
    print(f"  beta1  (E -> D)              : {params.beta1}")
    print(f"  beta2  (E -> I)              : {params.beta2}")
    print(f"  gamma  (D -> I)              : {params.gamma}")
    print(f"  mu1    (E -> S)              : {params.mu1}")
    print(f"  mu2    (D -> S)              : {params.mu2}")
    print(f"  mu3    (I -> S)              : {params.mu3}")
    print(f"  Approx. R0                   : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        modif_sedis_ode,
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
        (fr"modif_SEDIS Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\gamma$={params.gamma}"),
        params.t_end,
    )
    fig.tight_layout()
    save_figure(fig, "modif_SEDIS")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  modif_SEDIS model simulation complete.")
    return fig
