"""Public simulation functions — one per epidemic model.

Each ``model_*`` function takes its matching ``*Params`` dataclass, runs the
ODE solver, prints a summary, saves a PNG to ``figs/`` and returns the
matplotlib Figure object.
"""

import matplotlib.pyplot as plt

from constants import FIGURE_DPI, FIGURE_SIZE, COLORS
from odes import (
    si_ode,
    sis_ode,
    sir_ode,
    seir_ode,
    sepns_ode,
    sedis_ode,
    sedpnr_ode,
)
from params import (
    SIParams,
    SISParams,
    SIRParams,
    SEIRParams,
    SEPNSParams,
    SEDISParams,
    SEDPNRParams,
)
from plotting import (
    dominant_infected_curve,
    mark_peak,
    plot_lines,
    save_figure,
    solve,
    style_axes,
)


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


def model_sepns(params: SEPNSParams) -> plt.Figure:
    """Simulate and plot the SEPNS model for social-network misinformation spread.

    The infected pool is split by rumour *sentiment*:
      P – positively spreading (shares the rumour approvingly)
      N – negatively spreading (shares the rumour critically / emotionally)
    Both spreader classes return to Susceptible (no permanent disengagement).
    """
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


def model_sedis(params: SEDISParams) -> plt.Figure:
    """Simulate and plot the SEDIS model for misinformation with a Doubtful compartment.

    Adds human-selection behaviour: exposed individuals may enter a Doubtful state
    before deciding to spread (I) or return to Susceptible.  All spreaders eventually
    cycle back to Susceptible (rumour-specific; no permanent immunity).
    """
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
    print(f"  alpha  (S -> E)              : {params.alpha}")
    print(f"  beta1  (E -> D)              : {params.beta1}")
    print(f"  beta2  (E -> I)              : {params.beta2}")
    print(f"  gamma  (D -> I)              : {params.gamma}")
    print(f"  mu1    (E -> S)              : {params.mu1}")
    print(f"  mu2    (D -> S)              : {params.mu2}")
    print(f"  mu3    (I -> S)              : {params.mu3}")
    print(f"  Approx. R0                   : {r0:.3f}")
    print(f"  Simulation period            : {params.t_end} days")

    t, y = solve(
        sedis_ode,
        [S0, E0, D0, I0],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2, params.gamma,
         params.mu1, params.mu2, params.mu3, N),
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


def model_sedpnr(params: SEDPNRParams) -> plt.Figure:
    """Simulate and plot the SEDPNR model

    The most complete misinformation model in this suite:
      S  – Susceptible
      E  – Exposed (encountered the misinformation)
      D  – Doubtful (sceptical; fact-checking)
      P  – Positively Infected (spreading with positive sentiment)
      N  – Negatively Infected (spreading with negative sentiment)
      R  – Restrained (permanently stopped spreading)

    R0 = max(beta1 / lambda1, beta2 / lambda2)
    """
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
        t, compartments,
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
