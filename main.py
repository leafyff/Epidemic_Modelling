"""
epidemic_models.py
==================
Simulation of classic and social-network epidemic models:
  - SI     (Susceptible - Infected)
  - SIS    (Susceptible - Infected - Susceptible)
  - SIR    (Susceptible - Infected - Recovered)
  - SEIR   (Susceptible - Exposed - Infected - Recovered)
  - SEPNS  (Susceptible - Exposed - Positively Infected - Negatively Infected - Susceptible)
  - SEDIS  (Susceptible - Exposed - Doubtful - Infected - Susceptible)
  - SEDPNR (Susceptible - Exposed - Doubtful - Positively Infected - Negatively Infected - Restrained)

Each model is solved with scipy.integrate.solve_ivp (RK45) and plotted
with matplotlib. All parameters are configured in main().

Plot conventions
----------------
- Titles use proper Greek-letter subscripts via matplotlib mathtext,
  e.g. r"$\beta_1$", r"$\alpha$".
- The infection peak is marked with a single red dot; all numeric peak
  details are printed to stdout only.
- No annotation boxes or R₀ labels appear inside the figure; only the
  compartment colour legend is shown.

References
----------
Govindankutty & Gopalan (2024). "Epidemic modeling for misinformation spread
in digital networks through a social intelligence approach."
Scientific Reports 14, 19100. https://doi.org/10.1038/s41598-024-69657-0
"""

import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGURE_DPI     = 150
FIGURE_SIZE    = (10, 5)
LINE_WIDTH     = 2.2
FIGS_DIR       = "figs"

PEAK_DOT_COLOR = "#D32F2F"   # vivid red used for the peak marker
PEAK_DOT_SIZE  = 80          # scatter marker size (points²)

COLORS = {
    "S": "#2196F3",  # blue
    "E": "#FF9800",  # orange
    "I": "#F44336",  # red
    "R": "#4CAF50",  # green
    "P": "#9C27B0",  # purple  – Positively infected
    "N": "#E91E63",  # pink    – Negatively infected
    "D": "#795548",  # brown   – Doubtful
}


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SIParams:
    population      : int   = 10_000
    initial_infected: int   = 10
    beta            : float = 0.30
    t_end           : float = 60.0
    t_steps         : int   = 1_000


@dataclass
class SISParams:
    population      : int   = 10_000
    initial_infected: int   = 10
    beta            : float = 0.30
    gamma           : float = 0.10
    t_end           : float = 120.0
    t_steps         : int   = 1_000


@dataclass
class SIRParams:
    population        : int   = 10_000
    initial_infected  : int   = 10
    initial_recovered : int   = 0
    beta              : float = 0.30
    gamma             : float = 0.10
    t_end             : float = 160.0
    t_steps           : int   = 1_000


@dataclass
class SEIRParams:
    population      : int   = 10_000
    initial_exposed : int   = 0
    initial_infected: int   = 10
    beta            : float = 0.30
    sigma           : float = 0.20
    gamma           : float = 0.10
    t_end           : float = 200.0
    t_steps         : int   = 1_000


@dataclass
class SEPNSParams:
    """Parameters for the SEPNS (Susceptible-Exposed-Positively Infected-
    Negatively Infected-Susceptible) model.

    The infected state is split by the *sentiment* of the rumour being spread:
      P – individuals spreading misinformation with a positive tone.
      N – individuals spreading misinformation with a negative tone.
    Both P and N eventually return to S (no permanent immunity for social rumours).

    Transition rates
    ----------------
    alpha : S  -> E   exposure / contact rate
    beta1 : E  -> P   exposed adopts positive-sentiment spreading
    beta2 : E  -> N   exposed adopts negative-sentiment spreading
    mu1   : P  -> S   positive spreader loses interest / returns to susceptible
    mu2   : N  -> S   negative spreader loses interest / returns to susceptible
    mu_e  : E  -> S   exposed individual rejects information, returns to susceptible
    """
    population          : int   = 10_000
    initial_exposed     : int   = 0
    initial_pos_infected: int   = 5
    initial_neg_infected: int   = 5
    alpha               : float = 0.20   # S -> E
    beta1               : float = 0.15   # E -> P
    beta2               : float = 0.20   # E -> N  (negative spreads faster)
    mu1                 : float = 0.05   # P -> S
    mu2                 : float = 0.05   # N -> S
    mu_e                : float = 0.03   # E -> S  (rejection of information)
    t_end               : float = 200.0
    t_steps             : int   = 1_000


@dataclass
class SEDISParams:
    """Parameters for the SEDIS (Susceptible-Exposed-Doubtful-Infected-Susceptible)
    model of rumour propagation.

    Adds a *Doubtful* (D) compartment: individuals who have heard the rumour but
    have not yet decided whether to believe or reject it.  Both D and I can cycle
    back to S because social-media rumours offer no permanent immunity.

    Transition rates
    ----------------
    alpha : S -> E   exposure rate
    beta1 : E -> D   exposed becomes sceptical / doubtful
    beta2 : E -> I   exposed directly accepts and spreads
    gamma : D -> I   doubtful is eventually convinced, starts spreading
    mu1   : E -> S   exposed rejects information early
    mu2   : D -> S   doubtful rejects after rechecking
    mu3   : I -> S   spreader loses interest / information becomes stale
    """
    population       : int   = 10_000
    initial_exposed  : int   = 0
    initial_doubtful : int   = 0
    initial_infected : int   = 10
    alpha            : float = 0.20   # S -> E
    beta1            : float = 0.10   # E -> D
    beta2            : float = 0.15   # E -> I
    gamma            : float = 0.08   # D -> I
    mu1              : float = 0.04   # E -> S
    mu2              : float = 0.05   # D -> S
    mu3              : float = 0.05   # I -> S
    t_end            : float = 200.0
    t_steps          : int   = 1_000


@dataclass
class SEDPNRParams:
    """Parameters for the SEDPNR (Susceptible-Exposed-Doubtful-Positively Infected-
    Negatively Infected-Restrained) model.

    The full misinformation-spread model proposed by Govindankutty & Gopalan (2024).
    It combines sentiment-aware infection (P/N split) with a Doubtful state and a
    terminal Restrained state (individuals who permanently stop spreading).

    Transition rates  (see paper equations 5–10)
    ----------------
    alpha   : S  -> E   contact / exposure rate
    beta1   : E  -> P   exposed adopts positive-sentiment spreading
    beta2   : E  -> N   exposed adopts negative-sentiment spreading
    beta3   : D  -> P   doubtful converted to positive spreader
    beta4   : D  -> N   doubtful converted to negative spreader
    gamma   : E  -> D   exposed becomes doubtful
    lambda1 : P  -> R   positive spreader becomes restrained
    lambda2 : N  -> R   negative spreader becomes restrained
    mu1     : E  -> S   exposed rejects information, returns to susceptible
    mu2     : D  -> S   doubtful rejects after verification
    """
    population          : int   = 10_000
    initial_exposed     : int   = 10
    initial_doubtful    : int   = 10
    initial_pos_infected: int   = 5
    initial_neg_infected: int   = 5
    initial_restrained  : int   = 0
    alpha               : float = 0.20   # S -> E
    beta1               : float = 0.15   # E -> P
    beta2               : float = 0.20   # E -> N
    beta3               : float = 0.10   # D -> P
    beta4               : float = 0.12   # D -> N
    gamma               : float = 0.10   # E -> D
    lambda1             : float = 0.05   # P -> R
    lambda2             : float = 0.05   # N -> R
    mu1                 : float = 0.03   # E -> S
    mu2                 : float = 0.04   # D -> S
    t_end               : float = 100.0
    t_steps             : int   = 1_000


# ---------------------------------------------------------------------------
# ODE right-hand-side functions
# ---------------------------------------------------------------------------

def _si_ode(_t: float, y: list[float], beta: float, N: float) -> list[float]:
    """SI model ODEs.

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N
    """
    S, I = y
    new_infected = beta * S * I / N
    return [-new_infected, +new_infected]


def _sis_ode(
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


def _sir_ode(
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


def _seir_ode(
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


def _sepns_ode(
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


def _sedis_ode(
    _t: float, y: list[float],
    alpha: float, beta1: float, beta2: float, gamma: float,
    mu1: float, mu2: float, mu3: float, N: float,
) -> list[float]:
    """SEDIS model ODEs for rumour propagation with a Doubtful compartment.

    Compartments: S, E, D, I

    dS/dt = -(alpha * S * I / N)  +  mu1*E  +  mu2*D  +  mu3*I
    dE/dt = +(alpha * S * I / N)  -  (beta1 + beta2 + mu1) * E
    dD/dt = +beta1 * E            -  (gamma + mu2) * D
    dI/dt = +beta2 * E + gamma * D  -  mu3 * I
    """
    S, E, D, I = y
    exposure = alpha * S * I / N
    e_out    = (beta1 + beta2 + mu1) * E
    d_out    = (gamma + mu2) * D
    return [
        -exposure + mu1 * E + mu2 * D + mu3 * I,
        +exposure - e_out,
        +beta1 * E - d_out,
        +beta2 * E + gamma * D - mu3 * I,
    ]


def _sedpnr_ode(
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

    Note: alpha acts as a per-capita rate on S (mean-field / homogeneous-network
    formulation from the paper).
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_figs_dir() -> None:
    """Create the output figures directory if it does not exist."""
    os.makedirs(FIGS_DIR, exist_ok=True)


def _save_figure(fig: plt.Figure, model_name: str) -> None:
    """Save *fig* to FIGS_DIR/<model_name>_model_ex.png."""
    _ensure_figs_dir()
    path = os.path.join(FIGS_DIR, f"{model_name}_model_ex.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")


def _solve(
    fun: Any,
    y0: list[float],
    t_end: float,
    t_steps: int,
    args: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Run solve_ivp (RK45) and return (time, states) as plain ndarrays."""
    t_eval: np.ndarray = np.linspace(0.0, t_end, t_steps)
    result: Any = solve_ivp(
        fun=fun, t_span=(0.0, t_end), y0=y0,
        args=args, t_eval=t_eval, method="RK45",
    )
    return np.asarray(result.t), np.asarray(result.y)


def _style_axes(ax: plt.Axes, title: str, t_end: float) -> None:
    """Apply shared axes formatting.  Only the compartment legend is displayed."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Number of individuals", fontsize=11)
    ax.set_xlim(0, t_end)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _mark_peak(
    ax: plt.Axes, t: np.ndarray, values: np.ndarray
) -> tuple[float, float]:
    """Place a single red dot at the peak of *values*; return (peak_t, peak_v)."""
    idx    = int(np.argmax(values))
    peak_t = float(t[idx])
    peak_v = float(values[idx])
    ax.scatter(peak_t, peak_v, color=PEAK_DOT_COLOR, s=PEAK_DOT_SIZE, zorder=5)
    return peak_t, peak_v


def _plot_lines(ax: plt.Axes, t: np.ndarray, compartments: dict[str, np.ndarray]) -> None:
    """Draw one filled, labelled line per compartment."""
    for label, values in compartments.items():
        color = COLORS[label[0]]   # first letter maps to COLORS key (S/E/I/R/P/N/D)
        ax.fill_between(t, values, alpha=0.08, color=color)
        ax.plot(t, values, label=label, color=color, linewidth=LINE_WIDTH)


# ---------------------------------------------------------------------------
# Public model functions
# ---------------------------------------------------------------------------

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

    t, y = _solve(_si_ode, [S0, I0], params.t_end, params.t_steps, (params.beta, N))
    S, I = y

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {"Susceptible": S, "Infected": I})
    # SI infection only rises monotonically – peak is the final value
    _mark_peak(ax, t, I)

    _style_axes(
        ax,
        fr"SI Model  |  N={N:,.0f}  |  $\beta$={params.beta}",
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SI")
    print(f"  Final infected               : {float(I[-1]):,.0f} individuals")
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

    t, y = _solve(
        _sis_ode, [S0, I0], params.t_end, params.t_steps,
        (params.beta, params.gamma, N),
    )
    S, I = y

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {"Susceptible": S, "Infected": I})
    peak_t, peak_v = _mark_peak(ax, t, I)

    if r0 > 1.0:
        endemic_I = N * (1.0 - 1.0 / r0)
        ax.axhline(
            endemic_I, color=COLORS["I"], linestyle=":", linewidth=1.2, alpha=0.7,
            label=f"Endemic I* = {endemic_I:,.0f}",
        )
        ax.legend(fontsize=10, framealpha=0.9)

    _style_axes(
        ax,
        fr"SIS Model  |  N={N:,.0f}  |  $\beta$={params.beta}  |  $\gamma$={params.gamma}",
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SIS")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SIS model simulation complete.")
    return fig


def model_sir(params: SIRParams) -> plt.Figure:
    """Simulate and plot the SIR model (Kermack-McKendrick 1927)."""
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

    t, y = _solve(
        _sir_ode, [S0, I0, R0_val], params.t_end, params.t_steps,
        (params.beta, params.gamma, N),
    )
    S, I, R = y

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {"Susceptible": S, "Infected": I, "Recovered": R})
    peak_t, peak_v = _mark_peak(ax, t, I)

    _style_axes(
        ax,
        fr"SIR Model  |  N={N:,.0f}  |  $\beta$={params.beta}  |  $\gamma$={params.gamma}",
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SIR")
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

    t, y = _solve(
        _seir_ode, [S0, E0, I0, 0.0], params.t_end, params.t_steps,
        (params.beta, params.sigma, params.gamma, N),
    )
    S, E, I, R = y

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {"Susceptible": S, "Exposed": E, "Infected": I, "Recovered": R})
    peak_t, peak_v = _mark_peak(ax, t, I)

    _style_axes(
        ax,
        (fr"SEIR Model  |  N={N:,.0f}  |  $\beta$={params.beta}"
         fr"  |  $\sigma$={params.sigma}  |  $\gamma$={params.gamma}"),
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SEIR")
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

    t, y = _solve(
        _sepns_ode,
        [S0, E0, P0, N0],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2,
         params.mu1, params.mu2, params.mu_e, N),
    )
    S, E, P, N_comp = y

    total_spreaders = P + N_comp

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {
        "Susceptible"       : S,
        "Exposed"           : E,
        "Pos. Infected (P)" : P,
        "Neg. Infected (N)" : N_comp,
    })
    peak_t, peak_v = _mark_peak(ax, t, total_spreaders)

    _style_axes(
        ax,
        (fr"SEPNS Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\beta_2$={params.beta2}"),
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SEPNS")
    print(f"  Peak total spreaders (P+N)   : {peak_v:,.0f} individuals at day {peak_t:.1f}")
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

    t, y = _solve(
        _sedis_ode,
        [S0, E0, D0, I0],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2, params.gamma,
         params.mu1, params.mu2, params.mu3, N),
    )
    S, E, D, I = y

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {
        "Susceptible": S,
        "Exposed"    : E,
        "Doubtful"   : D,
        "Infected"   : I,
    })
    peak_t, peak_v = _mark_peak(ax, t, I)

    _style_axes(
        ax,
        (fr"SEDIS Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\gamma$={params.gamma}"),
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SEDIS")
    print(f"  Peak infection               : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SEDIS model simulation complete.")
    return fig


def model_sedpnr(params: SEDPNRParams) -> plt.Figure:
    """Simulate and plot the SEDPNR model (Govindankutty & Gopalan, 2024).

    The most complete misinformation model in this suite:
      S  – Susceptible
      E  – Exposed (encountered the misinformation)
      D  – Doubtful (sceptical; fact-checking)
      P  – Positively Infected (spreading with positive sentiment)
      N  – Negatively Infected (spreading with negative sentiment)
      R  – Restrained (permanently stopped spreading)

    R0 = max(beta1 / lambda1, beta2 / lambda2)  [paper equation 27].
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

    t, y = _solve(
        _sedpnr_ode,
        [S0, E0, D0, P0, N0, R0_v],
        params.t_end, params.t_steps,
        (params.alpha, params.beta1, params.beta2, params.beta3, params.beta4,
         params.gamma, params.lambda1, params.lambda2,
         params.mu1, params.mu2),
    )
    S, E, D, P, N_comp, R = y

    total_spreaders = P + N_comp

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    _plot_lines(ax, t, {
        "Susceptible"       : S,
        "Exposed"           : E,
        "Doubtful"          : D,
        "Pos. Infected (P)" : P,
        "Neg. Infected (N)" : N_comp,
        "Restrained"        : R,
    })
    peak_t, peak_v = _mark_peak(ax, t, total_spreaders)

    _style_axes(
        ax,
        (fr"SEDPNR Model  |  N={N:,.0f}  |  $\alpha$={params.alpha}"
         fr"  |  $\beta_1$={params.beta1}  |  $\beta_2$={params.beta2}"),
        params.t_end,
    )
    fig.tight_layout()
    _save_figure(fig, "SEDPNR")
    print(f"  Peak total spreaders (P+N)           : {peak_v:,.0f} individuals at day {peak_t:.1f}")
    print("  SEDPNR model simulation complete.")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Epidemic Model Simulator")
    print("=" * 60)

    si_params = SIParams(
        population       = 10_000,
        initial_infected = 10,
        beta             = 0.30,
        t_end            = 60.0,
        t_steps          = 1_000,
    )

    sis_params = SISParams(
        population       = 10_000,
        initial_infected = 10,
        beta             = 0.30,
        gamma            = 0.10,
        t_end            = 120.0,
        t_steps          = 1_000,
    )

    sir_params = SIRParams(
        population        = 10_000,
        initial_infected  = 10,
        initial_recovered = 0,
        beta              = 0.30,
        gamma             = 0.10,
        t_end             = 160.0,
        t_steps           = 1_000,
    )

    seir_params = SEIRParams(
        population       = 10_000,
        initial_exposed  = 0,
        initial_infected = 10,
        beta             = 0.30,
        sigma            = 0.20,   # 1/sigma = mean incubation period (days)
        gamma            = 0.10,
        t_end            = 200.0,
        t_steps          = 1_000,
    )

    sepns_params = SEPNSParams(
        population           = 10_000,
        initial_exposed      = 0,
        initial_pos_infected = 5,
        initial_neg_infected = 5,
        alpha                = 0.20,
        beta1                = 0.15,
        beta2                = 0.20,
        mu1                  = 0.05,
        mu2                  = 0.05,
        mu_e                 = 0.03,
        t_end                = 200.0,
        t_steps              = 1_000,
    )

    sedis_params = SEDISParams(
        population       = 10_000,
        initial_exposed  = 0,
        initial_doubtful = 0,
        initial_infected = 10,
        alpha            = 0.20,
        beta1            = 0.10,
        beta2            = 0.15,
        gamma            = 0.08,
        mu1              = 0.04,
        mu2              = 0.05,
        mu3              = 0.05,
        t_end            = 200.0,
        t_steps          = 1_000,
    )

    sedpnr_params = SEDPNRParams(
        population           = 10_000,
        initial_exposed      = 10,
        initial_doubtful     = 10,
        initial_pos_infected = 5,
        initial_neg_infected = 5,
        initial_restrained   = 0,
        alpha                = 0.20,
        beta1                = 0.15,
        beta2                = 0.20,
        beta3                = 0.10,
        beta4                = 0.12,
        gamma                = 0.10,
        lambda1              = 0.05,
        lambda2              = 0.05,
        mu1                  = 0.03,
        mu2                  = 0.04,
        t_end                = 100.0,
        t_steps              = 1_000,
    )

    model_si(si_params)
    model_sis(sis_params)
    model_sir(sir_params)
    model_seir(seir_params)
    model_sepns(sepns_params)
    model_sedis(sedis_params)
    model_sedpnr(sedpnr_params)

    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()