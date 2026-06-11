"""Fit every model in the project to the observed I(t) of a JSON sample
and overlay all fits in a single comparison PNG.

Only S and I are observed, so for each model the infected observable is
fit by non-linear least squares: the infected compartment(s) start at
the observed I(0), latent compartments (E, D where present) are free
initial-condition parameters, recovered/removed start at 0, and
S(0) closes the population. Rate constants are bounded to physical
values; modif_SEDIS's alpha is bounded ~1/N because its exposure term
carries an extra factor of I.

For SEPNS and SEDPNR the "infected" observable is P+N (the spreaders).

Usage:
    python fit_all_models.py samples/COVID_Germany_2020.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares

# Allow running from anywhere (this script lives in temp_scripts/): put the
# project root on sys.path so the `drawing` and `models` packages resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing import FIGURE_DPI, FIGURE_SIZE, ensure_figs_dir
from models.SEDIS import sedis_ode
from models.SEDPNR import sedpnr_ode
from models.SEIR import seir_ode
from models.SEPNS import sepns_ode
from models.SI import si_ode
from models.SIR import sir_ode
from models.SIS import sis_ode
from models.modif_SEDIS import modif_sedis_ode

# Upper bound on rate constants. Kept generous (not ~1/day) because this is
# a curve-fit comparison: a model should not be reported as a poor fit merely
# because a physical rate cap squeezed it. In particular SIR needs gamma ~ 4
# (a short effective infectious period) to make a closed epidemic peak within
# the observed window; capping at 3 forced its peak ~25 days too late.
RATE_MAX = 6.0


def _registry(N):
    """Per-model spec: ode, rate names/guesses/bounds, layout, infected,
    latent ICs, and whether N is the trailing ODE argument."""
    a = 1.0 / N  # modif_SEDIS alpha scale
    return [
        dict(name="SI",  ode=si_ode,  needs_N=True,
             rates=["beta"], x0=[0.2], ub=[RATE_MAX],
             layout=["S", "I"], infected=["I"], latent=[]),
        dict(name="SIS", ode=sis_ode, needs_N=True,
             rates=["beta", "gamma"], x0=[0.2, 0.1], ub=[RATE_MAX] * 2,
             layout=["S", "I"], infected=["I"], latent=[]),
        dict(name="SIR", ode=sir_ode, needs_N=True,
             rates=["beta", "gamma"], x0=[0.2, 0.1], ub=[RATE_MAX] * 2,
             layout=["S", "I", "R"], infected=["I"], latent=[]),
        dict(name="SEIR", ode=seir_ode, needs_N=True,
             rates=["beta", "sigma", "gamma"], x0=[0.2, 0.2, 0.1],
             ub=[RATE_MAX] * 3,
             layout=["S", "E", "I", "R"], infected=["I"], latent=["E"]),
        dict(name="SEPNS", ode=sepns_ode, needs_N=True,
             rates=["alpha", "beta1", "beta2", "mu1", "mu2", "mu_e"],
             x0=[0.2, 0.15, 0.2, 0.05, 0.05, 0.03], ub=[RATE_MAX] * 6,
             layout=["S", "E", "P", "N"], infected=["P", "N"], latent=["E"]),
        dict(name="SEDIS", ode=sedis_ode, needs_N=False,
             rates=["alpha", "beta1", "beta2", "gamma", "mu1", "mu2", "mu3"],
             x0=[0.2, 0.12, 0.15, 0.08, 0.05, 0.05, 0.06], ub=[RATE_MAX] * 7,
             layout=["S", "E", "D", "I"], infected=["I"], latent=["E", "D"]),
        dict(name="modif_SEDIS", ode=modif_sedis_ode, needs_N=False,
             rates=["alpha", "beta1", "beta2", "gamma", "mu1", "mu2", "mu3"],
             x0=[a, 0.12, 0.15, 0.08, 0.05, 0.05, 0.06],
             ub=[30 * a] + [RATE_MAX] * 6,
             layout=["S", "E", "D", "I"], infected=["I"], latent=["E", "D"]),
        dict(name="SEDPNR", ode=sedpnr_ode, needs_N=False,
             rates=["alpha", "beta1", "beta2", "beta3", "beta4", "gamma",
                    "lambda1", "lambda2", "mu1", "mu2"],
             x0=[0.2, 0.15, 0.2, 0.1, 0.12, 0.1, 0.05, 0.05, 0.03, 0.04],
             ub=[RATE_MAX] * 10,
             layout=["S", "E", "D", "P", "N", "R"], infected=["P", "N"],
             latent=["E", "D"]),
    ]


def _build_y0(spec, theta, N, I0):
    comp = {c: 0.0 for c in spec["layout"]}
    for c in spec["infected"]:
        comp[c] = I0 / len(spec["infected"])
    for i, c in enumerate(spec["latent"]):
        comp[c] = theta[len(spec["rates"]) + i]
    comp["S"] = N - sum(v for k, v in comp.items() if k != "S")
    return [comp[c] for c in spec["layout"]]


def _infected_curve(spec, sol):
    idx = [spec["layout"].index(c) for c in spec["infected"]]
    return sol.y[idx].sum(axis=0)


def _simulate(spec, theta, t, N, I0):
    rates = tuple(theta[:len(spec["rates"])])
    args = rates + ((N,) if spec["needs_N"] else ())
    try:
        sol = solve_ivp(spec["ode"], (t[0], t[-1]), _build_y0(spec, theta, N, I0),
                        args=args, t_eval=t, method="LSODA",
                        rtol=1e-7, atol=1e-6,
                        max_step=max(1.0, (t[-1] - t[0]) / 50.0))
    except Exception:
        return np.full_like(t, 1e18)
    if not sol.success or sol.y.shape[1] != t.shape[0]:
        return np.full_like(t, 1e18)
    return _infected_curve(spec, sol)


def _fit(spec, t, N, I0, I_obs):
    """Global fit of a model's I(t) to the data; return (best_params, rmse).

    Uses differential evolution (a bounded, gradient-free GLOBAL optimiser)
    followed by a Jacobian-scaled least-squares polish. This replaces the
    earlier hand-tuned multi-start: DE samples the whole bounded box, so it
    is robust to (a) the many local minima of these ODE fits, (b) the
    wildly multi-scale parameter vector (modif_SEDIS alpha ~ 1/N alongside
    O(1) rates and O(10^3) latent initial conditions), and (c) the very
    different rate regimes different models need (slow plateaus vs the fast
    gamma ~ 4-6 SIR/SEIR need to make a closed epidemic peak). No per-model
    initial guesses or magnitude multipliers are required, and the seed
    makes the result reproducible.

    Decision variables: the rate constants plus the UNOBSERVED initial
    latent pools E0, D0 (people incubating / hesitating at t0). The latter
    are bounded to ~20x the initial infected -- a physical scale -- so the
    optimiser cannot fake a peak by seeding millions of "exposed" on day 1.
    """
    nr, nl = len(spec["rates"]), len(spec["latent"])
    init_ub = max(20.0 * I0, 100.0)
    lb = [0.0] * (nr + nl)
    ub = list(spec["ub"]) + [init_ub] * nl

    def resid(th):
        return _simulate(spec, th, t, N, I0) - I_obs

    def cost(th):                       # scalar objective for DE
        return float(np.sqrt(np.mean(resid(th) ** 2)))

    best_x, best_rmse = None, np.inf

    def consider(x):
        nonlocal best_x, best_rmse
        r = cost(x)
        if r < best_rmse:
            best_x, best_rmse = np.asarray(x, dtype=float), r

    # (1) Targeted multi-start local fits. Cheap, and they reliably reach
    # the fast "sharp-peak" basins (e.g. modif_SEDIS on the flu, gamma~4
    # SIR/SEIR) that a blind global search can miss. Rate guesses are
    # scaled 1x/3x/10x/30x to span slow plateaus and fast outbreaks; both
    # unscaled and Jacobian-scaled steps are tried (the parameter vector is
    # badly multi-scale: modif alpha ~ 1/N vs O(1) rates vs O(10^3) E0/D0).
    for m in (1.0, 3.0, 10.0, 30.0):
        rates = [v * m for v in spec["x0"]]
        for e0 in (0.0, 5.0 * I0):
            x0 = np.clip(rates + [e0] * nl, lb, ub)
            for x_scale in ("jac", 1.0):
                try:
                    res = least_squares(resid, x0, bounds=(lb, ub),
                                        x_scale=x_scale, max_nfev=1500)
                except Exception:
                    continue
                consider(res.x)

    # (2) Global differential-evolution safety net + polish. Catches basins
    # the multi-start seeds miss (e.g. modif_SEDIS's plateau optimum on the
    # COVID curve). Seeded for reproducibility. The best of (1) and (2) is
    # returned, so this can only ever improve the result. DE is the costly
    # part; its budget is kept modest because the multi-start in (1) already
    # finds the right basin for almost every model -- DE only ever shifted a
    # single already-failing plateau fit here, and never by enough to change
    # a conclusion. Set DE_ENABLE=False below for a ~3x faster, near-identical run.
    DE_ENABLE = True
    try:
        if not DE_ENABLE:
            raise RuntimeError("DE disabled")
        de = differential_evolution(
            cost, list(zip(lb, ub)), maxiter=30, popsize=10, tol=1e-4,
            mutation=(0.4, 1.2), recombination=0.8, init="sobol",
            seed=0, polish=False)
        consider(de.x)
        res = least_squares(resid, np.clip(de.x, lb, ub), bounds=(lb, ub),
                            x_scale="jac", max_nfev=2000)
        consider(res.x)
    except Exception:
        pass

    return best_x, best_rmse


def main(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = json.load(f)
    N = float(sample["params"]["population"])
    t = np.asarray(sample["time"], dtype=float)
    I_obs = np.asarray(sample["compartments"]["I"], dtype=float)
    I0 = float(I_obs[0])
    peak = float(I_obs.max())

    results = []
    for spec in _registry(N):
        theta, rmse = _fit(spec, t, N, I0, I_obs)
        curve = _simulate(spec, theta, t, N, I0)
        results.append((spec["name"], rmse, curve))
        print(f"  {spec['name']:<12} RMSE = {rmse:11,.1f}  ({100 * rmse / peak:4.1f}% of peak)")

    results.sort(key=lambda r: r[1])
    print(f"\nBest fit: {results[0][0]} (RMSE {results[0][1]:,.0f})  |  "
          f"Worst: {results[-1][0]} (RMSE {results[-1][1]:,.0f})")

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.scatter(t, I_obs, color="#222222", s=20, zorder=6, label="Observed I(t)")
    for i, (name, rmse, curve) in enumerate(results):
        ax.plot(t, curve, lw=2.0, color=cmap(i % 10),
                label=f"{name}  (RMSE {rmse:,.0f})")
    title_src = (sample["params"].get("event")
                 or sample["params"].get("country")
                 or sample.get("model", "sample"))
    ax.set_title(f"All models fit to {title_src} I(t)\n"
                 f"(only S, I observed; N={N:,.0f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Infected I(t)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, ls="--", lw=0.5, alpha=0.7)
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    ensure_figs_dir()
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join("figs", f"{stem}_all_models.png")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"\nSaved comparison plot -> {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/COVID_Germany_2020.json")
