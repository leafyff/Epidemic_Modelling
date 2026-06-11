"""Fit SEDIS and modif_SEDIS to the I(t) curve of any JSON sample and
compare them, using ONLY the observed S and I.

Real data does not measure the latent Exposed (E) and Doubtful (D)
compartments, so this fitter treats them as unknowns:

  * I(0) is fixed to the observed first value.
  * E(0) and D(0) are *free parameters* (the unobserved initial latent
    pools), and S(0) = N - I(0) - E(0) - D(0) closes the population.
  * the 7 rate constants are fit under physical upper bounds
    (a rate > a few per day is meaningless for these models), so the
    optimiser cannot wander off to degenerate values such as mu = 300.
  * several starting points are tried (multi-start) and the best kept,
    because least_squares is a local method.

The objective is the residual on the observed I(t) only -- that is all
we can fit when E and D are unobserved. The two models differ only in
their exposure term (SEDIS: alpha*S, modif_SEDIS: alpha*S*I), so the
comparison isolates which exposure law better explains the curve.
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
from scipy.optimize import least_squares

# Allow running from anywhere (this script lives in temp_scripts/): put the
# project root on sys.path so the `drawing` and `models` packages resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing import COLORS, FIGURE_DPI, FIGURE_SIZE, ensure_figs_dir
from models.SEDIS import sedis_ode
from models.modif_SEDIS import modif_sedis_ode

RATE_NAMES = ["alpha", "beta1", "beta2", "gamma", "mu1", "mu2", "mu3"]
PARAM_NAMES = RATE_NAMES + ["E0", "D0"]
RATE_MAX = 3.0   # per day; an upper bound that is already very fast


def _y0_from(theta, N, I0):
    """Build [S0, E0, D0, I0] from a parameter vector (last two are E0, D0)."""
    E0, D0 = theta[7], theta[8]
    return [N - I0 - E0 - D0, E0, D0, I0]


def _simulate_I(ode, theta, t, N, I0):
    # LSODA auto-handles stiff regions the optimiser may probe; max_step
    # keeps any single integration bounded so the fit can never hang.
    try:
        sol = solve_ivp(ode, (t[0], t[-1]), _y0_from(theta, N, I0),
                        args=tuple(theta[:7]), t_eval=t,
                        method="LSODA", rtol=1e-6, atol=1e-6,
                        max_step=max(1.0, (t[-1] - t[0]) / 50.0))
    except Exception:
        return np.full_like(t, 1e18)
    if not sol.success or sol.y.shape[1] != t.shape[0]:
        return np.full_like(t, 1e18)
    return sol.y[3]


def _fit(ode, starts, t, N, I0, I_obs):
    """Multi-start bounded least squares; return (best_params, rmse)."""
    # Physically bound the unobserved initial latent pools (E0, D0): people
    # incubating/hesitating at t0 are on the order of the initial infected,
    # not millions. Without this the optimiser fakes a peak by seeding a
    # huge initial reservoir that drains into I.
    init_ub = max(20.0 * I0, 100.0)
    lb = [0.0] * 9
    ub = [RATE_MAX] * 7 + [init_ub, init_ub]

    best = None
    for x0 in starts:
        x0 = np.clip(x0, lb, ub)
        try:
            res = least_squares(
                lambda th: _simulate_I(ode, th, t, N, I0) - I_obs,
                x0, bounds=(lb, ub), max_nfev=500)
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rmse < best[1]:
            best = (res.x, rmse)
    params = dict(zip(PARAM_NAMES, [float(v) for v in best[0]]))
    return params, best[1]


def _starts(alpha_scale, I0):
    """A small grid of starting points (alpha magnitude x latent seeds)."""
    base = [0.2, 0.12, 0.15, 0.08, 0.05, 0.05, 0.06]
    out = []
    for a, e0 in ((0.3, 0.0), (1.0, 5.0 * I0), (2.0, 50.0 * I0)):
        out.append([a * alpha_scale] + base[1:] + [e0, 0.0])
    return out


def compare(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    N = float(sample["params"]["population"])
    t = np.asarray(sample["time"], dtype=float)
    I_obs = np.asarray(sample["compartments"]["I"], dtype=float)
    I0 = float(I_obs[0])

    est_s, rmse_s = _fit(sedis_ode,       _starts(1.0,     I0), t, N, I0, I_obs)
    est_m, rmse_m = _fit(modif_sedis_ode, _starts(1.0 / N, I0), t, N, I0, I_obs)
    I_s = _simulate_I(sedis_ode,       [est_s[k] for k in PARAM_NAMES], t, N, I0)
    I_m = _simulate_I(modif_sedis_ode, [est_m[k] for k in PARAM_NAMES], t, N, I0)

    for name, est, rmse in [("SEDIS  (alpha*S)", est_s, rmse_s),
                            ("modif_SEDIS (alpha*S*I)", est_m, rmse_m)]:
        print(f"\n--- {name} fit to {os.path.basename(path)} I(t) ---")
        for k in PARAM_NAMES:
            print(f"  {k:<6} = {est[k]:.6g}")
        print(f"  RMSE = {rmse:.3f}")
    winner = "SEDIS" if rmse_s < rmse_m else "modif_SEDIS"
    print(f"\nBetter fit: {winner} "
          f"(RMSE {min(rmse_s, rmse_m):.3f} vs {max(rmse_s, rmse_m):.3f})")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.scatter(t, I_obs, color="#222222", s=22, zorder=5, label="Observed I(t)")
    ax.plot(t, I_s, color=COLORS["I"], lw=2.4,
            label=f"SEDIS fit (RMSE {rmse_s:,.0f})")
    ax.plot(t, I_m, color=COLORS["S"], lw=2.4, ls="--",
            label=f"modif_SEDIS fit (RMSE {rmse_m:,.0f})")
    title_src = (sample["params"].get("event")
                 or sample["params"].get("country")
                 or sample.get("model", "?"))
    ax.set_title(f"SEDIS vs modif_SEDIS (fit to I, E/D latent)\n"
                 f"{title_src}  (N={N:,.0f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Infected I(t)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, ls="--", lw=0.5, alpha=0.7)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    ensure_figs_dir()
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join("figs", f"{stem}_SEDIS_vs_modif.png")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"\nSaved comparison plot -> {out}")
    return out


if __name__ == "__main__":
    compare(sys.argv[1] if len(sys.argv) > 1 else "samples/COVID_Germany_2020.json")
