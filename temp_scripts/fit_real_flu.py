"""Compare SEDIS vs modif_SEDIS on a REAL non-COVID dataset.

Dataset: 1978 influenza A/H1N1 outbreak in an English boarding school
(Anonymous, British Medical Journal, 4 March 1978, p.587; widely
redistributed as ``outbreaks::influenza_england_1978_school`` in R).
N = 763 boys; the daily "confined to bed" count is the actively
infectious / spreading population I(t) over 14 days.

Because no real dataset measures the latent Exposed (E) and Doubtful (D)
compartments, the two models are compared the methodologically correct
way: each model's I(t) is fit to the observed I(t) by non-negative
non-linear least squares (E and D left as unobserved latent states),
and the resulting RMSE / curves are compared. The only structural
difference between the models is the exposure term -- SEDIS uses
alpha*S, modif_SEDIS uses the mass-action alpha*S*I -- so this directly
tests which exposure law better explains a real epidemic curve.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Allow running from anywhere (this script lives in temp_scripts/): put the
# project root on sys.path so the `drawing` and `models` packages resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing import COLORS, FIGURE_DPI, FIGURE_SIZE, ensure_figs_dir
from models.SEDIS import sedis_ode
from models.modif_SEDIS import modif_sedis_ode

# --- real data -----------------------------------------------------------
N = 763.0
DATES = [f"1978-01-{d:02d}" for d in range(22, 32)] + \
        [f"1978-02-{d:02d}" for d in range(1, 5)]
IN_BED = np.array([3, 8, 26, 76, 225, 298, 258, 233,
                   189, 128, 68, 29, 14, 4], dtype=float)
t = np.arange(len(IN_BED), dtype=float)          # days 0..13
I_obs = IN_BED

I0 = float(IN_BED[0])
y0 = [N - I0, 0.0, 0.0, I0]                       # S, E, D, I

PARAM_NAMES = ["alpha", "beta1", "beta2", "gamma", "mu1", "mu2", "mu3"]


def simulate_I(ode, theta):
    """Integrate a SEDIS-type model and return the I compartment at t."""
    sol = solve_ivp(ode, (t[0], t[-1]), y0, args=tuple(theta),
                    t_eval=t, method="RK45", rtol=1e-9, atol=1e-9)
    if sol.y.shape[1] != t.shape[0]:
        return np.full_like(t, 1e9)
    return sol.y[3]


def residuals(theta, ode):
    return simulate_I(ode, theta) - I_obs


def fit(ode, x0):
    res = least_squares(residuals, x0, args=(ode,),
                        bounds=(0.0, np.inf), max_nfev=20000)
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return dict(zip(PARAM_NAMES, [float(v) for v in res.x])), rmse


# initial guesses (modif alpha ~ 1/N smaller because of the extra I factor)
x0_sedis = [0.8, 0.4, 0.8, 0.3, 0.05, 0.05, 0.4]
x0_modif = [0.8 / N, 0.4, 0.8, 0.3, 0.05, 0.05, 0.4]

est_sedis, rmse_sedis = fit(sedis_ode, x0_sedis)
est_modif, rmse_modif = fit(modif_sedis_ode, x0_modif)

I_sedis = simulate_I(sedis_ode, [est_sedis[k] for k in PARAM_NAMES])
I_modif = simulate_I(modif_sedis_ode, [est_modif[k] for k in PARAM_NAMES])


def report(name, est, rmse):
    print(f"\n--- {name} fit to real 1978 influenza I(t) ---")
    for k in PARAM_NAMES:
        print(f"  {k:<6} = {est[k]:.6g}")
    print(f"  RMSE (in-bed count) = {rmse:.3f}")


report("SEDIS  (exposure alpha*S)",   est_sedis, rmse_sedis)
report("modif_SEDIS (alpha*S*I)",      est_modif, rmse_modif)
winner = "SEDIS" if rmse_sedis < rmse_modif else "modif_SEDIS"
print(f"\nBetter fit to the real curve: {winner} "
      f"(RMSE {min(rmse_sedis, rmse_modif):.3f} vs "
      f"{max(rmse_sedis, rmse_modif):.3f})")

# --- plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
ax.scatter(t, I_obs, color="#222222", s=45, zorder=5,
           label="Observed (boys in bed)")
ax.plot(t, I_sedis, color=COLORS["I"], lw=2.4,
        label=f"SEDIS fit (RMSE {rmse_sedis:.1f})")
ax.plot(t, I_modif, color=COLORS["S"], lw=2.4, ls="--",
        label=f"modif_SEDIS fit (RMSE {rmse_modif:.1f})")
ax.set_title("SEDIS vs modif_SEDIS on real data\n"
             "1978 English boarding-school influenza (N=763)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Day", fontsize=11)
ax.set_ylabel("Infected (confined to bed)", fontsize=11)
ax.grid(True, ls="--", lw=0.5, alpha=0.7)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

ensure_figs_dir()
out = "figs/flu1978_SEDIS_vs_modif.png"
fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
print(f"\nSaved comparison plot -> {out}")

# --- save the real series as a JSON sample (provenance) ------------------
sample = {
    "model": "SEDIS",
    "params": {
        "population": int(N),
        "initial_infected": int(I0),
        "source": "BMJ 1978-03-04 p.587 / outbreaks::influenza_england_1978_school",
        "event": "1978 English boarding-school influenza A/H1N1",
        "I_def": "confined-to-bed count (active infectious)",
        "note": "E and D are latent (unobserved); fit to I(t) only",
    },
    "n_points": int(len(t)),
    "dates": DATES,
    "time": t.tolist(),
    "compartments": {"I": I_obs.tolist()},
}
with open("samples/flu1978_school.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2)
print("Saved real-data sample -> samples/flu1978_school.json")
