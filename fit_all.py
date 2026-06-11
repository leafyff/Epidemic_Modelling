"""Fit every model in the project to a sample's observed I(t) curve.

Public API: ``fit_all_models(path)`` returns the per-model fit result
(estimates, standard errors, fitted curve, RMSE) and saves an overlay
plot of every model's fit to ``figs/<stem>_all_models.png``.

How the fits are set up
-----------------------
Real samples only contain ``I`` (the actively spreading / infectious
compartment). For each candidate model:

* observable = ``I`` (or ``P + N`` for the sentiment-aware SEPNS / SEDPNR)
* latent compartments (``E``, ``D``) are *free initial-condition*
  parameters — the optimiser estimates how many people were
  incubating / hesitating on day 0
* ``R`` / restrained starts at 0
* ``S(0)`` closes the population

Optimisation: a small multi-start of bounded least-squares fits +
a Differential-Evolution global safety net + a final Jacobian-scaled
``least_squares`` polish. The polish is mandatory because we need its
Jacobian for the standard-error calculation.

Standard errors (Nakonechny & Shevchuk 2020, Theorem 38)
--------------------------------------------------------
For the non-linear LS objective ``Φ(θ) = Σ_i (I_i - I_model(t_i; θ))²``
the Hessian at the optimum is well-approximated by the Gauss-Newton
matrix ``A = J'J`` where ``J = ∂I_model/∂θ`` is the residual Jacobian
returned by ``scipy.optimize.least_squares``. The per-component
guaranteed error of Theorem 38

    σ_k = √[(A⁻¹)_kk] · √(β − Φ(θ̂))

with the noise budget ``β = Φ(θ̂) · n/(n−p)`` reduces to the standard
non-linear LS standard error

    σ̂²    = Φ(θ̂) / (n − p)
    Cov  = σ̂² · (J'J)⁻¹
    SE_k = √diag(Cov)_k

``J'J`` is inverted with the Moore-Penrose pseudoinverse whenever it
is rank-deficient (typical for over-parameterised models like SEDPNR
on the 14-point flu sample).
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares

from drawing import FIGURE_DPI, FIGURE_SIZE, ensure_figs_dir
from estimation import _gcv_lambda, find_extrema
from models.SEDIS import sedis_ode
from models.SEDPNR import sedpnr_ode
from models.SEIR import seir_ode
from models.SEPNS import sepns_ode
from models.SI import si_ode
from models.SIR import sir_ode
from models.SIS import sis_ode
from models.modif_SEDIS import modif_sedis_ode

# Upper bound on rate constants. Kept generous because this is a curve
# comparison: a model should not lose merely because a physical rate
# cap squeezed it. SIR/SEIR need gamma ~ 4 to close a fast peak in 14
# days for the flu sample.
RATE_MAX = 6.0


# ---------------------------------------------------------------------------
# Per-model registry
# ---------------------------------------------------------------------------

def _registry(N: float) -> list[dict[str, Any]]:
    """Per-model spec: ode, rate names/bounds/initial guesses, compartment
    layout, observable, latent (free-IC) compartments, and whether N is
    the trailing ODE argument."""
    a = 1.0 / N  # modif_SEDIS alpha scale (alpha*S*I exposure)
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


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------

def _build_y0(spec: dict[str, Any], theta: np.ndarray, N: float, I0: float) -> list[float]:
    comp = {c: 0.0 for c in spec["layout"]}
    for c in spec["infected"]:
        comp[c] = I0 / len(spec["infected"])
    for i, c in enumerate(spec["latent"]):
        comp[c] = theta[len(spec["rates"]) + i]
    comp["S"] = N - sum(v for k, v in comp.items() if k != "S")
    return [comp[c] for c in spec["layout"]]


def _infected_curve(spec: dict[str, Any], sol: Any) -> np.ndarray:
    idx = [spec["layout"].index(c) for c in spec["infected"]]
    return sol.y[idx].sum(axis=0)


def _simulate(spec: dict[str, Any], theta: np.ndarray,
              t: np.ndarray, N: float, I0: float) -> np.ndarray:
    rates = tuple(theta[:len(spec["rates"])])
    args  = rates + ((N,) if spec["needs_N"] else ())
    try:
        sol = solve_ivp(
            spec["ode"], (t[0], t[-1]), _build_y0(spec, theta, N, I0),
            args=args, t_eval=t, method="LSODA",
            rtol=1e-7, atol=1e-6,
            max_step=max(1.0, (t[-1] - t[0]) / 50.0),
        )
    except Exception:
        return np.full_like(t, 1e18)
    if not sol.success or sol.y.shape[1] != t.shape[0]:
        return np.full_like(t, 1e18)
    return _infected_curve(spec, sol)


# ---------------------------------------------------------------------------
# One-model fit (params, SE, RMSE)
# ---------------------------------------------------------------------------

COND_TRUST = 1.0e12   # cond(J'J) above which a fit is flagged unidentifiable


def _fit_one(spec: dict[str, Any], t: np.ndarray, N: float,
             I0: float, I_obs: np.ndarray,
             loss: str = "abs",
             fixed: dict[str, float] | None = None,
             ridge: Any = "auto") -> dict[str, Any]:
    """Multi-start + DE + polish for one model.

    ``loss`` selects the optimisation objective:

    * ``"abs"`` (default) — OLS: raw residual ``I_model - I_obs`` (RMSE in
      people; the peak dominates).
    * ``"gls"`` — GLS: inverse-variance / relative residual
      ``(I_model - I_obs)/max(|I_obs|,1)``; down-weights the high-count peak
      so every scale of the curve is heard. Helps long noisy series with a
      bounded tail (e.g. COVID); on curves decaying to ~0 prefer ``"abs"``.
    * ``"rel"`` — backwards-compatible alias of ``"gls"`` (identical objective).
    * ``"log"`` — ``log1p(I_model) - log1p(I_obs)``; fits the growth shape.

    ``fixed`` pins named rates to given values and **removes them from the
    optimisation vector** — e.g. fixing the recovery rate ``gamma`` to a
    known infectious period breaks the ``beta``/``gamma`` collinearity when
    ``S`` stays ~ ``N`` (the COVID failure mode).

    ``ridge`` adds a Tikhonov penalty ``λ·Σ(θ_k/scale_k)²`` (scale = the
    parameter bounds) to the nonlinear objective. It may be a float
    (``0`` = off, ``>0`` = manual λ) or ``"auto"`` (default) — then λ is
    chosen by GCV on the local Jacobian, engaged only for genuinely
    **rank-deficient** models (``cond(J'J) > COND_TRUST``, the same boundary
    as the ``[unident.]`` flag). There it bounds the exploding SE without
    hurting RMSE; it is deliberately *not* applied to merely
    moderately-conditioned models (a nonlinear refit with a local-GCV λ can
    over-shrink them). It does **not** separate a merely collinear pair
    (β≈γ when S≈N) — use ``fixed`` for that. The SE then uses the
    regularised Gauss-Newton matrix ``J'J + λ·diag(1/scale²)``.

    The reported RMSE is always the absolute one (for comparability);
    parameter SE is in parameter units regardless of ``loss``. Also returns
    the applied ``ridge`` λ, AIC/BIC/AICc, ``cond(J'J)`` (raw diagnostic) and
    a ``trustworthy`` flag (``cond < COND_TRUST`` and finite AICc).
    """
    fixed    = fixed or {}
    nr, nl   = len(spec["rates"]), len(spec["latent"])
    free_pos = [i for i, r in enumerate(spec["rates"]) if r not in fixed]
    n_free   = len(free_pos)
    p        = n_free + nl                            # estimated parameters
    init_ub  = max(20.0 * I0, 100.0)
    lb       = [0.0] * p
    ub       = [spec["ub"][i] for i in free_pos] + [init_ub] * nl

    def expand(th: np.ndarray) -> np.ndarray:
        """Free optimisation vector -> full [rates..., latent...] for _simulate."""
        full = np.empty(nr + nl)
        fi = 0
        for i, r in enumerate(spec["rates"]):
            if r in fixed:
                full[i] = fixed[r]
            else:
                full[i] = th[fi]; fi += 1
        full[nr:] = th[n_free:]
        return full

    def transform(curve: np.ndarray) -> np.ndarray:
        # "gls" is the inverse-variance (proportional-noise) WLS form of GLS
        # for a single observed series; it is the same objective as "rel".
        if loss in ("gls", "rel"):
            return (curve - I_obs) / np.maximum(np.abs(I_obs), 1.0)
        if loss == "log":
            return np.log1p(np.maximum(curve, 0.0)) - np.log1p(I_obs)
        return curve - I_obs

    def resid(th: np.ndarray) -> np.ndarray:
        curve = _simulate(spec, expand(th), t, N, I0)
        if curve.max() > 1e17:                        # _simulate failure sentinel
            return np.full_like(t, 1e9)
        return transform(curve)

    def cost(th: np.ndarray) -> float:
        return float(np.sqrt(np.mean(resid(th) ** 2)))

    best_x: np.ndarray | None = None
    best_cost                  = np.inf

    def consider(x: Any) -> None:
        nonlocal best_x, best_cost
        c = cost(x)
        if c < best_cost:
            best_x, best_cost = np.asarray(x, dtype=float), c

    # ---- multi-start LS ----
    for m in (1.0, 3.0, 10.0, 30.0):
        rates = [spec["x0"][i] * m for i in free_pos]
        for e0 in (0.0, 5.0 * I0):
            x0 = np.clip(rates + [e0] * nl, lb, ub)
            for x_scale in ("jac", 1.0):
                try:
                    res = least_squares(resid, x0, bounds=(lb, ub),
                                        x_scale=x_scale, max_nfev=1500)
                except Exception:
                    continue
                consider(res.x)

    # ---- DE safety net ----
    if p > 0:
        try:
            de = differential_evolution(
                cost, list(zip(lb, ub)), maxiter=30, popsize=10, tol=1e-4,
                mutation=(0.4, 1.2), recombination=0.8, init="sobol",
                seed=0, polish=False,
            )
            consider(de.x)
            res = least_squares(resid, np.clip(de.x, lb, ub), bounds=(lb, ub),
                                x_scale="jac", max_nfev=2000)
            consider(res.x)
        except Exception:
            pass

    if best_x is None:
        best_x = np.clip(
            np.asarray([spec["x0"][i] for i in free_pos] + [0.0] * nl, dtype=float),
            lb, ub,
        )

    # ---- unregularised polish: Jacobian for the cond diagnostic + ridge choice ----
    n     = len(I_obs)
    J: np.ndarray | None = None
    r_final: np.ndarray  = resid(best_x)
    try:
        res0    = least_squares(resid, np.clip(best_x, lb, ub), bounds=(lb, ub),
                                x_scale="jac", max_nfev=5000)
        best_x  = np.asarray(res0.x, dtype=float)
        r_final = res0.fun
        J       = np.asarray(res0.jac)
    except Exception:
        pass

    cond = np.nan
    if J is not None and J.shape == (n, p) and p > 0:
        try:
            cond = float(np.linalg.cond(J.T @ J))
        except np.linalg.LinAlgError:
            cond = np.inf

    # ---- choose Tikhonov lambda (penalty on theta/scale, scale = bounds) ----
    scale = np.asarray(ub, dtype=float)
    scale[scale <= 0] = 1.0
    if isinstance(ridge, str):
        key = ridge.strip().lower()
        if key in ("auto", "gcv"):
            lam = 0.0
            if (J is not None and J.shape == (n, p) and p > 0
                    and np.isfinite(cond) and cond > COND_TRUST):
                try:                                  # GCV on the local linearisation
                    U, s, _ = np.linalg.svd(J * scale[None, :], full_matrices=False)
                    g       = U.T @ r_final
                    bperp   = max(float(r_final @ r_final - g @ g), 0.0)
                    lam     = _gcv_lambda(s, g, bperp, n)
                except np.linalg.LinAlgError:
                    lam = 0.0
        elif key in ("off", "none"):
            lam = 0.0
        else:
            lam = float(ridge)
    else:
        lam = float(ridge)

    # ---- ridge refit: augment residual with sqrt(lam) * theta/scale ----
    if lam > 0.0 and p > 0:
        sqrt_l = float(np.sqrt(lam))

        def resid_ridge(th: np.ndarray) -> np.ndarray:
            return np.concatenate([resid(th), sqrt_l * (th / scale)])

        try:
            resR    = least_squares(resid_ridge, np.clip(best_x, lb, ub),
                                    bounds=(lb, ub), x_scale="jac", max_nfev=5000)
            best_x  = np.asarray(resR.x, dtype=float)
            J       = np.asarray(resR.jac)[:n]        # data-only Jacobian
            r_final = resR.fun[:n]                    # data-only residual
        except Exception:
            pass

    full_x = expand(best_x)
    curve  = _simulate(spec, full_x, t, N, I0)
    rmse   = float(np.sqrt(np.mean((curve - I_obs) ** 2)))   # always absolute

    # ---- SE via ridge Gauss-Newton: M = J'J + lam*diag(1/scale^2) ----
    se_free = np.full(p, np.nan)
    if J is not None and J.shape == (n, p) and p > 0:
        dof    = max(n - p, 1)
        sigma2 = float(np.sum(r_final ** 2) / dof)
        M      = J.T @ J
        if lam > 0.0:
            M = M + lam * np.diag(1.0 / scale ** 2)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        cov     = sigma2 * M_inv
        se_free = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # ---- information criteria (from absolute RSS) + trust flag ----
    rss = float(np.sum((curve - I_obs) ** 2))
    k   = p + 1                                       # +1 for the variance
    if rss > 0.0 and n > 0:
        aic  = n * np.log(rss / n) + 2 * k
        bic  = n * np.log(rss / n) + np.log(n) * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if (n - k - 1) > 0 else np.inf
    else:
        aic = bic = aicc = np.inf
    trustworthy = bool(np.isfinite(cond) and cond < COND_TRUST and np.isfinite(aicc))

    # ---- expand theta/SE back to full (fixed rates shown, SE 0) ----
    param_names = list(spec["rates"]) + [f"{c}0" for c in spec["latent"]]
    full_se: list[float] = []
    fi = 0
    for r in spec["rates"]:
        if r in fixed:
            full_se.append(0.0)
        else:
            full_se.append(float(se_free[fi])); fi += 1
    full_se.extend(float(se_free[n_free + j]) for j in range(nl))

    return {
        "name"        : spec["name"],
        "rates"       : list(spec["rates"]),
        "latent"      : list(spec["latent"]),
        "fixed"       : dict(fixed),
        "param_names" : param_names,
        "theta"       : [float(v) for v in full_x],
        "se"          : full_se,
        "rmse"        : rmse,
        "aic"         : float(aic),
        "bic"         : float(bic),
        "aicc"        : float(aicc),
        "n_params"    : p,
        "jac_cond"    : cond,
        "ridge"       : float(lam),
        "trustworthy" : trustworthy,
        "curve"       : curve.tolist(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_all_models(path: str, save_plot: bool = True,
                   plot_suffix: str = "_all_models",
                   loss: str = "abs",
                   fixed: dict[str, float] | None = None,
                   ridge: Any = "auto",
                   show: bool = False) -> dict[str, Any]:
    """Fit every model to the sample's I(t); save an overlay plot.

    ``loss`` (``"abs"`` = OLS default / ``"gls"`` = GLS / ``"rel"`` alias /
    ``"log"``) selects the optimisation objective and ``fixed`` pins named
    rates (see ``_fit_one``). When ``show`` is true the overlay figure is left
    open (not closed) so a caller can ``plt.show()`` it in a window. Returns
    ``{"sample", "results", "plot_path", "loss", "fixed", "ridge"}``.
    ``results`` is sorted by RMSE ascending; each entry carries ``aicc`` and
    ``trustworthy``.
    """
    fixed = fixed or {}
    with open(path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    N      = float(sample["params"]["population"])
    t      = np.asarray(sample["time"], dtype=float)
    I_obs  = np.asarray(sample["compartments"]["I"], dtype=float)
    I0     = float(I_obs[0])
    peak   = float(I_obs.max())

    extra = f", loss={loss}, ridge={ridge}"
    if fixed:
        extra += ", fixed=" + ",".join(f"{k}={v:g}" for k, v in fixed.items())
    print(f"\nFitting all models to {os.path.basename(path)}  "
          f"(N={N:,.0f}, {len(t)} points, peak I={peak:,.0f}{extra})")

    extrema = find_extrema(t, I_obs)
    if extrema:
        ex_str = ", ".join(
            f"{e['type']} t={e['t']:.2f} (I={e['value']:,.0f})" for e in extrema
        )
        print(f"  Observed I(t) extrema (finite-diff derivative): {ex_str}")

    results: list[dict[str, Any]] = []
    for spec in _registry(N):
        r = _fit_one(spec, t, N, I0, I_obs, loss=loss, fixed=fixed, ridge=ridge)
        results.append(r)
        flag = "" if r["trustworthy"] else "  [unident.]"
        rdg  = f"  ridge={r['ridge']:.1e}" if r.get("ridge", 0.0) > 0 else ""
        aicc_str = "inf" if not np.isfinite(r["aicc"]) else f"{r['aicc']:.1f}"
        print(f"  {r['name']:<12} RMSE = {r['rmse']:11,.2f}  "
              f"({100 * r['rmse'] / peak:5.2f}% of peak)   AICc = {aicc_str:>8}{flag}{rdg}")

    results.sort(key=lambda r: r["rmse"])
    trust = [r for r in results if r["trustworthy"]]
    pool  = trust if trust else results
    best_aicc = min(pool, key=lambda r: r["aicc"])
    n_untrust = len(results) - len(trust)

    print(f"\nBest by RMSE : {results[0]['name']} (RMSE {results[0]['rmse']:,.2f})")
    print(f"Best by AICc : {best_aicc['name']} (AICc "
          f"{best_aicc['aicc']:.1f}, {best_aicc['n_params']} params"
          f"{'' if best_aicc['trustworthy'] else ', NONE trustworthy!'})")
    if n_untrust:
        print(f"  {n_untrust} model(s) flagged unidentifiable "
              f"(cond(J'J) > {COND_TRUST:.0e}) -> SE/params unreliable.")

    plot_path: str | None = None
    if save_plot:
        cmap   = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        ax.scatter(t, I_obs, color="#222222", s=20, zorder=6, label="Observed I(t)")
        for i, r in enumerate(results):
            ax.plot(t, r["curve"], lw=2.0, color=cmap(i % 10),
                    label=f"{r['name']}  (RMSE {r['rmse']:,.1f})")
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
        stem      = os.path.splitext(os.path.basename(path))[0]
        plot_path = os.path.join("figs", f"{stem}{plot_suffix}.png")
        fig.savefig(plot_path, dpi=FIGURE_DPI, bbox_inches="tight")
        if not show:                       # keep open so the caller can plt.show()
            plt.close(fig)
        print(f"\nSaved comparison plot -> {plot_path}")

    return {"sample": sample, "results": results, "plot_path": plot_path,
            "loss": loss, "fixed": fixed, "ridge": ridge}


def print_se_report(result: dict[str, Any]) -> None:
    """Pretty-print rate-parameter estimates with standard errors."""
    print("\n" + "=" * 70)
    print("Parameter estimates ± standard errors (rates only; latent ICs hidden)")
    print("=" * 70)
    for r in result["results"]:
        cond_str = "—" if not np.isfinite(r["jac_cond"]) else f"{r['jac_cond']:.2e}"
        aicc_str = "inf" if not np.isfinite(r["aicc"]) else f"{r['aicc']:.1f}"
        trust    = "OK" if r["trustworthy"] else "UNIDENTIFIABLE"
        fixed    = r.get("fixed", {})
        print(f"\n--- {r['name']:<12} RMSE = {r['rmse']:,.2f}   AICc = {aicc_str}   "
              f"cond(J'J) = {cond_str}   [{trust}] ---")
        print(f"  {'Parameter':<10} {'Estimate':>16} {'Std. error':>16}")
        for name, val, se in zip(r["param_names"], r["theta"], r["se"]):
            # Hide latent initial-condition rows (suffix "0"); they are
            # nuisance parameters, not model rates.
            if name.endswith("0") and name not in r["rates"]:
                continue
            tag = "  (fixed)" if name in fixed else ""
            print(f"  {name:<10} {val:>16.6g} {se:>16.4e}{tag}")
        # Also report latent ICs, but on a separate line.
        latent_strs = [
            f"{name}={val:.4g} (±{se:.2e})"
            for name, val, se in zip(r["param_names"], r["theta"], r["se"])
            if name.endswith("0") and name not in r["rates"]
        ]
        if latent_strs:
            print(f"  latent ICs : {', '.join(latent_strs)}")
