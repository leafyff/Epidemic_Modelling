"""Real-data test (COVID-2020, flu-1978): does GLS/WLS weighting beat OLS?

Real data HAS noise but NO known truth -> we use a neutral, non-circular
metric: time-split hold-out forecasting. Fit each weight scheme on the
first `train_frac` of the curve, integrate the fitted model over the full
horizon, and score the prediction on the held-out tail. We report BOTH
absolute RMSE (favours high-count fit) and relative error (favours
low-count fit) so neither scheme's "home" metric decides the verdict.

Model is held fixed across schemes; only the weighting changes.
"""
import json
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, ".")
from fit_all import _registry, _simulate


def get_spec(name, N):
    return next(s for s in _registry(N) if s["name"] == name)


def fit_train(spec, t, N, I0, y, w, fixed=None):
    fixed = fixed or {}
    nr, nl = len(spec["rates"]), len(spec["latent"])
    free = [i for i, r in enumerate(spec["rates"]) if r not in fixed]
    nf, p = len(free), len(free) + nl
    lb = [0.0] * p
    ub = [spec["ub"][i] for i in free] + [max(20.0 * I0, 100.0)] * nl
    sw = np.sqrt(w)

    def expand(th):
        full = np.empty(nr + nl)
        fi = 0
        for i, r in enumerate(spec["rates"]):
            if r in fixed:
                full[i] = fixed[r]
            else:
                full[i] = th[fi]; fi += 1
        full[nr:] = th[nf:]
        return full

    def resid(th):
        c = _simulate(spec, expand(th), t, N, I0)
        return sw * (c - y) if c.max() < 1e17 else np.full_like(t, 1e9)

    best, bc = None, np.inf
    for m in (1.0, 3.0, 10.0):
        x0 = np.clip([spec["x0"][i] * m for i in free] + [2.0 * I0] * nl, lb, ub)
        try:
            r = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac", max_nfev=4000)
            c = float(np.sum(r.fun ** 2))
            if c < bc:
                best, bc = r.x, c
        except Exception:
            pass
    return expand(np.asarray(best)) if best is not None else expand(np.asarray(x0))


def holdout(path, model, train_frac=0.7, fixed=None):
    d = json.load(open(path, encoding="utf-8"))
    N = float(d["params"]["population"])
    t = np.asarray(d["time"], float)
    y = np.asarray(d["compartments"]["I"], float)
    k = int(len(t) * train_frac)
    ttr, ytr = t[:k], y[:k]
    I0 = float(ytr[0])
    spec = get_spec(model, N)

    schemes = {
        "ols": np.ones_like(ytr),
        "poisson": 1.0 / np.maximum(ytr, 1.0),
        "prop": 1.0 / np.maximum(ytr, 1.0) ** 2,
    }
    # feasible GLS: pilot OLS -> estimate Var ~ mu^q from fitted means -> w = 1/mu^q
    th0 = fit_train(spec, ttr, N, I0, ytr, np.ones_like(ytr), fixed)
    mu = np.maximum(_simulate(spec, th0, ttr, N, I0), 1.0)
    r2 = (ytr - mu) ** 2 + 1e-9
    A = np.vstack([np.ones_like(mu), np.log(mu)]).T
    q = float(np.clip(np.linalg.lstsq(A, np.log(r2), rcond=None)[0][1], 0.0, 2.0))
    schemes["fgls"] = 1.0 / mu ** q

    peak = float(y.max())
    print(f"\n=== {model} on {path.split('/')[-1]} | train={k}/{len(t)} pts, "
          f"predict last {len(t)-k} | peak I={peak:,.0f}"
          f"{' | fixed='+str(fixed) if fixed else ''} | fGLS q~{q:.2f} ===")
    print(f"  {'scheme':<9}{'holdout RMSE(abs)':>18}{'holdout rel.err':>17}   rates/R0")
    base_abs = base_rel = None
    for s, w in schemes.items():
        th = fit_train(spec, ttr, N, I0, ytr, w, fixed)
        full = _simulate(spec, th, t, N, I0)
        pred, obs = full[k:], y[k:]
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        rel = float(np.mean(np.abs(pred - obs) / np.maximum(np.abs(obs), 1.0)))
        rates = dict(zip(spec["rates"], th[:len(spec["rates"])]))
        if "beta" in rates and "gamma" in rates and rates["gamma"] > 0:
            extra = f"R0={rates['beta']/rates['gamma']:.2f} (b={rates['beta']:.3g},g={rates['gamma']:.3g})"
        else:
            extra = ",".join(f"{kk}={vv:.3g}" for kk, vv in list(rates.items())[:3])
        if s == "ols":
            base_abs, base_rel = rmse, rel
            ta = tr = "  (baseline)"
        else:
            ta = f" ({100*(rmse-base_abs)/base_abs:+.0f}%)"
            tr = f" ({100*(rel-base_rel)/base_rel:+.0f}%)"
        print(f"  {s:<9}{rmse:>14,.1f}{ta:<8}{rel:>10.1%}{tr:<8}   {extra}")


if __name__ == "__main__":
    # flu: small N -> S varies a lot -> beta,gamma identifiable (clean real case)
    holdout("samples/flu1978_school.json", "SIR", train_frac=0.7)
    holdout("samples/flu1978_school.json", "SEIR", train_frac=0.7)
    # COVID: huge N -> S~N -> fix gamma to remove the beta/gamma collinearity
    # confound, so the comparison isolates the WEIGHTING effect.
    holdout("samples/COVID_Germany_2020.json", "SIR", train_frac=0.7, fixed={"gamma": 0.1})
    holdout("samples/COVID_Germany_2020.json", "SIS", train_frac=0.7, fixed={"gamma": 0.1})
