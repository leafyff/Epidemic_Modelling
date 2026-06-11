"""Empirical test: does GLS/WLS weighting beat OLS for the fit-all nonlinear fit?

We take a synthetic curve with KNOWN true parameters, inject realistic
heteroscedastic noise, then fit the *correct* model with several weight
schemes and measure how well each recovers (a) the true parameters and
(b) the true (noise-free) curve. Averaged over many noise realisations.

Weight schemes for the residual r_i = sqrt(w_i) * (I_model_i - y_i):
  ols      w_i = 1                       (current fit-all default, loss=abs)
  poisson  w_i = 1/max(y_i,1)            (var proportional to mean; count data)
  prop     w_i = 1/max(y_i,1)^2          (var prop to mean^2; == loss=rel)
  fgls     feasible GLS: estimate the variance power from OLS residuals, reweight
"""
import json
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, ".")
from fit_all import _registry, _simulate


def get_spec(name, N):
    for s in _registry(N):
        if s["name"] == name:
            return s
    raise KeyError(name)


def fit(spec, t, N, I0, y, w):
    """Bounded multi-start weighted-LS fit of one model. Returns theta_full."""
    nr, nl = len(spec["rates"]), len(spec["latent"])
    p = nr + nl
    lb = [0.0] * p
    ub = [spec["ub"][i] for i in range(nr)] + [max(20.0 * I0, 100.0)] * nl
    sw = np.sqrt(w)

    def resid(th):
        curve = _simulate(spec, th, t, N, I0)
        if curve.max() > 1e17:
            return np.full_like(t, 1e9)
        return sw * (curve - y)

    best, bestc = None, np.inf
    for m in (1.0, 3.0, 10.0):
        x0 = np.clip([spec["x0"][i] * m for i in range(nr)] + [2.0 * I0] * nl, lb, ub)
        try:
            r = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac", max_nfev=3000)
            c = float(np.sum(r.fun ** 2))
            if c < bestc:
                best, bestc = r.x, c
        except Exception:
            pass
    return np.asarray(best) if best is not None else np.asarray(x0)


def fgls_weights(spec, t, N, I0, y):
    """Feasible GLS: OLS pilot -> estimate Var ~ mu^q -> w = 1/mu^q."""
    th = fit(spec, t, N, I0, y, np.ones_like(y))
    mu = _simulate(spec, th, t, N, I0)
    mu = np.maximum(mu, 1.0)
    r2 = (y - mu) ** 2 + 1e-9
    # regress log r2 on log mu  -> slope q is the variance power
    A = np.vstack([np.ones_like(mu), np.log(mu)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(r2), rcond=None)
    q = float(np.clip(coef[1], 0.0, 2.0))
    return 1.0 / mu ** q, q


def run(sample_path, model, noise, level, seeds=40, n_obs=60):
    d = json.load(open(sample_path, encoding="utf-8"))
    N = float(d["params"]["population"])
    t_full = np.asarray(d["time"], dtype=float)
    I_full = np.asarray(d["compartments"]["I"], dtype=float)
    # subsample to n_obs evenly spaced points (mimics real reporting cadence)
    idx = np.linspace(0, len(t_full) - 1, n_obs).round().astype(int)
    t, I_true = t_full[idx], I_full[idx]
    I0 = float(I_true[0])
    spec = get_spec(model, N)
    true = {r: float(d["params"][r]) for r in spec["rates"] if r in d["params"]}

    schemes = ["ols", "poisson", "prop", "fgls"]
    perr = {s: [] for s in schemes}      # mean relative param error
    crec = {s: [] for s in schemes}      # RMSE(fit curve, TRUE curve)
    qhat = []

    rng = np.random.default_rng(0)
    for _ in range(seeds):
        if noise == "poisson":          # Var = level * mean  (count-data style)
            y = I_true + np.sqrt(level * np.maximum(I_true, 1.0)) * rng.standard_normal(len(t))
        else:                            # proportional: Var = (level*mean)^2
            y = I_true * (1.0 + level * rng.standard_normal(len(t)))
        y = np.maximum(y, 0.0)

        wmap = {
            "ols": np.ones_like(y),
            "poisson": 1.0 / np.maximum(y, 1.0),
            "prop": 1.0 / np.maximum(y, 1.0) ** 2,
        }
        wf, q = fgls_weights(spec, t, N, I0, y)
        wmap["fgls"] = wf
        qhat.append(q)

        for s in schemes:
            th = fit(spec, t, N, I0, y, wmap[s])
            est = dict(zip(spec["rates"], th[:len(spec["rates"])]))
            pe = np.mean([abs(est[k] - true[k]) / abs(true[k]) for k in true])
            perr[s].append(pe)
            curve = _simulate(spec, th, t, N, I0)
            crec[s].append(np.sqrt(np.mean((curve - I_true) ** 2)))

    print(f"\n=== {model} on {sample_path.split('/')[-1]} | noise={noise} "
          f"level={level} | n_obs={n_obs} | seeds={seeds} ===")
    print(f"  true params: {true}")
    if noise == "poisson":
        print(f"  (optimal weight for Var~mean is 1/mean -> 'poisson'; fGLS q_hat~{np.mean(qhat):.2f})")
    else:
        print(f"  (optimal weight for Var~mean^2 is 1/mean^2 -> 'prop'; fGLS q_hat~{np.mean(qhat):.2f})")
    print(f"  {'scheme':<9}{'param rel.err (mean+-sd)':<30}{'curve-recovery RMSE':<22}")
    base_pe = np.mean(perr["ols"])
    base_cr = np.mean(crec["ols"])
    for s in schemes:
        pe_m, pe_s = np.mean(perr[s]), np.std(perr[s])
        cr_m = np.mean(crec[s])
        tagp = f"  ({100*(pe_m-base_pe)/base_pe:+.0f}% vs OLS)" if s != "ols" else "  (baseline)"
        tagc = f"  ({100*(cr_m-base_cr)/base_cr:+.0f}%)" if s != "ols" else ""
        print(f"  {s:<9}{pe_m:>8.1%} +- {pe_s:>6.1%}{'':<8}{cr_m:>12.2f}{tagc}{tagp}")


if __name__ == "__main__":
    # SIR (2 params) is the decisive cheap case -> many seeds, both noise regimes.
    # SEIR (4 params) and SEDIS (9 params) confirm it generalises -> fewer seeds.
    run("samples/SIR_fresh.json", "SIR", "poisson", level=8.0, seeds=40)
    run("samples/SIR_fresh.json", "SIR", "prop", level=0.15, seeds=40)
    run("samples/SIR_fresh.json", "SEIR", "poisson", level=8.0, seeds=10)
    run("samples/SEDIS_sample1000.json", "SEDIS", "poisson", level=8.0, seeds=12)
