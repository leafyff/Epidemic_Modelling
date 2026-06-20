"""Extended Kalman filter (EKF) for tracking a time-varying transmission rate
from an ``I(t)``-only series.

Motivation
----------
``fit_all`` recovers *constant* rate parameters by nonlinear least squares: it
forward-simulates each model's ODE and minimises the mismatch to the observed
infected curve. That is the right tool for a point estimate and for ranking
models, but it cannot answer the question every real outbreak raises —
**did the transmission rate change over time?** (an intervention, a behaviour
shift, a new variant). The honest caveat already in ``Documentation.md`` says
as much: when ``S ≈ N`` a constant-rate model cannot reproduce an
intervention-driven decline, and "the remaining misfit is the genuine signal
that a time-varying ``β(t)`` is needed".

Because the observation map (simulate the ODE, read off ``I``) is *nonlinear*
in the parameters, the recursive estimator is no longer plain RLS but its
nonlinear generalisation, the **extended Kalman filter**. This module
implements the EKF for *joint state + parameter* estimation, specialised to
the epidemiologically meaningful and **observable** case:

* the filter state is the model's compartment vector **augmented with the
  single transmission / exposure rate** (``beta`` for SI/SIS/SIR/SEIR,
  ``alpha`` for the misinformation models — always ``spec["rates"][0]``);
* every *other* rate is held fixed at the batch-LS value (estimating all
  rates jointly from ``I(t)`` alone is unobservable — see the note below);
* the augmented rate follows a random walk ``β_{k+1} = β_k + w_k`` whose
  process-noise variance is the tuning knob (the EKF analogue of the RLS
  forgetting factor): more process noise ⇒ faster tracking, noisier estimate.

The state propagates through the model ODE by sub-stepped forward Euler; the
covariance propagates through the analytically-structured, numerically-
evaluated Jacobian of that step. The scalar observation ``I_k`` (or
``P_k + N_k`` for the sentiment models) corrects both the compartments and the
rate via the Kalman gain (Joseph-form covariance update for stability).

Initialisation from the batch-LS fit (rates + latent initial conditions) is
deliberate: a cold-started EKF is notoriously sensitive to its initial guess,
so we seed it at the nonlinear-LS optimum and let it add the one thing LS
cannot — the *time profile* ``β̂(t)``.

Why only one rate is tracked
----------------------------
A single scalar observation per step carries roughly one degree of freedom.
Tracking the transmission rate (the dominant, intervention-sensitive one)
against a fixed backbone of the remaining rates is well-posed; augmenting the
state with several simultaneously-drifting rates is not identifiable from
``I(t)`` alone and makes the filter diverge — the same unidentifiability that
``fit_all`` flags as ``[unident.]`` for the over-parameterised models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Numerical Jacobians of the ODE right-hand side
# ---------------------------------------------------------------------------

def _jac_state(deriv: Any, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian ``∂f/∂x`` of the RHS at state ``x``."""
    nc = x.size
    J  = np.empty((nc, nc))
    for j in range(nc):
        h      = eps * max(abs(x[j]), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (deriv(xp) - deriv(xm)) / (2.0 * h)
    return J


def _jac_rate(deriv_b: Any, x: np.ndarray, beta: float,
              eps: float = 1e-6) -> np.ndarray:
    """Central-difference derivative ``∂f/∂β`` of the RHS at rate ``beta``."""
    h = eps * max(abs(beta), 1e-12)
    return (deriv_b(x, beta + h) - deriv_b(x, beta - h)) / (2.0 * h)


# ---------------------------------------------------------------------------
# Core EKF
# ---------------------------------------------------------------------------

def ekf_track_transmission(
    spec: dict[str, Any],
    t: np.ndarray,
    N: float,
    I_obs: np.ndarray,
    ls_rates: list[float],
    ls_latent: list[float],
    q_rel: float = 0.03,
    r_rel: float = 0.05,
    n_sub: int = 4,
    beta_p0_rel: float = 0.5,
) -> dict[str, Any]:
    """Track the time-varying transmission rate of one model from ``I_obs``.

    Parameters
    ----------
    spec
        One entry of ``fit_all._registry`` (ode, layout, rates, infected,
        latent, needs_N).
    t, N, I_obs
        Observation times, population, and the observed infected curve.
    ls_rates, ls_latent
        Batch-LS estimates (rate vector in ``spec["rates"]`` order, latent
        initial conditions in ``spec["latent"]`` order) used to seed the
        filter and to fix the non-tracked rates.
    q_rel
        Random-walk process-noise std of the tracked rate, as a fraction of
        its LS value, **per √day**. The EKF analogue of the RLS forgetting
        factor: larger ⇒ faster tracking, noisier ``β̂(t)``. ``q_rel→0``
        pins the rate constant (≈ a recursive re-derivation of the LS value).
    r_rel
        Observation-noise std as a fraction of the local infected count
        (proportional / inverse-variance noise, as for the GLS objective).
    n_sub
        Forward-Euler sub-steps per observation interval (covariance is
        propagated through the compounded sub-step Jacobians).
    beta_p0_rel
        Initial std of the tracked rate as a fraction of its LS value.

    Returns
    -------
    dict with ``name``, ``rate`` (tracked rate name), ``beta_path`` and
    ``beta_se`` (filtered estimate ± 1σ per step), ``beta_final``,
    ``beta_lo`` / ``beta_hi`` (min/max over the stabilised second half),
    ``curve`` (filtered infected fit), ``rmse``, ``time``, and ``q_rel`` /
    ``r_rel``. ``ok`` is ``False`` if the filter produced non-finite values.
    """
    t      = np.asarray(t, dtype=float)
    I_obs  = np.asarray(I_obs, dtype=float)
    layout = spec["layout"]
    nc     = len(layout)
    nz     = nc + 1                                  # compartments + tracked rate
    inf_idx = [layout.index(c) for c in spec["infected"]]
    s_idx   = layout.index("S")
    needs_N = spec["needs_N"]
    rate_name = spec["rates"][0]                     # transmission / exposure rate
    beta0   = float(ls_rates[0])
    others  = [float(v) for v in ls_rates]           # full rate vector (idx 0 overwritten)

    def rate_args(beta: float) -> tuple:
        rv    = list(others)
        rv[0] = beta
        return tuple(rv) + ((N,) if needs_N else ())

    def deriv(x: np.ndarray, beta: float) -> np.ndarray:
        return np.asarray(spec["ode"](0.0, x, *rate_args(beta)), dtype=float)

    # --- initial mean: seed compartments from LS latent ICs + observed I0 ----
    x0 = np.zeros(nc)
    for c in spec["infected"]:
        x0[layout.index(c)] = I_obs[0] / len(spec["infected"])
    for i, c in enumerate(spec["latent"]):
        x0[layout.index(c)] = max(float(ls_latent[i]), 0.0)
    x0[s_idx] = max(N - sum(x0[k] for k in range(nc) if k != s_idx), 0.0)
    z = np.concatenate([x0, [beta0]])

    # --- initial covariance --------------------------------------------------
    peak = max(float(np.max(I_obs)), 1.0)
    P = np.zeros((nz, nz))
    for k in range(nc):
        P[k, k] = (0.02 * peak) ** 2 + 1.0           # mild prior on compartments
    for i, c in enumerate(spec["latent"]):           # latent ICs: uncertain
        ki = layout.index(c)
        P[ki, ki] = (max(float(ls_latent[i]), 0.10 * peak)) ** 2 + 1.0
    P[nc, nc] = (beta_p0_rel * abs(beta0) + 1e-12) ** 2

    q_var_day = (q_rel * abs(beta0)) ** 2            # rate random-walk diffusion
    floor_var = (1e-4 * peak) ** 2                   # tiny state process-noise floor
    H = np.zeros(nz)
    H[inf_idx] = 1.0
    eye = np.eye(nz)

    def update(z: np.ndarray, P: np.ndarray, y: float) -> tuple[np.ndarray, np.ndarray]:
        R   = (r_rel * max(abs(y), 0.05 * peak)) ** 2 + 1.0
        S   = float(H @ P @ H + R)
        K   = (P @ H) / S
        z   = z + K * (y - float(H @ z))
        IKH = eye - np.outer(K, H)
        P   = IKH @ P @ IKH.T + np.outer(K, K) * R   # Joseph form
        return z, 0.5 * (P + P.T)

    n     = t.size
    beta_path = np.empty(n)
    beta_se   = np.empty(n)
    curve     = np.empty(n)

    z, P = update(z, P, I_obs[0])                    # absorb the first observation
    z[:nc] = np.clip(z[:nc], 0.0, 1.5 * N)
    z[nc]  = max(z[nc], 0.0)
    beta_path[0], beta_se[0] = z[nc], np.sqrt(max(P[nc, nc], 0.0))
    curve[0] = z[inf_idx].sum()

    ok = True
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        ds = dt / n_sub
        for _ in range(n_sub):
            x, beta = z[:nc], z[nc]
            d   = deriv(x, beta)
            Jx  = _jac_state(lambda xx: deriv(xx, beta), x)
            Jb  = _jac_rate(deriv, x, beta)
            A = eye.copy()
            A[:nc, :nc] = np.eye(nc) + ds * Jx
            A[:nc, nc]  = ds * Jb
            x_next = np.clip(x + ds * d, 0.0, 1.5 * N)
            z = np.concatenate([x_next, [beta]])
            P = A @ P @ A.T
            P[nc, nc] += q_var_day * ds              # rate diffusion over the sub-step
            P[np.diag_indices(nc)] += floor_var * ds
            P = 0.5 * (P + P.T)
        z, P = update(z, P, I_obs[k + 1])
        z[:nc] = np.clip(z[:nc], 0.0, 1.5 * N)
        z[nc]  = max(z[nc], 0.0)
        beta_path[k + 1] = z[nc]
        beta_se[k + 1]   = np.sqrt(max(P[nc, nc], 0.0))
        curve[k + 1]     = z[inf_idx].sum()
        if not (np.all(np.isfinite(z)) and np.isfinite(curve[k + 1])):
            ok = False
            break

    rmse = float(np.sqrt(np.mean((curve - I_obs) ** 2))) if ok else float("inf")
    half = beta_path[n // 2:]
    return {
        "name"      : spec["name"],
        "rate"      : rate_name,
        "beta_path" : beta_path,
        "beta_se"   : beta_se,
        "beta_final": float(beta_path[-1]),
        "beta_ls"   : beta0,
        "beta_lo"   : float(np.min(half)),
        "beta_hi"   : float(np.max(half)),
        "curve"     : curve,
        "rmse"      : rmse,
        "time"      : t,
        "q_rel"     : float(q_rel),
        "r_rel"     : float(r_rel),
        "ok"        : bool(ok),
    }
