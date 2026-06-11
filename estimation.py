"""Discrete-model parameter estimation via non-negative generalized least squares.

Reads a JSON sample produced by ``sampling.create_sample`` and recovers
the rate parameters of the underlying epidemic model assuming the
**forward-Euler discretization** of the original ODE system.

For a generic ODE system of the form

    dy/dt = f(y; θ)

where the right-hand side is *linear in the parameters* θ (every model in
this project has this structure), forward Euler gives

    (y_{n+1} - y_n) / dt  =  f(y_n; θ)

For each time step we therefore have one linear equation per compartment
in the unknowns θ.  Stacking all equations across all steps and
compartments produces a tall over-determined linear system A·θ = b.

Because every rate parameter in these ODE models is a 1/time quantity
that must satisfy θ ≥ 0 (a negative rate has no physical meaning — it
would correspond to mass flowing *backwards* between compartments), the
system is solved by **non-negative least squares**
(``scipy.optimize.nnls``).  This guarantees ``θ_i >= 0`` for every
recovered parameter; if the trajectory is best explained by a negative
value (a sign that the model is mis-specified or the parameter is
unidentifiable from the data), NNLS clips the corresponding component
to exactly zero and re-routes the residual into the remaining
parameters.

Weighting (GLS).  By default the linear system is solved by *weighted*
/ *generalized* least squares: every squared residual is divided by the
inverse of the noise variance, so a compartment with a larger dynamic
range cannot dominate the fit just because its dy/dt values are bigger:

    minimize  Σ_i  w_i · (b_i - A_i·θ)²        subject to θ ≥ 0,
              w_i = 1 / σ²_{c(i)}

The per-compartment variance ``σ̂²_c`` is estimated in one IRLS step:
a pilot OLS fit produces residuals ``r_i``, then ``σ̂²_c`` is set to
the mean of ``r_i²`` over the equations belonging to compartment c.
Passing ``weighting="uniform"`` reverts to OLS (the original NNLS
estimator).

Per-parameter standard errors.  Each estimate is reported together
with its **standard error** (and a large-sample 95% half-CI ≈
1.96·SE):

    Cov(θ̂)  = σ̂² · (Aᵀ W A)⁻¹
    SE(θ_k) = √[diag(Cov)]_k

where ``σ̂² = (Σ_i w_i r̂_i²) / (n_eq - n_param)``,  ``W = diag(w_i)``
and ``r̂_i = b_i - A_i·θ̂``.  This is the standard heteroscedastic-WLS
inference formula and matches the per-component guaranteed error of
Nakonechny & Shevchuk (2020, Theorem 38) when the noise budget is
taken to be ``β = Φ(θ̂) · n_eq / (n_eq - n_param)``.

Supported models (one builder per model, each producing the A, b, names
triple for the linear system):

    SI, SIS, SIR, SEIR, SEPNS, SEDIS, modif_SEDIS, SEDPNR

The discretization bias.  The samples in ``samples/`` are produced by
scipy's RK45 (a high-order method) applied to the *continuous* ODE.
Fitting them with forward Euler therefore recovers the parameters of the
*discrete* model that best matches the trajectory — typically a small
percent off the original continuous-model values for the resolutions used
here.  Smaller ``--n-points`` (larger dt) increases this gap; larger
``--n-points`` shrinks it.
"""

import json
import os
from typing import Any, Callable

import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Per-model design-matrix builders
# ---------------------------------------------------------------------------
# Each builder returns (A, b, param_names).  A has shape (n_eq, n_params),
# b has shape (n_eq,), and param_names lists the parameters in column order
# of A.  ``state`` is the dict of compartment arrays from the JSON sample,
# ``t`` is the time array, ``N`` is the total population.

def _builder_SI(t: np.ndarray, state: dict[str, np.ndarray], N: float):
    """Forward-Euler SI linear system.

        (S_{n+1} - S_n) / dt = -beta * S_n*I_n/N
        (I_{n+1} - I_n) / dt = +beta * S_n*I_n/N
    """
    S, I = state["S"], state["I"]
    dt = np.diff(t)

    dS_dt = np.diff(S) / dt
    dI_dt = np.diff(I) / dt

    Sl, Il = S[:-1], I[:-1]
    SI_N   = Sl * Il / N

    A = np.vstack([
        np.column_stack([-SI_N]),   # eq for dS
        np.column_stack([+SI_N]),   # eq for dI
    ])
    b = np.concatenate([dS_dt, dI_dt])
    return A, b, ["beta"]


def _builder_SIS(t: np.ndarray, state: dict[str, np.ndarray], N: float):
    """Forward-Euler SIS linear system.

        (S_{n+1} - S_n) / dt = -beta * S_n*I_n/N  +  gamma * I_n
        (I_{n+1} - I_n) / dt = +beta * S_n*I_n/N  -  gamma * I_n
    """
    S, I = state["S"], state["I"]
    dt = np.diff(t)

    dS_dt = np.diff(S) / dt
    dI_dt = np.diff(I) / dt

    Sl, Il = S[:-1], I[:-1]
    SI_N   = Sl * Il / N

    A = np.vstack([
        np.column_stack([-SI_N, +Il]),   # eq for dS
        np.column_stack([+SI_N, -Il]),   # eq for dI
    ])
    b = np.concatenate([dS_dt, dI_dt])
    return A, b, ["beta", "gamma"]


def _builder_SIR(t: np.ndarray, state: dict[str, np.ndarray], N: float):
    """Forward-Euler SIR linear system.

        (S_{n+1} - S_n) / dt = -beta * S_n*I_n/N
        (I_{n+1} - I_n) / dt = +beta * S_n*I_n/N  -  gamma * I_n
        (R_{n+1} - R_n) / dt =                     +gamma * I_n
    """
    S, I, R = state["S"], state["I"], state["R"]
    dt = np.diff(t)

    dS_dt = np.diff(S) / dt
    dI_dt = np.diff(I) / dt
    dR_dt = np.diff(R) / dt

    Sl, Il = S[:-1], I[:-1]
    SI_N   = Sl * Il / N
    zeros  = np.zeros_like(Sl)

    A = np.vstack([
        np.column_stack([-SI_N, zeros]),   # eq for dS
        np.column_stack([+SI_N, -Il   ]),  # eq for dI
        np.column_stack([zeros, +Il   ]),  # eq for dR
    ])
    b = np.concatenate([dS_dt, dI_dt, dR_dt])
    return A, b, ["beta", "gamma"]


def _builder_SEIR(t: np.ndarray, state: dict[str, np.ndarray], N: float):
    """Forward-Euler SEIR linear system.

        dS = -beta * S*I/N
        dE = +beta * S*I/N  -  sigma * E
        dI = +sigma * E      -  gamma * I
        dR =                   +gamma * I

    Columns: [beta, sigma, gamma]
    """
    S, E, I, R = state["S"], state["E"], state["I"], state["R"]
    dt = np.diff(t)

    dS_dt = np.diff(S) / dt
    dE_dt = np.diff(E) / dt
    dI_dt = np.diff(I) / dt
    dR_dt = np.diff(R) / dt

    Sl, El, Il = S[:-1], E[:-1], I[:-1]
    SI_N  = Sl * Il / N
    zeros = np.zeros_like(Sl)

    A = np.vstack([
        np.column_stack([-SI_N, zeros, zeros]),   # dS
        np.column_stack([+SI_N, -El,   zeros]),   # dE
        np.column_stack([zeros, +El,   -Il  ]),   # dI
        np.column_stack([zeros, zeros, +Il  ]),   # dR
    ])
    b = np.concatenate([dS_dt, dE_dt, dI_dt, dR_dt])
    return A, b, ["beta", "sigma", "gamma"]


def _builder_SEPNS(t: np.ndarray, state: dict[str, np.ndarray], N: float):
    """Forward-Euler SEPNS linear system.

    With spreaders = P + N_comp and exposure = alpha * S * spreaders / N:

        dS = -exposure + mu_e*E + mu1*P + mu2*N_comp
        dE = +exposure - beta1*E - beta2*E - mu_e*E
        dP = +beta1*E  - mu1*P
        dN = +beta2*E  - mu2*N_comp

    Columns: [alpha, beta1, beta2, mu1, mu2, mu_e]
    """
    S, E, P, Nc = state["S"], state["E"], state["P"], state["N"]
    dt = np.diff(t)

    dS_dt = np.diff(S)  / dt
    dE_dt = np.diff(E)  / dt
    dP_dt = np.diff(P)  / dt
    dN_dt = np.diff(Nc) / dt

    Sl, El, Pl, Ncl = S[:-1], E[:-1], P[:-1], Nc[:-1]
    spread_N = Sl * (Pl + Ncl) / N
    zeros    = np.zeros_like(Sl)

    A = np.vstack([
        # dS:  -alpha*spread_N + mu1*P + mu2*Nc + mu_e*E
        np.column_stack([-spread_N, zeros, zeros, +Pl,   +Ncl,  +El  ]),
        # dE:  +alpha*spread_N - beta1*E - beta2*E - mu_e*E
        np.column_stack([+spread_N, -El,   -El,   zeros, zeros, -El  ]),
        # dP:  +beta1*E - mu1*P
        np.column_stack([zeros,     +El,   zeros, -Pl,   zeros, zeros]),
        # dN:  +beta2*E - mu2*Nc
        np.column_stack([zeros,     zeros, +El,   zeros, -Ncl,  zeros]),
    ])
    b = np.concatenate([dS_dt, dE_dt, dP_dt, dN_dt])
    return A, b, ["alpha", "beta1", "beta2", "mu1", "mu2", "mu_e"]


def _builder_SEDIS(t: np.ndarray, state: dict[str, np.ndarray], _N: float):
    """Forward-Euler SEDIS linear system.

    SEDIS exposure is the per-S leakage form ``alpha * S`` (no I factor),
    so the column for alpha is built from S_n alone. Population is unused
    by the design matrix (the leading underscore reflects that).

        dS = -alpha*S + mu1*E + mu2*D + mu3*I
        dE = +alpha*S - beta1*E - beta2*E - mu1*E
        dD = +beta1*E - gamma*D - mu2*D
        dI = +beta2*E + gamma*D - mu3*I

    Columns: [alpha, beta1, beta2, gamma, mu1, mu2, mu3]
    """
    S, E, D, I = state["S"], state["E"], state["D"], state["I"]
    dt = np.diff(t)

    dS_dt = np.diff(S) / dt
    dE_dt = np.diff(E) / dt
    dD_dt = np.diff(D) / dt
    dI_dt = np.diff(I) / dt

    Sl, El, Dl, Il = S[:-1], E[:-1], D[:-1], I[:-1]
    zeros = np.zeros_like(Sl)

    A = np.vstack([
        # dS
        np.column_stack([-Sl,   zeros, zeros, zeros, +El,   +Dl,   +Il  ]),
        # dE
        np.column_stack([+Sl,   -El,   -El,   zeros, -El,   zeros, zeros]),
        # dD
        np.column_stack([zeros, +El,   zeros, -Dl,   zeros, -Dl,   zeros]),
        # dI
        np.column_stack([zeros, zeros, +El,   +Dl,   zeros, zeros, -Il  ]),
    ])
    b = np.concatenate([dS_dt, dE_dt, dD_dt, dI_dt])
    return A, b, ["alpha", "beta1", "beta2", "gamma", "mu1", "mu2", "mu3"]


def _builder_modif_SEDIS(t: np.ndarray, state: dict[str, np.ndarray], _N: float):
    """Forward-Euler modif_SEDIS linear system.

    Identical to SEDIS except the exposure term is alpha*S*I (mass-action,
    NOT divided by N).  Population is therefore unused.

        dS = -alpha*S*I + mu1*E + mu2*D + mu3*I
        dE = +alpha*S*I - beta1*E - beta2*E - mu1*E
        dD = +beta1*E   - gamma*D - mu2*D
        dI = +beta2*E + gamma*D - mu3*I

    Columns: [alpha, beta1, beta2, gamma, mu1, mu2, mu3]
    """
    S, E, D, I = state["S"], state["E"], state["D"], state["I"]
    dt = np.diff(t)

    dS_dt = np.diff(S) / dt
    dE_dt = np.diff(E) / dt
    dD_dt = np.diff(D) / dt
    dI_dt = np.diff(I) / dt

    Sl, El, Dl, Il = S[:-1], E[:-1], D[:-1], I[:-1]
    SI    = Sl * Il
    zeros = np.zeros_like(Sl)

    A = np.vstack([
        np.column_stack([-SI,   zeros, zeros, zeros, +El,   +Dl,   +Il  ]),  # dS
        np.column_stack([+SI,   -El,   -El,   zeros, -El,   zeros, zeros]),  # dE
        np.column_stack([zeros, +El,   zeros, -Dl,   zeros, -Dl,   zeros]),  # dD
        np.column_stack([zeros, zeros, +El,   +Dl,   zeros, zeros, -Il  ]),  # dI
    ])
    b = np.concatenate([dS_dt, dE_dt, dD_dt, dI_dt])
    return A, b, ["alpha", "beta1", "beta2", "gamma", "mu1", "mu2", "mu3"]


def _builder_SEDPNR(t: np.ndarray, state: dict[str, np.ndarray], _N: float):
    """Forward-Euler SEDPNR linear system.

        dS = -alpha*S + mu1*E + mu2*D
        dE = +alpha*S - beta1*E - beta2*E - gamma*E - mu1*E
        dD = +gamma*E - beta3*D - beta4*D - mu2*D
        dP = +beta1*E + beta3*D - lambda1*P
        dN = +beta2*E + beta4*D - lambda2*N_comp
        dR =                      +lambda1*P + lambda2*N_comp

    Columns: [alpha, beta1, beta2, beta3, beta4, gamma,
              lambda1, lambda2, mu1, mu2]
    """
    S, E, D, P, Nc, R = (state["S"], state["E"], state["D"],
                         state["P"], state["N"], state["R"])
    dt = np.diff(t)

    dS_dt = np.diff(S)  / dt
    dE_dt = np.diff(E)  / dt
    dD_dt = np.diff(D)  / dt
    dP_dt = np.diff(P)  / dt
    dN_dt = np.diff(Nc) / dt
    dR_dt = np.diff(R)  / dt

    Sl, El, Dl, Pl, Ncl = S[:-1], E[:-1], D[:-1], P[:-1], Nc[:-1]
    zeros = np.zeros_like(Sl)

    A = np.vstack([
        # dS: -alpha*S + mu1*E + mu2*D
        np.column_stack([-Sl,   zeros, zeros, zeros, zeros, zeros,
                         zeros, zeros, +El,   +Dl  ]),
        # dE: +alpha*S - (beta1 + beta2 + gamma + mu1)*E
        np.column_stack([+Sl,   -El,   -El,   zeros, zeros, -El,
                         zeros, zeros, -El,   zeros]),
        # dD: +gamma*E - (beta3 + beta4 + mu2)*D
        np.column_stack([zeros, zeros, zeros, -Dl,   -Dl,   +El,
                         zeros, zeros, zeros, -Dl  ]),
        # dP: +beta1*E + beta3*D - lambda1*P
        np.column_stack([zeros, +El,   zeros, +Dl,   zeros, zeros,
                         -Pl,   zeros, zeros, zeros]),
        # dN: +beta2*E + beta4*D - lambda2*Nc
        np.column_stack([zeros, zeros, +El,   zeros, +Dl,   zeros,
                         zeros, -Ncl,  zeros, zeros]),
        # dR: +lambda1*P + lambda2*Nc
        np.column_stack([zeros, zeros, zeros, zeros, zeros, zeros,
                         +Pl,   +Ncl,  zeros, zeros]),
    ])
    b = np.concatenate([dS_dt, dE_dt, dD_dt, dP_dt, dN_dt, dR_dt])
    return A, b, ["alpha", "beta1", "beta2", "beta3", "beta4", "gamma",
                  "lambda1", "lambda2", "mu1", "mu2"]


_BUILDERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, list[str]]]] = {
    "SI"         : _builder_SI,
    "SIS"        : _builder_SIS,
    "SIR"        : _builder_SIR,
    "SEIR"       : _builder_SEIR,
    "SEPNS"      : _builder_SEPNS,
    "SEDIS"      : _builder_SEDIS,
    "modif_SEDIS": _builder_modif_SEDIS,
    "SEDPNR"     : _builder_SEDPNR,
}


# ---------------------------------------------------------------------------
# Compartment row layout of each builder's linear system
# ---------------------------------------------------------------------------
# Every builder stacks its equations compartment-by-compartment with
# (n_points - 1) equations per compartment. This registry records the
# compartment order so weights, residuals and diagnostics can be attributed
# to the correct compartment.

_MODEL_COMPARTMENTS: dict[str, list[str]] = {
    "SI"         : ["S", "I"],
    "SIS"        : ["S", "I"],
    "SIR"        : ["S", "I", "R"],
    "SEIR"       : ["S", "E", "I", "R"],
    "SEPNS"      : ["S", "E", "P", "N"],
    "SEDIS"      : ["S", "E", "D", "I"],
    "modif_SEDIS": ["S", "E", "D", "I"],
    "SEDPNR"     : ["S", "E", "D", "P", "N", "R"],
}


# ---------------------------------------------------------------------------
# Generalized / weighted least squares (with non-negativity)
# ---------------------------------------------------------------------------

def _compute_weights(
    A: np.ndarray,
    b: np.ndarray,
    compartments: list[str],
    weighting: Any = "auto",
) -> tuple[np.ndarray, str]:
    """Return a per-equation weight vector w_i for the WLS fit.

    Three strategies are supported:

    * ``"uniform"`` — all weights equal one. Equivalent to ordinary
      least squares (the original NNLS estimator).
    * ``"auto"`` — one IRLS step: fit OLS first, then for each
      compartment ``c`` estimate the residual variance
      ``σ̂²_c = mean(r_i² for i in compartment c)`` and set
      ``w_i = 1/σ̂²_{c(i)}``. This equalises the contribution of every
      compartment regardless of its dynamic range.
    * ``np.ndarray`` of shape ``(n_eq,)`` — custom user-supplied weights.

    Returns ``(weights, label)``; ``label`` is the human-readable name
    of the strategy used (for the diagnostic printout).
    """
    n_eq = A.shape[0]
    n_per_compartment = n_eq // len(compartments)

    if isinstance(weighting, np.ndarray):
        if weighting.shape != (n_eq,):
            raise ValueError(
                f"Custom weighting must have shape ({n_eq},), got {weighting.shape}."
            )
        return weighting.astype(float), "custom"

    if weighting == "uniform":
        return np.ones(n_eq), "uniform (OLS)"

    if weighting == "auto":
        # Pilot OLS fit to estimate per-compartment residual variance.
        theta_pilot, _ = nnls(A, b)
        r = b - A @ theta_pilot
        weights = np.empty(n_eq)
        for c_idx in range(len(compartments)):
            start, end = c_idx * n_per_compartment, (c_idx + 1) * n_per_compartment
            var_c = float(np.mean(r[start:end] ** 2))
            if var_c < 1e-12:
                var_c = 1e-12  # Avoid division by zero for trivially fit compartments
            weights[start:end] = 1.0 / var_c
        return weights, "auto (GLS, per-compartment residual variance)"

    raise ValueError(
        f"Unknown weighting strategy: {weighting!r}. "
        f"Expected 'uniform', 'auto', or an ndarray of length {n_eq}."
    )


AUTO_RIDGE_COND = 1.0e6   # raw cond(AᵀWA) above which ridge="auto" engages


def _gcv_lambda(s: np.ndarray, g: np.ndarray, b_perp_sq: float, n: int) -> float:
    """Pick the Tikhonov ``λ`` that minimises the Generalized Cross-Validation
    score, from the SVD of the scaled design.

    For ``A_s = U diag(s) Vᵀ`` and ``g = Uᵀ b_w`` the ridge residual norm and
    the hat-matrix trace are

        ‖r(λ)‖² = ‖b_perp‖² + Σ_i [λ/(s_i²+λ)]² g_i²,
        tr H(λ) = Σ_i s_i²/(s_i²+λ),

    and ``GCV(λ) = n·‖r(λ)‖² / (n − tr H(λ))²`` is minimised over a log grid.
    GCV needs no held-out set and picks the λ that best trades fit against
    effective degrees of freedom.
    """
    s2 = s ** 2
    g2 = g ** 2
    if s2.size == 0:
        return 0.0
    smax2 = float(s2.max())
    grid  = np.logspace(np.log10(smax2) - 9.0, np.log10(smax2) + 1.0, 90)
    best_lam, best_gcv = 0.0, np.inf
    for lam in grid:
        filt  = lam / (s2 + lam)
        rss   = b_perp_sq + float(np.sum((filt ** 2) * g2))
        trH   = float(np.sum(s2 / (s2 + lam)))
        denom = n - trH
        if denom <= 0.0:
            continue
        gcv = n * rss / denom ** 2
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    return best_lam


def _solve_nn_wls(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    ridge: Any = 0.0,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Solve the non-negative weighted least-squares problem, with column
    equilibration and (optionally automatic) Tikhonov regularization for
    ill-conditioned design matrices.

        minimize  Σ_i w_i (b_i - A_i θ)²  +  λ·‖θ_s‖²      subject to θ ≥ 0

    Two conditioning safeguards run before the NNLS solve:

    * **Column equilibration** (``scale=True``, default). Every column of
      the whitened design ``A_w = W^{1/2} A`` is rescaled to unit Euclidean
      norm. Because compartments span many orders of magnitude (``S ~ 1e7``
      vs ``E ~ 1e2``) this drops the condition number of ``AᵀWA`` by several
      orders **without changing the minimiser** — it is the exact
      reparameterisation ``θ = D θ_s`` with ``D = diag(1/‖col‖)`` (and
      ``D > 0`` keeps the non-negativity cone intact).
    * **Tikhonov / ridge**. ``ridge`` may be a float (``0`` = off, ``>0`` =
      manual λ) or the string ``"auto"`` / ``"gcv"`` — then λ is chosen by
      Generalized Cross-Validation, but **only when the raw condition number
      exceeds ``AUTO_RIDGE_COND``** (well-conditioned problems are left
      untouched, so exact recovery is preserved). The scaled system is
      augmented with ``√λ·I`` rows, so the normal matrix becomes
      ``A_sᵀA_s + λ·I`` — always invertible, bounding rank-deficient /
      collinear parameter sets (SEDPNR ``mu1``/``mu2``; ``β≈γ`` on COVID).

    Both safeguards keep the problem a standard NNLS solved by
    ``scipy.optimize.nnls``.

    Covariance is computed from the SVD of the scaled design (so we never
    form ``AᵀWA`` and square its condition number):

        Cov(θ̂) = σ̂²·D·(A_sᵀA_s + λ·I)⁻¹·D,
        σ̂²     = (Σ_i w_i r̂_i²) / (n_eq - n_param)

    Returns ``(theta, cov, sigma2, cond, lam)``: ``cond`` is the 2-norm
    condition number of the *raw* (unscaled) ``AᵀWA`` (the difficulty
    diagnostic), and ``lam`` is the regularization strength actually applied.
    """
    n_eq, n_param = A.shape
    sqrt_w        = np.sqrt(weights)
    A_w           = sqrt_w[:, None] * A
    b_w           = sqrt_w * b

    # --- column equilibration (scaling) --------------------------------------
    if scale:
        col_norms = np.linalg.norm(A_w, axis=0)
        col_norms[col_norms < 1e-300] = 1.0          # guard all-zero columns
        D = 1.0 / col_norms
    else:
        D = np.ones(n_param)
    A_s = A_w * D                                     # scaled design: A_w = A_s/D

    # --- SVD of the scaled design (drives cond, GCV and covariance) ----------
    U, s, Vt = np.linalg.svd(A_s, full_matrices=False)
    s2 = s ** 2

    # raw (unscaled) condition number — difficulty diagnostic
    s_raw   = np.linalg.svd(A_w, compute_uv=False)
    pos_raw = s_raw[s_raw > 0]
    cond    = float((s_raw.max() / pos_raw.min()) ** 2) if pos_raw.size else np.inf

    # --- choose the ridge strength λ -----------------------------------------
    if isinstance(ridge, str):
        key = ridge.strip().lower()
        if key in ("auto", "gcv"):
            if cond > AUTO_RIDGE_COND:
                g         = U.T @ b_w
                b_perp_sq = max(float(b_w @ b_w - g @ g), 0.0)
                lam       = _gcv_lambda(s, g, b_perp_sq, n_eq)
            else:
                lam = 0.0
        elif key in ("off", "none"):
            lam = 0.0
        else:
            lam = float(ridge)
    else:
        lam = float(ridge)

    # --- NNLS solve (ridge-augmented if λ>0), in scaled coordinates ----------
    if lam > 0.0:
        A_aug = np.vstack([A_s, np.sqrt(lam) * np.eye(n_param)])
        b_aug = np.concatenate([b_w, np.zeros(n_param)])
    else:
        A_aug, b_aug = A_s, b_w

    theta_s, _ = nnls(A_aug, b_aug)
    theta      = D * theta_s                          # back to original coords

    # --- residual scale on the ORIGINAL (un-augmented) whitened system -------
    r_w    = b_w - A_w @ theta
    dof    = max(n_eq - n_param, 1)
    sigma2 = float(np.sum(r_w ** 2) / dof)

    # --- covariance via SVD of the scaled design (stable; no AᵀWA) -----------
    if lam > 0.0:
        inv_diag = 1.0 / (s2 + lam)
    else:
        # Moore-Penrose truncation (drop directions below numerical rank).
        tol      = (s.max() if s.size else 0.0) * max(n_eq, n_param) * np.finfo(float).eps
        inv_diag = np.where(s > tol, 1.0 / np.where(s > tol, s2, 1.0), 0.0)
    Minv_s = (Vt.T * inv_diag) @ Vt
    cov    = sigma2 * (D[:, None] * Minv_s * D[None, :])
    return theta, cov, sigma2, cond, lam


# ---------------------------------------------------------------------------
# Extremum detection via the finite-difference derivative
# ---------------------------------------------------------------------------

def _parabolic_vertex(t: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float]:
    """Refine an extremum at index *j* to sub-step resolution by fitting a
    parabola through ``(j-1, j, j+1)``; fall back to the grid point at the
    boundaries or for a near-linear (degenerate) fit."""
    if j - 1 < 0 or j + 1 >= len(t):
        return float(t[j]), float(y[j])
    xs = t[j - 1:j + 2]
    ys = y[j - 1:j + 2]
    a, b_coef, c = np.polyfit(xs, ys, 2)
    if abs(a) < 1e-300:
        return float(t[j]), float(y[j])
    tv = -b_coef / (2.0 * a)
    tv = min(max(tv, float(xs[0])), float(xs[-1]))    # clamp into the bracket
    return float(tv), float(a * tv * tv + b_coef * tv + c)


def _merge_same_type(y: np.ndarray,
                     cand: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Collapse adjacent same-type extrema, keeping the more extreme one
    (the higher of two maxima / the lower of two minima)."""
    i = 0
    cand = list(cand)
    while i < len(cand) - 1:
        if cand[i][1] == cand[i + 1][1]:
            a, b = cand[i], cand[i + 1]
            if cand[i][1] == "max":
                keep = a if y[a[0]] >= y[b[0]] else b
            else:
                keep = a if y[a[0]] <= y[b[0]] else b
            cand[i:i + 2] = [keep]
        else:
            i += 1
    return cand


def _prune_extrema(y: np.ndarray, cand: list[tuple[int, str]],
                   thr: float) -> list[tuple[int, str]]:
    """Persistence pruning: repeatedly remove the least-prominent max-min
    wiggle (value gap < ``thr``) and merge the same-type neighbours it
    exposes, so noise in the finite-difference derivative does not spawn
    spurious extrema while the dominant turning points survive."""
    cand = _merge_same_type(y, cand)
    while len(cand) >= 2:
        gaps = [abs(float(y[cand[i + 1][0]]) - float(y[cand[i][0]]))
                for i in range(len(cand) - 1)]
        m = int(np.argmin(gaps))
        if gaps[m] >= thr:
            break
        del cand[m:m + 2]
        cand = _merge_same_type(y, cand)
    return cand


def find_extrema(
    t: np.ndarray,
    series: np.ndarray,
    smooth: int = 0,
    min_prominence_frac: float = 0.05,
) -> list[dict[str, Any]]:
    """Locate interior extrema (peaks / troughs) of *series* via the
    finite-difference approximation of its derivative.

    The derivative is approximated by central differences
    (``numpy.gradient``); an extremum sits where it changes sign. Optional
    centred moving-average smoothing (``smooth`` = window length, forced
    odd) denoises the derivative for noisy real-world series. Wiggles whose
    amplitude is below ``min_prominence_frac`` of the series range are
    pruned, and each surviving extremum is refined to sub-step resolution by
    a local parabolic fit.

    Returns a list of ``{"index", "t", "value", "type"}`` dicts, with
    ``type`` either ``"max"`` or ``"min"``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(series, dtype=float)
    n = y.size
    if n < 3:
        return []

    ys = y
    if smooth and smooth >= 3:
        k = int(smooth) | 1                            # force odd window
        if k < n:
            ys = np.convolve(y, np.ones(k) / k, mode="same")

    dy = np.gradient(ys, t)                            # central-difference derivative
    sg = np.sign(dy)

    # Forward-fill zero derivatives with the last non-zero sign, so an
    # extremum that lands exactly on a sample (dy == 0 there) is not masked.
    filled = sg.copy()
    last   = 0.0
    for i in range(filled.size):
        if filled[i] != 0:
            last = filled[i]
        else:
            filled[i] = last

    cand: list[tuple[int, str]] = []
    for i in range(1, n):
        if filled[i - 1] != 0 and filled[i] != 0 and filled[i - 1] != filled[i]:
            kind = "max" if filled[i - 1] > 0 else "min"
            if kind == "max":
                j = i if ys[i] >= ys[i - 1] else i - 1
            else:
                j = i if ys[i] <= ys[i - 1] else i - 1
            cand.append((j, kind))

    thr  = (float(np.max(y) - np.min(y)) or 1.0) * float(min_prominence_frac)
    cand = _prune_extrema(y, cand, thr)

    out: list[dict[str, Any]] = []
    for j, kind in cand:
        ts, vs = _parabolic_vertex(t, y, j)
        out.append({"index": int(j), "t": ts, "value": vs, "type": kind})
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sample(path: str) -> dict[str, Any]:
    """Read a JSON sample produced by ``sampling.create_sample``.

    Raises ``FileNotFoundError`` with a friendly message if *path* does
    not exist, so every caller (CLI commands, library users) gets the
    same error rather than re-implementing the check.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_parameters(
    sample: dict[str, Any],
    weighting: Any = "auto",
    ridge: Any = "auto",
    scale: bool = True,
) -> dict[str, Any]:
    """Estimate the discrete-model parameters via non-negative GLS.

    Solves the over-determined linear system ``A·θ = b`` of the
    forward-Euler discretization by weighted least squares with a
    non-negativity constraint on ``θ``. Each estimate is returned
    together with its standard error.

    Parameters
    ----------
    sample
        Loaded JSON sample (as produced by ``sampling.create_sample``).
    weighting
        ``"auto"`` (default) — one-step IRLS GLS, with weights set to
        ``1/σ̂²_c`` per compartment (where ``σ̂²_c`` is the mean squared
        OLS residual in compartment ``c``).
        ``"uniform"`` — ordinary least squares, equivalent to the
        original NNLS estimator.
        ``np.ndarray`` — custom per-equation weights.
    ridge
        Tikhonov / ridge regularization. A float (``0`` = off, ``>0`` =
        manual ``λ``) or ``"auto"`` (default) — then ``λ`` is chosen by
        Generalized Cross-Validation, engaged only when the raw condition
        number exceeds ``AUTO_RIDGE_COND`` (well-conditioned problems are
        left untouched). Bounds ill-conditioned / unidentifiable sets.
    scale
        If ``True`` (default), equilibrate the columns of the whitened
        design matrix to unit norm before solving — an exact
        reparameterisation that sharply lowers the condition number.

    Returns
    -------
    result : dict
        Keys: ``model``, ``estimates`` (name → float),
        ``std_errors`` (name → float), ``true_params`` (name → float,
        where the generator rates are stored in the sample), ``rmse``
        (unweighted residual RMSE), ``sigma2`` (estimated WLS residual
        scale), ``cond`` (2-norm condition number of the raw ``AᵀWA``),
        ``ridge`` (λ used), ``scaled`` (whether column scaling was on),
        ``extrema`` (per-compartment finite-difference extremum points),
        ``weighting`` (string label of the strategy actually
        used), ``n_points``, ``dt`` (mean step size).

    Notes
    -----
    Standard error formula (heteroscedastic WLS with weights
    ``w_i = 1/σ²_i``):

        Cov(θ̂)  = σ̂² · (Aᵀ W A)⁻¹
        SE(θ_k) = √diag(Cov)_k

    where ``σ̂² = (Σ_i w_i r̂_i²) / (n_eq - n_param)``. This matches
    the per-component guaranteed-error formula of Nakonechny &
    Shevchuk (2020, Theorem 38),
    ``σ_k = √[(A⁻¹)_kk] · √(β - Φ(θ̂))``, with the noise budget
    ``β = Φ(θ̂) · n_eq / (n_eq - n_param)``.

    For NNLS solutions where a parameter is pinned to exactly zero the
    SE is reported using the unconstrained Hessian and should be read
    as a diagnostic ("data is consistent with zero") rather than a
    strict confidence interval.
    """
    model = sample["model"]
    if model not in _BUILDERS:
        supported = ", ".join(_BUILDERS) or "(none yet)"
        raise ValueError(
            f"No estimator registered for model {model!r}. Supported: {supported}."
        )

    population = float(sample["params"]["population"])
    t          = np.asarray(sample["time"], dtype=float)
    state      = {k: np.asarray(v, dtype=float)
                  for k, v in sample["compartments"].items()}

    A, b, names  = _BUILDERS[model](t, state, population)
    compartments = _MODEL_COMPARTMENTS[model]

    weights, weighting_label = _compute_weights(A, b, compartments, weighting)
    theta, cov, sigma2, cond, lam = _solve_nn_wls(A, b, weights, ridge=ridge, scale=scale)
    se                       = np.sqrt(np.maximum(np.diag(cov), 0.0))
    rmse                     = float(np.sqrt(np.mean((A @ theta - b) ** 2)))
    ridge_auto               = isinstance(ridge, str) and ridge.strip().lower() in ("auto", "gcv")

    estimates   = dict(zip(names, [float(v) for v in theta]))
    std_errors  = dict(zip(names, [float(s) for s in se]))
    true_params = {
        n: float(sample["params"][n])
        for n in names
        if n in sample["params"]
    }

    # Extremum points of each observed compartment via the finite-difference
    # derivative (np.gradient) — a curve-characterisation diagnostic.
    extrema: dict[str, list[dict[str, Any]]] = {}
    for c in compartments:
        if c in state:
            ex = find_extrema(t, state[c])
            if ex:
                extrema[c] = ex

    return {
        "model"      : model,
        "estimates"  : estimates,
        "std_errors" : std_errors,
        "true_params": true_params,
        "rmse"       : rmse,
        "sigma2"     : sigma2,
        "cond"       : cond,
        "ridge"      : float(lam),
        "ridge_auto" : ridge_auto,
        "scaled"     : bool(scale),
        "extrema"    : extrema,
        "weighting"  : weighting_label,
        "n_points"   : int(len(t)),
        "dt"         : float(np.mean(np.diff(t))),
    }


def print_summary(result: dict[str, Any], source: str) -> None:
    """Print a human-readable parameter-estimation report with standard errors."""
    model = result["model"]
    est   = result["estimates"]
    se    = result["std_errors"]
    true  = result["true_params"]

    if result.get("ridge", 0.0) > 0:
        src = "auto/GCV" if result.get("ridge_auto") else "manual"
        reg = f"ridge lambda={result['ridge']:.2e} ({src})"
    else:
        reg = "auto (cond below threshold -> off)" if result.get("ridge_auto") else "none"
    print(f"\n--- Parameter estimation: {model} ---")
    print(f"  Source file                  : {source}")
    print(f"  Number of points             : {result['n_points']}")
    print(f"  Step size dt                 : {result['dt']:.6f}")
    print(f"  Weighting                    : {result['weighting']}")
    print(f"  Column scaling               : {'on' if result.get('scaled', False) else 'off'}")
    print(f"  Regularization               : {reg}")
    print(f"  cond(A^T W A)                : {result.get('cond', float('nan')):.3e}")
    print(f"  Residual scale (sigma^2)     : {result['sigma2']:.4e}")
    print()
    print(
        f"  {'Parameter':<10} {'Estimate':>14} {'Std. error':>14} "
        f"{'95% CI half':>14} {'True':>14} {'Rel. error':>12}"
    )
    for name, value in est.items():
        sd      = se[name]
        ci_half = 1.96 * sd                       # large-sample normal approx
        t_val   = true.get(name)
        if t_val is not None:
            if t_val != 0:
                rel_err_str = f"{abs(value - t_val) / abs(t_val):>12.2%}"
            else:
                rel_err_str = f"{'nan':>12}"
            print(
                f"  {name:<10} {value:>14.6f} {sd:>14.4e} {ci_half:>14.4e} "
                f"{t_val:>14.6f} {rel_err_str}"
            )
        else:
            print(
                f"  {name:<10} {value:>14.6f} {sd:>14.4e} {ci_half:>14.4e} "
                f"{'-':>14} {'-':>12}"
            )
    print(f"  Residual RMSE (unweighted)   : {result['rmse']:.4e}")

    extrema = result.get("extrema", {})
    if extrema:
        print("\n  Extremum points (finite-difference derivative):")
        for c, exs in extrema.items():
            for e in exs:
                print(f"    {c:<3} {e['type']:>3} at t={e['t']:>11.4f}"
                      f"  value={e['value']:>16,.4f}")


def find_parameters(
    path: str,
    weighting: Any = "auto",
    ridge: Any = "auto",
    scale: bool = True,
) -> dict[str, Any]:
    """High-level helper used by the CLI: load, estimate, print, return."""
    sample = load_sample(path)
    result = estimate_parameters(sample, weighting=weighting, ridge=ridge, scale=scale)
    print_summary(result, path)
    return result
