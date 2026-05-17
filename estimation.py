"""Discrete-model parameter estimation via least squares.

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
compartments produces a tall over-determined linear system A·θ = b which
is solved in the least-squares sense by ``numpy.linalg.lstsq``.

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


# ---------------------------------------------------------------------------
# Per-model design-matrix builders
# ---------------------------------------------------------------------------
# Each builder returns (A, b, param_names).  A has shape (n_eq, n_params),
# b has shape (n_eq,), and param_names lists the parameters in column order
# of A.  ``state`` is the dict of compartment arrays from the JSON sample,
# ``t`` is the time array, ``N`` is the total population.

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


_BUILDERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, list[str]]]] = {
    "SIR": _builder_SIR,
    # Extend here: "SI", "SIS", "SEIR", "SEPNS", "SEDIS", "SEDPNR".
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sample(path: str) -> dict[str, Any]:
    """Read a JSON sample produced by ``sampling.create_sample``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_parameters(sample: dict[str, Any]) -> dict[str, Any]:
    """Estimate the discrete-model parameters from a sample dict.

    Returns a result dict with keys: ``model``, ``estimates`` (name -> float),
    ``true_params`` (the rates the sample was generated with, where present),
    ``rmse`` (least-squares residual RMSE), ``n_points``, ``dt`` (mean step).
    """
    model = sample["model"]
    if model not in _BUILDERS:
        supported = ", ".join(_BUILDERS) or "(none yet)"
        raise ValueError(
            f"No estimator registered for model {model!r}. Supported: {supported}."
        )

    population = float(sample["params"]["population"])
    t       = np.asarray(sample["time"],         dtype=float)
    state   = {k: np.asarray(v, dtype=float) for k, v in sample["compartments"].items()}

    A, b, names = _BUILDERS[model](t, state, population)

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    rmse      = float(np.sqrt(np.mean((A @ theta - b) ** 2)))

    estimates   = dict(zip(names, [float(v) for v in theta]))
    true_params = {
        n: float(sample["params"][n])
        for n in names
        if n in sample["params"]
    }

    return {
        "model"      : model,
        "estimates"  : estimates,
        "true_params": true_params,
        "rmse"       : rmse,
        "n_points"   : int(len(t)),
        "dt"         : float(np.mean(np.diff(t))),
    }


def print_summary(result: dict[str, Any], source: str) -> None:
    """Print a human-readable parameter-estimation report."""
    model = result["model"]
    est   = result["estimates"]
    true  = result["true_params"]

    print(f"\n--- Parameter estimation: {model} ---")
    print(f"  Source file                  : {source}")
    print(f"  Number of points             : {result['n_points']}")
    print(f"  Step size dt                 : {result['dt']:.6f}")
    print()
    print(f"  {'Parameter':<10} {'Estimated':>14} {'True':>14} {'Abs. error':>14} {'Rel. error':>12}")
    for name, value in est.items():
        t_val = true.get(name)
        if t_val is not None:
            abs_err = abs(value - t_val)
            rel_err = abs_err / abs(t_val) if t_val != 0 else float("nan")
            print(f"  {name:<10} {value:>14.6f} {t_val:>14.6f} {abs_err:>14.4e} {rel_err:>12.2%}")
        else:
            print(f"  {name:<10} {value:>14.6f} {'-':>14} {'-':>14} {'-':>12}")
    print(f"  Residual RMSE                : {result['rmse']:.4e}")


def find_parameters(path: str) -> dict[str, Any]:
    """High-level helper used by the CLI: load, estimate, print, return."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample file not found: {path}")
    sample = load_sample(path)
    result = estimate_parameters(sample)
    print_summary(result, path)
    return result
