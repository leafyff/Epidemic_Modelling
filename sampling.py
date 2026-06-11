"""
Sample-generation library.

The single public function ``create_sample`` accepts any of the seven
``*Params`` dataclasses defined in the ``models`` package.  It dispatches to
the correct ODE system, integrates with the shared ``drawing.solve``
helper, and writes a JSON document containing:

    {
        "model"       : "<MODEL NAME>",
        "params"      : <full parameter dict>,
        "time"        : [t0, t1, ...],
        "compartments": { "S": [...], "I": [...], ... }
    }

This module is a *library*: it does not run anything on import and has no
``__main__`` block.  All driver code (which params to run, what filenames
to write) lives in ``main.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from drawing import solve
from models.modif_SEDIS import ModifSEDISParams, modif_sedis_ode
from models.SEDIS import SEDISParams, sedis_ode
from models.SEDPNR import SEDPNRParams, sedpnr_ode
from models.SEIR import SEIRParams, seir_ode
from models.SEPNS import SEPNSParams, sepns_ode
from models.SI import SIParams, si_ode
from models.SIR import SIRParams, sir_ode
from models.SIS import SISParams, sis_ode

SAMPLES_DIR = "samples"


def _setup(params: Any) -> tuple[str, Any, list[float], tuple, list[str]]:
    """Map a Params instance to (model_name, ode_fn, y0, ode_args, compartments)."""
    N = float(params.population)

    if isinstance(params, SIParams):
        I0 = float(params.initial_infected)
        return ("SI", si_ode, [N - I0, I0],
                (params.beta, N), ["S", "I"])

    if isinstance(params, SISParams):
        I0 = float(params.initial_infected)
        return ("SIS", sis_ode, [N - I0, I0],
                (params.beta, params.gamma, N), ["S", "I"])

    if isinstance(params, SIRParams):
        I0  = float(params.initial_infected)
        R0v = float(params.initial_recovered)
        return ("SIR", sir_ode, [N - I0 - R0v, I0, R0v],
                (params.beta, params.gamma, N), ["S", "I", "R"])

    if isinstance(params, SEIRParams):
        E0 = float(params.initial_exposed)
        I0 = float(params.initial_infected)
        return ("SEIR", seir_ode, [N - E0 - I0, E0, I0, 0.0],
                (params.beta, params.sigma, params.gamma, N),
                ["S", "E", "I", "R"])

    if isinstance(params, SEPNSParams):
        E0  = float(params.initial_exposed)
        P0  = float(params.initial_pos_infected)
        Nc0 = float(params.initial_neg_infected)
        return ("SEPNS", sepns_ode, [N - E0 - P0 - Nc0, E0, P0, Nc0],
                (params.alpha, params.beta1, params.beta2,
                 params.mu1, params.mu2, params.mu_e, N),
                ["S", "E", "P", "N"])

    if isinstance(params, SEDISParams):
        E0 = float(params.initial_exposed)
        D0 = float(params.initial_doubtful)
        I0 = float(params.initial_infected)
        return ("SEDIS", sedis_ode, [N - E0 - D0 - I0, E0, D0, I0],
                (params.alpha, params.beta1, params.beta2, params.gamma,
                 params.mu1, params.mu2, params.mu3),
                ["S", "E", "D", "I"])

    if isinstance(params, ModifSEDISParams):
        E0 = float(params.initial_exposed)
        D0 = float(params.initial_doubtful)
        I0 = float(params.initial_infected)
        return ("modif_SEDIS", modif_sedis_ode, [N - E0 - D0 - I0, E0, D0, I0],
                (params.alpha, params.beta1, params.beta2, params.gamma,
                 params.mu1, params.mu2, params.mu3),
                ["S", "E", "D", "I"])

    if isinstance(params, SEDPNRParams):
        E0   = float(params.initial_exposed)
        D0   = float(params.initial_doubtful)
        P0   = float(params.initial_pos_infected)
        Nc0  = float(params.initial_neg_infected)
        R0v  = float(params.initial_restrained)
        return ("SEDPNR", sedpnr_ode,
                [N - E0 - D0 - P0 - Nc0 - R0v, E0, D0, P0, Nc0, R0v],
                (params.alpha,
                 params.beta1, params.beta2, params.beta3, params.beta4,
                 params.gamma, params.lambda1, params.lambda2,
                 params.mu1, params.mu2),
                ["S", "E", "D", "P", "N", "R"])

    raise TypeError(f"Unknown params type: {type(params).__name__}")


DEFAULT_N_POINTS = 1000


def create_sample(
    params: Any,
    filename: str,
    samples_dir: str = SAMPLES_DIR,
    n_points: int = DEFAULT_N_POINTS,
) -> str:
    """Run the model described by *params*, save the time series to JSON, return the path.

    Parameters
    ----------
    params       : any of SIParams / SISParams / SIRParams / SEIRParams /
                   SEPNSParams / SEDISParams / SEDPNRParams
    filename     : output filename, e.g. ``"SIR_sample1.json"``
    samples_dir  : output directory (created if missing).  Default ``"samples"``.
    n_points     : number of equally-spaced time points to record in the
                   sample. Overrides ``params.t_steps`` for the integration
                   grid that is written to the JSON file. Default 1000.
    """
    model_name, ode_fn, y0, ode_args, compartment_names = _setup(params)

    t, y = solve(ode_fn, y0, params.t_end, n_points, ode_args)

    os.makedirs(samples_dir, exist_ok=True)
    path = os.path.join(samples_dir, filename)

    payload = {
        "model"       : model_name,
        "params"      : asdict(params),
        "n_points"    : n_points,
        "time"        : t.tolist(),
        "compartments": {
            name: values.tolist()
            for name, values in zip(compartment_names, y)
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved {model_name} sample ({len(t)} points) -> {path}")
    return path
