"""
Epidemic Model Simulator — console entry point.

This is a thin argparse-based CLI on top of the rest of the project.
For full command examples and the project structure, see Documentation.md.

Five commands:

    python main.py run-all                       Run every model with defaults.
    python main.py run <MODEL>                   Run one model.
    python main.py sample <MODEL> <FILENAME>     Save a JSON sample to samples/.
    python main.py find-parameters <SAMPLE>      Recover rates from a sample (batch NNLS or RLS).
    python main.py plot-sample <SAMPLE>          Plot the time series of a JSON sample.

Parameter overrides for ``run`` and ``sample`` use ``--param KEY=VALUE``,
repeatable, where ``KEY`` is any field of the model's ``*Params`` dataclass.
"""

import argparse
import os
import sys
from dataclasses import fields
from typing import Any, get_type_hints

import matplotlib.pyplot as plt

from estimation import find_parameters
from fit_all import fit_all_models, print_se_report
from plot_sample import plot_sample
from models import (
    ModifSEDISParams,
    SEDISParams,
    SEDPNRParams,
    SEIRParams,
    SEPNSParams,
    SIParams,
    SIRParams,
    SISParams,
    model_modif_sedis,
    model_sedis,
    model_sedpnr,
    model_seir,
    model_sepns,
    model_si,
    model_sir,
    model_sis,
)
from sampling import DEFAULT_N_POINTS, SAMPLES_DIR, create_sample

MODEL_REGISTRY: dict[str, tuple[Any, Any]] = {
    "SI"         : (SIParams,         model_si),
    "SIS"        : (SISParams,        model_sis),
    "SIR"        : (SIRParams,        model_sir),
    "SEIR"       : (SEIRParams,       model_seir),
    "SEPNS"      : (SEPNSParams,      model_sepns),
    "SEDIS"      : (SEDISParams,      model_sedis),
    "SEDPNR"     : (SEDPNRParams,     model_sedpnr),
    "MODIF_SEDIS": (ModifSEDISParams, model_modif_sedis),
}


# ---------------------------------------------------------------------------
# Parameter override handling
# ---------------------------------------------------------------------------

def _coerce(raw: str, target_type: type) -> Any:
    """Coerce a string CLI value to *target_type* (int or float)."""
    if target_type is int:
        # Accept "10_000", "10000" and "1e4".
        return int(float(raw.replace("_", "")))
    if target_type is float:
        return float(raw)
    return raw


def _apply_overrides(params_cls: Any, overrides: list[str]) -> Any:
    """Build a ``params_cls`` instance with --param KEY=VALUE overrides applied.

    ``params_cls`` is one of the seven dataclasses re-exported by
    ``models``; it is annotated as ``Any`` to keep ``dataclasses.fields``
    and ``typing.get_type_hints`` happy without forcing a Protocol import.
    Unknown keys raise a friendly error listing the valid fields.
    """
    type_hints = get_type_hints(params_cls)
    valid_fields = {f.name for f in fields(params_cls)}

    kwargs: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            sys.exit(f"error: --param must be KEY=VALUE, got {item!r}")
        key, raw_val = item.split("=", 1)
        if key not in valid_fields:
            sys.exit(
                f"error: unknown parameter {key!r} for {params_cls.__name__}.\n"
                f"       valid keys: {', '.join(sorted(valid_fields))}"
            )
        kwargs[key] = _coerce(raw_val, type_hints[key])

    return params_cls(**kwargs)


def _resolve_model(name: str) -> tuple[Any, Any]:
    """Look up a (Params, model_fn) pair by case-insensitive name."""
    key = name.upper()
    if key not in MODEL_REGISTRY:
        sys.exit(
            f"error: unknown model {name!r}. "
            f"Choose from: {', '.join(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    params_cls, fn = _resolve_model(args.model)
    fn(_apply_overrides(params_cls, args.param))
    if not args.no_show:
        plt.show()


def cmd_run_all(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  Epidemic Model Simulator — running all models")
    print("=" * 60)
    for params_cls, fn in MODEL_REGISTRY.values():
        fn(params_cls())
    print("=" * 60)
    if not args.no_show:
        plt.show()


def cmd_sample(args: argparse.Namespace) -> None:
    params_cls, _ = _resolve_model(args.model)
    params = _apply_overrides(params_cls, args.param)
    create_sample(params, args.filename, n_points=args.n_points)


def _resolve_sample_path(name: str) -> str:
    """Accept either a bare filename (looked up in samples/) or a real path."""
    if os.path.exists(name):
        return name
    candidate = os.path.join(SAMPLES_DIR, name)
    if os.path.exists(candidate):
        return candidate
    sys.exit(
        f"error: sample file {name!r} not found "
        f"(looked at {name!r} and {candidate!r})"
    )


def cmd_find_parameters(args: argparse.Namespace) -> None:
    if not (0.0 < args.forgetting <= 1.0):
        sys.exit(
            f"error: --forgetting must be in (0, 1], got {args.forgetting}"
        )
    if args.rls_delta <= 0.0:
        sys.exit(f"error: --rls-delta must be > 0, got {args.rls_delta}")
    if args.method == "batch" and (args.forgetting != 1.0 or args.track):
        print("note: --forgetting/--track only apply to --method rls; ignoring "
              "them for the batch solve.")
    result = find_parameters(
        _resolve_sample_path(args.sample),
        weighting=args.weighting,
        ridge=args.ridge,
        scale=args.scale,
        method=args.method,
        forgetting=args.forgetting,
        rls_delta=args.rls_delta,
        track=args.track,
        show=not args.no_show,
    )
    if not args.no_show and result.get("plot_path"):
        plt.show()


def cmd_plot_sample(args: argparse.Namespace) -> None:
    plot_sample(_resolve_sample_path(args.sample))
    if not args.no_show:
        plt.show()


def cmd_fit_all(args: argparse.Namespace) -> None:
    fixed: dict[str, float] = {}
    for item in args.fix:
        if "=" not in item:
            sys.exit(f"error: --fix must be NAME=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            fixed[key] = float(raw)
        except ValueError:
            sys.exit(f"error: --fix value for {key!r} must be a number, got {raw!r}")
    result = fit_all_models(
        _resolve_sample_path(args.sample),
        loss=args.loss, fixed=fixed, ridge=args.ridge,
        ekf=args.ekf, ekf_q=args.ekf_q, ekf_r=args.ekf_r,
        show=not args.no_show,
    )
    print_se_report(result)
    if not args.no_show:
        plt.show()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epidemic-models",
        description=(
            "Epidemic / misinformation-spread model simulator.\n"
            "Full command examples and parameter reference are in Documentation.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    param_help = (
        "Override a parameter as KEY=VALUE. Repeatable. "
        "Valid keys are the fields of the model's Params dataclass "
        "(see Documentation.md). Example: --param beta=0.4 --param gamma=0.1"
    )
    model_choices_help = (
        "Model name (case-insensitive): "
        f"{', '.join(MODEL_REGISTRY)}"
    )

    # run -------------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run a single model and save its figure to figs/.",
    )
    p_run.add_argument("model", help=model_choices_help)
    p_run.add_argument("--param", action="append", default=[], help=param_help)
    p_run.add_argument("--no-show", action="store_true",
                       help="Do not open the matplotlib window after running.")
    p_run.set_defaults(func=cmd_run)

    # run-all ---------------------------------------------------------------
    p_all = sub.add_parser(
        "run-all",
        help="Run every model with default parameters.",
    )
    p_all.add_argument("--no-show", action="store_true",
                       help="Do not open the matplotlib window after running.")
    p_all.set_defaults(func=cmd_run_all)

    # sample ----------------------------------------------------------------
    p_sample = sub.add_parser(
        "sample",
        help="Run a model and write its time series as JSON in samples/.",
    )
    p_sample.add_argument("model", help=model_choices_help)
    p_sample.add_argument("filename", help="Output filename (placed in samples/).")
    p_sample.add_argument(
        "--n-points", type=int, default=DEFAULT_N_POINTS,
        help=f"Number of points in the sample (default {DEFAULT_N_POINTS}).",
    )
    p_sample.add_argument("--param", action="append", default=[], help=param_help)
    p_sample.set_defaults(func=cmd_sample)

    # find-parameters ------------------------------------------------------
    p_find = sub.add_parser(
        "find-parameters",
        aliases=["find_parameters"],
        help="Estimate discrete-model rates from a JSON sample (least squares).",
    )
    p_find.add_argument(
        "sample",
        help="Sample filename in samples/, or a path to a JSON sample.",
    )
    p_find.add_argument(
        "--weighting",
        choices=["auto", "uniform"],
        default="auto",
        help=(
            "Weighting strategy for the LS fit. "
            "'auto' (default) = GLS with per-compartment residual variance; "
            "'uniform' = OLS (original NNLS estimator)."
        ),
    )
    p_find.add_argument(
        "--ridge",
        type=str,
        default="auto",
        metavar="auto|off|LAMBDA",
        help=(
            "Tikhonov/ridge regularization. 'auto' (default) = GCV-selected "
            "lambda, engaged only when cond exceeds the threshold; 'off' (or 0) "
            "= disabled; a number = manual lambda. Stabilises ill-conditioned / "
            "structurally unidentifiable parameter sets (e.g. SEDPNR mu1/mu2)."
        ),
    )
    p_find.add_argument(
        "--no-scale",
        dest="scale",
        action="store_false",
        help=(
            "Disable per-column equilibration of the design matrix "
            "(column scaling is ON by default and improves conditioning)."
        ),
    )
    p_find.add_argument(
        "--method",
        choices=["batch", "rls"],
        default="batch",
        help=(
            "Linear-solve strategy. 'batch' (default) = one direct "
            "non-negative WLS solve; 'rls' = recursive least squares walked "
            "one time step at a time. With --forgetting 1 (default) RLS "
            "reproduces the batch estimate; with --forgetting <1 it tracks "
            "time-varying rates and saves a theta(t) trajectory plot."
        ),
    )
    p_find.add_argument(
        "--forgetting",
        type=float,
        default=1.0,
        metavar="LAMBDA",
        help=(
            "RLS forgetting factor lambda in (0,1] (used with --method rls). "
            "1.0 = no forgetting (equivalent to batch); <1 gives a sliding "
            "memory of ~1/(1-lambda) steps that tracks time-varying rates "
            "(implies a tracked trajectory). Typical: 0.97-0.999."
        ),
    )
    p_find.add_argument(
        "--rls-delta",
        type=float,
        default=1.0e-6,
        metavar="DELTA",
        help=(
            "RLS diffuse-prior scale P0 = I/delta (used with --method rls). "
            "Small delta (default 1e-6) makes the prior negligible."
        ),
    )
    p_find.add_argument(
        "--track",
        action="store_true",
        help=(
            "Record and plot the RLS parameter trajectory theta(t) even when "
            "--forgetting is 1 (shows the recursion converging to the batch "
            "estimate). Always on when --forgetting <1."
        ),
    )
    p_find.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the RLS trajectory plot window (the PNG is still saved).",
    )
    p_find.set_defaults(func=cmd_find_parameters, scale=True)

    # plot-sample ----------------------------------------------------------
    p_plot = sub.add_parser(
        "plot-sample",
        aliases=["plot_sample"],
        help="Plot the time series stored in a JSON sample (no model re-run).",
    )
    p_plot.add_argument(
        "sample",
        help="Sample filename in samples/, or a path to a JSON sample.",
    )
    p_plot.add_argument("--no-show", action="store_true",
                        help="Do not open the matplotlib window after plotting.")
    p_plot.set_defaults(func=cmd_plot_sample)

    # fit-all ---------------------------------------------------------------
    p_fitall = sub.add_parser(
        "fit-all",
        aliases=["fit_all"],
        help=("Fit every model in the project to the sample's I(t) curve "
              "and report per-parameter estimates with standard errors."),
    )
    p_fitall.add_argument(
        "sample",
        help="Sample filename in samples/, or a path to a JSON sample.",
    )
    p_fitall.add_argument(
        "--loss",
        choices=["abs", "gls", "rel", "log"],
        default="abs",
        help=("Optimisation objective: 'abs' (default) = OLS, raw residual "
              "(peak-dominated); 'gls' = GLS, inverse-variance/relative "
              "residual that balances the orders of magnitude an epidemic "
              "curve spans (alias: 'rel'); 'log' log-residual (growth shape)."),
    )
    p_fitall.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=("Pin a rate to a fixed value, removing it from the fit "
              "(repeatable). E.g. --fix gamma=0.1 --fix sigma=0.2 breaks the "
              "beta/gamma collinearity when S stays ~ N (the COVID failure mode)."),
    )
    p_fitall.add_argument(
        "--ridge",
        type=str,
        default="auto",
        metavar="auto|off|LAMBDA",
        help=("Tikhonov/ridge regularization of the nonlinear fit. 'auto' "
              "(default) = GCV-selected lambda, engaged only when cond(J'J) "
              "exceeds the threshold; 'off'/0 = disabled; a number = manual "
              "lambda. Bounds over-parameterised / rank-deficient models "
              "(SEDIS/SEDPNR); does NOT separate a collinear pair (use --fix)."),
    )
    p_fitall.add_argument(
        "--ekf",
        nargs="?",
        const="best",
        default=None,
        metavar="MODEL",
        help=("Run a post-fit extended Kalman filter that tracks a "
              "time-varying transmission rate beta(t)/alpha(t) from the I(t) "
              "series, seeded from the model's LS fit. Bare --ekf uses the "
              "AICc-best model; --ekf SIR picks a model by name. Saves a "
              "two-panel theta(t) plot to figs/<sample>_ekf_track.png."),
    )
    p_fitall.add_argument(
        "--ekf-q",
        type=float,
        default=0.03,
        metavar="Q_REL",
        help=("EKF process-noise std of the tracked rate, as a fraction of "
              "its LS value per sqrt(day) (default 0.03). Larger = faster "
              "tracking but noisier; ~0 pins the rate constant."),
    )
    p_fitall.add_argument(
        "--ekf-r",
        type=float,
        default=0.05,
        metavar="R_REL",
        help=("EKF observation-noise std as a fraction of the local infected "
              "count (default 0.05)."),
    )
    p_fitall.add_argument(
        "--no-show", action="store_true",
        help="Do not open the model-comparison plot window after fitting.",
    )
    p_fitall.set_defaults(func=cmd_fit_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
