"""
Epidemic Model Simulator — console entry point.

This is a thin argparse-based CLI on top of the rest of the project.
For full command examples and the project structure, see Documentation.md.

Three commands:

    python main.py run-all                       Run every model with defaults.
    python main.py run <MODEL>                   Run one model.
    python main.py sample <MODEL> <FILENAME>     Save a JSON sample to samples/.

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
from models import (
    SEDISParams,
    SEDPNRParams,
    SEIRParams,
    SEPNSParams,
    SIParams,
    SIRParams,
    SISParams,
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
    "SI"    : (SIParams,     model_si),
    "SIS"   : (SISParams,    model_sis),
    "SIR"   : (SIRParams,    model_sir),
    "SEIR"  : (SEIRParams,   model_seir),
    "SEPNS" : (SEPNSParams,  model_sepns),
    "SEDIS" : (SEDISParams,  model_sedis),
    "SEDPNR": (SEDPNRParams, model_sedpnr),
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
    find_parameters(_resolve_sample_path(args.sample))


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
    p_find.set_defaults(func=cmd_find_parameters)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
