"""
Epidemic Model Simulator — entry point.

================================================================================
HOW TO USE THIS PROJECT
================================================================================

This package simulates seven classic and social-network epidemic models:
    SI     – Susceptible → Infected (no recovery)
    SIS    – Susceptible → Infected → Susceptible
    SIR    – Susceptible → Infected → Recovered
    SEIR   – Susceptible → Exposed → Infected → Recovered
    SEPNS  – Susceptible → Exposed → Positively/Negatively Infected → Susceptible
    SEDIS  – Susceptible → Exposed → Doubtful → Infected → Susceptible
    SEDPNR – Susceptible → Exposed → Doubtful → P/N Infected → Restrained

Each model is solved with scipy.integrate.solve_ivp (RK45) and plotted with
matplotlib.  Plots are written to the ``figs/`` directory next to this file.

--------------------------------------------------------------------------------
1.  Project layout
--------------------------------------------------------------------------------

    main.py        – this file; usage docs + default scenario runner
    constants.py   – figure / colour / output-path constants
    params.py      – one dataclass per model holding its parameters
    odes.py        – right-hand-side functions for every model's ODE system
    plotting.py    – solver wrapper, axis styling, peak marker, save-to-PNG
    models.py      – model_si, model_sis, …, model_sedpnr public functions

--------------------------------------------------------------------------------
2.  Requirements
--------------------------------------------------------------------------------

    pip install numpy scipy matplotlib

--------------------------------------------------------------------------------
3.  Running the default scenario
--------------------------------------------------------------------------------

From the project root:

    python main.py

This runs all seven models with the default parameter sets defined in
``main()`` below, prints a per-model summary to the terminal, saves a PNG for
each model under ``figs/<NAME>_model_ex.png`` and finally opens an interactive
matplotlib window with every figure.

--------------------------------------------------------------------------------
4.  Customising a simulation
--------------------------------------------------------------------------------

Edit ``main()`` below to change parameters, or import the project from your
own script:

    from params import SIRParams
    from models import model_sir

    custom = SIRParams(
        population        = 50_000,
        initial_infected  = 100,
        beta              = 0.45,
        gamma             = 0.12,
        t_end             = 180.0,
    )
    fig = model_sir(custom)   # solves, prints a summary, saves figs/SIR_model_ex.png
    fig.show()

Every model follows the same pattern: import its ``*Params`` dataclass from
``params``, import its ``model_*`` function from ``models``, instantiate the
params, and call the function.  The returned object is a ``matplotlib.figure.Figure``
that you can further customise before display.

--------------------------------------------------------------------------------
5.  Running a single model
--------------------------------------------------------------------------------

Comment out the model calls you do not want in ``main()``, or write a tiny
driver script as shown in section 4.

================================================================================
"""

import matplotlib.pyplot as plt

from models import (
    model_sedis,
    model_sedpnr,
    model_seir,
    model_sepns,
    model_si,
    model_sir,
    model_sis,
)
from params import (
    SEDISParams,
    SEDPNRParams,
    SEIRParams,
    SEPNSParams,
    SIParams,
    SIRParams,
    SISParams,
)


def main() -> None:
    print("=" * 60)
    print("  Epidemic Model Simulator")
    print("=" * 60)

    si_params = SIParams(
        population       = 10_000,  # total network size
        initial_infected = 10,      # seed infections at t=0
        beta             = 0.30,    # S -> I   transmission rate
        t_end            = 60.0,    # simulation end time (days)
        t_steps          = 1_000,   # number of output time points
    )

    sis_params = SISParams(
        population       = 10_000,  # total network size
        initial_infected = 10,      # seed infections at t=0
        beta             = 0.30,    # S -> I   transmission rate
        gamma            = 0.10,    # I -> S   recovery rate (no permanent immunity)
        t_end            = 120.0,   # simulation end time (days)
        t_steps          = 1_000,   # number of output time points
    )

    sir_params = SIRParams(
        population        = 10_000,  # total network size
        initial_infected  = 10,      # seed infections at t=0
        initial_recovered = 0,       # pre-immune individuals at t=0
        beta              = 0.30,    # S -> I   transmission rate
        gamma             = 0.10,    # I -> R   recovery rate (permanent immunity)
        t_end             = 160.0,   # simulation end time (days)
        t_steps           = 1_000,   # number of output time points
    )

    seir_params = SEIRParams(
        population       = 10_000,  # total network size
        initial_exposed  = 0,       # latent infections at t=0
        initial_infected = 10,      # seed infections at t=0
        beta             = 0.30,    # S -> E   transmission / contact rate
        sigma            = 0.20,    # E -> I   incubation rate (1/sigma = mean incubation days)
        gamma            = 0.10,    # I -> R   recovery rate (permanent immunity)
        t_end            = 200.0,   # simulation end time (days)
        t_steps          = 1_000,   # number of output time points
    )

    sepns_params = SEPNSParams(
        population           = 10_000,  # total network size
        initial_exposed      = 0,       # exposed individuals at t=0
        initial_pos_infected = 5,       # positive-sentiment spreaders at t=0
        initial_neg_infected = 5,       # negative-sentiment spreaders at t=0
        alpha                = 0.20,    # S -> E   exposure / contact rate
        beta1                = 0.15,    # E -> P   positive-sentiment adoption rate
        beta2                = 0.20,    # E -> N   negative-sentiment adoption rate
        mu1                  = 0.05,    # P -> S   positive spreader loses interest
        mu2                  = 0.05,    # N -> S   negative spreader loses interest
        mu_e                 = 0.03,    # E -> S   exposed individual rejects information early
        t_end                = 200.0,   # simulation end time (days)
        t_steps              = 1_000,   # number of output time points
    )

    sedis_params = SEDISParams(
        population       = 10_000,  # total network size
        initial_exposed  = 0,       # exposed individuals at t=0
        initial_doubtful = 0,       # doubtful individuals at t=0
        initial_infected = 10,      # seed spreaders at t=0
        alpha            = 0.20,    # S -> E   exposure rate
        beta1            = 0.10,    # E -> D   exposed becomes doubtful
        beta2            = 0.15,    # E -> I   exposed directly accepts and spreads
        gamma            = 0.08,    # D -> I   doubtful individual becomes convinced
        mu1              = 0.04,    # E -> S   exposed rejects information early
        mu2              = 0.05,    # D -> S   doubtful rejects after verification
        mu3              = 0.05,    # I -> S   spreader loses interest
        t_end            = 200.0,   # simulation end time (days)
        t_steps          = 1_000,   # number of output time points
    )

    sedpnr_params = SEDPNRParams(
        population           = 10_000,  # total network size
        initial_exposed      = 10,      # exposed individuals at t=0
        initial_doubtful     = 10,      # doubtful individuals at t=0
        initial_pos_infected = 5,       # positive-sentiment spreaders at t=0
        initial_neg_infected = 5,       # negative-sentiment spreaders at t=0
        initial_restrained   = 0,       # restrained individuals at t=0
        alpha                = 0.20,    # S -> E   contact / exposure rate
        beta1                = 0.15,    # E -> P   exposed adopts positive-sentiment spreading
        beta2                = 0.20,    # E -> N   exposed adopts negative-sentiment spreading
        beta3                = 0.10,    # D -> P   doubtful converts to positive spreader
        beta4                = 0.12,    # D -> N   doubtful converts to negative spreader
        gamma                = 0.10,    # E -> D   exposed becomes doubtful
        lambda1              = 0.05,    # P -> R   positive spreader becomes restrained
        lambda2              = 0.05,    # N -> R   negative spreader becomes restrained
        mu1                  = 0.03,    # E -> S   exposed rejects information
        mu2                  = 0.04,    # D -> S   doubtful rejects after verification
        t_end                = 100.0,   # simulation end time (days)
        t_steps              = 1_000,   # number of output time points
    )

    model_si(si_params)
    model_sis(sis_params)
    model_sir(sir_params)
    model_seir(seir_params)
    model_sepns(sepns_params)
    model_sedis(sedis_params)
    model_sedpnr(sedpnr_params)

    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()
