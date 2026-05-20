# Documentation

This file covers the project structure and full CLI usage examples for the
epidemic model simulator.  All commands assume you are in the project root
and that `numpy`, `scipy` and `matplotlib` are installed
(`pip install numpy scipy matplotlib`).

---

## Project structure

```text
.
├── main.py             # Console entry point (argparse-based CLI)
├── drawing.py          # Figure constants + plotting / solver helpers
├── sampling.py         # Library: create_sample(params, filename, n_points)
├── estimation.py       # Library: forward-Euler least-squares parameter estimation
├── models/             # One file per epidemic model — each fully self-contained
│   ├── __init__.py     # Re-exports every *Params, *_ode and model_*
│   ├── SI.py
│   ├── SIS.py
│   ├── SIR.py
│   ├── SEIR.py
│   ├── SEPNS.py
│   ├── SEDIS.py
│   ├── modif_SEDIS.py
│   └── SEDPNR.py
├── figs/               # PNG plots written by each model_* function
├── samples/            # JSON samples written by sampling.create_sample()
├── Documentation.md    # This file
├── README.md           # Project overview, theory, references
├── LICENSE
└── Epidemic modeling for misinformation spread.pdf
```

Each `models/<NAME>.py` contains:

1. A long module docstring with the full theory of the model (compartments,
   governing ODEs, parameters, basic reproduction number, typical use cases).
2. A `<NAME>Params` dataclass listing every tunable parameter with defaults.
3. The ODE right-hand-side function (e.g. `sir_ode`).
4. The public simulation function (e.g. `model_sir`) which integrates the
   ODE, prints a summary and writes a PNG to `figs/`.

---

## CLI overview

The console interface is

```bash
python main.py <command> [options]
```

| Command | Purpose |
|---|---|
| `run-all` | Run every model with default parameters. |
| `run <MODEL>` | Run a single model. |
| `sample <MODEL> <FILENAME>` | Run a model and save its time series as JSON. |
| `find-parameters <SAMPLE>` | Estimate the discrete-model rate parameters from a JSON sample via least squares (alias: `find_parameters`). |

`<MODEL>` is case-insensitive and one of:
`SI`, `SIS`, `SIR`, `SEIR`, `SEPNS`, `SEDIS`, `modif_SEDIS`, `SEDPNR`.

Parameter overrides for `run` and `sample` use `--param KEY=VALUE`
(repeatable). `KEY` must match a field of the model's `*Params` dataclass —
see the parameter reference at the bottom of this file.

Help is available for every command:

```bash
python main.py --help
python main.py run --help
python main.py sample --help
```

---

## Examples: running models

Run every model with default parameters and display all figures:

```bash
python main.py run-all
```

Run one model at a time with defaults:

```bash
python main.py run SI
python main.py run SIS
python main.py run SIR
python main.py run SEIR
python main.py run SEPNS
python main.py run SEDIS
python main.py run modif_SEDIS
python main.py run SEDPNR
```

Override parameters (each override is one `--param KEY=VALUE`):

```bash
python main.py run SI     --param beta=0.5 --param t_end=90
python main.py run SIS    --param beta=0.4 --param gamma=0.08
python main.py run SIR    --param population=50000 --param beta=0.4 --param gamma=0.12 --param t_end=180
python main.py run SEIR   --param beta=0.45 --param sigma=0.2 --param gamma=0.12
python main.py run SEPNS  --param alpha=0.25 --param beta1=0.18 --param beta2=0.22
python main.py run SEDIS  --param alpha=0.25 --param beta1=0.12 --param beta2=0.18 --param gamma=0.10
python main.py run modif_SEDIS --param alpha=2.5e-5 --param beta1=0.12 --param beta2=0.18 --param gamma=0.10
python main.py run SEDPNR --param beta1=0.18 --param beta2=0.22 --param lambda1=0.06 --param lambda2=0.06
```

Skip the interactive matplotlib window (useful in CI or headless servers — the
PNG is still written to `figs/`):

```bash
python main.py run-all --no-show
python main.py run SIR --param beta=0.4 --no-show
```

---

## Examples: creating JSON samples

Create the canonical SIR sample with the defaults requested in the project
brief (β = 0.3, γ = 0.1):

```bash
python main.py sample SIR SIR_sample1.json
```

Override the number of points (`--n-points`, default 1000):

```bash
python main.py sample SIR  SIR_dense.json   --n-points 5000
python main.py sample SEIR SEIR_coarse.json --n-points 200
```

Override model parameters at the same time (combine `--param` with `--n-points`):

```bash
python main.py sample SIR    SIR_high_beta.json    --param beta=0.5 --param gamma=0.1
python main.py sample SEDPNR SEDPNR_alt.json       --n-points 2000 --param alpha=0.25 --param beta1=0.18
python main.py sample SEPNS  SEPNS_neg_dominant.json --param beta1=0.10 --param beta2=0.30 --n-points 1500
```

JSON files are written to `samples/<FILENAME>`. The layout is:

```json
{
  "model"       : "SIR",
  "params"      : { "population": 10000, "beta": 0.3, "gamma": 0.1, "...": "..." },
  "n_points"    : 1000,
  "time"        : [0.0, 0.16, 0.32, "..."],
  "compartments": {
    "S": ["..."],
    "I": ["..."],
    "R": ["..."]
  }
}
```

`n_points` controls the resolution of `time` and the compartment arrays for
that one sample; it does **not** mutate the dataclass's own `t_steps`.

---

## Examples: estimating parameters from a sample (`find-parameters`)

`find-parameters` reads a JSON sample, applies the **forward-Euler**
discretization of the model's ODE system, and solves the resulting
over-determined linear system in the least-squares sense (numpy's
`linalg.lstsq`) to recover the rate parameters.

End-to-end example using the SIR sample from the project brief
(β = 0.4, γ = 0.2):

```bash
# 1) generate the test sample
python main.py sample SIR SIR_sample1.json --param beta=0.4 --param gamma=0.2

# 2) recover the parameters
python main.py find-parameters SIR_sample1.json
```

Expected output (truncated):

```text
--- Parameter estimation: SIR ---
  Source file                  : samples\SIR_sample1.json
  Number of points             : 1000
  Step size dt                 : 0.160160

  Parameter       Estimated           True     Abs. error   Rel. error
  beta             0.399424       0.400000     5.7568e-04        0.14%
  gamma            0.200287       0.200000     2.8750e-04        0.14%
  Residual RMSE                : 9.7588e-01
```

The `find-parameters` argument accepts either a bare filename (resolved
against `samples/`) or an absolute / relative path to a JSON file. The
underscore alias `find_parameters` is also accepted:

```bash
python main.py find-parameters SIR_sample1.json
python main.py find_parameters samples/SIR_sample1.json
python main.py find-parameters /absolute/path/to/sample.json
```

### Why estimates differ slightly from the true rates

The samples are produced by SciPy's RK45 (a high-order method) applied to
the *continuous* ODE.  `find-parameters` fits the *forward-Euler*
discretization to the same trajectory.  The recovered rates are therefore
those of the discrete model that best reproduces the sampled trajectory,
which differ from the continuous-model rates by an amount proportional to
`dt` and the local curvature of the solution.  Increasing `--n-points`
when you generate the sample shrinks this gap; decreasing it widens it.

### Currently supported models

| Model | Status |
|---|---|
| SIR | Supported |
| SI / SIS / SEIR / SEPNS / SEDIS / modif_SEDIS / SEDPNR | Not yet — add a builder to `estimation._BUILDERS` to extend. |

---

## Parameter reference

Every parameter listed below is a valid `--param KEY=VALUE` key for that
model. The blocks below show the dataclass field name, the default value
used when you do not pass `--param` for it, and a short comment on its
meaning. Full mathematical descriptions live in the module docstring of
the corresponding `models/<NAME>.py` file.

### SI

```python
si_params = SIParams(
    population       = 10_000,  # total network size
    initial_infected = 10,      # seed infections at t=0
    beta             = 0.30,    # S -> I   transmission rate
    t_end            = 60.0,    # simulation end time (days)
    t_steps          = 1_000,   # number of output time points
)
```

### SIS

```python
sis_params = SISParams(
    population       = 10_000,  # total network size
    initial_infected = 10,      # seed infections at t=0
    beta             = 0.30,    # S -> I   transmission rate
    gamma            = 0.10,    # I -> S   recovery rate (no permanent immunity)
    t_end            = 120.0,   # simulation end time (days)
    t_steps          = 1_000,   # number of output time points
)
```

### SIR

```python
sir_params = SIRParams(
    population        = 10_000,  # total network size
    initial_infected  = 10,      # seed infections at t=0
    initial_recovered = 0,       # pre-immune individuals at t=0
    beta              = 0.30,    # S -> I   transmission rate
    gamma             = 0.10,    # I -> R   recovery rate (permanent immunity)
    t_end             = 160.0,   # simulation end time (days)
    t_steps           = 1_000,   # number of output time points
)
```

### SEIR

```python
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
```

### SEPNS

```python
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
```

### SEDIS

```python
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
```

### modif_SEDIS

Same compartments and transitions as SEDIS; the **only** difference is
that the exposure term gains an explicit `I` factor — `alpha * S * I`
instead of `alpha * S`. Because the rate now scales with the number of
spreaders, `alpha` here is a per-pair interaction rate and is
numerically much smaller than the SEDIS `alpha`.

```python
modif_sedis_params = ModifSEDISParams(
    population       = 10_000,  # total network size
    initial_exposed  = 0,       # exposed individuals at t=0
    initial_doubtful = 0,       # doubtful individuals at t=0
    initial_infected = 10,      # seed spreaders at t=0
    alpha            = 2.0e-5,  # S -> E   per-pair exposure rate (per S, per I)
    beta1            = 0.10,    # E -> D   exposed becomes doubtful
    beta2            = 0.15,    # E -> I   exposed directly accepts and spreads
    gamma            = 0.08,    # D -> I   doubtful individual becomes convinced
    mu1              = 0.04,    # E -> S   exposed rejects information early
    mu2              = 0.05,    # D -> S   doubtful rejects after verification
    mu3              = 0.05,    # I -> S   spreader loses interest
    t_end            = 200.0,   # simulation end time (days)
    t_steps          = 1_000,   # number of output time points
)
```

### SEDPNR

```python
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
```

### Notes on `t_end`, `t_steps` and `--n-points`

`t_end` is the simulation horizon in days (float). `t_steps` is the number
of equally-spaced output points used by `run` / `run-all`. For `sample`,
prefer the `--n-points` flag instead of `--param t_steps=...`; the flag
takes precedence over `t_steps`.

---

## Using the library from Python

The CLI is a thin wrapper. The same workflows work directly from Python:

```python
from models import ModifSEDISParams, SIRParams, model_modif_sedis, model_sir
from sampling import create_sample

# Run + plot
fig = model_sir(SIRParams(beta=0.4, gamma=0.12))

# Save a JSON sample
create_sample(SIRParams(beta=0.3, gamma=0.1),
              "SIR_sample1.json", n_points=1000)

# modif_SEDIS (alpha*S*I exposure)
model_modif_sedis(ModifSEDISParams(alpha=2.5e-5, beta1=0.12))
```
