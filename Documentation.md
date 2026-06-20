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
├── estimation.py       # Library: forward-Euler least-squares parameter estimation (batch + RLS)
├── kalman.py           # Library: extended Kalman filter for time-varying beta(t) tracking
├── plot_sample.py      # Library: render a JSON sample as a matplotlib figure
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
| `find-parameters <SAMPLE>` | Estimate the discrete-model rate parameters from a full-state JSON sample via least squares (alias: `find_parameters`). |
| `fit-all <SAMPLE>` | Fit every model to an `I(t)`-only sample and rank them by RMSE / AICc, with per-parameter standard errors (alias: `fit_all`). |
| `plot-sample <SAMPLE>` | Plot the time series stored in a JSON sample. Saves a PNG to `figs/` and opens a matplotlib window (alias: `plot_sample`). |

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
python main.py plot-sample --help
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
python main.py run modif_SEDIS --param alpha=0.02 --param beta1=0.12 --param beta2=0.18 --param gamma=0.10
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

## Examples: plotting a sample (`plot-sample`)

`plot-sample` reads a JSON sample produced by `sample` and draws the
compartments versus time. It does **not** re-run the ODE and does
**not** compare the trajectory against the model the sample came from —
the figure is a direct rendering of the arrays stored in the JSON file.

The figure uses the same palette as `run` / `run-all` (blue S, orange E,
red I, green R, brown D, purple P, pink N) and is saved to
`figs/<sample-stem>_sample.png`.

```bash
# Plot an existing sample
python main.py plot-sample SEDIS_sample1.json

# Headless: just write the PNG, do not open the matplotlib window
python main.py plot-sample SEDIS_sample1.json --no-show

# Underscore alias and explicit path are both accepted
python main.py plot_sample samples/SEDIS_sample1.json
python main.py plot-sample /absolute/path/to/sample.json
```

End-to-end with `sample`:

```bash
python main.py sample SEDIS SEDIS_demo.json
python main.py plot-sample SEDIS_demo.json
```

The console summary lists the source model, the compartment names found
in the file, the number of points, the time span, and the path of the
saved PNG.

---

## Examples: estimating parameters from a sample (`find-parameters`)

`find-parameters` reads a JSON sample, applies the **forward-Euler**
discretization of the model's ODE system, and solves the resulting
over-determined linear system in the *non-negative* least-squares sense
(`scipy.optimize.nnls`) to recover the rate parameters. The system is
linear in the rates, so it can be solved either in one **batch** pass
(default) or **recursively** one time step at a time
(`--method rls`) — the recursive form additionally tracks a
*time-varying* rate `θ̂(t)` (see
[Recursive least squares](#recursive-least-squares---method-rls--online-estimation-and-tracking-time-varying-rates) below).

Non-negative least squares is used because every rate constant in these
ODE models is a `1/time` quantity that must satisfy `θ ≥ 0` — a
negative recovered rate would correspond to mass flowing backwards
between compartments and is physically meaningless. NNLS guarantees
`θ_i >= 0` for every estimate; when the data would be best explained by
a negative value (a sign of mis-specification or unidentifiability), the
solver pins that component to exactly zero and re-routes the residual
into the remaining parameters. There is no upper bound: rates are not
probabilities, so `θ_i > 1` is allowed (e.g. `beta = 1.5` per day for a
fast-spreading disease).

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
  Weighting                    : auto (GLS, per-compartment residual variance)
  Column scaling               : on
  Regularization               : none
  cond(A^T W A)                : 9.921e+00
  Residual scale (sigma^2)     : 9.9459e-01

  Parameter        Estimate     Std. error    95% CI half           True   Rel. error
  beta             0.399224     9.6769e-05     1.8967e-04       0.400000        0.19%
  gamma            0.200137     3.5146e-05     6.8887e-05       0.200000        0.07%
  Residual RMSE (unweighted)   : 9.7751e-01

  Extremum points (finite-difference derivative):
    I   max at t=    33.7971  value=      1,540.0673
```

Every estimate is reported with its **standard error** and a large-sample
95% CI half-width (`1.96·SE`). The header lines report the **weighting**
strategy, whether **column scaling** is on, the **regularization** used,
and `cond(AᵀWA)` — the 2-norm condition number of the normal matrix
(a difficulty / identifiability diagnostic). The trailing **extremum
points** are the peaks/troughs of each compartment located from the
finite-difference derivative (`numpy.gradient`).

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

### Conditioning controls: column scaling and ridge regularization

Real samples and over-parameterised models can make the normal matrix
`AᵀWA` badly conditioned (near-collinear columns → unstable estimates,
huge standard errors). Two controls address this:

| Flag | Default | Effect |
|---|---|---|
| (always on) **column scaling** | on | Each column of the whitened design `W^{1/2}A` is rescaled to unit norm before the solve. This is an **exact reparameterisation** — it lowers `cond(AᵀWA)` by orders of magnitude **without changing the estimates** (only the numerics improve). Disable with `--no-scale`. |
| `--ridge auto\|off\|LAMBDA` | `auto` | **Tikhonov / ridge** regularization. `auto` selects `λ` by **Generalized Cross-Validation**, but only engages when `cond(AᵀWA)` exceeds the threshold (`AUTO_RIDGE_COND ≈ 1e6`) — well-conditioned samples are left untouched, so exact recovery is preserved. `off`/`0` disables it; a number sets `λ` manually. Augments the scaled system with `√λ·I` rows, i.e. minimises `Σ wᵢ(bᵢ−Aᵢθ)² + λ‖θ_s‖²` subject to `θ ≥ 0`. |

```bash
# Default: scaling on, ridge auto (engages only if cond is high), cond + extrema reported:
python main.py find-parameters COVID_Germany_2020.json

# Manual ridge strength, or disable regularization entirely:
python main.py find-parameters COVID_Germany_2020.json --ridge 1e-2
python main.py find-parameters COVID_Germany_2020.json --ridge off

# Reproduce the un-equilibrated numerics (rarely needed):
python main.py find-parameters SEDIS_random.json --no-scale
```

Standard errors and the covariance are computed from the **SVD of the
scaled design** (never forming `AᵀWA`, which would square the condition
number); a rank-deficient problem falls back to the Moore-Penrose
pseudoinverse automatically.

> **Note.** Ridge cures *numerical* ill-conditioning and bounds rank-deficient
> sets, but when a parameter pair is only collinear (e.g. `beta`≈`gamma` on
> COVID, where `S≈N`), GCV correctly applies almost no ridge — the data simply
> cannot separate them. The honest fix there is to **fix one rate** (see
> `fit-all --fix` below) so the other becomes identifiable.

### Recursive least squares (`--method rls`) — online estimation and tracking time-varying rates

The forward-Euler discretization makes the estimation problem **linear in
the rate parameters** (`A·θ = b`), and recursive least squares (RLS) is
precisely the *online* solver for a linear least-squares system. Instead of
forming the whole system once and solving it (the `batch` default), RLS
walks the trajectory **one time step at a time**, updating the estimate
`θ̂` and the inverse information matrix `P` with the Kalman-gain recursion

```text
e_n = b_n − A_n θ̂_{n-1}                          (innovation / prediction error)
K_n = P A_nᵀ (λ I + A_n P A_nᵀ)⁻¹                (gain)
θ̂_n = θ̂_{n-1} + K_n e_n                          (estimate update)
P   ← (P − K_n A_n P) / λ                         (information update)
```

where `A_n` stacks one forward-Euler equation **per compartment** at step
`n` (a block update, fed in *time order*). It is enabled with
`--method rls`:

| Flag | Default | Effect |
|---|---|---|
| `--method batch\|rls` | `batch` | `batch` = one direct non-negative WLS solve; `rls` = the same estimator run recursively. |
| `--forgetting LAMBDA` | `1.0` | Exponential forgetting factor `λ ∈ (0,1]`. `1.0` keeps all history; `λ < 1` gives a sliding memory of `≈ 1/(1−λ)` steps that **tracks time-varying rates** (and forces trajectory tracking). |
| `--rls-delta DELTA` | `1e-6` | Diffuse-prior scale `P₀ = I/δ`. Small `δ` ⇒ negligible prior (and a tiny ridge that keeps `P` well-defined). |
| `--track` | off | Record and plot the trajectory `θ̂(t)` even when `λ = 1` (shows the recursion *converging* to the batch estimate). Always on when `λ < 1`. |

Why this is worth having (two regimes):

**1. `λ = 1` — a streaming cross-check, identical to batch.** With the
diffuse prior, the recursion is algebraically the *same* estimator as the
batch solve (it equals ridge-WLS with ridge `δ`; at `δ = 1e-6` the bias is
negligible). It reproduces the batch numbers to ~4 significant figures
while never forming the full `A` — an `O(n·p²)` streaming/online estimator
suited to long or incrementally-arriving data:

```bash
# RLS with no forgetting -> matches `find-parameters --method batch`
python main.py find-parameters SEDIS_sample1.json --method rls
# Add --track to also save figs/<sample>_rls_track.png showing convergence
python main.py find-parameters SEDIS_sample1.json --method rls --track
```

**2. `λ < 1` — tracking a time-varying rate (the real payoff).** A constant
parameter is a modelling assumption, not a fact: real epidemics have a
transmission rate that changes with interventions and behaviour. The batch
fit can only return a single constant; RLS with forgetting returns the
whole trajectory `θ̂(t)` and a plot to `figs/<sample>_rls_track.png`:

```bash
python main.py find-parameters my_full_state_sample.json --method rls --forgetting 0.97
```

On a synthetic SIR trajectory whose transmission rate steps from
`β = 0.30` to `β = 0.12` on day 30 (an "intervention"), the **batch** fit
returns a meaningless average `β ≈ 0.18` that matches neither phase, while
**RLS tracks the change**:

```text
  day      I(t)   beta_hat  beta_true
   20       457   0.3000    0.30
   28      1570   0.3000    0.30
   32      1885   0.2050    0.12     <- step at day 30, estimate starts moving
   40      1492   0.1351    0.12
   55       799   0.1214    0.12     <- settled onto the new value
```

The print-out adds a **tracked-range** block (min..max of each rate over
the second half of the trajectory, past the diffuse-prior transient), and
the saved plot draws each `θ̂(t)` against the generating value where known.

Caveats specific to RLS:

* **No signal ⇒ no update.** A rate is only identifiable where its design
  column is non-zero. If `I(t) ≈ 0` (the epidemic has burned out) the
  `β·S·I/N` term carries no information and RLS correctly *holds* its last
  estimate rather than drifting — put the change you want to detect inside
  the active-transmission window.
* **Non-negativity** is enforced by projecting the final estimate (and each
  tracked point) onto `θ ≥ 0`, exactly as NNLS does — a component that
  wants to go negative is pinned to zero and read as "unidentifiable".
* **Forgetting widens the standard errors** (fewer effective observations),
  and `λ` trades responsiveness against noise: smaller `λ` reacts faster
  but is noisier. Column scaling and the `δ` prior handle conditioning the
  same way the batch solver does.

> **Scope note.** This brings the time-varying `β(t)` flagged as
> out-of-scope in the `fit-all` caveat *into* scope — but only for
> **full-state** samples (`find-parameters` needs every compartment, not
> just `I(t)`). Tracking from an `I(t)`-only series is nonlinear and needs
> the recursive *extended Kalman filter* in [`fit-all --ekf`](#ekf-tracking-a-time-varying-transmission-rate---ekf)
> instead of plain RLS.

### Extremum detection (finite-difference derivative)

Every `find-parameters` run also reports the **extremum points** (peaks
and troughs) of each compartment's trajectory. They are located by
approximating the derivative with central differences
(`numpy.gradient`), detecting sign changes (robust to an extremum landing
exactly on a sample), pruning sub-prominent noise wiggles, and refining
each turning point to sub-step resolution with a local parabolic fit.
`fit-all` reports the same for the observed `I(t)` curve — e.g. on the
1978 flu sample it recovers the epidemic peak at day ≈ 5.

### Supported models

`find-parameters` recognises every model in this project. The estimator
is dispatched from the sample's `"model"` field, and the columns of the
least-squares design matrix follow the parameter order shown below.

| Model         | Estimated parameters (in solver order)                                  |
|---            |---                                                                       |
| `SI`          | `beta`                                                                   |
| `SIS`         | `beta`, `gamma`                                                          |
| `SIR`         | `beta`, `gamma`                                                          |
| `SEIR`        | `beta`, `sigma`, `gamma`                                                 |
| `SEPNS`       | `alpha`, `beta1`, `beta2`, `mu1`, `mu2`, `mu_e`                          |
| `SEDIS`       | `alpha`, `beta1`, `beta2`, `gamma`, `mu1`, `mu2`, `mu3`                  |
| `modif_SEDIS` | `alpha`, `beta1`, `beta2`, `gamma`, `mu1`, `mu2`, `mu3`                  |
| `SEDPNR`      | `alpha`, `beta1`, `beta2`, `beta3`, `beta4`, `gamma`, `lambda1`, `lambda2`, `mu1`, `mu2` |

Note: `modif_SEDIS` uses the mass-action exposure `alpha * S * I` (not
`alpha * S * I / N`), so the column for `alpha` in its design matrix is
built from `S_n * I_n` rather than `S_n * I_n / N` — but the recovered
`alpha` is numerically tiny because it absorbs the missing `1/N`.

Example workflow for each model:

```bash
# Generate a sample and recover its rates.
python main.py sample SI          SI_demo.json          && python main.py find-parameters SI_demo.json
python main.py sample SIS         SIS_demo.json         && python main.py find-parameters SIS_demo.json
python main.py sample SEIR        SEIR_demo.json        && python main.py find-parameters SEIR_demo.json
python main.py sample SEPNS       SEPNS_demo.json       && python main.py find-parameters SEPNS_demo.json
python main.py sample SEDIS       SEDIS_demo.json       && python main.py find-parameters SEDIS_demo.json
python main.py sample MODIF_SEDIS modif_SEDIS_demo.json && python main.py find-parameters modif_SEDIS_demo.json
python main.py sample SEDPNR      SEDPNR_demo.json      && python main.py find-parameters SEDPNR_demo.json
```

### Note on identifiability

Some compartments contribute almost-collinear columns to the design
matrix — notably the slow-flux terms `mu1` and `mu2` in SEDPNR, and the
`mu_e` term in SEPNS. The least-squares fit still returns a residual
near zero, but individual parameter estimates can drift by 10–30%
because the trajectory is roughly equally well explained by neighbouring
values of those rates. With NNLS, an unidentifiable parameter that would
otherwise have come out negative will be pinned to exactly `0.000000` in
the report — a useful visual cue that the data did not support a
non-zero value for that rate. Increasing `--n-points` reduces the
discretization bias but does **not** fix this structural
ill-conditioning; longer / richer trajectories or informative priors
would.

---

## Examples: comparing all models on `I(t)` (`fit-all`)

`fit-all` is for **real data where only the infected curve `I(t)` is
observed** (and the true model is unknown). For each of the 8 models it
fits the rate parameters *and* the unobserved latent initial conditions
(`E0`, `D0`) by forward-simulating the ODE and minimising the mismatch to
`I(t)` (multi-start `least_squares` + a `differential_evolution` safety
net). It ranks the models and reports per-parameter standard errors.

```bash
python main.py fit-all flu1978_school.json
python main.py fit-all COVID_Germany_2020.json
```

At the end of a run `fit-all` **opens the overlay comparison plot in a
window** (every model's fitted `I(t)` over the observed points) and also
saves it to `figs/<sample>_all_models.png`. Pass `--no-show` to skip the
window (e.g. for scripting / CI).

Each model line shows RMSE and **AICc** (small-sample Akaike criterion);
models whose `cond(J'J)` exceeds `1e12` are flagged `[unident.]` and
excluded from the AICc-based "best model" pick. AICc penalises extra
parameters, so it can prefer a *simpler* model over one with marginally
lower RMSE (e.g. on the flu sample it prefers **SIR** over **SEIR**).

### `--loss {abs,gls,rel,log}` — OLS (default) vs GLS objective

The fit minimises a sum of squared residuals. **`--loss abs` (default) is
ordinary least squares (OLS)** — the raw residual `I_model - I_obs`. Because
epidemic curves span orders of magnitude, the absolute RMSE is dominated by
the peak, so OLS fits the peak well but can neglect the low-count rise/decline.

**`--loss gls` is generalized least squares (GLS)** — an inverse-variance
(proportional-noise) weighting `(I_model - I_obs)/max(|I_obs|,1)` that
down-weights the high-count peak so every scale of the curve is heard
(`--loss rel` is a backwards-compatible alias for the same objective).
`--loss log` (log-residual) is a related growth-shape objective.

On long, noisy real series with a **bounded tail** (e.g. COVID-2020) GLS can
markedly improve how the model tracks the off-peak phases; on short curves
that **decay toward zero** (e.g. the flu boarding-school sample) the
peak-focused OLS is usually more robust. RMSE is still reported in **absolute
people** for comparability; only the optimisation objective changes.

```bash
python main.py fit-all COVID_Germany_2020.json                 # OLS (default)
python main.py fit-all COVID_Germany_2020.json --loss gls      # GLS
```

### `--fix NAME=VALUE` — pin biological rates

When `S` stays ≈ `N` (e.g. national COVID data) the transmission and
recovery rates are collinear — only `β−γ` is identifiable, so the
optimiser rails `β` and `γ` to the upper bound (`RATE_MAX=6`,
non-physical). Pin a known rate to break the collinearity:

```bash
# fix the recovery/incubation rates to plausible values; fit only transmission
python main.py fit-all COVID_Germany_2020.json --fix gamma=0.1 --fix sigma=0.2 --loss log
```

The fixed rates leave the optimisation vector (their SE is shown as 0,
marked `(fixed)`). On the COVID SEIR fit this turns the railed
`β≈4.2` into a **physical `β≈0.145`** (`R0=β/γ≈1.45`), now identifiable.

### `--ridge auto|off|LAMBDA` — Tikhonov regularization of the fit

Adds a penalty `λ·Σ(θ_k/scale_k)²` (scale = parameter bounds) to the
nonlinear objective. `auto` (default) selects `λ` by GCV on the local
Jacobian and **engages only for genuinely rank-deficient models**
(`cond(J'J) > 1e12`, the same boundary as the `[unident.]` flag). There it
**bounds the exploding standard errors** of the over-parameterised models
(SEDIS/SEDPNR, `cond` up to `1e41`) without changing their RMSE.
Moderately-conditioned models (SI/SIR/SEIR) are deliberately left untouched
— a nonlinear refit with a local-GCV `λ` can over-shrink them.

```bash
python main.py fit-all COVID_Germany_2020.json              # ridge auto (default)
python main.py fit-all COVID_Germany_2020.json --ridge off  # disable
```

> **What ridge does and does not do.** It tames *rank-deficient*
> over-parameterisation, but it does **not** separate a merely *collinear*
> pair (`β≈γ` when `S≈N`) — GCV correctly applies almost no ridge there
> because the collinear fit already predicts `I(t)` well. Use `--fix` for
> collinearity, `--ridge` for over-parameterisation.

> **Honest caveat.** `--fix` removes the *ill-conditioning* (identifiable,
> physical parameters, finite SE), but constant-rate models still cannot
> reproduce an intervention-driven decline when `S≈N` — the remaining
> misfit is the genuine signal that a time-varying `β(t)` is needed. That
> is exactly what `--ekf` (below) recovers.

### EKF: tracking a time-varying transmission rate (`--ekf`)

The least-squares fit above returns a **constant** rate per model. But the
observation map (forward-simulate the ODE, read off `I`) is *nonlinear* in
the parameters, so the recursive estimator that can recover a *time-varying*
rate from `I(t)` is not plain RLS (as in `find-parameters`) but its
nonlinear generalisation, the **extended Kalman filter (EKF)**. It is
implemented in [`kalman.py`](kalman.py) and enabled as a **post-fit pass**
on one model:

```bash
python main.py fit-all COVID_Germany_2020.json --ekf            # AICc-best model
python main.py fit-all COVID_Germany_2020.json --ekf SIR        # a named model
python main.py fit-all COVID_Germany_2020.json --ekf SIR --ekf-q 0.05 --ekf-r 0.08
```

How it is set up (and why it is *observable*):

* the filter state is the model's compartment vector **augmented with the
  single transmission / exposure rate** (`β`, or `α` for the misinformation
  models — always the first rate);
* **every other rate is held fixed at its LS value.** A scalar observation
  per step carries ~one degree of freedom, so tracking *one* rate against a
  fixed backbone is well-posed; augmenting several simultaneously-drifting
  rates is not identifiable from `I(t)` alone (the same unidentifiability
  `fit-all` flags as `[unident.]`) and makes the filter diverge;
* the rate follows a random walk whose process-noise std `--ekf-q` (fraction
  of the LS value per √day) is the EKF analogue of the RLS forgetting factor
  — larger ⇒ faster tracking, noisier `β̂(t)`; `≈0` pins it constant;
* the filter is **seeded from the model's LS fit** (rates + latent ICs),
  which removes the EKF's notorious sensitivity to its starting point.

| Flag | Default | Effect |
|---|---|---|
| `--ekf [MODEL]` | off | Run the EKF on `MODEL` (or the AICc-best model if bare). Saves `figs/<sample>_ekf_track.png`. |
| `--ekf-q Q_REL` | `0.03` | Rate process-noise std, fraction of the LS rate per √day. Bigger tracks faster / noisier. |
| `--ekf-r R_REL` | `0.05` | Observation-noise std, fraction of the local infected count (proportional noise). |

The report compares the constant-rate baseline to the tracked rate and
prints a `time variation DETECTED / consistent with a constant rate`
verdict (drift over the second half vs the median 1σ band):

```text
EKF transmission tracking: SIR  (tracked rate: beta)
  EKF beta   final / range (2nd half) : 0.1268  [0.0294, 0.1275]
  Tracked drift vs median 1-sigma     : 0.098 vs 0.014  -> time variation DETECTED
  RMSE  constant-LS -> time-varying EKF : ...
```

The two-panel plot shows the observed `I(t)` with the constant-LS curve and
the EKF curve (top), and `β̂(t) ± 1σ` against the constant LS value
(bottom).

#### Effectiveness — where the EKF helps and where it does not

The EKF was evaluated against the constant-rate LS fit on synthetic and real
data:

* **Synthetic SIR with a β step `0.30 → 0.12` on day 30 (only `I(t)` shown
  to the filter).** The LS fit returns a meaningless average `β ≈ 0.18`; the
  EKF tracks `β̂` from `0.30` down to `0.12` within ~10 days of the step,
  and its `±1σ` band correctly *widens* once `I → 0` (no infections ⇒ no
  information about `β`). **Clear win.**
* **COVID-19 Germany 2020 (122 days, real).** Tracking `β` on SIR
  (`γ` fixed at `0.1`) recovers the first-wave history a constant `β` cannot:
  `R₀ = β/γ ≈ 3.3` during the March exponential growth, collapsing below 1
  (`R₀ ≈ 0.6`) after the lockdown, then rising again toward reopening
  (`R₀ ≈ 1.3`). **Clear win** — and exactly the signal the *Honest caveat*
  above predicts.
* **1978 flu boarding school (14 days, real, single clean wave).** Here a
  constant `β` is already adequate, so the EKF gives **no RMSE improvement**
  (slightly worse, as sub-stepped Euler on a fast `β ≈ 5` epidemic is less
  accurate than the LSODA-based LS simulation) and any "detected" drift is
  mostly filter transient. **Constant-rate LS is the better tool here.**

**Conclusion.** The EKF is effective precisely when the LS fit is *not*:
long enough series in which transmission genuinely varies (interventions,
behaviour, variants). On short, clean, single-wave epidemics the constant-
rate batch fit is preferable. Practical caveats: results depend on the
`--ekf-q` / `--ekf-r` tuning; non-negativity is enforced by clipping but the
LS upper bound (`RATE_MAX`) is **not**, so on a near-railed fit (flu)
`β̂(t)` can wander above it; and only the transmission rate is tracked by
design.

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
    alpha            = 0.20,    # S -> E   per-S exposure rate (independent of I)
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

The SEDIS exposure term is ``alpha * S`` (per-susceptible leakage,
independent of how many spreaders are currently active). modif_SEDIS
replaces this with the mass-action term ``alpha * S * I`` — that is the
*only* structural difference between the two models.

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
from plot_sample import plot_sample

# Run + plot
fig = model_sir(SIRParams(beta=0.4, gamma=0.12))

# Save a JSON sample
create_sample(SIRParams(beta=0.3, gamma=0.1),
              "SIR_sample1.json", n_points=1000)

# Re-plot the saved sample (no ODE re-run, no comparison)
plot_sample("samples/SIR_sample1.json")

# modif_SEDIS (alpha*S*I exposure)
model_modif_sedis(ModifSEDISParams(alpha=2.5e-5, beta1=0.12))
```
