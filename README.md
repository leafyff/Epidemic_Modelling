# Epidemic Model Simulator for Misinformation Spread

A Python simulation project for comparing classical epidemic models and misinformation-spread models in digital networks.

The project implements deterministic compartmental models using systems of ordinary differential equations. It includes traditional epidemic models such as SI, SIS, SIR, and SEIR, together with social-network misinformation models such as SEPNS, SEDIS, and SEDPNR.

## Overview

This repository studies how information, rumors, and misinformation can spread through a population by adapting epidemic-modeling ideas to digital networks.

The social-network models represent user behavior such as exposure to misinformation, uncertainty, sentiment-based spreading, rejection after verification, and eventual disengagement from spreading.

The simulator is a console application: `main.py` is an argparse-based CLI that drives the rest of the codebase. Each model is solved with `scipy.integrate.solve_ivp` using the RK45 method and visualized with `matplotlib`. Generated plots are saved as PNG files in the `figs/` directory, and JSON samples (for downstream analysis) are saved in the `samples/` directory.

## Implemented Models

| Model | Full Name | Main Idea | Model's ODE |
|---|---|---|---|
| SI | Susceptible, Infected | The simplest irreversible spread model. Susceptible individuals become infected, and infected individuals remain infected. | $\Large\frac{dS}{dt} = -\frac{\beta S I}{n}$ <br> $\Large\frac{dI}{dt} = \frac{\beta S I}{n}$ |
| SIS | Susceptible, Infected, Susceptible | Infected individuals can return to the susceptible state, so the spread can persist over time. | $\Large\frac{dS}{dt} = -\frac{\beta S I}{n} + \gamma I$ <br> $\Large\frac{dI}{dt} = \frac{\beta S I}{n} - \gamma I$ |
| SIR | Susceptible, Infected, Recovered | Infected individuals recover into a permanently immune or removed state. | $\Large\frac{dS}{dt} = -\frac{\beta S I}{n}$ <br> $\Large\frac{dI}{dt} = \frac{\beta S I}{n} - \gamma I$ <br> $\Large\frac{dR}{dt} = \gamma I$ |
| SEIR | Susceptible, Exposed, Infected, Recovered | Adds a latent exposed state before individuals become infectious. | $\Large\frac{dS}{dt} = -\frac{\beta S I}{n}$ <br> $\Large\frac{dE}{dt} = \frac{\beta S I}{n} - \sigma E$ <br> $\Large\frac{dI}{dt} = \sigma E - \gamma I$ <br> $\Large\frac{dR}{dt} = \gamma I$ |
| SEPNS | Susceptible, Exposed, Positively Infected, Negatively Infected, Susceptible | Splits misinformation spreaders into positive-sentiment and negative-sentiment spreaders. | $\Large\frac{dS}{dt} = -\frac{\alpha S (P + N)}{n} + \mu_1 P + \mu_2 N + \mu_e E$ <br> $\Large\frac{dE}{dt} = \frac{\alpha S (P + N)}{n} - (\beta_1 + \beta_2 + \mu_e) E$ <br> $\Large\frac{dP}{dt} = \beta_1 E - \mu_1 P$ <br> $\Large\frac{dN}{dt} = \beta_2 E - \mu_2 N$ |
| SEDIS | Susceptible, Exposed, Doubtful, Infected, Susceptible | Adds a doubtful state for users who question the information before accepting, rejecting, or spreading it. | $\Large\frac{dS}{dt} = \mu_1 E + \mu_2 D + \mu_3 I - \alpha S$ <br> $\Large\frac{dE}{dt} = \alpha S - (\beta_1 + \beta_2 + \mu_1) E$ <br> $\Large\frac{dD}{dt} = \beta_1 E - (\gamma + \mu_2) D$ <br> $\Large\frac{dI}{dt} = \gamma D + \beta_2 E - \mu_3 I$ |
| SEDPNR | Susceptible, Exposed, Doubtful, Positively Infected, Negatively Infected, Restrained | Combines doubt, sentiment-aware misinformation spreading, and a restrained state for users who permanently stop spreading the misinformation. | $\Large\frac{dS}{dt} = \mu_1 E + \mu_2 D - \alpha S$ <br> $\Large\frac{dE}{dt} = \alpha S - (\beta_1 + \beta_2 + \gamma + \mu_1) E$ <br> $\Large\frac{dD}{dt} = \gamma E - (\beta_3 + \beta_4 + \mu_2) D$ <br> $\Large\frac{dP}{dt} = \beta_1 E + \beta_3 D - \lambda_1 P$ <br> $\Large\frac{dN}{dt} = \beta_2 E + \beta_4 D - \lambda_2 N$ <br> $\Large\frac{dR}{dt} = \lambda_1 P + \lambda_2 N$ |

**Compartments (state variables):**

* $S$ — number of susceptible individuals (can still be infected/influenced)
* $E$ — number of exposed individuals (have encountered the information but are not yet active spreaders)
* $I$ — number of infected / actively spreading individuals
* $R$ — number of recovered (immune or removed) individuals
* $D$ — number of doubtful individuals (questioning the information before deciding)
* $P$ — number of positively-infected spreaders (spread with positive sentiment)
* $N$ — number of negatively-infected spreaders (spread with negative sentiment)
* $n$ — total (initial) population size, $n = S + E + I + \dots$

**Transmission / contact parameters:**

* $\beta$ — infection (transmission) rate from $S$ to $I$ in classic SI/SIS/SIR/SEIR models
* $\alpha$ — contact rate at which susceptibles become exposed in SEPNS/SEDIS/SEDPNR

**Progression parameters (out of $E$ and $D$):**

* $\sigma$ — rate at which exposed individuals become infectious (SEIR)
* $\beta_1$ — transition rate from $E$ to the next state (to $P$ in SEPNS/SEDPNR, to $D$ in SEDIS)
* $\beta_2$ — transition rate from $E$ to the next state (to $N$ in SEPNS/SEDPNR, to $I$ in SEDIS)
* $\gamma$ — recovery / progression rate (from $I$ to $R$ in SIR/SEIR; from $D$ to $I$ in SEDIS; from $E$ to $D$ in SEDPNR)
* $\beta_3$ — transition rate from $D$ to $P$ (SEDPNR)
* $\beta_4$ — transition rate from $D$ to $N$ (SEDPNR)

**Return / removal parameters:**

* $\mu_e$ — rate at which exposed individuals return to susceptible (SEPNS)
* $\mu_1$ — return-to-susceptible rate (from $P$ in SEPNS; from $E$ in SEDIS/SEDPNR)
* $\mu_2$ — return-to-susceptible rate (from $N$ in SEPNS; from $D$ in SEDIS/SEDPNR)
* $\mu_3$ — return-to-susceptible rate from $I$ (SEDIS)
* $\lambda_1$ — rate at which positively-infected spreaders move to the restrained state $R$ (SEDPNR)
* $\lambda_2$ — rate at which negatively-infected spreaders move to the restrained state $R$ (SEDPNR)


## Requirements

Python 3.9 or newer is recommended.

| Dependency   | Purpose                                     |
| ------------ | ------------------------------------------- |
| `numpy`      | Numerical arrays and simulation time grids  |
| `scipy`      | Solving ODE systems with `solve_ivp`        |
| `matplotlib` | Plotting and saving simulation figures      |

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install numpy scipy matplotlib
```

## Usage

This project is a console application. **Read [`Documentation.md`](Documentation.md)** for the project structure, the full list of CLI commands, parameter overrides, and worked examples for both running models and generating JSON samples.

A short taste of the CLI:

```bash
python main.py run-all                                 # run every model with defaults
python main.py run SIR --param beta=0.4 --param gamma=0.12   # run one model with overrides
python main.py sample SIR SIR_sample1.json             # save a JSON sample to samples/
python main.py --help                                  # full CLI help
```

`run` / `run-all` write PNG figures to `figs/` and open them in matplotlib; `sample` writes JSON to `samples/`. See [`Documentation.md`](Documentation.md) for everything else.

## Output Figures

| Model  | Preview                                            |
| ------ | -------------------------------------------------- |
| SI     | ![SI model result](figs/SI_model_ex.png)           |
| SIS    | ![SIS model result](figs/SIS_model_ex.png)         |
| SIR    | ![SIR model result](figs/SIR_model_ex.png)         |
| SEIR   | ![SEIR model result](figs/SEIR_model_ex.png)       |
| SEPNS  | ![SEPNS model result](figs/SEPNS_model_ex.png)     |
| SEDIS  | ![SEDIS model result](figs/SEDIS_model_ex.png)     |
| SEDPNR | ![SEDPNR model result](figs/SEDPNR_model_ex.png)   |

## Default Simulation Parameters

Each model's default parameters live in its `*Params` dataclass under `models/<NAME>.py`. The CLI applies these defaults unless overridden with `--param KEY=VALUE` — see [`Documentation.md`](Documentation.md) for the override reference.

| Model  | Population | Initial Conditions                                                                   | Main Rates                                                                                                                  | Simulation Length |
| ------ | ---------: | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ----------------: |
| SI     |     10,000 | 10 infected                                                                          | `beta = 0.30`                                                                                                               |           60 days |
| SIS    |     10,000 | 10 infected                                                                          | `beta = 0.30`, `gamma = 0.10`                                                                                               |          120 days |
| SIR    |     10,000 | 10 infected, 0 recovered                                                             | `beta = 0.30`, `gamma = 0.10`                                                                                               |          160 days |
| SEIR   |     10,000 | 0 exposed, 10 infected                                                               | `beta = 0.30`, `sigma = 0.20`, `gamma = 0.10`                                                                               |          200 days |
| SEPNS  |     10,000 | 0 exposed, 5 positively infected, 5 negatively infected                              | `alpha = 0.20`, `beta1 = 0.15`, `beta2 = 0.20`, `mu1 = 0.05`, `mu2 = 0.05`, `mu_e = 0.03`                                   |          200 days |
| SEDIS  |     10,000 | 0 exposed, 0 doubtful, 10 infected                                                   | `alpha = 0.20`, `beta1 = 0.10`, `beta2 = 0.15`, `gamma = 0.08`, `mu1 = 0.04`, `mu2 = 0.05`, `mu3 = 0.05`                    |          200 days |
| SEDPNR |     10,000 | 10 exposed, 10 doubtful, 5 positively infected, 5 negatively infected, 0 restrained  | `alpha = 0.20`, `beta1 = 0.15`, `beta2 = 0.20`, `beta3 = 0.10`, `beta4 = 0.12`, `gamma = 0.10`, `lambda1 = 0.05`, `lambda2 = 0.05`, `mu1 = 0.03`, `mu2 = 0.04` | 100 days |

## Customizing the Simulation

Parameters are overridden directly from the command line — no source edits required. For example, run SEDPNR with stronger restraint:

```bash
python main.py run SEDPNR --param lambda1=0.08 --param lambda2=0.08 --param t_end=120
```

Or save the same scenario as a JSON sample for downstream analysis:

```bash
python main.py sample SEDPNR SEDPNR_strong_restraint.json --param lambda1=0.08 --param lambda2=0.08 --n-points 2000
```

The full list of overridable keys per model, and many more examples, is in [`Documentation.md`](Documentation.md).

## Reproducibility

The current implementation is deterministic. It solves ordinary differential equations rather than running a random agent-based simulation. Running the same code with the same parameters should produce the same curves.

| Setting          |                       Value |
| ---------------- | --------------------------: |
| ODE solver       | `scipy.integrate.solve_ivp` |
| Solver method    | `RK45`                      |
| Figure size      | `10 x 5` inches             |
| Figure DPI       | `150`                       |
| Output directory | `figs`                      |

## Notes on Interpretation

The generated curves show compartment-level dynamics under the chosen parameters. They are useful for studying qualitative behavior such as infection peaks, exposed-population growth, sentiment-specific spread, decline of spreaders, and growth of the restrained population.

The plots provide qualitative model behavior rather than calibrated predictions for a specific real-world social network. Operational use would require empirical calibration, platform-specific assumptions, network topology, user-behavior data, and validation.

## Reference

Govindankutty, S., & Gopalan, S. P. (2024). Epidemic modeling for misinformation spread in digital networks through a social intelligence approach. *Scientific Reports, 14*, Article 19100. https://doi.org/10.1038/s41598-024-69657-0

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgment

This project is based on the ideas presented in the Scientific Reports article “Epidemic modeling for misinformation spread in digital networks through a social intelligence approach.”
