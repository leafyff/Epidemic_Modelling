# Epidemic Model Simulator for Misinformation Spread

A Python simulation project for comparing classical epidemic models and misinformation-spread models in digital networks.

The project implements deterministic compartmental models using systems of ordinary differential equations. It includes traditional epidemic models such as SI, SIS, SIR, and SEIR, together with social-network misinformation models such as SEPNS, SEDIS, and SEDPNR.

The project is based on the paper **“Epidemic modeling for misinformation spread in digital networks through a social intelligence approach”** by Sreeraag Govindankutty and Shynu Padinjappurath Gopalan, published in *Scientific Reports* in 2024.

DOI: https://doi.org/10.1038/s41598-024-69657-0

## Overview

This repository studies how information, rumors, and misinformation can spread through a population by adapting epidemic-modeling ideas to digital networks.

Traditional epidemic models describe how infection moves through biological populations. The social-network models in this project adapt that idea to misinformation spread by representing user states such as exposure, doubt, sentiment-based spreading, rejection after verification, and eventual restraint from further spreading.

The simulations are implemented in `main.py`. Each model is solved with `scipy.integrate.solve_ivp` using the RK45 method and visualized with `matplotlib`. Generated plots are saved as PNG files in the `figs/` directory.

## Implemented Models

| Model | Full Name | Main Idea |
|---|---|---|
| SI | Susceptible, Infected | The simplest irreversible spread model. Susceptible individuals become infected, and infected individuals remain infected. |
| SIS | Susceptible, Infected, Susceptible | Infected individuals can return to the susceptible state, so spread can persist over time. |
| SIR | Susceptible, Infected, Recovered | Infected individuals recover into a permanently immune or removed state. |
| SEIR | Susceptible, Exposed, Infected, Recovered | Adds a latent exposed state before individuals become infectious. |
| SEPNS | Susceptible, Exposed, Positively Infected, Negatively Infected, Susceptible | Splits misinformation spreaders into positive-sentiment and negative-sentiment spreaders. |
| SEDIS | Susceptible, Exposed, Doubtful, Infected, Susceptible | Adds a doubtful state for users who question the information before accepting, rejecting, or spreading it. |
| SEDPNR | Susceptible, Exposed, Doubtful, Positively Infected, Negatively Infected, Restrained | Combines doubt, sentiment-aware misinformation spreading, and a restrained state for users who permanently stop spreading the misinformation. |

## Project Structure

```text
.
├── main.py
├── Epidemic modeling for misinformation spread.pdf
├── LICENSE
└── figs
    ├── SI_model_ex.png
    ├── SIS_model_ex.png
    ├── SIR_model_ex.png
    ├── SEIR_model_ex.png
    ├── SEPNS_model_ex.png
    ├── SEDIS_model_ex.png
    └── SEDPNR_model_ex.png
```

## Requirements

Python 3.9 or newer is recommended.

| Dependency | Purpose |
|---|---|
| `numpy` | Numerical arrays and simulation time grids |
| `scipy` | Solving ODE systems with `solve_ivp` |
| `matplotlib` | Plotting and saving simulation figures |

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

Run the simulator:

```bash
python main.py
```

The script runs all implemented models, prints the main parameters and peak-spread information to the terminal, saves the generated PNG figures into the `figs/` directory, and displays the plots.

## Output Figures

In the equations below, `N_pop` denotes the total population. For models with a negatively infected compartment, `N_c` denotes the negatively infected compartment to avoid confusing it with the total population.

| Model | Model's ODE | Preview |
|---|---|---|
| SI | $\frac{dS}{dt}=-\beta \frac{SI}{N_{pop}}$<br>$\frac{dI}{dt}=\beta \frac{SI}{N_{pop}}$ | ![SI model result](figs/SI_model_ex.png) |
| SIS | $\frac{dS}{dt}=-\beta \frac{SI}{N_{pop}}+\gamma I$<br>$\frac{dI}{dt}=\beta \frac{SI}{N_{pop}}-\gamma I$ | ![SIS model result](figs/SIS_model_ex.png) |
| SIR | $\frac{dS}{dt}=-\beta \frac{SI}{N_{pop}}$<br>$\frac{dI}{dt}=\beta \frac{SI}{N_{pop}}-\gamma I$<br>$\frac{dR}{dt}=\gamma I$ | ![SIR model result](figs/SIR_model_ex.png) |
| SEIR | $\frac{dS}{dt}=-\beta \frac{SI}{N_{pop}}$<br>$\frac{dE}{dt}=\beta \frac{SI}{N_{pop}}-\sigma E$<br>$\frac{dI}{dt}=\sigma E-\gamma I$<br>$\frac{dR}{dt}=\gamma I$ | ![SEIR model result](figs/SEIR_model_ex.png) |
| SEPNS | $\frac{dS}{dt}=-\alpha \frac{S(P+N_c)}{N_{pop}}+\mu_1P+\mu_2N_c+\mu_eE$<br>$\frac{dE}{dt}=\alpha \frac{S(P+N_c)}{N_{pop}}-(\beta_1+\beta_2+\mu_e)E$<br>$\frac{dP}{dt}=\beta_1E-\mu_1P$<br>$\frac{dN_c}{dt}=\beta_2E-\mu_2N_c$ | ![SEPNS model result](figs/SEPNS_model_ex.png) |
| SEDIS | $\frac{dS}{dt}=-\alpha \frac{SI}{N_{pop}}+\mu_1E+\mu_2D+\mu_3I$<br>$\frac{dE}{dt}=\alpha \frac{SI}{N_{pop}}-(\beta_1+\beta_2+\mu_1)E$<br>$\frac{dD}{dt}=\beta_1E-(\gamma+\mu_2)D$<br>$\frac{dI}{dt}=\beta_2E+\gamma D-\mu_3I$ | ![SEDIS model result](figs/SEDIS_model_ex.png) |
| SEDPNR | $\frac{dS}{dt}=\mu_1E+\mu_2D-\alpha S$<br>$\frac{dE}{dt}=\alpha S-(\beta_1+\beta_2+\gamma+\mu_1)E$<br>$\frac{dD}{dt}=\gamma E-(\beta_3+\beta_4+\mu_2)D$<br>$\frac{dP}{dt}=\beta_1E+\beta_3D-\lambda_1P$<br>$\frac{dN_c}{dt}=\beta_2E+\beta_4D-\lambda_2N_c$<br>$\frac{dR}{dt}=\lambda_1P+\lambda_2N_c$ | ![SEDPNR model result](figs/SEDPNR_model_ex.png) |

## Model Descriptions

### SI Model

SI means **Susceptible, Infected**.

`S` represents susceptible individuals who have not yet adopted or received the spreading item. `I` represents infected individuals who are currently carrying or spreading it.

This is the simplest model in the project. Susceptible individuals become infected through contact with infected individuals. The model has no recovery, removal, doubt, immunity, or restraint state. Once an individual moves from `S` to `I`, that individual remains infected for the rest of the simulation.

In the context of information spread, this model represents an extreme baseline where exposure leads to permanent adoption or permanent spreading. Because this assumption is too strong for real misinformation behavior, the SI model is mainly useful as a comparison point for more realistic models.

### SIS Model

SIS means **Susceptible, Infected, Susceptible**.

The first `S` represents susceptible individuals. `I` represents infected individuals. The final `S` represents the return of infected individuals back to the susceptible state.

The SIS model adds recovery without permanent immunity. Infected individuals can stop spreading and return to the susceptible pool. Since they do not become permanently immune or permanently restrained, they may become infected again later.

In this implementation, `beta` controls the transition from `S` to `I`, and `gamma` controls the transition from `I` back to `S`. When the transmission rate is sufficiently larger than the recovery rate, the infection can persist as an endemic state instead of disappearing completely.

For misinformation, SIS can represent repeated exposure to similar claims where users may stop spreading a claim but remain vulnerable to accepting or resharing it again later.

### SIR Model

SIR means **Susceptible, Infected, Recovered**.

`S` represents susceptible individuals. `I` represents infected individuals. `R` represents recovered individuals who no longer participate in the spread.

The SIR model adds a permanently removed or immune state. Once infected individuals recover, they move to `R` and do not return to the susceptible state. This is a standard classical epidemic model and is useful for comparison with social-network misinformation models.

In this implementation, `beta` controls infection, and `gamma` controls recovery. The ratio `beta / gamma` is used as the basic reproduction number for the classical SIR dynamics.

For misinformation, the recovered state can be interpreted as users who have stopped spreading and are no longer susceptible to the same item. This is a stronger assumption than many social-network situations allow, because users can often become susceptible again when a rumor resurfaces, changes form, or receives social reinforcement.

### SEIR Model

SEIR means **Susceptible, Exposed, Infected, Recovered**.

`S` represents susceptible individuals. `E` represents exposed individuals who have encountered the spreading item but are not yet actively spreading it. `I` represents infected individuals who are actively spreading. `R` represents recovered individuals who are removed from the spreading process.

The SEIR model adds a latent stage between susceptibility and infection. This makes it more realistic than SIR for processes where exposure does not immediately create active spreading. In biological epidemic modeling, `E` often represents infected but not yet infectious individuals. In information-spread modeling, `E` can represent users who have seen the content but have not yet decided whether to spread it.

In this implementation, `beta` controls movement from `S` to `E`, `sigma` controls movement from `E` to `I`, and `gamma` controls movement from `I` to `R`. The mean exposed-stage duration is represented by `1 / sigma`.

### SEPNS Model

SEPNS means **Susceptible, Exposed, Positively Infected, Negatively Infected, Susceptible**.

`S` represents susceptible individuals. `E` represents exposed individuals. `P` represents positively infected individuals who spread misinformation with a positive or approving sentiment. `N` represents negatively infected individuals who spread misinformation with a negative, distrustful, critical, or emotionally reactive sentiment. The final `S` represents the return of spreaders back to susceptibility.

The SEPNS model adapts epidemic modeling to social-network rumor propagation by splitting the infected state according to sentiment. This is useful because misinformation is often not spread in a neutral way. Users may share the same false claim approvingly, fearfully, angrily, or critically, and those emotional tones can affect how the content propagates.

In this implementation, susceptible users move to the exposed state through contact with positive and negative spreaders. Exposed users may become positive spreaders, become negative spreaders, or reject the information and return to `S`. Positive and negative spreaders can lose interest and return to the susceptible state.

This model is useful when sentiment matters but a separate doubtful or restrained state is not required.

### SEDIS Model

SEDIS means **Susceptible, Exposed, Doubtful, Infected, Susceptible**.

`S` represents susceptible individuals. `E` represents exposed individuals who have encountered the misinformation. `D` represents doubtful individuals who are uncertain, skeptical, or checking the information. `I` represents infected individuals who accept and spread the misinformation. The final `S` represents the return of exposed, doubtful, or infected users back to susceptibility.

The SEDIS model adds a human-selection component through the doubtful state. This is important for misinformation modeling because users do not always accept or spread information immediately after exposure. Some users pause, question the content, compare it with other sources, or wait for more confirmation.

In this implementation, susceptible users become exposed through contact with infected users. Exposed users may become doubtful, directly become infected, or reject the information and return to susceptible. Doubtful users may eventually become infected after further exposure or persuasion, or they may reject the misinformation and return to susceptible. Infected users eventually lose interest or stop spreading and return to susceptible.

This model is more realistic than simple epidemic models for social-media rumor spread because it includes uncertainty and delayed belief formation.

### SEDPNR Model

SEDPNR means **Susceptible, Exposed, Doubtful, Positively Infected, Negatively Infected, Restrained**.

`S` represents susceptible individuals who have not adopted or engaged with the misinformation. `E` represents exposed individuals who have encountered the misinformation. `D` represents doubtful individuals who are skeptical, uncertain, or verifying the information. `P` represents positively infected users who spread the misinformation with positive sentiment. `N` represents negatively infected users who spread it with negative sentiment. `R` represents restrained users who have stopped spreading the misinformation.

The SEDPNR model is the most complete misinformation-spread model in this project. It combines the exposed and doubtful states with sentiment-aware spreading and a terminal restrained state. In this model, `R` means **Restrained**, not **Recovered**. This distinction matters because the model is intended for misinformation spread, where users may stop spreading because the topic becomes stale, because they verify the information, because they lose interest, or because interventions reduce further sharing.

In this implementation, susceptible users move to exposed at rate `alpha`. Exposed users may become positively infected, negatively infected, doubtful, or return to susceptible. Doubtful users may become positively or negatively infected, or they may return to susceptible after verification or loss of interest. Positive and negative spreaders eventually become restrained.

The article presents SEDPNR as a model for misinformation in digital networks that accounts for social intelligence, sentiment, doubt, and restraint. Compared with SIR and SEIR, SEDPNR is focused less on biological infection and more on user psychology, social influence, belief state, and sharing behavior.

## Default Simulation Parameters

The parameters are configured directly in `main.py`.

| Model | Population | Initial Conditions | Main Rates | Simulation Length |
|---|---:|---|---|---:|
| SI | 10,000 | 10 infected | `beta = 0.30` | 60 days |
| SIS | 10,000 | 10 infected | `beta = 0.30`, `gamma = 0.10` | 120 days |
| SIR | 10,000 | 10 infected, 0 recovered | `beta = 0.30`, `gamma = 0.10` | 160 days |
| SEIR | 10,000 | 0 exposed, 10 infected | `beta = 0.30`, `sigma = 0.20`, `gamma = 0.10` | 200 days |
| SEPNS | 10,000 | 0 exposed, 5 positively infected, 5 negatively infected | `alpha = 0.20`, `beta1 = 0.15`, `beta2 = 0.20`, `mu1 = 0.05`, `mu2 = 0.05`, `mu_e = 0.03` | 200 days |
| SEDIS | 10,000 | 0 exposed, 0 doubtful, 10 infected | `alpha = 0.20`, `beta1 = 0.10`, `beta2 = 0.15`, `gamma = 0.08`, `mu1 = 0.04`, `mu2 = 0.05`, `mu3 = 0.05` | 200 days |
| SEDPNR | 10,000 | 10 exposed, 10 doubtful, 5 positively infected, 5 negatively infected, 0 restrained | `alpha = 0.20`, `beta1 = 0.15`, `beta2 = 0.20`, `beta3 = 0.10`, `beta4 = 0.12`, `gamma = 0.10`, `lambda1 = 0.05`, `lambda2 = 0.05`, `mu1 = 0.03`, `mu2 = 0.04` | 100 days |

## Customizing the Simulation

To change a simulation, edit the corresponding parameter object in `main.py`.

For example, the SEDPNR parameters are configured like this:

```python
sedpnr_params = SEDPNRParams(
    population           = 10_000,
    initial_exposed      = 10,
    initial_doubtful     = 10,
    initial_pos_infected = 5,
    initial_neg_infected = 5,
    initial_restrained   = 0,
    alpha                = 0.20,
    beta1                = 0.15,
    beta2                = 0.20,
    beta3                = 0.10,
    beta4                = 0.12,
    gamma                = 0.10,
    lambda1              = 0.05,
    lambda2              = 0.05,
    mu1                  = 0.03,
    mu2                  = 0.04,
    t_end                = 100.0,
    t_steps              = 1_000,
)
```

After changing the parameters, rerun:

```bash
python main.py
```

The figures in `figs/` will be regenerated.

## Reproducibility

The current implementation is deterministic. It solves ordinary differential equations instead of running a random agent-based simulation. Running the same code with the same parameters should produce the same curves.

| Setting | Value |
|---|---:|
| ODE solver | `scipy.integrate.solve_ivp` |
| Solver method | `RK45` |
| Figure size | `10 x 5` inches |
| Figure DPI | `150` |
| Output directory | `figs` |

## Notes on Interpretation

The generated curves show compartment-level dynamics under the chosen parameters. They are useful for studying qualitative behavior such as infection peaks, exposed-population growth, sentiment-specific spread, decline of spreaders, and growth of the restrained population.

The plots provide qualitative model behavior rather than calibrated predictions for a specific real-world social network. Operational use would require empirical calibration, platform-specific assumptions, network topology, user-behavior data, and validation.

## References

Govindankutty, S., & Gopalan, S. P. (2024). Epidemic modeling for misinformation spread in digital networks through a social intelligence approach. *Scientific Reports, 14*, Article 19100. https://doi.org/10.1038/s41598-024-69657-0

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgment

This project is based on the ideas presented in the Scientific Reports article “Epidemic modeling for misinformation spread in digital networks through a social intelligence approach.”
