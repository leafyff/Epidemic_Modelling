"""ODE right-hand-side functions for every epidemic model.

Each function follows the signature expected by scipy.integrate.solve_ivp:
``fun(t, y, *args) -> dy/dt`` where *t* is unused for these autonomous systems.
"""


def si_ode(_t: float, y: list[float], beta: float, N: float) -> list[float]:
    """SI model ODEs.

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N
    """
    S, I = y
    new_infected = beta * S * I / N
    return [-new_infected, +new_infected]


def sis_ode(
    _t: float, y: list[float],
    beta: float, gamma: float, N: float,
) -> list[float]:
    """SIS model ODEs. Recovered individuals re-enter Susceptible (no immunity).

    dS/dt = -beta * S * I / N  +  gamma * I
    dI/dt = +beta * S * I / N  -  gamma * I
    """
    S, I = y
    new_infected = beta * S * I / N
    recoveries   = gamma * I
    return [-new_infected + recoveries, +new_infected - recoveries]


def sir_ode(
    _t: float, y: list[float],
    beta: float, gamma: float, N: float,
) -> list[float]:
    """SIR model ODEs. Recovered individuals gain permanent immunity.

    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N  -  gamma * I
    dR/dt = +gamma * I

    Epidemic threshold: R0 = beta / gamma > 1.
    """
    S, I, _R = y
    new_infected = beta * S * I / N
    recoveries   = gamma * I
    return [-new_infected, +new_infected - recoveries, +recoveries]


def seir_ode(
    _t: float, y: list[float],
    beta: float, sigma: float, gamma: float, N: float,
) -> list[float]:
    """SEIR model ODEs. Adds a latent Exposed compartment before Infected.

    dS/dt = -beta * S * I / N
    dE/dt = +beta * S * I / N  -  sigma * E
    dI/dt = +sigma * E          -  gamma * I
    dR/dt = +gamma * I

    Mean incubation period = 1 / sigma days.
    """
    S, E, I, _R = y
    new_exposed  = beta * S * I / N
    new_infected = sigma * E
    recoveries   = gamma * I
    return [
        -new_exposed,
        +new_exposed  - new_infected,
        +new_infected - recoveries,
        +recoveries,
    ]


def sepns_ode(
    _t: float, y: list[float],
    alpha: float, beta1: float, beta2: float,
    mu1: float, mu2: float, mu_e: float, N: float,
) -> list[float]:
    """SEPNS model ODEs for social-network rumour propagation.

    Compartments: S, E, P (pos. infected), N_comp (neg. infected)

    dS/dt = -(alpha * S * (P + N_comp) / N)  +  mu1*P  +  mu2*N_comp  +  mu_e*E
    dE/dt = +(alpha * S * (P + N_comp) / N)  -  (beta1 + beta2 + mu_e) * E
    dP/dt = +beta1 * E  -  mu1 * P
    dN/dt = +beta2 * E  -  mu2 * N_comp
    """
    S, E, P, N_comp = y
    total_spreaders = P + N_comp
    exposure        = alpha * S * total_spreaders / N
    e_out           = (beta1 + beta2 + mu_e) * E
    return [
        -exposure + mu1 * P + mu2 * N_comp + mu_e * E,
        +exposure - e_out,
        +beta1 * E - mu1 * P,
        +beta2 * E - mu2 * N_comp,
    ]


def sedis_ode(
    _t: float, y: list[float],
    alpha: float, beta1: float, beta2: float, gamma: float,
    mu1: float, mu2: float, mu3: float, N: float,
) -> list[float]:
    """SEDIS model ODEs for rumour propagation with a Doubtful compartment.

    Compartments: S, E, D, I

    dS/dt = -(alpha * S * I / N)  +  mu1*E  +  mu2*D  +  mu3*I
    dE/dt = +(alpha * S * I / N)  -  (beta1 + beta2 + mu1) * E
    dD/dt = +beta1 * E            -  (gamma + mu2) * D
    dI/dt = +beta2 * E + gamma * D  -  mu3 * I
    """
    S, E, D, I = y
    exposure = alpha * S * I / N
    e_out    = (beta1 + beta2 + mu1) * E
    d_out    = (gamma + mu2) * D
    return [
        -exposure + mu1 * E + mu2 * D + mu3 * I,
        +exposure - e_out,
        +beta1 * E - d_out,
        +beta2 * E + gamma * D - mu3 * I,
    ]


def sedpnr_ode(
    _t: float, y: list[float],
    alpha: float,
    beta1: float, beta2: float, beta3: float, beta4: float,
    gamma: float, lambda1: float, lambda2: float,
    mu1: float, mu2: float,
) -> list[float]:
    """SEDPNR model ODEs (Govindankutty & Gopalan 2024, equations 5–10).

    Compartments: S, E, D, P, N_comp, R

    dS/dt = mu1*E + mu2*D - alpha*S
    dE/dt = alpha*S - (beta1 + beta2 + gamma + mu1)*E
    dD/dt = gamma*E - (beta3 + beta4 + mu2)*D
    dP/dt = beta1*E + beta3*D - lambda1*P
    dN/dt = beta2*E + beta4*D - lambda2*N_comp
    dR/dt = lambda1*P + lambda2*N_comp

    Note: alpha acts as a per-capita rate on S (mean-field / homogeneous-network
    formulation from the paper).
    """
    S, E, D, P, N_comp, _R = y
    e_out = (beta1 + beta2 + gamma + mu1) * E
    d_out = (beta3 + beta4 + mu2) * D
    return [
        mu1 * E + mu2 * D - alpha * S,
        alpha * S - e_out,
        gamma * E - d_out,
        beta1 * E + beta3 * D - lambda1 * P,
        beta2 * E + beta4 * D - lambda2 * N_comp,
        lambda1 * P + lambda2 * N_comp,
    ]
