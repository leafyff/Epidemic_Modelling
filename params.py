"""Parameter dataclasses for every epidemic model in this package."""

from dataclasses import dataclass


@dataclass
class SIParams:
    """Parameters for the SI (Susceptible-Infected) model.

    The SI model is the simplest epidemic model used here. Individuals start
    as Susceptible (S) or Infected (I). Infection is irreversible in this
    model, so nobody recovers and the infected population can only increase.
    """
    population      : int   = 10_000  # total number of individuals in the network
    initial_infected: int   = 10      # number of infected individuals at t=0
    beta            : float = 0.30    # S -> I   transmission rate
    t_end           : float = 60.0    # simulation end time (days)
    t_steps         : int   = 1_000   # number of equally-spaced time points


@dataclass
class SISParams:
    """Parameters for the SIS (Susceptible-Infected-Susceptible) model.

    The SIS model adds recovery without permanent immunity. Infected
    individuals recover back into the Susceptible (S) compartment, so the
    infection can persist as an endemic state when beta / gamma is greater
    than one.
    """
    population      : int   = 10_000  # total number of individuals in the network
    initial_infected: int   = 10      # number of infected individuals at t=0
    beta            : float = 0.30    # S -> I   transmission rate
    gamma           : float = 0.10    # I -> S   recovery rate (no permanent immunity)
    t_end           : float = 120.0   # simulation end time (days)
    t_steps         : int   = 1_000   # number of equally-spaced time points


@dataclass
class SIRParams:
    """Parameters for the SIR (Susceptible-Infected-Recovered) model.

    The SIR model adds a Recovered (R) compartment. Infected individuals move
    to R after recovery and are treated as permanently immune, so they do not
    return to the Susceptible pool.
    """
    population        : int   = 10_000  # total number of individuals in the network
    initial_infected  : int   = 10      # number of infected individuals at t=0
    initial_recovered : int   = 0       # number of recovered (immune) individuals at t=0
    beta              : float = 0.30    # S -> I   transmission rate
    gamma             : float = 0.10    # I -> R   recovery rate (permanent immunity)
    t_end             : float = 160.0   # simulation end time (days)
    t_steps           : int   = 1_000   # number of equally-spaced time points


@dataclass
class SEIRParams:
    """Parameters for the SEIR (Susceptible-Exposed-Infected-Recovered) model.

    The SEIR model inserts an Exposed (E) compartment between Susceptible and
    Infected. Exposed individuals are infected but not yet infectious; sigma
    controls the transition from E to I, and gamma controls recovery from I to R.
    """
    population      : int   = 10_000  # total number of individuals in the network
    initial_exposed : int   = 0       # number of exposed (latent) individuals at t=0
    initial_infected: int   = 10      # number of infected individuals at t=0
    beta            : float = 0.30    # S -> E   transmission / contact rate
    sigma           : float = 0.20    # E -> I   incubation rate  (1/sigma = mean incubation days)
    gamma           : float = 0.10    # I -> R   recovery rate (permanent immunity)
    t_end           : float = 200.0   # simulation end time (days)
    t_steps         : int   = 1_000   # number of equally-spaced time points


@dataclass
class SEPNSParams:
    """Parameters for the SEPNS (Susceptible-Exposed-Positively Infected-Negatively Infected-Susceptible) model.

    The infected state is split by the *sentiment* of the rumour being spread:
      P – individuals spreading misinformation with a positive tone.
      N – individuals spreading misinformation with a negative tone.
    Both P and N eventually return to S (no permanent immunity for social rumours).
    """
    population          : int   = 10_000  # total number of individuals in the network
    initial_exposed     : int   = 0       # number of exposed individuals at t=0
    initial_pos_infected: int   = 5       # number of positive-sentiment spreaders at t=0
    initial_neg_infected: int   = 5       # number of negative-sentiment spreaders at t=0
    alpha               : float = 0.20    # S  -> E   exposure / contact rate
    beta1               : float = 0.15    # E  -> P   exposed adopts positive-sentiment spreading
    beta2               : float = 0.20    # E  -> N   exposed adopts negative-sentiment spreading (faster)
    mu1                 : float = 0.05    # P  -> S   positive spreader loses interest, returns to susceptible
    mu2                 : float = 0.05    # N  -> S   negative spreader loses interest, returns to susceptible
    mu_e                : float = 0.03    # E  -> S   exposed individual rejects information early
    t_end               : float = 200.0   # simulation end time (days)
    t_steps             : int   = 1_000   # number of equally-spaced time points


@dataclass
class SEDISParams:
    """Parameters for the SEDIS (Susceptible-Exposed-Doubtful-Infected-Susceptible)
    model of rumour propagation.

    Adds a *Doubtful* (D) compartment: individuals who have heard the rumour but
    have not yet decided whether to believe or reject it.  Both D and I can cycle
    back to S because social-media rumours offer no permanent immunity.
    """
    population       : int   = 10_000  # total number of individuals in the network
    initial_exposed  : int   = 0       # number of exposed individuals at t=0
    initial_doubtful : int   = 0       # number of doubtful individuals at t=0
    initial_infected : int   = 10      # number of infected (spreading) individuals at t=0
    alpha            : float = 0.20    # S  -> E   exposure rate
    beta1            : float = 0.10    # E  -> D   exposed becomes sceptical / doubtful
    beta2            : float = 0.15    # E  -> I   exposed directly accepts and spreads
    gamma            : float = 0.08    # D  -> I   doubtful is eventually convinced, starts spreading
    mu1              : float = 0.04    # E  -> S   exposed rejects information early
    mu2              : float = 0.05    # D  -> S   doubtful rejects after rechecking
    mu3              : float = 0.05    # I  -> S   spreader loses interest / information becomes stale
    t_end            : float = 200.0   # simulation end time (days)
    t_steps          : int   = 1_000   # number of equally-spaced time points


@dataclass
class SEDPNRParams:
    """Parameters for the SEDPNR (Susceptible-Exposed-Doubtful-Positively Infected-Negatively Infected-Restrained) model.

    The full misinformation-spread model proposed by Govindankutty & Gopalan (2024).
    It combines sentiment-aware infection (P/N split) with a Doubtful state and a
    terminal Restrained state (individuals who permanently stop spreading).
    See paper equations 5–10.
    """
    population          : int   = 10_000  # total number of individuals in the network
    initial_exposed     : int   = 10      # number of exposed individuals at t=0
    initial_doubtful    : int   = 10      # number of doubtful individuals at t=0
    initial_pos_infected: int   = 5       # number of positive-sentiment spreaders at t=0
    initial_neg_infected: int   = 5       # number of negative-sentiment spreaders at t=0
    initial_restrained  : int   = 0       # number of restrained (silent) individuals at t=0
    alpha               : float = 0.20    # S  -> E   contact / exposure rate
    beta1               : float = 0.15    # E  -> P   exposed adopts positive-sentiment spreading
    beta2               : float = 0.20    # E  -> N   exposed adopts negative-sentiment spreading
    beta3               : float = 0.10    # D  -> P   doubtful converted to positive spreader
    beta4               : float = 0.12    # D  -> N   doubtful converted to negative spreader
    gamma               : float = 0.10    # E  -> D   exposed becomes doubtful
    lambda1             : float = 0.05    # P  -> R   positive spreader becomes restrained
    lambda2             : float = 0.05    # N  -> R   negative spreader becomes restrained
    mu1                 : float = 0.03    # E  -> S   exposed rejects information, returns to susceptible
    mu2                 : float = 0.04    # D  -> S   doubtful rejects after verification
    t_end               : float = 100.0   # simulation end time (days)
    t_steps             : int   = 1_000   # number of equally-spaced time points
