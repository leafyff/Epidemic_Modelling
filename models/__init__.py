"""Epidemic models package.

Each module in this package is fully self-contained: it holds the parameter
dataclass, the ODE right-hand-side function and the public simulation function
for one model, along with a complete written description of the model.

Re-exports below let callers write::

    from models import SIRParams, model_sir

without needing to know which file a given symbol lives in.
"""

from .SEDIS import SEDISParams, model_sedis, sedis_ode
from .SEDPNR import SEDPNRParams, model_sedpnr, sedpnr_ode
from .SEIR import SEIRParams, model_seir, seir_ode
from .SEPNS import SEPNSParams, model_sepns, sepns_ode
from .SI import SIParams, model_si, si_ode
from .SIR import SIRParams, model_sir, sir_ode
from .SIS import SISParams, model_sis, sis_ode

__all__ = [
    "SIParams",     "si_ode",     "model_si",
    "SISParams",    "sis_ode",    "model_sis",
    "SIRParams",    "sir_ode",    "model_sir",
    "SEIRParams",   "seir_ode",   "model_seir",
    "SEPNSParams",  "sepns_ode",  "model_sepns",
    "SEDISParams",  "sedis_ode",  "model_sedis",
    "SEDPNRParams", "sedpnr_ode", "model_sedpnr",
]
