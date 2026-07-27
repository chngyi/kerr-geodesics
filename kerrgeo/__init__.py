"""kerrgeo -- geodesic integration in Kerr and Schwarzschild spacetimes.

Geometric units throughout: G = c = 1, lengths in units of M.
Signature (-, +, +, +).

Quick start
-----------
>>> import numpy as np
>>> from kerrgeo import Schwarzschild, trace, photon_from_impact_parameter
>>> from kerrgeo.events import horizon_event, escape_event
>>> bh = Schwarzschild()
>>> y0 = photon_from_impact_parameter(bh, r0=1e4, b=8.0)
>>> sol = trace(bh, y0, lam_max=5e4,
...             events=[horizon_event(bh), escape_event(2e4)])
>>> sol.status
'event'
"""

from .metrics.base import Metric
from .metrics.kerr import KerrBL, Schwarzschild
from .hamiltonian import (
    rhs,
    hamiltonian,
    momenta_from_constants,
    state_from_constants,
    photon_from_impact_parameter,
    circular_orbit,
    orbit_from_apsides,
)
from .invariants import norm, energy, angular_momentum, carter, drift_report
from .integrate import trace, rk4, gauss_legendre, Solution
from . import analytic, events, measure, separated

__all__ = [
    "Metric", "KerrBL", "Schwarzschild",
    "rhs", "hamiltonian", "momenta_from_constants", "state_from_constants",
    "photon_from_impact_parameter", "circular_orbit", "orbit_from_apsides",
    "norm", "energy", "angular_momentum", "carter", "drift_report",
    "trace", "rk4", "gauss_legendre", "Solution",
    "analytic", "events", "measure", "separated",
]

__version__ = "0.1.0"
