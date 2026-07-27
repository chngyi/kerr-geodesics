"""Constants of the motion, and what each one is good for as a diagnostic.

A Kerr geodesic has four constants.  They are *not* interchangeable as error
diagnostics -- they fail in different ways and detect different problems, which
is why the validation suite tracks all four separately rather than reporting a
single "error".

    mu^2 = -g^{ab} p_a p_b       the mass-shell / norm condition
    E    = -p_t                  energy at infinity      (Killing vector d_t)
    Lz   = +p_phi                axial angular momentum  (Killing vector d_phi)
    Q                            Carter's constant       (Killing TENSOR)

E and Lz: exact, so they detect bugs
------------------------------------
Because g^{ab} has no t or phi dependence, the code for dp_t/dl and dp_phi/dl
evaluates to identically zero -- see `metrics/base.py`.  E and Lz therefore
stay put to machine precision (~1e-16) regardless of integrator or step size.
Drift here means a coding error: a wrong index, a metric that accidentally
depends on phi, a bad initial condition.  It is *not* a step-size problem, and
tightening the tolerance will not fix it.

mu^2: measures truncation error, but weakly
-------------------------------------------
2H = g^{ab} p_a p_b is conserved by the exact flow, so its drift is a genuine
integration-error measure -- the one you already planned to use.  Two caveats.
It is a quadratic first integral, so some integrators (implicit Gauss-Legendre
in particular) conserve it far better than they conserve the actual trajectory,
which makes it look optimistic.  And in Schwarzschild it is nearly degenerate
with E and Lz for equatorial orbits, so it is a weak test there.

Q: the honest error measure in Kerr
-----------------------------------
Carter's constant is what makes Kerr integrable.  E and Lz come from the two
Killing *vectors*, and for a generic non-equatorial orbit that is not enough --
you would still have a chaotic-looking 2-D problem in (r, th).  Kerr also
admits an irreducible Killing *tensor* K_ab, and

    Q = K_ab p^a p^b - (Lz - aE)^2
      = p_th^2 + cos^2(th) [ a^2 (mu^2 - E^2) + Lz^2 / sin^2(th) ]

is the associated constant.  It is what separates the Hamilton-Jacobi equation
and reduces the motion to the decoupled first-order form in `separated.py`.

Crucially: Q is *not* protected by any cyclic coordinate.  Nothing in the code
forces it to be conserved.  So in Kerr, Q drift is the clean, sharp measure of
how much truncation error the integrator is actually committing -- exactly the
role energy drift played in your N-body code.  Track it.

Meaning of Q, briefly: Q = 0 is the equatorial plane.  Q > 0 means the orbit
oscillates in theta about the equator, reaching a maximum inclination set by
Theta(th) = 0.  Q < 0 is possible only for orbits confined to one hemisphere,
which exist inside the horizon.
"""

from __future__ import annotations

import numpy as np


def norm(y, metric):
    """g^{ab} p_a p_b.  Should equal -mu^2: -1 timelike, 0 null."""
    return float(np.asarray(metric.ginv(y[:4]), dtype=float) @ y[4:] @ y[4:])


def energy(y):
    """E = -p_t, the conserved energy per unit rest mass measured at infinity."""
    return float(-y[4])


def angular_momentum(y):
    """Lz = p_phi, the conserved angular momentum about the spin axis."""
    return float(y[7])


def carter(y, metric, mu=None):
    """Carter's constant Q.

    Parameters
    ----------
    mu : float or None
        Rest mass (1 timelike, 0 null).  If None it is inferred from the
        current norm, which is slightly circular but convenient; pass it
        explicitly when using Q as an error diagnostic so that norm drift does
        not leak into the Q measurement.
    """
    a = getattr(metric, "a", 0.0)
    th, p_th = y[2], y[6]
    E, Lz = energy(y), angular_momentum(y)
    if mu is None:
        mu2 = -norm(y, metric)
    else:
        mu2 = mu * mu
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    return float(p_th * p_th + c2 * (a * a * (mu2 - E * E) + Lz * Lz / s2))


def all_invariants(y, metric, mu=None):
    """Dict of all four, for logging."""
    return {
        "norm": norm(y, metric),
        "E": energy(y),
        "Lz": angular_momentum(y),
        "Q": carter(y, metric, mu),
    }


def drift_report(ys, metric, mu=1.0):
    """Max absolute drift of each invariant along a trajectory.

    Parameters
    ----------
    ys : ndarray, shape (8, N)
        Trajectory as returned by `kerrgeo.integrate.trace` (``sol.y``).

    Returns
    -------
    dict of str -> float
        Maximum |value - value_at_start| over the trajectory.  Absolute rather
        than relative because Q and the norm both legitimately pass through
        zero (equatorial orbits, photons).
    """
    ys = np.asarray(ys)
    keys = ("norm", "E", "Lz", "Q")
    first = all_invariants(ys[:, 0], metric, mu)
    worst = {k: 0.0 for k in keys}
    for i in range(ys.shape[1]):
        cur = all_invariants(ys[:, i], metric, mu)
        for k in keys:
            worst[k] = max(worst[k], abs(cur[k] - first[k]))
    return worst
