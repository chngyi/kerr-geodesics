"""The separated (Carter) first-order form -- an independent second opinion.

Kerr is not merely a solution of Einstein's equations; it is an *integrable*
one.  Carter (1968) showed that the Hamilton-Jacobi equation separates, so the
four-dimensional motion decouples into two one-dimensional problems:

    Sigma dr/dtau  = +/- sqrt(R(r))
    Sigma dth/dtau = +/- sqrt(Theta(th))

    R(r)     = P^2 - Delta [ mu^2 r^2 + (Lz - aE)^2 + Q ],    P = E(r^2+a^2) - aLz
    Theta(th) = Q - cos^2(th) [ a^2 (mu^2 - E^2) + Lz^2/sin^2(th) ]

plus quadratures for t and phi.  The radial equation involves only r, the polar
equation only theta.  That is what Carter's constant Q buys you, and it is the
entire reason Kerr geodesics are tractable at all -- without it you would have
a genuinely two-dimensional (r, th) dynamical system with no closed form.

Why this module exists
----------------------
Not for speed, though it is faster.  It exists so the validation suite can
compare two formulations that share no code.  The Hamiltonian integrator in
`kerrgeo.hamiltonian` can self-report that E, Lz and Q are conserved -- but a
formulation that *assumes* they are conserved cannot be checked that way, and
conversely, a bug in the metric would corrupt the Hamiltonian path while
leaving this one untouched.  Two independent derivations agreeing to 1e-11 on
the same generic inclined orbit is a far stronger statement than either one's
internal diagnostics.

Two implementation choices worth explaining
-------------------------------------------
1. Mino time.  Substituting dtau = Sigma dlambda_M removes the shared 1/Sigma
   factor and leaves dr/dlambda_M = +/- sqrt(R), dth/dlambda_M = +/- sqrt(Theta)
   -- completely decoupled.  Mino time is not proper time and has no direct
   physical meaning; it is a reparametrisation chosen because it makes the
   equations separate cleanly.

2. Second-order form, to avoid the square roots.  The obvious approach is to
   integrate the first-order equations directly and flip the +/- branch at every
   turning point.  That is exactly where sqrt(R) -> 0 and d/dr sqrt(R) -> infinity,
   so it is both numerically delicate and fiddly to code.  Differentiating
   (dr/dlambda_M)^2 = R(r) instead gives

       d^2 r / dlambda_M^2 = (1/2) R'(r)
       d^2 th/ dlambda_M^2 = (1/2) Theta'(th)

   with no square roots and no branches at all -- turning points are handled
   automatically, because the trajectory simply decelerates through them.  The
   cost is that we now integrate 6 equations rather than 2, and the constants
   R and Theta are enforced only through the initial conditions.  For a
   cross-check that is a good trade.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def radial_potential(r, a, E, Lz, Q, mu=1.0, M=1.0):
    """R(r), the radial effective potential.  dr/dlambda_M = +/- sqrt(R)."""
    P = E * (r * r + a * a) - a * Lz
    D = r * r - 2.0 * M * r + a * a
    K = (Lz - a * E) ** 2 + Q
    return P * P - D * (mu * mu * r * r + K)


def radial_potential_deriv(r, a, E, Lz, Q, mu=1.0, M=1.0):
    """dR/dr."""
    P = E * (r * r + a * a) - a * Lz
    D = r * r - 2.0 * M * r + a * a
    K = (Lz - a * E) ** 2 + Q
    dP = 2.0 * E * r
    dD = 2.0 * r - 2.0 * M
    return 2.0 * P * dP - dD * (mu * mu * r * r + K) - D * 2.0 * mu * mu * r


def polar_potential(th, a, E, Lz, Q, mu=1.0):
    """Theta(theta), the polar effective potential."""
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    return Q - c2 * (a * a * (mu * mu - E * E) + Lz * Lz / s2)


def polar_potential_deriv(th, a, E, Lz, Q, mu=1.0):
    """dTheta/dtheta.

    Using d(cos^2)/dth = -2 sin cos and d(cot^2)/dth = -2 cos/sin^3.
    """
    c, s = np.cos(th), np.sin(th)
    return 2.0 * c * s * a * a * (mu * mu - E * E) + 2.0 * Lz * Lz * c / s**3


def trace_separated(metric, x0, E, Lz, Q, mu=1.0, lam_max=100.0,
                    sign_r=-1.0, sign_theta=1.0, rtol=1e-12, atol=1e-12,
                    n_out=2000):
    """Integrate a geodesic using the separated Carter equations.

    Parameters
    ----------
    x0 : array_like, shape (4,)
        Initial (t, r, theta, phi).
    E, Lz, Q, mu : float
        The constants of motion.
    lam_max : float
        Range of *Mino* time -- not proper time and not the affine parameter
        used by `kerrgeo.integrate.trace`.  See `mino_to_affine` for converting
        between them when comparing trajectories.

    Returns
    -------
    dict with keys 'lam_mino', 'tau', 't', 'r', 'theta', 'phi'
    """
    a = getattr(metric, "a", 0.0)
    M = getattr(metric, "M", 1.0)

    R0 = radial_potential(x0[1], a, E, Lz, Q, mu, M)
    T0 = polar_potential(x0[2], a, E, Lz, Q, mu)
    if R0 < -1e-10 or T0 < -1e-10:
        raise ValueError(
            f"initial point is outside the allowed region: R = {R0:.6g}, "
            f"Theta = {T0:.6g} (both must be >= 0)."
        )
    dr0 = sign_r * np.sqrt(max(R0, 0.0))
    dth0 = sign_theta * np.sqrt(max(T0, 0.0))

    def deriv(lm, s):
        t, r, th, ph, dr, dth, _tau = s
        D = r * r - 2.0 * M * r + a * a
        S = r * r + a * a * np.cos(th) ** 2
        s2 = np.sin(th) ** 2
        P = E * (r * r + a * a) - a * Lz

        dt = -a * (a * E * s2 - Lz) + (r * r + a * a) * P / D
        dph = -(a * E - Lz / s2) + a * P / D
        ddr = 0.5 * radial_potential_deriv(r, a, E, Lz, Q, mu, M)
        ddth = 0.5 * polar_potential_deriv(th, a, E, Lz, Q, mu)
        # Proper time is carried as a state variable, dtau/dlambda_M = Sigma,
        # so it inherits the integrator's order.  Post-hoc trapezoid
        # integration of Sigma would be only 2nd-order accurate and would
        # dominate the error budget when comparing against the Hamiltonian
        # trajectory -- it cost 8e-3 in r before this was fixed.
        return [dt, dr, dth, dph, ddr, ddth, S]

    s0 = [x0[0], x0[1], x0[2], x0[3], dr0, dth0, 0.0]
    lam_eval = np.linspace(0.0, lam_max, n_out)
    sol = solve_ivp(deriv, (0.0, lam_max), s0, method="DOP853",
                    rtol=rtol, atol=atol, t_eval=lam_eval, dense_output=True)

    r, th = sol.y[1], sol.y[2]

    return {
        "lam_mino": sol.t,
        "tau": sol.y[6],
        "t": sol.y[0],
        "r": r,
        "theta": th,
        "phi": sol.y[3],
        "dr": sol.y[4],
        "dtheta": sol.y[5],
    }


def constants_from_state(y, metric, mu=1.0):
    """Extract (E, Lz, Q) from a Hamiltonian state vector.

    Bridge between the two formulations: run the Hamiltonian integrator, pull
    the constants out of the initial state with this, then feed them to
    `trace_separated` and compare the resulting trajectories.
    """
    from .invariants import angular_momentum, carter, energy

    return energy(y), angular_momentum(y), carter(y, metric, mu)
