"""Extracting physical observables from integrated geodesics.

These are the functions the validation suite compares against
`kerrgeo.analytic`.  Each one is a small piece of numerical craft in its own
right -- the naive version of both measurements below is wrong at a level that
would swamp the effect being measured.
"""

from __future__ import annotations

import numpy as np

from .events import escape_event, horizon_event, radial_turning_event
from .hamiltonian import photon_from_impact_parameter, rhs
from .integrate import trace


def measure_deflection(metric, b, r0=1e5, r_escape=None, lam_max=None,
                       method="DOP853", rtol=1e-12, atol=1e-12,
                       horizon_eps=1e-6):
    """Light deflection angle for a photon with impact parameter ``b``.

    Returns np.inf if the photon is captured.

    How the angle is extracted
    --------------------------
    The obvious approach -- take phi at the start and phi at the end and
    subtract pi -- carries an error of order b/r0, because at finite radius the
    photon has not yet reached its asymptote.  For b = 50 M and r0 = 1e5 M that
    is 5e-4 rad against a deflection of 8e-2 rad: a 0.6% error, which is larger
    than the entire second-order term we want to test.  Pushing r0 up to fix it
    is expensive and eventually runs into round-off in accumulated phi.

    Instead we read off the *direction of travel* at each end.  Projecting the
    state into the asymptotically-flat plane,

        vx = (dr/dl) cos(phi) - r sin(phi) (dphi/dl)
        vy = (dr/dl) sin(phi) + r cos(phi) (dphi/dl)

    gives the instantaneous heading, and the deflection is the angle between
    the incoming and outgoing headings.  This converges to the true asymptote
    as O(M log r / r) with no O(b/r) term at all, so a modest r0 suffices.
    """
    if r_escape is None:
        r_escape = r0
    if lam_max is None:
        lam_max = 20.0 * r0

    y0 = photon_from_impact_parameter(metric, r0=r0, b=b)
    sol = trace(
        metric, y0, lam_max,
        method=method, rtol=rtol, atol=atol,
        events=[horizon_event(metric, eps=horizon_eps),
                escape_event(r_escape)],
    )

    if sol.r[-1] < r_escape * 0.9:      # never got back out: captured
        return np.inf

    def heading(y):
        d = rhs(0.0, y, metric)
        r, phi = y[1], y[3]
        dr, dphi = d[1], d[3]
        vx = dr * np.cos(phi) - r * np.sin(phi) * dphi
        vy = dr * np.sin(phi) + r * np.cos(phi) * dphi
        return np.arctan2(vy, vx)

    h_in = heading(sol.y[:, 0])
    h_out = heading(sol.y[:, -1])
    return abs(np.unwrap([h_in, h_out])[1] - h_in)


def measure_precession(metric, r_peri, r_apo, n_orbits=1, method="DOP853",
                       rtol=1e-13, atol=1e-13, n_steps=None):
    """Periapsis advance per radial period, in radians.

    Integrates a bound equatorial orbit from periapsis and uses scipy's event
    root-finding to locate subsequent periapsis passages (p_r = 0 with p_r
    increasing).  The advance is

        Delta phi_prec = (phi at next periapsis - phi at this one) - 2 pi

    ``n_orbits`` averages the advance over several radial periods.  Note that
    measurement does *not* improve with n: the error is dominated by
    accumulated integration error in phi, which grows with the integration
    length, not by per-crossing event-detection error, which would average
    down.  Measured relative error at r_peri = 100, r_apo = 200 is 1.8e-12 for
    n = 1 and 2.7e-11 for n = 20.  Use n = 1 unless you specifically want to
    quantify accumulation.

    Precision note.  For a Mercury-like orbit the advance is ~5e-7 rad against
    a swept 2 pi, so the subtraction discards seven significant figures.  There
    is no way to fold the subtraction inside as we did in
    `analytic.precession_exact` -- phi comes out of the integrator as a single
    accumulated number.  This is the fundamental reason the *measured* Mercury
    precession is harder to get than the analytic one, and it is why the
    validation suite reports the achieved relative error rather than asserting
    a fixed tolerance.
    """
    from .hamiltonian import orbit_from_apsides

    y0, E, Lz = orbit_from_apsides(metric, r_peri, r_apo)

    # Radial period is roughly the Keplerian value; integrate generously past
    # the requested number of orbits so the last event is definitely captured.
    # Keplerian estimate of the radial period.  It is only an estimate: deep in
    # the strong field the true period is substantially longer, so integrate
    # well past it rather than trusting the estimate.
    a_sma = 0.5 * (r_peri + r_apo)
    T_est = 2.0 * np.pi * a_sma ** 1.5 / np.sqrt(getattr(metric, "M", 1.0))
    lam_max = 4.0 * n_orbits * T_est

    peri = radial_turning_event(direction=+1, terminal=False)
    kwargs = dict(method=method, events=[peri, horizon_event(metric)])
    if n_steps is not None:
        kwargs["n_steps"] = n_steps
    else:
        kwargs.update(rtol=rtol, atol=atol)
    sol = trace(metric, y0, lam_max, **kwargs)

    if not sol.y_events or len(sol.y_events[0]) == 0:
        raise RuntimeError("no periapsis crossings found; increase lam_max.")

    # The initial state IS a periapsis (p_r = 0 there by construction), so the
    # event function starts at exactly zero and scipy may report a crossing at
    # lambda = 0.  Discard only that one.  Filtering on a large fraction of the
    # estimated period would be wrong: in the strong field the true radial
    # period can be several times the Keplerian estimate, and a genuine
    # periapsis return would be thrown away with it.
    lam_events = np.asarray(sol.t_events[0])
    phi_events = np.asarray(sol.y_events[0])[:, 3]
    keep = lam_events > 1e-6 * T_est
    lam_events, phi_events = lam_events[keep], phi_events[keep]

    if len(phi_events) < n_orbits:
        raise RuntimeError(
            f"found only {len(phi_events)} periapsis returns, needed "
            f"{n_orbits}.  Increase lam_max."
        )

    phi_end = phi_events[n_orbits - 1]
    # Started exactly at periapsis with phi = 0.
    return (phi_end - 2.0 * np.pi * n_orbits) / n_orbits


def capture_threshold(metric, b_lo, b_hi, r0=1e4, tol=1e-10, prograde=True):
    """Bisect on impact parameter to find the photon capture threshold b_c.

    Below b_c the photon crosses the horizon; above it, the photon escapes.
    Exactly at b_c it asymptotes onto the (unstable) circular photon orbit.
    For Schwarzschild the answer is 3 sqrt(3) M and it is one of the sharpest
    single-number tests available: the threshold is a genuinely strong-field
    quantity, and bisection converges on it to any precision you like.
    """
    sgn = 1.0 if prograde else -1.0

    def escapes(b):
        y0 = photon_from_impact_parameter(metric, r0=r0, b=sgn * abs(b))
        sol = trace(metric, y0, 50.0 * r0,
                    events=[horizon_event(metric), escape_event(r0)])
        return sol.r[-1] > 0.9 * r0

    if escapes(b_lo) or not escapes(b_hi):
        raise ValueError(
            f"bracket [{b_lo}, {b_hi}] does not straddle the threshold: "
            f"need capture at b_lo and escape at b_hi."
        )
    while b_hi - b_lo > tol:
        mid = 0.5 * (b_lo + b_hi)
        if escapes(mid):
            b_hi = mid
        else:
            b_lo = mid
    return 0.5 * (b_lo + b_hi)


def reversibility_error(metric, y0, lam_max, method="DOP853", **kw):
    """Integrate forward then back, and report how far the state moved.

    A test that catches an entire class of bugs the invariants miss.  The
    conserved quantities are insensitive to errors *along* the trajectory --
    a geodesic can be at completely the wrong place on the right orbit and
    still report perfect E, Lz and Q.  Time-reversal exercises the actual
    trajectory, and it also catches asymmetric step-size control and mishandled
    event branches.

    Reverses by flipping the sign of all four momenta, which reverses the
    direction of travel while leaving the position untouched.
    """
    fwd = trace(metric, y0, lam_max, method=method, **kw)
    y_mid = fwd.y[:, -1].copy()
    y_mid[4:] *= -1.0
    back = trace(metric, y_mid, fwd.t[-1], method=method, **kw)
    y_end = back.y[:, -1].copy()
    y_end[4:] *= -1.0
    # t runs backwards too, so compare only the spatial coords and momenta.
    return float(np.max(np.abs(y_end[1:] - np.asarray(y0)[1:])))
