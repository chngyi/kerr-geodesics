"""Termination and detection events, including horizon handling.

--------------------------------------------------------------------------
The coordinate singularity at the horizon
--------------------------------------------------------------------------
Delta = r^2 - 2Mr + a^2 vanishes at r_pm = M +/- sqrt(M^2 - a^2).  In the
Boyer-Lindquist inverse metric this makes g^tt and g^tphi blow up like 1/Delta
while g^rr = Delta/Sigma goes to zero.  Nothing physical is happening: the
Kretschmann scalar (see `KerrBL.kretschmann`) is perfectly finite at r_+.  The
chart is what fails, because BL time t is the time measured by a distant static
observer, and that observer never sees anything cross.

Two consequences for us, one benign and one that constrains the whole project:

1. Because we integrate in an *affine* parameter, not t, the trajectory itself
   stays smooth right down to r_+.  Only dt/dl diverges.  So the correct
   handling for all exterior work is simply to stop: put a terminal event just
   outside r_+ and label the geodesic "captured".  Nothing that reaches the
   horizon comes back out, so no exterior physics is lost.  For ray tracing,
   captured = black pixel.

   The offset must be small enough not to distort results but large enough that
   1/Delta stays representable.  With eps = 1e-6, Delta ~ 1e-6 near r_+ and the
   metric components reach ~1e6 -- fine in double precision.  The validation
   suite includes a scan showing the deflection angle is insensitive to eps
   across several decades, which is the evidence that this choice is harmless.

2. The interior region -- stage 3 of this project, the closed timelike curves
   near the ring singularity -- is NOT reachable this way.  You cannot integrate
   *through* r_+ in BL coordinates at any tolerance, because the chart genuinely
   does not cover the crossing.  That requires a horizon-penetrating chart:
   ingoing Kerr / Kerr-Schild coordinates, related to BL by

       dv     = dt   + (r^2 + a^2)/Delta  dr
       dphi~  = dphi + a/Delta            dr

   in which the metric is g = eta + f l l with everything regular at both
   horizons.  This is exactly why `Metric` is an abstract, pluggable object
   rather than hard-coded BL: stage 3 is a new metric class, not a rewrite.
   (It is also the only way to reach r < 0 through the ring, which is where the
   CTC region g_phiphi < 0 actually lives.)
"""

from __future__ import annotations

import numpy as np


def horizon_event(metric, eps=1e-6, terminal=True):
    """Fires when the geodesic reaches (1 + eps) * r_+.

    For a = M the two horizons merge and r_+ = M; for a Schwarzschild hole
    r_+ = 2M.
    """
    r_stop = metric.r_plus * (1.0 + eps)

    def ev(lam, y, *args):
        return y[1] - r_stop

    ev.terminal = terminal
    ev.direction = -1.0  # only trigger on inward crossing
    return ev


def ring_event(metric, eps=1e-3, terminal=True):
    """Fires near the ring singularity, where Sigma = r^2 + a^2 cos^2 -> 0.

    This is the *genuine* singularity -- the Kretschmann scalar diverges as
    1/Sigma^6 -- so unlike the horizons there is no chart that continues
    through it.  Termination here is physics, not a coordinate artefact.
    Only equatorial approaches can reach it: off the equator Sigma >= a^2
    cos^2(theta) > 0 and the geodesic threads the disk instead.
    """
    a = getattr(metric, "a", 0.0)

    def ev(lam, y, *args):
        r, th = y[1], y[2]
        return r * r + a * a * np.cos(th) ** 2 - eps * eps

    ev.terminal = terminal
    ev.direction = -1.0
    return ev


def negative_r_escape_event(r_min=-50.0, terminal=True):
    """Fires when the geodesic sails deep into the negative-r sheet.

    In the maximally extended Kerr interior, passing through the r = 0 disk
    leads to an asymptotically flat region of *negative* r (where the mass
    reads as negative).  A geodesic getting here has genuinely left through
    the ring.
    """

    def ev(lam, y, *args):
        return y[1] - r_min

    ev.terminal = terminal
    ev.direction = -1.0
    return ev


def escape_event(r_max, terminal=True):
    """Fires when the geodesic gets out to r_max -- i.e. it escaped."""

    def ev(lam, y, *args):
        return y[1] - r_max

    ev.terminal = terminal
    ev.direction = +1.0
    return ev


def radial_turning_event(direction=0.0, terminal=False):
    """Fires at periapsis / apoapsis, where dr/dl = 0.

    Outside the horizon g^rr = Delta/Sigma > 0, so dr/dl = g^rr p_r vanishes
    exactly when p_r does -- we can watch the momentum component directly.

    direction = +1 catches periapsis (p_r going negative -> positive, i.e. the
    turn from infalling to outgoing); -1 catches apoapsis; 0 catches both.
    """

    def ev(lam, y, *args):
        return y[5]

    ev.terminal = terminal
    ev.direction = float(direction)
    return ev


def polar_turning_event(terminal=False):
    """Fires at maximum excursion in theta, where p_theta = 0."""

    def ev(lam, y, *args):
        return y[6]

    ev.terminal = terminal
    ev.direction = 0.0
    return ev


def equator_crossing_event(terminal=False):
    """Fires when the geodesic crosses the equatorial plane theta = pi/2.

    Used by the ray tracer to find where a ray meets a thin accretion disc.
    """

    def ev(lam, y, *args):
        return y[2] - np.pi / 2

    ev.terminal = terminal
    ev.direction = 0.0
    return ev
