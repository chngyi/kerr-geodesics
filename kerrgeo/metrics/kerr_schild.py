"""Kerr in ingoing Kerr (Eddington-Finkelstein-type) coordinates.

This is the stage-3 chart: the one that covers horizon crossing and the
interior, all the way through the disk at r = 0 into the negative-r region
where the closed timelike curves live.

--------------------------------------------------------------------------
Why Boyer-Lindquist fails, and what fixes it
--------------------------------------------------------------------------
BL time t is the clock of a distant static observer, and that observer never
sees anything reach r_+: along an infalling geodesic t diverges
logarithmically while the affine parameter stays finite.  The chart runs out
of numbers.  The fix is to relabel time along *infalling light*: define

    dv      = dt   + (r^2 + a^2) / Delta  dr
    dphi~   = dphi + a / Delta            dr

v is constant on ingoing principal null rays (it is the "advanced time"), and
phi~ absorbs the infinite winding that frame dragging imposes on anything
approaching the horizon in BL coordinates -- an infalling particle crosses
r_+ at finite phi~ even though its BL phi has wound around infinitely often.

--------------------------------------------------------------------------
The inverse metric: every 1/Delta cancels
--------------------------------------------------------------------------
Transforming the BL inverse metric with the Jacobian of the relabelling,
g'^{mn} = (dx'^m/dx^a)(dx'^n/dx^b) g^{ab}, the divergent pieces cancel
algebraically.  Two examples worth seeing once (primes suppressed, s = sin):

    g^vv    = g^tt + F'^2 g^rr
            = -A/(Sigma Delta) + [(r^2+a^2)^2/Delta^2](Delta/Sigma)
            = [ (r^2+a^2)^2 - A ] / (Sigma Delta)
            = a^2 s^2 Delta / (Sigma Delta)          =  a^2 sin^2(th)/Sigma

    g^vphi~ = g^tphi + F'G' g^rr
            = [ -2aMr + a(r^2+a^2) ] / (Sigma Delta) =  a/Sigma

and similarly for the rest.  The result is *simpler* than BL:

    g^vv     = a^2 sin^2(th) / Sigma        g^rr      = Delta / Sigma
    g^vr     = (r^2 + a^2) / Sigma          g^rphi~   = a / Sigma
    g^vphi~  = a / Sigma                    g^thth    = 1 / Sigma
    g^phi~phi~ = 1 / (Sigma sin^2(th))

Nothing blows up at Delta = 0: both horizons are ordinary places in this
chart, as the finite curvature always said they were.  The only true
singularities left are Sigma = 0 (the ring) and the polar axis.

The chart is also perfectly valid for r < 0: Sigma = r^2 + a^2 cos^2(th) > 0
off the ring, so geodesics can be followed through the r = 0 disk into the
negative-r sheet -- which is where g_phiphi < 0 and closed timelike curves
appear.

--------------------------------------------------------------------------
What carries over unchanged
--------------------------------------------------------------------------
The relabelling touches only t and phi, and only by functions of r.  So the
Killing vectors d_v = d_t and d_phi~ = d_phi are THE SAME vectors, and

    E  = -p_v      Lz = p_phi~      p_theta unchanged

are numerically identical to their BL values.  The chart is still stationary
and axisymmetric (cyclic = (0, 3)), so E and Lz remain *exactly* conserved,
and Carter's Q -- built from E, Lz and p_theta -- keeps its formula.  Every
diagnostic in `kerrgeo.invariants` works verbatim on ingoing states.

The one momentum that changes is radial:

    p_v = p_t,   p_phi~ = p_phi,   p_r' = p_r + [ (r^2+a^2)E - a Lz ] / Delta

The 1/Delta divergence of BL p_r for an infalling geodesic is exactly
cancelled by this shift: finite physics, finite numbers, once you stop using
the coordinate that was hiding it.
"""

from __future__ import annotations

import numpy as np

from .base import Metric


class KerrIngoing(Metric):
    """Kerr metric in ingoing Kerr coordinates, x = (v, r, theta, phi~).

    Signature (-, +, +, +), G = c = 1.  Valid for all r (both signs), both
    horizons included; singular only on the ring Sigma = 0 and the axis.
    """

    cyclic = (0, 3)     # stationary + axisymmetric: E and Lz exact, as in BL
    coords = ("v", "r", "theta", "phi~")

    def __init__(self, a: float = 0.0, M: float = 1.0):
        if M <= 0:
            raise ValueError("M must be positive")
        if abs(a) > M:
            raise ValueError(f"|a| = {abs(a)} > M = {M}: naked singularity")
        self.a = float(a)
        self.M = float(M)

    def __repr__(self):
        return f"KerrIngoing(a={self.a}, M={self.M})"

    # -- the two functions the integrator calls ------------------------------

    def ginv(self, x):
        """Inverse metric.  Analytic in x (complex-step safe), regular at
        Delta = 0, valid for r < 0."""
        a, M = self.a, self.M
        r, th = x[1], x[2]
        s2 = np.sin(th) ** 2
        c2 = np.cos(th) ** 2
        S = r * r + a * a * c2
        D = r * r - 2.0 * M * r + a * a

        gi = np.zeros((4, 4), dtype=np.result_type(r, th, 1.0))
        gi[0, 0] = a * a * s2 / S
        gi[0, 1] = gi[1, 0] = (r * r + a * a) / S
        gi[0, 3] = gi[3, 0] = a / S
        gi[1, 1] = D / S
        gi[1, 3] = gi[3, 1] = a / S
        gi[2, 2] = 1.0 / S
        gi[3, 3] = 1.0 / (S * s2)
        return gi

    def g(self, x):
        """Covariant metric, closed form (from the line element).

        Note g_rr = 0: r is null-adapted here.  g_phiphi is identical to its
        BL value -- the phi relabelling is by a function of r only -- which is
        why the CTC criterion g_phiphi < 0 can be checked in either chart.
        """
        a, M = self.a, self.M
        r, th = x[1], x[2]
        s2 = np.sin(th) ** 2
        c2 = np.cos(th) ** 2
        S = r * r + a * a * c2

        gd = np.zeros((4, 4), dtype=np.result_type(r, th, 1.0))
        gd[0, 0] = -(1.0 - 2.0 * M * r / S)
        gd[0, 1] = gd[1, 0] = 1.0
        gd[0, 3] = gd[3, 0] = -2.0 * a * M * r * s2 / S
        gd[1, 3] = gd[3, 1] = -a * s2
        gd[2, 2] = S
        gd[3, 3] = (r * r + a * a + 2.0 * a * a * M * r * s2 / S) * s2
        return gd

    # -- geometry ------------------------------------------------------------

    def horizons(self):
        a, M = self.a, self.M
        root = np.sqrt(M * M - a * a)
        return (M - root, M + root)

    @property
    def r_plus(self):
        return self.horizons()[1]

    @property
    def r_minus(self):
        return self.horizons()[0]

    def g_phiphi(self, r, theta):
        """Covariant g_phiphi -- the norm of the closed azimuthal circles.

        Where this is negative, the circle of constant (v, r, theta) traced
        by phi~ in [0, 2pi) is a closed *timelike* curve.  That happens only
        on the negative-r sheet near the ring: the 2 a^2 M r sin^2/Sigma term
        needs r < 0 and Sigma small to overpower r^2 + a^2.
        """
        a, M = self.a, self.M
        s2 = np.sin(theta) ** 2
        S = r * r + a * a * np.cos(theta) ** 2
        return (r * r + a * a + 2.0 * a * a * M * r * s2 / S) * s2

    def ctc_loop_proper_time(self, r, theta=np.pi / 2):
        """Proper time to traverse the closed timelike curve once:
        tau = 2 pi sqrt(-g_phiphi).  NaN where the circle is not timelike.

        This is a legitimate (accelerated, non-geodesic) worldline: an
        observer following it returns to the same event -- same v, same
        everything -- having aged this much.
        """
        gpp = self.g_phiphi(r, theta)
        return 2.0 * np.pi * np.sqrt(np.where(gpp < 0, -gpp, np.nan))

    # -- chart transfer (valid in the BL overlap region, r > r_+) ------------

    def tortoise(self, r):
        """r*(r) with dr*/dr = (r^2+a^2)/Delta, for r > r_+.

        v = t + r*.  The log divergence at r_+ is exactly the divergence of
        BL t along infalling rays -- moved into the coordinate relabelling
        where it belongs.
        """
        a, M = self.a, self.M
        rm, rp = self.horizons()
        if np.any(np.asarray(r) <= rp):
            raise ValueError("tortoise(r) implemented for the exterior only")
        if rp == rm:
            raise NotImplementedError("extremal case not needed here")
        return (r
                + (2.0 * M * rp / (rp - rm)) * np.log((r - rp) / (2.0 * M))
                - (2.0 * M * rm / (rp - rm)) * np.log((r - rm) / (2.0 * M)))

    def phi_shift(self, r):
        """G(r) with dG/dr = a/Delta, for r > r_+.  phi~ = phi + G."""
        a = self.a
        rm, rp = self.horizons()
        if np.any(np.asarray(r) <= rp):
            raise ValueError("phi_shift(r) implemented for the exterior only")
        return (a / (rp - rm)) * np.log((r - rp) / (r - rm))


# ---------------------------------------------------------------------------
# Moving states between the charts (exterior overlap only)
# ---------------------------------------------------------------------------

def bl_to_ingoing(metric_in: KerrIngoing, y_bl):
    """Transform a BL phase-space state (r > r_+) to ingoing coordinates.

    Positions:  v = t + r*(r),  phi~ = phi + G(r).
    Momenta:    p_v = p_t,  p_phi~ = p_phi,  p_theta unchanged,
                p_r' = p_r + [(r^2+a^2) E - a Lz] / Delta.
    """
    a, M = metric_in.a, metric_in.M
    t, r, th, ph = y_bl[:4]
    p_t, p_r, p_th, p_ph = y_bl[4:]
    D = r * r - 2.0 * M * r + a * a
    P = (r * r + a * a) * (-p_t) - a * p_ph

    return np.array([
        t + metric_in.tortoise(r),
        r, th,
        ph + metric_in.phi_shift(r),
        p_t,
        p_r + P / D,
        p_th,
        p_ph,
    ])


def ingoing_to_bl(metric_in: KerrIngoing, y_in):
    """Inverse of `bl_to_ingoing`.  Exterior only, like everything BL."""
    a, M = metric_in.a, metric_in.M
    v, r, th, ph = y_in[:4]
    p_v, p_r, p_th, p_ph = y_in[4:]
    D = r * r - 2.0 * M * r + a * a
    P = (r * r + a * a) * (-p_v) - a * p_ph

    return np.array([
        v - metric_in.tortoise(r),
        r, th,
        ph - metric_in.phi_shift(r),
        p_v,
        p_r - P / D,
        p_th,
        p_ph,
    ])


def state_from_constants_ingoing(metric, x, E, Lz, Q=0.0, mu=1.0,
                                 sign_theta=1.0, branch="ingoing"):
    """Build an ingoing-chart state from the constants of motion.

    The BL builder solved g^rr p_r^2 = ... because BL has no radial cross
    terms.  This chart does (g^vr, g^rphi~), so the mass-shell condition is a
    genuine quadratic in p_r, with the two roots

        p_r = ( P +/- sqrt(R) ) / Delta,      P = (r^2+a^2) E - a Lz

    and R the same radial potential as always.  The +/- picks outgoing vs
    ingoing; at the horizon Delta -> 0 and the outgoing root diverges (the
    physical pile-up of outgoing rays) while the ingoing root stays finite.

    Evaluating (P - sqrt(R))/Delta directly would subtract nearly equal
    numbers near the horizon; multiplying by the conjugate gives

        p_r(ingoing) = K / ( P + sqrt(R) ),   K = mu^2 r^2 + (Lz - aE)^2 + Q

    with no Delta anywhere -- manifestly finite at both horizons.  The same
    cancellation lesson as the periapsis initial conditions in stage 1.
    """
    from ..separated import polar_potential, radial_potential

    a, M = metric.a, metric.M
    r, th = x[1], x[2]

    Th = polar_potential(th, a, E, Lz, Q, mu)
    if Th < 0:
        if Th > -1e-12:
            Th = 0.0
        else:
            raise ValueError(f"Theta(theta) = {Th:.4g} < 0: forbidden angle")
    p_th = sign_theta * np.sqrt(Th)

    R = radial_potential(r, a, E, Lz, Q, mu, M)
    if R < 0:
        if R > -1e-12:
            R = 0.0
        else:
            raise ValueError(f"R(r) = {R:.4g} < 0: forbidden radius")

    P = (r * r + a * a) * E - a * Lz
    K = mu * mu * r * r + (Lz - a * E) ** 2 + Q
    if branch == "ingoing":
        p_r = K / (P + np.sqrt(R))
    elif branch == "outgoing":
        D = r * r - 2.0 * M * r + a * a
        if abs(D) < 1e-10:
            raise ValueError("outgoing branch is singular at a horizon")
        p_r = (P + np.sqrt(R)) / D
    else:
        raise ValueError("branch must be 'ingoing' or 'outgoing'")

    return np.concatenate((np.asarray(x, dtype=float),
                           [-E, p_r, p_th, Lz]))


def principal_null_ingoing(metric, r0, theta0):
    """The ingoing principal null congruence: the rays the chart is built on.

    In these coordinates they are breathtakingly simple.  With
    p = (-1, 0, 0, a sin^2 th) one finds, from dx/dlambda = g^{ab} p_b:

        dv/dlambda = 0,  dtheta/dlambda = 0,  dphi~/dlambda = 0,
        dr/dlambda = -1  ...exactly.

    The photon moves in a straight coordinate line r = r0 - lambda at
    constant (v, theta, phi~), through r_+, through r_-, through the disk at
    r = 0 (if theta0 != pi/2), and out into the negative-r sheet.  Its Carter
    constant is Q = -a^2 cos^4(theta0) < 0: these are 'vortical' geodesics,
    confined to a cone around the axis -- which is precisely what lets them
    pass through the disk instead of hitting the ring.

    Because the exact solution is linear, comparing the integrator against it
    through both horizons is the sharpest interior test available.
    """
    a = metric.a
    s2 = np.sin(theta0) ** 2
    x = np.array([0.0, r0, theta0, 0.0])
    p = np.array([-1.0, 0.0, 0.0, a * s2])
    return np.concatenate((x, p))
