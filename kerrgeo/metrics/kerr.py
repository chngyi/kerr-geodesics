"""Kerr spacetime in Boyer-Lindquist coordinates (and Schwarzschild as a=0).

Units are geometric: G = c = 1, and lengths/times are measured in units of M
(so ``M = 1.0`` by default and r = 6 means r = 6GM/c^2).  The spin parameter
``a`` has units of M and lies in [0, M] for a black hole; a = M is extremal
and a > M would be a naked singularity.

Signature is (-, +, +, +), so g_ab u^a u^b = -1 for a massive particle and 0
for a photon.

The line element is

    ds^2 = -(1 - 2Mr/S) dt^2 - (4 a M r sin^2 th / S) dt dphi
           + (S/D) dr^2 + S dth^2
           + (r^2 + a^2 + 2 a^2 M r sin^2 th / S) sin^2 th  dphi^2

with

    S (Sigma) = r^2 + a^2 cos^2 th
    D (Delta) = r^2 - 2Mr + a^2

We never actually need that expression.  What the integrator needs is the
inverse, which is pleasantly compact:

    g^tt   = -A / (S D)          A = (r^2 + a^2)^2 - a^2 D sin^2 th
    g^tphi = -2 a M r / (S D)
    g^rr   = D / S
    g^thth = 1 / S
    g^phiphi = (D - a^2 sin^2 th) / (S D sin^2 th)

Note where this blows up:

  * D -> 0 at the horizons r_pm = M +/- sqrt(M^2 - a^2).  This is a *coordinate*
    singularity -- the curvature is perfectly finite there (see
    `kretschmann`).  Boyer-Lindquist coordinates simply cannot describe
    horizon crossing, because BL time t diverges logarithmically as a worldline
    approaches r_+.  We handle this by terminating integration just outside
    r_+; see `kerrgeo/events.py`.  Anything that reaches the horizon is
    captured and never returns, so for all exterior physics this loses nothing.
    Interior work (closed timelike curves near the ring singularity) requires a
    horizon-penetrating chart instead -- that is why `Metric` is a pluggable
    object rather than hard-coded BL.

  * S -> 0 at r = 0, th = pi/2: the *ring* singularity.  This one is real --
    the Kretschmann scalar diverges.

  * sin th -> 0 on the polar axis.  Also a coordinate artefact (the usual
    spherical-polar one).  Orbits that pass exactly through the axis need care;
    generic ones do not.
"""

from __future__ import annotations

import numpy as np

from .base import Metric


class KerrBL(Metric):
    """Kerr metric in Boyer-Lindquist coordinates, x = (t, r, theta, phi)."""

    cyclic = (0, 3)  # stationary + axisymmetric  =>  E and Lz exactly conserved

    def __init__(self, a: float = 0.0, M: float = 1.0):
        if M <= 0:
            raise ValueError("M must be positive")
        if abs(a) > M:
            raise ValueError(
                f"|a| = {abs(a)} exceeds M = {M}: that is a naked singularity, "
                "not a black hole.  Use 0 <= |a| <= M."
            )
        self.a = float(a)
        self.M = float(M)

    def __repr__(self):
        return f"KerrBL(a={self.a}, M={self.M})"

    # -- the two functions the integrator actually calls ---------------------

    def ginv(self, x):
        """Inverse metric g^{ab}.  Analytic in ``x`` (complex-step safe)."""
        a, M = self.a, self.M
        r, th = x[1], x[2]

        # Use sin/cos directly rather than any branching -- keeps this analytic
        # so the complex-step derivative in Metric.dginv stays valid.
        s2 = np.sin(th) ** 2
        c2 = np.cos(th) ** 2

        S = r * r + a * a * c2                       # Sigma
        D = r * r - 2.0 * M * r + a * a              # Delta
        A = (r * r + a * a) ** 2 - a * a * D * s2

        gi = np.zeros((4, 4), dtype=np.result_type(r, th, 1.0))
        gi[0, 0] = -A / (S * D)
        gi[0, 3] = gi[3, 0] = -2.0 * a * M * r / (S * D)
        gi[1, 1] = D / S
        gi[2, 2] = 1.0 / S
        gi[3, 3] = (D - a * a * s2) / (S * D * s2)
        return gi

    def g(self, x):
        """Covariant metric g_ab, in closed form (cheaper and better
        conditioned near the horizon than inverting ``ginv`` numerically)."""
        a, M = self.a, self.M
        r, th = x[1], x[2]
        s2 = np.sin(th) ** 2
        c2 = np.cos(th) ** 2
        S = r * r + a * a * c2
        D = r * r - 2.0 * M * r + a * a

        gd = np.zeros((4, 4), dtype=np.result_type(r, th, 1.0))
        gd[0, 0] = -(1.0 - 2.0 * M * r / S)
        gd[0, 3] = gd[3, 0] = -2.0 * a * M * r * s2 / S
        gd[1, 1] = S / D
        gd[2, 2] = S
        gd[3, 3] = (r * r + a * a + 2.0 * a * a * M * r * s2 / S) * s2
        return gd

    # -- geometry: closed-form landmarks used for validation -----------------

    def horizons(self):
        """(r_-, r_+).  Roots of Delta = 0."""
        a, M = self.a, self.M
        root = np.sqrt(M * M - a * a)
        return (M - root, M + root)

    @property
    def r_plus(self):
        """Outer event horizon."""
        return self.horizons()[1]

    def r_ergo(self, theta):
        """Outer boundary of the ergosphere (the static limit surface), where
        g_tt = 0.  Inside it no observer can remain at fixed phi: frame
        dragging is not merely strong but compulsory.

        Touches the horizon on the axis (theta = 0, pi) and bulges to r = 2M
        in the equatorial plane.
        """
        a, M = self.a, self.M
        return M + np.sqrt(M * M - a * a * np.cos(theta) ** 2)

    def r_isco(self, prograde: bool = True):
        """Innermost stable circular orbit, equatorial.

        Bardeen, Press & Teukolsky (1972), eq. (2.21).  Reduces to 6M at a=0,
        and for a = M gives 1M prograde / 9M retrograde.
        """
        a, M = abs(self.a), self.M
        x = a / M
        Z1 = 1.0 + (1.0 - x * x) ** (1 / 3) * (
            (1.0 + x) ** (1 / 3) + (1.0 - x) ** (1 / 3)
        )
        Z2 = np.sqrt(3.0 * x * x + Z1 * Z1)
        sign = -1.0 if prograde else 1.0
        return M * (3.0 + Z2 + sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))

    def r_photon(self, prograde: bool = True):
        """Radius of the equatorial circular photon orbit.

        Bardeen et al. (1972), eq. (2.18).  Gives 3M at a=0 (the photon
        sphere), and 1M / 4M for a = M.
        """
        a, M = abs(self.a), self.M
        sign = -1.0 if prograde else 1.0
        return 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(sign * a / M)))

    def kretschmann(self, r, theta):
        """R_abcd R^abcd -- a coordinate-independent curvature scalar.

        Used in the validation suite to demonstrate that nothing physical
        happens at r_+ (this stays finite) while the ring singularity at
        S = 0 is genuine (this diverges).
        """
        a, M = self.a, self.M
        c2 = np.cos(theta) ** 2
        S = r * r + a * a * c2
        ac = a * np.cos(theta)
        return (
            48.0 * M * M * (r * r - ac * ac) * (S * S - 16.0 * r * r * ac * ac)
            / S**6
        )


def Schwarzschild(M: float = 1.0) -> KerrBL:
    """Schwarzschild is Kerr with a = 0.

    Deliberately *not* a separate implementation.  Sharing the code path means
    the Schwarzschild validation suite (which has many exact analytic answers
    to check against) is simultaneously a test of the Kerr code, where far
    fewer closed-form answers exist.
    """
    return KerrBL(a=0.0, M=M)
