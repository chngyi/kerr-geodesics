"""The geodesic equation in Hamiltonian form, and how to start a geodesic.

--------------------------------------------------------------------------
Why Hamiltonian rather than Christoffel symbols
--------------------------------------------------------------------------
The textbook geodesic equation is second order,

    d^2 x^a / dl^2  =  - Gamma^a_{bc}  (dx^b/dl)(dx^c/dl)

which you turn into 8 first-order ODEs for (x^a, u^a).  It works, but for Kerr
it means coding 40 independent Christoffel symbols, each a page of algebra, and
every one of them a place to make a sign error that you will then spend a week
hunting.

The equivalent Hamiltonian description takes

    H(x, p) = (1/2) g^{ab}(x) p_a p_b

with p_a the *covariant* momentum (index down), and gives Hamilton's equations

    dx^a/dl = + dH/dp_a  =  g^{ab} p_b
    dp_a/dl = - dH/dx^a  = -(1/2) (d_a g^{bc}) p_b p_c

Three concrete advantages, in increasing order of importance:

1. You need only the inverse metric and its gradient -- five nonzero
   components for Kerr, not forty Christoffels.

2. H is itself conserved (it has no explicit l-dependence), and
   2H = g^{ab} p_a p_b = -mu^2.  So the four-velocity norm you wanted as an
   error diagnostic is exactly the conserved Hamiltonian.  Its drift is a
   direct measure of how badly the integrator is violating the geometry.

3. t and phi do not appear in g^{ab} for Kerr in BL coordinates, so
   dp_t/dl and dp_phi/dl are *identically zero in the code itself*.  Energy
   E = -p_t and axial angular momentum Lz = p_phi are conserved to machine
   precision no matter what integrator you use or how big the step is.

Point 3 is worth dwelling on, because it changes what validation means.  In
your N-body code, energy drift measured truncation error.  Here E and Lz drift
measures *bugs* -- if they move at all, something is wrong with the code, not
the step size.  The quantity that actually measures truncation error in Kerr is
Carter's constant Q, which arises from a Killing tensor rather than a cyclic
coordinate and gets no such protection.  See `kerrgeo/invariants.py`.

--------------------------------------------------------------------------
What about the fully separated first-order (Carter) form?
--------------------------------------------------------------------------
Kerr is special: the Hamilton-Jacobi equation separates, and you can reduce the
motion to

    S dr/dl   = +/- sqrt(R(r))
    S dth/dl  = +/- sqrt(Theta(th))

plus quadratures for t and phi, with R and Theta built from E, Lz, Q and mu.
That is far faster -- two nontrivial ODEs instead of eight.

We implement it in `kerrgeo/separated.py`, but as a *cross-check*, not as the
workhorse, for two reasons.  First, the +/- branches must be flipped by hand at
every radial and polar turning point, exactly where sqrt(R) -> 0 and the
derivative is infinite, which is numerically delicate.  Second, and more
importantly, it assumes E, Lz and Q are constant -- so it cannot be validated
by checking that they are.  Two independent formulations agreeing is a much
stronger statement than either one self-reporting.

--------------------------------------------------------------------------
The affine parameter
--------------------------------------------------------------------------
We integrate with respect to l, an affine parameter.  For a massive particle
l = proper time tau (given the normalisation below); for a photon proper time
is identically zero and l is just some affine parameter along the null ray --
it has no direct physical meaning, only its ratios do.

Do NOT parametrise by coordinate time t.  Near the horizon dt/dl -> infinity
(that logarithmic freezing is the whole reason infalling matter appears to
stall at r_+), which makes the system arbitrarily stiff in t while it stays
perfectly smooth in l.  Affine parametrisation is what makes horizon approach
numerically benign.
"""

from __future__ import annotations

import numpy as np


def rhs(lam, y, metric):
    """Right-hand side of Hamilton's equations.

    Parameters
    ----------
    lam : float
        Affine parameter (unused -- the system is autonomous; scipy wants it).
    y : ndarray, shape (8,)
        Phase-space state [t, r, theta, phi, p_t, p_r, p_theta, p_phi].
    metric : Metric

    Returns
    -------
    ndarray, shape (8,)
    """
    x, p = y[:4], y[4:]
    gi = metric.ginv(x)
    dgi = metric.dginv(x)

    dx = gi @ p                                    # dx^a/dl = g^{ab} p_b
    dp = -0.5 * np.einsum("cab,a,b->c", dgi, p, p)  # dp_c/dl = -(1/2) d_c g^{ab} p_a p_b
    return np.concatenate((dx, dp))


def hamiltonian(y, metric):
    """2H = g^{ab} p_a p_b.  Equals -mu^2: -1 for a massive particle
    (with l = proper time), 0 for a photon."""
    return float(np.asarray(metric.ginv(y[:4])) @ y[4:] @ y[4:])


# ---------------------------------------------------------------------------
# Building initial conditions
# ---------------------------------------------------------------------------

def momenta_from_constants(metric, x, E, Lz, Q=0.0, mu=1.0,
                           sign_r=-1.0, sign_theta=1.0):
    """Construct p_a at position ``x`` from the conserved quantities.

    This is the natural way to specify a geodesic in Kerr: rather than picking
    a velocity direction, you pick the constants of motion that label the
    orbit, and the momenta follow.

        p_t   = -E          (E = energy per unit mass, measured at infinity)
        p_phi = +Lz         (axial angular momentum per unit mass)

        p_theta^2 = Q - cos^2(th) [ a^2 (mu^2 - E^2) + Lz^2 / sin^2(th) ]
                  = Theta(th)

        p_r    from the mass-shell condition g^{ab} p_a p_b = -mu^2

    The last step is easy because the BL inverse metric has no g^{r a} cross
    terms: p_r appears only as g^{rr} p_r^2, so

        p_r^2 = ( -mu^2 - [g^tt E^2 - 2 g^tphi E Lz
                            + g^thth p_th^2 + g^phiphi Lz^2] ) / g^rr

    Parameters
    ----------
    mu : float
        1.0 for a massive particle (timelike), 0.0 for a photon (null).
    sign_r, sign_theta : float
        Which branch of the square roots -- i.e. whether the geodesic starts
        moving inward/outward and north/south.  Default sign_r = -1 is
        infalling, which is what you want for a ray coming in from far away.

    Raises
    ------
    ValueError
        If the requested constants are not attainable at ``x`` -- i.e. the
        point lies outside the allowed region for that orbit (Theta < 0 or
        R < 0).  This is a physical statement, not a numerical failure: it
        means you asked for a turning point you have already passed.
    """
    a = getattr(metric, "a", 0.0)
    th = x[2]
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2

    theta_pot = Q - c2 * (a * a * (mu * mu - E * E) + Lz * Lz / s2)
    if theta_pot < 0:
        if theta_pot > -1e-12:
            theta_pot = 0.0  # on the turning point, up to rounding
        else:
            raise ValueError(
                f"Theta(theta) = {theta_pot:.6g} < 0 at theta = {th:.6g}: this "
                "polar angle is forbidden for the requested (E, Lz, Q)."
            )
    p_th = sign_theta * np.sqrt(theta_pot)

    gi = np.asarray(metric.ginv(x), dtype=float)
    rest = (gi[0, 0] * E * E
            - 2.0 * gi[0, 3] * E * Lz
            + gi[2, 2] * p_th * p_th
            + gi[3, 3] * Lz * Lz)
    r_pot = (-mu * mu - rest) / gi[1, 1]
    if r_pot < 0:
        if r_pot > -1e-12:
            r_pot = 0.0
        else:
            raise ValueError(
                f"R(r) = {r_pot:.6g} < 0 at r = {x[1]:.6g}: this radius is "
                "forbidden for the requested (E, Lz, Q).  You are inside a "
                "potential barrier / outside a turning point."
            )
    p_r = sign_r * np.sqrt(r_pot)

    return np.array([-E, p_r, p_th, Lz], dtype=float)


def state_from_constants(metric, x, E, Lz, Q=0.0, mu=1.0,
                         sign_r=-1.0, sign_theta=1.0):
    """Full 8-vector initial state.  Convenience wrapper."""
    p = momenta_from_constants(metric, x, E, Lz, Q, mu, sign_r, sign_theta)
    return np.concatenate((np.asarray(x, dtype=float), p))


def photon_from_impact_parameter(metric, r0, b, phi0=0.0, theta0=np.pi / 2):
    """An equatorial photon incoming from radius ``r0`` with impact parameter b.

    For a photon only the ratio b = Lz/E is physical (rescaling the affine
    parameter rescales E and Lz together), so we set E = 1 and Lz = b.

    Sign convention: b > 0 is prograde (co-rotating with the hole), b < 0 is
    retrograde.  In Schwarzschild the two are equivalent by reflection; in Kerr
    they are emphatically not, which is the whole point of the frame-dragging
    study.
    """
    x = np.array([0.0, r0, theta0, phi0])
    return state_from_constants(metric, x, E=1.0, Lz=b, Q=0.0, mu=0.0,
                                sign_r=-1.0, sign_theta=0.0)


def circular_orbit(metric, r, prograde=True):
    """Equatorial circular timelike orbit at radius ``r``: returns (state, E, Lz).

    Closed form for Kerr (Bardeen, Press & Teukolsky 1972, eqs. 2.12-2.13):

        E  = (r^2 - 2Mr +/- a sqrt(Mr)) / (r sqrt(r^2 - 3Mr +/- 2a sqrt(Mr)))
        Lz = +/- sqrt(Mr) (r^2 -/+ 2a sqrt(Mr) + a^2)
                / (r sqrt(r^2 - 3Mr +/- 2a sqrt(Mr)))

    with the upper sign for prograde.  The square root in the denominator goes
    imaginary inside the photon orbit -- there are no circular timelike orbits
    there at all -- which is a useful thing to see the code refuse to do.
    """
    a, M = getattr(metric, "a", 0.0), getattr(metric, "M", 1.0)
    s = 1.0 if prograde else -1.0
    rt = np.sqrt(M * r)

    disc = r * r - 3.0 * M * r + 2.0 * s * a * rt
    if disc <= 0:
        raise ValueError(
            f"No circular timelike orbit at r = {r:.6g} "
            f"(r_photon = {metric.r_photon(prograde):.6g} for this spin/sense)."
        )
    denom = r * np.sqrt(disc)

    E = (r * r - 2.0 * M * r + s * a * rt) / denom
    Lz = s * rt * (r * r - 2.0 * s * a * rt + a * a) / denom

    x = np.array([0.0, r, np.pi / 2, 0.0])
    p = np.array([-E, 0.0, 0.0, Lz])
    return np.concatenate((x, p)), E, Lz


def zamo_drop_state(metric, r0):
    """A zero-angular-momentum particle released from radial rest at r0.

    The cleanest statement of frame dragging.  With Lz = p_phi = 0 the
    particle carries no angular momentum whatsoever, yet its azimuthal
    velocity is

        dphi/dt = g^{phi t} p_t / (g^{tt} p_t) = -g_tphi / g_phiphi = omega(r)

    -- it corotates with the hole at exactly the frame-drag rate, spiralling
    as it falls with no torque ever acting on it.  (The two expressions above
    are equal by the 2x2 block-inverse identity, so `KerrBL.omega` gives the
    same number by an independent formula -- a consistency check the tests
    use.)

    The energy is fixed by requiring the release to be from radial rest:
    R(r0) = 0 with Lz = Q = 0 gives

        E = r0 sqrt( Delta / [ (r0^2 + a^2)^2 - a^2 Delta ] ) = r0 sqrt(Delta/A)

    which is also, not coincidentally, the redshift factor of the ZAMO frame.
    """
    a, M = getattr(metric, "a", 0.0), getattr(metric, "M", 1.0)
    D = r0 * r0 - 2.0 * M * r0 + a * a
    if D <= 0:
        raise ValueError(f"r0 = {r0:g} is inside the horizon")
    A = (r0 * r0 + a * a) ** 2 - a * a * D          # equatorial
    E = r0 * np.sqrt(D / A)
    x = np.array([0.0, r0, np.pi / 2, 0.0])
    p = np.array([-E, 0.0, 0.0, 0.0])               # at the turning point exactly
    return np.concatenate((x, p))


def spherical_photon_orbit(metric, r):
    """A photon on the (unstable) spherical orbit of constant BL radius r.

    Kerr has photon orbits at every radius between the prograde and retrograde
    equatorial circles, each with its own (xi, eta) from
    `analytic.kerr_photon_orbit_constants`.  For eta > 0 the orbit leaves the
    equatorial plane and winds on a sphere between polar turning points --
    the fully three-dimensional generalisation of the Schwarzschild photon
    sphere, and the skeleton on which the black-hole shadow is built.

    Starts on the equator heading north.  Since R(r) = R'(r) = 0 there by
    construction, p_r is set to exactly zero rather than computed through a
    square root of a rounding-level quantity (same lesson as
    `circular_orbit`).  Being unstable, the orbit will eventually peel off --
    watching |r - r0| grow from ~1e-9 is a direct measurement of accumulated
    integration error feeding the instability.
    """
    from .analytic import kerr_photon_orbit_constants

    a = getattr(metric, "a", 0.0)
    M = getattr(metric, "M", 1.0)
    if abs(a) < 1e-12:
        raise ValueError("for a = 0 use the equatorial photon sphere at r = 3M")
    lo, hi = metric.r_photon(True), metric.r_photon(False)
    if not (lo <= r <= hi):
        raise ValueError(
            f"no spherical photon orbit at r = {r:g}: the allowed range for "
            f"a = {a:g} is [{lo:.4f}, {hi:.4f}]")
    xi, eta = kerr_photon_orbit_constants(r, a, M)
    if eta < 0:
        raise ValueError(
            f"eta = {eta:.4g} < 0 at r = {r:g}: this radius' orbit is not "
            "reachable from the equator (it lies at the extreme prograde edge)")
    x = np.array([0.0, r, np.pi / 2, 0.0])
    p = np.array([-1.0, 0.0, np.sqrt(eta), xi])     # E = 1 normalisation
    return np.concatenate((x, p))


def spherical_orbit(metric, r, Q, prograde=True):
    """A timelike orbit of constant BL radius r with Carter constant Q > 0:
    an inclined circle, precessing about the spin axis.

    Solves R(r) = 0, R'(r) = 0 for (E, Lz) at the given Q -- the same
    conditions that define a circular orbit, but off the equator.  These are
    the orbits on which Lense-Thirring nodal precession is measured cleanly:
    r is exactly constant, so the node advance per polar period is the *only*
    secular drift in the problem.

    The inclination follows from Q: cos(i) = Lz / sqrt(Lz^2 + Q), so small Q
    means nearly equatorial.  Returns (state, E, Lz); the state starts on the
    equator heading north (p_theta = +sqrt(Q), exact since Theta(pi/2) = Q).
    """
    from scipy.optimize import fsolve

    from .separated import radial_potential, radial_potential_deriv

    a = getattr(metric, "a", 0.0)
    M = getattr(metric, "M", 1.0)

    _, E0, L0 = circular_orbit(metric, r, prograde)   # Q = 0 starting guess

    def residual(v):
        Ev, Lv = v
        return [radial_potential(r, a, Ev, Lv, Q, 1.0, M),
                radial_potential_deriv(r, a, Ev, Lv, Q, 1.0, M)]

    (E, Lz), info, ok, msg = fsolve(residual, [E0, L0], full_output=True)
    if ok != 1:
        raise RuntimeError(f"spherical-orbit solve failed at r = {r:g}: {msg}")

    x = np.array([0.0, r, np.pi / 2, 0.0])
    p = np.array([-E, 0.0, np.sqrt(Q), Lz])
    return np.concatenate((x, p)), float(E), float(Lz)


def orbit_from_apsides(metric, r_peri, r_apo, prograde=True):
    """Equatorial bound timelike orbit with the given periapsis and apoapsis.

    Solves the two simultaneous conditions R(r_peri) = R(r_apo) = 0 for (E, Lz).
    For the Schwarzschild case this has the clean closed form below; for a != 0
    we fall back to a 2-D root find on the effective potential.

    Returned state starts *at periapsis* moving outward, which makes
    periapsis-to-periapsis precession measurement trivial: you just look for
    the next p_r = 0 crossing in the same direction.
    """
    a, M = getattr(metric, "a", 0.0), getattr(metric, "M", 1.0)
    if r_apo <= r_peri:
        raise ValueError("need r_apo > r_peri")

    if a == 0.0:
        rp, ra = r_peri, r_apo
        # dr/dl = 0 at both apsides means the effective-potential condition
        #   E^2 = (1 - 2M/rp)(1 + L^2/rp^2) = (1 - 2M/ra)(1 + L^2/ra^2)
        # holds at each.  Subtracting gives a linear equation for L^2:
        #   L^2 [ (1-2M/rp)/rp^2 - (1-2M/ra)/ra^2 ] = 2M/rp - 2M/ra
        #
        # Do NOT evaluate that quotient as written.  For a weak-field orbit
        # (rp ~ 3e7 M, as for Mercury) each bracket is a difference of numbers
        # within 1e-8 of each other, and you lose eight significant figures to
        # cancellation before the integration even starts.  Clearing the
        # denominators analytically -- the common factor (ra - rp) cancels
        # exactly -- gives the algebraically identical but numerically stable
        L2 = (2.0 * M * rp * rp * ra * ra
              / (rp * ra * (ra + rp) - 2.0 * M * (ra * ra + ra * rp + rp * rp)))
        Lz = np.sqrt(L2) * (1.0 if prograde else -1.0)
        E = np.sqrt((1.0 - 2.0 * M / rp) * (1.0 + L2 / (rp * rp)))
    else:
        from scipy.optimize import fsolve

        def residual(v):
            Ev, Lv = v
            out = []
            for rr in (r_peri, r_apo):
                D = rr * rr - 2.0 * M * rr + a * a
                out.append((Ev * (rr * rr + a * a) - a * Lv) ** 2
                           - D * (rr * rr + (Lv - a * Ev) ** 2))
            return out

        guess_L = np.sqrt(M * (r_peri + r_apo) / 2.0) * (1.0 if prograde else -1.0)
        E, Lz = fsolve(residual, [0.97, guess_L], full_output=False)

    # p_r = 0 is set exactly, not obtained from momenta_from_constants.  At a
    # turning point the radial potential R(r) vanishes, so computing
    # p_r = sqrt(R/g^rr) evaluates a square root of a quantity that is zero up
    # to rounding -- and sqrt turns a 1e-16 absolute error into 1e-8.  For a
    # Mercury-like orbit that starts the particle a measurable distance off
    # periapsis and produces a spurious precession ~1000x larger than the real
    # 5e-7 rad/orbit signal.  We are at periapsis by construction; say so.
    x = np.array([0.0, r_peri, np.pi / 2, 0.0])
    p = np.array([-E, 0.0, 0.0, Lz], dtype=float)
    return np.concatenate((x, p)), float(E), float(Lz)
