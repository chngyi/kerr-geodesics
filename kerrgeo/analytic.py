"""Closed-form and quadrature reference values to validate the integrator against.

The point of this module is to give the validation suite something to compare
to that shares *no code* with the integrator.  Everything here is either an
algebraic formula or a one-dimensional quadrature of the orbit equation -- no
geodesic integration, no metric objects, no Hamiltonian.  If the integrator and
these agree to 1e-10, the agreement means something.

Two of these deserve comment because they are stronger tests than the textbook
weak-field formulas usually quoted:

`deflection_exact` and `precession_exact` reduce the orbit to the standard
u = 1/r form and evaluate the resulting elliptic integral by quadrature, after
a substitution that removes the inverse-square-root endpoint singularities
analytically.  They are exact at *any* impact parameter or eccentricity, not
just in the weak field.  So they test the strong-field regime near the photon
sphere, where the weak-field series is useless and where all the interesting
errors live.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Schwarzschild landmarks
# ---------------------------------------------------------------------------

def photon_sphere(M=1.0):
    """Radius of the unstable circular photon orbit: r = 3GM/c^2."""
    return 3.0 * M


def isco(M=1.0):
    """Innermost stable circular orbit for Schwarzschild: r = 6GM/c^2."""
    return 6.0 * M


def critical_impact_parameter(M=1.0):
    """b_c = 3 sqrt(3) M ~= 5.196 M.

    A photon with b < b_c is captured; with b > b_c it escapes; at b = b_c it
    asymptotes onto the photon sphere and orbits forever.  This is also the
    apparent radius of the black hole shadow as seen from far away -- note that
    it is 5.196 M, appreciably larger than the 3M photon sphere itself, because
    of the gravitational lensing between the sphere and the observer.
    """
    return 3.0 * np.sqrt(3.0) * M


def horizon(M=1.0):
    """Schwarzschild radius r_s = 2GM/c^2."""
    return 2.0 * M


# ---------------------------------------------------------------------------
# Light deflection
# ---------------------------------------------------------------------------

def deflection_weak(b, M=1.0, order=3):
    """Weak-field series for the light deflection angle.

        alpha = 4M/b + (15 pi/4)(M/b)^2 + (128/3)(M/b)^3 + ...

    (Keeton & Petters 2005.)  The leading term is Einstein's 1915 result -- the
    one that is twice the Newtonian value and got measured at Sobral in 1919.

    Checking that the integrator reproduces the *second* term is a much sharper
    test than the first: 4M/b alone is recovered by almost any approximately
    correct calculation, whereas the 15 pi/4 coefficient is sensitive to the
    actual spatial curvature of the metric.
    """
    x = M / np.asarray(b, dtype=float)
    a = 4.0 * x
    if order >= 2:
        a = a + (15.0 * np.pi / 4.0) * x**2
    if order >= 3:
        a = a + (128.0 / 3.0) * x**3
    return a


def deflection_exact(b, M=1.0):
    """Exact Schwarzschild deflection angle for impact parameter ``b``.

    Null geodesics in Schwarzschild satisfy, with u = 1/r,

        (du/dphi)^2 = 1/b^2 - u^2 + 2M u^3   =:  f(u)

    so the total swept angle is  2 * integral_0^{u0} du / sqrt(f(u)),  where u0
    is the turning point (the smallest positive root of f), and the deflection
    is that minus pi.

    The integrand diverges as (u0 - u)^{-1/2} at the endpoint.  Rather than ask
    a quadrature routine to cope, we remove it exactly: writing
    f(u) = 2M (u - r1)(u - r2)(u - r3) and substituting u = u0 - t^2 cancels
    the offending factor against du = -2t dt, leaving a smooth integrand.

    Returns np.inf for b <= b_c, where the photon is captured and there is no
    turning point.
    """
    b = float(b)
    if b <= critical_impact_parameter(M):
        return np.inf

    # f(u) = 2M u^3 - u^2 + 1/b^2
    roots = np.roots([2.0 * M, -1.0, 0.0, 1.0 / (b * b)])
    real = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
    positive = real[real > 0]
    if positive.size < 2:
        return np.inf
    u0 = positive[0]                       # turning point: smallest positive root
    others = [r for r in real if abs(r - u0) > 1e-14]

    def integrand(t):
        u = u0 - t * t
        # sqrt(f) = t * sqrt(-2M (u - other1)(u - other2))
        prod = -2.0 * M * (u - others[0]) * (u - others[1])
        return 2.0 / np.sqrt(prod)

    # `val` is the one-sided sweep, from u = 0 (infinity) in to the turning
    # point.  The full trajectory sweeps twice that, in and back out again.
    val, _ = quad(integrand, 0.0, np.sqrt(u0), limit=200, epsabs=1e-13,
                  epsrel=1e-13)
    return 2.0 * val - np.pi


# ---------------------------------------------------------------------------
# Perihelion precession
# ---------------------------------------------------------------------------

def precession_weak(a_sma, e, M=1.0):
    """Weak-field perihelion advance per orbit, in radians:

        Delta phi = 6 pi G M / (c^2 a (1 - e^2))

    Accurate to relative order M / (a(1-e^2)), so for a test at the 1e-6 level
    you need a semi-major axis of ~1e6 M or a comparison against
    `precession_exact`.
    """
    return 6.0 * np.pi * M / (a_sma * (1.0 - e * e))


def precession_exact(r_peri, r_apo, M=1.0):
    """Exact Schwarzschild periapsis advance per radial period, in radians.

    Same u = 1/r reduction as `deflection_exact`.  For a timelike equatorial
    orbit,

        (du/dphi)^2 = (E^2 - 1)/L^2 + (2M/L^2) u - u^2 + 2M u^3  =: F(u)

    with F(u) = 2M (u - r1)(u - r2)(u - r3) and r1 = 1/r_apo, r2 = 1/r_peri the
    two turning points.  The substitution

        u = (r1 + r2)/2 + ((r2 - r1)/2) sin(psi)

    removes *both* endpoint singularities at once, since
    (u - r1)(r2 - u) = ((r2-r1)/2)^2 cos^2(psi) cancels against du.  What
    remains is a smooth integral over psi in [-pi/2, pi/2].

    Because the two turning points fix two of the three roots and the cubic's
    coefficients fix their sum, the third root follows from Vieta:
    r1 + r2 + r3 = 1/(2M).

    Returns the advance, i.e. (swept angle per radial period) - 2 pi.

    Note the subtraction of 1 inside the integrand.  The swept angle is
    2 * integral(g) and the advance is that minus 2 pi; since the psi interval
    has length pi, 2 * integral(1) = 2 pi exactly, so we may fold the -2 pi
    inside as 2 * integral(g - 1).  This matters: in the weak field g is within
    ~1e-5 of 1, so computing the integral and then subtracting 2 pi would throw
    away five significant digits to cancellation, and for a Mercury-like orbit
    (g - 1 ~ 1e-8) it would throw away eight and leave nothing.  Folding the
    subtraction inside makes the integrand itself the small quantity, and the
    result stays accurate to full precision at any semi-major axis.
    """
    r1, r2 = 1.0 / r_apo, 1.0 / r_peri       # r1 < r2
    r3 = 1.0 / (2.0 * M) - r1 - r2           # Vieta on  2M u^3 - u^2 + ...
    m, d = 0.5 * (r1 + r2), 0.5 * (r2 - r1)

    def integrand(psi):
        u = m + d * np.sin(psi)
        return 1.0 / np.sqrt(2.0 * M * (r3 - u)) - 1.0

    val, _ = quad(integrand, -np.pi / 2, np.pi / 2, limit=200,
                  epsabs=1e-16, epsrel=1e-13)
    return 2.0 * val


def apsides_from_elements(a_sma, e):
    """(r_peri, r_apo) from semi-major axis and eccentricity."""
    return a_sma * (1.0 - e), a_sma * (1.0 + e)


# ---------------------------------------------------------------------------
# Mercury, as a physical sanity check with a number everyone knows
# ---------------------------------------------------------------------------

#: Solar mass in geometric units: GM_sun / c^2, in metres.
GM_SUN_OVER_C2 = 1476.6250385

MERCURY = {
    "a_sma_m": 5.790905e10,      # semi-major axis, metres
    "e": 0.20563,                # eccentricity
    "period_days": 87.9691,
    "observed_gr_precession_arcsec_per_century": 42.98,
}


def mercury_precession_arcsec_per_century():
    """The 43 arcseconds per century, from `precession_weak`.

    Included because it converts an abstract 1e-7 radians per orbit into the
    number that actually convinced people in 1915, and because getting it right
    exercises the unit conversion between geometric and SI units -- an
    easy place to be quietly wrong by a factor of 2.
    """
    a_geo = MERCURY["a_sma_m"] / GM_SUN_OVER_C2       # semi-major axis in units of M
    per_orbit = precession_weak(a_geo, MERCURY["e"], M=1.0)   # radians
    orbits_per_century = 36525.0 / MERCURY["period_days"]
    rad_per_century = per_orbit * orbits_per_century
    return rad_per_century * (180.0 / np.pi) * 3600.0


# ---------------------------------------------------------------------------
# Kerr landmarks (used from stage 2 onward)
# ---------------------------------------------------------------------------

def kerr_photon_orbit_constants(r, a, M=1.0):
    """(xi, eta) for the spherical photon orbit of Boyer-Lindquist radius r.

    Kerr admits photon orbits at constant r for a whole *range* of radii,
    r_ph(prograde) <= r <= r_ph(retrograde) -- not just the two equatorial
    circles.  Each radius carries specific values of xi = Lz/E and
    eta = Q/E^2 (Bardeen 1973):

        xi  = [ M(r^2 - a^2) - r Delta ] / [ a (r - M) ]
        eta = r^3 [ 4 M Delta - r (r - M)^2 ] / [ a^2 (r - M)^2 ]

    obtained by solving R(r) = R'(r) = 0 for the two ratios.  At the
    equatorial endpoints eta = 0 (no polar motion); in between eta > 0 and the
    orbit winds on a sphere, filling an annulus between polar turning points.
    A photon with exactly these constants asymptotically orbits forever; these
    are the orbits whose instability makes the black-hole shadow edge.

    Only the ratios xi, eta are meaningful for photons -- rescaling the affine
    parameter rescales (E, Lz, Q) together -- which is why the shadow of a
    black hole is independent of photon energy.

    Undefined at a = 0 (the formulas are 0/0: with no spin every photon orbit
    sits at r = 3M and xi becomes a free parameter), and singular at r = M
    (reached only in the extremal prograde limit).
    """
    if abs(a) < 1e-12:
        raise ValueError("xi, eta are undefined at a = 0; use r = 3M directly")
    D = r * r - 2.0 * M * r + a * a
    xi = (M * (r * r - a * a) - r * D) / (a * (r - M))
    eta = r**3 * (4.0 * M * D - r * (r - M) ** 2) / (a * a * (r - M) ** 2)
    return xi, eta


def kerr_critical_impact_parameter(a, prograde=True, M=1.0):
    """Capture threshold b_c for equatorial photons, signed.

    This is xi evaluated at the equatorial photon-orbit radius.  Prograde
    (co-rotating) photons return b > 0; retrograde b < 0.  The magnitudes are
    *very* different: for a = M the prograde threshold is 2M while the
    retrograde is 7M -- the hole is three and a half times "bigger" for light
    that fights the spin than for light that surfs it.  This asymmetry is the
    cleanest quantitative signature of frame dragging, and it is what shifts
    the black-hole shadow sideways.

    Reduces to +/- 3 sqrt(3) M at a = 0.
    """
    if abs(a) < 1e-12:
        return (3.0 * np.sqrt(3.0) * M) * (1.0 if prograde else -1.0)
    sign = -1.0 if prograde else 1.0
    r_ph = 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(sign * a / M)))
    xi, _ = kerr_photon_orbit_constants(r_ph, a, M)
    return xi


def kerr_circular_frequencies(r, a, prograde=True, M=1.0):
    """(Omega_phi, Omega_theta, Omega_r) for a circular equatorial orbit.

    Coordinate-time frequencies of the azimuthal motion and of small
    oscillations about the circle (Wilkins 1972; Merloni et al. 1999), with
    the upper sign prograde:

        Omega_phi   = +/- sqrt(M) / (r^{3/2} +/- a sqrt(M))
        Omega_theta = Omega_phi sqrt(1 -/+ 4 a sqrt(M) r^{-3/2} + 3 a^2 r^{-2})
        Omega_r     = Omega_phi sqrt(1 - 6M/r +/- 8 a sqrt(M) r^{-3/2} - 3 a^2 r^{-2})

    In Schwarzschild all three degenerate pairwise consequences hold:
    Omega_theta = Omega_phi exactly (orbital planes are fixed -- no frame
    dragging), and Omega_r < Omega_phi is the periapsis advance.  Kerr splits
    all three, and each splitting is an observable:

      * Omega_phi - Omega_theta  is the Lense-Thirring nodal precession: the
        orbital plane of an inclined orbit is dragged around the spin axis.
      * Omega_phi - Omega_r      is the (relativistic) periapsis advance.
      * Omega_r -> 0             marks the ISCO, where radial oscillations
        cease to be restored.

    Omega_phi is signed (negative for retrograde); the other two are returned
    as positive frequencies.
    """
    s = 1.0 if prograde else -1.0
    sq = np.sqrt(M)
    Om_phi = s * sq / (r**1.5 + s * a * sq)
    x = a * sq / r**1.5
    Om_th = abs(Om_phi) * np.sqrt(1.0 - 4.0 * s * x + 3.0 * a * a / (r * r))
    Om_r = abs(Om_phi) * np.sqrt(
        1.0 - 6.0 * M / r + 8.0 * s * x - 3.0 * a * a / (r * r))
    return Om_phi, Om_th, Om_r


def kerr_shadow_boundary(a, theta_obs=np.pi / 2, n=720, M=1.0):
    """Analytic outline of the Kerr black hole shadow (Bardeen 1973).

    Parametrised by the radius r of the spherical photon orbit that the ray
    asymptotes onto.  The critical impact parameters are

        xi   = [ (r^2 + a^2) - 4 M r (r^2 + a^2)/(r^2 - ... ) ] ... see below
        eta  = r^3 [ 4 M a^2 - r (r - 3M)^2 ] / (a^2 (r - M)^2)

    with the standard closed forms

        xi  = ( M(r^2 - a^2) - r(r^2 - 2Mr + a^2) ) / ( a (r - M) )
        eta = r^3 ( 4 M (r^2 - 2Mr + a^2) - r (r - M)^2 ) / ( a^2 (r - M)^2 )

    and the celestial coordinates seen by a distant observer at inclination
    theta_obs

        alpha = -xi / sin(theta_obs)
        beta  = +/- sqrt( eta + a^2 cos^2(theta_obs) - xi^2 cot^2(theta_obs) )

    At a = 0 this degenerates to the circle of radius 3 sqrt(3) M; the code
    special-cases that since the expressions above divide by a.

    This is the reference curve for the stage-4 ray tracer: the numerically
    traced shadow edge should land on it.
    """
    if abs(a) < 1e-12:
        t = np.linspace(0.0, 2.0 * np.pi, n)
        rc = critical_impact_parameter(M)
        return rc * np.cos(t), rc * np.sin(t)

    # Range of spherical photon orbit radii, from prograde to retrograde.
    r1 = 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-abs(a) / M)))
    r2 = 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(+abs(a) / M)))
    r = np.linspace(min(r1, r2), max(r1, r2), n // 2)

    xi, eta = kerr_photon_orbit_constants(r, a, M)

    st, ct = np.sin(theta_obs), np.cos(theta_obs)
    alpha = -xi / st
    beta2 = eta + a * a * ct * ct - xi * xi * (ct / st) ** 2
    ok = beta2 >= 0
    alpha, beta = alpha[ok], np.sqrt(beta2[ok])

    # Close the curve: upper branch out, lower branch back.
    return (np.concatenate([alpha, alpha[::-1]]),
            np.concatenate([beta, -beta[::-1]]))
