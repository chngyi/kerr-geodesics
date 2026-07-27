"""Stage-2 tests: Kerr physics against closed forms.

Everything here checks an *integrated* result against an independent analytic
expression -- Bardeen photon-orbit constants, Wilkins orbital frequencies, the
frame-drag rate, the phi-turnaround radius.  The stage-1 suite established
that the machinery conserves what it should; this one establishes that the
physics coming out of it is Kerr's.
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrgeo import (
    KerrBL,
    Schwarzschild,
    analytic,
    circular_orbit,
    measure,
    norm,
    rhs,
    separated,
    spherical_orbit,
    spherical_photon_orbit,
    state_from_constants,
    trace,
    zamo_drop_state,
)
from kerrgeo.events import horizon_event


# ---------------------------------------------------------------------------
# Photon capture thresholds: the cleanest frame-dragging observable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prograde", [True, False])
def test_capture_threshold_matches_bardeen(prograde):
    """Bisection on 'does this photon reach the horizon' against xi evaluated
    at the equatorial photon-orbit radius.  At a = 0.9 the thresholds are
    2.84 M prograde vs 6.83 M retrograde -- the spin asymmetry in one pair of
    numbers."""
    a = 0.9
    m = KerrBL(a=a)
    bc_analytic = analytic.kerr_critical_impact_parameter(a, prograde)
    lo, hi = abs(bc_analytic) * 0.9, abs(bc_analytic) * 1.1
    bc = measure.capture_threshold(m, lo, hi, r0=500.0, tol=1e-6,
                                   prograde=prograde)
    assert bc == pytest.approx(abs(bc_analytic), abs=1e-5)


def test_kerr_critical_impact_parameters_reduce_to_schwarzschild():
    assert analytic.kerr_critical_impact_parameter(0.0, True) == pytest.approx(
        3 * np.sqrt(3))
    assert analytic.kerr_critical_impact_parameter(0.0, False) == pytest.approx(
        -3 * np.sqrt(3))
    # Known extremal limits: 2M prograde, -7M retrograde.
    assert analytic.kerr_critical_impact_parameter(1.0 - 1e-9, True) == \
        pytest.approx(2.0, abs=1e-3)
    assert analytic.kerr_critical_impact_parameter(1.0, False) == \
        pytest.approx(-7.0, abs=1e-9)


def test_deflection_asymmetry_appears_with_spin_and_only_with_spin():
    """Same |b| on both sides of an a = 0.9 hole: the retrograde ray bends
    ~3x more.  At a = 0 the two must agree exactly -- reflection symmetry is
    a property Schwarzschild has and Kerr does not."""
    mk = KerrBL(a=0.9)
    d_pro = measure.measure_deflection(mk, +7.0, r0=1e5)
    d_ret = measure.measure_deflection(mk, -7.0, r0=1e5)
    assert d_ret > 2.5 * d_pro

    ms = Schwarzschild()
    d1 = measure.measure_deflection(ms, +7.0, r0=1e5)
    d2 = measure.measure_deflection(ms, -7.0, r0=1e5)
    assert d1 == pytest.approx(d2, rel=1e-10)


# ---------------------------------------------------------------------------
# Frame dragging, directly
# ---------------------------------------------------------------------------

def test_zamo_corotates_at_exactly_omega():
    """A particle with Lz = 0 -- zero angular momentum, no torque -- orbits
    at dphi/dt = omega(r).  The trajectory value and the closed form come from
    different expressions (g^{t phi}/g^{tt} vs -g_tphi/g_phiphi), so this
    checks the metric algebra as well as Lz staying pinned at 0."""
    m = KerrBL(a=0.9)
    y0 = zamo_drop_state(m, 8.0)
    assert norm(y0, m) == pytest.approx(-1.0, abs=1e-12)

    sol = trace(m, y0, 200.0, rtol=1e-12, atol=1e-12,
                events=[horizon_event(m)])
    assert sol.status == "event"          # it falls in
    worst = 0.0
    for i in range(0, sol.y.shape[1], max(1, sol.y.shape[1] // 60)):
        d = rhs(0.0, sol.y[:, i], m)
        worst = max(worst, abs(d[3] / d[0] - m.omega(sol.y[1, i])))
    assert worst < 1e-12

    # And at the horizon everything corotates at Omega_H = a / (2 M r_+).
    d_end = rhs(0.0, sol.y[:, -1], m)
    assert d_end[3] / d_end[0] == pytest.approx(m.Omega_H, rel=1e-4)


def test_omega_falls_off_as_gravitomagnetic_dipole():
    """omega r^3 -> 2 a M far away: frame dragging is the gravitational
    analogue of a magnetic dipole field."""
    m = KerrBL(a=0.9)
    assert m.omega(1000.0) * 1000.0**3 == pytest.approx(1.8, rel=1e-5)


@pytest.mark.parametrize("b", [-2.0, -3.0, -5.0])
def test_retrograde_photon_phi_reversal_radius(b):
    """A captured retrograde photon's azimuthal motion reverses at exactly
    r = 2M (1 + a/|b|) -- outside the static limit.  Frame dragging beats the
    photon's own angular momentum before the ergosphere is even reached."""
    m = KerrBL(a=0.9)
    r_flip = measure.measure_phi_turnaround(m, b)
    assert r_flip == pytest.approx(2.0 * (1.0 + 0.9 / abs(b)), rel=1e-9)


def test_negative_energy_orbits_exist_only_inside_ergosphere():
    """The Penrose process in potential form: for a = 0.9 there are timelike
    geodesics with E < 0 (energy at infinity!), but their allowed region lies
    strictly inside the static limit -- R(r) < 0 everywhere outside.  A
    negative-energy fragment is causally committed to the hole."""
    a, E, Lz = 0.9, -0.1, -3.0
    m = KerrBL(a=a)

    assert separated.radial_potential(1.6, a, E, Lz, 0.0, 1.0, 1.0) > 0
    r_out = np.linspace(m.r_ergo(np.pi / 2), 50.0, 400)
    R_out = separated.radial_potential(r_out, a, E, Lz, 0.0, 1.0, 1.0)
    assert np.all(R_out < 0)

    # The state is a genuine timelike geodesic, and it is captured.
    y0 = state_from_constants(m, np.array([0.0, 1.6, np.pi / 2, 0.0]),
                              E=E, Lz=Lz, Q=0.0, mu=1.0, sign_r=-1)
    assert norm(y0, m) == pytest.approx(-1.0, abs=1e-10)
    sol = trace(m, y0, 50.0, rtol=1e-12, atol=1e-12, events=[horizon_event(m)])
    assert sol.status == "event"
    assert sol.y[1].max() <= 1.6 + 1e-9   # never leaves the ergosphere


# ---------------------------------------------------------------------------
# Orbital frequencies and the two precessions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prograde", [True, False])
def test_orbital_frequency_is_kepler_with_a_spin_correction(prograde):
    """Omega_phi = sqrt(M)/(r^{3/2} +/- a sqrt(M)), measured by timing one
    integrated revolution in coordinate time."""
    m = KerrBL(a=0.9)
    got = measure.measure_orbital_frequency(m, 10.0, prograde)
    want = analytic.kerr_circular_frequencies(10.0, 0.9, prograde)[0]
    assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("prograde", [True, False])
def test_nodal_precession_matches_wilkins_frequencies(prograde):
    """Lense-Thirring: the plane of an inclined orbit is dragged around the
    spin axis.  Measured from phi between successive ascending nodes of an
    exactly spherical orbit; the closed-form ratio is the i -> 0 limit, so
    agreement is O(Q/Lz^2) ~ 1e-7 at Q = 1e-6."""
    m = KerrBL(a=0.9)
    Om_phi, Om_th, _ = analytic.kerr_circular_frequencies(12.0, 0.9, prograde)
    ratio, adv = measure.measure_nodal_precession(m, 12.0, Q=1e-6,
                                                  prograde=prograde)
    assert ratio == pytest.approx(Om_phi / Om_th, rel=1e-6)
    if prograde:
        assert adv > 0        # nodes dragged forward, with the spin


def test_nodal_precession_vanishes_without_spin():
    """Control: in Schwarzschild orbital planes are fixed, so the ratio is
    exactly 1 and the advance is zero to integration accuracy."""
    ratio, adv = measure.measure_nodal_precession(Schwarzschild(), 12.0, Q=1e-6)
    assert abs(ratio - 1.0) < 1e-8


def test_kerr_periapsis_advance_near_circular():
    """Equatorial periapsis advance vs 2 pi (Omega_phi/Omega_r - 1), on a
    nearly circular orbit (formula exact in the e -> 0 limit; O(e^2) here)."""
    m = KerrBL(a=0.9)
    Om_phi, _, Om_r = analytic.kerr_circular_frequencies(10.0, 0.9, True)
    want = 2.0 * np.pi * (abs(Om_phi) / Om_r - 1.0)
    got = measure.measure_precession(m, 9.9, 10.1)
    assert got == pytest.approx(want, rel=2e-3)


def test_kerr_frequencies_reduce_to_schwarzschild():
    """At a = 0: Omega_theta = Omega_phi exactly (no nodal precession) and
    Omega_r vanishes at the ISCO."""
    Om_phi, Om_th, Om_r = analytic.kerr_circular_frequencies(10.0, 0.0, True)
    assert Om_th == pytest.approx(Om_phi)
    assert analytic.kerr_circular_frequencies(6.0, 0.0, True)[2] == \
        pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Spherical photon orbits
# ---------------------------------------------------------------------------

def test_photon_orbit_constants_satisfy_their_defining_identities():
    """(xi, eta) are defined by R(r) = R'(r) = 0.  Check both to machine
    precision across the allowed radius range -- pure algebra, no
    integration."""
    a = 0.9
    m = KerrBL(a=a)
    for r in np.linspace(m.r_photon(True) * 1.01, m.r_photon(False) * 0.99, 7):
        xi, eta = analytic.kerr_photon_orbit_constants(r, a)
        R = separated.radial_potential(r, a, 1.0, xi, eta, mu=0.0)
        dR = separated.radial_potential_deriv(r, a, 1.0, xi, eta, mu=0.0)
        assert abs(R) < 1e-10
        assert abs(dR) < 1e-10


def test_spherical_photon_orbit_holds_r_then_peels_off_exponentially():
    """The orbit is a genuine solution (r constant to ~1e-9 over 20 M) but an
    *unstable* one: rounding-level perturbations e-fold until the photon
    departs.  Both halves are physics -- the instability is what makes these
    orbits the shadow edge."""
    m = KerrBL(a=0.9)
    y0 = spherical_photon_orbit(m, 2.6)
    sol = trace(m, y0, 60.0, rtol=1e-13, atol=1e-13)
    drift = np.abs(sol.y[1] - 2.6)
    assert drift[sol.t <= 20.0].max() < 1e-8      # holds the sphere
    assert drift.max() > 1e-3                     # ...but not forever


def test_spherical_photon_orbit_rejects_bad_radii():
    m = KerrBL(a=0.9)
    with pytest.raises(ValueError, match="allowed range"):
        spherical_photon_orbit(m, 5.0)


def test_spherical_timelike_orbit_is_exactly_spherical():
    """The inclined-circle builder solves R = R' = 0 with Q > 0; r must stay
    put to integration accuracy over many polar periods."""
    m = KerrBL(a=0.9)
    y0, E, Lz = spherical_orbit(m, 12.0, Q=0.5)
    sol = trace(m, y0, 2000.0, rtol=1e-12, atol=1e-12)
    assert np.abs(sol.y[1] - 12.0).max() < 1e-7
    # It genuinely leaves the plane, exactly as far as the polar potential
    # allows: the maximum excursion is the root of Theta(theta) = 0.  (The
    # flat-space inclination formula cos i = Lz/sqrt(Lz^2+Q) is off at the
    # 1e-3 level here -- the a^2(1-E^2)cos^2(theta) term in Theta matters.)
    from scipy.optimize import brentq

    theta_turn = brentq(
        lambda th: separated.polar_potential(th, 0.9, E, Lz, 0.5, 1.0),
        1e-3, np.pi / 2 - 1e-6)
    i_exact = np.pi / 2 - theta_turn
    theta_excursion = np.abs(sol.y[2] - np.pi / 2).max()
    # Discrete solver samples never land exactly on a turning point, so the
    # sampled max can only *undershoot* the true excursion (by ~1e-5 here).
    assert theta_excursion <= i_exact + 1e-9
    assert theta_excursion == pytest.approx(i_exact, rel=1e-4)


# ---------------------------------------------------------------------------
# Energetics
# ---------------------------------------------------------------------------

def test_isco_binding_energy_ladder():
    """1 - E at the prograde ISCO: 5.72% for Schwarzschild, 32.1% at
    a = 0.998 (Thorne's spin-equilibrium limit), climbing toward
    1 - 1/sqrt(3) = 42.3% at extremality.  This is why accretion onto a
    spinning hole is the most efficient energy source known."""
    _, E0, _ = circular_orbit(Schwarzschild(), 6.0)
    assert 1.0 - E0 == pytest.approx(1.0 - np.sqrt(8.0 / 9.0), rel=1e-12)

    m = KerrBL(a=0.998)
    _, E1, _ = circular_orbit(m, m.r_isco(True), True)
    assert 1.0 - E1 == pytest.approx(0.3210, abs=5e-4)
