"""Test suite.  Run with `pytest -q` from the repository root.

Organised by what each test actually protects:

  * metric algebra          -- g g^-1 = I, closed forms, complex-step derivatives
  * closed-form landmarks   -- horizons, ergosphere, ISCO, photon orbits
  * conservation            -- E and Lz exact; norm and Q to tolerance
  * physical observables    -- deflection, capture threshold, precession
  * formulation agreement   -- Hamiltonian vs separated Carter equations
  * integrator properties   -- convergence order, reversibility
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrgeo import (
    KerrBL,
    Schwarzschild,
    analytic,
    circular_orbit,
    drift_report,
    hamiltonian,
    measure,
    norm,
    orbit_from_apsides,
    photon_from_impact_parameter,
    separated,
    state_from_constants,
    trace,
)
from kerrgeo.events import escape_event, horizon_event

SPINS = [0.0, 0.3, 0.6, 0.9, 0.998]


# ---------------------------------------------------------------------------
# Metric algebra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a", SPINS)
@pytest.mark.parametrize("r,th", [(3.5, 0.4), (7.0, np.pi / 2), (25.0, 2.5)])
def test_inverse_metric_really_is_the_inverse(a, r, th):
    """g_ab and g^{ab} are computed from independent closed forms, so their
    product being the identity is a genuine check on both."""
    m = KerrBL(a=a)
    x = np.array([0.0, r, th, 0.0])
    prod = m.g(x) @ np.asarray(m.ginv(x), dtype=float)
    assert np.allclose(prod, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("a", [0.0, 0.5, 0.9])
def test_complex_step_matches_finite_difference(a):
    """Complex-step derivatives agree with a central difference to the
    difference's own accuracy (~1e-11), confirming the technique is being
    applied correctly rather than silently returning garbage."""
    m = KerrBL(a=a)
    x = np.array([0.0, 6.2, 1.05, 0.7])
    dg = m.dginv(x)
    h = 1e-5
    for c in (1, 2):
        xp, xm = x.copy(), x.copy()
        xp[c] += h
        xm[c] -= h
        fd = (np.asarray(m.ginv(xp), float) - np.asarray(m.ginv(xm), float)) / (2 * h)
        assert np.abs(fd - dg[c]).max() < 1e-8


@pytest.mark.parametrize("a", SPINS)
def test_cyclic_derivatives_are_identically_zero(a):
    """This is what makes E and Lz exact.  If it ever fails, every
    conservation guarantee in the project is void."""
    m = KerrBL(a=a)
    dg = m.dginv(np.array([0.0, 5.0, 1.2, 0.3]))
    assert np.all(dg[0] == 0.0)
    assert np.all(dg[3] == 0.0)


def test_naked_singularity_is_rejected():
    with pytest.raises(ValueError, match="naked singularity"):
        KerrBL(a=1.5)


# ---------------------------------------------------------------------------
# Closed-form landmarks
# ---------------------------------------------------------------------------

def test_schwarzschild_landmarks():
    m = Schwarzschild()
    assert m.r_plus == pytest.approx(2.0)
    assert m.r_isco() == pytest.approx(6.0)
    assert m.r_photon() == pytest.approx(3.0)
    assert analytic.critical_impact_parameter() == pytest.approx(3 * np.sqrt(3))


def test_extremal_kerr_landmarks():
    """a = M is the sharpest case: horizon, prograde ISCO and prograde photon
    orbit all collapse to r = M in Boyer-Lindquist coordinates."""
    m = KerrBL(a=1.0)
    assert m.r_plus == pytest.approx(1.0)
    assert m.r_isco(prograde=True) == pytest.approx(1.0, abs=1e-9)
    assert m.r_isco(prograde=False) == pytest.approx(9.0)
    assert m.r_photon(prograde=True) == pytest.approx(1.0, abs=1e-9)
    assert m.r_photon(prograde=False) == pytest.approx(4.0)


@pytest.mark.parametrize("a", [0.3, 0.9])
def test_ergosphere_geometry(a):
    """The static limit touches the horizon on the axis and reaches 2M in the
    equatorial plane, independent of spin."""
    m = KerrBL(a=a)
    assert m.r_ergo(0.0) == pytest.approx(m.r_plus)
    assert m.r_ergo(np.pi / 2) == pytest.approx(2.0)
    assert m.r_ergo(1.0) > m.r_plus


@pytest.mark.parametrize("a", [0.0, 0.9])
def test_gtt_changes_sign_at_the_static_limit(a):
    """Inside the ergosphere g_tt > 0, so d/dt is spacelike and no observer can
    remain at fixed (r, theta, phi).  Frame dragging becomes compulsory."""
    m = KerrBL(a=a) if a else KerrBL(a=0.5)
    th = 1.0
    re = m.r_ergo(th)
    assert m.g(np.array([0.0, re * 1.01, th, 0.0]))[0, 0] < 0   # outside: timelike
    assert m.g(np.array([0.0, re * 0.99, th, 0.0]))[0, 0] > 0   # inside: spacelike


def test_kretschmann_finite_at_horizon_but_diverges_at_ring():
    """Evidence that the horizon is only a coordinate singularity while the
    ring singularity is physical."""
    m = KerrBL(a=0.9)
    at_horizon = abs(m.kretschmann(m.r_plus, 1.0))
    assert np.isfinite(at_horizon) and at_horizon < 1e3
    near_ring = abs(m.kretschmann(1e-4, np.pi / 2))
    assert near_ring > 1e20


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a", [0.0, 0.6, 0.9])
def test_energy_and_angular_momentum_are_exact(a):
    """Not "small": bitwise unchanged.  These are protected by the structure of
    the equations, not by the tolerance."""
    m = KerrBL(a=a)
    # A genuinely bound inclined orbit: these constants place r = 10 strictly
    # between two radial turning points, with Theta(pi/3) > 0, for every spin
    # tested.  (Getting this wrong is easy -- reduce the total angular momentum
    # a little and the "orbit" is a plunge, which then hits the horizon and the
    # BL diagnostics blow up rather than merely drifting.)
    y0 = state_from_constants(m, np.array([0.0, 10.0, np.pi / 3, 0.0]),
                              E=0.985, Lz=3.0, Q=6.25, mu=1.0)
    sol = trace(m, y0, 600.0, rtol=1e-12, atol=1e-12)
    d = drift_report(sol.y, m, mu=1.0)
    assert d["E"] == 0.0
    assert d["Lz"] == 0.0


@pytest.mark.parametrize("a", [0.0, 0.9])
def test_carter_constant_and_norm_are_conserved_to_tolerance(a):
    """Q is the honest truncation-error diagnostic: nothing in the code forces
    it to hold, unlike E and Lz."""
    m = KerrBL(a=a)
    # A genuinely bound inclined orbit: these constants place r = 10 strictly
    # between two radial turning points, with Theta(pi/3) > 0, for every spin
    # tested.  (Getting this wrong is easy -- reduce the total angular momentum
    # a little and the "orbit" is a plunge, which then hits the horizon and the
    # BL diagnostics blow up rather than merely drifting.)
    y0 = state_from_constants(m, np.array([0.0, 10.0, np.pi / 3, 0.0]),
                              E=0.985, Lz=3.0, Q=6.25, mu=1.0)
    sol = trace(m, y0, 600.0, rtol=1e-12, atol=1e-12)
    d = drift_report(sol.y, m, mu=1.0)
    assert d["Q"] < 1e-9
    assert d["norm"] < 1e-10


def test_photon_norm_starts_null():
    m = KerrBL(a=0.7)
    y0 = photon_from_impact_parameter(m, r0=500.0, b=7.0)
    assert abs(hamiltonian(y0, m)) < 1e-12


def test_circular_orbit_stays_circular():
    """A circular orbit is a fixed point of the radial motion; any drift in r
    is pure integration error and shows up immediately."""
    m = KerrBL(a=0.5)
    y0, E, Lz = circular_orbit(m, 10.0)
    sol = trace(m, y0, 2000.0, rtol=1e-12, atol=1e-12)
    assert np.ptp(sol.y[1]) < 1e-8


def test_no_circular_orbit_inside_photon_radius():
    m = KerrBL(a=0.5)
    with pytest.raises(ValueError, match="No circular timelike orbit"):
        circular_orbit(m, m.r_photon(prograde=True) * 0.9)


def test_forbidden_initial_conditions_are_rejected():
    """Asking for a radius outside the allowed region should raise a clear
    physical error, not silently produce a NaN trajectory."""
    m = Schwarzschild()
    with pytest.raises(ValueError, match=r"R\(r\)"):
        state_from_constants(m, np.array([0.0, 8.0, np.pi / 2, 0.0]),
                             E=0.5, Lz=6.0, Q=0.0, mu=1.0)


# ---------------------------------------------------------------------------
# Physical observables against independent analytic references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", [5.5, 6.0, 10.0, 50.0, 200.0])
def test_deflection_matches_exact_quadrature(b):
    """Spans the strong field (b = 5.5, deflection 2.55 rad) to the weak field
    (b = 200, deflection 0.02 rad)."""
    m = Schwarzschild()
    got = measure.measure_deflection(m, b, r0=1e5)
    want = analytic.deflection_exact(b)
    assert got == pytest.approx(want, rel=1e-8)


def test_deflection_reproduces_the_second_order_coefficient():
    """4M/b alone is recovered by any roughly-correct calculation.  The
    15 pi/4 (M/b)^2 term is sensitive to the actual spatial curvature, so
    matching it is the sharper statement."""
    m = Schwarzschild()
    b = 400.0
    got = measure.measure_deflection(m, b, r0=1e6)
    # Strip the Einstein term and the known third-order term; what is left must
    # be the second-order coefficient.  (Keeping the third-order term in would
    # leave a 0.9% residual -- it is not negligible at this b.)
    second_order = got - 4.0 / b - (128.0 / 3.0) / b**3
    assert second_order == pytest.approx(15 * np.pi / 4 / b**2, rel=1e-3)


def test_photon_capture_threshold_is_3root3():
    m = Schwarzschild()
    bc = measure.capture_threshold(m, 5.0, 5.5, r0=1e4, tol=1e-8)
    assert bc == pytest.approx(3 * np.sqrt(3), abs=1e-7)


@pytest.mark.parametrize("rp,ra", [(10.0, 20.0), (30.0, 60.0), (100.0, 200.0)])
def test_precession_matches_exact_quadrature(rp, ra):
    m = Schwarzschild()
    got = measure.measure_precession(m, rp, ra)
    want = analytic.precession_exact(rp, ra)
    assert got == pytest.approx(want, rel=1e-9)


def test_precession_approaches_the_weak_field_formula():
    """6 pi M / (a(1-e^2)) is the leading term; the exact result should
    converge onto it as the orbit weakens."""
    prev = None
    for a_sma in (1e3, 1e4, 1e5):
        rp, ra = analytic.apsides_from_elements(a_sma, 0.2)
        ratio = analytic.precession_exact(rp, ra) / analytic.precession_weak(a_sma, 0.2)
        assert ratio == pytest.approx(1.0, abs=5e-3)
        if prev is not None:
            assert abs(ratio - 1) < abs(prev - 1)
        prev = ratio


def test_mercury_precession_is_43_arcsec_per_century():
    assert analytic.mercury_precession_arcsec_per_century() == pytest.approx(42.98, abs=0.01)


# ---------------------------------------------------------------------------
# The two formulations must agree
# ---------------------------------------------------------------------------

def test_hamiltonian_and_separated_forms_agree():
    """The strongest single test in the suite: two derivations sharing no code
    -- one integrating Hamilton's equations from the metric, one integrating
    Carter's separated equations from (E, Lz, Q) -- tracing the same generic
    inclined Kerr orbit."""
    from scipy.interpolate import CubicSpline

    m = KerrBL(a=0.9)
    E, Lz, Q = 0.95, 2.8, 3.0
    x0 = np.array([0.0, 12.0, np.pi / 3, 0.0])
    y0 = state_from_constants(m, x0, E, Lz, Q, mu=1.0, sign_r=-1, sign_theta=1)

    solH = trace(m, y0, 400.0, rtol=1e-13, atol=1e-13)
    sep = separated.trace_separated(m, x0, E, Lz, Q, mu=1.0,
                                    lam_max=25.0, n_out=20000)

    tau = np.linspace(1.0, 390.0, 400)
    for idx, key in ((1, "r"), (2, "theta"), (3, "phi"), (0, "t")):
        a_ = CubicSpline(solH.t, solH.y[idx])(tau)
        b_ = CubicSpline(sep["tau"], sep[key])(tau)
        assert np.abs(a_ - b_).max() < 1e-4, f"formulations disagree on {key}"


def test_constants_round_trip_through_the_state_vector():
    m = KerrBL(a=0.8)
    E, Lz, Q = 0.94, -2.2, 5.0
    y0 = state_from_constants(m, np.array([0.0, 9.0, 1.1, 0.0]), E, Lz, Q, mu=1.0)
    gotE, gotL, gotQ = separated.constants_from_state(y0, m, mu=1.0)
    assert (gotE, gotL) == pytest.approx((E, Lz))
    assert gotQ == pytest.approx(Q)


# ---------------------------------------------------------------------------
# Integrator properties
# ---------------------------------------------------------------------------

def test_rk4_converges_at_fourth_order():
    """Halving the step should cut the error by ~16.  Confirms the integrator
    is actually RK4 and not accidentally something lower order."""
    m = Schwarzschild()
    y0, _, _ = orbit_from_apsides(m, 12.0, 24.0)
    lam = 400.0
    ref = trace(m, y0, lam, rtol=1e-13, atol=1e-13).y[:, -1]

    # Step sizes must be coarse enough that truncation error dominates.  At
    # n = 2000 (h = 0.2) RK4 is already down at ~4e-12, on the round-off floor,
    # where the ratio between successive errors is meaningless.
    errs = []
    for n in (100, 200, 400):
        got = trace(m, y0, lam, method="RK4", n_steps=n).y[:, -1]
        errs.append(np.abs(got[1:4] - ref[1:4]).max())
    for lo, hi in zip(errs[1:], errs[:-1]):
        assert hi / lo > 10.0   # 16 in theory; 10 leaves room for round-off


def test_symplectic_integrator_conserves_the_norm_by_construction():
    """Gauss-Legendre conserves quadratic first integrals exactly, and
    g^{ab} p_a p_b is quadratic in p.  Worth knowing when reading diagnostics:
    it makes the norm look flattering relative to the trajectory error."""
    m = Schwarzschild()
    y0, _, _ = orbit_from_apsides(m, 12.0, 24.0)
    sol = trace(m, y0, 400.0, method="GL2", n_steps=4000)
    assert drift_report(sol.y, m, mu=1.0)["norm"] < 1e-13


@pytest.mark.parametrize("method", ["DOP853", "GL2"])
def test_time_reversibility(method):
    """Catches errors the invariants cannot see: a geodesic can sit at
    completely the wrong place on the right orbit and still report perfect
    E, Lz and Q."""
    m = KerrBL(a=0.6)
    y0 = photon_from_impact_parameter(m, r0=80.0, b=6.5)
    kw = dict(n_steps=8000) if method == "GL2" else dict(rtol=1e-13, atol=1e-13)
    assert measure.reversibility_error(m, y0, 120.0, method=method, **kw) < 1e-7


def test_horizon_event_terminates_a_captured_photon():
    m = KerrBL(a=0.9)
    y0 = photon_from_impact_parameter(m, r0=200.0, b=1.0)
    sol = trace(m, y0, 5000.0,
                events=[horizon_event(m), escape_event(400.0)])
    assert sol.status == "event"
    assert sol.r[-1] == pytest.approx(m.r_plus, rel=1e-4)


def test_deflection_is_insensitive_to_the_horizon_cutoff():
    """The epsilon offset used to avoid 1/Delta blowing up must not
    contaminate physical answers."""
    m = Schwarzschild()
    vals = []
    for eps in (1e-3, 1e-5, 1e-7):
        y0 = photon_from_impact_parameter(m, r0=1e4, b=6.0)
        sol = trace(m, y0, 1e5,
                    events=[horizon_event(m, eps=eps), escape_event(1e4)])
        vals.append(sol.y[3, -1])
    assert np.ptp(vals) < 1e-9
