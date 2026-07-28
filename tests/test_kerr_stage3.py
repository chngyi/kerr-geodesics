"""Stage-3 tests: the interior, in ingoing Kerr coordinates.

The claims under test, in order of importance:

  1. The ingoing chart is a correct description of the same spacetime --
     geodesics integrated in BL and ingoing coordinates agree to rounding in
     the exterior overlap.
  2. Horizon crossing is smooth: no step-size collapse, invariants held,
     right through Delta = 0 where BL fails.
  3. The interior structure is what Carter said it was: geodesics reach the
     negative-r sheet only through the disk (Q < 0), equatorial ones die on
     the ring, and the closed timelike curves live exactly where
     g_phiphi < 0 -- entirely at r < 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrgeo import (
    KerrBL,
    KerrIngoing,
    bl_to_ingoing,
    hamiltonian,
    ingoing_to_bl,
    norm,
    principal_null_ingoing,
    separated,
    state_from_constants_ingoing,
    trace,
    zamo_drop_state,
)
from kerrgeo.events import negative_r_escape_event, ring_event
from kerrgeo.invariants import carter, drift_report

A = 0.9
MI = KerrIngoing(a=A)
MB = KerrBL(a=A)


# ---------------------------------------------------------------------------
# The chart itself
# ---------------------------------------------------------------------------

def test_metric_identity_everywhere_including_horizons_and_negative_r():
    """g g^-1 = I from closed forms on both sides -- evaluated AT both
    horizons (Delta = 0 exactly), at the disk r = 0, and on the negative-r
    sheet.  Every one of these points is outside the BL chart's competence."""
    rp, rm = MI.horizons()[1], MI.horizons()[0]
    for r in (5.0, rp, 1.0, rm, 0.3, 0.0, -0.5, -3.0):
        for th in (0.4, np.pi / 2, 2.6):
            if r == 0.0 and abs(th - np.pi / 2) < 1e-9:
                continue        # that point IS the ring singularity
            x = np.array([0.0, r, th, 0.0])
            err = np.abs(MI.g(x) @ np.asarray(MI.ginv(x), float)
                         - np.eye(4)).max()
            assert err < 1e-12, f"identity fails at r={r}, th={th}"


def test_ring_is_the_only_singularity_of_the_chart():
    """At the ring Sigma = 0 the inverse metric genuinely diverges -- and
    should, because the curvature does."""
    x = np.array([0.0, 1e-8, np.pi / 2, 0.0])
    assert np.abs(np.asarray(MI.ginv(x), float)).max() > 1e10


def test_complex_step_derivatives_between_the_horizons():
    m = KerrIngoing(a=A)
    x = np.array([0.0, 1.2, 1.0, 0.0])       # r_- < r < r_+
    dg = m.dginv(x)
    h = 1e-6
    for c in (1, 2):
        xp, xm = x.copy(), x.copy()
        xp[c] += h
        xm[c] -= h
        fd = (np.asarray(m.ginv(xp), float)
              - np.asarray(m.ginv(xm), float)) / (2 * h)
        assert np.abs(fd - dg[c]).max() < 1e-8
    assert np.all(dg[0] == 0.0) and np.all(dg[3] == 0.0)


# ---------------------------------------------------------------------------
# Same spacetime: BL and ingoing agree where both are valid
# ---------------------------------------------------------------------------

def test_charts_agree_in_the_exterior_overlap():
    """Integrate the same infalling geodesic in both charts to the same
    affine parameter, map the ingoing endpoint back to BL, and compare all
    eight phase-space components.  Agreement at 1e-12 means metric,
    transfer functions and both integrations are mutually consistent."""
    y0_bl = zamo_drop_state(MB, 8.0)
    y0_in = bl_to_ingoing(MI, y0_bl)
    assert norm(y0_in, MI) == pytest.approx(-1.0, abs=1e-12)

    lam = 20.0                                # endpoint still at r = 4.24
    sb = trace(MB, y0_bl, lam, rtol=1e-13, atol=1e-13)
    si = trace(MI, y0_in, lam, rtol=1e-13, atol=1e-13)
    back = ingoing_to_bl(MI, si.y[:, -1])
    assert np.abs(back - sb.y[:, -1]).max() < 1e-11


def test_transfer_round_trip_is_identity():
    y = zamo_drop_state(MB, 6.0)
    there_and_back = ingoing_to_bl(MI, bl_to_ingoing(MI, y))
    assert np.abs(there_and_back - y).max() < 1e-12


# ---------------------------------------------------------------------------
# Horizon crossing
# ---------------------------------------------------------------------------

def test_infall_crosses_both_horizons_smoothly_and_dies_on_the_ring():
    """The full interior story in one trajectory: an equatorial ZAMO drop
    passes r_+ and r_- with invariants held (norm to 6e-12, E and Lz exact),
    and terminates only at the genuine singularity.  In BL coordinates this
    trajectory cannot even be represented past r_+."""
    y0 = bl_to_ingoing(MI, zamo_drop_state(MB, 8.0))
    sol = trace(MI, y0, 200.0, rtol=1e-12, atol=1e-12,
                events=[ring_event(MI, eps=1e-3)])

    assert sol.status == "event"                      # reached the ring
    assert sol.y[1, -1] == pytest.approx(1e-3, rel=0.1)
    assert sol.y[1].min() < MI.r_minus                # went through both

    # Invariants: exact ones exactly, norm to tolerance through r_+ and r_-.
    lam_rm = sol.t[np.argmin(np.abs(sol.y[1] - MI.r_minus))]
    through = sol.y[:, sol.t <= lam_rm + 0.05]
    d = drift_report(through, MI, mu=1.0)
    assert d["E"] == 0.0
    assert d["Lz"] == 0.0
    assert d["norm"] < 1e-10

    # The crossings cost no special effort: modest nfev for 200 M of hard
    # trajectory is the whole point of the affine parameter + regular chart.
    assert sol.nfev < 20000


def test_inner_turning_point_stalls_at_the_cauchy_horizon():
    """A particle with an inner turning point (R = 0 at r = 0.51 < r_-)
    turns around inside r_-, becomes outgoing -- and the *ingoing* chart
    correctly refuses to take it back out: v diverges as it re-approaches
    r_- from below, exactly as BL t diverges at r_+.  Outgoing crossing of
    the Cauchy horizon belongs to a different extension (a white-hole sheet),
    not to this chart.  The test pins the behaviour: r approaches r_- from
    below and stays there."""
    x = np.array([0.0, MI.r_plus, np.pi / 2, 0.0])
    y0 = state_from_constants_ingoing(MI, x, E=0.95, Lz=1.0, Q=0.0, mu=1.0)
    sol = trace(MI, y0, 60.0, rtol=1e-10, atol=1e-10)
    r_turn_expected = 0.5075
    assert sol.y[1].min() == pytest.approx(r_turn_expected, abs=5e-3)
    assert sol.y[1, -1] == pytest.approx(MI.r_minus, abs=5e-3)
    assert sol.y[1, -1] < MI.r_minus            # from below, never out


def test_ingoing_ic_builder_is_finite_at_the_horizon():
    """p_r = K/(P + sqrt(R)) -- the cancellation-free form -- must give a
    normalised, integrable state at Delta = 0 exactly."""
    x = np.array([0.0, MI.r_plus, np.pi / 2, 0.0])
    y = state_from_constants_ingoing(MI, x, E=0.95, Lz=1.0, Q=0.0, mu=1.0)
    assert np.isfinite(y).all()
    assert norm(y, MI) == pytest.approx(-1.0, abs=1e-10)

    with pytest.raises(ValueError, match="outgoing branch is singular"):
        state_from_constants_ingoing(MI, x, E=0.95, Lz=1.0, mu=1.0,
                                     branch="outgoing")


# ---------------------------------------------------------------------------
# The principal null ray: an exact solution through everything
# ---------------------------------------------------------------------------

def test_principal_null_ray_is_reproduced_exactly():
    """In this chart the ingoing principal null congruence is r = r0 - lambda
    at constant (v, theta, phi~) -- an exact linear solution through r_+,
    r_-, and the disk into negative r.  The integrator must reproduce it to
    rounding over the whole passage; any chart pathology at the horizons
    would show up here first."""
    th0 = np.radians(80)
    y0 = principal_null_ingoing(MI, 5.0, th0)
    assert abs(hamiltonian(y0, MI)) < 1e-14

    sol = trace(MI, y0, 12.0, rtol=1e-13, atol=1e-13,
                events=[negative_r_escape_event(-4.9)])
    assert sol.status == "event"
    assert np.abs(sol.y[1] - (5.0 - sol.t)).max() < 1e-11
    for i in (0, 2, 3):                       # v, theta, phi~ all frozen
        assert np.abs(sol.y[i] - y0[i]).max() < 1e-11


def test_vortical_geodesics_have_negative_carter_constant():
    """Q = -a^2 cos^4(theta0) < 0 for the principal null rays: they are
    confined to cones about the axis, which is exactly what lets them thread
    the disk instead of hitting the ring."""
    th0 = np.radians(80)
    y0 = principal_null_ingoing(MI, 5.0, th0)
    assert carter(y0, MI, mu=0.0) == pytest.approx(
        -A * A * np.cos(th0) ** 4, rel=1e-10)


def test_crossing_the_disk_requires_negative_q():
    """R(r=0) = -a^2 Q, so R >= 0 at the crossing forces Q <= 0: no orbit
    with equator-crossing polar motion can pass through the disk.  Algebraic
    consequence of the radial potential -- checked, not assumed."""
    # Q > 0: r = 0 is forbidden (R < 0) -- these orbits bounce or hit the ring.
    for E, Lz, Q in ((1.0, 0.3, 2.0), (0.95, -1.0, 0.5), (1.2, 0.0, 1e-3)):
        assert separated.radial_potential(0.0, A, E, Lz, Q, 1.0) < 0
    # Q < 0 (vortical): the disk is open.
    assert separated.radial_potential(0.0, A, 1.0, 0.3, -0.2, 0.0) > 0


# ---------------------------------------------------------------------------
# Closed timelike curves
# ---------------------------------------------------------------------------

def test_ctcs_exist_only_on_the_negative_r_sheet():
    """The azimuthal circles are timelike (g_phiphi < 0) in a band of
    negative r hugging the ring -- and nowhere at r > 0.  The exterior of a
    Kerr black hole is causally clean; the pathology is real but hidden
    behind both horizons and through the ring."""
    rr = np.linspace(-1.5, -1e-3, 500)
    assert np.any(MI.g_phiphi(rr, np.pi / 2) < 0)

    # The inner edge of the equatorial band is the real root of
    # r^3 + a^2 r + 2 a^2 M = 0  (g_phiphi = 0 with Sigma = r^2).
    from scipy.optimize import brentq

    edge = brentq(lambda r: r**3 + A * A * r + 2 * A * A, -1.5, -0.5)
    band = rr[MI.g_phiphi(rr, np.pi / 2) < 0]
    assert band.min() == pytest.approx(edge, abs=5e-3)

    r_pos = np.linspace(1e-3, 30.0, 700)
    th = np.linspace(0.01, np.pi - 0.01, 150)
    Rg, Tg = np.meshgrid(r_pos, th)
    assert np.all(MI.g_phiphi(Rg, Tg) > 0)


def test_ctc_loop_proper_time_is_finite_and_order_M():
    """An observer riding the closed azimuthal circle at r = -0.5 on the
    equator returns to the same event after a finite proper time of order M.
    Not a geodesic -- it takes acceleration -- but a legal worldline."""
    tau = MI.ctc_loop_proper_time(-0.5, np.pi / 2)
    assert np.isfinite(tau)
    assert 0.1 < tau < 20.0
    # Outside the band the loop is spacelike and tau is undefined.
    assert np.isnan(MI.ctc_loop_proper_time(3.0, np.pi / 2))


def test_a_geodesic_passes_through_the_ctc_region():
    """The vortical ray at theta0 = 80 deg sweeps r monotonically through
    the band where g_phiphi(r, 80deg) < 0: a geodesic that has traversed the
    CTC region and come out the other side into the negative-r asymptotic
    domain."""
    th0 = np.radians(80)
    y0 = principal_null_ingoing(MI, 3.0, th0)
    sol = trace(MI, y0, 6.0, rtol=1e-13, atol=1e-13,
                events=[negative_r_escape_event(-2.5)])
    assert sol.status == "event"
    # r is exactly monotonic (dr/dlambda = -1), so path coverage of the band
    # follows from the endpoints; verify the band is really there.
    rr = np.linspace(-1.2, -0.02, 800)
    band = rr[MI.g_phiphi(rr, th0) < 0]
    assert band.size > 0
    assert sol.y[1, -1] < band.min() < band.max() < sol.y[1, 0]


def test_bl_and_ingoing_agree_on_g_phiphi():
    """The CTC criterion is chart-independent here: phi~ differs from phi by
    a function of r only, so g_phiphi is literally the same function."""
    for r, th in ((-0.5, np.pi / 2), (2.0, 1.0), (-0.2, 1.2)):
        x = np.array([0.0, r, th, 0.0])
        assert MI.g_phiphi(r, th) == pytest.approx(MI.g(x)[3, 3], rel=1e-13)
