"""Stage-4 tests: the backwards ray tracer.

Division of labour under test: the vectorised batch tracer must agree with
the trusted per-ray integrator away from the critical boundary (imaging), and
the boundary itself is measured with the trusted integrator and compared to
Bardeen's analytic shadow curve (metrology).
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrgeo import KerrBL, analytic, circular_orbit, rhs, trace
from kerrgeo import render as R
from kerrgeo.events import escape_event, horizon_event

A = 0.9


# ---------------------------------------------------------------------------
# The batch integrator must be the same physics
# ---------------------------------------------------------------------------

def test_batch_rhs_matches_the_single_ray_rhs():
    """`_batch_rhs` restates the BL metric as array formulas -- duplicated
    physics, which is a bug farm unless pinned.  Component-by-component
    agreement with `kerrgeo.rhs` on random phase-space points keeps the two
    implementations from drifting apart."""
    rng = np.random.default_rng(7)
    m = KerrBL(a=A)
    for _ in range(30):
        y = np.array([0.0, rng.uniform(2.2, 40.0), rng.uniform(0.3, 2.8),
                      rng.uniform(0.0, 6.0), -1.0, rng.uniform(-0.5, 0.5),
                      rng.uniform(-3.0, 3.0), rng.uniform(-6.0, 6.0)])
        single = rhs(0.0, y, m)
        batch = R._batch_rhs(y.reshape(8, 1), A, 1.0)[:, 0]
        assert np.abs(single - batch).max() < 1e-13


def test_batch_tracer_agrees_with_adaptive_integrator_off_the_edge():
    """For pixels comfortably inside/outside the shadow (10% off the
    boundary) the imaging tracer and DOP853 must classify identically.
    (On the hairline the batch RK4 is *expected* to blur -- the photon-shell
    instability e-folds its truncation error -- which is exactly why
    metrology uses the adaptive integrator.)"""
    m = KerrBL(a=A)
    bc_pro = analytic.kerr_critical_impact_parameter(A, True)
    bc_ret = analytic.kerr_critical_impact_parameter(A, False)
    pixels = [(-bc_pro * 0.9, 0.0, True), (-bc_pro * 1.1, 0.0, False),
              (-bc_ret * 0.9, 0.0, True), (-bc_ret * 1.1, 0.0, False),
              (0.0, 4.7, True), (0.0, 5.7, False)]
    for al, be, want_captured in pixels:
        y0 = R.pixel_rays(al, be, 200.0, np.pi / 2, A)
        out = R.trace_rays(y0, A, r_esc=210.0, h_scale=0.01,
                           max_steps=100000)
        got_batch = out["status"][0] == R.CAPTURED
        sol = trace(m, y0[:, 0], 12000.0, rtol=1e-12, atol=1e-12,
                    events=[horizon_event(m), escape_event(210.0)])
        got_adaptive = not sol.y[1, -1] > 200.0
        assert got_batch == got_adaptive == want_captured, (al, be)


def test_capture_is_not_confused_by_the_bl_divergence():
    """Regression for a real bug: an RK4 step whose stages sample the BL
    metric divergence at Delta = 0 used to catapult plunging rays to
    |r| ~ 1e5 -- sometimes outward past the escape radius, silently
    misclassifying a captured ray.  This pixel is one that did exactly
    that.  The step-shrink guard and the two-step escape condition must
    keep it captured, with a finite final state."""
    y0 = R.pixel_rays(-1.3929, 4.15e-16, 200.0, np.pi / 2, A)
    out = R.trace_rays(y0, A, r_esc=210.0, h_scale=0.004, max_steps=250000)
    assert out["status"][0] == R.CAPTURED
    assert np.isfinite(out["y"][:, 0]).all()


# ---------------------------------------------------------------------------
# The shadow against Bardeen
# ---------------------------------------------------------------------------

def test_shadow_edges_hit_the_capture_thresholds():
    """Along beta = 0 the shadow boundary IS the pair of equatorial capture
    thresholds -- the sharpest single line to check."""
    al, _ = R.shadow_edge(np.pi, 200.0, np.pi / 2, A, tol=1e-4)
    assert al == pytest.approx(-analytic.kerr_critical_impact_parameter(A, True),
                               abs=2e-4)
    al, _ = R.shadow_edge(0.0, 200.0, np.pi / 2, A, tol=1e-4)
    assert al == pytest.approx(-analytic.kerr_critical_impact_parameter(A, False),
                               abs=2e-4)


def test_schwarzschild_shadow_is_a_circle_of_radius_3root3():
    psis = np.linspace(0.0, 2 * np.pi, 4, endpoint=False) + 0.4
    al, be = R.shadow_edges(psis, 200.0, np.pi / 2, 0.0, tol=1e-4)
    r_edge = np.hypot(al, be)
    assert np.all(np.abs(r_edge - 3.0 * np.sqrt(3.0)) < 3e-4)
    assert np.ptp(r_edge) < 3e-4                    # and it is round


def test_kerr_shadow_boundary_matches_bardeen_off_axis():
    """A non-equatorial angle at a non-equatorial inclination: the parts of
    the curve that no single closed-form threshold checks."""
    ac, bc = analytic.kerr_shadow_boundary(A, np.radians(60.0), n=40000)
    al, be = R.shadow_edges([2.2, 4.4], 200.0, np.radians(60.0), A, tol=2e-4)
    for x, y in zip(al, be):
        assert np.hypot(ac - x, bc - y).min() < 1e-3


# ---------------------------------------------------------------------------
# Disk physics
# ---------------------------------------------------------------------------

def test_doppler_factor_reduces_to_the_orbit_normalisation():
    """For a face-on photon (xi = 0), g = 1/u^t.  Check u^t against an
    independent construction: raise the (E, Lz) covariant momentum of
    `circular_orbit` with the inverse metric."""
    m = KerrBL(a=A)
    for r in (3.0, 6.0, 12.0):
        y, E, Lz = circular_orbit(m, r, True)
        x = np.array([0.0, r, np.pi / 2, 0.0])
        ut = float(np.asarray(m.ginv(x), float)[0] @ y[4:])
        assert R.doppler_factor(r, 0.0, A) == pytest.approx(1.0 / ut,
                                                            rel=1e-12)


def test_doppler_beaming_favours_the_approaching_side():
    """xi > 0 photons leave prograde material moving toward the observer.
    The asymmetry between the two sides is what lights up half of every real
    accretion-disk image.  (Note g_app barely misses 1.0 here: at r = 6M the
    gravitational redshift almost exactly cancels the Doppler blueshift --
    the observed g is a product of both, and only the *ratio* across the
    disk is pure Doppler.)"""
    g_app = R.doppler_factor(4.0, +5.0, A)
    g_mid = R.doppler_factor(4.0, 0.0, A)
    g_rec = R.doppler_factor(4.0, -5.0, A)
    assert g_app > g_mid > g_rec
    assert g_app > 1.0          # this deep in, beaming beats the redshift
    assert g_app / g_rec > 3.0
    # In brightness (g^4) that is two orders of magnitude of asymmetry.
    assert (g_app / g_rec) ** 4 > 80.0


def test_gravitational_redshift_wins_at_small_radius():
    """Face-on (no Doppler): light from deeper in the well arrives more
    redshifted, monotonically."""
    g = [R.doppler_factor(r, 0.0, A) for r in (2.4, 4.0, 10.0, 100.0)]
    assert np.all(np.diff(g) > 0)
    assert g[-1] == pytest.approx(1.0, abs=2e-2)


# ---------------------------------------------------------------------------
# The renderer end to end
# ---------------------------------------------------------------------------

def test_render_scene_smoke():
    """A small render must classify every pixel, find all three fates, and
    produce finite positive disk intensities."""
    scene = R.render_scene(A, np.radians(80.0), nx=64, ny=48, fov=14.0,
                           r_obs=400.0, h_scale=0.02, max_steps=40000)
    st = scene["status"]
    assert not np.any(st == R.FLYING)
    assert (st == R.CAPTURED).mean() > 0.01        # a shadow exists
    assert (st == R.DISK).mean() > 0.05            # the disk is visible
    assert (st == R.ESCAPED).mean() > 0.3          # most of the sky is sky
    I = scene["intensity"]
    assert np.isfinite(I[st == R.DISK]).all()
    assert (I[st == R.DISK] > 0).all()


def test_render_shadow_shifts_with_spin():
    """The a = 0.9 shadow centroid is displaced toward positive alpha
    (retrograde side) relative to a = 0 -- frame dragging as seen in the
    image, cheaply measured from classified pixels."""
    kw = dict(nx=96, ny=72, fov=10.0, r_obs=400.0, h_scale=0.02,
              max_steps=40000)
    c = {}
    for a in (0.0, A):
        scene = R.render_scene(a, np.pi / 2, disk_rout=0.0, **kw)
        st, alpha = scene["status"], scene["alpha"]
        cap = (st == R.CAPTURED)
        c[a] = (cap * alpha[None, :]).sum() / cap.sum()
    assert abs(c[0.0]) < 0.15                      # centred without spin
    assert c[A] > 1.0                              # dragged sideways with it