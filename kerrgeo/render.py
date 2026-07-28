"""Backwards ray tracing: what a Kerr black hole looks like.

The idea is standard and slightly upside down: instead of following light
from sources to the eye (almost all of it misses), follow each camera pixel's
ray *backwards* -- reverse the photon momentum and integrate away from the
camera -- and ask where it came from.  Three fates:

    captured   the backwards ray crosses the horizon: nothing can have come
               from there.  Black pixel.  The set of such pixels is the
               black-hole shadow.
    disk       the ray meets the equatorial plane inside the accretion disk:
               the pixel shows disk material, shifted and beamed by the
               emitter's motion and the climb out of the potential well.
    escaped    the ray reaches large radius: the pixel shows (lensed)
               background sky.

--------------------------------------------------------------------------
The camera: conserved quantities, not angles
--------------------------------------------------------------------------
Each pixel is labelled by Bardeen's celestial coordinates (alpha, beta) --
apparent displacement on the sky, in units of M, perpendicular and parallel
to the projected spin axis.  For an observer at infinity these map exactly to
the photon's conserved quantities:

    xi  = -alpha sin(theta_obs)                       (= Lz/E)
    eta = beta^2 + (alpha^2 - a^2) cos^2(theta_obs)   (= Q/E^2)

and the pixel's backwards ray is just `state_from_constants` with E = 1.
Building the camera this way has a sharp payoff for validation: whether a
photon is captured depends only on its constants, not on where along the ray
you start integrating.  So the shadow boundary measured by *this* camera must
match Bardeen's analytic curve at any camera radius, to bisection tolerance,
with no finite-distance aberration to argue about.  (A finite-r camera built
from a local tetrad would see the shadow slightly differently; that
correction is O(M/r_obs) bookkeeping, not physics, and is skipped here.)

--------------------------------------------------------------------------
Why a separate vectorised integrator
--------------------------------------------------------------------------
An image is 10^4..10^5 rays.  scipy's solve_ivp handles one ODE system at a
time, and Python-level looping over rays costs minutes to hours.  Instead,
`trace_rays` integrates ALL rays simultaneously: the state is an (8, N)
array, the RK4 stages are numpy expressions over the whole batch, and each
ray carries its own step size h ~ r (small near the hole, large far away).
That turns rendering from hours into tens of seconds -- RK4's fitness for
exactly this single-pass bulk workload was the stage-1 integrator-choice
conclusion, now cashed in.

The batch right-hand side necessarily re-states the BL inverse metric as
array formulas (`_ginv_bl`).  Duplicated physics is a bug farm, so the very
first stage-4 test pins `_batch_rhs` against `kerrgeo.rhs` component by
component on random states; if the two implementations ever drift apart, the
suite says so.  Derivatives use the same complex-step trick as the Metric
class -- exact to rounding, no hand-derived gradients.
"""

from __future__ import annotations

import numpy as np

_CSTEP = 1e-20


# ---------------------------------------------------------------------------
# Vectorised metric and RHS (Boyer-Lindquist, mirrors KerrBL.ginv)
# ---------------------------------------------------------------------------

def _ginv_bl(r, th, a, M):
    """The five nonzero BL inverse-metric components, as arrays.

    Complex-step safe: works elementwise on complex arrays.
    """
    s2 = np.sin(th) ** 2
    c2 = np.cos(th) ** 2
    S = r * r + a * a * c2
    D = r * r - 2.0 * M * r + a * a
    A = (r * r + a * a) ** 2 - a * a * D * s2
    return (-A / (S * D),            # g^tt
            -2.0 * a * M * r / (S * D),   # g^tphi
            D / S,                   # g^rr
            1.0 / S,                 # g^thth
            (D - a * a * s2) / (S * D * s2))  # g^phiphi


def _batch_rhs(y, a, M):
    """Hamilton's equations for an (8, N) batch of BL states."""
    r, th = y[1], y[2]
    p0, p1, p2, p3 = y[4], y[5], y[6], y[7]

    gtt, gtp, grr, gqq, gpp = _ginv_bl(r, th, a, M)

    dx0 = gtt * p0 + gtp * p3
    dx1 = grr * p1
    dx2 = gqq * p2
    dx3 = gtp * p0 + gpp * p3

    # d_r g^ab and d_th g^ab by complex step; t, phi derivatives are zero,
    # so p_t and p_phi are exactly constant (same guarantee as everywhere).
    cr = _ginv_bl(r + 1j * _CSTEP, th, a, M)
    ct = _ginv_bl(r, th + 1j * _CSTEP, a, M)

    def quad(comps):
        gtt_, gtp_, grr_, gqq_, gpp_ = comps
        return (gtt_ * p0 * p0 + 2.0 * gtp_ * p0 * p3 + grr_ * p1 * p1
                + gqq_ * p2 * p2 + gpp_ * p3 * p3)

    dp1 = -0.5 * np.imag(quad(cr)) / _CSTEP
    dp2 = -0.5 * np.imag(quad(ct)) / _CSTEP

    zero = np.zeros_like(r)
    return np.array([dx0, dx1, dx2, dx3, zero, dp1, dp2, zero])


# ---------------------------------------------------------------------------
# The camera
# ---------------------------------------------------------------------------

def pixel_constants(alpha, beta, theta_obs, a):
    """(xi, eta) for the photon arriving at celestial position (alpha, beta)."""
    s, c = np.sin(theta_obs), np.cos(theta_obs)
    xi = -np.asarray(alpha) * s
    eta = np.asarray(beta) ** 2 + (np.asarray(alpha) ** 2 - a * a) * c * c
    return xi, eta


def pixel_rays(alpha, beta, r_obs, theta_obs, a, M=1.0):
    """Initial (8, N) batch of backwards rays for a grid of pixels.

    E = 1 normalisation; sign_r = -1 (the backwards ray leaves the camera
    heading inward); the sign of p_theta is the sign of beta (a pixel above
    the spin axis looks over the pole).

    p_r is built from the radial potential exactly as in
    `momenta_from_constants`; pixels whose R(r_obs) < 0 cannot correspond to
    any photon reaching the observer and are flagged by NaN (they only occur
    for absurd fields of view).
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float)).ravel()
    beta = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    xi, eta = pixel_constants(alpha, beta, theta_obs, a)

    s2 = np.sin(theta_obs) ** 2
    c2 = np.cos(theta_obs) ** 2
    Th = eta + a * a * c2 - xi * xi * c2 / s2       # Theta(theta_obs), mu=0
    Th = np.where(np.abs(Th) < 1e-14, 0.0, Th)
    p_th = np.sign(beta) * np.sqrt(np.maximum(Th, 0.0))

    # R(r) = P^2 - Delta K, photon form.
    D = r_obs * r_obs - 2.0 * M * r_obs + a * a
    P = (r_obs * r_obs + a * a) - a * xi
    K = (xi - a) ** 2 + eta
    R = P * P - D * K
    p_r = -np.sqrt(np.maximum(R, 0.0)) / D          # ingoing (BL, exterior)
    p_r = np.where(R < 0, np.nan, p_r)

    n = alpha.size
    y0 = np.zeros((8, n))
    y0[1] = r_obs
    y0[2] = theta_obs
    y0[4] = -1.0
    y0[5] = p_r
    y0[6] = p_th
    y0[7] = xi
    return y0


# ---------------------------------------------------------------------------
# The batch tracer
# ---------------------------------------------------------------------------

#: status codes
FLYING, CAPTURED, ESCAPED, DISK = 0, 1, 2, 3


def trace_rays(y0, a, M=1.0, r_esc=None, disk=None, eps_horizon=5e-3,
               h_scale=0.04, h_min=0.02, h_max=25.0, max_steps=6000):
    """Integrate an (8, N) batch of rays to their fates.

    Per-ray adaptive step h = clip(h_scale * r, h_min, h_max): fine spirals
    near the photon orbits get resolved, the long haul to the camera does not
    burn steps.  Finished rays are frozen (masked out of the update), so cost
    is dominated by the few near-critical rays that wind longest.

    Horizon handling deserves a comment, because the naive version fails in
    a memorable way.  This tracer works in BL coordinates, whose metric
    diverges at Delta = 0 -- and an RK4 step is not a point evaluation: its
    STAGES sample ahead of the current state.  If a plunging ray is allowed
    to take an ordinary-sized step near r_+, a stage can land in the
    divergent zone, and the resulting garbage step catapults the ray to
    |r| ~ 1e5 in either direction -- including *outward past the escape
    radius*, silently misclassifying a captured ray as escaped.  Two
    defences: the step is shrunk geometrically as the buffer is approached
    (h <= 0.3 (r - r_stop), floored), so no stage can reach the divergence
    before the capture flag fires; and 'escaped' additionally requires the
    ray to have been at r > r_esc/2 on the previous step, so a single-step
    teleport can never satisfy it.

    disk : (r_in, r_out) or None
        Opaque thin disk in the equatorial plane.  A crossing is detected by
        the sign change of (theta - pi/2) across a step and located by linear
        interpolation -- adequate because near the disk h is already small.

    Returns dict with 'status', 'r_hit' (disk rays), 'y' (final states).
    """
    a = float(a)
    y = np.array(y0, dtype=float)
    n = y.shape[1]
    if r_esc is None:
        r_esc = 1.05 * np.nanmax(y[1])
    r_plus = M + np.sqrt(M * M - a * a)
    r_stop = r_plus * (1.0 + eps_horizon)

    status = np.full(n, FLYING, dtype=np.int8)
    status[~np.isfinite(y[5])] = CAPTURED           # unphysical pixels: black
    r_hit = np.full(n, np.nan)
    r_min = np.full(n, np.inf)                      # closest approach, per ray

    for _ in range(max_steps):
        act = status == FLYING
        if not act.any():
            break
        ya = y[:, act]
        h = np.clip(h_scale * np.abs(ya[1]), h_min, h_max)
        # Geometric approach to the capture buffer: no RK4 stage may sample
        # the BL divergence at Delta = 0 (see docstring).
        near = ya[1] < r_stop + 0.6
        h = np.where(near,
                     np.minimum(h, np.maximum(0.3 * (ya[1] - r_stop), 1e-3)),
                     h)
        # Near the polar axis phi winds at 1/sin^2(theta): rays with small
        # |xi| swing over the pole faster than a radius-scaled step can
        # track, leaving a seam of mistraced pixels along alpha = 0.  Scale
        # h with sin(theta) so the swing is resolved; only near-polar rays
        # pay the extra steps.
        s_ax = np.abs(np.sin(ya[2]))
        h = h * np.clip(s_ax / 0.25, 0.15, 1.0)

        k1 = _batch_rhs(ya, a, M)
        k2 = _batch_rhs(ya + 0.5 * h * k1, a, M)
        k3 = _batch_rhs(ya + 0.5 * h * k2, a, M)
        k4 = _batch_rhs(ya + h * k3, a, M)
        yn = ya + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        new_status = np.full(yn.shape[1], FLYING, dtype=np.int8)
        new_status[yn[1] < r_stop] = CAPTURED
        new_status[~np.isfinite(yn[1])] = CAPTURED  # belt and braces
        # Genuine escapes climb out; teleports from a corrupted step do not.
        new_status[(yn[1] > r_esc) & (ya[1] > 0.5 * r_esc)] = ESCAPED

        if disk is not None:
            r_in, r_out = disk
            f0, f1 = ya[2] - np.pi / 2, yn[2] - np.pi / 2
            crossed = (np.sign(f0) != np.sign(f1)) & (new_status == FLYING)
            if crossed.any():
                w = f0[crossed] / (f0[crossed] - f1[crossed])
                r_c = ya[1, crossed] * (1 - w) + yn[1, crossed] * w
                hit = (r_c >= r_in) & (r_c <= r_out)
                ci = np.where(crossed)[0][hit]
                new_status[ci] = DISK
                idx = np.where(act)[0][ci]
                r_hit[idx] = r_c[hit]

        y[:, act] = yn
        status[act] = new_status
        idx_act = np.where(act)[0]
        r_min[idx_act] = np.minimum(r_min[idx_act], yn[1])

    return {"status": status, "r_hit": r_hit, "y": y, "r_min": r_min}


def trace_rays_refined(y0, a, M=1.0, shell_margin=1.15, refine_factor=0.2,
                       **kw):
    """Two-pass tracing: full batch at the requested step, then re-trace the
    suspect rays at a finer one.

    'Suspect' = classified escaped, but having passed inside the photon
    shell region (r_min < shell_margin * r_photon_retrograde).  Those are
    the rays whose winding near the unstable shell can amplify fixed-step
    error into a wrong escape; genuinely escaping rays that never came close
    are beyond suspicion.  Re-tracing the few-percent suspect population at
    h_scale/5 cuts their step error by ~600x (RK4 is 4th order) for a few
    percent of the original cost, removing the speckle of misclassified
    pixels inside the shadow.
    """
    from .metrics.kerr import KerrBL

    out = trace_rays(y0, a, M, **kw)
    r_shell_out = KerrBL(a=a, M=M).r_photon(prograde=False)
    suspect = (out["status"] == ESCAPED) & \
              (out["r_min"] < shell_margin * r_shell_out)
    if suspect.any():
        kw2 = dict(kw)
        kw2["h_scale"] = refine_factor * kw.get("h_scale", 0.04)
        kw2["max_steps"] = int(kw.get("max_steps", 6000) / refine_factor)
        redo = trace_rays(y0[:, suspect], a, M, **kw2)
        idx = np.where(suspect)[0]
        out["status"][idx] = redo["status"]
        out["r_hit"][idx] = redo["r_hit"]
        out["y"][:, idx] = redo["y"]
    return out


# ---------------------------------------------------------------------------
# Shadow boundary by bisection
# ---------------------------------------------------------------------------

def _is_captured(alpha, beta, r_obs, theta_obs, a, M=1.0):
    """Classify one pixel with the trusted adaptive integrator.

    Deliberately NOT the batch tracer.  Near-critical rays wind along the
    unstable photon shell, where the shell e-folds integration error (the
    stage-2 instability measurement, now working against us): fixed-step RK4
    leaves thin annuli of misclassified pixels near the boundary no matter
    how the step is tuned, and radial bisection will happily lock onto one.
    The boundary *measurement* therefore uses DOP853 at rtol 1e-12 -- the
    same integrator stack every other validation in this project rests on --
    and the batch tracer is reserved for imaging, where a hairline
    misclassification annulus is invisible at pixel resolution.

    A ray still winding when the affine budget runs out is counted as
    captured; the resulting edge bias is below the bisection tolerance.

    Tolerance choice: adaptivity means the trajectory error stays ~rtol no
    matter how long the ray winds -- the misclassified annulus has width
    ~rtol in impact parameter.  1e-9 is five orders below the bisection
    tolerance and an order of magnitude faster than 1e-12.
    """
    from .integrate import trace
    from .events import horizon_event, escape_event
    from .metrics.kerr import KerrBL

    m = KerrBL(a=a, M=M)
    y0 = pixel_rays(alpha, beta, r_obs, theta_obs, a, M)[:, 0]
    sol = trace(m, y0, 10.0 * r_obs, rtol=1e-9, atol=1e-9,
                events=[horizon_event(m), escape_event(1.05 * r_obs)])
    return not sol.y[1, -1] > r_obs


def shadow_edges(psis, r_obs, theta_obs, a, M=1.0, tol=1e-4):
    """Measure the shadow boundary along the given position angles.

    Radial bisection from the centroid of the analytic curve.  The analytic
    curve enters only through the bracket seed (+/-30%); a boundary in
    genuine disagreement would escape the bracket and fail loudly rather
    than be quietly confirmed.

    Returns (alpha, beta) arrays of the measured edge points.
    """
    from .analytic import kerr_shadow_boundary

    psis = np.atleast_1d(np.asarray(psis, dtype=float))
    ac, bc = kerr_shadow_boundary(a, theta_obs, n=4000, M=M)
    a0 = 0.5 * (ac.min() + ac.max())               # centroid alpha offset

    rr = np.hypot(ac - a0, bc)
    ang = np.mod(np.arctan2(bc, ac - a0), 2 * np.pi)
    o = np.argsort(ang)
    r_seed = np.interp(np.mod(psis, 2 * np.pi), ang[o], rr[o])

    out_a = np.empty_like(psis)
    out_b = np.empty_like(psis)
    for i, (psi, rs) in enumerate(zip(psis, r_seed)):
        cs, sn = np.cos(psi), np.sin(psi)
        lo, hi = 0.70 * rs, 1.30 * rs
        if not _is_captured(a0 + lo * cs, lo * sn, r_obs, theta_obs, a, M):
            raise RuntimeError(f"inner bracket escaped at psi={psi:.3f}")
        if _is_captured(a0 + hi * cs, hi * sn, r_obs, theta_obs, a, M):
            raise RuntimeError(f"outer bracket captured at psi={psi:.3f}")
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if _is_captured(a0 + mid * cs, mid * sn, r_obs, theta_obs, a, M):
                lo = mid
            else:
                hi = mid
        r_edge = 0.5 * (lo + hi)
        out_a[i], out_b[i] = a0 + r_edge * cs, r_edge * sn
    return out_a, out_b


def shadow_edge(psi, r_obs, theta_obs, a, M=1.0, tol=1e-4):
    """Single-angle convenience wrapper around `shadow_edges`."""
    al, be = shadow_edges([psi], r_obs, theta_obs, a, M, tol)
    return float(al[0]), float(be[0])


# ---------------------------------------------------------------------------
# Disk physics
# ---------------------------------------------------------------------------

def doppler_factor(r, xi, a, M=1.0):
    """g = E_obs / E_emit for a photon leaving the (prograde, circular,
    equatorial) disk and reaching the static observer at infinity.

    Both energies are projections of the SAME conserved photon momentum:
    E_obs = -p.u_obs = E = 1 at infinity, and

        E_emit = -p.u_emit = u^t (1 - Omega xi)

    for the emitter on a circular orbit with angular velocity Omega.  So

        g = 1 / [ u^t (1 - Omega xi) ]

    xi > 0 rays (emitted by material approaching the observer on the
    prograde side) are blueshifted -- this is what makes one side of every
    real accretion-disk image brighter.  u^t is computed from the
    normalisation u.u = -1 rather than a separate closed form, so this
    function has no independent physics to get wrong.
    """
    Om = np.sqrt(M) / (r ** 1.5 + a * np.sqrt(M))
    # BL covariant components on the equator (Sigma = r^2):
    gtt = -(1.0 - 2.0 * M / r)
    gtp = -2.0 * a * M / r
    gpp = r * r + a * a + 2.0 * a * a * M / r
    ut = 1.0 / np.sqrt(-(gtt + 2.0 * Om * gtp + Om * Om * gpp))
    return 1.0 / (ut * (1.0 - Om * xi))


def render_scene(a, theta_obs, nx=320, ny=240, fov=24.0, r_obs=1000.0,
                 disk_rin=None, disk_rout=18.0, M=1.0, **trace_kw):
    """Render the classic image: shadow + lensed thin disk + checkered sky.

    Returns dict of image-shaped arrays: 'status', 'intensity' (disk, Doppler
    boosted, NaN elsewhere), 'checker' (background parity, NaN elsewhere),
    plus the pixel axes.  Plotting is left to the caller -- this module owes
    the physics, not the colormap.
    """
    from .metrics.kerr import KerrBL

    if disk_rin is None:
        disk_rin = KerrBL(a=a).r_isco(True)

    al = np.linspace(-fov, fov, nx)
    be = np.linspace(-fov * ny / nx, fov * ny / nx, ny)
    A_, B_ = np.meshgrid(al, be)

    y0 = pixel_rays(A_.ravel(), B_.ravel(), r_obs, theta_obs, a, M)
    out = trace_rays_refined(y0, a, M, r_esc=1.05 * r_obs,
                             disk=(disk_rin, disk_rout), **trace_kw)

    status = out["status"].reshape(ny, nx)
    r_hit = out["r_hit"].reshape(ny, nx)
    xi = y0[7].reshape(ny, nx)

    # Disk brightness: bolometric flux transforms as g^4; emissivity falls
    # off as r^-2 (a stand-in profile -- the geometry and Doppler pattern,
    # not the disk astrophysics, are the point here).
    g = doppler_factor(r_hit, xi, a, M)
    intensity = np.where(status == DISK, g ** 4 * (disk_rin / r_hit) ** 2,
                         np.nan)

    # Background: a checker in the escaped ray's arrival direction.  NOT a
    # (theta, phi) checker: any longitude-based pattern is genuinely
    # discontinuous through the poles (all meridians converge there), and
    # rays with xi ~ 0 pass over the pole -- so a lat/long checker paints a
    # spurious-looking seam down the alpha = 0 column of the image.  Cells
    # built by flooring the Cartesian components of the direction vector
    # have no singular points anywhere on the sphere.
    yf = out["y"]
    thf = yf[2].reshape(ny, nx)
    phf = yf[3].reshape(ny, nx)
    nx_v = np.sin(thf) * np.cos(phf)
    ny_v = np.sin(thf) * np.sin(phf)
    nz_v = np.cos(thf)
    cell = 0.35
    checker = np.where(
        status == ESCAPED,
        (np.floor((nx_v + 1) / cell) + np.floor((ny_v + 1) / cell)
         + np.floor((nz_v + 1) / cell)) % 2,
        np.nan)

    return {"status": status, "intensity": intensity, "checker": checker,
            "alpha": al, "beta": be, "r_hit": r_hit}
