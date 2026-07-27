"""Integrators, and the reasoning behind which one to use where.

--------------------------------------------------------------------------
Why your N-body RK4 drifted, and whether that carries over
--------------------------------------------------------------------------
Fixed-step RK4 is not symplectic, and on a bound orbit its truncation error has
a *secular* component -- a small bias in the same direction every orbit, so
energy error grows roughly linearly with the number of orbits rather than
averaging out.  Over a solar-system integration of many thousands of orbits
that dominates everything.

Whether it matters here depends entirely on what you are computing:

  * Light deflection, capture cross-sections, ray tracing.  A single pass, no
    repeated orbits, nothing to accumulate.  Secular drift is irrelevant; what
    matters is local accuracy through periapsis.  RK4 is fine.
  * Perihelion precession over hundreds of orbits.  This is the N-body
    situation again, and it is worse than it looks: a spurious secular change
    in E or Lz would masquerade as *extra precession*, i.e. the numerical error
    contaminates the exact quantity you are trying to measure.
  * Long-term bound orbits, inspirals.  Symplectic territory.

--------------------------------------------------------------------------
Why you cannot just use leapfrog
--------------------------------------------------------------------------
The obvious reflex -- "use velocity Verlet like a good N-body person" -- does
not work.  Leapfrog and its relatives require a *separable* Hamiltonian,
H = T(p) + V(x), so that each half-step can be solved exactly.  Ours is

    H = (1/2) g^{ab}(x) p_a p_b

which is quadratic in p but with position-dependent coefficients: x and p are
inseparably coupled.  There is no exact drift/kick split.

Three real options remain, and all three are implemented here so the repo can
show the comparison rather than assert it:

1. `rk4` -- fixed-step, explicit, 4th order.  Fast, simple, non-symplectic.
   The baseline, and genuinely the right tool for ray tracing.

2. `dop853` (via scipy) -- adaptive, explicit, 8th order.  The default.
   Adaptivity matters here for a reason independent of accuracy: near periapsis
   and near the photon sphere the coordinate velocity varies by orders of
   magnitude, so any fixed step is simultaneously wasteful far out and
   inadequate close in.  At rtol = 1e-12 the norm drift over thousands of M is
   ~1e-13 -- far below anything physical -- and it is still not symplectic, but
   with error that small over a non-repeating trajectory it does not matter.

3. `gauss_legendre` -- implicit Runge-Kutta at the Gauss points, 4th order
   (2-stage) or 6th (3-stage).  This *is* symplectic, and unlike leapfrog it
   works for a general non-separable H.  The price is that each step needs a
   nonlinear solve; we use fixed-point iteration, which converges quickly
   because the step is small.  It also exactly conserves quadratic first
   integrals -- which for us means the norm g^{ab} p_a p_b is preserved to
   machine precision automatically.  Use it for long-term bound orbits.

Recommendation: `dop853` for everything in stages 1-2, `rk4` for bulk ray
tracing in stage 4, `gauss_legendre` when you want to make a claim about
long-term orbital dynamics.  The validation suite quantifies the difference
instead of taking anyone's word for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .hamiltonian import rhs


@dataclass
class Solution:
    """Uniform result object across integrators.

    Mirrors the parts of scipy's OdeResult we use, so downstream code does not
    care which integrator produced it.
    """

    t: np.ndarray                       # affine parameter values, shape (N,)
    y: np.ndarray                       # states, shape (8, N)
    status: str = "ok"                  # "ok" | "captured" | "escaped" | "event"
    t_events: list = field(default_factory=list)
    y_events: list = field(default_factory=list)
    nfev: int = 0
    method: str = ""

    @property
    def r(self):
        return self.y[1]

    @property
    def theta(self):
        return self.y[2]

    @property
    def phi(self):
        return self.y[3]

    def cartesian(self):
        """Pseudo-Cartesian coordinates for plotting.

        A caution: BL (r, theta, phi) are not spherical polars in flat space,
        so x = r sin(th) cos(ph) etc. is a *visualisation convention*, not a
        coordinate transformation.  It is asymptotically correct at large r,
        which is where we do all the plotting, and it is what everyone in the
        field plots.  (The Kerr-Schild convention, which uses
        sqrt(r^2+a^2) sin(th) for the transverse radius, is more faithful near
        the hole; the difference is invisible beyond a few M.)
        """
        r, th, ph = self.y[1], self.y[2], self.y[3]
        return (r * np.sin(th) * np.cos(ph),
                r * np.sin(th) * np.sin(ph),
                r * np.cos(th))


# ---------------------------------------------------------------------------
# Explicit fixed-step RK4
# ---------------------------------------------------------------------------

def rk4(metric, y0, lam_max, n_steps, events=None):
    """Classic fixed-step RK4.  Events are checked by sign change per step."""
    events = list(events or [])
    h = lam_max / n_steps
    ys = np.empty((8, n_steps + 1))
    ts = np.empty(n_steps + 1)
    ys[:, 0] = y0
    ts[0] = 0.0

    y = np.array(y0, dtype=float)
    lam = 0.0
    status = "ok"
    n = 0
    for i in range(1, n_steps + 1):
        k1 = rhs(lam, y, metric)
        k2 = rhs(lam + h / 2, y + h / 2 * k1, metric)
        k3 = rhs(lam + h / 2, y + h / 2 * k2, metric)
        k4 = rhs(lam + h, y + h * k3, metric)
        y_new = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        lam_new = lam + h
        n += 4

        hit = False
        for ev in events:
            if ev(lam, y) * ev(lam_new, y_new) < 0 and getattr(ev, "terminal", False):
                hit = True
        y, lam = y_new, lam_new
        ys[:, i] = y
        ts[i] = lam
        if hit:
            ys = ys[:, : i + 1]
            ts = ts[: i + 1]
            status = "event"
            break

    return Solution(t=ts, y=ys, status=status, nfev=n, method="RK4")


# ---------------------------------------------------------------------------
# Implicit symplectic Gauss-Legendre
# ---------------------------------------------------------------------------

# 2-stage Gauss-Legendre: order 4, symplectic, A-stable.
_GL2_C = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
_GL2_A = np.array([[0.25, 0.25 - np.sqrt(3) / 6],
                   [0.25 + np.sqrt(3) / 6, 0.25]])
_GL2_B = np.array([0.5, 0.5])

# 3-stage Gauss-Legendre: order 6.
_s15 = np.sqrt(15)
_GL3_C = np.array([0.5 - _s15 / 10, 0.5, 0.5 + _s15 / 10])
_GL3_A = np.array([
    [5 / 36,                2 / 9 - _s15 / 15,   5 / 36 - _s15 / 30],
    [5 / 36 + _s15 / 24,    2 / 9,               5 / 36 - _s15 / 24],
    [5 / 36 + _s15 / 30,    2 / 9 + _s15 / 15,   5 / 36],
])
_GL3_B = np.array([5 / 18, 4 / 9, 5 / 18])


def gauss_legendre(metric, y0, lam_max, n_steps, stages=2,
                   tol=1e-14, max_iter=50, events=None):
    """Implicit Gauss-Legendre Runge-Kutta: symplectic, fixed step.

    The stage equations  K_i = f(y + h sum_j a_ij K_j)  are solved by fixed-point
    iteration.  For a smooth problem and a reasonable step this converges in a
    handful of iterations; if it stops converging the step is far too large for
    the local curvature, which is itself useful information.

    Being symplectic, this conserves a *modified* Hamiltonian exactly, which
    means the error in the true H oscillates with bounded amplitude instead of
    growing secularly.  It also conserves quadratic first integrals exactly, so
    the norm g^{ab} p_a p_b holds to machine precision by construction -- worth
    knowing when interpreting the diagnostics, since it makes the norm look
    flattering compared to what the trajectory error actually is.
    """
    if stages == 2:
        C, A, B = _GL2_C, _GL2_A, _GL2_B
    elif stages == 3:
        C, A, B = _GL3_C, _GL3_A, _GL3_B
    else:
        raise ValueError("stages must be 2 or 3")

    events = list(events or [])
    h = lam_max / n_steps
    s = len(C)
    ys = np.empty((8, n_steps + 1))
    ts = np.empty(n_steps + 1)
    ys[:, 0] = y0
    ts[0] = 0.0

    y = np.array(y0, dtype=float)
    lam = 0.0
    status = "ok"
    nfev = 0

    for i in range(1, n_steps + 1):
        K = np.tile(rhs(lam, y, metric), (s, 1))  # shape (s, 8), initial guess
        nfev += 1
        for _ in range(max_iter):
            K_new = np.empty_like(K)
            for j in range(s):
                stage_y = y + h * (A[j] @ K)
                K_new[j] = rhs(lam + C[j] * h, stage_y, metric)
                nfev += 1
            if np.max(np.abs(K_new - K)) < tol:
                K = K_new
                break
            K = K_new
        else:
            raise RuntimeError(
                f"Gauss-Legendre stage iteration failed to converge at "
                f"lambda = {lam:.6g}, r = {y[1]:.6g}.  The step h = {h:.3g} is "
                "too large for the local curvature -- use more steps."
            )

        y_new = y + h * (B @ K)
        lam_new = lam + h

        hit = any(
            ev(lam, y) * ev(lam_new, y_new) < 0 and getattr(ev, "terminal", False)
            for ev in events
        )
        y, lam = y_new, lam_new
        ys[:, i] = y
        ts[i] = lam
        if hit:
            ys, ts = ys[:, : i + 1], ts[: i + 1]
            status = "event"
            break

    return Solution(t=ts, y=ys, status=status, nfev=nfev,
                    method=f"GL{stages} (symplectic)")


# ---------------------------------------------------------------------------
# Adaptive explicit -- the default
# ---------------------------------------------------------------------------

def trace(metric, y0, lam_max, method="DOP853", rtol=1e-12, atol=1e-12,
          events=None, dense_output=False, max_step=np.inf, n_steps=None):
    """Integrate a geodesic.  The general entry point.

    Parameters
    ----------
    metric : Metric
    y0 : array_like, shape (8,)
        Initial [t, r, theta, phi, p_t, p_r, p_theta, p_phi].
    lam_max : float
        How far to integrate in affine parameter.
    method : str
        "DOP853" or "RK45" (adaptive, via scipy), "RK4", or "GL2"/"GL3"
        (fixed step -- requires ``n_steps``).
    events : list
        Callables from `kerrgeo.events`.  Terminal ones stop the integration.

    Returns
    -------
    Solution
    """
    y0 = np.asarray(y0, dtype=float)
    events = list(events or [])

    if method == "RK4":
        if n_steps is None:
            raise ValueError("RK4 is fixed-step: pass n_steps")
        return rk4(metric, y0, lam_max, n_steps, events)
    if method in ("GL2", "GL3"):
        if n_steps is None:
            raise ValueError(f"{method} is fixed-step: pass n_steps")
        return gauss_legendre(metric, y0, lam_max, n_steps,
                              stages=int(method[-1]), events=events)

    sol = solve_ivp(
        rhs, (0.0, lam_max), y0, args=(metric,),
        method=method, rtol=rtol, atol=atol,
        events=events or None, dense_output=dense_output, max_step=max_step,
    )

    status = "ok"
    if sol.status == 1:  # a terminal event fired
        status = "event"
    return Solution(t=sol.t, y=sol.y, status=status,
                    t_events=list(sol.t_events) if sol.t_events else [],
                    y_events=list(sol.y_events) if sol.y_events else [],
                    nfev=sol.nfev, method=method)
