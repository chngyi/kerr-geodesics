"""Generate the validation figures.

    python scripts/make_figures.py [--out figures]

Each figure is a claim the repository makes about its own correctness, plotted
so the claim can be checked by eye as well as by the test suite.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import style  # noqa: E402
from kerrgeo import (  # noqa: E402
    KerrBL,
    Schwarzschild,
    analytic,
    drift_report,
    measure,
    orbit_from_apsides,
    photon_from_impact_parameter,
    trace,
)
from kerrgeo.events import escape_event, horizon_event  # noqa: E402
from kerrgeo.invariants import all_invariants  # noqa: E402

style.use()
CAT = style.CAT


# ---------------------------------------------------------------------------

def fig_photon_trajectories(out):
    """Photon paths across the capture threshold.

    The physics to look for: rays with b < 3sqrt(3) M spiral in and are
    captured; rays with b just above it loop around the photon sphere one or
    more times before escaping; far rays bend gently.  The dashed circle at
    r = 3M is the photon sphere.
    """
    bh = Schwarzschild()
    bc = analytic.critical_impact_parameter()
    bs = [3.0, 4.5, 5.19, bc + 1e-3, bc + 0.05, 6.0, 8.0, 12.0, 20.0]
    colors = style.sequential(len(bs), lo=0.38, hi=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, lim, title in (
        (axes[0], 30, "Photon trajectories in Schwarzschild\n"
                      "colour: light to dark with increasing b (3M to 20M)"),
        (axes[1], 8, "Detail: the photon sphere at r = 3M\n"
                     "rays near $b_c$ wind before escaping"),
    ):
        for b, c in zip(bs, colors):
            y0 = photon_from_impact_parameter(bh, r0=60.0, b=b)
            sol = trace(bh, y0, 900.0, rtol=1e-12, atol=1e-12,
                        events=[horizon_event(bh), escape_event(60.0)])
            r, phi = sol.y[1], sol.y[3]
            ax.plot(r * np.cos(phi), r * np.sin(phi), color=c, lw=1.5,
                    solid_capstyle="round", zorder=3)
        style.draw_hole(ax, 2.0, r_photon=3.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.72, lim * 0.72)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x  [M]")
        ax.grid(alpha=0.55)
    axes[0].set_ylabel("y  [M]")

    # Direct labels instead of a nine-entry legend.
    axes[0].annotate("captured\n$b < 5.196\\,M$", (-19, -14), color=colors[3],
                     fontsize=8.5, ha="center", fontweight="medium")
    axes[0].annotate("escaping\n$b > 5.196\\,M$", (-19, 13), color=colors[-1],
                     fontsize=8.5, ha="center", fontweight="medium")
    axes[1].annotate("photon sphere\n$r = 3M$", (3.4, -5.2),
                     color=style.INK_MUTED, fontsize=8.5)

    fig.savefig(os.path.join(out, "photon_trajectories.png"))
    plt.close(fig)


def fig_deflection(out):
    """Measured deflection against the exact result and the weak-field series.

    The lower panel is the real content: it shows the integrator tracking the
    exact quadrature to ~1e-10 relative across three decades of impact
    parameter, while the first-order Einstein formula 4M/b is off by 20% at
    b = 6M and the third-order series is still off by 1e-3 there.
    """
    bh = Schwarzschild()
    bs = np.geomspace(5.4, 3000.0, 34)
    meas = np.array([measure.measure_deflection(bh, b, r0=max(1e5, 400 * b))
                     for b in bs])
    exact = np.array([analytic.deflection_exact(b) for b in bs])
    w1 = analytic.deflection_weak(bs, order=1)
    w3 = analytic.deflection_weak(bs, order=3)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.4, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.12})

    ax0.loglog(bs, exact, color=style.INK, lw=1.8, zorder=3,
               label="exact (quadrature)")
    ax0.loglog(bs, w1, color=CAT[1], lw=1.6, linestyle=style.DASH, zorder=2,
               label=r"weak field, $4M/b$")
    ax0.loglog(bs, w3, color=CAT[2], lw=1.6, linestyle=style.DOT, zorder=2,
               label="weak field, 3 terms")
    ax0.loglog(bs, meas, "o", color=CAT[0], ms=5.0, zorder=4,
               label="integrated geodesics")
    ax0.axvline(analytic.critical_impact_parameter(), color=style.INK_MUTED,
                lw=0.9, linestyle=style.FINEDOT, zorder=1)
    ax0.annotate(r"capture: $b_c=3\sqrt{3}\,M$", (5.45, 3.2e-3),
                 color=style.INK_MUTED, fontsize=8.5, rotation=90)
    ax0.set_ylabel(r"deflection $\alpha$  [rad]")
    ax0.set_title("Light deflection: integrated geodesics vs closed form")
    ax0.legend(loc="upper right")

    ax1.loglog(bs, np.abs(meas / exact - 1), "o-", color=CAT[0], ms=4.5,
               lw=1.4, label="integrator vs exact")
    ax1.loglog(bs, np.abs(w1 / exact - 1), color=CAT[1], lw=1.6, linestyle=style.DASH,
               label=r"$4M/b$ vs exact")
    ax1.loglog(bs, np.abs(w3 / exact - 1), color=CAT[2], lw=1.6,
               linestyle=style.DOT, label="3-term series vs exact")
    ax1.set_xlabel("impact parameter  b  [M]")
    ax1.set_ylabel("relative error")
    ax1.legend(loc="upper right")
    ax1.set_ylim(1e-12, 3.0)

    fig.savefig(os.path.join(out, "deflection.png"))
    plt.close(fig)
    return meas, exact, bs


def fig_precession(out):
    """A strongly precessing orbit, and how well the advance is measured."""
    bh = Schwarzschild()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                                   gridspec_kw={"wspace": 0.26})

    # Left: the orbit itself, deep in the strong field so precession is visible.
    y0, E, Lz = orbit_from_apsides(bh, 12.0, 30.0)
    sol = trace(bh, y0, 3200.0, rtol=1e-13, atol=1e-13)
    r, phi = sol.y[1], sol.y[3]
    ax0.plot(r * np.cos(phi), r * np.sin(phi), color=CAT[0], lw=1.3,
             solid_capstyle="round", zorder=3)
    style.draw_hole(ax0, 2.0, r_photon=3.0)
    ax0.set_aspect("equal")
    ax0.set_xlim(-34, 34)
    ax0.set_ylim(-34, 34)
    ax0.set_xlabel("x  [M]")
    ax0.set_ylabel("y  [M]")
    adv = analytic.precession_exact(12.0, 30.0)
    ax0.set_title(f"Precessing orbit, $r_p=12M$, $r_a=30M$\n"
                  f"advance {np.degrees(adv):.1f}$^\\circ$ per orbit")

    # Right: measurement accuracy vs orbit size -- the precision story.
    rps = np.geomspace(8.0, 3e6, 22)
    rel, exact_vals = [], []
    for rp in rps:
        ra = 1.6 * rp
        ex = analytic.precession_exact(rp, ra)
        got = measure.measure_precession(bh, rp, ra)
        exact_vals.append(ex)
        rel.append(abs(got / ex - 1))
    ax1.loglog(rps, rel, "o", color=CAT[0], ms=4.5, lw=1.4, linestyle=style.SOLID,
               label="relative error in measured advance")
    ax1.loglog(rps, 2.5e-16 / np.array(exact_vals), color=CAT[3], lw=1.6,
               linestyle=style.DASH,
               label=r"single-rounding floor: $\epsilon_{64}/\Delta\phi_{\rm prec}$")
    ax1.set_xlabel(r"periapsis  $r_p$  [M]")
    ax1.set_ylabel("relative error")
    ax1.set_title("Measuring the advance gets harder as the orbit weakens")
    ax1.legend(loc="upper left")

    a_merc = analytic.MERCURY["a_sma_m"] / analytic.GM_SUN_OVER_C2
    rp_m = a_merc * (1 - analytic.MERCURY["e"])
    ax1.axvline(rp_m, color=style.INK_MUTED, lw=0.9, linestyle=style.FINEDOT, zorder=1)
    ax1.annotate("Mercury", (rp_m * 0.55, 3e-13), color=style.INK_MUTED,
                 fontsize=8.5, rotation=90)

    fig.savefig(os.path.join(out, "precession.png"))
    plt.close(fig)


def fig_conservation(out):
    """The secular-vs-bounded distinction, and convergence order.

    The left panel is the direct answer to "will RK4 drift like it did in my
    N-body code".  Over a few orbits you cannot tell the schemes apart; the
    difference only appears over ~100 radial periods, where RK4's Carter-constant
    error climbs steadily while the symplectic integrator's stays bounded and
    merely oscillates.  That is the whole practical distinction between
    non-symplectic and symplectic, and it is why the answer depends on what you
    are computing rather than being "always use X".
    """
    bh = KerrBL(a=0.9)
    from kerrgeo import state_from_constants
    y0 = state_from_constants(bh, np.array([0.0, 10.0, np.pi / 3, 0.0]),
                              E=0.985, Lz=3.0, Q=6.25, mu=1.0)
    lam = 60000.0
    h = 1.0
    n = int(lam / h)

    runs = [
        (f"RK4, h = {h}", dict(method="RK4", n_steps=n), CAT[1], style.DASH),
        (f"Gauss-Legendre (symplectic), h = {h}",
         dict(method="GL2", n_steps=n), CAT[2], style.SOLID),
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.7),
                                   gridspec_kw={"wspace": 0.27})

    for label, kw, color, dash in runs:
        sol = trace(bh, y0, lam, **kw)
        stride = max(1, sol.y.shape[1] // 900)
        idx = np.arange(0, sol.y.shape[1], stride)
        q0 = all_invariants(sol.y[:, 0], bh, mu=1.0)["Q"]
        qs = np.array([all_invariants(sol.y[:, i], bh, mu=1.0)["Q"] for i in idx])
        ax0.semilogy(sol.t[idx], np.abs(qs - q0) + 1e-18, color=color, lw=1.5,
                     linestyle=dash, label=label, zorder=3)
    ax0.set_xlabel(r"affine parameter  $\lambda$  [M]   ($\sim$130 radial periods)")
    ax0.set_ylabel(r"$|\Delta Q|$  (Carter constant drift)")
    ax0.set_title("RK4 drifts secularly; the symplectic scheme stays bounded\n"
                  "(E and $L_z$ are exactly zero-drift for both)")
    ax0.legend(loc="lower right")
    ax0.set_ylim(1e-9, 3e-4)

    # Convergence order.
    bhS = Schwarzschild()
    y0s, _, _ = orbit_from_apsides(bhS, 12.0, 24.0)
    L = 400.0
    ref = trace(bhS, y0s, L, rtol=1e-13, atol=1e-13).y[:, -1]
    ns = np.array([50, 100, 200, 400, 800, 1600])
    hs = L / ns
    anchor = None
    for label, method, color, dash in (
        ("RK4", "RK4", CAT[1], style.DASH),
        ("Gauss-Legendre (2-stage)", "GL2", CAT[2], style.SOLID),
    ):
        errs = np.array([np.abs(trace(bhS, y0s, L, method=method,
                                      n_steps=int(k)).y[1:4, -1] - ref[1:4]).max()
                         for k in ns])
        anchor = errs[0] if anchor is None else anchor
        ax1.loglog(hs, errs, "o", color=color, ms=5.0, linestyle=dash, lw=1.5,
                   label=label, zorder=3)
    # Reference slope anchored to the data, offset upward so it reads as a guide.
    ax1.loglog(hs, 6.0 * anchor * (hs / hs[0]) ** 4, color=style.INK_MUTED,
               lw=1.2, linestyle=style.FINEDOT, label=r"$\propto h^4$", zorder=2)
    ax1.set_xlabel("step size  h  [M]")
    ax1.set_ylabel("error in final position  [M]")
    ax1.set_title("Both fixed-step schemes converge at 4th order")
    ax1.legend(loc="lower right")
    ax1.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    fig.savefig(os.path.join(out, "conservation.png"))
    plt.close(fig)


def fig_horizon(out):
    """Evidence that the horizon is a coordinate artefact, handled safely.

    The left panel is plotted at several polar angles for one spin, rather than
    at several spins in the equatorial plane.  That is deliberate: on the
    equator cos(theta) = 0, so Sigma = r^2 and the Kretschmann scalar collapses
    to 48 M^2 / r^6 with *no spin dependence at all* -- three spins would plot
    exactly on top of each other and show nothing.  Varying theta instead shows
    the actual structure: the singularity is a ring in the equatorial plane, so
    only the theta = pi/2 curve diverges as r -> 0.  Every curve is finite at
    r_+ (markers), which is the point about the horizon.

    The downward spikes on the off-equatorial curves are not artefacts: the
    Kretschmann scalar genuinely changes sign away from the equator, and we
    plot its absolute value on a log axis, so each sign change shows as a notch.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"wspace": 0.28})

    m = KerrBL(a=0.9)
    r = np.geomspace(0.02, 30.0, 1400)
    angles = [(np.pi / 2, r"$\theta=\pi/2$ (through the ring)", CAT[3], style.SOLID),
              (np.pi / 3, r"$\theta=\pi/3$", CAT[0], style.DASH),
              (np.pi / 4, r"$\theta=\pi/4$", CAT[2], style.DOT),
              (0.0, r"$\theta=0$ (spin axis)", CAT[1], style.DASHDOT)]
    for th, label, color, dash in angles:
        ax0.loglog(r, np.abs(m.kretschmann(r, th)), color=color, lw=1.7,
                   linestyle=dash, label=label, zorder=3)
        ax0.plot([m.r_plus], [abs(m.kretschmann(m.r_plus, th))], "o",
                 color=color, ms=7.0, zorder=6,
                 markeredgecolor=style.SURFACE, markeredgewidth=1.8)
    ax0.axvline(m.r_plus, color=style.INK_MUTED, lw=0.9,
                linestyle=style.FINEDOT, zorder=1)
    ax0.annotate(r"$r_+$", (m.r_plus * 1.12, 1e-6), color=style.INK_MUTED,
                 fontsize=9)
    ax0.set_xlabel("r  [M]")
    ax0.set_ylabel(r"$|R_{abcd}R^{abcd}|$  [$M^{-4}$]")
    ax0.set_title("Kerr, a = 0.9M: curvature is finite at $r_+$ (markers)\n"
                  "and diverges only on the equatorial ring")
    ax0.legend(loc="upper right")
    ax0.set_ylim(1e-8, 1e14)

    # The cutoff epsilon does not contaminate physical answers.
    bh = Schwarzschild()
    eps = np.geomspace(1e-9, 1e-2, 15)
    ref = analytic.deflection_exact(6.0)
    vals = [abs(measure.measure_deflection(bh, 6.0, r0=1e4, horizon_eps=e) / ref - 1)
            for e in eps]
    ax1.loglog(eps, np.maximum(vals, 1e-16), "o", color=CAT[0], ms=5.0,
               linestyle=style.SOLID, lw=1.5,
               label=r"deflection error at $b=6M$")
    ax1.set_xlabel(r"horizon cutoff  $\epsilon$   ($r_{\rm stop}=(1+\epsilon)r_+$)")
    ax1.set_ylabel("relative error in deflection")
    ax1.set_title("Seven decades of cutoff, no effect on the answer:\n"
                  "an escaping ray never goes near $r_+$")
    ax1.legend(loc="upper left")
    ax1.set_ylim(1e-14, 1e-8)

    fig.savefig(os.path.join(out, "horizon.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 2: Kerr
# ---------------------------------------------------------------------------

def fig_frame_dragging(out):
    """Frame dragging three ways: ZAMO spirals, a retrograde photon forced to
    reverse, and the drag rate omega(r) against its integrated measurement."""
    from kerrgeo import rhs, zamo_drop_state
    from kerrgeo.measure import measure_phi_turnaround

    m = KerrBL(a=0.9)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 5.0),
                                   gridspec_kw={"wspace": 0.26})

    # -- left: trajectories -------------------------------------------------
    # Three ZAMO drops (Lz = 0), released 120 degrees apart: each spirals
    # prograde purely because spacetime rotates under it.
    for k in range(3):
        y0 = zamo_drop_state(m, 6.0)
        y0[3] = 2.0 * np.pi * k / 3.0
        sol = trace(m, y0, 400.0, rtol=1e-12, atol=1e-12,
                    events=[horizon_event(m)])
        r, ph = sol.y[1], sol.y[3]
        ax0.plot(r * np.cos(ph), r * np.sin(ph), color=CAT[0], lw=1.6,
                 solid_capstyle="round", zorder=3,
                 label="ZAMO drops ($L_z=0$)" if k == 0 else None)

    # A retrograde photon (b = -3): sweeps backwards, is reversed at
    # r_flip = 2M(1 + a/|b|), crosses the horizon corotating.
    b = -3.0
    y0 = photon_from_impact_parameter(m, r0=30.0, b=b)
    sol = trace(m, y0, 300.0, rtol=1e-12, atol=1e-12,
                events=[horizon_event(m)])
    r, ph = sol.y[1], sol.y[3]
    ax0.plot(r * np.cos(ph), r * np.sin(ph), color=CAT[1], lw=1.8,
             linestyle=style.DASH, zorder=4, label=f"retrograde photon, $b=-3M$")
    r_flip = 2.0 * (1.0 + 0.9 / abs(b))
    ax0.add_patch(plt.Circle((0, 0), r_flip, fill=False, lw=1.0,
                             edgecolor=CAT[1], linestyle=(0, (1.5, 2.5)),
                             zorder=2))
    ax0.annotate(r"$r_{\rm flip}=2M(1+a/|b|)$", (r_flip * 0.05, -r_flip - 0.75),
                 color=CAT[1], fontsize=8.5, ha="center")

    # Ergosphere drawn muted so the orange flip circle (which belongs to the
    # photon's story) is the only orange ring.
    style.draw_hole(ax0, m.r_plus)
    ax0.add_patch(plt.Circle((0, 0), m.r_ergo(np.pi / 2), fill=False, lw=1.1,
                             edgecolor=style.INK_MUTED,
                             linestyle=(0, (4, 3)), zorder=4))
    ax0.annotate("static limit", (-2.6, 2.2), color=style.INK_MUTED,
                 fontsize=8.5, ha="right")
    ax0.set_aspect("equal")
    ax0.set_xlim(-8.5, 8.5)
    ax0.set_ylim(-8.5, 8.5)
    ax0.set_xlabel("x  [M]")
    ax0.set_ylabel("y  [M]")
    ax0.set_title("Frame dragging, a = 0.9M")
    ax0.legend(loc="upper left")

    # -- right: the drag rate, measured vs closed form ----------------------
    rr = np.geomspace(m.r_plus * 1.001, 60.0, 400)
    ax1.loglog(rr, m.omega(rr), color=style.INK, lw=1.8, zorder=3,
               label=r"$\omega = 2aMr/A$ (closed form)")
    ax1.loglog(rr, 1.8 / rr**3, color=CAT[2], lw=1.5, linestyle=style.DOT,
               zorder=2, label=r"$2aM/r^3$ (gravitomagnetic dipole)")

    y0 = zamo_drop_state(m, 40.0)
    sol = trace(m, y0, 3000.0, rtol=1e-12, atol=1e-12,
                events=[horizon_event(m)])
    idx = np.unique(np.geomspace(1, sol.y.shape[1] - 1, 26).astype(int))
    r_s = sol.y[1, idx]
    om_s = np.array([rhs(0.0, sol.y[:, i], m)[3] / rhs(0.0, sol.y[:, i], m)[0]
                     for i in idx])
    ax1.loglog(r_s, om_s, "o", color=CAT[0], ms=5.0, zorder=4,
               label=r"$d\phi/dt$ of an infalling ZAMO")

    ax1.plot([m.r_plus], [m.Omega_H], "o", color=CAT[3], ms=7.0, zorder=5,
             markeredgecolor=style.SURFACE, markeredgewidth=1.6)
    ax1.annotate(r"$\Omega_H = a/2Mr_+$", (m.r_plus * 1.1, m.Omega_H * 1.12),
                 color=CAT[3], fontsize=8.5)
    ax1.set_xlabel("r  [M]")
    ax1.set_ylabel(r"angular velocity  [$M^{-1}$]")
    ax1.set_title(r"The drag rate $\omega(r)$, measured vs closed form")
    ax1.legend(loc="lower left")
    ax1.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    fig.savefig(os.path.join(out, "frame_dragging.png"))
    plt.close(fig)


def fig_kerr_asymmetry(out):
    """How the landmark radii and capture thresholds split with spin."""
    from kerrgeo.analytic import kerr_critical_impact_parameter
    from kerrgeo.measure import capture_threshold

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                                   gridspec_kw={"wspace": 0.24})

    a_grid = np.linspace(0.0, 1.0, 400)
    mets = [KerrBL(a=float(a)) for a in a_grid]

    # Labels placed mid-curve where the curves are well separated -- at a = 1
    # the horizon, prograde ISCO and prograde photon orbit all converge on
    # r = M and edge labels would pile up unreadably.
    curves = [
        ("ISCO retrograde", [m.r_isco(False) for m in mets],
         CAT[1], style.DASH, 0.42, 8.0, 17),
        ("ISCO prograde", [m.r_isco(True) for m in mets],
         CAT[0], style.SOLID, 0.13, 5.15, -16),
        ("photon retrograde", [m.r_photon(False) for m in mets],
         CAT[3], style.DASHDOT, 0.56, 3.95, 6),
        ("photon prograde", [m.r_photon(True) for m in mets],
         CAT[2], style.DOT, 0.24, 2.42, -8),
        ("horizon $r_+$", [m.r_plus for m in mets],
         style.INK_MUTED, style.SOLID, 0.13, 1.62, -3),
    ]
    for label, vals, color, ls, lx, ly, rot in curves:
        ax0.plot(a_grid, vals, color=color, lw=1.7, linestyle=ls, zorder=3)
        ax0.annotate(label, (lx, ly), color=color, fontsize=8.5,
                     rotation=rot, rotation_mode="anchor")
    ax0.axhline(2.0, color=CAT[4], lw=1.1, linestyle=style.FINEDOT, zorder=2)
    ax0.annotate("static limit (equator)", (0.63, 2.1), color=CAT[4],
                 fontsize=8.5)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 9.5)
    ax0.set_xlabel("spin  a/M")
    ax0.set_ylabel("r  [M]")
    ax0.set_title("Landmark radii vs spin")

    # -- right: capture thresholds, measured vs Bardeen ---------------------
    bc_pro = [kerr_critical_impact_parameter(float(a), True) for a in a_grid]
    bc_ret = [-kerr_critical_impact_parameter(float(a), False) for a in a_grid]
    ax1.plot(a_grid, bc_ret, color=CAT[1], lw=1.7, linestyle=style.DASH,
             zorder=3, label="retrograde $|b_c|$ (Bardeen)")
    ax1.plot(a_grid, bc_pro, color=CAT[0], lw=1.7, zorder=3,
             label="prograde $b_c$ (Bardeen)")

    a_pts = (0.3, 0.6, 0.9)
    for pro, color in ((True, CAT[0]), (False, CAT[1])):
        pts = []
        for a in a_pts:
            m = KerrBL(a=a)
            guess = abs(kerr_critical_impact_parameter(a, pro))
            pts.append(capture_threshold(m, guess * 0.9, guess * 1.1,
                                         r0=500.0, tol=1e-5, prograde=pro))
        ax1.plot(a_pts, pts, "o", color=color, ms=6.5, zorder=4,
                 markeredgecolor=style.SURFACE, markeredgewidth=1.4,
                 label=("integrated (bisection)" if pro else None))

    ax1.annotate(r"$3\sqrt{3}\,M$ at $a=0$", (0.03, 4.5),
                 color=style.INK_MUTED, fontsize=8.5)
    ax1.annotate("7M", (0.93, 7.15), color=CAT[1], fontsize=8.5)
    ax1.annotate("2M", (0.93, 2.25), color=CAT[0], fontsize=8.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 7.6)
    ax1.set_xlabel("spin  a/M")
    ax1.set_ylabel(r"capture threshold  $|b_c|$  [M]")
    ax1.set_title("Photon capture thresholds vs spin")
    ax1.legend(loc="lower left")

    fig.savefig(os.path.join(out, "kerr_asymmetry.png"))
    plt.close(fig)


def fig_spherical_photon(out):
    """A spherical photon orbit: winds on a sphere, then the instability
    that makes the shadow edge takes over."""
    from scipy.optimize import brentq

    from kerrgeo import spherical_photon_orbit
    from kerrgeo.analytic import kerr_photon_orbit_constants
    from kerrgeo.separated import polar_potential

    m = KerrBL(a=0.9)
    r0 = 2.6
    xi, eta = kerr_photon_orbit_constants(r0, 0.9)
    y0 = spherical_photon_orbit(m, r0)
    # Terminate once the instability has fully taken over: after departure the
    # photon plunges, and integrating the final approach to r_+ is pure cost.
    sol = trace(m, y0, 60.0, rtol=1e-13, atol=1e-13,
                events=[horizon_event(m, eps=1e-3), escape_event(10.0)])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.2, 4.4),
                                        gridspec_kw={"wspace": 0.3})

    # -- x-y projection while it still holds the sphere ---------------------
    hold = sol.t <= 42.0
    r, th, ph = sol.y[1, hold], sol.y[2, hold], sol.y[3, hold]
    x, y = r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph)
    ax0.plot(x, y, color=CAT[0], lw=1.3, solid_capstyle="round", zorder=3)
    style.draw_hole(ax0, m.r_plus, r_ergo=m.r_ergo(np.pi / 2))
    ax0.set_aspect("equal")
    ax0.set_xlim(-3.1, 3.1)
    ax0.set_ylim(-3.1, 3.1)
    ax0.set_xlabel("x  [M]")
    ax0.set_ylabel("y  [M]")
    ax0.set_title(f"Spherical photon orbit, r = {r0}M, a = 0.9M\n"
                  f"$\\xi = {xi:.3f}$, $\\eta = {eta:.2f}$")

    # -- the winding on the sphere ------------------------------------------
    th_turn = brentq(lambda t: polar_potential(t, 0.9, 1.0, xi, eta, 0.0),
                     1e-3, np.pi / 2 - 1e-6)
    ax1.plot(np.degrees(ph), np.degrees(th), color=CAT[0], lw=1.3, zorder=3)
    for t in (th_turn, np.pi - th_turn):
        ax1.axhline(np.degrees(t), color=CAT[3], lw=1.1,
                    linestyle=style.DASH, zorder=2)
    ax1.annotate(r"polar turning points: $\Theta(\theta)=0$",
                 (np.degrees(ph).min() + 6, np.degrees(th_turn) - 5),
                 color=CAT[3], fontsize=8.5)
    ax1.set_xlabel(r"$\phi$  [deg]")
    ax1.set_ylabel(r"$\theta$  [deg]")
    ax1.invert_yaxis()
    ax1.set_title("Winds between the latitudes the\npolar potential allows")

    # -- the instability ------------------------------------------------------
    drift = np.abs(sol.y[1] - r0) + 1e-16
    ax2.semilogy(sol.t, drift, color=CAT[0], lw=1.5, zorder=3)
    lam_fit = np.array([15.0, 38.0])
    d0 = np.interp(lam_fit[0], sol.t, drift)
    kappa = np.log(np.interp(lam_fit[1], sol.t, drift) / d0) / np.ptp(lam_fit)
    ax2.semilogy(lam_fit, d0 * np.exp(kappa * (lam_fit - lam_fit[0])),
                 color=style.INK_MUTED, lw=1.2, linestyle=style.FINEDOT,
                 zorder=2)
    ax2.annotate(f"$e$-folds every {1 / kappa:.1f} M",
                 (lam_fit[0] + 1.5, d0 * 8), color=style.INK_MUTED,
                 fontsize=8.5, rotation=28)
    ax2.set_xlabel(r"affine parameter  $\lambda$  [M]")
    ax2.set_ylabel(r"$|r - r_0|$  [M]")
    ax2.set_title("Unstable by design: rounding error\n$e$-folds until the photon departs")

    fig.savefig(os.path.join(out, "spherical_photon.png"))
    plt.close(fig)


def fig_kerr_precession(out):
    """The two Kerr precessions against the Wilkins frequencies."""
    from kerrgeo.analytic import kerr_circular_frequencies
    from kerrgeo.measure import measure_nodal_precession, measure_precession

    m = KerrBL(a=0.9)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.7),
                                   gridspec_kw={"wspace": 0.27})

    # -- nodal (Lense-Thirring) ----------------------------------------------
    rr = np.geomspace(6.0, 120.0, 200)
    adv_exact = np.array([
        2 * np.pi * (kerr_circular_frequencies(r, 0.9, True)[0]
                     / kerr_circular_frequencies(r, 0.9, True)[1] - 1)
        for r in rr])
    ax0.loglog(rr, adv_exact, color=style.INK, lw=1.8, zorder=3,
               label="exact (Wilkins frequencies)")
    ax0.loglog(rr, 4 * np.pi * 0.9 / rr**1.5, color=CAT[2], lw=1.5,
               linestyle=style.DOT, zorder=2,
               label=r"weak field: $4\pi a\, r^{-3/2}$")
    r_pts = np.array([6.0, 9.0, 14.0, 22.0, 35.0, 60.0, 100.0])
    adv_meas = [measure_nodal_precession(m, r, Q=1e-6)[1] for r in r_pts]
    ax0.loglog(r_pts, adv_meas, "o", color=CAT[0], ms=5.5, zorder=4,
               markeredgecolor=style.SURFACE, markeredgewidth=1.2,
               label="integrated orbits")
    ax0.set_xlabel("orbit radius  r  [M]")
    ax0.set_ylabel("node advance per polar period  [rad]")
    ax0.set_title("Lense-Thirring nodal precession, a = 0.9M\n"
                  "the orbital plane itself is dragged around the spin axis")
    ax0.legend(loc="lower left")

    # -- periapsis, prograde vs retrograde -----------------------------------
    rr = np.geomspace(10.5, 120.0, 200)
    for pro, color, ls, label in ((True, CAT[0], style.SOLID, "prograde"),
                                  (False, CAT[1], style.DASH, "retrograde")):
        exact = []
        for r in rr:
            Om_phi, _, Om_r = kerr_circular_frequencies(r, 0.9, pro)
            exact.append(2 * np.pi * (abs(Om_phi) / Om_r - 1))
        ax1.loglog(rr, exact, color=color, lw=1.7, linestyle=ls, zorder=3,
                   label=f"{label} (exact)")
        r_pts = np.array([10.5, 16.0, 25.0, 45.0, 80.0])
        meas = [measure_precession(m, r * 0.99, r * 1.01, prograde=pro)
                for r in r_pts]
        ax1.loglog(r_pts, meas, "o", color=color, ms=5.5, zorder=4,
                   markeredgecolor=style.SURFACE, markeredgewidth=1.2)
    ax1.set_xlabel("orbit radius  r  [M]")
    ax1.set_ylabel("periapsis advance per orbit  [rad]")
    ax1.set_title("Periapsis advance splits with orbit direction\n"
                  "same radius, same hole -- different precession")
    ax1.legend(loc="lower left")

    fig.savefig(os.path.join(out, "kerr_precession.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 3: the interior
# ---------------------------------------------------------------------------

def fig_horizon_crossing(out):
    """The chart handoff: BL time diverges at r_+, ingoing v sails through --
    and the trajectory continues through both horizons to where the physics
    actually ends."""
    from kerrgeo import (KerrIngoing, bl_to_ingoing, principal_null_ingoing,
                         zamo_drop_state)
    from kerrgeo.events import negative_r_escape_event, ring_event
    from kerrgeo.invariants import drift_report

    a = 0.9
    mi = KerrIngoing(a=a)
    mb = KerrBL(a=a)
    rp, rm = mi.r_plus, mi.r_minus

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                                   gridspec_kw={"wspace": 0.25})

    # -- left: the same infall in both time coordinates ----------------------
    y0_bl = zamo_drop_state(mb, 8.0)
    sol_bl = trace(mb, y0_bl, 200.0, rtol=1e-12, atol=1e-12,
                   events=[horizon_event(mb, eps=1e-7)])
    y0_in = bl_to_ingoing(mi, y0_bl)
    sol_in = trace(mi, y0_in, 200.0, rtol=1e-12, atol=1e-12,
                   events=[ring_event(mi, eps=1e-3)])

    # Offset both clocks to zero at release so the curves are comparable.
    ax0.plot(sol_bl.y[1], sol_bl.y[0] - sol_bl.y[0, 0], color=CAT[1],
             lw=1.8, linestyle=style.DASH, zorder=3,
             label="Boyer-Lindquist  t")
    ax0.plot(sol_in.y[1], sol_in.y[0] - sol_in.y[0, 0], color=CAT[0],
             lw=1.8, zorder=4, label="ingoing Kerr  v")
    for rx, name in ((rp, r"$r_+$"), (rm, r"$r_-$")):
        ax0.axvline(rx, color=style.INK_MUTED, lw=0.9,
                    linestyle=style.FINEDOT, zorder=1)
        ax0.annotate(name, (rx + 0.08, 105), color=style.INK_MUTED,
                     fontsize=9)
    ax0.plot([sol_in.y[1, -1]], [sol_in.y[0, -1] - sol_in.y[0, 0]], "o",
             color=CAT[3], ms=7, zorder=5,
             markeredgecolor=style.SURFACE, markeredgewidth=1.5)
    ax0.annotate("ring singularity\n(the genuine end)",
                 (0.35, sol_in.y[0, -1] - sol_in.y[0, 0] - 4), color=CAT[3],
                 fontsize=8.5)
    ax0.set_xlim(0, 8.2)
    ax0.set_ylim(0, 120)
    ax0.invert_xaxis()                       # infall reads left to right
    ax0.set_xlabel("r  [M]")
    ax0.set_ylabel("time coordinate  [M]")
    ax0.set_title("One infalling particle, two clocks")
    ax0.legend(loc="upper left")

    # -- right: r(lambda) through everything ---------------------------------
    ax1.plot(sol_in.t, sol_in.y[1], color=CAT[0], lw=1.8, zorder=4,
             label="equatorial infall (timelike)")
    y0p = principal_null_ingoing(mi, 8.0, np.radians(80))
    solp = trace(mi, y0p, 12.0, rtol=1e-13, atol=1e-13,
                 events=[negative_r_escape_event(-3.0)])
    ax1.plot(solp.t, solp.y[1], color=CAT[2], lw=1.8, linestyle=style.DOT,
             zorder=3, label=r"vortical photon ($Q<0$, through the disk)")
    for ry, name in ((rp, r"$r_+$"), (rm, r"$r_-$"), (0.0, "disk  r = 0")):
        ax1.axhline(ry, color=style.INK_MUTED, lw=0.9,
                    linestyle=style.FINEDOT, zorder=1)
        ax1.annotate(name, (0.4, ry + 0.12), color=style.INK_MUTED,
                     fontsize=8.5)
    d = drift_report(sol_in.y[:, sol_in.y[1] > rm * 0.99], mi, mu=1.0)
    ax1.annotate("through both horizons:\n"
                 rf"$|\Delta$norm$| < ${d['norm']:.0e},  "
                 r"$\Delta E = \Delta L_z = 0$",
                 (12.5, 5.6), fontsize=8.5, color=style.INK_MUTED)
    ax1.set_xlim(0, 26)
    ax1.set_ylim(-3.2, 8.2)
    ax1.set_xlabel(r"affine parameter  $\lambda$  [M]")
    ax1.set_ylabel("r  [M]")
    ax1.set_title("r($\\lambda$) is smooth through both horizons")
    ax1.legend(loc="upper right")

    fig.savefig(os.path.join(out, "horizon_crossing.png"))
    plt.close(fig)


def fig_interior_map(out):
    """The causal map of the interior, and the CTC loop proper period."""
    from kerrgeo import KerrIngoing

    a = 0.9
    mi = KerrIngoing(a=a)
    rp, rm = mi.r_plus, mi.r_minus

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                                   gridspec_kw={"wspace": 0.26,
                                                "width_ratios": [1.35, 1.0]})

    # -- left: (r, theta) causal map -----------------------------------------
    rr = np.linspace(-1.25, 3.0, 700)
    th = np.linspace(0.02, np.pi - 0.02, 500)
    Rg, Tg = np.meshgrid(rr, th)
    gpp = mi.g_phiphi(Rg, Tg)

    # CTC region: a filled patch, labelled directly.
    ax0.contourf(Rg, np.degrees(Tg), (gpp < 0).astype(float),
                 levels=[0.5, 1.5], colors=[CAT[3]], alpha=0.30, zorder=2)
    ax0.contour(Rg, np.degrees(Tg), gpp, levels=[0.0], colors=[CAT[3]],
                linewidths=1.4, zorder=3)
    ax0.annotate("closed timelike curves\n$g_{\\phi\\phi}<0$", (-0.72, 118),
                 color=CAT[3], fontsize=8.5, ha="center", zorder=6)

    # Ergosphere boundary r_E(theta) and the static limit shading direction.
    th_line = np.linspace(0.02, np.pi - 0.02, 300)
    ax0.plot(mi.M + np.sqrt(mi.M**2 - a * a * np.cos(th_line) ** 2),
             np.degrees(th_line), color=CAT[1], lw=1.4,
             linestyle=style.DASH, zorder=3)
    ax0.annotate("ergosphere", (2.15, 96), color=CAT[1], fontsize=8.5)

    for rx, name in ((rp, "$r_+$"), (rm, "$r_-$")):
        ax0.axvline(rx, color=style.INK_MUTED, lw=1.1, zorder=3)
        ax0.annotate(name, (rx + 0.04, 6), color=style.INK_MUTED, fontsize=9)
    ax0.axvline(0.0, color=style.INK, lw=1.0, linestyle=style.FINEDOT,
                zorder=3)
    ax0.annotate("disk r = 0", (0.03, 168), color=style.INK, fontsize=8.5,
                 rotation=90)
    ax0.plot([0.0], [90.0], "o", color=style.INK, ms=8, zorder=6,
             markeredgecolor=style.SURFACE, markeredgewidth=1.6)
    ax0.annotate("ring singularity", (0.09, 86), color=style.INK, fontsize=8.5,
                 zorder=6)

    # The two stage-3 trajectories, in this plane.
    ax0.plot([3.0, -1.25], [80, 80], color=CAT[2], lw=1.8,
             linestyle=style.DOT, zorder=5)
    ax0.annotate("vortical photon", (1.62, 74), color=CAT[2], fontsize=8.5)
    ax0.plot([3.0, 0.02], [90, 90], color=CAT[0], lw=1.8, zorder=5,
             solid_capstyle="round")
    ax0.annotate("equatorial infall", (1.35, 94), color=CAT[0], fontsize=8.5)

    ax0.set_xlabel("r  [M]")
    ax0.set_ylabel(r"$\theta$  [deg]")
    ax0.set_ylim(180, 0)
    ax0.set_title("Causal map of the interior, a = 0.9M")

    # -- right: proper period of the closed loops ----------------------------
    rr = np.linspace(-1.05, -0.005, 600)
    for th_deg, color, ls in ((90, CAT[3], style.SOLID),
                              (80, CAT[0], style.DASH),
                              (70, CAT[2], style.DOT)):
        tau = mi.ctc_loop_proper_time(rr, np.radians(th_deg))
        ax1.plot(rr, tau, color=color, lw=1.7, linestyle=ls,
                 label=rf"$\theta = {th_deg}^\circ$")
    ax1.set_xlabel("r  [M]")
    ax1.set_ylabel(r"proper time per loop  $2\pi\sqrt{-g_{\phi\phi}}$  [M]")
    ax1.set_title("Proper time to ride a closed loop once")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 6)

    fig.savefig(os.path.join(out, "interior_map.png"))
    plt.close(fig)


ALL_FIGURES = (
    ("photon_trajectories", fig_photon_trajectories),
    ("deflection", fig_deflection),
    ("precession", fig_precession),
    ("conservation", fig_conservation),
    ("horizon", fig_horizon),
    ("frame_dragging", fig_frame_dragging),
    ("kerr_asymmetry", fig_kerr_asymmetry),
    ("spherical_photon", fig_spherical_photon),
    ("kerr_precession", fig_kerr_precession),
    ("horizon_crossing", fig_horizon_crossing),
    ("interior_map", fig_interior_map),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "figures"))
    ap.add_argument("--only", default=None,
                    help="comma-separated figure names to regenerate")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    wanted = set(args.only.split(",")) if args.only else None
    for name, fn in ALL_FIGURES:
        if wanted is not None and name not in wanted:
            continue
        t0 = time.time()
        fn(out)
        print(f"  {name:22s} {time.time() - t0:6.1f}s")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()



