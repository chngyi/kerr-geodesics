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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "figures"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    for name, fn in (
        ("photon_trajectories", fig_photon_trajectories),
        ("deflection", fig_deflection),
        ("precession", fig_precession),
        ("conservation", fig_conservation),
        ("horizon", fig_horizon),
    ):
        t0 = time.time()
        fn(out)
        print(f"  {name:22s} {time.time() - t0:6.1f}s")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()


