"""Run every quantitative validation check and write a report.

    python scripts/run_validation.py            # print to stdout
    python scripts/run_validation.py --md VALIDATION.md

The test suite asserts these same facts pass/fail; this script reports the
actual numbers, which is what belongs in a write-up.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kerrgeo import (  # noqa: E402
    KerrBL,
    Schwarzschild,
    analytic,
    circular_orbit,
    drift_report,
    measure,
    orbit_from_apsides,
    photon_from_impact_parameter,
    separated,
    state_from_constants,
    trace,
)
from kerrgeo.events import escape_event, horizon_event  # noqa: E402

ROWS: list[tuple] = []


def row(section, quantity, computed, reference, note=""):
    if reference is None or reference == 0:
        err = ""
    else:
        err = f"{abs(computed / reference - 1):.2e}"
    ROWS.append((section, quantity, computed, reference, err, note))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None, help="also write a markdown report here")
    args = ap.parse_args()
    t_start = time.time()

    # -- closed-form landmarks ---------------------------------------------
    for a in (0.0, 0.5, 0.9, 1.0):
        m = KerrBL(a=a)
        if a == 0.0:
            row("Landmarks", "Schwarzschild horizon r_+", m.r_plus, 2.0)
            row("Landmarks", "Schwarzschild photon sphere", m.r_photon(), 3.0)
            row("Landmarks", "Schwarzschild ISCO", m.r_isco(), 6.0)
        if a == 1.0:
            row("Landmarks", "extremal r_+", m.r_plus, 1.0)
            row("Landmarks", "extremal ISCO (prograde)", m.r_isco(True), 1.0)
            row("Landmarks", "extremal ISCO (retrograde)", m.r_isco(False), 9.0)
            row("Landmarks", "extremal photon orbit (retro)", m.r_photon(False), 4.0)
        row("Landmarks", f"ergosphere at equator, a={a}", m.r_ergo(np.pi / 2), 2.0)

    row("Landmarks", "critical impact parameter b_c",
        analytic.critical_impact_parameter(), 3 * np.sqrt(3))

    # -- photon capture threshold, by bisection ----------------------------
    bh = Schwarzschild()
    bc = measure.capture_threshold(bh, 5.0, 5.5, r0=1e4, tol=1e-9)
    row("Strong field", "capture threshold (bisection)", bc, 3 * np.sqrt(3),
        "integrated, not assumed")

    # -- light deflection ---------------------------------------------------
    for b in (5.5, 6.0, 10.0, 50.0, 500.0):
        got = measure.measure_deflection(bh, b, r0=max(1e5, 400 * b))
        row("Deflection", f"alpha at b = {b:g} M", got,
            analytic.deflection_exact(b), "vs exact quadrature")

    b = 400.0
    got = measure.measure_deflection(bh, b, r0=1e6)
    second = got - 4.0 / b - (128.0 / 3.0) / b**3
    row("Deflection", "2nd-order coefficient  15*pi/4",
        second * b**2, 15 * np.pi / 4, "isolated from the integrated result")

    # -- precession ---------------------------------------------------------
    for rp, ra in ((10.0, 20.0), (30.0, 60.0), (100.0, 200.0), (1e4, 1.5e4)):
        got = measure.measure_precession(bh, rp, ra)
        row("Precession", f"advance, r_p={rp:g} r_a={ra:g}", got,
            analytic.precession_exact(rp, ra), "rad/orbit, vs exact")

    row("Precession", "Mercury (from weak-field formula)",
        analytic.mercury_precession_arcsec_per_century(), 42.98,
        "arcsec/century, observed 42.98")

    a_merc = analytic.MERCURY["a_sma_m"] / analytic.GM_SUN_OVER_C2
    rp_m, ra_m = analytic.apsides_from_elements(a_merc, analytic.MERCURY["e"])
    conv = (36525.0 / analytic.MERCURY["period_days"]) * (180 / np.pi) * 3600
    got = measure.measure_precession(bh, rp_m, ra_m, rtol=1e-13, atol=1e-14)
    row("Precession", "Mercury (integrated geodesic)", got * conv, 42.98,
        "arcsec/century -- limited by float64, see README")

    # -- conservation -------------------------------------------------------
    mk = KerrBL(a=0.9)
    y0 = state_from_constants(mk, np.array([0.0, 10.0, np.pi / 3, 0.0]),
                              E=0.985, Lz=3.0, Q=6.25, mu=1.0)
    for label, kw in (("DOP853 rtol=1e-12", dict(rtol=1e-12, atol=1e-12)),
                      ("RK4 h=0.5", dict(method="RK4", n_steps=6000)),
                      ("GL2 h=0.5", dict(method="GL2", n_steps=6000))):
        sol = trace(mk, y0, 3000.0, **kw)
        d = drift_report(sol.y, mk, mu=1.0)
        for k in ("E", "Lz", "norm", "Q"):
            row("Conservation", f"a=0.9 inclined orbit, 3000 M -- {label}: max drift in {k}", d[k], None)

    # -- circular orbit stability ------------------------------------------
    y0c, E, Lz = circular_orbit(KerrBL(a=0.5), 10.0)
    solc = trace(KerrBL(a=0.5), y0c, 2000.0, rtol=1e-12, atol=1e-12)
    row("Conservation", "a=0.5 circular orbit at r=10M, 2000 M: peak-to-peak r",
        float(np.ptp(solc.y[1])), None, "should be pure integration error")

    # -- formulation cross-check -------------------------------------------
    E, Lz, Q = 0.95, 2.8, 3.0
    x0 = np.array([0.0, 12.0, np.pi / 3, 0.0])
    yH = state_from_constants(mk, x0, E, Lz, Q, mu=1.0, sign_r=-1, sign_theta=1)
    solH = trace(mk, yH, 400.0, rtol=1e-13, atol=1e-13)
    sep = separated.trace_separated(mk, x0, E, Lz, Q, mu=1.0,
                                    lam_max=25.0, n_out=20000)
    tau = np.linspace(1.0, 390.0, 500)
    for idx, key in ((1, "r"), (2, "theta"), (3, "phi"), (0, "t")):
        diff = np.abs(CubicSpline(solH.t, solH.y[idx])(tau)
                      - CubicSpline(sep["tau"], sep[key])(tau)).max()
        row("Hamiltonian vs separated Carter", f"max deviation in {key} over 390 M",
            float(diff), None, "two independent formulations")

    # -- reversibility ------------------------------------------------------
    yp = photon_from_impact_parameter(KerrBL(a=0.6), r0=80.0, b=6.5)
    row("Reversibility", "photon, forward+back over 120 M",
        measure.reversibility_error(KerrBL(a=0.6), yp, 120.0,
                                    rtol=1e-13, atol=1e-13), None)

    # -- horizon cutoff insensitivity --------------------------------------
    ref = analytic.deflection_exact(6.0)
    vals = [measure.measure_deflection(bh, 6.0, r0=1e4, horizon_eps=e)
            for e in (1e-9, 1e-6, 1e-3)]
    row("Horizon handling", "spread in alpha(b=6M) across eps 1e-9 to 1e-3",
        float(np.ptp(vals)), None, "cutoff has no effect on escaping rays")

    m9 = KerrBL(a=0.9)
    row("Horizon handling", "Kretschmann at r_+ (finite)",
        float(m9.kretschmann(m9.r_plus, np.pi / 3)), None, "coordinate artefact")
    row("Horizon handling", "Kretschmann at r=1e-4 on ring (divergent)",
        float(m9.kretschmann(1e-4, np.pi / 2)), None, "physical singularity")

    # ======================================================================
    # Stage 2: Kerr physics
    # ======================================================================

    m9 = KerrBL(a=0.9)

    # -- capture thresholds: the frame-dragging asymmetry --------------------
    for pro in (True, False):
        bc_an = analytic.kerr_critical_impact_parameter(0.9, pro)
        bc = measure.capture_threshold(m9, abs(bc_an) * 0.9, abs(bc_an) * 1.1,
                                       r0=500.0, tol=1e-6, prograde=pro)
        row("Kerr capture thresholds (a=0.9)",
            f"|b_c| {'prograde' if pro else 'retrograde'} (bisection)",
            bc, abs(bc_an), "vs Bardeen closed form")

    d_pro = measure.measure_deflection(m9, +7.0, r0=1e5)
    d_ret = measure.measure_deflection(m9, -7.0, r0=1e5)
    row("Kerr capture thresholds (a=0.9)", "deflection ratio at |b|=7M",
        d_ret / d_pro, None, "retrograde vs prograde: same |b|, 3.2x the bend")

    # -- frame dragging -------------------------------------------------------
    from kerrgeo import rhs, zamo_drop_state

    y0 = zamo_drop_state(m9, 8.0)
    sol = trace(m9, y0, 200.0, rtol=1e-12, atol=1e-12,
                events=[horizon_event(m9)])
    worst = max(abs(rhs(0.0, sol.y[:, i], m9)[3] / rhs(0.0, sol.y[:, i], m9)[0]
                    - m9.omega(sol.y[1, i]))
                for i in range(0, sol.y.shape[1], 5))
    row("Frame dragging (a=0.9)", "ZAMO infall: max |dphi/dt - omega(r)|",
        worst, None, "Lz=0 particle corotates at exactly the drag rate")
    d_end = rhs(0.0, sol.y[:, -1], m9)
    row("Frame dragging (a=0.9)", "dphi/dt at the horizon", d_end[3] / d_end[0],
        m9.Omega_H, "everything crosses corotating at Omega_H")

    r_flip = measure.measure_phi_turnaround(m9, -3.0)
    row("Frame dragging (a=0.9)", "retrograde photon phi-reversal radius, b=-3M",
        r_flip, 2.0 * (1.0 + 0.9 / 3.0), "vs 2M(1 + a/|b|)")

    # -- orbital frequencies and the two precessions -------------------------
    for pro in (True, False):
        got = measure.measure_orbital_frequency(m9, 10.0, pro)
        want = analytic.kerr_circular_frequencies(10.0, 0.9, pro)[0]
        row("Kerr frequencies and precessions",
            f"Omega_phi at r=10M, {'prograde' if pro else 'retrograde'}",
            got, want, "Kepler + spin correction, timed over one revolution")

    Om = analytic.kerr_circular_frequencies(12.0, 0.9, True)
    ratio, adv = measure.measure_nodal_precession(m9, 12.0, Q=1e-6)
    row("Kerr frequencies and precessions",
        "Lense-Thirring node ratio Omega_phi/Omega_theta, r=12M",
        ratio, Om[0] / Om[1], f"advance {adv:.4e} rad per polar period")
    ratio0, _ = measure.measure_nodal_precession(Schwarzschild(), 12.0, Q=1e-6)
    row("Kerr frequencies and precessions",
        "same measurement at a=0 (control)", ratio0, 1.0,
        "orbital planes are fixed without spin")

    Om_pro = analytic.kerr_circular_frequencies(10.0, 0.9, True)
    got = measure.measure_precession(m9, 9.9, 10.1)
    row("Kerr frequencies and precessions",
        "periapsis advance, near-circular r=10M prograde",
        got, 2 * np.pi * (abs(Om_pro[0]) / Om_pro[2] - 1),
        "vs 2 pi (Omega_phi/Omega_r - 1); O(e^2) formula error")

    # -- spherical photon orbits ---------------------------------------------
    from kerrgeo import spherical_photon_orbit

    y0 = spherical_photon_orbit(m9, 2.6)
    sol = trace(m9, y0, 60.0, rtol=1e-13, atol=1e-13,
                events=[horizon_event(m9, eps=1e-3), escape_event(10.0)])
    drift = np.abs(sol.y[1] - 2.6)
    row("Spherical photon orbit (a=0.9, r=2.6M)",
        "max |r - r0| over first 20 M", float(drift[sol.t <= 20.0].max()),
        None, "holds the sphere at integration accuracy")
    row("Spherical photon orbit (a=0.9, r=2.6M)",
        "max |r - r0| by 60 M", float(drift.max()), None,
        "unstable orbit: rounding e-folds every ~1.7 M until departure")

    # -- energetics -----------------------------------------------------------
    E_neg, Lz_neg = -0.1, -3.0
    r_out = np.linspace(m9.r_ergo(np.pi / 2), 50.0, 400)
    row("Kerr energetics",
        "negative-energy orbit (E=-0.1): max R(r) outside the ergosphere",
        float(separated.radial_potential(r_out, 0.9, E_neg, Lz_neg,
                                         0.0, 1.0, 1.0).max()),
        None, "R < 0 everywhere outside: E < 0 states are confined (Penrose)")

    for a_spin, label in ((0.0, "a=0"), (0.998, "a=0.998")):
        mm = KerrBL(a=a_spin)
        _, E_isco, _ = circular_orbit(mm, mm.r_isco(True), True)
        row("Kerr energetics", f"ISCO binding energy 1-E, {label}",
            1.0 - E_isco,
            1.0 - np.sqrt(8.0 / 9.0) if a_spin == 0.0 else 0.3210,
            "radiative efficiency of accretion")

    # ======================================================================
    # Stage 3: the interior, in ingoing Kerr coordinates
    # ======================================================================
    from scipy.optimize import brentq

    from kerrgeo import (KerrIngoing, bl_to_ingoing, ingoing_to_bl,
                         principal_null_ingoing, zamo_drop_state)
    from kerrgeo.events import negative_r_escape_event, ring_event
    from kerrgeo.invariants import drift_report as _drift

    mi = KerrIngoing(a=0.9)

    # chart overlap
    y0_bl = zamo_drop_state(mk, 8.0)
    y0_in = bl_to_ingoing(mi, y0_bl)
    sb = trace(mk, y0_bl, 20.0, rtol=1e-13, atol=1e-13)
    si = trace(mi, y0_in, 20.0, rtol=1e-13, atol=1e-13)
    row("Ingoing chart vs BL (a=0.9)",
        "same geodesic, both charts, endpoint state diff at lam=20",
        float(np.abs(ingoing_to_bl(mi, si.y[:, -1]) - sb.y[:, -1]).max()),
        None, "all 8 phase-space components")

    # smooth double crossing
    sol = trace(mi, y0_in, 200.0, rtol=1e-12, atol=1e-12,
                events=[ring_event(mi, eps=1e-3)])
    lam_rm = sol.t[np.argmin(np.abs(sol.y[1] - mi.r_minus))]
    d_thru = _drift(sol.y[:, sol.t <= lam_rm + 0.05], mi, mu=1.0)
    row("Horizon crossing (a=0.9)", "norm drift through r+ AND r-",
        d_thru["norm"], None, "E and Lz drift exactly 0; BL cannot represent this")
    row("Horizon crossing (a=0.9)", "termination radius (ring approach)",
        float(sol.y[1, -1]), 1e-3, "stops only at the genuine singularity")

    # exact principal null ray
    y0p = principal_null_ingoing(mi, 5.0, np.radians(80))
    solp = trace(mi, y0p, 12.0, rtol=1e-13, atol=1e-13,
                 events=[negative_r_escape_event(-4.9)])
    row("Principal null ray (exact solution)",
        "max |r - (r0 - lambda)| through r+, r-, disk, to r=-4.9",
        float(np.abs(solp.y[1] - (5.0 - solp.t)).max()), None,
        "chart-adapted ray: dr/dlambda = -1 exactly")
    row("Principal null ray (exact solution)",
        "max drift in (v, theta, phi~)",
        float(max(np.abs(solp.y[i] - y0p[i]).max() for i in (0, 2, 3))),
        None, "all three frozen along the exact solution")

    # CTCs
    edge = brentq(lambda r: r**3 + 0.81 * r + 2 * 0.81, -1.5, -0.5)
    rr = np.linspace(-1.5, -1e-3, 4000)
    band = rr[mi.g_phiphi(rr, np.pi / 2) < 0]
    row("Closed timelike curves (a=0.9)",
        "equatorial CTC band inner edge", float(band.min()), float(edge),
        "vs real root of r^3 + a^2 r + 2 a^2 M = 0")
    r_pos = np.linspace(1e-3, 30.0, 800)
    th_g = np.linspace(0.01, np.pi - 0.01, 200)
    Rg, Tg = np.meshgrid(r_pos, th_g)
    row("Closed timelike curves (a=0.9)",
        "min g_phiphi over the whole r>0 sheet",
        float(mi.g_phiphi(Rg, Tg).min()), None,
        "positive everywhere: no CTCs outside the ring")
    row("Closed timelike curves (a=0.9)",
        "proper time to ride the loop at r=-0.5 (equator)",
        float(mi.ctc_loop_proper_time(-0.5)), None,
        "finite, order M: return to the same event after this much aging")

    # -- output -------------------------------------------------------------
    buf = io.StringIO()
    w = buf.write
    w("# Validation report\n\n")
    w(f"Generated by `scripts/run_validation.py` in {time.time() - t_start:.0f} s.\n")
    w("Geometric units, G = c = M = 1.\n\n")

    last = None
    for section, quantity, computed, reference, err, note in ROWS:
        if section != last:
            w(f"\n## {section}\n\n")
            w("| quantity | computed | reference | rel. error | note |\n")
            w("|---|---:|---:|---:|---|\n")
            last = section
        cs = f"{computed:.10g}"
        rs = f"{reference:.10g}" if reference is not None else "--"
        w(f"| {quantity} | `{cs}` | `{rs}` | `{err or '--'}` | {note} |\n")

    text = buf.getvalue()
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nwritten to {args.md}")


if __name__ == "__main__":
    main()

