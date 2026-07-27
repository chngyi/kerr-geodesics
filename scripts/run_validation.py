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
from kerrgeo.events import horizon_event  # noqa: E402

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

