# Kerr geodesics

A numerical integrator for geodesics in Kerr and Schwarzschild spacetime, built
around the Hamiltonian formulation and validated against closed-form results
wherever they exist.

Geometric units throughout: `G = c = 1`, lengths in units of `M`, signature
`(-, +, +, +)`.

![Photon trajectories](figures/photon_trajectories.png)

**Status.** Stage 1 (Schwarzschild) is complete and validated. The metric layer,
integrators and diagnostics are already spin-general, so the Kerr machinery is
in place and unit-tested; the Kerr *physics* study (frame dragging, the
ergosphere) is the next stage. See [Roadmap](#roadmap).

---

## Quick start

```bash
git clone https://github.com/chngyi/kerr-geodesics
cd kerr-geodesics
python -m venv .venv
```

Activate it — pick the line for your shell:

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\Activate.ps1     # Windows, PowerShell
.venv\Scripts\activate.bat     # Windows, cmd.exe
```

Then:

```bash
pip install -r requirements.txt

pytest -q                                        # 60 tests, ~25 s
python scripts/run_validation.py --md VALIDATION.md   # regenerate the report
python scripts/make_figures.py                   # regenerate figures/ (~80 s)
```

`run_validation.py` prints to stdout by default; `--md` is what writes the file.
Both `VALIDATION.md` and `figures/` are committed, so you only need to
regenerate them if you have changed the code.

```python
import numpy as np
from kerrgeo import Schwarzschild, trace, photon_from_impact_parameter
from kerrgeo.events import horizon_event, escape_event

bh = Schwarzschild()
y0 = photon_from_impact_parameter(bh, r0=1e4, b=6.0)     # b in units of M
sol = trace(bh, y0, lam_max=5e4,
            events=[horizon_event(bh), escape_event(2e4)])

print(sol.status, sol.r[-1])            # 'event' 20000.0  -- it escaped
```

---

## 1. Formulation: why Hamiltonian, not Christoffel symbols

The textbook geodesic equation is second order,

$$\frac{d^2x^a}{d\lambda^2} = -\Gamma^a{}_{bc}\,\frac{dx^b}{d\lambda}\frac{dx^c}{d\lambda}$$

which you split into 8 first-order ODEs for $(x^a, u^a)$. It works, but for Kerr
it means coding 40 independent Christoffel symbols, each of them a page of
algebra and a place to hide a sign error.

This project uses the equivalent Hamiltonian description. With
$H = \tfrac12 g^{ab}(x)\,p_a p_b$ and $p_a$ the **covariant** momentum,

$$\frac{dx^a}{d\lambda} = g^{ab}p_b, \qquad
\frac{dp_a}{d\lambda} = -\tfrac12\,(\partial_a g^{bc})\,p_b p_c$$

Three advantages, in increasing order of importance:

1. **Only the inverse metric is needed** — five non-zero components for Kerr,
   not forty Christoffels. And the Kerr inverse metric is genuinely simple:

   $$g^{tt} = -\frac{A}{\Sigma\Delta},\quad g^{t\phi} = -\frac{2aMr}{\Sigma\Delta},\quad
   g^{rr} = \frac{\Delta}{\Sigma},\quad g^{\theta\theta} = \frac{1}{\Sigma},\quad
   g^{\phi\phi} = \frac{\Delta - a^2\sin^2\theta}{\Sigma\Delta\sin^2\theta}$$

2. **The norm is the Hamiltonian.** $2H = g^{ab}p_ap_b = -\mu^2$ is conserved
   because $H$ has no explicit $\lambda$-dependence. The four-velocity norm you
   wanted as an error diagnostic is not a separate quantity to track — it *is*
   the conserved Hamiltonian.

3. **$E$ and $L_z$ become exact.** Kerr in Boyer–Lindquist coordinates is
   stationary and axisymmetric, so $g^{ab}$ has no $t$ or $\phi$ dependence, so
   $\partial_t g^{bc} = \partial_\phi g^{bc} = 0$ *identically*. The code for
   $dp_t/d\lambda$ and $dp_\phi/d\lambda$ therefore evaluates to exactly zero,
   and $E = -p_t$, $L_z = p_\phi$ are conserved to machine precision regardless
   of integrator or step size.

Point 3 changes what validation means, and it is worth being precise about:

| quantity | protected by | what its drift means |
|---|---|---|
| $E = -p_t$ | cyclic coordinate $t$ | **a bug.** Measured drift is exactly `0.0`. |
| $L_z = p_\phi$ | cyclic coordinate $\phi$ | **a bug.** Measured drift is exactly `0.0`. |
| $\mu^2 = -g^{ab}p_ap_b$ | conserved $H$ | truncation error, but a weak test |
| **$Q$ (Carter)** | Killing *tensor* — **not** enforced by the code | **the honest truncation-error measure** |

In your N-body code, energy drift measured truncation error. Here it does not:
$E$ and $L_z$ are structurally exact, and if they ever move, tightening the
tolerance will not help because the problem is a coding error. The quantity that
plays the role energy played for you is **Carter's constant**.

### Carter's constant and why Kerr is tractable at all

The two Killing vectors give $E$ and $L_z$; with the norm that is three
constants for a four-dimensional problem, which leaves a genuinely
two-dimensional $(r,\theta)$ system with no closed form. Kerr also admits an
irreducible Killing **tensor**, and the associated fourth constant

$$Q = p_\theta^2 + \cos^2\theta\left[a^2(\mu^2 - E^2) + \frac{L_z^2}{\sin^2\theta}\right]$$

is what separates the Hamilton–Jacobi equation and reduces the motion to
decoupled first-order form. $Q = 0$ is the equatorial plane; $Q > 0$ orbits
oscillate in $\theta$ about the equator.

Because $Q$ comes from a Killing tensor rather than a cyclic coordinate,
nothing in the code forces it to hold — which is exactly what makes it a good
diagnostic.

### The separated form is used as a cross-check, not the workhorse

The separation gives

$$\Sigma\frac{dr}{d\tau} = \pm\sqrt{R(r)},\qquad
\Sigma\frac{d\theta}{d\tau} = \pm\sqrt{\Theta(\theta)}$$

which is much faster — two nontrivial ODEs instead of eight. It is implemented
in [`kerrgeo/separated.py`](kerrgeo/separated.py), but as a second opinion
rather than the main path, for two reasons. The $\pm$ branches must be flipped
by hand at every turning point, exactly where $\sqrt{R}\to 0$ and its derivative
blows up. And more fundamentally, it *assumes* $E$, $L_z$, $Q$ are constant, so
it cannot be validated by checking that they are.

Two implementation notes there. We integrate in **Mino time**
($d\tau = \Sigma\,d\lambda_M$), which removes the shared $1/\Sigma$ and
decouples $r$ from $\theta$ completely. And we differentiate the first-order
equations into second-order form, $d^2r/d\lambda_M^2 = \tfrac12 R'(r)$, which
eliminates the square roots and the branch-flipping entirely — turning points
are then handled automatically, because the trajectory simply decelerates
through them.

**Result:** the two formulations, sharing no code, agree to
`2.4e-6` in every coordinate over 390 M of proper time on a generic inclined
$a = 0.9$ orbit. That is a far stronger statement than either one's internal
diagnostics.

### Exact metric derivatives without hand-derived algebra

`Metric.dginv` uses **complex-step differentiation**:

$$f'(x) = \frac{\mathrm{Im}\,[\,f(x + ih)\,]}{h} + O(h^2)$$

Because the derivative is recovered from the *imaginary* part there is no
subtractive cancellation, so $h$ can be taken absurdly small (we use `1e-20`)
and the $O(h^2)$ truncation error vanishes. The result is correct to full double
precision — effectively an exact derivative, without hand-deriving ten messy
quotient-rule expressions per metric.

The price: every metric must be written using only analytic operations (no
`abs`, no branching on coordinate values). That constraint is documented on
`Metric.ginv`, and a unit test checks the complex-step result against a central
difference.

---

## 2. Integrator choice

**Short answer: adaptive DOP853 by default; RK4 for bulk ray tracing;
Gauss–Legendre when you want to claim something about long-term orbits.**

### Your leapfrog reflex will not work here

The natural instinct — "use velocity Verlet like a good N-body person" — fails.
Leapfrog and friends need a **separable** Hamiltonian, $H = T(p) + V(x)$, so
each half-step can be solved exactly. Ours is $H = \tfrac12 g^{ab}(x)p_ap_b$:
quadratic in $p$, but with position-dependent coefficients. There is no exact
drift/kick split. This rules out the entire family of explicit symplectic
methods you would reach for first.

### Whether RK4's drift actually matters depends on what you compute

| task | secular drift matters? | recommendation |
|---|---|---|
| Deflection, capture, ray tracing | No — single pass, nothing accumulates | **RK4** is fine and fastest |
| Precession over hundreds of orbits | **Yes**, and worse than it looks | DOP853 or GL2 |
| Long-term bound orbits, inspirals | Yes | **GL2** (symplectic) |

The precession case deserves the warning: a spurious secular change in $E$ or
$L_z$ would masquerade as *extra precession* — the numerical error contaminates
precisely the quantity being measured. (In this code that particular failure is
structurally impossible, since $E$ and $L_z$ are exact. But the trajectory error
is not.)

![Conservation and convergence](figures/conservation.png)

The left panel is the direct measurement. Over a few orbits the schemes are
indistinguishable; over ~130 radial periods, RK4's Carter-constant error climbs
steadily from `1e-6` to `1e-4` while the symplectic scheme's stays bounded and
merely oscillates. That is the whole practical difference between symplectic and
non-symplectic, and it is why the honest answer is "it depends on the task"
rather than "always use X".

### Why adaptive stepping, for a reason unrelated to accuracy

Near periapsis and near the photon sphere the coordinate velocity varies by
orders of magnitude along a single trajectory. Any fixed step is simultaneously
wasteful far out and inadequate close in. DOP853 at `rtol = 1e-12` holds the
Carter constant to `1.8e-11` over 3000 M — three to four orders of magnitude
better than fixed-step RK4 at `h = 0.5`, for comparable cost.

### Available integrators

| method | order | properties | use for |
|---|---|---|---|
| `RK4` | 4 | explicit, fixed step | ray tracing, speed |
| `DOP853` | 8 | explicit, adaptive | **default** |
| `GL2` / `GL3` | 4 / 6 | implicit, **symplectic** | long-term bound orbits |

`GL2`/`GL3` are Gauss–Legendre implicit Runge–Kutta, solved by fixed-point
iteration. Unlike leapfrog they work for a general non-separable $H$. They also
conserve quadratic first integrals exactly — which means the norm
$g^{ab}p_ap_b$ holds to machine precision *by construction*. Worth knowing when
reading the diagnostics: it makes the norm look flattering relative to the
actual trajectory error, which is one more reason to watch $Q$ instead.

---

## 3. Validation

Full numbers in [`VALIDATION.md`](VALIDATION.md), regenerated by
`scripts/run_validation.py`. 60 tests in `tests/test_kerrgeo.py`.

The strategy is to compare against things that **share no code with the
integrator** — closed forms, independent quadratures, and a second formulation.
Self-reported conservation is necessary but nowhere near sufficient.

### Closed-form landmarks (exact)

Horizon $r_\pm = M \pm\sqrt{M^2-a^2}$; photon sphere $3M$; ISCO $6M$; critical
impact parameter $b_c = 3\sqrt{3}M$; ergosphere $r_E = M + \sqrt{M^2 - a^2\cos^2\theta}$.
For extremal $a = M$: horizon, prograde ISCO and prograde photon orbit all
collapse to $r = M$, retrograde ISCO is $9M$, retrograde photon orbit is $4M$.
All reproduced to `0.00e+00`.

### Light deflection

![Deflection validation](figures/deflection.png)

The integrated geodesics track the exact quadrature to **1e-12 to 1e-10
relative across three decades** of impact parameter — while over the same range
the first-order Einstein formula $4M/b$ is off by 20% at $b = 6M$ and the
three-term series by `1e-3`.

Two checks here are sharper than they look:

- **The second-order coefficient.** $4M/b$ alone is recovered by almost any
  roughly-correct calculation. Isolating the next term from the integrated
  result gives `11.78205` against $15\pi/4 = 11.78097$ — a relative error of
  `9.1e-5`, which is the size of the *fourth*-order term we did not subtract.
  That coefficient is sensitive to the actual spatial curvature of the metric.
- **The capture threshold by bisection.** Bisecting on "does this photon reach
  the horizon" gives `5.196152423`, matching $3\sqrt{3}$ to `6.5e-11`. This is a
  genuinely strong-field quantity obtained by integration, not by formula.

A note on method: extracting the deflection by subtracting initial and final
$\phi$ carries an $O(b/r_0)$ error that would swamp the second-order term. We
read off the *direction of travel* at each end instead, which converges as
$O(M\log r/r)$ with no $O(b/r)$ term at all. See `measure.measure_deflection`.

### Perihelion precession

![Precession](figures/precession.png)

Against an exact quadrature of the orbit equation: `1.6e-13` at $r_p = 10M$,
`5.7e-12` at $r_p = 100M$.

**Mercury: an honest negative result.** The weak-field formula
$6\pi GM/c^2a(1-e^2)$ gives **42.981 arcsec/century** against the observed
42.98. But *integrating the geodesic* gives 42.55 — about 1%, and it does not
improve as `rtol` is tightened.

This is a precision limit, not an integrator failure, and it is worth
understanding because it is the kind of thing that looks like a bug. Mercury's
advance is `5.02e-7` rad against a swept $2\pi$: measuring it by subtraction
throws away seven significant figures, leaving ~9 digits of a 16-digit float.
The right panel shows the error growing smoothly with orbit size, tracking about
two decades above the single-rounding floor (the gap is accumulation over many
steps). The fix, if you want one, is extended precision (`mpmath`) — not a
better integrator.

The *analytic* path avoids this by folding the $-2\pi$ inside the integrand, so
the small quantity is computed directly rather than as a difference of large
ones. That trick is not available to the measurement, because $\phi$ comes out
of the integrator as a single accumulated number.

### Two independent formulations

Hamiltonian vs separated Carter equations, generic inclined $a = 0.9$ orbit
($E = 0.95$, $L_z = 2.8$, $Q = 3$), compared over 390 M of proper time:

| coordinate | max deviation |
|---|---|
| $r$ | `2.4e-6` |
| $\theta$ | `1.8e-6` |
| $\phi$ | `2.3e-6` |
| $t$ | `2.1e-6` |

### Other checks worth having

- **Time reversibility** — integrate forward then back: `8.0e-13`. This catches
  a class of bug the invariants cannot see, because a geodesic can sit at
  completely the wrong place on the right orbit and still report perfect $E$,
  $L_z$ and $Q$.
- **Convergence order** — halving $h$ cuts the error by ~16 for both fixed-step
  schemes, confirming they are the order they claim to be. (Measure this at
  coarse steps: at $h = 0.2$ RK4 is already on the round-off floor, where the
  ratio is meaningless.)
- **Circular orbits stay circular** — peak-to-peak $r$ over 2000 M: `1.3e-11`.
- **Forbidden initial conditions raise** rather than silently producing NaN
  trajectories, with a message saying which of $R(r)$ or $\Theta(\theta)$ went
  negative.

---

## 4. Coordinate singularities at the horizons

![Horizon handling](figures/horizon.png)

$\Delta = r^2 - 2Mr + a^2$ vanishes at $r_\pm$, making $g^{tt}$ and $g^{t\phi}$
blow up like $1/\Delta$. **Nothing physical happens there.** The left panel
shows the Kretschmann scalar $R_{abcd}R^{abcd}$ is perfectly finite at $r_+$
(markers) — it is the chart that fails, because Boyer–Lindquist $t$ is the time
of a distant static observer, and that observer never sees anything cross.

The same panel shows what *is* singular: only the $\theta = \pi/2$ curve
diverges as $r \to 0$. The singularity is a **ring** in the equatorial plane,
and off-equatorial worldlines pass the origin with finite curvature.

### The practical handling, and why it is enough

Because we integrate in an **affine parameter rather than $t$**, the trajectory
stays smooth all the way down to $r_+$; only $dt/d\lambda$ diverges. This is the
main reason to parametrise by $\lambda$: it converts a stiff problem into a
benign one. So the correct handling for all exterior work is simply to stop —
a terminal event at $(1+\epsilon)r_+$, labelling the geodesic "captured".
Nothing that reaches the horizon comes back, so no exterior physics is lost. For
ray tracing, captured = black pixel.

The right panel is the evidence that this is harmless: varying $\epsilon$ over
**seven decades** ($10^{-9}$ to $10^{-2}$) changes the deflection at $b = 6M$ by
exactly `0`. An escaping ray never goes near $r_+$, so the cutoff is invisible
to it.

### Stage 3 needs a different chart — plan for it now

The interior region and the closed timelike curves near the ring are **not
reachable this way**. You cannot integrate *through* $r_+$ in Boyer–Lindquist
coordinates at any tolerance, because the chart genuinely does not cover the
crossing. That needs a horizon-penetrating chart — ingoing Kerr / Kerr–Schild,

$$dv = dt + \frac{r^2+a^2}{\Delta}dr, \qquad
d\tilde\phi = d\phi + \frac{a}{\Delta}dr$$

in which $g = \eta + f\,l\,l$ is regular at both horizons. It is also the only
way to reach $r < 0$ through the ring, which is where the CTC region
$g_{\phi\phi} < 0$ actually lives.

This is why `Metric` is an abstract, pluggable object rather than hard-coded BL:
**stage 3 is a new metric subclass, not a rewrite.** Implement `ginv` with
analytic operations, declare which coordinates are cyclic, and everything
else — integrators, events, invariants, diagnostics — works unchanged.

---

## Layout

```
kerrgeo/
  metrics/base.py     Metric ABC; complex-step derivatives; cyclic-coordinate handling
  metrics/kerr.py     Kerr in Boyer-Lindquist (a=0 gives Schwarzschild)
  hamiltonian.py      Hamilton's equations; building initial conditions
  invariants.py       norm, E, Lz, Carter Q; drift reporting
  integrate.py        RK4, DOP853, Gauss-Legendre symplectic; Solution type
  events.py           horizon / escape / turning-point / equator-crossing events
  separated.py        Carter first-order form, as an independent cross-check
  measure.py          deflection, precession, capture threshold, reversibility
  analytic.py         closed forms and exact quadratures to validate against
scripts/
  run_validation.py   quantitative report -> VALIDATION.md
  make_figures.py     the five figures in figures/
  style.py            plotting style; CVD-checked palette
tests/test_kerrgeo.py 60 tests
```

Schwarzschild is deliberately *not* a separate implementation — it is
`KerrBL(a=0)`. Sharing the code path means the Schwarzschild validation suite,
which has many exact answers to check against, is simultaneously a test of the
Kerr code, where far fewer closed forms exist.

## Roadmap

- [x] **Stage 1 — Schwarzschild.** Deflection, photon sphere, capture threshold,
      precession. Validated.
- [ ] **Stage 2 — Kerr physics.** Frame dragging, the ergosphere, prograde vs
      retrograde asymmetry, spherical photon orbits. *Machinery is in place and
      unit-tested at several spins; the study is next.*
- [ ] **Stage 3 — Interior.** Requires the Kerr–Schild chart (see above). CTCs
      near the ring, $r < 0$ region.
- [ ] **Stage 4 — Backwards ray tracing.** `analytic.kerr_shadow_boundary`
      already provides the Bardeen (1973) analytic shadow outline to validate a
      rendered image against.

## References

- Carter, *Global structure of the Kerr family of gravitational fields*, Phys. Rev. **174** (1968) 1559 — the separation and the fourth constant.
- Bardeen, Press & Teukolsky, ApJ **178** (1972) 347 — ISCO, photon orbits, circular-orbit $E$ and $L_z$.
- Bardeen, in *Black Holes* (Les Houches, 1973) — the analytic shadow outline.
- Keeton & Petters, Phys. Rev. D **72** (2005) 104006 — the deflection series beyond $4M/b$.
- Hairer, Lubich & Wanner, *Geometric Numerical Integration* (2006) — Gauss–Legendre as a symplectic method for non-separable $H$.
- Squire & Trapp, SIAM Rev. **40** (1998) 110 — complex-step differentiation.

Open-source Kerr integrators worth comparing against:
[curvedpy](https://github.com/bldevries/curvedpy) (Python, Schwarzschild + Kerr),
[SIM5](https://github.com/mbursa/sim5) (C with Python bindings),
[AART](https://github.com/iAART/aart) (adaptive analytic ray tracing, photon rings),
[KerrP2P](https://github.com/AuroraDysis/KerrP2P) (point-to-point null geodesics).
