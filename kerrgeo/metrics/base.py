"""Spacetime metrics, and the machinery for differentiating them.

The integrator never asks a spacetime for Christoffel symbols.  It asks for
exactly two things:

    ginv(x)   -> g^{ab}(x)          the *inverse* metric,   shape (4, 4)
    dginv(x)  -> d_c g^{ab}(x)      its first derivatives,  shape (4, 4, 4)

Why the inverse metric instead of the 40 Christoffel symbols?  Because we
integrate Hamilton's equations for

    H(x, p) = (1/2) g^{ab}(x) p_a p_b

and those equations reference g^{ab} and its gradient and nothing else.  See
`docs/formulation.md` for the derivation.

--------------------------------------------------------------------------
How the derivatives are computed: the complex-step trick
--------------------------------------------------------------------------
A naive central difference,  (f(x+h) - f(x-h)) / 2h, loses about half the
available precision to catastrophic cancellation: the best you can do is
~1e-10 relative error, which would show up directly as spurious drift in the
Carter constant and pollute our headline error diagnostic.

Instead we use the *complex step* derivative.  For a function f that is
analytic (built only from +, -, *, /, powers, sin, cos, exp, ...),

    f(x + i h) = f(x) + i h f'(x) - h^2 f''(x)/2 - ...

so

    f'(x) = Im[ f(x + i h) ] / h  +  O(h^2)

The key point: the derivative is recovered from the *imaginary* part, so it is
never contaminated by subtracting two nearly-equal real numbers.  There is no
cancellation, which means h can be made absurdly small (we use 1e-20) and the
O(h^2) truncation error vanishes entirely.  The result is correct to full
double precision -- it is, for practical purposes, an exact derivative, but it
is obtained without hand-deriving ten messy quotient-rule expressions per
metric (and without the bugs that come with them).

The cost: every metric you write must use only analytic operations.  No abs(),
no min/max, no branching on the value of a coordinate, no np.sqrt of something
that could be a negative real.  This is documented on `Metric.ginv`.

--------------------------------------------------------------------------
Why only some derivatives are computed
--------------------------------------------------------------------------
Kerr in Boyer-Lindquist coordinates is stationary (no t-dependence) and
axisymmetric (no phi-dependence).  So

    d_t g^{ab} = d_phi g^{ab} = 0     identically.

Hamilton's equation for the momenta is  dp_c/dl = -(1/2) d_c g^{ab} p_a p_b,
so the t and phi components of the RHS are *identically zero as written in the
code*, not merely small.  Energy E = -p_t and axial angular momentum
Lz = p_phi are therefore conserved to machine precision by construction, for
any integrator, at any step size.

That is not just an efficiency win.  It turns E and Lz into a pure *bug*
detector: if they ever move by more than ~1e-15, the cause is a coding error,
not truncation error.  Carter's constant Q, by contrast, comes from a Killing
*tensor* rather than a cyclic coordinate and is not protected this way -- so Q
drift is the honest measure of integration error.  See `kerrgeo/invariants.py`.
"""

from __future__ import annotations

import numpy as np

# Complex step size.  Because there is no cancellation this can be far below
# sqrt(eps); 1e-20 puts the truncation error O(h^2) = 1e-40 well under the
# 1e-16 rounding floor, while keeping h*f' comfortably above the 1e-308
# underflow limit.
_CSTEP = 1e-20


class Metric:
    """Base class for a spacetime in a particular coordinate chart.

    Subclasses must set ``cyclic`` and implement ``ginv``.
    """

    #: Coordinate indices the metric components do not depend on.  Their
    #: conjugate momenta are then exactly conserved.  (0, 3) = (t, phi) for a
    #: stationary axisymmetric spacetime in BL-like coordinates.
    cyclic: tuple[int, ...] = ()

    #: Human-readable coordinate names, for plots and error messages.
    coords: tuple[str, ...] = ("t", "r", "theta", "phi")

    def ginv(self, x):
        """Inverse metric g^{ab} at position ``x``, shape (4, 4).

        MUST be implemented using only operations that are analytic in the
        complex sense (arithmetic, powers, sin, cos, ...), so that the
        complex-step derivative in `dginv` is valid.  In particular ``x`` may
        arrive with complex dtype; write the body so it propagates that dtype
        rather than casting to float.
        """
        raise NotImplementedError

    def g(self, x):
        """Covariant metric g_ab, obtained by inverting ``ginv``.

        Only used for diagnostics and for building initial conditions from a
        local frame -- never inside the integration loop.
        """
        return np.linalg.inv(np.asarray(self.ginv(x), dtype=float))

    def dginv(self, x):
        """Derivatives d_c g^{ab}, shape (4, 4, 4), indexed [c, a, b].

        Rows for cyclic coordinates are left exactly zero -- see module
        docstring.
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        out = np.zeros((n, n, n))
        for c in range(n):
            if c in self.cyclic:
                continue  # identically zero; leaving it so protects E and Lz
            xc = x.astype(complex)
            xc[c] += 1j * _CSTEP
            out[c] = np.imag(self.ginv(xc)) / _CSTEP
        return out

    # -- geometry helpers, overridden where a closed form exists -------------

    def horizons(self):
        """(r_minus, r_plus), or None if the spacetime has no horizon."""
        return None

    def __repr__(self):
        return f"{type(self).__name__}()"
