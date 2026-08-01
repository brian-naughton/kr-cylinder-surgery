#!/usr/bin/env python3
"""Shared objects for the R-partial visit (RD-OPS.md, A22).

Everything transcribed here is Dubouloz--Moser-Jauslin--Poloni's,
arXiv:0903.4278v1 §7 (Prop 7.1, Prop 7.2, Cor 7.3).  The transcription
is CERTIFIED by ``rd_1_transcribe.py`` (ledger RD-1-TRANSCRIBE.json):
the paper's own printed identities were machine-verified before any of
this was used downstream.  No novelty is claimed on these objects.

This module supplies the verified objects plus the working machinery
every later block needs:

  * ``AW_t_valuation``  -- the t-adic valuation of an element of A[w]
    (t is prime in A[w] because A[w]/(t) = C[x,y,z,w]/(x^2 y + z^2 + x)
    is a domain), returned with an explicit witness quotient;
  * ``in_ideal_xwz``    -- divisibility by the irreducible x w + z;
  * ``iota``            -- the faithful model A[w] --> C[x^{+-1},z,t,w].
"""

from __future__ import annotations

import sympy as sp

x, y, z, t, w = sp.symbols("x y z t w")
GENS = (x, y, z, t, w)
E = sp.expand

P = x**2 * y + z**2 + x + t**3          # Koras-Russell cubic (DMJP)
S = x * y + z**2 + x + t**3             # the auxiliary threefold X_1
Q0 = E(P.subs(t, 0))                    # P mod t = x^2 y + z^2 + x

PHI = {
    x: x,
    y: x * y - x * w**2 - 2 * z * w,
    z: z + x * w,
    t: t,
    w: 2 * w + y * z + 3 * x * y * w - 3 * z * w**2 - x * w**3,
}

PSI = {
    x: x,
    y: -(y + y**2 + w * z) / t**3 - (y * z - x * w) ** 2 / (4 * t**6),
    z: z - x * (y * z - x * w) / (2 * t**3),
    t: t,
    w: (y * z - x * w) / (2 * t**3),
}


def ap(hom: dict, expr: sp.Expr) -> sp.Expr:
    """Apply a ring endomorphism given on the generators (H1-safe)."""
    return E(expr.xreplace(hom))


def delta(f: sp.Expr) -> sp.Expr:
    """Delta = t^6 ( -2 z d/dx + (y+1) d/dz )   (DMJP Prop 7.2)."""
    return E(t**6 * (-2 * z * sp.diff(f, x) + (y + 1) * sp.diff(f, z)))


# The four components of partial = Phi o Delta o Psi, as polynomial
# representatives in C[x,y,z,t,w].  partial(t) = 0.
DPART = {v: ap(PHI, delta(ap(PSI, v))) for v in GENS}
DX, DY, DZ, DW = (E(DPART[v]) for v in (x, y, z, w))

# The two kernel generators of Cor-7.1 transport: images under Phi of
# Delta's kernel generators y and w  (see rd_3_kernel.py).
KER_P = E(PHI[y])          # Phi(y) = x y - x w^2 - 2 z w
KER_Q = E(PHI[w])          # Phi(w) = 2w + yz + 3xyw - 3zw^2 - xw^3

# --- the faithful localisation model  A[w] --> C[x^{+-1},z,t,w] ---------
IOTA_Y = -(z**2 + x + t**3) / x**2
IOTA = {y: IOTA_Y}


def iota(e: sp.Expr) -> sp.Expr:
    return sp.cancel(E(e.xreplace(IOTA)))


def D_model(f: sp.Expr, comps=None) -> sp.Expr:
    """The derivation partial read in the model C[x^{+-1},z,t,w]."""
    dx_, dz_, dw_ = comps if comps else (iota(DX), iota(DZ), iota(DW))
    return sp.cancel(E(dx_ * sp.diff(f, x) + dz_ * sp.diff(f, z)
                       + dw_ * sp.diff(f, w)))


# --- exact t-adic valuation inside A[w] --------------------------------
def is_poly(e: sp.Expr) -> bool:
    """Is e a polynomial in x,y,z,t,w over Q?

    NOTE: ``sp.denom(sp.together(e)) == 1`` is NOT this test -- it also
    rejects rational coefficients such as the 1/2 that pervades these
    components.  Use the Poly constructor, which is the actual question.
    """
    try:
        sp.Poly(E(e), *GENS, domain="QQ")
        return True
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed,
            sp.polys.polyerrors.GeneratorsNeeded):
        return False


def _divide_by_Q0(g: sp.Expr):
    """Write g = a * Q0 exactly in Q[x,y,z,w], or return None.

    Q0 = x^2 y + z^2 + x is linear in y and primitive (gcd(x^2, z^2+x)
    = 1), so by Gauss's lemma a zero remainder over the fraction field
    already forces a to be a polynomial -- asserted, not assumed.
    """
    F = sp.QQ.frac_field(x, z, w)
    quo, rem = sp.div(sp.Poly(g, y, domain=F), sp.Poly(Q0, y, domain=F))
    if rem.as_expr() != 0:
        return None
    a = sp.cancel(quo.as_expr())
    assert is_poly(a), "Gauss's lemma violated"
    return E(a)


def AW_t_valuation(f: sp.Expr, cap: int = 24):
    """Return (k, g) with f = t^k * g in A[w] and g not in t*A[w].

    t is prime in A[w] (A[w]/(t) = C[x,y,z,w]/(x^2 y + z^2 + x), and
    x^2 y + z^2 + x is irreducible), so k is well defined.  Each step
    tests f|_{t=0} in (Q0) and, on success, produces the witness
    quotient (f - a*P)/t explicitly -- no ideal-membership oracle and
    no Groebner call.
    """
    g, k = E(f), 0
    if g == 0:
        return (None, sp.Integer(0))
    while k < cap:
        a = _divide_by_Q0(E(g.subs(t, 0)))
        if a is None:
            return (k, g)
        nxt = sp.cancel(E(g - a * P) / t)
        assert is_poly(nxt), "t-strip left a t-denominator"
        g, k = E(nxt), k + 1
    raise RuntimeError(f"t-valuation exceeded cap {cap}")


def in_ideal_xwz(f: sp.Expr) -> bool:
    """Is f in the ideal (x w + z) of A[w]?

    x w + z is irreducible in A[w] (degree 1 in w, primitive: x and z
    have no common factor in the UFD A, since V(x,z) has codimension 3
    in X).  A[w]/(xw+z) is a domain that injects into A[1/x], so the
    test is: substitute w -> -z/x and y -> -(z^2+x+t^3)/x^2 and ask for
    identical vanishing.
    """
    return sp.cancel(E(iota(f).subs(w, -z / x))) == 0
