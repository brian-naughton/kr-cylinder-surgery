#!/usr/bin/env python3
"""Shared machinery for the RG visit (RG-OPS.md, master A24).

RE-LOAD DISCIPLINE (A24, binding). Nothing here is taken from an
earlier session's conversation. The DMJP objects come from
``rd_common.py`` (certified by ledger RD-1-TRANSCRIBE.json); the R∂
results are re-derived in-process and re-checked by ``verify_carryover``
below, which every RG runner calls before doing anything else.

Attribution: Φ, Ψ, ∆, ∂, ∂₁, ∂₂ are Dubouloz--Moser-Jauslin--Poloni's,
arXiv:0903.4278v1 §7. No novelty is claimed on them.

THE L-PRESENTATION.  RD-4 established

    A[w][1/(t(p+1))] = C[p,q,t][1/(t(p+1))][Z],   Z = z + x w,

with ker(∂|A[w]) = C[p,q,t] (RD-3). Write

    L = C[P_, Q_, T_][1/(T_ (P_+1))][ZZ]

for the localised ring in its own coordinates. ``to_L`` sends an
element of A[w] to L; ``TO_L`` holds the images of the five generators.
The divisor is REDUCIBLE and is always handled as two components,
{T_ = 0} and {P_+1 = 0}, never as one.
"""

from __future__ import annotations

import itertools

import sympy as sp

from checker import is_in_ideal
from rd_common import (DPART, E, GENS, KER_P, KER_Q, P, S, ap, PHI, PSI,
                       AW_t_valuation, is_poly, x, y, z, t, w)
from rd_4_plinth import partial_red_components

# --- the L-coordinates -------------------------------------------------
Pp, Qq, Tt, ZZ = sp.symbols("P_ Q_ T_ ZZ")
LGENS = (Pp, Qq, Tt, ZZ)
KGENS = (Pp, Qq, Tt)

#: the two boundary components, kept apart throughout (RG-OPS §1)
BOUNDARY = {"t": Tt, "p+1": Pp + 1}

# the A[w] generators, read in L (RD-4 recovery formulas, re-derived here
# from the same closed forms and re-verified in-process by
# verify_carryover)
XL = -(ZZ**2 + Tt**3) / (Pp + 1)
WL = sp.cancel((Pp * ZZ - XL * Qq) / (2 * Tt**3))
YL = sp.cancel(-(Pp + Pp**2 + Qq * ZZ) / Tt**3 - WL**2)
ZL = sp.cancel(ZZ - XL * WL)
TL = Tt
TO_L = {x: XL, y: YL, z: ZL, t: TL, w: WL}

#: the three fractional generators that carry A[w] beyond C[p,q,t,Z]
#: (established and certified in RG-2)
H_L = sp.cancel((Pp + Pp**2 + Qq * ZZ) / Tt**3)


def to_L(expr: sp.Expr) -> sp.Expr:
    """Image in L of a polynomial representative in C[x,y,z,t,w]."""
    return sp.cancel(sp.together(E(expr).xreplace(TO_L)))


def z_split(expr: sp.Expr):
    """Write an element of L as a list of ZZ-coefficients over K."""
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    pn = sp.Poly(E(num), ZZ)
    return [sp.cancel(c / den) for c in reversed(pn.all_coeffs())]


def z_degree(expr: sp.Expr) -> int:
    cs = z_split(expr)
    d = -1
    for i, c in enumerate(cs):
        if sp.cancel(c) != 0:
            d = i
    return d


def order_along(expr: sp.Expr, comp: sp.Expr, cap: int = 12) -> int:
    """Exact order of vanishing of expr in L along a boundary component.

    Negative means a pole. ``comp`` is T_ or P_+1; the two are always
    measured SEPARATELY (RG-OPS §1). Returns the least k such that
    comp**(-k) * expr is regular at comp = 0, computed by clearing and
    then dividing exactly.
    """
    e = sp.cancel(sp.together(expr))
    if e == 0:
        return None
    num, den = sp.fraction(e)
    return _ord(num, comp, cap) - _ord(den, comp, cap)


def _ord(poly: sp.Expr, comp: sp.Expr, cap: int) -> int:
    p0 = E(poly)
    if p0 == 0:
        return cap
    k = 0
    while k < cap:
        quo, rem = sp.div(sp.Poly(p0, Pp, Qq, Tt, ZZ),
                          sp.Poly(comp, Pp, Qq, Tt, ZZ))
        if rem.as_expr() != 0:
            return k
        p0 = quo.as_expr()
        k += 1
    return k


def in_C_pqtZ(expr: sp.Expr) -> bool:
    """Is an element of L already a POLYNOMIAL in p, q, t, Z?"""
    e = sp.cancel(sp.together(expr))
    try:
        sp.Poly(e, *LGENS, domain="QQ")
        return True
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed,
            sp.polys.polyerrors.GeneratorsNeeded):
        return False


def in_C_pqt(expr: sp.Expr) -> bool:
    """Is an element of L a POLYNOMIAL in p, q, t alone (i.e. in ker ∂)?"""
    e = sp.cancel(sp.together(expr))
    if sp.diff(e, ZZ) != 0:
        return False
    try:
        sp.Poly(e, *KGENS, domain="QQ")
        return True
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed,
            sp.polys.polyerrors.GeneratorsNeeded):
        return False


# --- Krull dimension via a maximal independent set ---------------------
def krull_dim(gens, vs, order="grevlex"):
    """dim V(gens) in affine space on ``vs``; -1 for the empty variety.

    dim I = dim in(I), and dim in(I) is the size of the largest subset U
    of the variables carrying NO leading monomial. Exact, and cheap at
    these sizes.
    """
    gens = [E(g) for g in gens if E(g) != 0]
    if not gens:
        return len(vs)
    G = sp.groebner(gens, *vs, order=order)
    if list(G.exprs) == [1]:
        return -1
    lms = []
    for f in G.exprs:
        lm = sp.Poly(f, *vs).monoms(order=order)[0]
        lms.append(set(i for i, e in enumerate(lm) if e > 0))
    best = 0
    idx = range(len(vs))
    for k in range(len(vs), 0, -1):
        for U in itertools.combinations(idx, k):
            Us = set(U)
            if all(not lm.issubset(Us) for lm in lms):
                return k
    return best


# --- carry-over verification (called by every RG runner) ---------------
def verify_carryover(check):
    """Re-verify, IN PROCESS, every R∂ fact the RG blocks lean on.

    ``check(name, predicate, note)`` is the caller's Runner.check. This
    is the A24 re-load rule made executable: nothing below is assumed
    from an earlier session.
    """
    red = partial_red_components()

    check("carry-over: Phi(S) = P (DMJP Prop 7.1)",
          lambda: E(ap(PHI, S) - P) == 0)
    check("carry-over: partial(x) = -2 t^6 (z + x w) (DMJP Prop 7.2)",
          lambda: E(DPART[x] + 2 * t**6 * (z + x * w)) == 0)
    check("carry-over: content is t^3 (RD-2)",
          lambda: min(AW_t_valuation(E(DPART[v]))[0]
                      for v in (x, y, z, w)) == 3)
    check("carry-over: partial_red(x) = -2 t^3 (x w + z)",
          lambda: E(red[x] + 2 * t**3 * (x * w + z)) == 0)
    check("carry-over: p and q are in ker partial (RD-3)",
          lambda: all(is_in_ideal(E(sum(DPART[v] * sp.diff(g, v)
                                        for v in GENS)), P, y)
                      for g in (KER_P, KER_Q)))
    check("carry-over: the L-images of x,y,z,t,w are correct (RD-4)",
          lambda: all(_back_ok(v) for v in GENS),
          "each recovery formula re-substituted and tested in (P)")
    check("carry-over: partial_red(Z) = t^3 (p+1), Z = z + x w",
          lambda: is_in_ideal(
              E(sum(red[v] * sp.diff(E(z + x * w), v) for v in GENS)
                - t**3 * (KER_P + 1)), P, y))
    check("carry-over: the degeneracy line {x=z=t=w=0} is fixed (RD-4)",
          lambda: all(E(red[v].subs({x: 0, z: 0, t: 0, w: 0})) == 0
                      for v in GENS))
    return red


_BACK = {Pp: KER_P, Qq: KER_Q, Tt: t, ZZ: E(z + x * w)}
_CLEAR = E(t**6 * (KER_P + 1) ** 3)


def _back_ok(v) -> bool:
    """Substituting p,q,t,Z back into the L-image returns the generator."""
    num = E(sp.cancel(E((TO_L[v].xreplace(_BACK) - v) * _CLEAR)))
    return is_poly(num) and is_in_ideal(num, P, y)


def iota_model_rank():
    """Jacobian rank of (p, q, t, Z) in the model C[x^{+-1},z,t,w].

    Rank 4 certifies that p, q, t, Z are algebraically independent, i.e.
    that C[p,q,t,Z] really is a polynomial ring in four variables.
    """
    from rd_common import iota as _iota
    iy = -(z**2 + x + t**3) / x**2
    gens4 = [E(KER_P.xreplace({y: iy})), E(KER_Q.xreplace({y: iy})),
             t, E((z + x * w).xreplace({y: iy}))]
    M = sp.Matrix([[sp.cancel(sp.diff(sp.cancel(g), v))
                    for v in (x, z, t, w)] for g in gens4])
    return 4 if sp.cancel(M.det()) != 0 else M.rank()
