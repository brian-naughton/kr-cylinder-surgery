#!/usr/bin/env python3
"""Shared machinery for the RS visit (RS-OPS.md, master A28).

THE PICTURE BEING CERTIFIED (A28, verified at source in
`docs_sol_review_3.md` §1): Y_1 = Q_{(3,1), z^2-1/4} in the
Asok--Dubouloz--Ostvaer family (arXiv:2112.08241, Ex. 4.2.5),
A^1-equivalent to P^1 wedge-smash P^1 -- a motivic 4-sphere model with
complex realisation S^4. So the commuting square's smooth route reads

    X x A^1 (contractible)  -->  Y_1 (motivic S^4)  -->  A^4,

and the (p+1)-arrow is an explicit algebraic surgery killing Y_1's
H_4 generator. This module supplies the exact bookkeeping tools.

TERMINOLOGY (binding, A28): "flat family with generic fibre A^3 and one
quadric degeneration" -- never "A^3-fibration" unqualified. "Standard
centre" only with the defined equivalence (Aut(A^4), unit multiple of
the divisor, saturation of the centre).

ATTRIBUTION: ADO arXiv:2112.08241 for the family, its smoothness and
its A^1-homotopy type; DMJP arXiv:0903.4278 §7 for p, q, Z and the
t-localised bridge. Cited at every use.

A24 RE-LOAD RULE still in force: `rg_common.verify_carryover`,
`rm_common.verify_rg_carryover` and `verify_rm_carryover` below are run
by every RS runner before anything else.
"""

from __future__ import annotations

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import Pp, Qq, Tt, ZZ, H_L, TO_L, krull_dim, verify_carryover
from rm_common import N1, N2, R1, Xx, in_ideal, verify_rg_carryover

#: the Lefschetz class, and the Hodge-Deligne variables
L = sp.Symbol("L")
uu, vvv = sp.symbols("u v")

Hh = sp.Symbol("H_")
A4GENS = (Pp, Qq, Tt, ZZ)
B_GENS = (Pp, Qq, Tt, ZZ, Xx)          # X_1 x A^1  (relation R1)
Y_GENS = (Pp, Qq, Tt, ZZ, Hh)          # Y_1        (relation W)

#: W defines Y_1 in A^5; R1 defines X_1 x A^1 in A^5
W = E(Tt**3 * Hh - Pp - Pp**2 - Qq * ZZ)
QSURF = E(Qq * ZZ + Pp + Pp**2)        # the t = 0 quadric: qZ = -p(1+p)


def chi(cls):
    """Compactly-supported Euler characteristic: specialise L -> 1."""
    return sp.expand(sp.sympify(cls).subs(L, 1))


def epoly(cls):
    """Hodge-Deligne E-polynomial: L -> uv.

    Valid here because EVERY stratum used in this visit is a product of
    affine spaces, tori and points -- all of Hodge-Tate type -- so the
    E-polynomial is determined by the class. RS-3 records that collapse
    as a finding rather than hiding it.
    """
    return sp.expand(sp.sympify(cls).subs(L, uu * vvv))


# --- certification helpers ---------------------------------------------
def cert_graph(eq, solved, solution, gens, denom=sp.Integer(1)):
    """Certify {eq = 0, denom != 0} is the graph of solved = solution.

    Substituting the solution must annihilate eq identically after
    clearing the denominator. This certifies the stratum is isomorphic
    to the affine/torus base, with the projection as inverse.
    """
    sub = E(eq.xreplace({solved: solution}))
    return E(sp.cancel(sub * denom**8)) == 0 or E(sp.cancel(sub)) == 0


def cert_iso(fwd, inv, src_vars, tgt_vars):
    """Certify a parametrisation and its inverse compose to the identity.

    fwd: dict tgt_var -> expression in src_vars
    inv: dict src_var -> expression in tgt_vars
    """
    back = all(E(sp.cancel(inv[s].xreplace(fwd) - s)) == 0 for s in src_vars)
    fore = all(E(sp.cancel(fwd[g].xreplace(inv) - g)) == 0 for g in tgt_vars)
    return back and fore


def sum_strata(strata):
    """Total class of a list of (name, class) strata."""
    return sp.expand(sum(sp.sympify(c) for _, c in strata))


# --- the cuspidal curve, used everywhere -------------------------------
ss = sp.Symbol("s_")
CUSP_EQ = E(ZZ**2 + Tt**3)
CUSP_PARAM = {ZZ: ss**3, Tt: -ss**2}       # bijective onto the cusp
CUSP_INV = {ss: -ZZ / Tt}                  # valid off the cusp point
CUSP_CLASS = L                             # (L-1) punctured + 1 point


# --- carry-over from RM, re-verified in process ------------------------
def verify_rm_carryover(check):
    """Re-verify the RM facts the RS blocks lean on (A24 re-load rule)."""
    Zc, H_A = E(z + x * w), E(-(y + w**2))

    check("RM carry-over: W = t^3 H - p - p^2 - qZ defines Y_1, irreducible",
          lambda: len(sp.factor_list(W, *Y_GENS)[1]) == 1
          and krull_dim([W], Y_GENS) == 4)
    check("RM carry-over: Y_1 is SMOOTH",
          lambda: krull_dim([W] + [sp.diff(W, g) for g in Y_GENS],
                            Y_GENS) == -1)
    check("RM carry-over: X_1 x A^1 is SINGULAR along a line",
          lambda: krull_dim([R1] + [sp.diff(R1, g) for g in B_GENS],
                            B_GENS) == 1)
    check("RM carry-over: the transition identity holds in A[w]",
          lambda: is_in_ideal(E(KER_Q * Zc + KER_P + KER_P**2
                                + t**3 * (y + w**2)), P, y))
    check("RM carry-over: R4  2(p+1)w = HZ + q  in A[w]",
          lambda: is_in_ideal(E(2 * (KER_P + 1) * w - H_A * Zc - KER_Q),
                              P, y))
    check("RM carry-over: the compressed divisor F = t^3(p+1)",
          lambda: sp.cancel(TO_L[x] * Tt**3 * (Pp + 1)
                            + Tt**3 * (ZZ**2 + Tt**3)) == 0)
    return True
