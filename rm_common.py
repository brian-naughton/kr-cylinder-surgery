#!/usr/bin/env python3
"""Shared machinery for the RM visit (RM-OPS.md, master A26).

RE-LOAD DISCIPLINE (A24, still binding). Nothing is taken from an
earlier session's conversation. `rg_common.verify_carryover` re-verifies
the R∂ facts in process; `verify_rg_carryover` below adds the RG facts
this visit leans on. Every RM runner calls both before doing anything.

STANDING CORRECTIONS (A26 / RM-OPS §1), adopted in all prose here:
  * geometric arrows run  X x A^1 -> X_1 x A^1 -> A^4  (Spec reverses
    the ring inclusions); the tower factors ONE smooth-to-smooth
    birational affine morphism through a SINGULAR intermediate model;
  * X_1 x A^k is never affine space (A26) -- neither single step can be
    an isomorphism, only the composite matters;
  * p, q, Z and the t-localised bridge are DMJP's (arXiv:0903.4278 §7);
  * our results are properties of the ACTION, never separating
    invariants of the space.

This module adds the commutative-algebra tools RM needs and sympy
lacks: saturation, ideal equality/containment, and verified minimal
primes.
"""

from __future__ import annotations

import itertools

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import (Pp, Qq, Tt, ZZ, H_L, TO_L, KGENS, LGENS, krull_dim,
                       order_along, to_L, verify_carryover, z_degree)

#: the A^4 = Spec C[p,q,t,Z] coordinates, and the X_1-cylinder's extra one
Xx = sp.Symbol("X_")                      # the step-1 adjoined generator
A4GENS = (Pp, Qq, Tt, ZZ)
B_GENS = (Pp, Qq, Tt, ZZ, Xx)

#: the tower's two relations, in the (p,q,t,Z,X) presentation
R1 = E(Xx * (Pp + 1) + ZZ**2 + Tt**3)      # = Phi(S): defines X_1 x A^1
N1 = E(Pp + Pp**2 + Qq * ZZ)               # numerator of H over t^3
N2 = E(Pp * ZZ - Xx * Qq)                  # numerator of w over 2t^3

#: the compressed divisor candidate
F_COMP = E(Tt**3 * (Pp + 1))


# --- ideal tools --------------------------------------------------------
def gb(gens, vs, order="grevlex"):
    gens = [E(g) for g in gens if E(g) != 0]
    if not gens:
        return None
    return sp.groebner(gens, *vs, order=order)


def in_ideal(f, gens, vs) -> bool:
    G = gb(gens, vs)
    if G is None:
        return E(f) == 0
    return G.reduce(E(f))[1] == 0


def ideal_contains(gens_big, gens_small, vs) -> bool:
    """Is every generator of the small ideal in the big one?"""
    return all(in_ideal(g, gens_big, vs) for g in gens_small)


def ideal_eq(g1, g2, vs) -> bool:
    return ideal_contains(g1, g2, vs) and ideal_contains(g2, g1, vs)


def saturate(gens, f, vs):
    """I : f^oo, by the Rabinowitsch trick + elimination.

    I : f^oo = (I + (1 - s f)) cap k[vs], with s a fresh variable. An
    ELIMINATION -- measured by the caller before running.
    """
    s = sp.Symbol("s_sat")
    G = sp.groebner([E(g) for g in gens] + [E(1 - s * f)], s, *vs,
                    order="lex")
    return [g for g in G.exprs if s not in g.free_symbols]


def radical_power(f, gens, vs, cap=8):
    """Least k with f^k in the ideal, or None."""
    G = gb(gens, vs)
    if G is None:
        return None
    for k in range(1, cap + 1):
        if G.reduce(E(f**k))[1] == 0:
            return k
    return None


def verify_minimal_primes(gens, primes, vs, cap=8):
    """Certify sqrt(I) = intersection of the given primes.

    Returns a dict of the evidence: each prime contains I; each
    generator of the intersection has a power in I; and the claimed
    intersection is computed as an ideal so the check is two-sided.
    The primes' primality is the caller's separate check.
    """
    ev = {"each_prime_contains_I": all(
        ideal_contains(pr, gens, vs) for pr in primes)}
    # the intersection, computed the standard way: I cap J via elimination
    inter = primes[0]
    for pr in primes[1:]:
        inter = _intersect(inter, pr, vs)
    ev["intersection_generators"] = [str(sp.factor(g)) for g in inter]
    ev["powers_into_I"] = {str(sp.factor(g)): radical_power(g, gens, vs, cap)
                           for g in inter}
    ev["all_powers_found"] = all(v is not None
                                 for v in ev["powers_into_I"].values())
    return ev, inter


def _intersect(g1, g2, vs):
    """I cap J = (u*I + (1-u)*J) cap k[vs]."""
    u = sp.Symbol("u_int")
    G = sp.groebner([E(u * a) for a in g1] + [E((1 - u) * b) for b in g2],
                    u, *vs, order="lex")
    return [g for g in G.exprs if u not in g.free_symbols]


# --- carry-over from RG, re-verified in process -------------------------
def verify_rg_carryover(check):
    """Re-verify the RG facts the RM blocks lean on. A24 re-load rule."""
    Zc = E(z + x * w)
    H_A = E(-(y + w**2))

    check("RG carry-over: R1  x(p+1) + Z^2 + t^3 = 0 in A[w]",
          lambda: is_in_ideal(E(x * (KER_P + 1) + Zc**2 + t**3), P, y))
    check("RG carry-over: R3  t^3 H = p + p^2 + q Z in A[w]",
          lambda: is_in_ideal(E(t**3 * H_A - KER_P - KER_P**2
                                - KER_Q * Zc), P, y))
    check("RG carry-over: R2  2 t^3 w = p Z - x q in A[w]",
          lambda: is_in_ideal(E(2 * t**3 * w - KER_P * Zc + x * KER_Q),
                              P, y))
    check("RG carry-over: the L-forms of x, w, H match the tower data",
          lambda: sp.cancel(to_L(H_A) - H_L) == 0
          and sp.cancel(to_L(x) - TO_L[x]) == 0
          and sp.cancel(to_L(w) - TO_L[w]) == 0)
    check("RG carry-over: pole orders x(0,-1)  H(-3,0)  w(-3,-1)",
          lambda: (order_along(TO_L[x], Tt), order_along(TO_L[x], Pp + 1),
                   order_along(H_L, Tt), order_along(H_L, Pp + 1),
                   order_along(TO_L[w], Tt), order_along(TO_L[w], Pp + 1))
          == (0, -1, -3, 0, -3, -1))
    check("RG carry-over: the two plinth generators t^3(p+1) and q(p+1)",
          lambda: _plinth_gens_ok())
    return True


def _plinth_gens_ok() -> bool:
    from rd_4_plinth import partial_red_components
    red = partial_red_components()

    def d_red(f):
        return E(sum(red[g] * sp.diff(f, g) for g in GENS))
    Zc = E(z + x * w)
    return (is_in_ideal(E(d_red(Zc) - t**3 * (KER_P + 1)), P, y)
            and is_in_ideal(E(d_red(E(y + w**2)) + KER_Q * (KER_P + 1)),
                            P, y))
