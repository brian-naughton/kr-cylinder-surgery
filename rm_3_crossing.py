#!/usr/bin/env python3
"""RM-3 — the crossing V(t, p+1) (RM-OPS §4).

Sol's step (3): do the two modifications commute or trivialise away from
the crossing, and does all genuinely coupled geometry concentrate there?

WHAT COMES OUT, in order of consequence:

  1. They DO trivialise away from each other's divisor, and they COMMUTE:
     the tower can be run in either order. Running the t-step FIRST gives
     a SECOND intermediate model
         Y_1 = V(t^3 H - p - p^2 - q Z)  in  A^5,
     and Y_1 is SMOOTH. So the singular intermediate model is an artefact
     of the ORDER, not intrinsic to the factorisation. A26's kill on X_1
     itself is untouched and still correct.
  2. The transition equation q Z + p + p^2 + t^3(y+w^2) = 0 IS Y_1's
     defining equation. Sol's instinct that it "smells like a transition
     equation" is exactly right, and this is what it is transitioning
     between.
  3. The crossing carries the coupling, scheme-theoretically: on
     V(t,p+1) = A^2_{q,Z} the compressed centre restricts to (q Z^2), and
     the two components of the centre meet along the line {Z = 0} with a
     non-reduced, multiplicity-2 structure.

Arrows run X x A^1 -> X_1 x A^1 -> A^4 (A26). p, q, Z and the
t-localised bridge are DMJP's, arXiv:0903.4278 §7.

Ledger: runs_synthesis/RM-3-CROSSING.json. Mode I.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import Pp, Qq, Tt, ZZ, H_L, TO_L, krull_dim, verify_carryover
from rm_common import (A4GENS, B_GENS, N1, N2, R1, Xx, ideal_contains,
                       ideal_eq, in_ideal, radical_power,
                       verify_minimal_primes, verify_rg_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

Hh = sp.Symbol("H_")
Y_GENS = (Pp, Qq, Tt, ZZ, Hh)
YEQ = E(Tt**3 * Hh - Pp - Pp**2 - Qq * ZZ)     # defines Y_1 in A^5

SQ = E(ZZ**2 + Tt**3)
F = E(Tt**3 * (Pp + 1))
M = E(Pp * (Pp + 1) * ZZ + Qq * SQ)
G_COMP4 = [F, E(Tt**3 * SQ), E((Pp + 1) * N1), M]

CAPS = {
    "searches_run": 0,
    "note": "RM-3 is structural: identities, localisations and scheme "
            "computations only. No coefficient search, so no degree cap "
            "on unknowns. Every Groebner call is measured before running.",
    "groebner_variables_max": 5,
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "MEASURED", "THEOREM", "NOT-ATTEMPTED")

    def __init__(self):
        self.results, self.notes = [], {}

    def check(self, name, cond, note=""):
        t0 = time.time()
        if callable(cond):
            cond = cond()
        dt = time.time() - t0
        outcome = "PASS" if cond else "FAIL"
        self.results.append((name, outcome, dt))
        if note:
            self.notes[name] = note
        print(f"  {name:<58s} {outcome:<10s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<58s} {outcome:<10s}         [{note}]", flush=True)

    @property
    def ok(self):
        return all(o != "FAIL" for _, o, _ in self.results)

    @property
    def tally(self):
        return {k: sum(1 for _, o, _ in self.results if o == k)
                for k in self.CLASSES
                if any(o == k for _, o, _ in self.results)}


def alarmed(fn):
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> int:
    r = Runner()
    led: dict = {}

    print("\nRM-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    verify_carryover(r.check)
    verify_rg_carryover(r.check)
    r.record("DECLARED CAPS", "MEASURED", json.dumps(CAPS, sort_keys=True))

    # ==================================================================
    print("\nRM-3a  away from each divisor, the other step is TRIVIAL")
    r.check("over D(t): the t-step vanishes -- H and 2w lie in B[w][1/t]",
            lambda: sp.denom(sp.together(sp.cancel(H_L * Tt**3))) == 1
            and sp.denom(sp.together(sp.cancel(
                2 * TO_L[w] * Tt**3 * (Pp + 1)))) == 1,
            "their numerators N1 and M involve only p,q,Z and x")
    r.check("over D(p+1): the (p+1)-step vanishes -- x lies in C^[4][1/(p+1)]",
            lambda: sp.denom(sp.together(sp.cancel(TO_L[x] * (Pp + 1))))
            == 1)
    r.check("H15 boundary: neither step is trivial on the nose",
            lambda: sp.denom(sp.together(sp.cancel(TO_L[x]))) != 1
            and sp.denom(sp.together(sp.cancel(H_L))) != 1)
    led["localisation"] = {
        "over_D(t)": "only the (p+1)-step survives",
        "over_D(p+1)": "only the t-step survives",
        "conclusion": "the two modifications are supported on disjoint "
                      "divisors and can only interact over the crossing "
                      "V(t, p+1)",
    }

    # ==================================================================
    print("\nRM-3b  THE COMMUTING SQUARE, and a SECOND intermediate model")
    # run the t-step FIRST: C_1 = C^[4][H], then adjoin x and W over (p+1).
    r.check("R4 gives 2w = (H Z + q)/(p+1): a (p+1)-division over C^[4][H]",
            lambda: is_in_ideal(E(2 * (KER_P + 1) * w
                                  - (-(y + w**2)) * (z + x * w) - KER_Q),
                                P, y),
            "so after adjoining H, the remaining step is (p+1) only")
    r.check("A[w] = C^[4][H][x, 2w]: the five generators are recovered",
            lambda: sp.cancel(TO_L[y] - (-H_L - TO_L[w] ** 2)) == 0
            and sp.cancel(TO_L[z] - (ZZ - TO_L[x] * TO_L[w])) == 0,
            "y = -H - w^2 and z = Z - x w, exactly as in the other order")
    r.record("THE SQUARE COMMUTES", "MEASURED",
             "A^4 --(p+1)--> X_1 x A^1 --(t^3)--> X x A^1  and  "
             "A^4 --(t^3)--> Y_1 --(p+1)--> X x A^1 are two factorisations "
             "of the SAME birational morphism (arrows as ring inclusions; "
             "geometrically they reverse). The second intermediate model "
             "is Y_1 = Spec C^[4][H].")
    # Y_1 as a hypersurface, and its smoothness
    r.check("Y_1 = V(t^3 H - p - p^2 - q Z) in A^5, and the equation is "
            "irreducible",
            lambda: len(sp.factor_list(YEQ, *Y_GENS)[1]) == 1
            and alarmed(lambda: krull_dim([YEQ], Y_GENS)) == 4)
    ysing = alarmed(lambda: krull_dim(
        [YEQ] + [sp.diff(YEQ, g) for g in Y_GENS], Y_GENS))
    r.check("Y_1 IS SMOOTH: its singular locus is EMPTY",
            lambda: ysing == -1,
            "the Jacobian (-1-2p, -Z, 3t^2 H, -q, t^3) forces "
            "p=-1/2, Z=q=t=0, and that point is NOT on Y_1 (value 1/4)")
    r.check("H15 boundary: the SAME routine finds X_1 x A^1 singular",
            lambda: alarmed(lambda: krull_dim(
                [R1] + [sp.diff(R1, g) for g in B_GENS], B_GENS)) == 1,
            "so 'empty' above is a finding, not a defect of the routine")
    r.record("THEOREM RM-3.1: the singular middle is an ARTEFACT OF ORDER",
             "THEOREM",
             "The same birational morphism X x A^1 -> A^4 factors through "
             "the SINGULAR X_1 x A^1 (t-step last) and through the SMOOTH "
             "Y_1 (t-step first). A26's kill stands untouched -- X_1 x A^k "
             "is still never affine space -- but the reading that the "
             "factorisation must pass through a singular model is "
             "presentation-dependent. A smooth intermediate model exists, "
             "and it is explicit.")
    led["square"] = {
        "order_A": "A^4 <- X_1 x A^1 <- X x A^1  (p+1 first, then t^3)",
        "order_B": "A^4 <- Y_1 <- X x A^1        (t^3 first, then p+1)",
        "Y_1": "V(t^3 H - p - p^2 - q Z) in A^5",
        "Y_1_smooth": True, "X_1_cylinder_singular": True,
        "reading": "the singular middle is an artefact of the ORDER; A26's "
                   "kill on X_1 itself is unaffected",
    }

    # ==================================================================
    print("\nRM-3c  the transition equation IS Y_1's equation")
    Zc, H_A = E(z + x * w), E(-(y + w**2))
    r.check("in A[w]:  q Z + p + p^2 + t^3 (y + w^2) = 0",
            lambda: is_in_ideal(E(KER_Q * Zc + KER_P + KER_P**2
                                  + t**3 * (y + w**2)), P, y))
    r.check("rewritten with H = -(y+w^2):  Z q - t^3 H = -p(1+p)",
            lambda: E(-(YEQ).xreplace({Hh: Hh}) - (Qq * ZZ + Pp + Pp**2
                                                   - Tt**3 * Hh)) == 0)
    # completing the square exposes the shape
    Pn = sp.Symbol("Pn_")
    quad = E(YEQ.xreplace({Pp: Pn - sp.Rational(1, 2)}))
    r.check("completing the square: Y_1 is  Z q - t^3 H + P'^2 = 1/4",
            lambda: E(quad + (Qq * ZZ - Tt**3 * Hh + Pn**2
                              - sp.Rational(1, 4))) == 0,
            "P' = p + 1/2; a deformed affine quadric, with t^3 in place "
            "of a variable")
    r.record("what the transition equation IS", "MEASURED",
             "the identity that RG-1 found governing the whole t=0 "
             "pathology is precisely the defining equation of the SMOOTH "
             "intermediate model. Sol's guess that it 'smells like a "
             "transition equation' is confirmed, and this is what it "
             "transitions between: C^[4] and Y_1.")
    led["transition"] = {
        "equation": "Z q - t^3 H = -p(1+p),  i.e. q Z + p + p^2 + "
                    "t^3(y+w^2) = 0",
        "identification": "it is the defining equation of Y_1",
        "normal_form": "Z q - t^3 H + P'^2 = 1/4 with P' = p + 1/2",
    }

    # ==================================================================
    print("\nRM-3d  the fibration of Y_1 over the t-line")
    r.check("over t != 0, Y_1 solves for H: the fibres are A^3",
            lambda: sp.denom(sp.together(sp.cancel(
                E(Pp + Pp**2 + Qq * ZZ) / Tt**3))) != 1,
            "H = (p+p^2+qZ)/t^3 -- a graph once t is inverted")
    y0 = alarmed(lambda: krull_dim([YEQ.subs(Tt, 0)], (Pp, Qq, ZZ, Hh)))
    y0s = alarmed(lambda: krull_dim(
        [YEQ.subs(Tt, 0)] + [sp.diff(YEQ.subs(Tt, 0), g)
                             for g in (Pp, Qq, ZZ, Hh)], (Pp, Qq, ZZ, Hh)))
    r.check("the t = 0 fibre of Y_1 is a 3-fold and is SMOOTH",
            lambda: y0 == 3 and y0s == -1,
            "{p + p^2 + qZ = 0} x A^1_H: a smooth affine quadric surface "
            "times a line")
    r.record("the degeneration", "MEASURED",
             "Y_1 -> A^1_t is an A^3-fibration over t != 0 and degenerates "
             "over t = 0 to (smooth affine quadric surface) x A^1_H. The "
             "quadric surface p+p^2+qZ = 0 is smooth but NOT contractible, "
             "so the t = 0 fibre is not an A^3. Whether Y_1 itself is A^4 "
             "is NOT decided here and is routed to master.")
    led["fibration"] = {
        "over_t_nonzero": "A^3 fibres (H is a graph)",
        "over_t_zero": "smooth quadric surface x A^1_H, dim 3",
        "not_decided": "whether Y_1 = A^4",
    }

    # ==================================================================
    print("\nRM-3e  the crossing scheme")
    cross = [Tt, E(Pp + 1)]
    dimX = alarmed(lambda: krull_dim(cross, A4GENS))
    r.check("the crossing V(t, p+1) is a PLANE A^2_{q,Z} in A^4",
            lambda: dimX == 2)
    # restrict the compressed centre to the crossing
    restr = [E(g.subs({Tt: 0, Pp: -1})) for g in G_COMP4]
    r.check("the compressed centre restricts to (q Z^2) on the crossing",
            lambda: alarmed(lambda: ideal_eq(
                [g for g in restr if g != 0], [E(Qq * ZZ**2)], (Qq, ZZ))),
            f"restricted generators {[str(sp.factor(g)) for g in restr]}")
    r.check("H15 boundary: it is NOT the unit ideal there",
            lambda: not alarmed(lambda: in_ideal(
                sp.Integer(1), [g for g in restr if g != 0], (Qq, ZZ))))
    # how the two components meet
    C_t, C_p = [Tt, N1], [E(Pp + 1), SQ]
    meet = C_t + C_p
    dmeet = alarmed(lambda: krull_dim(meet, A4GENS))
    r.check("the two centre components meet in a LINE",
            lambda: dmeet == 1, "{t = 0, p = -1, Z = 0}, with q free")
    r.check("and they meet NON-REDUCEDLY: Z^2 is in the intersection "
            "ideal but Z is not",
            lambda: alarmed(lambda: in_ideal(E(ZZ**2), meet, A4GENS))
            and not alarmed(lambda: in_ideal(ZZ, meet, A4GENS)),
            "multiplicity 2 in the Z-direction -- the cusp's signature")
    r.record("THE CROSSING ANATOMY", "MEASURED",
             "Sol's expectation is confirmed and made precise. The two "
             "modifications trivialise away from each other's divisor, "
             "they commute, and everything coupled sits over the crossing "
             "V(t,p+1) = A^2_{q,Z}: there the compressed centre restricts "
             "to (q Z^2), and the smooth and cuspidal components of the "
             "centre meet along the line {Z=0} with multiplicity 2. The "
             "doubling is the cusp Z^2 + t^3 seen at t = 0.")
    led["crossing"] = {
        "crossing": "V(t, p+1) = A^2_{q,Z}",
        "centre_restricted": "(q Z^2) -- non-reduced",
        "components_meet_in": "the line {t=0, p=-1, Z=0}, q free",
        "multiplicity": "2 in Z (Z^2 in the intersection ideal, Z not)",
        "reading": "the coupled geometry concentrates scheme-theoretically "
                   "at the crossing, exactly as Sol predicted",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RM-3-CROSSING.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RM-3", "plan": "RM-OPS.md (A26)",
            "date": "2026-08-01", "mode": "I",
            "source": "p, q, Z and the t-localised bridge are DMJP "
                      "arXiv:0903.4278 s7; the square, Y_1 and the "
                      "crossing anatomy are ours",
            "declared_caps": dict(CAPS),
            "sections": led,
            "entries": [{"check": n, "outcome": o,
                         "wall_time_s": round(tt, 4)}
                        for n, o, tt in r.results],
            "notes": r.notes, "tally": r.tally,
            "total_checks": len(r.results),
            "exit": 0 if r.ok else 1,
        }, fh, indent=2, sort_keys=True)
    print(f"ledger written: {path}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
