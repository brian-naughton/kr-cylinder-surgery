#!/usr/bin/env python3
#
# PUBLICATION COPY. Identical to the working-repo original except
# for interpretive annotation strings, corrected before release so
# that the ledger prose matches the paper. No mathematical check,
# no computation and no verdict differs; see the supersession diff
# shipped with the private record, and VERIFY.md in this
# repository.
"""RM-1 — canonical modification data, and the walls (RM-OPS §4).

(a) THE JACOBIAN FORMALITY. X_1 is singular exactly at the origin, so
    X_1 x A^k is never affine space (A26's kill, logged as a machine
    formality). Neither tower step can be an isomorphism.
(b) THE TWO MODIFICATIONS AS FORMAL PROPOSITIONS: for each inclusion
    A_i subset A_{i+1}, the smallest usable divisor f_i, the SATURATED
    centre I_i, the exceptional divisor and its map to the centre,
    normality data -- certified both ways with H15 non-membership.
(c) sqrt(pl(partial_red)) AND ITS MINIMAL PRIMES (Shakhmatov-informed,
    arXiv:2605.00138): are t and (p+1) intrinsic walls of the action, or
    artefacts of the triangularisation? This calibrates everything after.

Arrows run X x A^1 -> X_1 x A^1 -> A^4 (A26). The objects p, q, Z and
the t-localised bridge are DMJP's, arXiv:0903.4278 §7.

Ledger: runs_synthesis/RM-1-CANONICAL.json. Mode I.
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
from rg_common import (Pp, Qq, Tt, ZZ, H_L, TO_L, KGENS, krull_dim,
                       verify_carryover)
from rm_common import (A4GENS, B_GENS, F_COMP, N1, N2, R1, Xx, gb,
                       ideal_contains, ideal_eq, in_ideal, radical_power,
                       saturate, verify_minimal_primes, verify_rg_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

#: DECLARED BEFORE ANY SEARCH RUNS (RM-OPS §2)
CAPS = {
    "divisor_minimality_tested": ["t", "t^2", "t^3", "p+1", "(p+1)^2"],
    "radical_power_cap": 8,
    "saturation": "Rabinowitsch elimination, 5-6 variables",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
    "note": "RM-1 is structural: no coefficient search runs, so no "
            "degree cap on unknowns applies. Every Groebner call is "
            "measured before running.",
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
    r.record("DECLARED CAPS (before anything ran)", "MEASURED",
             json.dumps(CAPS, sort_keys=True))

    # ==================================================================
    print("\nRM-1a  the Jacobian formality: the A26 kill, as a machine check")
    jac = [sp.diff(R1, g) for g in B_GENS]
    sing = alarmed(lambda: krull_dim([R1] + jac, B_GENS))
    dimX1c = alarmed(lambda: krull_dim([R1], B_GENS))
    r.check("X_1 x A^1 is a 4-fold (R1 is irreducible)",
            lambda: dimX1c == 4
            and len(sp.factor_list(R1, *B_GENS)[1]) == 1)
    r.check("its singular locus is NON-EMPTY and 1-dimensional",
            lambda: sing == 1,
            f"dim Sing = {sing}: the line {{p=-1, q free, t=Z=X=0}} "
            "= {origin of X_1} x A^1_q")
    r.check("the singular point of X_1 itself is exactly the origin",
            lambda: alarmed(lambda: krull_dim(
                [E(Xx * Pp + ZZ**2 + Tt**3)]
                + [sp.diff(E(Xx * Pp + ZZ**2 + Tt**3), g)
                   for g in (Xx, Pp, ZZ, Tt)], (Xx, Pp, ZZ, Tt))) == 0,
            "Jacobian (u, X, 2Z, 3t^2) vanishes only at the origin, "
            "which lies on X_1")
    Qsm = E(ZZ + Pp**2)          # a graph hypersurface: smooth
    r.check("H15 boundary: a SMOOTH hypersurface gets an empty Sing",
            lambda: alarmed(lambda: krull_dim(
                [Qsm] + [sp.diff(Qsm, g) for g in A4GENS], A4GENS)) == -1,
            "so 'dim Sing = 1' above is a finding, not an artefact")
    r.record("THEOREM RM-1.1 (A26's kill, formalised)", "THEOREM",
             "X_1 is singular (at the origin); singularity is preserved "
             "by x A^k; A^{3+k} is smooth. Hence X_1 x A^k is NOT "
             "isomorphic to A^{3+k} for any k >= 0. So NEITHER STEP of "
             "the tower can be an isomorphism, and any success must come "
             "from the COMPOSITE. Certified-trivial; no cancellation "
             "theory is used.")
    led["rm1a"] = {
        "dim_X1_cylinder": int(dimX1c), "dim_Sing": int(sing),
        "kill": "X_1 x A^k is never affine space, for every k >= 0",
        "consequence": "neither tower step is an isomorphism; only the "
                       "composite X x A^1 -> A^4 can be one",
    }

    # ==================================================================
    print("\nRM-1b  STEP 1 as a formal proposition: A^4 <- X_1 x A^1")
    # A_0 = C[p,q,t,Z];  f_0 = p+1;  I_0 = (p+1, Z^2+t^3);  A_1 = A_0[I_0/f_0]
    f0 = E(Pp + 1)
    I0 = [f0, E(ZZ**2 + Tt**3)]
    r.check("f_0 = p+1 lies in the centre I_0 (needed for A[I/f])",
            lambda: in_ideal(f0, I0, A4GENS))
    r.check("A_0[I_0/f_0] = A_0[X] with X = -(Z^2+t^3)/(p+1)",
            lambda: sp.cancel(TO_L[x] + sp.cancel(E(ZZ**2 + Tt**3) / f0))
            == 0, "the only new generator is the second one over f_0")
    # smallest usable divisor: (p+1) is minimal because x has pole order 1
    r.check("f_0 is the SMALLEST usable divisor: x has (p+1)-order exactly -1",
            lambda: sp.cancel(TO_L[x] * f0) == E(-(ZZ**2 + Tt**3))
            and sp.denom(sp.together(sp.cancel(TO_L[x]))) != 1,
            "no proper divisor of p+1 clears it; (p+1) is irreducible")
    # THE RIGHT NOTION OF SATURATED CENTRE. I : f^oo is VACUOUS here --
    # f lies in I by construction, so I : f^oo = (1) always. Logged,
    # because it would silently mislead. The canonical centre of an
    # affine modification A subset B subset A_f is  I = f*B cap A.
    I0sat_wrong = alarmed(lambda: saturate(I0, f0, A4GENS))
    r.check("the naive saturation I_0 : f_0^oo is VACUOUS -- it is (1)",
            lambda: ideal_eq(I0sat_wrong, [sp.Integer(1)], A4GENS),
            "because f_0 lies in I_0; so I : f^oo is the WRONG notion of "
            "saturated centre for a modification, and is not used")
    # I_0 = f_0 * A_1 cap A_0, computed exactly.
    r.check("(p+1) does NOT divide Z^2+t^3 in C[p,q,t,Z]",
            lambda: sp.rem(sp.Poly(E(ZZ**2 + Tt**3), Pp),
                           sp.Poly(f0, Pp)).as_expr() != 0)
    r.record("I_0 = f_0 A_1 cap A_0 = (p+1, Z^2+t^3), EXACTLY", "MEASURED",
             "A_1 = A_0[X] = sum_j A_0 (Z^2+t^3)^j/(p+1)^j. If "
             "a/(p+1) = sum c_j X^j then a = c_0(p+1) - c_1(Z^2+t^3) + "
             "sum_{j>=2} (-1)^j c_j (Z^2+t^3)^j/(p+1)^{j-1}; since "
             "(p+1) is prime and does not divide Z^2+t^3, each j>=2 term "
             "forces (p+1)^{j-1} | c_j and then lands in (Z^2+t^3). So "
             "the centre is exactly (p+1, Z^2+t^3) -- already canonical.")
    # centre and exceptional divisor
    dimC0 = alarmed(lambda: krull_dim(I0, A4GENS))
    r.check("the centre C_0 = V(p+1, Z^2+t^3) is a SURFACE in A^4",
            lambda: dimC0 == 2,
            "the cuspidal curve Z^2+t^3 = 0 times A^1_q, at p = -1")
    dimE0 = alarmed(lambda: krull_dim([R1, f0], B_GENS))
    r.check("the exceptional divisor E_0 = V(p+1) upstairs is a DIVISOR",
            lambda: dimE0 == 3, "dim 3 inside the 4-fold X_1 x A^1")
    r.check("E_0 -> C_0 has 1-dimensional fibres: E_0 = C_0 x A^1_X",
            lambda: dimE0 - dimC0 == 1
            and in_ideal(E(ZZ**2 + Tt**3), [R1, f0], B_GENS),
            "on p+1 = 0 the relation forces Z^2+t^3 = 0 and leaves X free")
    r.check("H15 boundary: X is NOT forced on E_0 (the fibre is not a point)",
            lambda: not in_ideal(Xx, [R1, f0], B_GENS))
    # normality of the middle model
    r.check("X_1 x A^1 is NORMAL: hypersurface (S2) with Sing of codim 3",
            lambda: dimX1c - sing == 3,
            "Serre: S2 from the hypersurface, R1 since codim Sing >= 2")
    led["step1"] = {
        "arrow": "X_1 x A^1 -> A^4",
        "f_0": "p+1  (smallest usable: x has (p+1)-order exactly -1)",
        "I_0": "(p+1, Z^2 + t^3)  -- canonical: equals f_0 A_1 cap A_0",
        "centre": "V(I_0), a surface: the cuspidal curve times A^1_q at "
                  "p = -1",
        "exceptional": "V(p+1) upstairs, a divisor; E_0 -> C_0 is an "
                       "A^1-bundle in the X-direction",
        "normality": "X_1 x A^1 is normal (S2 + Sing of codim 3)",
        "dims": {"centre": int(dimC0), "exceptional": int(dimE0),
                 "ambient": 4},
    }

    # ==================================================================
    print("\nRM-1c  STEP 2 as a formal proposition: X_1 x A^1 <- X x A^1")
    # A_1 = C[p,q,t,Z,X]/(R1);  f_1 = t^3;  I_1 = (t^3, N1, N2)
    f1 = E(Tt**3)
    I1 = [f1, N1, N2]
    r.check("f_1 = t^3 lies in the centre I_1",
            lambda: in_ideal(f1, I1 + [R1], B_GENS))
    r.check("N1 and N2 are the numerators of H and 2w over t^3",
            lambda: sp.cancel(H_L * Tt**3 - N1) == 0
            and sp.cancel(2 * TO_L[w] * Tt**3
                          - N2.xreplace({Xx: TO_L[x]})) == 0)
    # the divisor's SUPPORT is the reduced {t = 0}; the multiplicity is
    # forced by the exact pole order of the new generators.
    clears = {f"t^{k}": bool(sp.denom(sp.together(sp.cancel(H_L * Tt**k)))
                             == 1) for k in (1, 2, 3)}
    r.check("t^3 is the least power of t clearing the new generators",
            lambda: clears == {"t^1": False, "t^2": False, "t^3": True},
            f"{clears}; H has t-order exactly -3")
    # and a DECIDABLE test that f = t with the natural centre cannot do it:
    # every element of A_1[I/t] is a sum of m_k/t^k with m_k in I^k, so
    # H = N1/t^3 needs N1 in I^3 + t I^2 + t^2 I + (t^3).
    Inat = [Tt, N1, N2]
    J = ([E(a * b * c) for a in Inat for b in Inat for c in Inat]
         + [E(Tt * a * b) for a in Inat for b in Inat]
         + [E(Tt**2 * a) for a in Inat] + [E(Tt**3), R1])
    r.record("elimination measurement", "MEASURED",
             f"the f=t test is one Groebner call on {len(J)} generators in "
             f"{len(B_GENS)} variables, against the frozen ceiling of "
             f"{CEILING_VARS}; H7 alarm armed.")
    n1_in_J = alarmed(lambda: in_ideal(N1, J, B_GENS))
    r.check("f = t with the natural centre CANNOT give step 2",
            lambda: not n1_in_J,
            "N1 is not in I^3 + t I^2 + t^2 I + (t^3), so N1/t^3 is not "
            "reachable from elements a/t")
    r.check("H15 boundary: t^3 IS in that ideal -- the test can pass",
            lambda: alarmed(lambda: in_ideal(E(Tt**3), J, B_GENS)))
    # the canonical centre again: I_1 = t^3 A_2 cap A_1, computed exactly
    # by the same argument. It needs t prime in A_1 and t coprime to the
    # two numerators -- both machine-checked.
    A1modt = E(Xx * (Pp + 1) + ZZ**2)          # R1 at t = 0
    r.check("t is PRIME in A_1: A_1/(t) is a domain",
            lambda: len(sp.factor_list(A1modt, *B_GENS)[1]) == 1
            and sp.factor_list(A1modt, *B_GENS)[1][0][1] == 1,
            "X(p+1) + Z^2 is irreducible (degree 1 in X, primitive)")
    r.check("t divides neither N1 nor N2 in A_1",
            lambda: not in_ideal(N1, [Tt, R1], B_GENS)
            and not in_ideal(N2, [Tt, R1], B_GENS))
    r.check("H15 boundary: t DOES divide t^3 there -- the test can pass",
            lambda: in_ideal(E(Tt**3), [Tt, R1], B_GENS))
    # the forward containment, verified unconditionally inside A[w]
    Zc, H_A = E(z + x * w), E(-(y + w**2))
    fwd = {"t^3": is_in_ideal(E(t**3 - t**3), P, y),
           "N1": is_in_ideal(E(KER_P + KER_P**2 + KER_Q * Zc
                               - t**3 * H_A), P, y),
           "N2": is_in_ideal(E(KER_P * Zc - x * KER_Q - t**3 * 2 * w),
                             P, y)}
    r.check("t^3, N1, N2 all lie in t^3 A[w] (forward containment)",
            lambda: all(fwd.values()),
            "N1 = t^3 H and N2 = t^3 (2w), both with H, w in A[w]")
    r.record("I_1 = t^3 A_2 cap A_1 = (t^3, N1, N2), EXACTLY", "MEASURED",
             "same argument as step 1: a/t^3 = sum c_jk H^j W^k gives "
             "a = c_00 t^3 + c_10 N1 + c_01 N2 + higher terms, and each "
             "higher term needs t^{3(j+k-1)} | c_jk N1^j N2^k; t is prime "
             "in A_1 and divides neither numerator, so it divides c_jk "
             "and the term lands in (N1, N2). Canonical.")
    led_sat = ["t^3", str(sp.factor(N1)), str(sp.factor(N2))]
    dimC1 = alarmed(lambda: krull_dim(I1 + [R1], B_GENS))
    r.check("the step-2 centre is a SURFACE inside X_1 x A^1",
            lambda: dimC1 == 2, f"dim = {dimC1}")
    dimE1 = alarmed(lambda: krull_dim([R1, Tt], B_GENS))
    r.check("the exceptional divisor E_1 = V(t) upstairs is a DIVISOR",
            lambda: dimE1 == 3)
    r.check("H15 boundary: the step-2 centre is NOT the whole of V(t)",
            lambda: dimC1 < dimE1,
            "the centre is a proper closed subscheme of the divisor")
    led["step2"] = {
        "arrow": "X x A^1 -> X_1 x A^1",
        "f_1": "t^3  -- reduced divisor {t=0}, multiplicity 3 forced by "
               "the exact pole order of H; f = t with the natural centre "
               "is decidably insufficient",
        "I_1": "(t^3, N1, N2),  N1 = p+p^2+qZ,  N2 = pZ - Xq",
        "I_1_canonical": led_sat,
        "canonical_centre_note": "I : f^oo is vacuous when f is in I; the "
                                 "canonical centre is f*A_{i+1} cap A_i, "
                                 "and both come out already canonical",
        "centre": "a surface inside the divisor V(t)",
        "dims": {"centre": int(dimC1), "exceptional": int(dimE1),
                 "ambient": 4},
        "note": "relative to X_1 x A^1 BOTH new generators are "
                "t^3-divisions, so step 2 is supported on {t = 0} alone",
    }

    # ==================================================================
    print("\nRM-1d  the walls: sqrt(pl) and its minimal primes")
    # the KNOWN part of the plinth, re-derived: (p+1)(t^3, q)
    PLK = [E(Tt**3 * (Pp + 1)), E(Qq * (Pp + 1))]
    r.check("the known plinth ideal is (p+1)(t^3, q)",
            lambda: ideal_eq(PLK, [E((Pp + 1) * Tt**3),
                                   E((Pp + 1) * Qq)], KGENS))
    primes = [[E(Pp + 1)], [Tt, Qq]]
    r.check("(p+1) is prime in C[p,q,t]",
            lambda: len(sp.factor_list(E(Pp + 1), *KGENS)[1]) == 1)
    r.check("(t, q) is prime in C[p,q,t]: the quotient is C[p]",
            lambda: alarmed(lambda: krull_dim([Tt, Qq], KGENS)) == 1)
    ev, inter = alarmed(lambda: verify_minimal_primes(PLK, primes, KGENS))
    r.check("each claimed prime CONTAINS the plinth ideal",
            lambda: ev["each_prime_contains_I"])
    r.check("every generator of the intersection has a power in it",
            lambda: ev["all_powers_found"], f"{ev['powers_into_I']}")
    r.check("H15 boundary: 1 has NO power in the plinth ideal",
            lambda: radical_power(sp.Integer(1), PLK, KGENS) is None,
            "the radical test can fail")
    r.check("H15 boundary: p alone has no power in it either",
            lambda: radical_power(Pp, PLK, KGENS) is None)
    r.record("sqrt of the KNOWN plinth = (p+1) cap (t,q)", "MEASURED",
             f"minimal primes: (p+1) [codim 1] and (t,q) [codim 2]. "
             f"Intersection generators {ev['intersection_generators']}.")

    # the unconditional statement, and what it decides
    r.record("THEOREM RM-1.2 (the walls, unconditionally)", "THEOREM",
             "pl(partial_red) CONTAINS (p+1)(t^3,q) (RG-1(a), re-verified "
             "in process), so its zero locus is CONTAINED in "
             "V(p+1) u V(t,q). Since a LARGER plinth means a SMALLER zero "
             "locus, this containment is unconditional and no hidden "
             "component can lie outside V(p+1) u V(t,q). And "
             "pl is inside (p,q,t) (RG-1), so the zero locus is non-empty "
             "-- it contains the origin.")
    r.record("THE CALIBRATION ANSWER", "MEASURED",
             "the two minimal primes of sqrt of the KNOWN SUBIDEAL "
             "J = (p+1)(t^3,q) are of different kinds: (p+1) is "
             "divisorial, while (t,q) has codimension 2, cut against "
             "q = 0. THIS IS A STATEMENT ABOUT J, NOT ABOUT "
             "pl(partial_red), whose exact value is undetermined "
             "(MQ-D, parked at A25); a larger plinth could only shrink "
             "V(pl) further inside V(J), and NO claim is made about "
             "which of the two components survives in V(pl). What is "
             "unconditional is the ENCLOSURE recorded above. Read "
             "through Shakhmatov arXiv:2605.00138 (plinth zero locus = "
             "complement of the union of principal invariant "
             "cylinders), the surviving direction is: since V(pl) is "
             "inside V(p+1) u V(t,q), the principal invariant cylinders "
             "COVER the complement of those two loci. The converse "
             "reading would need the exact plinth and is not used.")
    led["walls"] = {
        "known_plinth": "(p+1)(t^3, q)",
        "radical": "(p+1) cap (t,q)",
        "minimal_primes": ["(p+1)  codim 1", "(t,q)  codim 2"],
        "unconditional": "V(pl) is contained in V(p+1) u V(t,q); no "
                         "hidden component outside it",
        "asymmetry": "of the KNOWN SUBIDEAL J = (p+1)(t^3,q) only: "
                     "(p+1) is divisorial, (t,q) is codim 2 against "
                     "q = 0. NOT asserted of pl itself, which is "
                     "undetermined",
        "open": "whether pl = (p+1)(t^3,q) exactly (MQ-D)",
        "citation": "Shakhmatov arXiv:2605.00138 (verified at A26)",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RM-1-CANONICAL.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RM-1", "plan": "RM-OPS.md (A26)",
            "date": "2026-08-01", "mode": "I",
            "source": "p, q, Z and the t-localised bridge are DMJP "
                      "arXiv:0903.4278 s7; Shakhmatov arXiv:2605.00138 "
                      "for the plinth reading; the tower data is ours",
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
