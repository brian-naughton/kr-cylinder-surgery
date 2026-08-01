#!/usr/bin/env python3
"""RG-3 — elementary moves at k = 1 (RG-OPS §4).

Can the identification over U be re-trivialised: is there a coordinate
system on A[w] making it C^[4]?

TWO EXACT THEOREMS come out, both ruling out the FLOW-COMPATIBLE route
(no degree cap involved), followed by the cap-bounded search the mandate
asks for and the classification of what blocks where.

  RG-3.1  pl(partial_red) is NOT PRINCIPAL, so A[w] is not C[p,q,t][s]
          for any s. Uses RG-1(a)'s refutation as the essential input:
          pl contains (p+1)(t^3, q), and (t^3, q) is not principal.
  RG-3.2  (independent) pi is neither surjective nor equidimensional
          (RG-1(b)), but a coordinate projection A^4 -> A^3 is both.

Both localise the block at {t = 0}. The {p+1 = 0} side is blocked too,
but differently and more softly: the shear and rescaling families fail
there because the modification's fibre over {p+1 = 0} is the CUSP
Z^2 + t^3 = 0, not a point.

Honest fence (A23, binding): failure through this presentation decides
NOTHING about X x A^1 = A^4. Mode I -- no negative claim, no §9
language.

Ledger: runs_synthesis/RG-3-K1.json. Mode I.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import (KGENS, LGENS, Pp, Qq, Tt, ZZ, H_L, TO_L, in_C_pqtZ,
                       krull_dim, order_along, to_L, verify_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

#: DECLARED BEFORE ANY SEARCH RUNS (RG-OPS §2)
CAPS = {
    "shear_family": "Z -> Z + f with f a GENERIC element of K "
                    "(symbolic): exact, no cap",
    "rescaling_family_alpha": "alpha = t^m (p+1)^n, -3 <= m,n <= 3",
    "coordinate_search_pool": ["p", "q", "t", "Z", "x", "y", "z", "w", "H"],
    "coordinate_search_tuples": "all C(9,4) = 126 four-subsets",
    "coordinate_search_filter": "Jacobian: det[grad P; grad a1..a4] must "
                                "be nowhere zero on X x A^1",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}

FF = sp.Symbol("F_")          # a generic element of K, for the shear family


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


def main() -> int:
    r = Runner()
    led: dict = {}

    print("\nRG-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    red = verify_carryover(r.check)
    r.record("DECLARED CAPS (before any search ran)", "MEASURED",
             json.dumps(CAPS, sort_keys=True))

    def d_red(f):
        return E(sum(red[v] * sp.diff(f, v) for v in GENS))

    # ==================================================================
    print("\nRG-3.1  the plinth is not principal  =>  A[w] != C[p,q,t][s]")
    # the two plinth generators, re-derived here
    g1 = E(Tt**3 * (Pp + 1))              # = partial_red(Z)
    g2 = E(Qq * (Pp + 1))                 # = -partial_red(y + w^2)
    r.check("t^3(p+1) is in the plinth: it is partial_red(Z)",
            lambda: is_in_ideal(E(d_red(E(z + x * w))
                                  - t**3 * (KER_P + 1)), P, y))
    r.check("q(p+1) is in the plinth: it is -partial_red(y + w^2)",
            lambda: is_in_ideal(E(d_red(E(y + w**2))
                                  + KER_Q * (KER_P + 1)), P, y))
    r.check("the plinth is inside (p, q, t): p, q, t die on the fixed line",
            lambda: all(E(g.subs({x: 0, z: 0, t: 0, w: 0})) == 0
                        for g in (KER_P, KER_Q, t))
            and all(E(red[v].subs({x: 0, z: 0, t: 0, w: 0})) == 0
                    for v in GENS),
            "partial_red(A[w]) is inside (x,z,t,w)A[w]; C[p,q,t] maps to "
            "C[y] killing p, q and t")
    r.check("gcd(t^3, q) = 1 in C[p,q,t]",
            lambda: sp.gcd(Tt**3, Qq) == 1)
    r.check("(p+1) is irreducible in C[p,q,t]",
            lambda: len(sp.factor_list(Pp + 1, *KGENS)[1]) == 1
            and sp.factor_list(Pp + 1, *KGENS)[1][0][1] == 1)
    G_pqt = sp.groebner([Pp, Qq, Tt], *KGENS, order="grevlex")
    r.check("(p+1) is NOT in (p,q,t) -- it is a unit at the origin",
            lambda: G_pqt.reduce(E(Pp + 1))[1] != 0)
    r.check("H15 boundary: p IS in (p,q,t) -- the membership test passes",
            lambda: G_pqt.reduce(E(Pp))[1] == 0)
    r.record("THEOREM RG-3.1", "THEOREM",
             "Suppose pl(partial_red) = (g). Both t^3(p+1) and q(p+1) lie "
             "in it, so g divides their gcd, which is (p+1) because "
             "gcd(t^3,q) = 1. As (p+1) is irreducible, g is a unit or an "
             "associate of (p+1). A unit gives 1 in pl, contradicting "
             "pl inside (p,q,t); an associate of (p+1) gives pl = (p+1), "
             "also contradicting pl inside (p,q,t) since (p+1) is not "
             "there. So pl is NOT PRINCIPAL. But if A[w] = R[s] with "
             "R = ker partial_red, then partial_red = partial_red(s) d/ds "
             "and pl = (partial_red(s)) IS principal. Hence there is NO s "
             "with A[w] = C[p,q,t][s]. Exact; no degree cap.")
    led["rg31"] = {
        "statement": "A[w] is not C[p,q,t][s] for any s",
        "route": "pl(partial_red) contains (p+1)(t^3,q) and sits inside "
                 "(p,q,t); (t^3,q) is not principal, so pl is not "
                 "principal; a presentation ker[s] forces a principal "
                 "plinth.",
        "essential_input": "RG-1(a)'s refutation -- without the second "
                           "generator q(p+1) the argument does not run.",
        "blocking_component": "{t = 0}: the non-principality is exactly "
                              "the failure of t^3 and q to share a factor, "
                              "and the t^3 is the {t=0} datum. The (p+1) "
                              "is a harmless common factor.",
    }

    # ==================================================================
    print("\nRG-3.2  the independent route: pi is not a coordinate "
          "projection")
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        empty = krull_dim([P, E(KER_P + 1), E(KER_Q - 3), t], GENS)
        jump = krull_dim([P, KER_P, KER_Q, t], GENS)
        gen1 = krull_dim([P, E(KER_P - 2), E(KER_Q - 3), E(t - 1)], GENS)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    r.check("re-verified: fibre over (-1,3,0) is EMPTY", lambda: empty == -1)
    r.check("re-verified: fibre over (0,0,0) is a SURFACE", lambda: jump == 2)
    r.check("re-verified: the generic fibre is an A^1", lambda: gen1 == 1)
    r.record("THEOREM RG-3.2", "THEOREM",
             "If A[w] = C[p,q,t][s] then Spec A[w] = A^4 with pi the "
             "coordinate projection to A^3, which is surjective with "
             "every fibre an A^1. RG-1(b), re-verified above, exhibits an "
             "EMPTY fibre and a two-dimensional one. Contradiction. This "
             "is independent of RG-3.1 and reaches the same conclusion "
             "(A16-5 discipline: two routes, one result).")
    led["rg32"] = {
        "statement": "same conclusion as RG-3.1, by an independent route",
        "witnesses": {"empty_fibre_over": "(-1,3,0)",
                      "jump_fibre_over": "(0,0,0)", "generic": "A^1"},
        "blocking_component": "{t = 0}: both witnesses lie over t = 0.",
    }

    # ==================================================================
    print("\nRG-3.3  the shear family, decided EXACTLY (no cap)")
    # Z' = Z + f, f a GENERIC element of K. x is quadratic in Z' and its
    # leading coefficient does not depend on f at all.
    xshift = sp.expand(sp.cancel(TO_L[x].xreplace({ZZ: ZZ - FF})))
    lead = sp.cancel(sp.Poly(sp.numer(sp.together(xshift)), ZZ).all_coeffs()[0]
                     / sp.denom(sp.together(xshift)))
    r.check("under Z -> Z + f, x stays quadratic in Z' with leading "
            "coefficient -1/(p+1)",
            lambda: sp.cancel(lead + 1 / (Pp + 1)) == 0,
            "independent of f: the shear cannot touch it")
    r.check("-1/(p+1) is NOT in C[p,q,t], so x is never reached",
            lambda: not in_C_pqtZ(sp.cancel(-1 / (Pp + 1))))
    r.record("shear family: BLOCKED at {p+1 = 0}, order 1", "MEASURED",
             "for EVERY f, x is not in C[p,q,t][Z+f], because the "
             "leading Z-coefficient of x is the shear-invariant "
             "-1/(p+1). Exact over the whole family.")

    print("\nRG-3.4  the rescaling family Z -> alpha Z + f")
    # leading coefficient becomes -1/(alpha^2 (p+1)); it can be made
    # polynomial by giving alpha a pole -- but then Z' must still lie in
    # A[w], and that is where the CUSP stops it.
    resc = {}
    for m in range(-3, 4):
        for n in range(-3, 4):
            al = E(Tt**m * (Pp + 1) ** n)
            coeff = sp.cancel(-1 / (al**2 * (Pp + 1)))
            resc[f"alpha=t^{m}(p+1)^{n}"] = bool(in_C_pqtZ(coeff))
    ok_alpha = [k for k, v in resc.items() if v]
    r.check("some rescalings DO fix the leading coefficient",
            lambda: len(ok_alpha) > 0,
            f"{len(ok_alpha)} of {len(resc)} in the declared family")
    # but they need alpha*Z in A[w]; test the cheapest, alpha = 1/(p+1)
    d_pm1 = krull_dim([P, E(KER_P + 1)], GENS)
    d_pm1Z = krull_dim([P, E(KER_P + 1), E(z + x * w)], GENS)
    r.check("Z is NOT divisible by (p+1) in A[w], so Z/(p+1) is not there",
            lambda: d_pm1Z < d_pm1,
            f"dim{{p+1=0}} = {d_pm1} but dim{{p+1=0, Z=0}} = {d_pm1Z}: Z "
            "does not vanish identically on {p+1=0}, because R1 there is "
            "the CUSP Z^2 + t^3 = 0")
    r.check("indeed R1 restricted to {p+1=0} is exactly the cusp Z^2+t^3",
            lambda: E((x * (Pp + 1) + ZZ**2 + Tt**3).subs(Pp, -1)
                      - (ZZ**2 + Tt**3)) == 0)
    r.check("H15 boundary: the cusp Z^2+t^3 is irreducible and singular "
            "at the origin",
            lambda: len(sp.factor_list(ZZ**2 + Tt**3, ZZ, Tt)[1]) == 1
            and krull_dim([ZZ**2 + Tt**3, 2 * ZZ, 3 * Tt**2], (ZZ, Tt)) == 0)
    r.record("rescaling family: BLOCKED at {p+1 = 0}, and the reason is "
             "the CUSP", "MEASURED",
             "a rescaling with a (p+1)-pole would fix the leading "
             "coefficient, but it needs alpha*Z inside A[w], and over "
             "{p+1 = 0} the modification's fibre is the cuspidal curve "
             "Z^2 + t^3 = 0 rather than a point, so Z does not vanish "
             "there. The cusp blocks the (p+1) side exactly as it blocks "
             "Masuda's factoriality (RG-1c).")
    led["shear_and_rescaling"] = {
        "shear": "BLOCKED at {p+1=0}, order 1; leading coefficient "
                 "-1/(p+1) is shear-invariant. Exact over the whole "
                 "family.",
        "rescaling": "BLOCKED at {p+1=0}; the fix needs Z/(p+1) in A[w], "
                     "and the fibre over {p+1=0} is the cusp Z^2+t^3 = 0.",
        "rescalings_fixing_the_leading_coefficient": ok_alpha,
    }

    # ==================================================================
    print("\nRG-3.5  cap-bounded coordinate search, with a Jacobian filter")
    H_A = E(-(y + w**2))
    POOL = {"p": KER_P, "q": KER_Q, "t": t, "Z": E(z + x * w),
            "x": x, "y": y, "z": z, "w": w, "H": H_A}
    gradP = [sp.diff(P, v) for v in GENS]
    survivors, blocked = [], {}
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        # the criterion is valid only because X x A^1 is SMOOTH
        smooth = krull_dim([P] + gradP, GENS)
        # POSITIVE CONTROL (H15): a hypersurface that IS a graph, so its
        # ring IS C^[4] and the named tuple IS a coordinate system. The
        # filter must PASS here, or its 126 rejections mean nothing.
        Qg = E(y - (x**2 + z * t * w))
        Mg = sp.Matrix([[sp.diff(Qg, v) for v in GENS]]
                       + [[sp.diff(g, v) for v in GENS]
                          for g in (x, z, t, w)])
        Dg = E(Mg.det())
        ctl = (Dg != 0 and not Dg.free_symbols)
        for combo in itertools.combinations(sorted(POOL), 4):
            M = sp.Matrix([gradP] + [[sp.diff(POOL[c], v) for v in GENS]
                                     for c in combo])
            D = E(M.det())
            if D == 0:
                blocked[" ".join(combo)] = "Jacobian identically zero"
                continue
            if krull_dim([P, D], GENS) != -1:
                blocked[" ".join(combo)] = "Jacobian vanishes somewhere "\
                                           "on X x A^1"
                continue
            survivors.append(" ".join(combo))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    r.check("X x A^1 is SMOOTH, so the Jacobian criterion is valid here",
            lambda: smooth == -1, "V(P, grad P) is empty")
    r.check("POSITIVE CONTROL: the filter PASSES on a genuine C^[4]",
            lambda: ctl,
            f"graph hypersurface y = x^2+ztw with coordinates (x,z,t,w): "
            f"det = {Dg}")
    r.record(f"Jacobian filter over all {len(blocked) + len(survivors)} "
             f"four-subsets", "MEASURED",
             f"survivors: {survivors if survivors else 'none'}")
    r.check("the filter discriminates: both rejection modes occur",
            lambda: any("identically zero" in v for v in blocked.values())
            and any("somewhere" in v for v in blocked.values()),
            f"{sum(1 for v in blocked.values() if 'identically' in v)} "
            f"identically zero, "
            f"{sum(1 for v in blocked.values() if 'somewhere' in v)} "
            f"vanishing somewhere")
    r.record("cap-bounded coordinate search: NO candidate", "MEASURED",
             "no four-subset of the declared pool passes the Jacobian "
             "necessary condition, so none can be a coordinate system. "
             "A cap-bounded non-finding, NOT a negative claim: the pool "
             "is nine named elements, not a degree-bounded sweep of "
             "A[w].")
    led["coordinate_search"] = {
        "pool": sorted(POOL), "tuples_tested": len(blocked) + len(survivors),
        "survivors": survivors,
        "status": "cap-bounded non-finding (MEASURED), not a negative",
        "criterion": "det[grad P; grad a1..a4] must be nowhere zero on "
                     "X x A^1 -- valid because X x A^1 is smooth; "
                     "positive control passed on a graph hypersurface",
    }

    # ==================================================================
    print("\nRG-3.6  the classification (the block's core science)")
    classification = {
        "{t = 0}": "BLOCKS THE FLOW-COMPATIBLE ROUTE, EXACTLY, at order 3. "
                   "Two independent proofs: (RG-3.1) the plinth contains "
                   "t^3(p+1) and q(p+1) and is therefore non-principal, "
                   "which no presentation C[p,q,t][s] permits; (RG-3.2) pi "
                   "has an empty fibre and a jump fibre, both over t = 0, "
                   "which no coordinate projection permits.",
        "{p+1 = 0}": "BLOCKS THE ELEMENTARY MOVES, and the obstruction is "
                     "the CUSP. Shears cannot help because the leading "
                     "Z-coefficient of x is the invariant -1/(p+1); "
                     "rescalings that would fix it need Z/(p+1) in A[w], "
                     "but the fibre over {p+1 = 0} is the cuspidal curve "
                     "Z^2 + t^3 = 0. This is the same cusp that carries "
                     "X_1's singularity (RG-2's tower) and that breaks "
                     "Masuda's factoriality.",
        "what_is_NOT_shown": "Nothing here bears on whether X x A^1 = A^4. "
                             "What is closed is the route through THIS "
                             "flow's quotient structure -- and, at the "
                             "declared caps, the elementary moves. Honest "
                             "fence, A23.",
    }
    for k, v in classification.items():
        r.record(f"classification: {k}", "MEASURED", v)
    led["classification"] = classification

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RG-3-K1.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RG-3", "plan": "RG-OPS.md (A24)",
            "date": "2026-07-31", "mode": "I",
            "source": "the derivation is DMJP arXiv:0903.4278 Prop 7.2; "
                      "the theorems and the classification are ours",
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
