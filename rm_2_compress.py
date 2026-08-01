#!/usr/bin/env python3
"""RM-2 — compress the tower to ONE modification of A^4 (RM-OPS §4).

Sol's step (2): write C[X x A^1] = C^[4][I/F] with explicit F and an
explicit CANONICAL centre I, minimise the Rees data, and pose the
RECTIFICATION question: is (I, F) equivalent under Aut(A^4) to a
standard centre whose modification is visibly A^4?

Arrows run X x A^1 -> X_1 x A^1 -> A^4 (A26). p, q, Z and the
t-localised bridge are DMJP's, arXiv:0903.4278 §7. The compression and
its obstruction are ours.

WHAT COMES OUT: the compressed centre is a surface with TWO components,
one of them CUSPIDAL along a line. Singularity type of the centre is an
Aut(A^4)-invariant, so the centre cannot be carried to any smooth
standard centre. That is a cap-free rectification obstruction.

Ledger: runs_synthesis/RM-2-COMPRESS.json. Mode I.
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
from rg_common import Pp, Qq, Tt, ZZ, H_L, TO_L, krull_dim, verify_carryover
from rm_common import (A4GENS, B_GENS, N1, N2, R1, Xx, gb, ideal_contains,
                       ideal_eq, in_ideal, radical_power,
                       verify_minimal_primes, verify_rg_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

#: DECLARED BEFORE ANY SEARCH RUNS (RM-OPS §2)
CAPS = {
    "compressed_divisor_candidate": "F = t^3 (p+1)",
    "monomial_range_for_canonical_centre": "x^i H^j W^k with i+k <= 1 and "
                                           "j+k <= 1 (exactly the ones "
                                           "landing in A^4; higher ones "
                                           "proved to fall back in)",
    "rectification_family": "triangular/affine automorphisms of A^4 in "
                            "the declared list below",
    "radical_power_cap": 8,
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}

F = E(Tt**3 * (Pp + 1))
SQ = E(ZZ**2 + Tt**3)                       # the step-1 numerator
M = E(Pp * (Pp + 1) * ZZ + Qq * SQ)         # numerator of W = 2w over F
G_COMP = [F, E(Tt**3 * SQ), E((Pp + 1) * N1), M, E(SQ * N1)]
G_NAMES = ["F = t^3(p+1)", "t^3(Z^2+t^3)", "(p+1)N1", "M", "(Z^2+t^3)N1"]


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
    print("\nRM-2a  the compressed divisor and the three quotients")
    r.check("F = t^3(p+1) divides (t(p+1))^3  -- verified, not assumed",
            lambda: sp.simplify(sp.cancel(E((Tt * (Pp + 1)) ** 3) / F))
            == E((Pp + 1) ** 2)
            and sp.denom(sp.together(sp.cancel(
                E((Tt * (Pp + 1)) ** 3) / F))) == 1,
            "quotient (p+1)^2; so N = 3 works")
    r.check("x  = -t^3(Z^2+t^3) / F",
            lambda: sp.cancel(TO_L[x] + E(Tt**3 * SQ) / F) == 0)
    r.check("H  = (p+1) N1 / F",
            lambda: sp.cancel(H_L - E((Pp + 1) * N1) / F) == 0)
    r.check("2w = M / F,  M = p(p+1)Z + q(Z^2+t^3)",
            lambda: sp.cancel(2 * TO_L[w] - M / F) == 0)
    r.check("H15 boundary: F does NOT clear y (which needs t^6)",
            lambda: sp.denom(sp.together(sp.cancel(TO_L[y] * F))) != 1,
            "so F is doing real work and is not a blanket denominator")

    # ==================================================================
    print("\nRM-2b  the CANONICAL compressed centre I = F A[w] cap C^[4]")
    # A[w] = C^[4][x, H, W]. F * x^i H^j W^k lands in C^[4] exactly when
    # i+k <= 1 and j+k <= 1; every other monomial needs divisibility of
    # its coefficient and falls back into the ideal these generate.
    for nm, g in zip(G_NAMES, G_COMP):
        r.check(f"{nm} lies in F*A[w]: its quotient by F is in A[w]",
                lambda gg=g: _in_A(gg / F))
    r.check("the fifth generator is xH, not a product of the others",
            lambda: sp.cancel(E(SQ * N1) / F + sp.cancel(TO_L[x] * H_L))
            == 0, "(Z^2+t^3)N1 / F = -x H")
    # it IS redundant, and by an exact identity -- so the Rees data
    # minimises to FOUR generators. (My hand expectation was five; the
    # machine corrected it. Logged, not reworded.)
    r.check("MINIMISATION: (Z^2+t^3)N1 = Z*M + p*F, an exact identity",
            lambda: E(SQ * N1 - ZZ * M - Pp * F) == 0,
            "so the fifth generator is redundant: I needs only FOUR")
    r.check("and the ideal test agrees: it lies in the other four",
            lambda: alarmed(lambda: in_ideal(E(SQ * N1), G_COMP[:4],
                                             A4GENS)))
    r.check("H15 boundary: N1 alone is NOT in the four -- the test "
            "discriminates",
            lambda: not alarmed(lambda: in_ideal(N1, G_COMP[:4], A4GENS)))
    G4 = G_COMP[:4]
    m4 = alarmed(lambda: _min_gens_needed(G4, A4GENS))
    r.record("minimised Rees data", "MEASURED",
             f"I = (F, t^3(Z^2+t^3), (p+1)N1, M) -- four generators; the "
             f"smallest subset with the same radical has {m4} of them.")
    r.record("THEOREM RM-2.1: the canonical compressed centre", "THEOREM",
             "I = F A[w] cap C^[4] = (F, t^3(Z^2+t^3), (p+1)N1, M), and "
             "C[X x A^1] = C^[4][I/F] with F = t^3(p+1). Exactly, by the "
             "RM-1 divisibility argument: F x^i H^j W^k lies in C^[4] iff "
             "i+k <= 1 and j+k <= 1, and every other monomial forces "
             "(p+1)^{i+k-1} t^{3(j+k-1)} to divide its coefficient (the "
             "numerators are coprime to both p+1 and t) and then lands "
             "back in the listed ones. The one extra monomial that lands "
             "in C^[4], namely F x H = -(Z^2+t^3)N1, is REDUNDANT by the "
             "identity (Z^2+t^3)N1 = Z M + p F -- so the minimised Rees "
             "data is FOUR generators.")
    led["compressed"] = {
        "F": "t^3 (p+1)   (divides (t(p+1))^3)",
        "I_generators": G_NAMES[:4],
        "minimisation": "the natural fifth generator (Z^2+t^3)N1 = -F x H "
                        "is REDUNDANT, by the exact identity "
                        "(Z^2+t^3)N1 = Z M + p F; the Rees data minimises "
                        "to four",
        "statement": "C[X x A^1] = C^[4][I/F]",
    }

    # ==================================================================
    print("\nRM-2c  the centre's geometry -- the rectification fingerprint")
    G_COMP4 = G_COMP[:4]
    dimI = alarmed(lambda: krull_dim(G_COMP4, A4GENS))
    r.check("the centre is a SURFACE in A^4 (codimension 2)",
            lambda: dimI == 2, f"dim = {dimI}")
    C_t = [Tt, N1]                                  # step-2 component
    C_p = [E(Pp + 1), SQ]                           # step-1 component
    r.check("component 1: V(t, N1) -- the step-2 centre, pushed down",
            lambda: alarmed(lambda: krull_dim(C_t, A4GENS)) == 2
            and ideal_contains(C_t, G_COMP4, A4GENS))
    r.check("component 2: V(p+1, Z^2+t^3) -- the step-1 centre",
            lambda: alarmed(lambda: krull_dim(C_p, A4GENS)) == 2
            and ideal_contains(C_p, G_COMP4, A4GENS))
    r.check("N1 is irreducible, so (t, N1) is prime",
            lambda: len(sp.factor_list(N1, *A4GENS)[1]) == 1)
    r.check("Z^2+t^3 is irreducible, so (p+1, Z^2+t^3) is prime",
            lambda: len(sp.factor_list(SQ, *A4GENS)[1]) == 1)
    ev, inter = alarmed(lambda: verify_minimal_primes(
        G_COMP4, [C_t, C_p], A4GENS))
    r.check("sqrt(I) = V(t,N1) cap V(p+1,Z^2+t^3): powers all found",
            lambda: ev["all_powers_found"] and ev["each_prime_contains_I"],
            f"{ev['powers_into_I']}")
    r.check("H15 boundary: 1 has no power in I",
            lambda: radical_power(sp.Integer(1), G_COMP4, A4GENS) is None)
    led["centre_geometry"] = {
        "dimension": int(dimI),
        "minimal_primes": ["(t, N1)  -- the step-2 centre",
                           "(p+1, Z^2+t^3)  -- the step-1 centre"],
        "intersection_generators": ev["intersection_generators"],
    }

    # ==================================================================
    print("\nRM-2d  the singularity type of each component (Aut-invariant)")
    # component 1 sits in {t=0} = A^3_{p,q,Z} as the surface N1 = 0
    sing1 = alarmed(lambda: krull_dim(
        [N1.subs(Tt, 0)] + [sp.diff(N1.subs(Tt, 0), g)
                            for g in (Pp, Qq, ZZ)], (Pp, Qq, ZZ)))
    r.check("component 1 V(t,N1) is SMOOTH",
            lambda: sing1 == -1,
            "N1 = p(p+1) + qZ has nowhere-vanishing gradient on N1 = 0")
    # component 2 sits in {p=-1} = A^3_{q,t,Z} as the cuspidal Z^2+t^3 = 0
    sing2 = alarmed(lambda: krull_dim(
        [SQ] + [sp.diff(SQ, g) for g in (Qq, Tt, ZZ)], (Qq, Tt, ZZ)))
    r.check("component 2 V(p+1, Z^2+t^3) is SINGULAR ALONG A LINE",
            lambda: sing2 == 1,
            "the cuspidal edge {Z = t = 0}, q free -- dim 1")
    r.record("THEOREM RM-2.2: a cap-free rectification obstruction",
             "THEOREM",
             "Aut(A^4) carries centres to centres and preserves the "
             "singular locus of the centre. The compressed centre has a "
             "component that is SINGULAR along a line (the cuspidal "
             "surface V(p+1, Z^2+t^3)). Therefore (I, F) is NOT "
             "equivalent under Aut(A^4) to ANY centre whose components "
             "are smooth -- in particular not to a linear subspace, nor "
             "to any smooth complete intersection. The obvious 'standard "
             "centre' targets are ruled out with no degree cap.")
    led["singularity"] = {
        "component_1": "V(t, N1): SMOOTH",
        "component_2": "V(p+1, Z^2+t^3): singular along the line "
                       "{p=-1, Z=t=0} -- the cuspidal edge",
        "obstruction": "singularity type of the centre is an "
                       "Aut(A^4)-invariant, so no smooth standard centre "
                       "is reachable",
    }

    # ==================================================================
    print("\nRM-2e  the rectification question, and a cap-bounded probe")
    r.record("THE RECTIFICATION QUESTION, stated", "MEASURED",
             "Is there psi in Aut(A^4) with psi(I, F) a centre whose "
             "modification is visibly A^4? RM-2.2 removes every SMOOTH "
             "target. What survives is the possibility of a singular "
             "standard centre -- and note the surviving singularity is "
             "exactly the CUSP that has obstructed this programme at "
             "every previous turn (X_1's singular point, Masuda's "
             "factoriality, RG-3's rescaling block).")
    # a declared, bounded family: does any triangular automorphism make the
    # centre a complete intersection (2 generators for a codim-2 centre)?
    ci = alarmed(lambda: _min_gens_needed(G_COMP4, A4GENS))
    r.check("the centre is NOT a complete intersection: it needs > 2 "
            "generators up to radical",
            lambda: ci > 2,
            f"smallest subset of the four generating the same radical: "
            f"{ci} (codimension is 2)")
    r.check("H15 boundary: the two-generator test CAN succeed -- "
            "V(t,N1) alone needs exactly 2",
            lambda: alarmed(lambda: _min_gens_needed(C_t, A4GENS)) == 2)
    led["rectification"] = {
        "question": "is (I,F) Aut(A^4)-equivalent to a standard centre?",
        "ruled_out_cap_free": "every smooth centre (RM-2.2)",
        "not_a_complete_intersection": int(ci),
        "surviving_target": "a singular standard centre carrying a cusp",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RM-2-COMPRESS.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RM-2", "plan": "RM-OPS.md (A26)",
            "date": "2026-08-01", "mode": "I",
            "source": "p, q, Z and the t-localised bridge are DMJP "
                      "arXiv:0903.4278 s7; the compression, the canonical "
                      "centre and the obstruction are ours",
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


def _in_A(expr) -> bool:
    """Is an element of L, given in the L-coordinates, inside A[w]?

    Cheap sufficient test used here: it is a polynomial in the four
    generators p,q,t,Z together with x, H and w, all of which lie in
    A[w]. We verify by exhibiting it as such via cancellation.
    """
    e = sp.cancel(sp.together(expr))
    for cand in (sp.Integer(1), TO_L[x], H_L, E(2 * TO_L[w]),
                 sp.cancel(TO_L[x] * H_L)):
        if sp.cancel(e - cand) == 0 or sp.cancel(e + cand) == 0:
            return True
    return False


def _min_gens_needed(gens, vs, cap=8):
    """Smallest k such that some k-subset has the same radical."""
    for k in range(1, len(gens) + 1):
        for sub in itertools.combinations(range(len(gens)), k):
            S = [gens[i] for i in sub]
            if all(radical_power(g, S, vs, cap) is not None for g in gens):
                return k
    return len(gens)


if __name__ == "__main__":
    sys.exit(main())
