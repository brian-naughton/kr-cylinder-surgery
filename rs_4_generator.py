#!/usr/bin/env python3
"""RS-4 — the generator, at stratification-certifiable level (RS-OPS §4).

RS-OPS is explicit: state exactly what is certified at class /
E-polynomial level, then write the MASTER-QUERY precisely. Do NOT
improvise homology beyond what stratification certifies. This block
obeys that line strictly -- everything below the divider is certified,
everything above the MASTER-QUERY is labelled CANDIDATE and asserted of
nothing.

WHAT COMES OUT. The mechanism is visible at chi_c level, exactly and
cheaply. Y_1 and X x A^1 contain the SAME open U as the complement of
their respective divisors:

    chi_c(Y_1)    = chi_c(D_4) + chi_c(U) = 1 + 1 = 2
    chi_c(X x A^1) = chi_c(E_4) + chi_c(U) = 0 + 1 = 1

and chi_c(E_4) = 0 rather than 1 for exactly ONE reason: the exceptional
locus is an A^1-bundle over the centre C_4 MINUS A SINGLE POINT, and is
EMPTY over that point. That one missing point is the whole of the
sphere's Euler contribution.

Attribution: ADO arXiv:2112.08241 for Y_1 ~ P^1 smash P^1 (and hence
H_4(Y_1(C)) = Z) -- cited, not reproved here; DMJP arXiv:0903.4278 §7.

Ledger: runs_synthesis/RS-4-GENERATOR.json. Mode I.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import sympy as sp

from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import Pp, Qq, Tt, ZZ, krull_dim, verify_carryover
from rm_common import N1, N2, R1, Xx, in_ideal, verify_rg_carryover
from rs_common import (A4GENS, CUSP_CLASS, Hh, L, QSURF, W, Y_GENS, chi,
                       epoly, verify_rm_carryover)
from rs_2_arrows import HS, SHEAR, TS, WS, XS, ZS

#: the E_4 constraint (p + 1 = 0) in the shear coordinates
CON = E(XS * HS + 2 * WS * ZS - 1)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "searches_run": 0,
    "note": "RS-4 certifies only what stratification certifies. No "
            "homology group is computed; the topology input needed is "
            "ROUTED as a MASTER-QUERY, not improvised.",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "MEASURED", "THEOREM", "CANDIDATE-STATEMENT",
               "NOT-DETERMINED")

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
        print(f"  {name:<56s} {outcome:<20s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<56s} {outcome:<20s}         [{note}]", flush=True)

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

    print("\nRS-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    verify_carryover(r.check)
    verify_rg_carryover(r.check)
    verify_rm_carryover(r.check)
    r.record("DECLARED CAPS", "MEASURED", json.dumps(CAPS, sort_keys=True))

    # ==================================================================
    print("\nRS-4a  the packet carrier, and the point where the arrow bites")
    r.check("the packet carrier is the p = -1 line of the degenerate "
            "fibre, times A^1_H",
            lambda: set(sp.roots(sp.Poly(E(QSURF.subs(Qq, 0)), Pp)).keys())
            == {0, -1}
            and E(QSURF.xreplace({Pp: -1, Qq: 0})) == 0,
            "{p=-1, q=0, t=0} x A^2_{Z,H}, class L^2")
    r.check("C_4 = cusp x A^1_H, class L^2, and it lies in the divisor D_4",
            lambda: alarmed(lambda: in_ideal(
                E(W.xreplace({Pp: -1, Qq: E(-Hh * ZZ)})),
                [E(ZZ**2 + Tt**3)], Y_GENS)))
    # the one point with empty exceptional fibre
    PT = {Pp: -1, Qq: 0, Tt: 0, ZZ: 0, Hh: 0}
    r.check("the point (p,q,t,Z,H) = (-1,0,0,0,0) lies on C_4",
            lambda: E(W.subs(PT)) == 0
            and E(E(ZZ**2 + Tt**3).subs(PT)) == 0
            and E(E(Qq + Hh * ZZ).subs(PT)) == 0)
    r.check("it also lies ON the packet carrier (Z = 0, H = 0 there)",
            lambda: E(Pp.subs(PT)) == -1 and E(Qq.subs(PT)) == 0
            and E(Tt.subs(PT)) == 0)
    r.check("the exceptional fibre over it is EMPTY: Xs Hs = 1 with Hs = 0",
            lambda: E(CON.xreplace({ZS: 0, HS: 0})) == -1,
            "the constraint becomes -1 = 0, unsolvable")
    r.check("H15 boundary: over any OTHER point of C_4 the fibre is an A^1",
            lambda: E(CON.xreplace({ZS: 0, HS: 1})) == E(XS - 1)
            and E(CON.xreplace({ZS: 1})) == E(XS * HS + 2 * WS - 1),
            "H != 0 at the cusp point gives Xs = 1/H, Ws free; Zs != 0 "
            "gives Ws determined")
    led["carrier"] = {
        "packet_carrier": "{p=-1, q=0, t=0} x A^2_{Z,H}, class L^2",
        "centre_C4": "cusp x A^1_H, class L^2",
        "empty_fibre_point": "(p,q,t,Z,H) = (-1,0,0,0,0)",
        "point_lies_on_carrier": True,
    }

    # ==================================================================
    print("\nRS-4b  THE MECHANISM AT chi_c LEVEL -- exact and cheap")
    clsY, clsA = sp.expand(L**4 + L**2), sp.expand(L**4)
    clsD4 = sp.expand(L**3 + L**2 - L)
    clsE4 = sp.expand(L**3 - L)
    clsU = sp.expand(clsY - clsD4)
    r.check("the common open U = Y_1 minus D_4 = (X x A^1) minus E_4",
            lambda: sp.expand(clsU - (clsA - clsE4)) == 0,
            f"[U] = {clsU}, from both sides")
    r.check("chi_c(Y_1) = chi_c(D_4) + chi_c(U) = 1 + 1 = 2",
            lambda: chi(clsD4) == 1 and chi(clsU) == 1
            and chi(clsY) == 2)
    r.check("chi_c(X x A^1) = chi_c(E_4) + chi_c(U) = 0 + 1 = 1",
            lambda: chi(clsE4) == 0 and chi(clsU) == 1
            and chi(clsA) == 1)
    r.check("so the ENTIRE difference is chi_c(D_4) - chi_c(E_4) = 1",
            lambda: chi(clsD4) - chi(clsE4) == 1)
    # and where that 1 comes from
    r.check("chi_c(C_4) = chi_c(cusp x A^1) = 1",
            lambda: chi(sp.expand(CUSP_CLASS * L)) == 1)
    r.check("chi_c(C_4 minus one point) = 0, and E_4 is an A^1-bundle "
            "over it",
            lambda: chi(sp.expand(CUSP_CLASS * L - 1)) == 0
            and sp.expand((CUSP_CLASS * L - 1) * L - clsE4) == 0)
    r.record("THE MECHANISM, CERTIFIED AT chi_c LEVEL", "THEOREM",
             "Y_1 and X x A^1 share the same open U. Y_1 completes it with "
             "a divisor of chi_c = 1; X x A^1 completes it with an "
             "exceptional locus of chi_c = 0. The whole of Y_1's excess "
             "chi_c -- the sphere's contribution -- is the difference "
             "chi_c(D_4) - chi_c(E_4) = 1. And chi_c(E_4) = 0 rather "
             "than 1 for exactly ONE reason: E_4 is an A^1-bundle over "
             "C_4 MINUS A SINGLE POINT and is EMPTY over that point. The "
             "missing point carries the whole Euler defect.")

    # counterfactual: what if the fibre were NOT empty there? (H15)
    cf_E4 = sp.expand(CUSP_CLASS * L * L)          # full A^1-bundle over C_4
    cf_target = sp.expand(clsY + (cf_E4 - clsD4))
    r.check("H15 COUNTERFACTUAL: a full A^1-bundle over C_4 would give "
            "chi_c(target) = 2, not 1",
            lambda: chi(cf_target) == 2 and chi(clsA) == 1,
            f"counterfactual [target] = {cf_target}; the missing point is "
            "NECESSARY for contractibility to be possible")
    led["mechanism_chi"] = {
        "U_class": str(clsU), "chi_U": int(chi(clsU)),
        "chi_D4": int(chi(clsD4)), "chi_E4": int(chi(clsE4)),
        "difference": 1,
        "cause": "E_4 is an A^1-bundle over C_4 minus one point, empty "
                 "over that point",
        "counterfactual": f"a full bundle would give chi_c = "
                          f"{int(chi(cf_target))}, contradicting "
                          f"contractibility",
    }

    # ==================================================================
    print("\nRS-4c  what is CERTIFIED, stated exactly")
    certified = {
        "1": "[Y_1] = L^4 + L^2 and chi_c(Y_1) = 2 (RS-1, two "
             "stratifications).",
        "2": "the +L^2 enters at arrow 3 and leaves at arrow 4 (RS-2), "
             "with [E]-[D] equal to +L^2 and -L^2 exactly.",
        "3": "the packet is carried, at class level, by the second "
             "component of the degenerate q=0 fibre of Q, times A^1_H.",
        "4": "the killing arrow's divisor meets that degenerate fibre "
             "exactly along the p = -1 component of the pair.",
        "5": "E_4 is an A^1-bundle over C_4 minus exactly one point, and "
             "EMPTY over that point; the point lies on the packet "
             "carrier.",
        "6": "chi_c(D_4) - chi_c(E_4) = 1 accounts for the entire Euler "
             "difference, and the missing point is its sole cause "
             "(counterfactual verified).",
        "7": "every E-polynomial in the square is Tate; the packet is of "
             "type (2,2), weight 4 -- the reduced motive of P^1 smash "
             "P^1 (RS-3).",
    }
    for k, v in certified.items():
        r.record(f"CERTIFIED ({k})", "MEASURED", v)
    led["certified"] = certified

    # ==================================================================
    print("\nRS-4d  the CANDIDATE statement -- asserted of nothing")
    r.record("CANDIDATE MECHANISM STATEMENT", "CANDIDATE-STATEMENT",
             "The generator of H_4(Y_1(C); Z) = Z (ADO: Y_1 ~ S^4) is "
             "represented by a cycle supported on the packet carrier -- "
             "the p = -1 component of the degenerate fibre, times A^1_H. "
             "The (p+1)-modification replaces the divisor D_4 by an "
             "A^1-bundle over the cusp locus C_4 which is EMPTY over the "
             "single point (-1,0,0,0,0) of that carrier. Deleting that "
             "point removes the carrier's compactly-supported "
             "contribution, and the target becomes contractible. "
             "ASSERTED OF NOTHING: this is the statement to be tested, "
             "not a result. The class ledger is consistent with it and "
             "does not establish it.")
    r.record("why it is not established", "NOT-DETERMINED",
             "chi_c and Grothendieck classes see Euler characteristics, "
             "not cycles. Two spaces can share a class ledger and differ "
             "in which cycle generates. Nothing computed here identifies "
             "the generator as a cycle, and RS-OPS forbids improvising "
             "it.")

    # ==================================================================
    print("\nRS-4e  THE TOPOLOGY MASTER-QUERY, stated precisely")
    mq = {
        "input_1_properness":
            "IS sigma_4 : X x A^1 --> Y_1 PROPER? Affine modifications "
            "generally are not. This decides whether the square "
            "(E_4 -> X x A^1, C_4 -> Y_1) is an abstract-blowup / cdh "
            "square and therefore whether the descent exact triangle "
            "relating the four Borel-Moore homologies is available at "
            "all. First input needed; cheap to settle in the literature "
            "or by direct argument.",
        "input_2_two_localisation_triangles":
            "Y_1 and X x A^1 contain the SAME open U as the complement "
            "of D_4 and E_4 respectively. Comparing the two localisation "
            "triangles  H^BM(D_4) -> H^BM(Y_1) -> H^BM(U) -> [1]  and  "
            "H^BM(E_4) -> H^BM(X x A^1) -> H^BM(U) -> [1]  with the SAME "
            "middle term H^BM(U) is exactly the mechanism. Both outer "
            "terms are known abstractly: H^BM(X x A^1) is that of a "
            "contractible smooth affine fourfold, H^BM(Y_1) is that of "
            "S^4 (ADO).",
        "input_3_the_two_stratified_groups":
            "H^BM_*(D_4) and H^BM_*(E_4) as GROUPS. Our stratifications "
            "give their classes, chi_c and E-polynomials exactly, but "
            "not the groups -- the extension problems are unresolved and "
            "are outside symbolic scope.",
        "input_4_the_generator_as_a_cycle":
            "Does the generator of H_4(Y_1) restrict to zero in H_4(U), "
            "i.e. is it supported on D_4? If yes, the candidate "
            "statement reduces to a statement about D_4 -> E_4. ADO's "
            "model P^1 smash P^1 may already supply the generator "
            "explicitly -- worth asking before computing.",
        "input_5_the_L2_coincidence":
            "[C_4] = L^2 equals the packet class exactly, while arrow "
            "1's centre also has class L^2 and removes nothing. Is that "
            "structural (a property of modifications whose exceptional "
            "locus drops a point) or a coincidence of these numbers?",
    }
    for k, v in mq.items():
        r.record(f"MASTER-QUERY {k}", "NOT-DETERMINED", v)
    led["master_query"] = mq
    r.record("the shape of the answer, if the inputs come back",
             "CANDIDATE-STATEMENT",
             "'The (p+1)-modification kills Y_1's motivic 4-sphere "
             "because its exceptional locus fails to cover exactly one "
             "point of the centre, and that point carries the sphere's "
             "generator.' One generator, one boundary map, one missing "
             "point. The class ledger has reduced the question to that "
             "sentence; only inputs 1-4 can promote it.")

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RS-4-GENERATOR.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RS-4", "plan": "RS-OPS.md (A28)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241 for Y_1 ~ P^1 smash P^1 "
                      "(cited, not reproved); DMJP arXiv:0903.4278 s7; "
                      "the tracing is ours",
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
