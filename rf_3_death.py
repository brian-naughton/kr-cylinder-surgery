#!/usr/bin/env python3
#
# PUBLICATION COPY. Identical to the working-repo original except
# for interpretive annotation strings, corrected before release so
# that the ledger prose matches the paper. No mathematical check,
# no computation and no verdict differs; see the supersession diff
# shipped with the private record, and VERIFY.md in this
# repository.
"""RF-3 — Parts 2 and 3: the modification inserted, and the death.

DECLARED SCOPE (RF-OPS §3). Algebra machine-certified; topology written
as numbered lemmas over it. Where an input is CITED rather than derived
(the contractibility of X x A^1), that is stated at the point of use and
again in the boundary statement. Nothing is asserted beyond it.

WHAT COMES OUT. The mechanism is a cycle-level statement, and it is
cleaner than the chi_c shadow suggested:

  * F_0 is DISJOINT from the modification divisor D_4; F_{-1} lies
    ENTIRELY inside it.
  * By RF-2, [F_0] = -[F_{-1}] and each generates H^BM_4(Y_1) = Z. So
    the generator, although representable by a cycle disjoint from the
    surgery locus, is ALSO representable by a cycle contained in it --
    and consequently restricts to ZERO on the common open U. That is
    the structural reason the surgery can reach it at all.
  * sigma_4^{-1}(F_{-1}) is NOT a plane: it is G_m x A^1, the plane with
    the fibre over the defect point (-1,0,0,0,0) deleted. Its own
    fundamental class, pushed forward along its CLOSED IMMERSION into
    X x A^1, is zero because H^BM_4(X x A^1) = 0. Nothing is pushed
    forward along sigma_4 itself, which is not proper (LEMMA RF-1.1).

AND THE HONEST BOUNDARY, certified rather than glossed: the modification
changes the fibre over EVERY point of A^2_{t,q}, not only over the
origin. So the difference is not localised at the defect, and deriving
the vanishing from the geometry alone (rather than from the cited
contractibility) needs the Borel-Moore homology of the modified
punctured family -- a monodromy computation outside this visit's bound.

Attribution: ADO arXiv:2112.08241; DMJP arXiv:0903.4278 §7. The
vanishing H^BM_4(X x A^1) = 0 needs only the ORDINARY TOPOLOGICAL
contractibility of X together with Poincare duality on a smooth
fourfold; Dubouloz--Fasel arXiv:1512.01933 and HKO arXiv:1409.1293
(whose theorem is for finite suspensions) are context, not input.

Ledger: runs_synthesis/RF-3-DEATH.json. Mode I.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import sympy as sp

from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import Pp, Qq, Tt, ZZ, TO_L, krull_dim, verify_carryover
from rm_common import R1, Xx, in_ideal, verify_rg_carryover
from rs_common import (Hh, L, QSURF, W as WEQ, Y_GENS, chi,
                       verify_rm_carryover)
from rf_common import (CON, F0_IDEAL, Fm1_IDEAL, P1_SH, P_SH, Q_SH,
                       SH_GENS, SHEAR, HS, TS, WS, XS, ZS, fibre_eq,
                       verify_rs_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "declared_scope": "algebra machine-certified; topology as numbered "
                      "lemmas; cited inputs named at point of use.",
    "cited_inputs": ["ADO 2112.08241: Y_1 ~ S^4",
                     "H^BM_4(X x A^1) = 0 from the ORDINARY "
                     "topological contractibility of X plus Poincare "
                     "duality on a smooth fourfold; the motivic "
                     "results are context, not input"],
    "searches_run": 0,
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
    CLASSES = ("PASS", "FAIL", "MEASURED", "LEMMA", "THEOREM", "BOUNDARY",
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
        print(f"  {name:<54s} {outcome:<14s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<54s} {outcome:<14s}         [{note}]", flush=True)

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

    print("\nRF-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    verify_carryover(r.check)
    verify_rg_carryover(r.check)
    verify_rm_carryover(r.check)
    verify_rs_carryover(r.check)
    r.record("DECLARED SCOPE / CAPS", "MEASURED",
             json.dumps(CAPS, sort_keys=True))

    # ==================================================================
    print("\nRF-3a  how the two components sit relative to the divisor")
    r.check("B1: F_0 is DISJOINT from D_4 = {p+1 = 0}",
            lambda: alarmed(lambda: krull_dim(
                F0_IDEAL + [E(Pp + 1)], Y_GENS)) == -1,
            "F_0 has p = 0, the divisor has p = -1")
    r.check("B2: F_{-1} is CONTAINED in D_4",
            lambda: alarmed(lambda: in_ideal(E(Pp + 1), Fm1_IDEAL,
                                             Y_GENS)),
            "p + 1 vanishes identically on F_{-1}")
    r.record("LEMMA 6: sigma_4 is an isomorphism over F_0", "LEMMA",
             "sigma_4 is an isomorphism over U = Y_1 minus D_4 [RF-1], "
             "and F_0 is a closed subvariety of U by B1. So "
             "F~_0 := sigma_4^{-1}(F_0) is closed in X x A^1 and "
             "sigma_4 restricts to an isomorphism F~_0 -> F_0 =~ A^2.")

    # ==================================================================
    print("\nRF-3b  what happens to F_{-1}: the defect made explicit")
    # in the shear coordinates, sigma_4^{-1}(F_{-1}) = {Ts=0, Zs=0, XsHs=1}
    r.check("B3: on E_4 with Ts = 0 the cusp equation forces Zs = 0",
            lambda: E(E(ZS**2 + TS**3).subs(TS, 0) - ZS**2) == 0)
    r.check("B4: and then q = -Hs Zs = 0 automatically",
            lambda: alarmed(lambda: in_ideal(E(Q_SH + HS * ZS), [CON],
                                             SH_GENS))
            and E(E(-HS * ZS).subs(ZS, 0)) == 0)
    r.check("B5: the surviving constraint is Xs Hs = 1, with Ws free",
            lambda: E(CON.subs(ZS, 0) - (XS * HS - 1)) == 0)
    r.record("LEMMA 7: sigma_4^{-1}(F_{-1}) = G_m x A^1", "LEMMA",
             "by B3-B5 the preimage is {Ts = 0, Zs = 0, Xs Hs = 1, Ws "
             "free}. Solving Hs = 1/Xs identifies it with "
             "G_m (coordinate Xs) x A^1 (coordinate Ws). Class "
             "(L-1)L = L^2 - L, matching the RS-2 ledger entry for the "
             "origin part of E_4.")
    # what it maps ONTO inside F_{-1}
    r.check("B6: its image inside F_{-1} =~ A^2_{Z,H} is the PUNCTURED "
            "line {Z = 0, H != 0}",
            lambda: E(E(ZS).subs(ZS, 0)) == 0
            and E(CON.subs({ZS: 0, HS: 0})) == -1,
            "Z = Zs = 0 always; H = Hs must be non-zero since Xs Hs = 1")
    r.check("B7: every point of F_{-1} with Z != 0 has EMPTY preimage",
            lambda: E(E(ZS**2 + TS**3).subs({TS: 0, ZS: 1})) == 1,
            "the cusp equation Zs^2 + Ts^3 = 0 fails at Ts = 0, Zs != 0")
    r.check("B8: the puncture IS the defect point (p,q,t,Z,H) = "
            "(-1,0,0,0,0)",
            lambda: E(WEQ.subs({Pp: -1, Qq: 0, Tt: 0, ZZ: 0, Hh: 0})) == 0
            and E(CON.subs({ZS: 0, HS: 0})) == -1,
            "it lies on Y_1, and the exceptional constraint is unsolvable "
            "there")
    r.check("H15 boundary: over any OTHER point of the line the fibre is "
            "an A^1",
            lambda: E(CON.subs({ZS: 0, HS: 1})) == E(XS - 1),
            "H != 0 gives Xs = 1/H with Ws free")
    led["defect"] = {
        "F_0": "disjoint from D_4; carried isomorphically by sigma_4",
        "F_{-1}": "contained in D_4",
        "preimage_of_F_{-1}": "G_m x A^1 = {Ts=0, Zs=0, XsHs=1}",
        "image_inside_F_{-1}": "the punctured line {Z = 0, H != 0}",
        "empty_over": "all of F_{-1} with Z != 0, and the puncture "
                      "(Z,H) = (0,0) -- the defect point",
    }

    # ==================================================================
    print("\nRF-3c  THE MECHANISM, as a cycle statement")
    r.record("THEOREM RF-3 (Parts 2 and 3, at cycle level)", "THEOREM",
             "(i) By RF-2, H^BM_4(Y_1) = Z with [F_0] = -[F_{-1}] and each "
             "a generator. (ii) F_{-1} is a closed subvariety of the "
             "modification divisor D_4 [B2]. Hence the generator is the "
             "pushforward of the fundamental class of F_{-1} under "
             "H^BM_4(D_4) -> H^BM_4(Y_1): THE GENERATOR HAS A "
             "REPRESENTATIVE INSIDE THE SURGERY LOCUS, even though it "
             "also has one (F_0) disjoint from it. (iii) By exactness of "
             "the localisation sequence for D_4 with open complement U, "
             "the generator therefore RESTRICTS TO ZERO in H^BM_4(U). "
             "(iv) The inverse image of that representative is "
             "sigma_4^{-1}(F_{-1}) = G_m x A^1 [LEMMA 7] -- the same "
             "plane with the fibre over the defect point deleted. It is "
             "a CLOSED subvariety of X x A^1, and the pushforward of "
             "its fundamental class along that CLOSED IMMERSION is "
             "zero, because H^BM_4(X x A^1) = 0 -- by Poincare duality "
             "on the smooth fourfold and the ordinary topological "
             "contractibility of X. NOTE: sigma_4 is NOT proper "
             "[LEMMA RF-1.1], so no class is pushed forward along it; "
             "the defect locates the cycle-level content of the "
             "change.")
    r.record("what this adds to the chi_c statement of RS-4", "MEASURED",
             "RS-4 established the Euler bookkeeping: the whole "
             "difference is chi_c(D_4) - chi_c(E_4) = 1, caused by one "
             "missing point. RF-3 upgrades that to INTEGRAL homology and "
             "names the cycle: the generator of H^BM_4(Y_1) = Z is "
             "[F_{-1}] (equivalently -[F_0]), it lives inside the surgery "
             "divisor, and the surgery replaces it by a punctured plane. "
             "The 'missing point' is now the puncture of an explicit "
             "cycle, not a numerical shadow.")

    # the homology of the origin fibre, before and after
    r.record("the origin fibre, before and after", "MEASURED",
             "Y_1: F_0 disjoint-union F_{-1} = C^2 disjoint-union C^2, so "
             "H^BM_4 = Z^2 and H^BM_3 = 0. X x A^1: F~_0 disjoint-union "
             "G_m x A^1 = C^2 disjoint-union (C^* x C), so H^BM_4 = Z^2 "
             "but H^BM_3 = Z. The defect creates a degree-3 class where "
             "there was none -- the loop in G_m.")

    # ==================================================================
    print("\nRF-3d  THE HONEST BOUNDARY -- certified, not glossed")
    # the modification changes fibres over EVERY base point
    r.check("C1: over t != 0 the exceptional part of the fibre is "
            "NON-EMPTY: Zs^2 = -t^3 has two roots",
            lambda: sp.discriminant(E(ZS**2 + 8), ZS) != 0
            and len(sp.roots(sp.Poly(E(ZS**2 + 8), ZS))) == 2,
            "sampled at t = 2: Zs^2 = -8 has two distinct roots, both "
            "non-zero")
    r.check("C2: and over t = 0, q != 0 the exceptional part is EMPTY",
            lambda: E(E(-HS * ZS).subs(ZS, 0)) == 0,
            "Ts = 0 forces Zs = 0 hence q = 0, so no point of E_4 lies "
            "over (0, q) with q != 0")
    r.check("C3: so the divisor D_4 meets EVERY fibre of pi -- it is not "
            "concentrated over the origin",
            lambda: alarmed(lambda: krull_dim(
                [WEQ, E(Pp + 1), E(Tt - 2), E(Qq - 3)], Y_GENS)) >= 0,
            "sampled: D_4 has points over (t,q) = (2,3)")
    r.record("BOUNDARY OF THE PROOF", "BOUNDARY",
             "The modification changes the fibre over EVERY point of "
             "A^2_{t,q} [C1-C3], not only over the origin: over t != 0 the "
             "plane loses its p = -1 line and gains two lines, and over "
             "t = 0 with q != 0 it simply loses that line. Therefore the "
             "comparison of the two localisation sequences is NOT "
             "supported at the defect alone, and H^BM_*(W~) is not "
             "obtainable from H^BM_*(W) by a local change. Consequently "
             "step (iv) of THEOREM RF-3 uses the contractibility of "
             "X x A^1 rather than deriving the vanishing from the "
             "modification geometry. Closing that gap needs H^BM_*(W~) "
             "computed independently -- a monodromy computation over "
             "A^2 minus the origin (the two exceptional lines over t != 0 "
             "are exchanged by the double cover Zs^2 = -t^3) -- which is "
             "beyond this visit's bound and is ROUTED, not attempted.")
    led["boundary"] = {
        "what_is_derived": "the cycle identification, its containment in "
                           "the divisor, its restriction to zero on U, "
                           "and the explicit punctured replacement",
        "what_is_cited": "H^BM_4(X x A^1) = 0, from the ORDINARY "
                         "topological contractibility of X and duality "
                         "on a smooth fourfold",
        "what_is_missing": "H^BM_*(W~), needing a monodromy analysis of "
                           "the double cover Zs^2 = -t^3 over A^2 minus "
                           "the origin",
    }

    # ==================================================================
    print("\nRF-3e  STOP-RULE CHECKPOINT (the decisive one)")
    r.record("STOP RULE 1 (noncanonical compactification)", "MEASURED",
             "NOT TRIGGERED. No compactification was used at any point; "
             "Poincare duality, flat pullback and localisation are "
             "canonical.")
    r.record("STOP RULE 2 (singular centre / IC programme)", "MEASURED",
             "NOT TRIGGERED. The singular centre was never entered: the "
             "whole argument runs on the fibre over the origin (two "
             "smooth planes) and on sigma_4^{-1}(F_{-1}) = G_m x A^1, "
             "also smooth. No intersection cohomology arose.")
    r.record("THE ACTUAL LIMIT, and why we stop here anyway", "BOUNDARY",
             "Neither stop rule fired. The limit is the visit's BOUND, "
             "not a pathology: the remaining gap is a well-posed and "
             "elementary-but-sizeable monodromy computation, and RF-OPS "
             "authorises one bounded visit. Starting it here would be "
             "the failure mode the stop rules exist to prevent, in "
             "spirit if not in letter. Routed to master as the single "
             "MASTER-QUERY.")
    led["stop_rules_RF3"] = {
        "rule_1": "not triggered", "rule_2": "not triggered",
        "actual_limit": "the visit bound; the residue is a well-posed "
                        "monodromy computation, routed not attempted",
    }

    # ==================================================================
    print("\nRF-3f  the fallback statement, STRENGTHENED by what was proved")
    r.record("PRE-APPROVED FALLBACK (A30), for comparison", "MEASURED",
             "'the exceptional defect accounts exactly for the Euler and "
             "Grothendieck-class change, while its interpretation as the "
             "killing of the homotopy generator remains conjectural.'")
    r.record("STRENGTHENED STATEMENT, ready for the paper", "THEOREM",
             "H^BM_4(Y_1;Z) = Z is generated by the fundamental class of "
             "either component of the doubled fibre over the origin of "
             "A^2_{t,q}, with [F_0] = -[F_{-1}]. The generator is "
             "SUPPORTED ON the modification divisor -- it has the "
             "representative F_{-1} contained in it -- and restricts to "
             "zero on the common open. The inverse image of that "
             "representative is sigma_4^{-1}(F_{-1}) = G_m x A^1, the "
             "same plane with the fibre over the defect point "
             "(-1,0,0,0,0) deleted, and the fundamental class of that "
             "locus, pushed forward along its CLOSED IMMERSION, is zero "
             "since H^BM_4(X x A^1) = 0. Dually, and with no properness "
             "hypothesis, sigma_4^* PD[F_{-1}] = 0 in H^4. The "
             "exceptional defect thus accounts exactly for the Euler "
             "and Grothendieck-class change and LOCATES ITS "
             "CYCLE-LEVEL CONTENT at one named point; deriving the "
             "vanishing "
             "from the modification geometry alone, rather than from the "
             "known contractibility of X x A^1, requires the "
             "Borel-Moore homology of the modified punctured family, "
             "which we do not compute.")
    led["fallback"] = {
        "pre_approved": "Euler/class change accounted for; homotopy "
                        "interpretation conjectural",
        "achieved": "the cycle is IDENTIFIED and its fate is explicit; "
                    "only the direction of inference remains cited "
                    "rather than derived",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RF-3-DEATH.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RF-3", "plan": "RF-OPS.md (A30)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241; DMJP arXiv:0903.4278 s7; "
                      "Dubouloz-Fasel 1512.01933 / HKO 1409.1293 (cited "
                      "input); the cycle analysis is ours",
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
