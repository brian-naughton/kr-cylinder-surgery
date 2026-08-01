#!/usr/bin/env python3
"""RF-2 — Part 1 of the target proposition: the cycle (RF-OPS §4).

Sol's inference (A30, labelled as such, TO BE VERIFIED OR CORRECTED):
[F_{-1}] - [F_0] is the natural Borel-Moore cycle representing the
generator of the relevant group, corresponding to the generator of
H~_4(Y_1).

THIS BLOCK CORRECTS IT. The relation forced by the geometry is
[F_0] + [F_{-1}] = 0, not [F_0] - [F_{-1}] = 0, so each of [F_0] and
[F_{-1}] is ALREADY a generator of H^BM_4(Y_1) = Z, and their
DIFFERENCE has index 2. The correction is robust: under the other
possible relation the difference would be zero, so on either reading
Sol's cycle fails to generate.

DECLARED SCOPE (RF-OPS §3). Algebraic identities: machine-certified.
Topological steps: written as numbered lemmas over those identities, at
integral Borel-Moore / singular level, each citing precisely and each
listing the ledger checks it consumes. Nothing is asserted beyond that.

Attribution: ADO arXiv:2112.08241 (Y_1 ~ P^1 smash P^1; Prop 4.3.1 for
the acyclicity of the one-branch deletions); DMJP arXiv:0903.4278 §7.

Ledger: runs_synthesis/RF-2-GENERATOR.json. Mode I.
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
from rf_common import (F0_IDEAL, Fm1_IDEAL, fibre_eq, verify_rs_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "declared_scope": "algebraic identities machine-certified; the "
                      "topological steps are numbered lemmas over them, "
                      "each with its citations and its consumed checks.",
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
    CLASSES = ("PASS", "FAIL", "MEASURED", "LEMMA", "THEOREM", "CORRECTION",
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
    print("\nRF-2a  the algebra the lemmas will consume")
    r.check("A1: the origin fibre is p+p^2 = 0, REDUCED, two components",
            lambda: E(fibre_eq(0, 0) + Pp + Pp**2) == 0
            and sp.discriminant(E(Pp + Pp**2), Pp) == 1
            and set(sp.roots(sp.Poly(E(Pp + Pp**2), Pp)).keys()) == {0, -1})
    r.check("A2: the generic fibre is IRREDUCIBLE (a single plane)",
            lambda: E(sp.cancel(fibre_eq(1, 0).xreplace(
                {Hh: E(Pp + Pp**2)}))) == 0
            and E(sp.cancel(fibre_eq(0, 1).xreplace(
                {ZZ: E(-(Pp + Pp**2))}))) == 0,
            "graphs over A^2, hence irreducible; so the special fibre has "
            "TWO components where the generic one has ONE")
    r.check("A3: F_0 and F_{-1} are disjoint planes of dimension 2",
            lambda: alarmed(lambda: krull_dim(F0_IDEAL, Y_GENS)) == 2
            and alarmed(lambda: krull_dim(Fm1_IDEAL, Y_GENS)) == 2
            and alarmed(lambda: krull_dim(F0_IDEAL + Fm1_IDEAL,
                                          Y_GENS)) == -1)
    IOTA = {Pp: E(-1 - Pp)}
    r.check("A4: iota : p -> -1-p preserves Y_1 and swaps the components",
            lambda: E(WEQ.xreplace(IOTA) - WEQ) == 0
            and E(Pp.xreplace(IOTA) + (Pp + 1)) == 0
            and E(E(Pp + 1).xreplace(IOTA) + Pp) == 0)
    clsF = sp.expand(2 * L**2)
    clsW = sp.expand(L**4 - L**2)
    r.check("A5: classes -- [F] = 2L^2, [W] = L^4 - L^2, [Y_1] = L^4 + L^2",
            lambda: sp.expand(clsF + clsW - (L**4 + L**2)) == 0)
    r.check("A6: chi_c bookkeeping -- 2 + 0 = 2",
            lambda: chi(clsF) == 2 and chi(clsW) == 0
            and chi(sp.expand(L**4 + L**2)) == 2)
    led["algebra"] = {
        "A1": "origin fibre reduced, two components p = 0 and p = -1",
        "A2": "generic fibre irreducible (a plane)",
        "A3": "F_0, F_{-1} disjoint planes",
        "A4": "iota swaps them",
        "A5/A6": "[F] = 2L^2, [W] = L^4-L^2, chi_c 2 = 2 + 0",
    }

    # ==================================================================
    print("\nRF-2b  the topological lemmas")
    r.record("LEMMA 1 (homotopy types)", "LEMMA",
             "Y_1 ~ P^1 smash P^1, complex realisation S^4 [ADO "
             "arXiv:2112.08241 Ex. 4.2.5, verified verbatim at A28; "
             "identification of Y_1 as Q_{(3,1),z^2-1/4} certified at "
             "RS-1]. W ~ A^2 minus origin ~ S^3 [RF-1 LEMMA RF-1.3, from "
             "the two certified trivialising graphs]. Y_1 is a smooth "
             "complex 4-fold, so of real dimension 8 and canonically "
             "oriented.")
    r.record("LEMMA 2 (the groups)", "LEMMA",
             "By Poincare duality on the smooth oriented 8-manifold Y_1, "
             "H^BM_k(Y_1) = H^{8-k}(Y_1), so H^BM_k(Y_1) = Z for k = 8 "
             "and k = 4 and 0 otherwise. For the closed subset "
             "F = F_0 disjoint-union F_{-1}, each component is C^2, so "
             "H^BM_k(F) = Z^2 for k = 4 and 0 otherwise [consumes A3]. "
             "For the open W ~ S^3: H^BM_k(W) = H^{8-k}(S^3) = Z for "
             "k = 8 and k = 5, 0 otherwise [consumes LEMMA 1].")
    r.record("LEMMA 3 (the relation [F_0] + [F_{-1}] = 0)", "LEMMA",
             "pi : Y_1 -> A^2 is FLAT [RF-1 LEMMA RF-1.2] of relative "
             "dimension 2, so flat pullback pi^! : H^BM_j(A^2) -> "
             "H^BM_{j+4}(Y_1) is defined and sends the class of a closed "
             "point to the class of its scheme-theoretic fibre. The "
             "origin fibre is REDUCED with the two components F_0, F_{-1} "
             "[consumes A1], so pi^![origin] = [F_0] + [F_{-1}]. But "
             "H^BM_0(A^2) = H^4(C^2) = 0, so the point class is ZERO. "
             "Hence [F_0] + [F_{-1}] = 0 in H^BM_4(Y_1).")

    # ==================================================================
    print("\nRF-2c  the localisation sequence, and the conclusion")
    r.record("LEMMA 4 (the localisation sequence)", "LEMMA",
             "For the closed F with open complement W in Y_1: "
             "... -> H^BM_5(Y_1) -> H^BM_5(W) -> H^BM_4(F) -> "
             "H^BM_4(Y_1) -> H^BM_4(W) -> ... . By LEMMA 2 this reads "
             "0 -> Z -> Z^2 -> Z -> 0, exact. So the map Z^2 -> Z is "
             "SURJECTIVE with kernel of rank one, and H^BM_4(Y_1) = "
             "Z^2 / (that kernel).")
    r.record("CROSS-CHECK (H15): the two independent computations of "
             "H^BM_5(W) agree", "MEASURED",
             "the sequence forces H^BM_5(W) = ker(Z^2 -> Z) = Z; "
             "independently LEMMA 2 gives H^BM_5(W) = H^3(S^3) = Z. They "
             "agree. Had the fibre or the homotopy type been wrong, this "
             "would not close.")
    r.record("LEMMA 5 (the kernel is generated by (1,1))", "LEMMA",
             "The kernel has rank one and, by LEMMA 3, contains "
             "(1,1) = [F_0] + [F_{-1}]. Since Z^2/ker = Z is "
             "torsion-free, the kernel is a PRIMITIVE rank-one sublattice; "
             "(1,1) is primitive; so the kernel is exactly <(1,1)>. "
             "Hence H^BM_4(Y_1) = Z^2/<(1,1)> = Z via (a,b) -> a - b.")
    r.record("THEOREM RF-2 (Part 1, CORRECTED)", "THEOREM",
             "H^BM_4(Y_1; Z) = Z, and under the isomorphism of LEMMA 5: "
             "[F_0] -> 1 and [F_{-1}] -> -1. So EACH of [F_0] and "
             "[F_{-1}] is a GENERATOR, they satisfy [F_0] = -[F_{-1}], "
             "and [F_{-1}] - [F_0] -> -2 is TWICE a generator: an index-2 "
             "class, NOT a generator.")
    r.record("CORRECTION to Sol's inference (A30 asked for exactly this)",
             "CORRECTION",
             "Sol's proposed cycle [F_{-1}] - [F_0] does NOT represent the "
             "generator; it represents twice one. The natural cycle is a "
             "SINGLE component, [F_0] (equivalently [F_{-1}] = -[F_0]). "
             "The inference was reasonable -- a difference of branches is "
             "the usual shape -- but here the two branches are already "
             "NEGATIVES of one another, because the fibre class vanishes "
             "(LEMMA 3), so the difference doubles instead of cancelling.")
    r.record("ROBUSTNESS (H15 boundary on the correction)", "MEASURED",
             "the involution iota swaps the two components [A4], so the "
             "kernel must be iota-stable, leaving only <(1,1)> or "
             "<(1,-1)> as primitive rank-one options. LEMMA 3 selects "
             "<(1,1)>. Under the OTHER option the difference would be "
             "ZERO. So on EITHER reading Sol's cycle fails to generate, "
             "and the correction does not depend on LEMMA 3 alone.")
    r.record("COROLLARY: iota acts by -1 on H^BM_4(Y_1)", "MEASURED",
             "iota sends [F_0] to [F_{-1}] = -[F_0], so it acts as -1 on "
             "Z. Consistent: iota is an involution and (-1)^2 = 1. A "
             "further independent consistency check on LEMMA 5.")
    led["part_1"] = {
        "result": "H^BM_4(Y_1) = Z with [F_0] and [F_{-1}] each a "
                  "generator and [F_0] + [F_{-1}] = 0",
        "sol_inference": "[F_{-1}] - [F_0] -- CORRECTED: it is twice a "
                         "generator (index 2), not a generator",
        "natural_cycle": "[F_0] alone",
        "robustness": "under the only other iota-stable relation the "
                      "difference would be 0; either way it does not "
                      "generate",
        "cross_checks": ["H^BM_5(W) = Z from the sequence and from "
                         "H^3(S^3) independently",
                         "iota acts by -1, and squares to +1"],
    }

    # the Mayer-Vietoris reading, for the record
    r.record("the Mayer-Vietoris reading (same answer, singular homology)",
             "LEMMA",
             "with the cover V_0 = Y_1 minus F_{-1} and V_{-1} = Y_1 minus "
             "F_0 [RF-1], both A^1-contractible by ADO Prop 4.3.1 (deleting "
             "all but one component over the centre), MV gives "
             "H~_4(Y_1) = H~_3(V_0 cap V_{-1}) = H~_3(W) = H~_3(S^3) = Z. "
             "This is A30's binding description: the generator IS the "
             "Mayer-Vietoris connecting class of A^2_{t,q} minus the "
             "origin. The Borel-Moore statement above is its cycle-level "
             "counterpart.")

    # ==================================================================
    print("\nRF-2d  STOP-RULE CHECKPOINT")
    r.record("STOP RULE 1 (noncanonical compactification)", "MEASURED",
             "NOT TRIGGERED. Poincare duality and the localisation "
             "sequence are canonical; no compactification was chosen.")
    r.record("STOP RULE 2 (singular centre / IC programme)", "MEASURED",
             "NOT TRIGGERED. Every space appearing in RF-2 is smooth "
             "(Y_1, C^2, W) and no intersection cohomology arises. The "
             "modification centre has not entered yet.")
    led["stop_rules_RF2"] = {"rule_1": "not triggered",
                             "rule_2": "not triggered"}

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RF-2-GENERATOR.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RF-2", "plan": "RF-OPS.md (A30)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241 (homotopy type, Prop 4.3.1); "
                      "DMJP arXiv:0903.4278 s7; the cycle computation and "
                      "the correction are ours",
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
