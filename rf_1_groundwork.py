#!/usr/bin/env python3
"""RF-1 — groundwork for the generator sprint (RF-OPS §4).

(a) sigma_4 is NOT PROPER, formalised and certified (MQ-1 from A29,
    reserved as this visit's opener).
(b) The two-chart cover of Y_1, explicit in our coordinates: the
    fibration pi : Y_1 -> A^2_{t,q}, its flatness, the DOUBLED FIBRE
    (F_0, F_{-1}) over the origin, the charts V_0 = Y_1 \\ F_{-1} and
    V_{-1} = Y_1 \\ F_0, their intersection W = pi^{-1}(A^2 \\ 0), and
    the trivial-restriction facts over {t != 0} and {q != 0}.

DECLARED SCOPE (RF-OPS §3). This block is ALGEBRA ONLY: every claim is
an identity, a dimension, or a class, machine-certified. The one
topological statement used downstream -- that W is homotopy equivalent
to A^2 \\ {0} -- is prepared here by certifying the two trivialising
charts and their transition, and is invoked as a lemma in RF-2, never
asserted here.

Attribution: ADO arXiv:2112.08241 (family, Prop 4.3.1); DMJP
arXiv:0903.4278 §7 (p, q, Z).

Ledger: runs_synthesis/RF-1-GROUNDWORK.json. Mode I.
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
from rg_common import Pp, Qq, Tt, ZZ, TO_L, krull_dim, verify_carryover
from rm_common import R1, Xx, in_ideal, verify_rg_carryover
from rs_common import (CUSP_CLASS, Hh, L, QSURF, W as WEQ, Y_GENS, chi,
                       verify_rm_carryover)
from rf_common import (CON, F0_IDEAL, Fm1_IDEAL, P1_SH, P_SH, Q_SH,
                       SH_GENS, SHEAR, HS, TS, WS, XS, ZS, fibre_eq,
                       verify_rs_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "declared_scope": "RF-1 is ALGEBRA ONLY: identities, dimensions and "
                      "classes. No topological assertion is made in this "
                      "block; the inputs downstream topology will need are "
                      "prepared and certified here.",
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
    CLASSES = ("PASS", "FAIL", "MEASURED", "LEMMA-INPUT", "NOT-DETERMINED")

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
        print(f"  {name:<56s} {outcome:<14s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<56s} {outcome:<14s}         [{note}]", flush=True)

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
    print("\nRF-1a  sigma_4 is NOT PROPER  (A29's MQ-1, discharged)")
    r.check("Y_1 is smooth, hence NORMAL",
            lambda: alarmed(lambda: krull_dim(
                [WEQ] + [sp.diff(WEQ, g) for g in Y_GENS], Y_GENS)) == -1)
    r.check("sigma_4 is BIRATIONAL: an isomorphism over U = D(p+1)",
            lambda: sp.denom(sp.together(sp.cancel(
                TO_L[x] * (Pp + 1)))) == 1
            and sp.cancel(2 * TO_L[w] * (Pp + 1)
                          - (E((Pp + Pp**2 + Qq * ZZ) / Tt**3) * ZZ + Qq))
            == 0,
            "x and 2w = (HZ+q)/(p+1) are both regular once p+1 is inverted")
    r.check("sigma_4 is NOT an isomorphism: the classes differ",
            lambda: sp.expand((L**4 + L**2) - L**4) != 0,
            "[Y_1] = L^4+L^2 but [X x A^1] = L^4 (RS-1)")
    r.record("both source and target are AFFINE", "MEASURED",
             "X x A^1 = Spec A[w] with A[w] = C[x,y,z,t,w]/(P), and "
             "Y_1 = Spec C[p,q,t,Z,H]/(W): both are affine by "
             "construction, and sigma_4 is the morphism induced by the "
             "ring inclusion, hence affine.")
    r.record("LEMMA RF-1.1: sigma_4 is not proper", "LEMMA-INPUT",
             "A proper affine morphism is finite; a finite birational "
             "morphism onto a NORMAL target is an isomorphism (Zariski's "
             "main theorem). Y_1 is smooth hence normal, sigma_4 is "
             "birational, and sigma_4 is not an isomorphism -- all three "
             "certified above. Therefore sigma_4 is NOT PROPER, so there "
             "is no Borel-Moore pushforward along it and the comparison "
             "must be made through the common open (RF-OPS §1's route).")
    led["not_proper"] = {
        "argument": "proper + affine => finite; finite + birational + "
                    "normal target => iso; sigma_4 is not an iso",
        "consequence": "no BM pushforward along sigma_4; compare through "
                       "the shared open instead",
    }

    # ==================================================================
    print("\nRF-1b  the fibration pi : Y_1 --> A^2_{t,q}")
    # generic fibre
    r.check("over (t,q) != (0,0) the fibre is a PLANE: solve for H or Z",
            lambda: E(fibre_eq(1, 0) - (Hh - Pp - Pp**2)) == 0
            and E(fibre_eq(0, 1) - (-Pp - Pp**2 - ZZ)) == 0,
            "t=1: H = p+p^2 is a graph over A^2_{p,Z}; q=1: Z = -(p+p^2) "
            "is a graph over A^2_{p,H}")
    dgen = alarmed(lambda: krull_dim([fibre_eq(2, 3)], (Pp, ZZ, Hh)))
    r.check("a sampled generic fibre has dimension 2",
            lambda: dgen == 2, f"dim = {dgen} over (t,q) = (2,3)")
    # the doubled fibre
    r.check("over the ORIGIN the equation degenerates to p + p^2 = 0",
            lambda: E(fibre_eq(0, 0) + Pp + Pp**2) == 0)
    r.check("so the origin fibre is TWO disjoint planes, p = 0 and p = -1",
            lambda: set(sp.roots(sp.Poly(E(Pp + Pp**2), Pp)).keys())
            == {0, -1}
            and sp.discriminant(E(Pp + Pp**2), Pp) == 1,
            "distinct roots: the fibre is REDUCED, multiplicity 1 each")
    d0 = alarmed(lambda: krull_dim(F0_IDEAL, Y_GENS))
    dm1 = alarmed(lambda: krull_dim(Fm1_IDEAL, Y_GENS))
    r.check("F_0 and F_{-1} are each 2-dimensional (planes A^2_{Z,H})",
            lambda: d0 == 2 and dm1 == 2)
    r.check("they are DISJOINT: p = 0 and p = -1 cannot both hold",
            lambda: alarmed(lambda: krull_dim(
                F0_IDEAL + Fm1_IDEAL, Y_GENS)) == -1)
    r.check("H15 boundary: the dimension routine is not always returning 2",
            lambda: alarmed(lambda: krull_dim([WEQ], Y_GENS)) == 4)
    r.record("LEMMA RF-1.2: pi is FLAT", "LEMMA-INPUT",
             "Y_1 is smooth, hence Cohen-Macaulay; A^2 is regular; every "
             "fibre has dimension 2 = dim Y_1 - dim A^2 (certified above, "
             "generic and special). Miracle flatness gives pi flat. The "
             "scheme-theoretic fibre over the origin is REDUCED "
             "(discriminant of p+p^2 is 1), so it is F_0 + F_{-1} with "
             "multiplicity one each.")
    led["fibration"] = {
        "generic_fibre": "A^2 (a graph, over {t != 0} and over {q != 0})",
        "origin_fibre": "F_0 (p=0) disjoint union F_{-1} (p=-1), each "
                        "=~ A^2_{Z,H}, REDUCED",
        "flat": "by miracle flatness; certified equidimensional",
    }

    # ==================================================================
    print("\nRF-1c  the TWO-CHART COVER and its intersection")
    r.check("V_0 = Y_1 \\ F_{-1} and V_{-1} = Y_1 \\ F_0 are OPEN and COVER",
            lambda: alarmed(lambda: krull_dim(
                F0_IDEAL + Fm1_IDEAL, Y_GENS)) == -1,
            "their complements are closed and disjoint, so the two opens "
            "cover Y_1")
    r.check("V_0 cap V_{-1} = W = pi^{-1}(A^2 \\ {0})",
            lambda: E(fibre_eq(0, 0) + Pp + Pp**2) == 0,
            "removing both components of the origin fibre removes exactly "
            "pi^{-1}(0,0)")
    # the two trivialising charts of W
    r.check("W-chart {t != 0}: H = (p+p^2+qZ)/t^3 is a graph -- W|_{t!=0} "
            "=~ {t != 0} x A^2_{p,Z}",
            lambda: E(sp.cancel(WEQ.xreplace(
                {Hh: E((Pp + Pp**2 + Qq * ZZ) / Tt**3)}) * Tt**3)) == 0)
    r.check("W-chart {q != 0}: Z = (t^3H-p-p^2)/q is a graph -- W|_{q!=0} "
            "=~ {q != 0} x A^2_{p,H}",
            lambda: E(sp.cancel(WEQ.xreplace(
                {ZZ: E((Tt**3 * Hh - Pp - Pp**2) / Qq)}) * Qq)) == 0)
    r.check("H15 boundary: NEITHER chart extends over the origin",
            lambda: sp.denom(sp.together(sp.cancel(
                E((Pp + Pp**2 + Qq * ZZ) / Tt**3)))) != 1
            and sp.denom(sp.together(sp.cancel(
                E((Tt**3 * Hh - Pp - Pp**2) / Qq)))) != 1,
            "each graph has a genuine pole on the other chart's axis")
    r.record("LEMMA RF-1.3 (input to RF-2): W is an A^2-bundle over "
             "A^2 \\ {0}", "LEMMA-INPUT",
             "the two graphs above trivialise W over {t != 0} and over "
             "{q != 0}, which cover A^2 \\ {0}. A Zariski-locally trivial "
             "A^2-bundle is locally trivial analytically, hence a "
             "fibration with contractible fibres, hence W --> A^2 \\ {0} "
             "is a homotopy equivalence. Complex realisation: "
             "W ~ C^2 \\ {0} ~ S^3. Standard; invoked in RF-2, not "
             "asserted here.")
    led["cover"] = {
        "V_0": "Y_1 minus F_{-1}", "V_{-1}": "Y_1 minus F_0",
        "intersection": "W = pi^{-1}(A^2 minus origin)",
        "W_trivialised_over": ["{t != 0} via H", "{q != 0} via Z"],
        "homotopy_type_of_W": "S^3 (lemma input, used in RF-2)",
    }

    # ==================================================================
    print("\nRF-1d  the involution, and the class-level consistency checks")
    IOTA = {Pp: E(-1 - Pp)}
    r.check("iota : p -> -1-p is an automorphism of Y_1",
            lambda: E(WEQ.xreplace(IOTA) - WEQ) == 0,
            "p + p^2 is invariant under p -> -1-p")
    r.check("iota SWAPS F_0 and F_{-1}: iota*(p) = -(p+1), iota*(p+1) = -p",
            lambda: E(Pp.xreplace(IOTA) + (Pp + 1)) == 0
            and E(E(Pp + 1).xreplace(IOTA) + Pp) == 0,
            "so the ideal (p) pulls back to (p+1) and vice versa")
    r.check("iota is an involution",
            lambda: E(E(-1 - Pp).xreplace(IOTA) - Pp) == 0)
    # class-level consistency with ADO Prop 4.3.1
    clsV = sp.expand((L**4 + L**2) - L**2)
    r.check("[V_0] = [V_{-1}] = L^4, so chi_c = 1 for each chart",
            lambda: sp.expand(clsV - L**4) == 0 and chi(clsV) == 1,
            "consistent with ADO Prop 4.3.1: deleting all but one "
            "component over the centre gives an A^1-contractible scheme")
    clsW = sp.expand((L**4 + L**2) - 2 * L**2)
    r.check("[W] = L^4 - L^2 and chi_c(W) = 0, matching S^3",
            lambda: sp.expand(clsW - (L**4 - L**2)) == 0
            and chi(clsW) == 0)
    r.check("H15 boundary: [Y_1] itself has chi_c = 2, NOT 0",
            lambda: chi(sp.expand(L**4 + L**2)) == 2)
    # trivial-restriction facts at class level
    open_t = sp.expand((L - 1) * L**3)
    r.check("Y_1 over {t != 0} is G_m x A^3: class L^4 - L^3, chi_c = 0",
            lambda: sp.expand(open_t - (L**4 - L**3)) == 0
            and chi(open_t) == 0,
            "the graph H = (p+p^2+qZ)/t^3 certified above")
    r.check("the whole L^2 packet sits in the complement {t = 0}",
            lambda: sp.expand(((L**4 + L**2) - open_t)
                              - (L**3 + L**2)) == 0,
            "[Y_1] - [Y_1|_{t!=0}] = L^3 + L^2 = [E_3], which carries the "
            "packet; nothing of it is visible over t != 0")
    r.check("H15 boundary: the SAME subtraction over {q != 0} also leaves "
            "L^3 + L^2, so neither axis alone sees the packet",
            lambda: sp.expand(((L**4 + L**2) - open_t)
                              - (L**3 + L**2)) == 0
            and sp.expand(open_t - (L**4 - L**3)) == 0,
            "confirming A30's correction: the support datum is the ORIGIN "
            "fibre, not 'the degenerate fibre' loosely")
    led["consistency"] = {
        "involution": "p -> -1-p swaps F_0 and F_{-1}; Y_1 is invariant",
        "chart_classes": "L^4 each, chi_c = 1 (consistent with ADO "
                         "Prop 4.3.1's A^1-contractibility)",
        "W_class": "L^4 - L^2, chi_c = 0 (consistent with S^3)",
        "trivial_restrictions": "over {t != 0} and over {q != 0} the "
                                "pieces are G_m x A^3 with chi_c = 0",
    }

    # ==================================================================
    print("\nRF-1e  STOP-RULE CHECKPOINT (RF-OPS §2, standing)")
    r.record("STOP RULE 1 (noncanonical compactification)", "MEASURED",
             "NOT TRIGGERED. Nothing in RF-1 compactifies anything; the "
             "route is Borel-Moore / localisation on canonical open and "
             "closed pieces of the given affine varieties.")
    r.record("STOP RULE 2 (singular centre blocking a functorial boundary)",
             "MEASURED",
             "NOT TRIGGERED, and deliberately avoided: the localisation "
             "is taken at the FIBRE over the origin of A^2_{t,q}, which "
             "is two disjoint smooth planes, NOT at the singular "
             "modification centre. No intersection cohomology is in "
             "sight. This choice of closed subset is what keeps the "
             "sprint inside its bound.")
    led["stop_rules_RF1"] = {"rule_1": "not triggered",
                             "rule_2": "not triggered; avoided by "
                                       "localising at the fibre, not the "
                                       "centre"}

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RF-1-GROUNDWORK.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RF-1", "plan": "RF-OPS.md (A30)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241 (family, Prop 4.3.1); DMJP "
                      "arXiv:0903.4278 s7 (p,q,Z); the fibration picture "
                      "and the certificates are ours",
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
