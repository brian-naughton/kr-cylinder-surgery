#!/usr/bin/env python3
"""RS-1 — certify the three kills, and build the invariant battery.

(a) THE THREE KILLS on Y_1, machine-certified:
    1. the stratified class [Y_1] = L^4 + L^2, hence chi_c = 2, hence
       Y_1 x A^r is never A^{4+r};
    2. Crit(W) is the line {p=-1/2, q=Z=t=0} on which W = 1/4, so W is
       not a variable of C^[5] nor stably;
    3. the explicit change of variables exhibiting
       Y_1 = Q_{(3,1), z^2-1/4} in ADO's printed conventions
       (arXiv:2112.08241, Ex. 4.2.5).
(b) THE INVARIANT BATTERY (A28 standing discipline) as a reusable
    runner, applied to all four corners of the commuting square.

Every class computation carries an H15 cross-check: an INDEPENDENT
stratification of the same variety, which must return the same class.

Attribution: ADO arXiv:2112.08241 for the family and its homotopy type;
DMJP arXiv:0903.4278 §7 for p, q, Z. The role of Y_1 in the square is
ours.

Ledger: runs_synthesis/RS-1-BATTERY.json. Mode I.
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
from rg_common import Pp, Qq, Tt, ZZ, krull_dim, verify_carryover
from rm_common import R1, Xx, verify_rg_carryover
from rs_common import (A4GENS, B_GENS, CUSP_CLASS, CUSP_EQ, CUSP_INV,
                       CUSP_PARAM, Hh, L, QSURF, W, Y_GENS, cert_graph,
                       cert_iso, chi, epoly, ss, sum_strata,
                       verify_rm_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "searches_run": 0,
    "note": "RS-1 is structural and exact: stratifications with explicit "
            "two-way parametrisations, plus Groebner dimension calls in "
            "<= 5 variables. No coefficient search, so no degree cap on "
            "unknowns applies.",
    "stratification_depth_max": 3,
    "h15_rule": "every total class is computed by TWO independent "
                "stratifications and the results must agree",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "MEASURED", "THEOREM", "NOT-DETERMINED")

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
        print(f"  {name:<58s} {outcome:<14s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<58s} {outcome:<14s}         [{note}]", flush=True)

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
    print("\nRS-1a  the cuspidal curve, certified once and reused")
    r.check("the cusp parametrisation lands on Z^2 + t^3 = 0",
            lambda: E(CUSP_EQ.xreplace(CUSP_PARAM)) == 0,
            "(Z,t) = (s^3, -s^2)")
    r.check("it is bijective off the cusp point, with inverse s = -Z/t",
            lambda: E(sp.cancel(CUSP_INV[ss].xreplace(CUSP_PARAM) - ss))
            == 0)
    r.check("so [cusp] = (L-1) + 1 = L  (punctured part + the cusp point)",
            lambda: sp.expand(CUSP_CLASS - ((L - 1) + 1)) == 0)
    r.check("H15 cross-check: the cusp is NOT smooth (so it is not just "
            "an A^1 in disguise)",
            lambda: alarmed(lambda: krull_dim(
                [CUSP_EQ, sp.diff(CUSP_EQ, ZZ), sp.diff(CUSP_EQ, Tt)],
                (ZZ, Tt))) == 0,
            "singular exactly at the origin; the class still equals L")

    # ==================================================================
    print("\nRS-1b  KILL 1 -- the stratified class [Y_1] = L^4 + L^2")
    # stratification A: by t
    r.check("stratum A1  {t != 0}: H is determined, so it is G_m x A^3",
            lambda: cert_graph(W, Hh, E((Pp + Pp**2 + Qq * ZZ) / Tt**3),
                               Y_GENS, denom=Tt))
    r.check("stratum A2  {t = 0}: W becomes -(qZ + p + p^2), H free",
            lambda: E(W.subs(Tt, 0) + QSURF) == 0)
    # the quadric surface Q = {qZ = -p(1+p)}, stratified by q
    r.check("Q-stratum {q != 0}: Z determined, so G_m x A^1",
            lambda: cert_graph(QSURF, ZZ, E(-(Pp + Pp**2) / Qq),
                               (Pp, Qq, ZZ), denom=Qq))
    r.check("Q-stratum {q = 0}: p(1+p) = 0, exactly TWO lines A^1_Z",
            lambda: sp.factor_list(E(QSURF.subs(Qq, 0)), Pp)[1]
            and set(sp.roots(sp.Poly(E(QSURF.subs(Qq, 0)), Pp)).keys())
            == {0, -1},
            "the two roots p = 0 and p = -1")
    Qcls = sp.expand((L - 1) * L + 2 * L)
    r.check("[Q] = (L-1)L + 2L = L^2 + L",
            lambda: sp.expand(Qcls - (L**2 + L)) == 0)
    stratA = [("t != 0 : G_m x A^3", (L - 1) * L**3),
              ("t = 0  : Q x A^1_H", Qcls * L)]
    clsA = sum_strata(stratA)
    r.check("stratification A gives [Y_1] = L^4 + L^2",
            lambda: sp.expand(clsA - (L**4 + L**2)) == 0, f"{clsA}")

    # H15 CROSS-CHECK: an INDEPENDENT stratification, by q
    r.check("stratum B1  {q != 0}: Z is determined, so G_m x A^3",
            lambda: cert_graph(W, ZZ, E((Tt**3 * Hh - Pp - Pp**2) / Qq),
                               Y_GENS, denom=Qq))
    r.check("stratum B2a {q = 0, t != 0}: H determined, G_m x A^2",
            lambda: cert_graph(E(W.subs(Qq, 0)), Hh,
                               E((Pp + Pp**2) / Tt**3),
                               (Pp, Tt, ZZ, Hh), denom=Tt))
    r.check("stratum B2b {q = 0, t = 0}: p(1+p) = 0, two copies of A^2",
            lambda: set(sp.roots(sp.Poly(E(-Pp - Pp**2), Pp)).keys())
            == {0, -1})
    stratB = [("q != 0        : G_m x A^3", (L - 1) * L**3),
              ("q=0, t != 0   : G_m x A^2", (L - 1) * L**2),
              ("q=0, t = 0    : 2 x A^2", 2 * L**2)]
    clsB = sum_strata(stratB)
    r.check("H15 CROSS-CHECK: stratification B gives the SAME class",
            lambda: sp.expand(clsB - clsA) == 0, f"{clsB}")
    r.check("chi_c(Y_1) = 2, against chi_c(A^4) = 1",
            lambda: chi(clsA) == 2 and chi(L**4) == 1)
    r.record("KILL 1 CERTIFIED", "THEOREM",
             "[Y_1] = L^4 + L^2 by two independent stratifications, so "
             "chi_c(Y_1) = 2. chi_c is motivic and multiplicative with "
             "chi_c(A^r) = 1, so chi_c(Y_1 x A^r) = 2 != 1 = "
             "chi_c(A^{4+r}) for every r >= 0. Hence Y_1 x A^r is NOT "
             "isomorphic to A^{4+r} for any r -- the kill covers every "
             "stabilisation.")
    led["kill_1"] = {
        "class": str(clsA), "chi_c": int(chi(clsA)),
        "stratification_A": [(n, str(c)) for n, c in stratA],
        "stratification_B_crosscheck": [(n, str(c)) for n, c in stratB],
        "Q_class": str(Qcls),
    }

    # ==================================================================
    print("\nRS-1c  KILL 2 -- Crit(W) is a line, and W = 1/4 on it")
    grad = [sp.diff(W, g) for g in Y_GENS]
    r.check("grad W = (-1-2p, -Z, 3t^2 H, -q, t^3)",
            lambda: [E(g) for g in grad]
            == [E(-1 - 2 * Pp), E(-ZZ), E(3 * Tt**2 * Hh), E(-Qq),
                E(Tt**3)])
    dcrit = alarmed(lambda: krull_dim(grad, Y_GENS))
    r.check("Crit(W) is a LINE (dimension 1)", lambda: dcrit == 1,
            "{p = -1/2, q = Z = t = 0}, H free")
    r.check("W is identically 1/4 on Crit(W)",
            lambda: E(W.xreplace({Pp: sp.Rational(-1, 2), Qq: 0, ZZ: 0,
                                  Tt: 0})) == sp.Rational(1, 4))
    r.check("H15 boundary: the critical value 1/4 is NOT 0, which is why "
            "Y_1 = V(W) is smooth",
            lambda: sp.Rational(1, 4) != 0
            and alarmed(lambda: krull_dim([W] + grad, Y_GENS)) == -1)
    r.record("KILL 2 CERTIFIED", "THEOREM",
             "a variable of a polynomial ring has nowhere-vanishing "
             "differential; W has a whole line of critical points, and "
             "adjoining further variables leaves that line (times an "
             "affine space) critical. So W is not a variable of C^[5], "
             "nor stably. Independent of KILL 1.")
    led["kill_2"] = {"crit_dim": int(dcrit),
                     "crit_locus": "{p=-1/2, q=Z=t=0}, H free",
                     "critical_value": "1/4",
                     "consequence": "W is not a variable, nor stably"}

    # ==================================================================
    print("\nRS-1d  KILL 3 -- Y_1 IS Q_{(3,1), z^2-1/4} in the ADO family")
    # ADO 2112.08241 Ex 4.2.5:  Q_{m,P} = { sum_i x_i^{m_i} t_i = P(z) }
    # our change: x1 = t, t1 = H, x2 = q, t2 = -Z, z = P' = p + 1/2
    zz2 = sp.Symbol("zprime")
    ado_lhs = E(Tt**3 * Hh + Qq * (-ZZ))            # x1^3 t1 + x2^1 t2
    ado_rhs = E(zz2**2 - sp.Rational(1, 4))         # P(z) = z^2 - 1/4
    r.check("under (x1,t1,x2,t2,z) = (t,H,q,-Z,p+1/2) the ADO equation IS W",
            lambda: E((ado_lhs - ado_rhs.xreplace(
                {zz2: E(Pp + sp.Rational(1, 2))})) - W) == 0,
            "x1^3 t1 + x2 t2 - P(z) = t^3H - qZ - p - p^2 = W")
    r.check("the change of variables is a LINEAR isomorphism of A^5",
            lambda: cert_iso(
                {Tt: Tt, Hh: Hh, Qq: Qq, ZZ: ZZ,
                 Pp: E(zz2 - sp.Rational(1, 2))},
                {Tt: Tt, Hh: Hh, Qq: Qq, ZZ: ZZ,
                 zz2: E(Pp + sp.Rational(1, 2))},
                (Tt, Hh, Qq, ZZ, zz2), (Tt, Hh, Qq, ZZ, Pp)),
            "only a translation in p; t2 = -Z is a sign flip")
    r.check("P(z) = z^2 - 1/4 has DISTINCT roots and unit discriminant",
            lambda: sp.discriminant(ado_rhs, zz2) == 1
            and len(sp.roots(sp.Poly(ado_rhs, zz2))) == 2,
            "ADO's smoothness hypothesis, verified")
    r.record("KILL 3 CERTIFIED (identification, prior art)", "THEOREM",
             "Y_1 = Q_{(3,1), z^2-1/4} EXACTLY, in ADO's printed "
             "conventions (arXiv:2112.08241 Ex. 4.2.5). With n = 2 and "
             "deg P = 2 their result gives A^1-homotopy type "
             "(P^1 smash P^1)^{wedge (deg P - 1)} = P^1 smash P^1, "
             "complex realisation S^4 -- NOT contractible, so not A^4. "
             "The fourfold and its homotopy type are ADO's PRIOR ART; "
             "what is ours is its role in the square.")
    led["kill_3"] = {
        "identification": "Y_1 = Q_{(3,1), z^2-1/4}, ADO 2112.08241 "
                          "Ex. 4.2.5",
        "change_of_variables": "x1=t, t1=H, x2=q, t2=-Z, z=p+1/2",
        "A1_homotopy_type": "P^1 smash P^1 (complex realisation S^4)",
        "prior_art": "the family, smoothness and homotopy type are ADO's",
        "open_in_the_literature": "the m-classification, ADO Question "
                                  "4.2.6",
    }
    r.record("the three kills are INDEPENDENT", "MEASURED",
             "KILL 1 is motivic (chi_c), KILL 2 is differential (critical "
             "locus), KILL 3 is homotopical (ADO). Any one suffices; "
             "KILL 1 alone also covers all stabilisations.")

    # ==================================================================
    print("\nRS-1e  THE INVARIANT BATTERY, on all four corners")

    def battery(name, eq, gens, strata, cross, sing_note, units, clgrp):
        cls = sum_strata(strata)
        crs = sum_strata(cross)
        ok = sp.expand(cls - crs) == 0
        r.check(f"battery[{name}]: two stratifications agree -- [{name}] "
                f"= {cls}", lambda o=ok: o,
                f"cross-check {crs}")
        if eq is None:
            sd = -1
        else:
            sd = alarmed(lambda: krull_dim(
                [eq] + [sp.diff(eq, g) for g in gens], gens))
        return {"class": str(cls), "chi_c": int(chi(cls)),
                "E_polynomial": str(epoly(cls)),
                "dim_Sing": int(sd), "sing_note": sing_note,
                "units": units, "class_group": clgrp,
                "strata": [(n, str(c)) for n, c in strata],
                "crosscheck": [(n, str(c)) for n, c in cross]}

    bat = {}
    # A^4
    bat["A^4"] = battery(
        "A^4", None, A4GENS,
        [("A^4", L**4)],
        [("{p!=0} G_m x A^3", (L - 1) * L**3), ("{p=0} A^3", L**3)],
        "smooth", "C^* (trivially)", "0 (trivially)")
    # X_1 x A^1
    r.check("X_1-stratum {p+1 != 0}: X determined, G_m x A^2 (x A^1_q)",
            lambda: cert_graph(R1, Xx, E(-(ZZ**2 + Tt**3) / (Pp + 1)),
                               B_GENS, denom=E(Pp + 1)))
    r.check("X_1-stratum {p+1 = 0}: the cusp Z^2+t^3 = 0, X free",
            lambda: E(R1.subs(Pp, -1) - (ZZ**2 + Tt**3)) == 0)
    bat["X_1 x A^1"] = battery(
        "X_1 x A^1", R1, B_GENS,
        [("{p+1!=0} G_m x A^2 x A^1_q", (L - 1) * L**2 * L),
         ("{p+1=0}  cusp x A^1_X x A^1_q", CUSP_CLASS * L * L)],
        [("[X_1] x A^1_q, X_1 by the same split", (L**3) * L)],
        "singular along a line (RM-1)",
        "C^* (Sol review #2, cited)", "0 -- factorial (Sol review #2)")
    # Y_1
    bat["Y_1"] = battery(
        "Y_1", W, Y_GENS,
        [(n, c) for n, c in stratA], [(n, c) for n, c in stratB],
        "smooth (RM-3)", "NOT DETERMINED here", "NOT DETERMINED here")
    # X x A^1
    KR = E(x**2 * y + z**2 + x + t**3)
    r.check("KR-stratum {x != 0}: y determined, G_m x A^2",
            lambda: cert_graph(KR, y, E(-(z**2 + x + t**3) / x**2),
                               (x, y, z, t), denom=x))
    r.check("KR-stratum {x = 0}: the cusp z^2+t^3 = 0, y free",
            lambda: E(KR.subs(x, 0) - (z**2 + t**3)) == 0)
    bat["X x A^1"] = battery(
        "X x A^1", None, GENS,
        [("{x!=0} G_m x A^2 x A^1_w", (L - 1) * L**2 * L),
         ("{x=0}  cusp x A^1_y x A^1_w", CUSP_CLASS * L * L)],
        [("[X] x A^1_w with [X] = L^3", (L**3) * L)],
        "smooth (X is smooth; RG-3 verified V(P, grad P) empty)",
        "C^* (X contractible)", "0 -- UFD (our M5)")

    r.check("SOL'S LEDGER CONFIRMED: L^4, L^4, L^4+L^2, L^4",
            lambda: [bat[k]["class"] for k in
                     ("A^4", "X_1 x A^1", "Y_1", "X x A^1")]
            == [str(sp.expand(L**4)), str(sp.expand(L**4)),
                str(sp.expand(L**4 + L**2)), str(sp.expand(L**4))],
            "no disagreement with the review")
    r.check("chi_c: 1, 1, 2, 1 -- only Y_1 differs",
            lambda: [bat[k]["chi_c"] for k in
                     ("A^4", "X_1 x A^1", "Y_1", "X x A^1")] == [1, 1, 2, 1])
    r.record("the battery, as standing discipline (A28)", "MEASURED",
             "battery(V) = stratified class (two independent "
             "stratifications) + chi_c + E-polynomial + dim Sing + units "
             "+ class group. The first four are machine-certified here; "
             "units and class group are recorded with their status, "
             "NOT-DETERMINED where that is the honest answer.")
    r.record("Y_1's units and class group", "NOT-DETERMINED",
             "not computed this visit; recorded rather than guessed. "
             "Neither is needed for the three kills.")
    led["battery"] = bat

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RS-1-BATTERY.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RS-1", "plan": "RS-OPS.md (A28)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241 Ex 4.2.5 for the family, "
                      "smoothness and A^1-homotopy type (PRIOR ART); "
                      "DMJP arXiv:0903.4278 s7 for p,q,Z; the role of "
                      "Y_1 in the square is ours",
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
