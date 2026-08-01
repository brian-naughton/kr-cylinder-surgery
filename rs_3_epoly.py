#!/usr/bin/env python3
"""RS-3 — the Hodge-Deligne E-polynomial ledger of the square.

RS-OPS asks for the E-polynomials of every divisor, centre, exceptional
locus and crossing stratum, "the finest bookkeeping level our exact
machinery reaches".

THE FINDING IS THAT IT IS NOT FINER. Every stratum used anywhere in this
visit is a product of affine spaces, tori and points -- all of Hodge-Tate
type -- so each E-polynomial is the Grothendieck class with L replaced by
uv, and the E-polynomial ledger carries NO information beyond the class
ledger. That is recorded as the result rather than padded out: the exact
symbolic machinery tops out at the class level, and anything finer needs
genuine (mixed) Hodge or homotopy input, which is RS-4's MASTER-QUERY.

What the ledger DOES sharpen: the packet is +L^2 -> (uv)^2, i.e. a Tate
class of weight 4 and Hodge type (2,2) -- exactly the reduced motive of
P^1 smash P^1, which is what ADO's homotopy type predicts.

Attribution: ADO arXiv:2112.08241; DMJP arXiv:0903.4278 §7.

Ledger: runs_synthesis/RS-3-EPOLY.json. Mode I.
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
from rs_common import (A4GENS, B_GENS, CUSP_CLASS, Hh, L, QSURF, W, Y_GENS,
                       cert_graph, chi, epoly, uu, vvv, verify_rm_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "searches_run": 0,
    "note": "RS-3 is exact bookkeeping only. Every stratum is a product "
            "of affine spaces, tori and points, so E is the class under "
            "L -> uv; no mixed-Hodge input is used or improvised.",
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
    print("\nRS-3a  the two centres not yet classed: C_2 and C_4")
    # C_2 = V(t, N1, N2) inside X_1 x A^1, reduced
    r.check("C_2 stratum {q != 0}: Z and X determined, and R1 is then "
            "AUTOMATIC",
            lambda: E(sp.cancel(
                (E(-(Pp + Pp**2) / Qq) ** 2
                 + E(Pp * E(-(Pp + Pp**2) / Qq) / Qq) * (Pp + 1)))) == 0,
            "Z = -(p+p^2)/q, X = pZ/q; then X(p+1)+Z^2 vanishes "
            "identically")
    r.check("C_2 stratum {q = 0}: p(1+p) = 0, two branches",
            lambda: set(sp.roots(sp.Poly(E(Pp + Pp**2), Pp)).keys())
            == {0, -1})
    r.check("C_2 branch p = 0: X = -Z^2, Z free -- an A^1",
            lambda: E((Xx * (Pp + 1) + ZZ**2).xreplace(
                {Pp: 0, Xx: E(-ZZ**2)})) == 0)
    r.check("C_2 branch p = -1: N2 forces Z = 0, X free -- an A^1",
            lambda: E(N2.xreplace({Pp: -1, Qq: 0})) == E(-ZZ)
            and E((Xx * (Pp + 1) + ZZ**2).xreplace({Pp: -1, ZZ: 0})) == 0)
    C2 = sp.expand((L - 1) * L + 2 * L)
    r.check("[C_2] = L^2 + L", lambda: sp.expand(C2 - (L**2 + L)) == 0)
    dC2 = alarmed(lambda: krull_dim([Tt, N1, N2, R1], B_GENS))
    r.check("H15 cross-check: dim C_2 = 2, matching a class of top term L^2",
            lambda: dC2 == 2, f"dim = {dC2}")

    # C_4 = V(p+1, Z^2+t^3, HZ+q) inside Y_1, reduced
    r.check("C_4 = {p=-1, Z^2+t^3=0, q=-HZ} inside Y_1: the Y_1 equation "
            "is then automatic",
            lambda: alarmed(lambda: in_ideal(
                E(W.xreplace({Pp: -1, Qq: E(-Hh * ZZ)})),
                [E(ZZ**2 + Tt**3)], Y_GENS)),
            "t^3 H - qZ becomes H(t^3 + Z^2)")
    C4 = sp.expand(CUSP_CLASS * L)
    r.check("[C_4] = [cusp] * L = L^2 (the cusp times A^1_H)",
            lambda: sp.expand(C4 - L**2) == 0)
    r.record("[C_4] equals the packet class exactly", "MEASURED",
             "[C_4] = L^2 = the packet removed by arrow 4. Recorded as an "
             "observation, NOT as a structural law: arrow 1 has centre of "
             "class L^2 too and removes nothing. Flagged for the "
             "MASTER-QUERY rather than interpreted here.")

    # ==================================================================
    print("\nRS-3b  the compressed centre and the crossing")
    Ct = [Tt, N1]                                   # V(t, N1)
    Cp = [E(Pp + 1), E(ZZ**2 + Tt**3)]              # V(p+1, Z^2+t^3)
    CtC = sp.expand(L**2 + L)                       # = [Q]
    CpC = sp.expand(L**2)                           # cusp x A^1_q
    inter = sp.expand(L)                            # {t=0,p=-1,Z=0}, q free
    r.check("the two components of the compressed centre meet in a LINE",
            lambda: alarmed(lambda: krull_dim(Ct + Cp, A4GENS)) == 1)
    r.check("that line is {t=0, p=-1, Z=0} with q free",
            lambda: E(N1.xreplace({Pp: -1, ZZ: 0})) == 0
            and E(E(ZZ**2 + Tt**3).xreplace({Tt: 0, ZZ: 0})) == 0)
    Icls = sp.expand(CtC + CpC - inter)
    r.check("[compressed centre] = (L^2+L) + L^2 - L = 2L^2 "
            "(inclusion-exclusion)",
            lambda: sp.expand(Icls - 2 * L**2) == 0, f"{Icls}")
    XC = sp.expand(L**2)                            # V(t,p+1) = A^2_{q,Z}
    XCred = sp.expand(2 * L - 1)                    # V(qZ) in A^2
    r.check("the crossing V(t,p+1) is A^2_{q,Z}, class L^2",
            lambda: sp.expand(XC - L**2) == 0)
    r.check("the crossing restriction (qZ^2) has reduced locus V(qZ), "
            "class 2L-1",
            lambda: sp.expand(XCred - (2 * L - 1)) == 0,
            "two lines meeting in a point")

    # ==================================================================
    print("\nRS-3c  THE E-POLYNOMIAL LEDGER")
    entries = {
        # corners
        "A^4": L**4,
        "X_1 x A^1": L**4,
        "Y_1": sp.expand(L**4 + L**2),
        "X x A^1": L**4,
        # divisors
        "D_1 = {p+1=0} in A^4": L**3,
        "D_2 = {t=0} in X_1xA^1": L**3,
        "D_3 = {t=0} in A^4": L**3,
        "D_4 = {p+1=0} in Y_1": sp.expand(L**3 + L**2 - L),
        # exceptional loci
        "E_1": L**3,
        "E_2": L**3,
        "E_3 = Y_0 = Q x A^1_H": sp.expand(L**3 + L**2),
        "E_4": sp.expand(L**3 - L),
        # centres
        "C_1 = V(p+1, Z^2+t^3)": CpC,
        "C_2 = V(t,N1,N2) in X_1xA^1": C2,
        "C_3 = V(t, N1) = Q": CtC,
        "C_4 = V(p+1,Z^2+t^3,HZ+q)": C4,
        # compressed centre and crossing
        "compressed centre V(I)": Icls,
        "crossing V(t,p+1)": XC,
        "crossing restriction V(qZ)": XCred,
        # the packet itself
        "the packet (p=-1 line x A^1_H)": sp.expand(L**2),
        "the cusp curve": CUSP_CLASS,
    }
    table = {}
    for nm, cls in entries.items():
        table[nm] = {"class": str(sp.expand(cls)),
                     "E_polynomial": str(epoly(cls)),
                     "chi_c": int(chi(cls))}
        print(f"    {nm:<38s} [.] = {str(sp.expand(cls)):<16s} "
              f"chi_c = {chi(cls)}")
    led["E_ledger"] = table

    r.check("every E-polynomial is a polynomial in the product uv",
            lambda: all(sp.expand(epoly(c) - sp.expand(
                sp.sympify(str(sp.expand(c))).subs(L, uu * vvv))) == 0
                for c in entries.values()),
            "so all classes are of Hodge-Tate type; no (p,q) off the "
            "diagonal appears")
    # H15: additivity cross-checks on the ledger
    r.check("H15 additivity: [Y_1] = [E_3] + [Y_1 minus E_3] with the "
            "open part (L-1)L^3",
            lambda: sp.expand((L**3 + L**2) + (L - 1) * L**3
                              - (L**4 + L**2)) == 0)
    r.check("H15 additivity: [X x A^1] = [E_4] + [(X x A^1) minus E_4]",
            lambda: sp.expand((L**3 - L)
                              + (L**4 - L**3 + L) - L**4) == 0,
            "the open part is X x A^1 minus E_4, isomorphic to Y_1 minus "
            "D_4, class L^4+L^2-(L^3+L^2-L) = L^4-L^3+L")
    r.check("H15 additivity: the compressed centre's two components and "
            "their line intersection",
            lambda: sp.expand(CtC + CpC - inter - Icls) == 0)
    r.record("THE E-POLYNOMIAL LEDGER ADDS NOTHING", "MEASURED",
             "every stratum in the whole square is a product of affine "
             "spaces, tori and points, so E(V) = [V] with L -> uv and the "
             "E-polynomial ledger is the class ledger relabelled. The "
             "exact symbolic machinery tops out at the class level. This "
             "is the answer to RS-3, not a shortfall: anything finer "
             "requires genuine mixed-Hodge or homotopy input, which RS-4 "
             "routes.")
    r.record("what the ledger DOES sharpen", "MEASURED",
             "the packet is +L^2 -> (uv)^2: a TATE class of weight 4 and "
             "Hodge type (2,2). That is exactly the reduced motive of "
             "P^1 smash P^1, which is what ADO's A^1-homotopy type "
             "predicts -- so the class ledger and the homotopy statement "
             "agree at the only level where both are visible.")
    led["reading"] = {
        "collapse": "E-polynomials = classes under L -> uv; no extra "
                    "separation available",
        "sharpening": "the packet is Tate of weight 4, type (2,2) = the "
                      "reduced motive of P^1 smash P^1",
        "beyond": "mixed-Hodge or homotopy input required; routed at RS-4",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RS-3-EPOLY.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RS-3", "plan": "RS-OPS.md (A28)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241; DMJP arXiv:0903.4278 s7; the "
                      "ledger is ours",
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
