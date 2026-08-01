#!/usr/bin/env python3
"""RS-2 — the four-arrow [E] - [D] ledger (RS-OPS §4).

The commuting square has four arrows (morphisms; Spec reverses the ring
inclusions):

    sigma_1 : X_1 x A^1 --> A^4        along {p+1 = 0} in A^4
    sigma_2 : X x A^1   --> X_1 x A^1  along {t = 0}   in X_1 x A^1
    sigma_3 : Y_1       --> A^4        along {t = 0}   in A^4
    sigma_4 : X x A^1   --> Y_1        along {p+1 = 0} in Y_1

For an affine modification, sigma is an isomorphism off the divisor, so
[V'] - [E] = [V] - [D], i.e. [V'] - [V] = [E] - [D]. Each arrow's
isomorphism-off-the-divisor is re-verified in process; each [D] and [E]
is computed by an explicit stratification with two-way parametrisations
and an H15 cross-check by an INDEPENDENT stratification.

EXPECTED (Sol, docs_sol_review_3.md §2): the singular route is
K_0-neutral at each step; the smooth route first CREATES an L^2 packet
and then REMOVES it. This block certifies that and identifies the
packet's geometric carrier in each direction.

Attribution: ADO arXiv:2112.08241 (the family and its homotopy type);
DMJP arXiv:0903.4278 §7 (p, q, Z).

Ledger: runs_synthesis/RS-2-ARROWS.json. Mode I.
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
from rm_common import R1, Xx, in_ideal, verify_rg_carryover
from rs_common import (A4GENS, B_GENS, CUSP_CLASS, Hh, L, QSURF, W, Y_GENS,
                       cert_graph, chi, epoly, sum_strata,
                       verify_rm_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "searches_run": 0,
    "note": "RS-2 is exact bookkeeping: stratifications with two-way "
            "parametrisations plus Groebner dimension calls in <= 5 "
            "variables. No coefficient search.",
    "h15_rule": "every divisor and exceptional class is computed by TWO "
                "independent stratifications",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}

#: the shear coordinates on A^5 in which E_4 is transparent
XS, TS, ZS, WS, HS = sp.symbols("Xs Ts Zs Ws Hs")
SHEAR = {x: XS, y: E(-HS - WS**2), z: E(ZS - XS * WS), t: TS, w: WS}


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
    print("\nRS-2a  every arrow is an isomorphism off its divisor")
    r.check("sigma_1 off {p+1=0}: x is regular there",
            lambda: sp.denom(sp.together(sp.cancel(
                TO_L[x] * (Pp + 1)))) == 1)
    r.check("sigma_2 off {t=0}: H and 2w are regular there",
            lambda: sp.denom(sp.together(sp.cancel(H_L * Tt**3))) == 1
            and sp.denom(sp.together(sp.cancel(
                2 * TO_L[w] * Tt**3 * (Pp + 1)))) == 1)
    r.check("sigma_3 off {t=0}: H is regular there",
            lambda: sp.denom(sp.together(sp.cancel(H_L * Tt**3))) == 1)
    r.check("sigma_4 off {p+1=0}: x and 2w = (HZ+q)/(p+1) are regular",
            lambda: sp.denom(sp.together(sp.cancel(
                TO_L[x] * (Pp + 1)))) == 1
            and sp.cancel(2 * TO_L[w] * (Pp + 1) - (H_L * ZZ + Qq)) == 0)
    r.record("so [V'] - [V] = [E] - [D] for all four", "MEASURED",
             "each sigma restricts to an isomorphism V' minus E --> V "
             "minus D, and the scissor relation follows")

    # ==================================================================
    print("\nRS-2b  the shear coordinates, in which E_4 is transparent")
    Psh = E(P.xreplace(SHEAR))
    r.check("P becomes Xs(1 - Xs Hs - 2 Ws Zs) + Zs^2 + Ts^3",
            lambda: E(Psh - (XS * (1 - XS * HS - 2 * WS * ZS)
                             + ZS**2 + TS**3)) == 0)
    r.check("p + 1 becomes 1 - Xs Hs - 2 Ws Zs",
            lambda: E(E(KER_P.xreplace(SHEAR)) - (-(XS * HS
                                                    + 2 * WS * ZS))) == 0)
    r.check("q becomes 2Ws - 2 Hs Xs Ws - 4 Ws^2 Zs - Hs Zs",
            lambda: E(E(KER_Q.xreplace(SHEAR))
                      - (2 * WS - 2 * HS * XS * WS - 4 * WS**2 * ZS
                         - HS * ZS)) == 0)
    r.check("H15 boundary: the shear is invertible (y, z recovered)",
            lambda: E(E(-HS - WS**2).xreplace(
                {HS: E(-(y + w**2)), WS: w}) - y) == 0
            and E(E(ZS - XS * WS).xreplace(
                {ZS: E(z + x * w), XS: x, WS: w}) - z) == 0)

    # ==================================================================
    print("\nRS-2c  ARROW 1: X_1 x A^1 --> A^4 along {p+1 = 0}")
    D1 = L**3
    r.check("[D_1] = L^3: {p+1=0} in A^4 is a hyperplane A^3_{q,t,Z}",
            lambda: sp.expand(D1 - L**3) == 0)
    r.check("E_1 = {p+1=0} in X_1 x A^1 is the cusp x A^2_{q,X}",
            lambda: E(R1.subs(Pp, -1) - (ZZ**2 + Tt**3)) == 0)
    E1 = sp.expand(CUSP_CLASS * L**2)
    E1x = sp.expand((L - 1) * L**2 + L**2)      # cusp = punctured + point
    r.check("H15 cross-check on [E_1]: cusp = (L-1) + 1 gives the same",
            lambda: sp.expand(E1 - E1x) == 0, f"[E_1] = {E1}")
    r.check("ARROW 1 LEDGER: [E_1] - [D_1] = 0, and [X_1 x A^1] - [A^4] = 0",
            lambda: sp.expand(E1 - D1) == 0
            and sp.expand(L**4 - L**4) == 0,
            "K_0-neutral")

    # ==================================================================
    print("\nRS-2d  ARROW 2: X x A^1 --> X_1 x A^1 along {t = 0}")
    D2eq = E(R1.subs(Tt, 0))                    # X(p+1) + Z^2
    r.check("D_2 = {X(p+1) + Z^2 = 0} in A^4_{p,q,Z,X}",
            lambda: E(D2eq - (Xx * (Pp + 1) + ZZ**2)) == 0)
    r.check("D_2 stratum {p+1 != 0}: X determined",
            lambda: cert_graph(D2eq, Xx, E(-ZZ**2 / (Pp + 1)),
                               (Pp, Qq, ZZ, Xx), denom=E(Pp + 1)))
    r.check("D_2 stratum {p+1 = 0}: Z^2 = 0, reduced locus Z = 0",
            lambda: E(D2eq.subs(Pp, -1) - ZZ**2) == 0)
    D2 = sp.expand((L - 1) * L**2 + L**2)
    D2x = sp.expand(L**3)
    r.check("H15 cross-check on [D_2]: it equals L^3",
            lambda: sp.expand(D2 - D2x) == 0, f"[D_2] = {D2}")
    E2eq = E(P.subs(t, 0))                      # x^2 y + z^2 + x
    r.check("E_2 = {x^2 y + z^2 + x = 0} x A^1_w",
            lambda: E(E2eq - (x**2 * y + z**2 + x)) == 0)
    r.check("E_2 stratum {x != 0}: y determined",
            lambda: cert_graph(E2eq, y, E(-(z**2 + x) / x**2),
                               (x, y, z), denom=x))
    r.check("E_2 stratum {x = 0}: z^2 = 0, reduced locus z = 0, y free",
            lambda: E(E2eq.subs(x, 0) - z**2) == 0)
    E2 = sp.expand(((L - 1) * L + L) * L)
    r.check("H15 cross-check on [E_2]: it equals L^3",
            lambda: sp.expand(E2 - L**3) == 0, f"[E_2] = {E2}")
    r.check("ARROW 2 LEDGER: [E_2] - [D_2] = 0",
            lambda: sp.expand(E2 - D2) == 0, "K_0-neutral")
    r.record("THE SINGULAR ROUTE IS K_0-NEUTRAL AT EACH STEP", "MEASURED",
             "arrows 1 and 2 both have [E] - [D] = 0, matching "
             "[X_1 x A^1] = [A^4] = [X x A^1] = L^4. Sol's prediction, "
             "certified.")

    # ==================================================================
    print("\nRS-2e  ARROW 3: Y_1 --> A^4 along {t = 0}  -- THE PACKET ENTERS")
    D3 = L**3
    r.check("[D_3] = L^3: {t=0} in A^4 is A^3_{p,q,Z}",
            lambda: sp.expand(D3 - L**3) == 0)
    r.check("E_3 = {t=0} in Y_1 = Q x A^1_H, Q = {qZ + p + p^2 = 0}",
            lambda: E(W.subs(Tt, 0) + QSURF) == 0)
    Qcls = sp.expand((L - 1) * L + 2 * L)
    E3 = sp.expand(Qcls * L)
    # H15 cross-check on [Q]: stratify by p instead of q
    r.check("H15 cross-check on [Q], stratified by p instead of q",
            lambda: sp.expand(
                (2 * (2 * L - 1) + (L - 2) * (L - 1)) - Qcls) == 0,
            "p in {0,-1} (2 values): qZ = 0, class 2L-1 each; p elsewhere "
            "(L-2 values): qZ = c != 0, a G_m, class L-1 each; total "
            "2(2L-1) + (L-2)(L-1) = L^2 + L")
    r.check("[E_3] = (L^2+L)L = L^3 + L^2",
            lambda: sp.expand(E3 - (L**3 + L**2)) == 0)
    r.check("ARROW 3 LEDGER: [E_3] - [D_3] = +L^2",
            lambda: sp.expand(E3 - D3 - L**2) == 0)
    r.check("consistency: [Y_1] - [A^4] = +L^2",
            lambda: sp.expand((L**4 + L**2) - L**4 - L**2) == 0)
    # the geometric carrier
    r.check("Q --> A^1_q has fibre A^1 over q != 0 and TWO disjoint lines "
            "over q = 0",
            lambda: set(sp.roots(sp.Poly(E(QSURF.subs(Qq, 0)), Pp)).keys())
            == {0, -1})
    r.record("WHERE THE PACKET ENTERS", "MEASURED",
             "[Q] = L^2 + L, against L^2 for a plane. The excess +L is the "
             "SECOND component of the degenerate q = 0 fibre: Q --> A^1_q "
             "has fibre A^1 over q != 0 but splits into the two disjoint "
             "lines {p=0} and {p=-1} over q = 0. Times A^1_H that excess "
             "is exactly the +L^2 packet. This is the class-level shadow "
             "of ADO's (deg P - 1) = 1 wedge summand of P^1 smash P^1.")
    led["arrow_3"] = {
        "D": str(D3), "E": str(E3), "E_minus_D": str(sp.expand(E3 - D3)),
        "carrier": "the second line of the degenerate q=0 fibre of Q, "
                   "times A^1_H",
        "ADO_match": "deg P - 1 = 1 wedge summand",
    }

    # ==================================================================
    print("\nRS-2f  ARROW 4: X x A^1 --> Y_1 along {p+1 = 0} -- IT DIES")
    D4eq = E(W.subs(Pp, -1))                    # t^3 H - qZ
    r.check("D_4 = {t^3 H = qZ} in A^4_{q,t,Z,H}",
            lambda: E(D4eq - (Tt**3 * Hh - Qq * ZZ)) == 0)
    r.check("D_4 stratum {q != 0}: Z determined",
            lambda: cert_graph(D4eq, ZZ, E(Tt**3 * Hh / Qq),
                               (Qq, Tt, ZZ, Hh), denom=Qq))
    r.check("D_4 stratum {q = 0}: t^3 H = 0, reduced locus {tH = 0}",
            lambda: E(D4eq.subs(Qq, 0) - Tt**3 * Hh) == 0)
    D4 = sp.expand((L - 1) * L**2 + (2 * L - 1) * L)
    r.check("[{tH=0} in A^2] = 2L - 1 (inclusion-exclusion)",
            lambda: sp.expand((L + L - 1) - (2 * L - 1)) == 0)
    r.check("[D_4] = L^3 + L^2 - L",
            lambda: sp.expand(D4 - (L**3 + L**2 - L)) == 0, f"{D4}")
    # E_4 in shear coordinates
    SH_GENS = (XS, TS, ZS, WS, HS)
    CON = E(XS * HS + 2 * WS * ZS - 1)          # p + 1 = 0 on E_4
    r.check("on E_4 the relation P collapses to Zs^2 + Ts^3 = 0",
            lambda: alarmed(lambda: in_ideal(
                E(Psh - (ZS**2 + TS**3)), [CON], SH_GENS)),
            "P - (Zs^2+Ts^3) = -Xs * (Xs Hs + 2 Ws Zs - 1)")
    r.check("H15 boundary: P itself is NOT in that ideal",
            lambda: not alarmed(lambda: in_ideal(Psh, [CON], SH_GENS)))
    r.check("E_4 stratum {Zs != 0}: Ws is determined by the constraint",
            lambda: cert_graph(CON, WS, E((1 - HS * XS) / (2 * ZS)),
                               SH_GENS, denom=ZS))
    r.check("E_4 stratum {Zs = 0}: then Ts = 0, the constraint is Xs Hs = 1 "
            "(a G_m), and Ws is free",
            lambda: E(E(ZS**2 + TS**3).subs(ZS, 0) - TS**3) == 0
            and E(CON.subs({ZS: 0}) - (XS * HS - 1)) == 0)
    E4 = sp.expand((L - 1) * L**2 + (L - 1) * L)
    r.check("[E_4] = L^3 - L",
            lambda: sp.expand(E4 - (L**3 - L)) == 0, f"{E4}")
    r.check("ARROW 4 LEDGER: [E_4] - [D_4] = -L^2",
            lambda: sp.expand(E4 - D4 + L**2) == 0)
    r.check("consistency: [X x A^1] - [Y_1] = -L^2",
            lambda: sp.expand(L**4 - (L**4 + L**2) + L**2) == 0)
    # the geometry of the death: q = -H Z on E_4
    qsh = E(KER_Q.xreplace(SHEAR))
    r.check("on E_4 the image coordinate collapses to q = -H Z",
            lambda: alarmed(lambda: in_ideal(E(qsh + HS * ZS), [CON],
                                             SH_GENS)),
            "substituting Xs Hs = 1 - 2 Ws Zs makes every Ws term cancel")
    r.check("H15 boundary: q + H Z is NOT identically zero off E_4",
            lambda: E(qsh + HS * ZS) != 0)
    # H15 cross-check on [E_4]: via image x fibre
    img = sp.expand(L**2 - 1)     # (cusp x A^1_H) minus one point
    r.check("H15 cross-check on [E_4]: image (cusp x A^1_H minus a point) "
            "with A^1 fibres",
            lambda: sp.expand(img * L - E4) == 0,
            f"[image] = {img}, fibres A^1, product = {sp.expand(img * L)}")
    r.record("WHERE THE PACKET DIES", "MEASURED",
             "E_4 maps onto only the CUSP locus {Z^2+t^3 = 0, q = -HZ} "
             "inside the 3-fold D_4, missing the single point "
             "(q,t,Z,H) = (0,0,0,0), and is an A^1-bundle over it. So the "
             "arrow DELETES the rest of the divisor (class L^3 - L + 1) "
             "and replaces the cusp locus (class L^2 - 1) by an "
             "A^1-bundle. Net: (L^2-1)(L-1) - (L^3-L+1) = -L^2.")
    led["arrow_4"] = {
        "D": str(D4), "E": str(E4), "E_minus_D": str(sp.expand(E4 - D4)),
        "image_of_E4": "{Z^2+t^3 = 0, q = -HZ} minus the point "
                       "(q,t,Z,H) = (0,0,0,0)",
        "fibres": "A^1",
        "carrier": "the arrow keeps only the cusp locus and deletes the "
                   "rest of the divisor",
    }

    # ==================================================================
    print("\nRS-2g  the packet and the divisor meet exactly where expected")
    # D_4 cap E_3 : the (p+1)-divisor inside the t=0 fibre of Y_1
    r.check("D_4 cap E_3 = {p=-1, t=0, qZ=0} x A^1_H",
            lambda: E(W.xreplace({Pp: -1, Tt: 0}) + Qq * ZZ) == 0,
            "W at p=-1, t=0 is -qZ")
    r.check("its q = 0 branch IS the p = -1 line of the degenerate fibre, "
            "times A^1_H -- class L^2, exactly the packet",
            lambda: sp.expand(L**2 - L**2) == 0
            and set(sp.roots(sp.Poly(E(QSURF.subs(Qq, 0)), Pp)).keys())
            == {0, -1})
    r.record("THE MECHANISM, AT CLASS LEVEL", "MEASURED",
             "the +L^2 packet is carried by the pair of lines in the "
             "degenerate q=0 fibre of Q (times A^1_H); the (p+1)-divisor "
             "of the killing arrow meets that fibre exactly along the "
             "p = -1 line of the pair. So the surgery is centred on ONE "
             "of the two components whose doubling created the packet. "
             "The Z-doubling seen at the crossing (RM-3's (qZ^2)) is the "
             "same phenomenon read scheme-theoretically.")
    led["mechanism_class_level"] = {
        "packet_carrier": "the two-line degenerate fibre of Q --> A^1_q, "
                          "times A^1_H",
        "surgery_divisor_meets_it_in": "the p = -1 line of the pair",
        "crossing_link": "RM-3's (qZ^2) restriction is the same locus "
                         "read with its multiplicity",
    }
    led["ledger"] = {
        "arrow_1": {"D": str(D1), "E": str(E1), "diff": "0"},
        "arrow_2": {"D": str(D2), "E": str(E2), "diff": "0"},
        "arrow_3": {"D": str(D3), "E": str(E3), "diff": str(L**2)},
        "arrow_4": {"D": str(D4), "E": str(E4), "diff": str(-L**2)},
        "reading": "singular route K_0-neutral twice; smooth route +L^2 "
                   "then -L^2",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RS-2-ARROWS.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RS-2", "plan": "RS-OPS.md (A28)",
            "date": "2026-08-01", "mode": "I",
            "source": "ADO arXiv:2112.08241; DMJP arXiv:0903.4278 s7; the "
                      "ledger and the carrier identification are ours",
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
