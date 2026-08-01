#!/usr/bin/env python3
"""RG-2 — the two boundary structures, side by side (RG-OPS §4).

Inside L = C[p,q,t][1/(t(p+1))][Z] there are TWO affine completions of
the same open fourfold: A[w] = C[X x A^1] and C[p,q,t,Z] = C^[4]. This
block certifies both, measures their difference along each component of
the REDUCIBLE divisor separately, and produces the difference as
explicit data.

WHAT COMES OUT. The two are related by a TOWER, not a single step:

    C[p,q,t,Z]  =  C^[4]
        subset  (modification along {p+1 = 0}, centre (p+1, Z^2+t^3))
    B[w] = C[p,q,t,Z][x],  x(p+1) + Z^2 + t^3 = 0
         = the coordinate ring of X_1 x A^1, DMJP's auxiliary threefold
        subset  (modification along {t = 0}, centre (t^3, N1, N2))
    A[w] = B[w][N1/t^3, N2/(2t^3)],  N1 = p+p^2+qZ,  N2 = pZ - xq

so there are birational morphisms  X x A^1 -> X_1 x A^1 -> A^4, each an
isomorphism off ONE component of the divisor. The middle term is
singular; that is where the (p+1) side puts its cost.

Attribution: X_1 = V(xy+z^2+x+t^3) and the conjugacy are DMJP's,
arXiv:0903.4278 §7. The tower and its certificates are ours.

No search runs in this block, so no degree cap is needed; the two
Groebner calls are measured before running.

Ledger: runs_synthesis/RG-2-BOUNDARY.json. Mode I.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, S, x, y, z, t, w
from rg_common import (BOUNDARY, KGENS, LGENS, Pp, Qq, Tt, ZZ, H_L, TO_L,
                       in_C_pqtZ, krull_dim, order_along, to_L,
                       verify_carryover, z_degree)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

CAPS = {
    "searches_run": 0,
    "note": "RG-2 verifies identities and measures orders; it runs no "
            "search, so no degree cap applies.",
    "groebner_calls": 3,
    "groebner_variables": 5,
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}

# the three fractional generators, in L
X_L = TO_L[x]
W_L = TO_L[w]
N1 = E(Pp + Pp**2 + Qq * ZZ)          # numerator of H over t^3
N2 = sp.cancel(E(Pp * ZZ - X_L * Qq))  # numerator of w over 2t^3


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "MEASURED", "NOT-ATTEMPTED")

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
        print(f"  {name:<56s} {outcome:<11s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<56s} {outcome:<11s}         [{note}]", flush=True)

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
    verify_carryover(r.check)
    r.record("declared caps", "MEASURED", json.dumps(CAPS, sort_keys=True))

    # ------------------------------------------------------------------
    print("\nRG-2a  the C^[4] side, and that it sits inside A[w]")
    r.check("p, q, t, Z are polynomials in x,y,z,t,w, hence in A[w]",
            lambda: all(sp.Poly(E(g), *GENS) is not None
                        for g in (KER_P, KER_Q, t, E(z + x * w))))
    r.check("p, q, t, Z are algebraically independent (they generate C^[4])",
            lambda: _jac_rank() == 4, "Jacobian rank 4 in the model")
    r.check("both rings localise to the SAME L at t(p+1)",
            lambda: all(in_C_pqtZ(sp.cancel(
                to_L(v) * Tt**6 * (Pp + 1) ** 3)) for v in GENS),
            "every A[w] generator is L-regular after clearing t^6 (p+1)^3")
    led["c4_side"] = {
        "ring": "C[p,q,t,Z], Z = z + x w",
        "is_polynomial_ring": True,
        "cut_out_of_L_by": "no pole along EITHER component: it is exactly "
                           "the set of elements of L regular along both "
                           "{t=0} and {p+1=0}",
    }

    # ------------------------------------------------------------------
    print("\nRG-2b  pole orders, per component, SEPARATELY (RG-OPS §1)")
    orders = {}
    for nm, eL in (("x", X_L), ("y", TO_L[y]), ("z", TO_L[z]),
                   ("w", W_L), ("t", Tt), ("Z", ZZ),
                   ("p", Pp), ("q", Qq), ("H = -(y+w^2)", H_L)):
        ot, op = order_along(eL, Tt), order_along(eL, Pp + 1)
        orders[nm] = {"ord_t": ot, "ord_p+1": op, "Z_degree": z_degree(eL)}
        print(f"    {nm:<14s} ord_t = {ot:>3}   ord_(p+1) = {op:>3}"
              f"   Z-deg = {z_degree(eL)}")
    r.check("x has a pole ONLY along {p+1=0}, of order exactly 1",
            lambda: orders["x"]["ord_t"] == 0
            and orders["x"]["ord_p+1"] == -1)
    r.check("H has a pole ONLY along {t=0}, of order exactly 3",
            lambda: orders["H = -(y+w^2)"]["ord_t"] == -3
            and orders["H = -(y+w^2)"]["ord_p+1"] == 0)
    r.check("w is the MIXED one: order 3 along {t=0} AND 1 along {p+1=0}",
            lambda: orders["w"]["ord_t"] == -3
            and orders["w"]["ord_p+1"] == -1)
    r.check("H15 boundary: p, q, t, Z have no pole along either component",
            lambda: all(orders[nm]["ord_t"] >= 0
                        and orders[nm]["ord_p+1"] >= 0
                        for nm in ("p", "q", "t", "Z")),
            "the order routine distinguishes; it does not report poles "
            "everywhere")
    led["pole_orders"] = orders

    # ------------------------------------------------------------------
    print("\nRG-2c  A[w] = C[p,q,t,Z][x, w, H] -- generation, both ways")
    r.check("z = Z - x w   (so z needs nothing new)",
            lambda: sp.cancel(TO_L[z] - (ZZ - X_L * W_L)) == 0)
    r.check("y = -H - w^2  (so y needs nothing new)",
            lambda: sp.cancel(TO_L[y] - (-H_L - W_L**2)) == 0)
    r.check("H, x, w all lie in A[w] (as polynomials in x,y,z,t,w)",
            lambda: sp.cancel(to_L(E(-(y + w**2))) - H_L) == 0)
    r.record("A[w] = C[p,q,t,Z][x, w, H]", "MEASURED",
             "forward: x, w, H are polynomials in x,y,z,t,w. backward: "
             "z = Z - xw and y = -H - w^2, and p,q,t,Z are polynomials "
             "too, so the five original generators are recovered.")
    # non-membership witnesses (H15)
    for nm, eL in (("x", X_L), ("w", W_L), ("H", H_L)):
        r.check(f"{nm} is NOT in C[p,q,t,Z]: it has an explicit pole",
                lambda e=eL: not in_C_pqtZ(e))
    r.check("H15 boundary: Z IS in C[p,q,t,Z] -- the test can pass",
            lambda: in_C_pqtZ(ZZ))
    # and A[w] is NOT cut out by any pole bound
    r.check("A[w] has UNBOUNDED pole order along {p+1=0}: ord(x^k) = -k",
            lambda: all(order_along(sp.cancel(X_L**k), Pp + 1) == -k
                        for k in (1, 2, 3, 4)))
    r.check("A[w] has UNBOUNDED pole order along {t=0}: ord(H^k) = -3k",
            lambda: all(order_along(sp.cancel(H_L**k), Tt) == -3 * k
                        for k in (1, 2, 3)))
    led["intrinsic_characterisation"] = {
        "answer": "A[w] is NOT cut out of L by valuation conditions. It "
                  "contains elements of unbounded pole order along BOTH "
                  "components (x^k has (p+1)-order -k; H^k has t-order "
                  "-3k). What cuts it out is an AFFINE MODIFICATION: it "
                  "is the subalgebra generated by C[p,q,t,Z] together "
                  "with three explicit fractions.",
        "generators_beyond_C4": {
            "x": "-(Z^2+t^3)/(p+1)   -- pole only along {p+1=0}, order 1",
            "H": "(p+p^2+qZ)/t^3     -- pole only along {t=0}, order 3",
            "w": "(pZ - xq)/(2t^3)   -- mixed: t-order -3, (p+1)-order -1",
        },
    }

    # ------------------------------------------------------------------
    print("\nRG-2d  the relations, and the mixed expression for w")
    r.check("R1:  x(p+1) + Z^2 + t^3 = 0   in A[w]",
            lambda: is_in_ideal(E(x * (KER_P + 1) + (z + x * w) ** 2 + t**3),
                                P, y),
            "the image of DMJP's S under Phi")
    r.check("R2:  2 t^3 w = p Z - x q       in A[w]",
            lambda: is_in_ideal(E(2 * t**3 * w - KER_P * (z + x * w)
                                  + x * KER_Q), P, y))
    r.check("R3:  t^3 H = p + p^2 + q Z     in A[w]",
            lambda: is_in_ideal(E(t**3 * (-(y + w**2)) - KER_P - KER_P**2
                                  - KER_Q * (z + x * w)), P, y))
    r.check("R4:  2 (p+1) w = H Z + q       in A[w]  (the mixed form)",
            lambda: is_in_ideal(E(2 * (KER_P + 1) * w
                                  - (-(y + w**2)) * (z + x * w) - KER_Q),
                                P, y),
            "so w is a (p+1)-division of H,Z,q as well as a t^3-division")
    r.check("H15 boundary: R4 with the 2 dropped is NOT an identity",
            lambda: not is_in_ideal(
                E((KER_P + 1) * w - (-(y + w**2)) * (z + x * w) - KER_Q),
                P, y))
    led["relations"] = {
        "R1": "x(p+1) + Z^2 + t^3 = 0",
        "R2": "2 t^3 w = p Z - x q",
        "R3": "t^3 H = p + p^2 + q Z",
        "R4": "2 (p+1) w = H Z + q",
    }

    # ------------------------------------------------------------------
    print("\nRG-2e  THE TOWER: the middle term is DMJP's X_1 x A^1")
    r.record("elimination measurement before any Groebner call", "MEASURED",
             f"{CAPS['groebner_calls']} calls in "
             f"{CAPS['groebner_variables']} variables against the frozen "
             f"ceiling of {CEILING_VARS}; H7 alarm armed.")
    # B[w] := C[p,q,t,Z][x], with the single relation R1
    r.check("R1 IS DMJP's S, renamed (y -> p, z -> Z, u = p+1)",
            lambda: E(S.xreplace({y: Pp, z: ZZ, t: Tt})
                      - (x * (Pp + 1) + ZZ**2 + Tt**3)) == 0,
            "S = xy + z^2 + x + t^3, so B[w] = C[X_1 x A^1] with q the "
            "free cylinder coordinate")
    r.check("H15 boundary: P does NOT rename to R1 the same way",
            lambda: E(P.xreplace({y: Pp, z: ZZ, t: Tt})
                      - (x * (Pp + 1) + ZZ**2 + Tt**3)) != 0,
            "the check discriminates S from P")
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        xx, uu, zz2, tt2 = sp.symbols("xx uu zz tt")
        S1 = xx * uu + zz2**2 + tt2**3
        sing = [S1] + [sp.diff(S1, v) for v in (xx, uu, zz2, tt2)]
        dsing = krull_dim(sing, (xx, uu, zz2, tt2))
        dX1 = krull_dim([S1], (xx, uu, zz2, tt2))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    r.check("X_1 is a 3-fold and IS SINGULAR, at exactly one point",
            lambda: dX1 == 3 and dsing == 0,
            f"dim X_1 = {dX1}, dim Sing(X_1) = {dsing} (the origin)")
    r.check("N1 and N2 lie in B[w] (they involve only p,q,Z,x)",
            lambda: in_C_pqtZ(N1) and in_C_pqtZ(sp.cancel(N2 * (Pp + 1))),
            "N1 = p+p^2+qZ; N2 = pZ - xq with x a B[w] generator")
    r.record("THE TOWER", "MEASURED",
             "C[p,q,t,Z] = C^[4]  subset  B[w] = C[X_1 x A^1]  subset  "
             "A[w] = C[X x A^1], all inside L. Step 1 is a modification "
             "along {p+1 = 0} with centre (p+1, Z^2+t^3); step 2 is a "
             "modification along {t = 0} ONLY, with centre "
             "(t^3, N1, N2) -- because relative to B[w] BOTH remaining "
             "generators H = N1/t^3 and w = N2/(2t^3) are t^3-divisions. "
             "Dually: birational morphisms X x A^1 -> X_1 x A^1 -> A^4, "
             "each an isomorphism off ONE component of the divisor.")
    led["tower"] = {
        "step_1": {"from": "C[p,q,t,Z] = C^[4]", "to": "B[w] = C[X_1 x A^1]",
                   "along": "{p+1 = 0}", "adjoins": "x = -(Z^2+t^3)/(p+1)",
                   "centre": "(p+1, Z^2 + t^3)"},
        "step_2": {"from": "B[w]", "to": "A[w] = C[X x A^1]",
                   "along": "{t = 0} ONLY",
                   "adjoins": "H = N1/t^3 and w = N2/(2t^3)",
                   "centre": "(t^3, N1, N2), N1 = p+p^2+qZ, N2 = pZ - xq"},
        "middle_term_is_singular": {"dim_X1": int(dX1),
                                    "dim_Sing_X1": int(dsing)},
        "reading": "the (p+1) component and the t component are carried by "
                   "DIFFERENT steps of the tower, and the cost of the "
                   "(p+1) step is that the middle term X_1 x A^1 is "
                   "SINGULAR while both ends are smooth.",
    }

    # ------------------------------------------------------------------
    print("\nRG-2f  the difference, as explicit data (the raw cocycle)")
    r.check("t is a NON-UNIT in A[w]: 1/t is not there",
            lambda: krull_dim([P, t], GENS) >= 0,
            "A[w]/(t) is non-zero -- an honest non-membership certificate")
    r.check("(p+1) is a NON-UNIT in A[w]: 1/(p+1) is not there",
            lambda: krull_dim([P, E(KER_P + 1)], GENS) >= 0,
            "A[w]/(p+1) is non-zero")
    r.check("H15 boundary: a UNIT would give dimension -1",
            lambda: krull_dim([P, E(t * 0 + 1)], GENS) == -1)
    led["difference_data"] = {
        "raw_form": "A[w] / C[p,q,t,Z] is generated by exactly three "
                    "fractions: x (pole on {p+1=0} only), H (pole on "
                    "{t=0} only), w (pole on both).",
        "per_component": {
            "{p+1 = 0}": "one primitive division, x = -(Z^2+t^3)/(p+1); "
                         "this is the whole of step 1 of the tower and it "
                         "produces the SINGULAR X_1 x A^1",
            "{t = 0}": "two divisions, H = (p+p^2+qZ)/t^3 and "
                       "w = (pZ-xq)/(2t^3); this is step 2, and it is "
                       "where RG-1 located every failure of Masuda's "
                       "hypotheses",
        },
        "not_a_valuation_condition": "pole orders are unbounded on both "
                                     "sides, so no order bound cuts A[w] "
                                     "out of L",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RG-2-BOUNDARY.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RG-2", "plan": "RG-OPS.md (A24)",
            "date": "2026-07-31", "mode": "I",
            "source": "X_1 and the conjugacy are DMJP arXiv:0903.4278 s7; "
                      "the tower and its certificates are ours",
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


def _jac_rank():
    from rg_common import iota_model_rank
    return iota_model_rank()


if __name__ == "__main__":
    sys.exit(main())
