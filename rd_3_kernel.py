#!/usr/bin/env python3
"""RD-3 — the kernel of the published LND, by conjugacy transport and
t-saturation (NOT elimination).  This is the heart of the visit.

The derivation is DMJP's (arXiv:0903.4278 Prop 7.2).  The kernel
computation is ours.

THE ROUTE (RD-OPS §4, RD-3), in three steps.

(1) LOCALISED, ON THE OTHER SIDE.  In Bt[w] put u = y + 1; the relation
    S = xy + z^2 + x + t^3 becomes x u + z^2 + t^3, so Bt[w] is the
    localised Danielewski-type ring C[x,u,z,t^{+-1},w]/(xu + z^2 + t^3),
    and Delta = t^6(-2z d/dx + u d/dz) kills u, t, w.  Inverting u
    solves the relation for x, so Bt[w][1/u] = C[t^{+-1}, w, u^{+-1}, z]
    on which Delta is exactly t^6 u d/dz -- a polynomial ring in ONE
    variable z over its kernel.  Hence
        ker(Delta | Bt[w][1/u]) = C[t^{+-1}, w, u^{+-1}],
    and since u is prime in Bt[w] (Bt[w]/(u) = C[x,z,t^{+-1},w]/(z^2+t^3)
    is a domain, z^2 + t^3 being irreducible over C[t^{+-1}]),
        ker(Delta | Bt[w]) = C[t^{+-1}][u, w] = C[t^{+-1}][y, w].

(2) TRANSPORT.  Prop 7.1 makes Phi : Bt[w] -> At[w] an isomorphism with
    inverse Psi, and partial = Phi o Delta o Psi, so
        ker(partial | At[w]) = Phi(ker Delta) = C[t^{+-1}][p, q],
        p = Phi(y) = x y - x w^2 - 2 z w,
        q = Phi(w) = 2w + y z + 3 x y w - 3 z w^2 - x w^3.

(3) t-SATURATION, not elimination.  ker(partial | A[w]) is the
    intersection A[w] cap C[t^{+-1}][p,q].  The saturation is TRIVIAL
    exactly when the reductions p, q mod t stay algebraically
    independent in the domain A[w]/(t) = C[x,y,z,w]/(x^2 y + z^2 + x):
    then H(p,q,t) in t A[w] forces H_0 = 0, i.e. t | H in C[p,q,t].

Ledger: runs_synthesis/RD-3-KERNEL.json.  Mode I.
"""

from __future__ import annotations

import json
import os
import sys
import time

import sympy as sp

from checker import is_in_ideal
from checker_twin import member as member_twin
from rd_common import (AW_t_valuation, DPART, DX, E, GENS, IOTA_Y, KER_P,
                       KER_Q, P, PHI, PSI, S, ap, delta, iota, is_poly,
                       x, y, z, t, w)

u = sp.Symbol("u")


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
        if isinstance(cond, tuple):
            cond, extra = cond[0], cond[1]
            note = f"{note} {extra}".strip() if note else str(extra)
        outcome = "PASS" if cond else "FAIL"
        self.results.append((name, outcome, dt))
        if note:
            self.notes[name] = note
        print(f"  {name:<50s} {outcome:<11s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<50s} {outcome:<11s}         [{note}]", flush=True)

    @property
    def ok(self):
        return all(o != "FAIL" for _, o, _ in self.results)

    @property
    def tally(self):
        return {k: sum(1 for _, o, _ in self.results if o == k)
                for k in self.CLASSES
                if any(o == k for _, o, _ in self.results)}


def jac_rank(fs, vs) -> int:
    """Rank of the Jacobian over the fraction field, by maximal minors.

    Robust where Matrix.rank() over a rational-function entry set is
    fragile: compute every maximal minor and stop at the first nonzero.
    """
    M = sp.Matrix([[sp.cancel(sp.diff(f, v)) for v in vs] for f in fs])
    k = min(M.shape)
    import itertools
    for size in range(k, 0, -1):
        for rows in itertools.combinations(range(M.shape[0]), size):
            for cols in itertools.combinations(range(M.shape[1]), size):
                if sp.cancel(M[rows, cols].det()) != 0:
                    return size
    return 0


def main() -> int:
    r = Runner()
    led: dict = {}

    r.check("tripwire: partial(x) = -2 t^6 (z + x w)",
            lambda: E(DX + 2 * t**6 * (z + x * w)) == 0)

    # ------------------------------------------------------------------
    print("\nRD-3a  ker(Delta) on Bt[w]: the localised Danielewski model")
    Su = E(S.subs(y, u - 1))
    r.check("S becomes x u + z^2 + t^3 under u = y + 1",
            lambda: E(Su - (x * u + z**2 + t**3)) == 0)
    r.check("Delta(y) = 0, Delta(t) = 0, Delta(w) = 0",
            lambda: delta(y) == 0 and delta(t) == 0 and delta(w) == 0)
    r.check("Delta(x) = -2 t^6 z and Delta(z) = t^6 u  (u = y+1)",
            lambda: E(delta(x) + 2 * t**6 * z) == 0
            and E(delta(z) - t**6 * (y + 1)) == 0)

    # inverting u solves the relation for x
    x_of = -(z**2 + t**3) / u
    r.check("inverting u: x = -(z^2+t^3)/u solves the relation",
            lambda: sp.cancel(E(x_of * u + z**2 + t**3)) == 0)
    # Delta read on C[t^{+-1}, w, u^{+-1}, z] must reproduce Delta(x)
    r.check("Delta = t^6 u d/dz on Bt[w][1/u]: reproduces Delta(x)",
            lambda: sp.cancel(E(t**6 * u * sp.diff(x_of, z)
                                - (-2 * t**6 * z))) == 0,
            "so ker(Delta|Bt[w][1/u]) = C[t^{+-1}, w, u^{+-1}]")
    # u is prime in Bt[w]
    r.check("z^2 + t^3 is irreducible over C[t^{+-1}] (u is prime)",
            lambda: len(sp.factor_list(z**2 + t**3, z, t)[1]) == 1
            and sp.factor_list(z**2 + t**3, z, t)[1][0][1] == 1,
            "Bt[w]/(u) is a domain")
    r.record("ker(Delta | Bt[w]) = C[t^{+-1}][y, w]", "MEASURED",
             "u = y+1 prime, so the u-saturation of C[t^{+-1}][w][u] "
             "inside Bt[w] is itself")
    led["ker_delta"] = {
        "statement": "ker(Delta | Bt[w]) = C[t^{+-1}][y, w]",
        "route": "u = y+1; Bt[w] = C[x,u,z,t^{+-1},w]/(xu+z^2+t^3); "
                 "Bt[w][1/u] = C[t^{+-1},w,u^{+-1},z] with Delta = "
                 "t^6 u d/dz; u prime since z^2+t^3 is irreducible "
                 "over C[t^{+-1}].",
    }

    # ------------------------------------------------------------------
    print("\nRD-3b  transport through the conjugacy (Prop 7.1)")
    p, q = KER_P, KER_Q
    print(f"    p = Phi(y) = {p}")
    print(f"    q = Phi(w) = {q}")
    # partial(f) for a polynomial representative f
    def dpart(f):
        return E(sum(DPART[v] * sp.diff(f, v) for v in GENS))

    r.check("partial(p) = 0 in At[w]  [twin A]",
            lambda: is_in_ideal(dpart(p), P, y))
    r.check("partial(p) = 0 in At[w]  [twin B]",
            lambda: member_twin(dpart(p), P, y))
    r.check("partial(q) = 0 in At[w]  [twin A]",
            lambda: is_in_ideal(dpart(q), P, y))
    r.check("partial(q) = 0 in At[w]  [twin B]",
            lambda: member_twin(dpart(q), P, y))
    r.check("partial(t) = 0", lambda: E(DPART[t]) == 0)
    r.check("x is NOT in ker partial (DMJP Prop 7.2)",
            lambda: E(DX) != 0)
    # the transport is exact: Psi carries p, q back to y, w mod (S)
    r.check("Psi(p) = y mod (S)",
            lambda: is_in_ideal(E(ap(PSI, p) - y) * t**12, S, y),
            "cleared of t-denominators before the membership test")
    r.check("Psi(q) = w mod (S)",
            lambda: is_in_ideal(E(ap(PSI, q) - w) * t**18, S, y))
    # H15 boundary: a near-miss must fail
    r.check("H15 boundary: partial(p + x) != 0",
            lambda: not is_in_ideal(dpart(E(p + x)), P, y))
    led["transport"] = {
        "p": str(p), "q": str(q),
        "statement": "ker(partial | At[w]) = C[t^{+-1}][p, q]",
    }

    # ------------------------------------------------------------------
    print("\nRD-3c  p, q, t are algebraically independent")
    MODEL_VARS = (x, z, t, w)
    ip, iq = iota(p), iota(q)
    rk = jac_rank([ip, iq, t], MODEL_VARS)
    r.check("Jacobian rank of (p, q, t) in C[x^{+-1},z,t,w] is 3",
            lambda: rk == 3, f"rank {rk}")
    r.check("H15 boundary: a dependent triple scores rank < 3",
            lambda: jac_rank([ip, iq, E(ip + 2 * iq)], MODEL_VARS) == 2,
            "the rank test can fail")
    led["independence"] = {"jacobian_rank_pqt": rk}

    # ------------------------------------------------------------------
    print("\nRD-3d  the t-saturation: is A[w] cap C[t^{+-1}][p,q] bigger?")
    # A[w]/(t) = C[x,y,z,w]/(x^2 y + z^2 + x); model y -> -(z^2+x)/x^2.
    ybar = -(z**2 + x) / x**2
    pbar = sp.cancel(E(p.subs(t, 0).xreplace({y: ybar})))
    qbar = sp.cancel(E(q.subs(t, 0).xreplace({y: ybar})))
    r.check("A[w]/(t) is a domain: x^2 y + z^2 + x is irreducible",
            lambda: len(sp.factor_list(x**2 * y + z**2 + x, x, y, z)[1]) == 1)
    rk2 = jac_rank([pbar, qbar], (x, z, w))
    r.check("p mod t and q mod t stay algebraically independent",
            lambda: rk2 == 2, f"Jacobian rank {rk2} in C[x^{{+-1}},z,w]")
    r.check("H15 boundary: the mod-t rank test can fail",
            lambda: jac_rank([pbar, E(3 * pbar)], (x, z, w)) == 1)
    # the saturation conclusion, plus a direct spot-check
    r.check("p is not in t*A[w] and q is not in t*A[w]",
            lambda: AW_t_valuation(p)[0] == 0 and AW_t_valuation(q)[0] == 0,
            "so p/t, q/t are not in A[w]")
    for expr, nm in ((E(p * q), "p q"), (E(p**2 + q), "p^2 + q"),
                     (E(p**3 - q**2), "p^3 - q^2"),
                     (E(p * q + p), "pq + p")):
        r.check(f"t-valuation of {nm} in A[w] is 0 (saturation spot-check)",
                lambda ee=expr: AW_t_valuation(ee)[0] == 0)
    r.record("ker(partial | A[w]) = C[p, q, t]", "MEASURED",
             "the t-saturation of C[p,q,t] inside A[w] is itself, "
             "because p mod t and q mod t remain algebraically "
             "independent in the domain A[w]/(t)")
    led["saturation"] = {
        "jacobian_rank_pbar_qbar": rk2,
        "argument":
            "H in C[p,q,t] with H in t A[w] reduces mod t to H_0(pbar,"
            "qbar) = 0 in the DOMAIN A[w]/(t); algebraic independence of "
            "pbar, qbar forces H_0 = 0, i.e. t | H in C[p,q,t]. Induct. "
            "Hence A[w] cap C[t^{+-1}][p,q] = C[p,q,t].",
        "conclusion": "ker(partial | A[w]) = C[p,q,t], a polynomial ring "
                      "in three variables (C^[3]).",
    }

    # ------------------------------------------------------------------
    print("\nRD-3e  consequences")
    r.check("ker(partial_red) = ker(partial)  (t^3 is a kernel element)",
            lambda: E(DPART[t]) == 0,
            "content division does not change the kernel")
    r.check("A[w] is strictly bigger than the kernel: x is outside",
            lambda: E(DX) != 0)
    led["consequences"] = {
        "kernel_is_polynomial":
            "ker(partial | A[w]) = C[p,q,t] = C^[3]. So the invariant "
            "ring of this Ga-action on X x A^1 IS a polynomial ring in "
            "three variables -- the RD-5 question 'is ker partial = "
            "C^[3]?' is answered YES, and the whole weight of the "
            "cancellation question moves onto the SLICE.",
        "explicit_generators": {"p": str(p), "q": str(q), "t": "t"},
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RD-3-KERNEL.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RD-3", "plan": "RD-OPS.md (A22)",
            "date": "2026-07-31", "mode": "I",
            "source": "the derivation is DMJP arXiv:0903.4278 Prop 7.2; "
                      "the kernel computation is ours",
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
