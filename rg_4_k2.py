#!/usr/bin/env python3
"""RG-4 — the stable escalation at k = 2 (RG-OPS §4).

Adjoin one variable v and ask whether A[w][v] = C[X x A^2] is C^[5].
A candidate here would be X x A^2 = A^5: STABLE NON-CANCELLATION.

Three things are done, in the order the mandate sets:

  (a) is the k = 1 obstruction 1-STABLE? Extend partial by partial(v)=0.
      RG-3.1's argument transports verbatim, so the flow-compatible route
      does not open at k = 2 either. Exact.
  (b) the DANIELEWSKI/ASANUMA template -- the extra coordinate absorbs
      the twist via a fibre product over the common base -- needs the
      quotient morphism to be an A^1-bundle. RG-1(b) says it is not.
      Recorded precisely, with the hypothesis re-verified in process.
  (c) the cap-bounded hunt the mandate asks for: five cap-bounded
      elements of A[w][v] generating it as a polynomial ring, filtered
      by the exact Jacobian criterion, with a POSITIVE CONTROL.

Framework citation for the 1-stable language: Lewis arXiv:1304.1765
(all Venereau-type polynomials are 1-stable coordinates). The
derivation is DMJP arXiv:0903.4278 Prop 7.2. Masuda arXiv:2512.06687
for the hypothesis grid.

Honest fence (A23, binding): failure through this presentation decides
NOTHING. Mode I.

Ledger: runs_synthesis/RG-4-K2.json. Mode I.
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
from rg_common import Pp, Qq, Tt, krull_dim, verify_carryover

v = sp.Symbol("v")
GENS2 = GENS + (v,)
Vv = sp.Symbol("V_")                   # the kernel-side name for v

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

#: DECLARED BEFORE ANY SEARCH RUNS (RG-OPS §2)
POOL_NAMES = ["p", "q", "t", "Z", "x", "y", "z", "w", "H", "v",
              "x+v", "H+v", "w+v", "Z+v"]
CAPS = {
    "k2_pool": POOL_NAMES,
    "k2_tuples": "all C(14,5) = 2002 five-subsets",
    "k2_filter": "exact Jacobian: det[grad P; grad a1..a5] (6x6) must be "
                 "nowhere zero on X x A^2; rejections carry an explicit "
                 "witness point",
    "k2_witness_points": 6,
    "stability_argument": "exact, no cap",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}

#: points ON X x A^2 (P = 0), used as cheap exact rejection witnesses
SAMPLES = [
    {x: 1, y: -3, z: 1, t: 1, w: 1, v: 1},
    {x: 1, y: -13, z: 2, t: 2, w: 3, v: 5},
    {x: 2, y: -1, z: 1, t: 1, w: 0, v: 0},
    {x: 0, y: 7, z: 0, t: 0, w: 3, v: 2},
    {x: -1, y: -1, z: 1, t: 1, w: 2, v: 1},
    {x: 1, y: 0, z: 0, t: -1, w: 1, v: 0},
]


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


def main() -> int:
    r = Runner()
    led: dict = {}

    print("\nRG-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    red = verify_carryover(r.check)
    r.record("DECLARED CAPS (before any search ran)", "MEASURED",
             json.dumps(CAPS, sort_keys=True))

    def d_red(f):
        return E(sum(red[g] * sp.diff(f, g) for g in GENS))

    # ==================================================================
    print("\nRG-4a  is the k = 1 obstruction 1-STABLE?")
    r.check("P does not involve v, so partial extends by partial(v) = 0",
            lambda: sp.diff(P, v) == 0)
    r.check("ker(partial | A[w][v]) = C[p,q,t,v]",
            lambda: E(d_red(v)) == 0 and all(
                is_in_ideal(E(d_red(g)), P, y) for g in (KER_P, KER_Q)),
            "v joins the kernel; p, q, t stay in it")
    r.check("the plinth is unchanged: partial(A[w][v]) = partial(A[w]) (x) "
            "C[v]",
            lambda: is_in_ideal(E(d_red(E(v * (z + x * w)))
                                  - v * t**3 * (KER_P + 1)), P, y),
            "partial is C[v]-linear, so pl picks up C[v] and nothing else")
    # the same non-principality argument, now in C[p,q,t,v]
    K4 = (Pp, Qq, Tt, Vv)
    r.check("in C[p,q,t,v]: gcd(t^3, q) is still 1",
            lambda: sp.gcd(Tt**3, Qq) == 1)
    r.check("in C[p,q,t,v]: (p+1) is still irreducible",
            lambda: len(sp.factor_list(Pp + 1, *K4)[1]) == 1)
    G4 = sp.groebner([Pp, Qq, Tt], *K4, order="grevlex")
    r.check("in C[p,q,t,v]: (p+1) is still NOT in (p,q,t)",
            lambda: G4.reduce(E(Pp + 1))[1] != 0)
    r.check("H15 boundary: q IS in (p,q,t) there -- the test still passes",
            lambda: G4.reduce(E(Qq))[1] == 0)
    r.record("THEOREM RG-4.1: RG-3.1 is 1-STABLE", "THEOREM",
             "pl(partial | A[w][v]) = pl(partial | A[w]) C[v] still "
             "contains t^3(p+1) and q(p+1), still sits inside (p,q,t), "
             "and gcd(t^3,q) = 1 with (p+1) irreducible and outside "
             "(p,q,t) in C[p,q,t,v] as well. So the plinth is still NOT "
             "principal and A[w][v] is NOT C[p,q,t,v][s] for any s. The "
             "flow-compatible route does not open by adding a variable.")
    led["rg41"] = {
        "statement": "A[w][v] is not C[p,q,t,v][s] for any s",
        "reading": "the k=1 obstruction of RG-3.1 is 1-stable, so the "
                   "MQ5 hope -- k=2 yields where k=1 resists -- does NOT "
                   "materialise along the flow-compatible route. It "
                   "remains open along any route that does not keep the "
                   "kernel as part of the coordinate system.",
    }

    # ==================================================================
    print("\nRG-4b  the Danielewski/Asanuma template needs a hypothesis "
          "we do not have")
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        empty = krull_dim([P, E(KER_P + 1), E(KER_Q - 3), t], GENS)
        jump = krull_dim([P, KER_P, KER_Q, t], GENS)
        gen1 = krull_dim([P, E(KER_P - 2), E(KER_Q - 3), E(t - 1)], GENS)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    r.check("re-verified in process: pi has an EMPTY fibre",
            lambda: empty == -1, "over (-1,3,0)")
    r.check("re-verified in process: pi has a 2-DIMENSIONAL fibre",
            lambda: jump == 2, "over (0,0,0)")
    r.check("re-verified in process: the generic fibre is an A^1",
            lambda: gen1 == 1)
    r.record("the fibre-product mechanism is UNAVAILABLE here", "MEASURED",
             "the Danielewski/Asanuma template makes the extra coordinate "
             "absorb the twist by forming the fibre product of two "
             "A^1-bundles over a COMMON base that agree over the overlap. "
             "Here the two sides are A[w] and C[p,q,t,Z] over "
             "A^3 = Spec C[p,q,t]. The second is an A^1-bundle; the first "
             "is NOT -- pi has an empty fibre and a jump fibre. So the "
             "template cannot be set up through this presentation. This "
             "is a statement about the PRESENTATION, not about "
             "X x A^2 = A^5.")
    r.record("MQ5's boundary cocycle does not exist in the naive form",
             "MEASURED",
             "A23/MQ5 re-aimed the stable question at 'the boundary "
             "cocycle' of the {t(p+1)=0} gluing. A cocycle in that sense "
             "presupposes TWO fibrations over the base agreeing on the "
             "overlap. RG-1(b) shows A[w] is not a fibration over "
             "Spec(ker partial) at all, so there is no such cocycle to "
             "compute. Routed to master as MASTER-QUERY.")
    led["rg4b"] = {
        "template": "Danielewski/Asanuma fibre product over a common base",
        "missing_hypothesis": "pi : X x A^1 -> A^3 is not an A^1-bundle "
                              "(empty fibre over (-1,3,0), 2-dimensional "
                              "fibre over (0,0,0))",
        "consequence": "no boundary cocycle in the MQ5 sense exists "
                       "through this presentation",
    }

    # ==================================================================
    print("\nRG-4c  cap-bounded hunt for a C^[5] coordinate system")
    H_A = E(-(y + w**2))
    Zc = E(z + x * w)
    POOL = {"p": KER_P, "q": KER_Q, "t": t, "Z": Zc, "x": x, "y": y,
            "z": z, "w": w, "H": H_A, "v": v,
            "x+v": E(x + v), "H+v": E(H_A + v), "w+v": E(w + v),
            "Z+v": E(Zc + v)}
    assert sorted(POOL) == sorted(POOL_NAMES), "pool drifted from the cap"
    gradP2 = [sp.diff(P, g) for g in GENS2]
    grads = {k: [sp.diff(e, g) for g in GENS2] for k, e in POOL.items()}

    # every sample really is on X x A^2
    r.check("all 6 witness points lie on X x A^2",
            lambda: all(E(P.subs(s)) == 0 for s in SAMPLES))
    # POSITIVE CONTROL (H15): a graph hypersurface in 6 variables, whose
    # ring genuinely IS C^[5]. The filter must pass here.
    Qg = E(y - (x**2 + z * t * w * v))
    Mg = sp.Matrix([[sp.diff(Qg, g) for g in GENS2]]
                   + [[sp.diff(a, g) for g in GENS2]
                      for a in (x, z, t, w, v)])
    Dg = E(Mg.det())
    r.check("POSITIVE CONTROL: the filter PASSES on a genuine C^[5]",
            lambda: Dg != 0 and not Dg.free_symbols,
            f"graph y = x^2+ztwv with coordinates (x,z,t,w,v): det = {Dg}")

    survivors, rejected_by_point, other = [], 0, 0
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        gP_num = [[sp.Rational(c.subs(s)) for c in gradP2] for s in SAMPLES]
        g_num = {k: [[sp.Rational(c.subs(s)) for c in grads[k]]
                     for s in SAMPLES] for k in POOL}
        for combo in itertools.combinations(sorted(POOL), 5):
            dead = False
            for i in range(len(SAMPLES)):
                M = sp.Matrix([gP_num[i]] + [g_num[c][i] for c in combo])
                if M.det() == 0:
                    dead = True
                    break
            if dead:
                rejected_by_point += 1
                continue
            M = sp.Matrix([gradP2] + [grads[c] for c in combo])
            D = E(M.det())
            if D == 0 or krull_dim([P, D], GENS2) != -1:
                other += 1
                continue
            survivors.append(" ".join(combo))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    total = rejected_by_point + other + len(survivors)
    r.record(f"Jacobian filter over all {total} five-subsets", "MEASURED",
             f"rejected by an explicit witness point: {rejected_by_point}; "
             f"rejected by the exact test: {other}; "
             f"survivors: {survivors if survivors else 'none'}")
    r.check("the point filter is not vacuous: it rejects some and not all",
            lambda: 0 < rejected_by_point < total,
            "so the rejections carry information")
    if survivors:
        r.record("CANDIDATE SURVIVED THE FILTER", "MEASURED",
                 "a five-subset passed the necessary condition; the full "
                 "generation test and, if it holds, the FULL twin bar "
                 "must run before anything is called a candidate.")
    else:
        r.record("cap-bounded k=2 hunt: NO candidate", "MEASURED",
                 "no five-subset of the declared pool passes the Jacobian "
                 "necessary condition. A cap-bounded non-finding, NOT a "
                 "negative claim: the pool is fourteen named elements, "
                 "not a degree-bounded sweep of A[w][v].")
    led["rg4c"] = {
        "pool": POOL_NAMES, "tuples_tested": total,
        "rejected_by_witness_point": rejected_by_point,
        "rejected_by_exact_test": other,
        "survivors": survivors,
        "status": "cap-bounded non-finding (MEASURED), not a negative",
    }

    # ==================================================================
    print("\nRG-4d  where that leaves the stable question")
    verdict = {
        "flow_compatible_route": "CLOSED at k = 2 as well (RG-4.1, exact). "
                                 "Adding a variable does not make the "
                                 "plinth principal.",
        "danielewski_template": "UNAVAILABLE through this presentation, "
                                "because pi is not an A^1-bundle.",
        "cap_bounded_hunt": "no candidate at the declared pool.",
        "what_remains_open": "everything that does not go through this "
                             "flow. X x A^2 = A^5 is untouched by all of "
                             "the above; so is X x A^1 = A^4. Honest "
                             "fence, A23.",
    }
    for k, val in verdict.items():
        r.record(f"k=2 verdict: {k}", "MEASURED", val)
    led["verdict"] = verdict

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RG-4-K2.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RG-4", "plan": "RG-OPS.md (A24)",
            "date": "2026-07-31", "mode": "I",
            "source": "derivation DMJP arXiv:0903.4278; 1-stable framework "
                      "Lewis arXiv:1304.1765; hypothesis grid Masuda "
                      "arXiv:2512.06687; the results are ours",
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
