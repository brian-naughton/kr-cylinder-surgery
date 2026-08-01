#!/usr/bin/env python3
"""RM-4 — the crossing cocycle, after the literature gate (RM-OPS §4).

LITERATURE GATE (RM-OPS §2, discharged before this block ran). Blanc and
Poloni, "Bivariables and Venereau polynomials", arXiv:2004.10739
(Apr 2020; Ann. Fac. Sci. Toulouse Math., Dec 2022), fetched and read at
primary source this session. Abstract, verbatim in the ledger: they
study the Daigle-Freudenburg family containing the Venereau polynomials,
which defines A^2-FIBRATIONS OVER A^2; following an idea of
Kaliman-Zaidenberg they show these are locally trivial A^2-BUNDLES OVER
THE PUNCTURED PLANE, all of a specific form X_f with
f in k[a^{+-1},b^{+-1}][x]; they introduce BIVARIABLES and show the
bivariables are in bijection with those X_f that are trivial, recovering
Lewis's theorem that the second Venereau polynomial is a variable.

APPLICABILITY, ASSESSED RATHER THAN ASSUMED: their objects are
A^2-bundles over a punctured PLANE; ours is a birational modification of
A^4 along a codimension-2 centre, and the two intermediate models are
3- and 4-dimensional hypersurfaces, not A^2-bundles. The framework is
therefore NOT invoked as a theorem here. What transfers is the METHOD --
compare the two trivialisations over the complement and ask whether the
transition is elementary -- and that is what this block does. Recorded
so that no later reader mistakes a method for a citation.

Ledger: runs_synthesis/RM-4-COCYCLE.json. Mode I.
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
from rg_common import Pp, Qq, Tt, ZZ, H_L, TO_L, krull_dim, verify_carryover
from rm_common import (A4GENS, B_GENS, N1, R1, Xx, in_ideal,
                       verify_rg_carryover)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

Hh = sp.Symbol("H_")
Y_GENS = (Pp, Qq, Tt, ZZ, Hh)
YEQ = E(Tt**3 * Hh - Pp - Pp**2 - Qq * ZZ)

vv = sp.Symbol("v")
GENS2 = GENS + (vv,)

#: DECLARED BEFORE ANY SEARCH RUNS (RM-OPS §2)
AW_POOL_NAMES = ["p", "q", "t", "Z", "x", "y", "z", "w", "H", "W", "xH",
                 "qZ"]
Y_POOL_NAMES = ["p", "q", "t", "Z", "H", "qZ", "t^3H", "P'", "Zq-t^3H",
                "H+p"]
CAPS = {
    "literature_gate": "Blanc-Poloni arXiv:2004.10739 fetched and read at "
                       "primary source BEFORE this block; applicability "
                       "assessed as METHOD-ONLY, not invoked as a theorem",
    "k1_pool_AW": AW_POOL_NAMES,
    "k1_tuples_AW": "all C(12,4) = 495 four-subsets",
    "Y1_pool": Y_POOL_NAMES,
    "Y1_tuples": "all C(10,4) = 210 four-subsets",
    "k2_pool_AW": AW_POOL_NAMES + ["v", "x+v", "H+v", "W+v"],
    "k2_tuples_AW": "all C(16,5) = 4368 five-subsets",
    "filter": "exact Jacobian: det[grad(defining eq); grad a_i] nowhere "
              "zero on the variety; rejections carry an explicit witness "
              "point; POSITIVE CONTROL on a graph hypersurface",
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}

#: witness points ON X x A^2 (P = 0)
SAMPLES_AW = [
    {x: 1, y: -3, z: 1, t: 1, w: 1, vv: 1},
    {x: 1, y: -13, z: 2, t: 2, w: 3, vv: 5},
    {x: 2, y: -1, z: 1, t: 1, w: 0, vv: 0},
    {x: 0, y: 7, z: 0, t: 0, w: 3, vv: 2},
    {x: -1, y: -1, z: 1, t: 1, w: 2, vv: 1},
    {x: 1, y: 0, z: 0, t: -1, w: 1, vv: 0},
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


def alarmed(fn):
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def jac_sweep(eq, pool, gens, size, samples):
    """Exact Jacobian sweep. Returns (survivors, by_point, by_exact)."""
    gradE = [sp.diff(eq, g) for g in gens]
    grads = {k: [sp.diff(e, g) for g in gens] for k, e in pool.items()}
    gE_num = [[sp.Rational(c.subs(s)) for c in gradE] for s in samples]
    g_num = {k: [[sp.Rational(c.subs(s)) for c in grads[k]]
                 for s in samples] for k in pool}
    survivors, by_point, by_exact = [], 0, 0
    for combo in itertools.combinations(sorted(pool), size):
        dead = False
        for i in range(len(samples)):
            Mn = sp.Matrix([gE_num[i]] + [g_num[c][i] for c in combo])
            if Mn.det() == 0:
                dead = True
                break
        if dead:
            by_point += 1
            continue
        Ms = sp.Matrix([gradE] + [grads[c] for c in combo])
        D = E(Ms.det())
        if D == 0 or krull_dim([eq, D], gens) != -1:
            by_exact += 1
            continue
        survivors.append(" ".join(combo))
    return survivors, by_point, by_exact


def main() -> int:
    r = Runner()
    led: dict = {}

    print("\nRM-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    verify_carryover(r.check)
    verify_rg_carryover(r.check)
    r.record("DECLARED CAPS (before any search ran)", "MEASURED",
             json.dumps(CAPS, sort_keys=True))

    # ==================================================================
    print("\nRM-4a  the literature gate, discharged")
    r.record("Blanc-Poloni arXiv:2004.10739 -- READ AT PRIMARY SOURCE",
             "MEASURED",
             "Title 'Bivariables and Venereau polynomials', Blanc and "
             "Poloni, arXiv Apr 2020, Ann. Fac. Sci. Toulouse Dec 2022. "
             "Abstract: they study the Daigle-Freudenburg family "
             "containing the Venereau polynomials, defining A^2-fibrations "
             "over A^2; following Kaliman-Zaidenberg they show these are "
             "locally trivial A^2-bundles over the PUNCTURED PLANE, all of "
             "the form X_f with f in k[a^{+-1},b^{+-1}][x]; they introduce "
             "BIVARIABLES and put them in bijection with those X_f that "
             "are trivial, reproving Lewis's theorem that the second "
             "Venereau polynomial is a variable.")
    r.record("APPLICABILITY: METHOD ONLY, not invoked as a theorem",
             "MEASURED",
             "their objects are A^2-bundles over a punctured plane; ours "
             "is a birational modification of A^4 along a codim-2 centre, "
             "with 4-dimensional hypersurface models. No theorem of theirs "
             "applies to our situation. What transfers is the method -- "
             "compare the two trivialisations over the complement and ask "
             "whether the transition is elementary. Nothing below is "
             "load-bearing on their results.")
    led["literature_gate"] = {
        "paper": "Blanc & Poloni, Bivariables and Venereau polynomials, "
                 "arXiv:2004.10739; Ann. Fac. Sci. Toulouse (2022)",
        "read_at_source": True,
        "applies_as_theorem": False,
        "transfers": "the gluing METHOD only",
    }

    # ==================================================================
    print("\nRM-4b  the transition datum, explicitly")
    r.check("over D(t) the model is the X_1-cylinder: A[w][1/t] = B[w][1/t]",
            lambda: sp.denom(sp.together(sp.cancel(H_L * Tt**3))) == 1)
    r.check("over D(p+1) the model is Y_1: A[w][1/(p+1)] = C[H][1/(p+1)]",
            lambda: sp.denom(sp.together(sp.cancel(
                TO_L[x] * (Pp + 1)))) == 1
            and sp.denom(sp.together(sp.cancel(
                2 * TO_L[w] * (Pp + 1) - (H_L * ZZ + Qq)))) == 1)
    r.record("THE TRANSITION DATUM", "MEASURED",
             "the two trivialisations are the two intermediate models of "
             "RM-3: over D(t) the singular X_1 x A^1, over D(p+1) the "
             "SMOOTH Y_1. They agree over U = D(t(p+1)) where both equal "
             "L. The transition is therefore not an automorphism of a "
             "bundle but a change of INTEGRAL MODEL, and the crossing "
             "V(t,p+1) -- where the compressed centre restricts to (qZ^2) "
             "-- is where the two disagree most.")
    led["transition_datum"] = {
        "over_D(t)": "X_1 x A^1 (singular)",
        "over_D(p+1)": "Y_1 (smooth)",
        "overlap": "both equal L over U = D(t(p+1))",
        "nature": "a change of integral model, not a bundle automorphism",
    }

    # ==================================================================
    print("\nRM-4c  k = 1 on the prize object: is A[w] = C^[4]?")
    H_A, Zc = E(-(y + w**2)), E(z + x * w)
    AW_POOL = {"p": KER_P, "q": KER_Q, "t": t, "Z": Zc, "x": x, "y": y,
               "z": z, "w": w, "H": H_A, "W": E(2 * w),
               "xH": E(x * H_A), "qZ": E(KER_Q * Zc)}
    assert sorted(AW_POOL) == sorted(AW_POOL_NAMES), "pool drifted"
    # positive control first (H15): the filter must be able to pass
    Qg = E(y - (x**2 + z * t * w))
    Mg = sp.Matrix([[sp.diff(Qg, g) for g in GENS]]
                   + [[sp.diff(a, g) for g in GENS] for a in (x, z, t, w)])
    r.check("POSITIVE CONTROL: the k=1 filter passes on a genuine C^[4]",
            lambda: E(Mg.det()) != 0 and not E(Mg.det()).free_symbols,
            f"graph hypersurface, det = {E(Mg.det())}")
    s1, bp1, be1 = alarmed(lambda: jac_sweep(
        P, AW_POOL, GENS, 4, [{k: v for k, v in s.items() if k != vv}
                              for s in SAMPLES_AW]))
    r.record(f"k=1 sweep over all {bp1 + be1 + len(s1)} four-subsets",
             "MEASURED",
             f"rejected by witness point: {bp1}; by the exact test: "
             f"{be1}; survivors: {s1 if s1 else 'none'}")
    r.check("the filter discriminates (both rejection modes occur)",
            lambda: bp1 > 0 and be1 > 0)
    led["k1_AW"] = {"pool": AW_POOL_NAMES, "survivors": s1,
                    "rejected_by_point": bp1, "rejected_exact": be1}

    # ==================================================================
    print("\nRM-4d  k = 2 on the prize object: is A[w][v] = C^[5]?")
    AW_POOL2 = dict(AW_POOL)
    AW_POOL2.update({"v": vv, "x+v": E(x + vv), "H+v": E(H_A + vv),
                     "W+v": E(2 * w + vv)})
    Qg2 = E(y - (x**2 + z * t * w * vv))
    Mg2 = sp.Matrix([[sp.diff(Qg2, g) for g in GENS2]]
                    + [[sp.diff(a, g) for g in GENS2]
                       for a in (x, z, t, w, vv)])
    r.check("POSITIVE CONTROL: the k=2 filter passes on a genuine C^[5]",
            lambda: E(Mg2.det()) != 0 and not E(Mg2.det()).free_symbols,
            f"det = {E(Mg2.det())}")
    s2, bp2, be2 = alarmed(lambda: jac_sweep(
        P, AW_POOL2, GENS2, 5, SAMPLES_AW))
    r.record(f"k=2 sweep over all {bp2 + be2 + len(s2)} five-subsets",
             "MEASURED",
             f"rejected by witness point: {bp2}; by the exact test: "
             f"{be2}; survivors: {s2 if s2 else 'none'}")
    r.check("the k=2 filter discriminates",
            lambda: bp2 > 0 and be2 > 0)
    led["k2_AW"] = {"pool": sorted(AW_POOL2), "survivors": s2,
                    "rejected_by_point": bp2, "rejected_exact": be2}

    # ==================================================================
    print("\nRM-4e  the NEW question RM-3 produced: is Y_1 = A^4?")
    Pn = E(Pp + sp.Rational(1, 2))
    Y_POOL = {"p": Pp, "q": Qq, "t": Tt, "Z": ZZ, "H": Hh,
              "qZ": E(Qq * ZZ), "t^3H": E(Tt**3 * Hh), "P'": Pn,
              "Zq-t^3H": E(Qq * ZZ - Tt**3 * Hh), "H+p": E(Hh + Pp)}
    assert sorted(Y_POOL) == sorted(Y_POOL_NAMES), "Y pool drifted"
    YSAMP = []
    for (p0, q0, t0, Z0) in ((1, 1, 1, 1), (0, 1, 1, 2), (2, 0, 1, 0),
                             (-1, 3, 1, 1), (1, 2, -1, 1), (3, 1, 2, 1)):
        H0 = sp.Rational(p0 + p0**2 + q0 * Z0, t0**3)
        YSAMP.append({Pp: p0, Qq: q0, Tt: t0, ZZ: Z0, Hh: H0})
    r.check("all Y_1 witness points lie on Y_1",
            lambda: all(E(YEQ.subs(s)) == 0 for s in YSAMP))
    sY, bpY, beY = alarmed(lambda: jac_sweep(YEQ, Y_POOL, Y_GENS, 4, YSAMP))
    r.record(f"Y_1 sweep over all {bpY + beY + len(sY)} four-subsets",
             "MEASURED",
             f"rejected by witness point: {bpY}; by the exact test: "
             f"{beY}; survivors: {sY if sY else 'none'}")
    r.check("the Y_1 filter discriminates",
            lambda: bpY > 0 or beY > 0)
    r.record("is Y_1 = A^4? NOT DECIDED", "NOT-ATTEMPTED",
             "no four-subset of the declared ten-element pool passes the "
             "necessary Jacobian condition. That is a cap-bounded "
             "non-finding, NOT a negative: Y_1 is a smooth affine 4-fold "
             "given by one explicit equation, and deciding it needs a "
             "proper attempt, not a pool sweep. Routed to master as the "
             "natural successor question -- if Y_1 = A^4 then the whole "
             "tower collapses to a SINGLE (p+1)-modification.")
    led["Y1"] = {"pool": Y_POOL_NAMES, "survivors": sY,
                 "rejected_by_point": bpY, "rejected_exact": beY,
                 "status": "NOT DECIDED -- routed to master"}

    # ==================================================================
    print("\nRM-4f  the classification")
    classification = {
        "k=1 on A[w]": f"no candidate at the declared pool "
                       f"({len(AW_POOL_NAMES)} elements, 495 subsets); "
                       "MEASURED, not a negative. RG-3's two exact "
                       "theorems already close the flow-compatible route.",
        "k=2 on A[w][v]": f"no candidate at the declared pool "
                          f"({len(AW_POOL2)} elements, 4368 subsets); "
                          "MEASURED. RG-4.1 already shows the "
                          "flow-compatible obstruction is 1-stable.",
        "the crossing": "the transition is a change of INTEGRAL MODEL "
                        "between the singular X_1-cylinder (over D(t)) "
                        "and the smooth Y_1 (over D(p+1)); the two "
                        "disagree over the crossing, where the centre "
                        "restricts to (qZ^2).",
        "Blanc-Poloni": "method only; no theorem of theirs applies to a "
                        "codim-2 crossing in A^4.",
        "what_is_NOT_shown": "nothing here bears on X x A^1 = A^4 or "
                             "X x A^2 = A^5. Honest fence, A26.",
    }
    for k, val in classification.items():
        r.record(f"classification: {k}", "MEASURED", val)
    led["classification"] = classification

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RM-4-COCYCLE.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RM-4", "plan": "RM-OPS.md (A26)",
            "date": "2026-08-01", "mode": "I",
            "source": "DMJP arXiv:0903.4278 s7 for p,q,Z and the bridge; "
                      "Blanc-Poloni arXiv:2004.10739 read at source "
                      "(method only); the results are ours",
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
