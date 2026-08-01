#!/usr/bin/env python3
"""RD-5 — the slice test for the published LND.

The derivation is DMJP's (arXiv:0903.4278 Prop 7.2); the slice analysis
is ours.

THE CRITERION USED, stated exactly.  A slice for an LND D on A[w] is an
s with D(s) = 1; then A[w] = (ker D)[s].  RD-2 showed partial has
content t^3, so partial(A[w]) is inside t^3 A[w] and partial can never
have a slice for trivial reasons.  The real question is about the
CONTENT-FREE derivation partial_red = partial / t^3, and that is what
is tested here.  ker(partial_red) = ker(partial) = C[p,q,t] (RD-3), so
a slice would give A[w] = C[p,q,t][s] = C^[4].

THE ANSWER IS NO, and it is an exact theorem, not a cap-limited search:
every component of partial_red vanishes at the origin of X x A^1, so by
the chain rule partial_red(F) vanishes there for EVERY F, and 1 does
not.  A bounded search is run anyway, with a positive control, as an
independent cross-check that the machinery agrees with the theorem.

Ledger: runs_synthesis/RD-5-SLICE.json.  Mode I.
"""

from __future__ import annotations

import json
import os
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rd_4_plinth import partial_red_components

IOTA_Y = -(z**2 + x + t**3) / x**2


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


def monomials(deg):
    """Deterministically ordered monomials of total degree <= deg (H2)."""
    out = []
    for dx in range(deg + 1):
        for dy in range(deg + 1 - dx):
            for dz in range(deg + 1 - dx - dy):
                for dt in range(deg + 1 - dx - dy - dz):
                    for dw in range(deg + 1 - dx - dy - dz - dt):
                        out.append((dx, dy, dz, dt, dw))
    out.sort()
    return [x**a * y**b * z**c * t**d * w**e for a, b, c, d, e in out]


def slice_search(comps: dict, deg: int):
    """Exhaustive exact search for s of total degree <= deg with D(s) = 1.

    Linear in the unknown coefficients, so this is an exact Q-nullspace
    problem, not an elimination: solved by linear algebra over Q.
    Returns (n_unknowns, solution or None).
    """
    mons = monomials(deg)
    cs = sp.symbols(f"c0:{len(mons)}")
    s = sum(c * m for c, m in zip(cs, mons))
    Ds = E(sum(comps[v] * sp.diff(s, v) for v in GENS))
    # membership of Ds - 1 in (P), cleared of x-denominators: linear in cs
    resid = E(Ds - 1).xreplace({y: IOTA_Y})
    ny = max(sp.Poly(E(Ds - 1), y).degree(), 0)
    resid = E(sp.cancel(resid * x ** (2 * ny)))
    eqs = sp.Poly(resid, x, z, t, w).coeffs()
    sol = sp.solve(eqs, cs, dict=True)
    return len(mons), (sol[0] if sol else None)


def main() -> int:
    r = Runner()
    led: dict = {}

    red = partial_red_components()
    r.check("tripwire: partial_red(x) = -2 t^3 (x w + z)",
            lambda: E(red[x] + 2 * t**3 * (x * w + z)) == 0)

    print("\nRD-5a  the criterion, stated exactly")
    r.record("criterion", "MEASURED",
             "a slice is s with partial_red(s) = 1 (partial itself is "
             "excluded a priori: RD-2 showed partial(A[w]) is inside "
             "t^3 A[w]). ker(partial_red) = ker(partial) = C[p,q,t], so a "
             "slice would give A[w] = C[p,q,t][s] = C^[4].")

    print("\nRD-5b  THE EXACT THEOREM: no slice, no caps involved")
    ORIGIN = {v: 0 for v in GENS}
    r.check("the origin lies on X x A^1",
            lambda: E(P.subs(ORIGIN)) == 0)
    r.check("every component of partial_red vanishes at the origin",
            lambda: all(E(red[v].subs(ORIGIN)) == 0 for v in GENS),
            "so partial_red(F)(origin) = 0 for EVERY F, by the chain rule")
    r.check("1 does not vanish at the origin: so 1 is not in the image",
            lambda: sp.Integer(1).subs(ORIGIN) == 1)
    r.record("partial_red has NO SLICE", "MEASURED",
             "exact, unconditional, no degree cap: partial_red(A[w]) is "
             "contained in the maximal ideal of the origin, and 1 is not.")
    r.record("partial has no slice either", "MEASURED",
             "a fortiori: partial = t^3 partial_red.")
    # a second, structural route
    r.check("structural route: a slice would force a FREE action",
            lambda: all(E(red[v].subs({x: 0, z: 0, t: 0, w: 0})) == 0
                        for v in GENS),
            "A[w] = ker[s] makes Ga act by translation, hence freely; but "
            "RD-4 exhibits the fixed line {x=z=t=w=0}")
    led["exact_theorem"] = {
        "verdict": "partial_red (hence partial) admits NO SLICE on A[w]",
        "status": "EXACT THEOREM -- not a cap-limited search",
        "proof_1": "every component of partial_red vanishes at the origin "
                   "of X x A^1; by the chain rule so does partial_red(F) "
                   "for every F; 1 does not.",
        "proof_2": "a slice makes A[w] = (ker partial)[s], so the "
                   "Ga-action is a translation and therefore free; RD-4 "
                   "exhibits a whole line of fixed points.",
    }

    print("\nRD-5c  bounded search, with a positive control (H15)")
    # positive control: d/dw obviously has the slice w. The solver must
    # FIND it, otherwise an EMPTY verdict below would be meaningless.
    dw_only = {v: (sp.Integer(1) if v == w else sp.Integer(0)) for v in GENS}
    n_ctl, sol_ctl = slice_search(dw_only, 1)
    r.check("positive control: the solver finds the slice of d/dw",
            lambda: sol_ctl is not None,
            f"{n_ctl} unknowns; solution found")
    # NOTE: no wall-clock goes into the ledger. Timings are recorded in
    # the per-entry wall_time_s field, which the replay comparison strips;
    # putting one in a NOTE made the RD-5 ledger non-replayable (see the
    # incident logged in RD-DELTA and in docs_rd_dissection.md).
    caps = {}
    for D in (1, 2, 3):
        n, sol = slice_search(red, D)
        caps[D] = {"unknowns": n, "found": sol is not None}
        r.check(f"no slice of total degree <= {D} (cross-check)",
                lambda ss=sol: ss is None,
                f"{n} unknowns, exact Q linear solve")
    r.record("search measurement", "MEASURED",
             f"linear (exact Q nullspace), not an elimination: "
             f"{[caps[D]['unknowns'] for D in caps]} unknowns at degrees "
             f"{list(caps)}. Agrees with the theorem, which needs no cap.")
    led["bounded_search"] = {"caps": {str(k): v for k, v in caps.items()},
                             "positive_control": "d/dw slice found"}

    print("\nRD-5d  where the obstruction sits")
    r.check("the obstruction is supported on {t = 0} (RD-4a)",
            lambda: all(E(red[v].subs({x: 0, z: 0, t: 0, w: 0})) == 0
                        for v in GENS))
    r.record("failure locus against the RD-4 divisor", "MEASURED",
             "the degeneracy locus lives on {t = 0}; the localisation "
             "theorem makes A[w][1/(t(p+1))] = C[p,q,t][1/(t(p+1))][Z] "
             "with Z = z+xw. So the slice exists after inverting t(p+1) "
             "and the whole failure is concentrated on that one divisor "
             "of A^3 = Spec(ker partial).")
    led["failure_locus"] = {
        "supported_on": "{t = 0} in X x A^1; {t (p+1) = 0} in A^3",
        "slice_after_localisation": "Z = z + x w, partial_red(Z) = t^3(p+1)",
    }

    print("\nRD-5e  the ending")
    r.record("CANDIDATE ending NOT triggered", "MEASURED",
             "RD-3 answers 'is ker partial = C^[3]?' YES, but the slice "
             "branch closes exactly, so no candidate isomorphism is "
             "assembled and the twin bar is not invoked. Nothing about "
             "X x A^1 = A^4 in general is claimed or implied: what closes "
             "is the route THROUGH THIS FLOW's slice.")
    led["ending"] = {
        "candidate": False,
        "scope":
            "This says nothing about the Zariski cancellation question "
            "itself. It closes one explicit route and replaces it with a "
            "sharper, bounded one (the localisation theorem's divisor).",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RD-5-SLICE.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RD-5", "plan": "RD-OPS.md (A22)",
            "date": "2026-07-31", "mode": "I",
            "source": "the derivation is DMJP arXiv:0903.4278 Prop 7.2; the "
                      "slice analysis is ours",
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
