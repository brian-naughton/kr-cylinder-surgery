#!/usr/bin/env python3
"""RD-6 — brackets of the published LND with DMJP's Cor-7.3 derivations.

partial (Prop 7.2), partial_1 = 2z d/dy - x^2 d/dz and
partial_2 = 3t^2 d/dy - x^2 d/dt (Cor 7.3) are DMJP's; d/dw is the
obvious one.  The bracket computation and what is read off it is ours.

Hunted, as RD-OPS §4 RD-6 asks: commuting pairs, a triangular flag, a
generically free Ga^r structure, candidate coordinates as successive
kernels and slices.

Ledger: runs_synthesis/RD-6-BRACKETS.json.  Mode I.
"""

from __future__ import annotations

import json
import os
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, P, iota, x, y, z, t, w
from rd_4_plinth import partial_red_components


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "MEASURED", "ERRATUM", "NOT-ATTEMPTED")

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
        print(f"  {name:<48s} {outcome:<11s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<48s} {outcome:<11s}         [{note}]", flush=True)

    @property
    def ok(self):
        return all(o != "FAIL" for _, o, _ in self.results)

    @property
    def tally(self):
        return {k: sum(1 for _, o, _ in self.results if o == k)
                for k in self.CLASSES
                if any(o == k for _, o, _ in self.results)}


def apply_der(D, f):
    return E(sum(D[v] * sp.diff(f, v) for v in GENS))


def bracket(D, Ed):
    return {v: E(apply_der(D, Ed[v]) - apply_der(Ed, D[v])) for v in GENS}


def is_zero_in_AW(D):
    return all(is_in_ideal(E(D[v]), P, y) for v in GENS)


def nilpotency(D, cap=20):
    """Depths of D on the generators, in the faithful localisation model."""
    cx, cz, cw = iota(D[x]), iota(D[z]), iota(D[w])
    ct = iota(D[t])
    iy = -(z**2 + x + t**3) / x**2

    def step(f):
        return sp.cancel(E(cx * sp.diff(f, x) + cz * sp.diff(f, z)
                           + ct * sp.diff(f, t) + cw * sp.diff(f, w)))
    out = {}
    for nm, e0 in (("x", x), ("y", iy), ("z", z), ("t", t), ("w", w)):
        e, k = e0, 0
        while sp.cancel(E(e)) != 0 and k < cap:
            e = step(e)
            k += 1
        out[nm] = k
    return out


def main() -> int:
    r = Runner()
    led: dict = {}

    red = partial_red_components()
    r.check("tripwire: partial_red(x) = -2 t^3 (x w + z)",
            lambda: E(red[x] + 2 * t**3 * (x * w + z)) == 0)

    D1 = {x: sp.Integer(0), y: 2 * z, z: -x**2, t: sp.Integer(0),
          w: sp.Integer(0)}
    D2 = {x: sp.Integer(0), y: 3 * t**2, z: sp.Integer(0), t: -x**2,
          w: sp.Integer(0)}
    DW = {x: sp.Integer(0), y: sp.Integer(0), z: sp.Integer(0),
          t: sp.Integer(0), w: sp.Integer(1)}
    NAMED = {"partial_red": red, "partial_1": D1, "partial_2": D2,
             "d/dw": DW}

    print("\nRD-6a  the Cor-7.3 derivations are LNDs on A[w]")
    for nm, D in (("partial_1", D1), ("partial_2", D2), ("d/dw", DW)):
        r.check(f"{nm} annihilates P exactly (so it preserves (P))",
                lambda DD=D: E(apply_der(DD, P)) == 0)
        dep = nilpotency(D)
        r.check(f"{nm} is locally nilpotent", lambda d=dep: max(d.values()) < 20,
                f"depths {dep}")

    print("\nRD-6b  all six brackets")
    names = list(NAMED)
    table, zeros = {}, []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            Br = bracket(NAMED[a], NAMED[b])
            zero = is_zero_in_AW(Br)
            table[f"[{a}, {b}]"] = ("0" if zero else
                                    {str(v): str(sp.factor(Br[v]))
                                     for v in GENS if E(Br[v]) != 0})
            if zero:
                zeros.append(f"[{a}, {b}]")
            r.record(f"[{a}, {b}] " + ("= 0" if zero else "!= 0"),
                     "MEASURED",
                     "commuting pair" if zero else "does not commute")
    # make the above non-vacuous: assert exactly the expected commuting set
    expected = {"[partial_1, partial_2]", "[partial_1, d/dw]",
                "[partial_2, d/dw]"}
    r.check("EXACTLY the three Cor-7.3 pairs commute",
            lambda: set(zeros) == expected, f"commuting: {sorted(zeros)}")
    led["brackets"] = table
    led["commuting_pairs"] = sorted(zeros)

    print("\nRD-6c  the commuting triple, and why it cannot give coordinates")
    r.record("{partial_1, partial_2, d/dw} is a commuting triple", "MEASURED",
             "so A[w] carries a Ga^3-action -- but its invariant ring is "
             "C[x] (DMJP Cor 7.3, repaired below), and our own Lemma L1 "
             "(M1-PREREG, frozen d423397) says x can NEVER be a coordinate "
             "of A^4, because A[w]/(x) is the NON-NORMAL cusp cylinder. So "
             "this flag is structurally dead as a coordinate system -- a "
             "banked lemma of ours closing a published structure.")
    # the generic frame
    M = sp.Matrix([[iota(NAMED[n][v]) for v in (x, z, t, w)] for n in names])
    det = sp.cancel(M.det())
    r.check("the four derivations form a GENERIC FRAME on X x A^1",
            lambda: det != 0,
            f"4x4 determinant in the model is nonzero")
    r.check("H15 boundary: dropping partial_red destroys the frame",
            lambda: sp.Matrix([[iota(NAMED[n][v]) for v in (x, z, t, w)]
                               for n in names[1:]]).rank() == 3,
            "the Cor-7.3 triple alone spans only 3 directions -- x is fixed")
    led["frame"] = {
        "det_nonzero": True,
        "reading": "partial_red is exactly the direction the Cor-7.3 triple "
                   "cannot reach: the triple fixes x, the frame needs it. "
                   "That is Cor 7.3's ML argument, seen as linear algebra.",
    }

    print("\nRD-6d  Cor 7.3 as printed, and its repair")
    r.check("w lies in Ker(partial_1) and in Ker(partial_2)",
            lambda: E(apply_der(D1, w)) == 0 and E(apply_der(D2, w)) == 0)
    r.record("ERRATUM (venial) in DMJP Cor 7.3", "ERRATUM",
             "the text says 'Ker(partial_1) cap Ker(partial_2) = C[x]' for "
             "the derivations ON A[w]; but w lies in both kernels, so on "
             "A[w] the intersection contains C[x][w]. The classical "
             "statement is on A (Makar-Limanov); on A[w] one must adjoin "
             "d/dw, which the paper's own framework supplies. Cor 7.3's "
             "CONCLUSION is unaffected -- see the repair below.")
    r.check("repair: x is killed by partial_1, partial_2 and d/dw",
            lambda: all(E(apply_der(D, x)) == 0 for D in (D1, D2, DW)))
    r.check("repair: partial_red(x) != 0, so ML(A[w]) = C follows",
            lambda: not is_in_ideal(E(red[x]), P, y),
            "any f(x) in ML would need partial_red(f) = f'(x) partial_red(x) "
            "= 0 in a domain, forcing f constant")
    led["cor_7_3"] = {
        "erratum": "as printed the intersection is stated on A[w] but omits "
                   "w; adjoining d/dw repairs it and the conclusion "
                   "ML(A[w]) = C stands.",
    }

    print("\nRD-6e  does partial_red interact with the RD-3 kernel?")
    r.check("partial_1(p) and partial_2(p) are not both zero",
            lambda: not (is_in_ideal(E(apply_der(D1, KER_P)), P, y)
                         and is_in_ideal(E(apply_der(D2, KER_P)), P, y)),
            "the Cor-7.3 derivations do NOT preserve ker partial pointwise")
    r.record("no triangular flag through partial_red was found", "MEASURED",
             "partial_red commutes with none of partial_1, partial_2, d/dw, "
             "and the three that do commute have the structurally dead "
             "invariant ring C[x] (Lemma L1). No new candidate coordinate "
             "system came out of the bracket structure. Stated as a "
             "measurement of what these six brackets show, not as a claim "
             "that no flag exists.")
    led["verdict"] = (
        "The bracket structure yields no new coordinate route. The only "
        "commuting sub-algebra is the Cor-7.3 triple, whose invariants are "
        "C[x], and Lemma L1 rules x out as a coordinate. partial_red is "
        "precisely the direction that triple cannot reach.")

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RD-6-BRACKETS.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RD-6", "plan": "RD-OPS.md (A22)",
            "date": "2026-07-31", "mode": "I",
            "source": "partial, partial_1, partial_2 are DMJP "
                      "arXiv:0903.4278 s7; the bracket analysis is ours",
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
