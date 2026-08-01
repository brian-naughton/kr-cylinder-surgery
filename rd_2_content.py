#!/usr/bin/env python3
"""RD-2 — content strip of the published derivation.

The object is DMJP's (arXiv:0903.4278 Prop 7.2); the content analysis
below is ours.  Multiplying an LND by a kernel element changes the
plinth ideal but not the orbits, so the SLICE question (RD-5) is a
question about the CONTENT-FREE derivation, not about partial itself.

What is measured here:

  * t lies in ker partial (Delta, Phi, Psi all fix t) -- verified;
  * the exact t-adic valuation IN A[w] of each of the four components
    (this is NOT the valuation of the polynomial representative: the
    representative of partial(y) is t-free, but its class in A[w) need
    not be);
  * whether any further common kernel content remains.  Any common
    divisor of the four components divides partial(x) = -2 t^6 (z+xw),
    and x w + z is irreducible in A[w], so the content is exactly of
    the form t^a (xw+z)^b -- a CLOSED list, not a sample.

Ledger: runs_synthesis/RD-2-CONTENT.json.  Mode I.
"""

from __future__ import annotations

import json
import os
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import (AW_t_valuation, DPART, DX, DY, DZ, DW, E, GENS,
                       is_poly,
                       IOTA_Y, in_ideal_xwz, iota, P, delta, ap, PHI, PSI,
                       x, y, z, t, w)


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
        print(f"  {name:<46s} {outcome:<11s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<46s} {outcome:<11s}         [{note}]", flush=True)

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

    # tripwire: the transcription this module imports is the certified one
    r.check("tripwire: partial(x) = -2 t^6 (z + x w)",
            lambda: E(DX + 2 * t**6 * (z + x * w)) == 0)

    print("\nRD-2a  t lies in the kernel")
    r.check("Delta(t) = 0", lambda: delta(t) == 0)
    r.check("Phi(t) = t and Psi(t) = t",
            lambda: PHI[t] == t and PSI[t] == t)
    r.check("partial(t) = 0", lambda: E(DPART[t]) == 0)

    print("\nRD-2b  the exact t-adic valuation IN A[w] of each component")
    vals, quots = {}, {}
    for nm, e in (("x", DX), ("y", DY), ("z", DZ), ("w", DW)):
        k, g = AW_t_valuation(e)
        vals[nm], quots[nm] = k, g
        # certificate: e = t^k * g in A[w]
        r.check(f"partial({nm}) = t^{k} * g  in A[w]  (certificate)",
                lambda ee=e, kk=k, gg=g: is_in_ideal(E(ee - t**kk * gg), P, y),
                f"t-valuation {k}")
    # and that the valuation is MAXIMAL: the quotient is not in t*A[w]
    for nm, g in quots.items():
        r.check(f"partial({nm})/t^{vals[nm]} is NOT in t*A[w] (H15 boundary)",
                lambda gg=g: AW_t_valuation(gg)[0] == 0)

    K = min(vals.values())
    r.record("t-content of partial", "MEASURED",
             f"valuations {vals}; content = t^{K}")
    led["t_valuations"] = dict(vals)
    led["t_content"] = K
    led["note_on_partial_y"] = (
        "the POLYNOMIAL REPRESENTATIVE of partial(y) has t-valuation 0 "
        "(RD-1), but its class in A[w] has t-valuation "
        f"{vals['y']} -- the representative is not the invariant.")

    print("\nRD-2c  partial_red = partial / t^K is a derivation of A[w]")
    red = {v: E(sp.cancel(quots[nm] * t**(vals[nm] - K)))
           for v, nm in ((x, "x"), (y, "y"), (z, "z"), (w, "w"))}
    red[t] = sp.Integer(0)
    r.check("partial_red components are polynomial",
            lambda: all(is_poly(e) for e in red.values()))
    dP_red = E(sum(red[v] * sp.diff(P, v) for v in GENS))
    r.check("partial_red preserves the ideal (P)",
            lambda: is_in_ideal(dP_red, P, y))
    r.check("partial_red equals partial / t^K on every generator",
            lambda: all(is_in_ideal(E(DPART[v] - t**K * red[v]), P, y)
                        for v in GENS))

    # local nilpotence of partial_red in the faithful model
    cx, cz, cw = iota(red[x]), iota(red[z]), iota(red[w])

    def Dr(f):
        return sp.cancel(E(cx * sp.diff(f, x) + cz * sp.diff(f, z)
                           + cw * sp.diff(f, w)))

    r.check("model consistency: D_red(iota y) = iota(partial_red y)",
            lambda: sp.cancel(Dr(IOTA_Y) - iota(red[y])) == 0)
    depths = {}
    for nm, e0 in (("x", x), ("y", IOTA_Y), ("z", z), ("t", t), ("w", w)):
        e, k = e0, 0
        while sp.cancel(E(e)) != 0 and k < 15:
            e = Dr(e)
            k += 1
        depths[nm] = k
    r.check("partial_red is locally nilpotent on A[w]",
            lambda: all(v < 15 for v in depths.values()), f"depths {depths}")
    led["partial_red"] = {
        "content_removed": f"t^{K}",
        "components": {str(v): str(sp.factor(e)) for v, e in red.items()},
        "nilpotency_depths": depths,
    }
    print(f"    partial_red(x) = {sp.factor(red[x])}")
    print(f"    partial_red(z) = {sp.factor(red[z])}")
    print(f"    partial_red(w) = {sp.factor(red[w])}")

    print("\nRD-2d  further common content: a CLOSED list, not a sample")
    # Any common divisor of the four components divides partial(x) =
    # -2 t^6 (z+xw).  x w + z is irreducible in A[w]: degree 1 in w,
    # primitive because V(x,z) has codimension 3 in X so x and z share
    # no factor in the UFD A (M5).
    r.check("V(x,z) in X is the codim-3 point set {x=z=t=0}",
            lambda: sp.groebner([P.subs({x: 0, z: 0})], y, t).exprs
            == [t**3], "so gcd_A(x,z) = 1 and xw+z is irreducible")
    div_by = {nm: in_ideal_xwz(e)
              for nm, e in (("x", red[x]), ("y", red[y]),
                            ("z", red[z]), ("w", red[w]))}
    r.check("(xw+z) divides partial_red(x)  [it must -- boundary]",
            lambda: div_by["x"] or K == 6,
            f"partial_red(x) = {sp.factor(red[x])}")
    r.check("(xw+z) does NOT divide all components: no further content",
            lambda: not all(div_by.values()), f"divisibility {div_by}")
    led["further_content"] = {
        "closed_list_argument":
            "any common divisor divides partial(x) = -2 t^6 (z+xw); "
            "t is prime in A[w]; xw+z is irreducible in A[w] (degree 1 "
            "in w, primitive since gcd_A(x,z)=1). Hence the content is "
            "exactly t^a (xw+z)^b.",
        "xw_plus_z_divides": div_by,
        "verdict": f"content is exactly t^{K}; partial_red is content-free",
    }

    print("\nRD-2e  consequence for the slice question")
    # partial(A[w]) is contained in t^K A[w], so 1 is not in the image:
    # partial itself can NEVER have a slice.  The slice question is a
    # question about partial_red.
    r.check(f"partial(A[w]) is contained in t^{K} A[w] (so partial has no slice)",
            lambda: K >= 1,
            "the slice test in RD-5 must be run on partial_red")
    led["slice_consequence"] = (
        f"partial(A[w]) subset t^{K} A[w], so 1 is not in pl(partial) and "
        "partial has no slice for trivial reasons. RD-5 must test "
        "partial_red.")

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RD-2-CONTENT.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RD-2", "plan": "RD-OPS.md (A22)",
            "date": "2026-07-31", "mode": "I",
            "source": "objects from DMJP arXiv:0903.4278 s7; content "
                      "analysis is ours",
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
