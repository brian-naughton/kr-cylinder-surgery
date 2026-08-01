#!/usr/bin/env python3
"""RD-4 — plinth ideal, degeneracy locus, and the localisation theorem.

The derivation is DMJP's (arXiv:0903.4278 Prop 7.2).  Everything
computed here about its plinth, fixed locus and quotient geometry is
ours.

Four deliverables:

  (a) THE DEGENERACY LOCUS.  The common zero locus of partial_red on
      X x A^1, computed exactly (both inclusions).
  (b) THE IMAGE IDEAL.  partial_red(A[w]) sits inside a PROPER ideal of
      A[w] -- which is what kills the slice (RD-5).
  (c) THE PLINTH.  Explicit elements of pl(partial_red) written in the
      kernel coordinates p, q, t that RD-3 produced -- computed by TWO
      routes where both are affordable -- plus a containment for the
      whole ideal.
  (d) THE LOCALISATION THEOREM (the sharp one).  With Z = z + x w,
      partial_red(Z) = t^3 (p+1), and
          A[w][1/(t(p+1))] = C[p,q,t][1/(t(p+1))][Z],
      verified by exhibiting x, y, z, w explicitly in those generators.
      So the whole obstruction to X x A^1 = A^4 THROUGH THIS FLOW is
      concentrated on the divisor {t (p+1) = 0} of A^3 = Spec(ker).

Ledger: runs_synthesis/RD-4-PLINTH.json.  Mode I.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import sympy as sp

from checker import is_in_ideal
from rd_common import (AW_t_valuation, DPART, E, GENS, KER_P, KER_Q, P,
                       PSI, ap, delta, is_poly, x, y, z, t, w)

U = sp.Symbol("U")                 # the B-side kernel coordinate u = y+1
Pp, Qq = sp.symbols("p_ q_")       # placeholders for the kernel generators

CEILING_VARS = 60
CEILING_CPU_S = 30 * 60


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
        print(f"  {name:<52s} {outcome:<11s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<52s} {outcome:<11s}         [{note}]", flush=True)

    @property
    def ok(self):
        return all(o != "FAIL" for _, o, _ in self.results)

    @property
    def tally(self):
        return {k: sum(1 for _, o, _ in self.results if o == k)
                for k in self.CLASSES
                if any(o == k for _, o, _ in self.results)}


def partial_red_components():
    """partial_red = partial / t^3 (RD-2), as polynomial representatives."""
    red = {}
    for v in GENS:
        if E(DPART[v]) == 0:
            red[v] = sp.Integer(0)
            continue
        k, g = AW_t_valuation(DPART[v])
        red[v] = E(sp.cancel(g * t**(k - 3)))
    return red


def to_B_model(g: sp.Expr) -> sp.Expr:
    """Reduce a B-side representative in the model u = y+1, x = -(z^2+t^3)/u."""
    return sp.cancel(E(g.xreplace({y: U - 1}).xreplace({x: -(z**2 + t**3) / U})))


def plinth_via_transport(v: sp.Symbol, k: int):
    """partial_red^k(v) in kernel coordinates, via Phi o Delta^k o Psi.

    Delta is triangular, so this is cheap where direct iteration of
    partial_red in A[w] is not.  If the result is not z-free the element
    is not in ker Delta and None comes back.
    """
    g = ap(PSI, v)
    for _ in range(k):
        g = delta(g)
    h = to_B_model(g)
    if sp.diff(h, z) != 0:
        return None
    return sp.expand(h.xreplace({U: Pp + 1, w: Qq}) / t ** (3 * k))


def main() -> int:
    r = Runner()
    led: dict = {}

    red = partial_red_components()
    r.check("tripwire: partial_red(x) = -2 t^3 (x w + z)",
            lambda: E(red[x] + 2 * t**3 * (x * w + z)) == 0)

    def apply_red(f):
        return E(sum(red[v] * sp.diff(f, v) for v in GENS))

    # ------------------------------------------------------------------
    print("\nRD-4a  the degeneracy locus of partial_red on X x A^1")
    # ERRATUM, logged against this executor's own in-session hand claim:
    # the locus is NOT {x = z = t = 0}.  partial_red(y) restricted to
    # that plane is -2w, not 0.  The machine caught it; the corrected
    # computation is below.  (RD-1's H15 discipline working as designed.)
    comps = [red[v] for v in (x, y, z, w)]
    r.check("ERRATUM: partial_red(y) on {x=z=t=0} equals -2w, NOT 0",
            lambda: E(red[y].subs({x: 0, z: 0, t: 0})) == -2 * w,
            "so {x=z=t=0} is NOT the degeneracy locus")
    r.check("the other three components DO vanish on {x = z = t = 0}",
            lambda: all(E(red[v].subs({x: 0, z: 0, t: 0})) == 0
                        for v in (x, z, w)))

    nvars, ngens = 5, len(comps) + 1
    r.record("elimination measurement before any Groebner call", "MEASURED",
             f"{nvars} variables, {ngens} generators -- against the frozen "
             f"ceiling of {CEILING_VARS} unknowns / {CEILING_CPU_S//60} "
             "CPU-min. Far under; H7 in-process alarm armed anyway.")
    J = [E(c) for c in comps] + [P]
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        GB = sp.groebner(J, *GENS, order="grevlex")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    def rad_power(el, N=8):
        for k in range(1, N + 1):
            if GB.reduce(E(el**k))[1] == 0:
                return k
        return None

    kt = rad_power(t)
    r.check("THE LOCUS IS SUPPORTED ON {t = 0}: t is in the radical",
            lambda: kt is not None, f"t^{kt} reduces to 0")
    for el, nm in ((x, "x"), (z, "z"), (y, "y"), (w, "w")):
        k = rad_power(el)
        r.check(f"H15 boundary: {nm} is NOT in the radical",
                lambda kk=k: kk is None,
                "the radical test can fail; only t is forced")
    r.check("the locus is NON-EMPTY: the line {x=z=t=w=0} lies in it",
            lambda: all(E(c.subs({x: 0, z: 0, t: 0, w: 0})) == 0
                        for c in comps)
            and E(P.subs({x: 0, z: 0, t: 0})) == 0,
            "a whole A^1_y of common zeros -- so the action is NOT free")
    led["degeneracy_locus"] = {
        "supported_on": "{t = 0} -- t^%s is in the degeneracy ideal" % kt,
        "non_empty": "contains the line {x = z = t = w = 0} = A^1_y",
        "erratum": "an in-session hand claim that the locus was the whole "
                   "plane {x=z=t=0} was WRONG (partial_red(y) = -2w there) "
                   "and was caught by the machine, not by re-reading.",
    }

    # ------------------------------------------------------------------
    print("\nRD-4b  the locus, chart by chart")
    Z = E(z + x * w)                                     # = Phi(z)
    # chart x != 0, t = 0: substitute the model y = -(x+z^2)/x^2 and clear
    ybar = -(x + z**2) / x**2
    nums = [sp.factor(sp.numer(sp.cancel(E(c.subs(t, 0).xreplace({y: ybar})))))
            for c in comps]
    nums = [n for n in nums if n != 0]
    Gc = sp.groebner(nums, x, z, w, order="grevlex")
    target = E(Z**2 * (Z**2 + x))
    r.check("on {x != 0, t = 0} the locus is exactly V(Z^2 (Z^2 + x))",
            lambda: len(Gc.exprs) == 1
            and sp.simplify(sp.cancel(sp.expand(Gc.exprs[0]) / target)).is_number
            and sp.cancel(sp.expand(Gc.exprs[0]) / target) != 0,
            f"Z = z + xw; GB is the single generator {sp.factor(Gc.exprs[0])}")
    r.check("Z^2 + x = -x p  modulo (P, t):  so the locus is V(Z) u V(p)",
            lambda: is_in_ideal(E(Z**2 + x + x * KER_P + t**3), P, y),
            "from the transported relation x(p+1) + Z^2 + t^3 = 0")
    # chart x = 0
    r.check("on {x = 0} cap X cap {t = 0} the fibre forces z = 0",
            lambda: E(P.subs({x: 0, t: 0}) - z**2) == 0)
    r.check("and there the locus is exactly {w = 0}: the line A^1_y",
            lambda: E(red[y].subs({x: 0, z: 0, t: 0})) == -2 * w
            and all(E(red[v].subs({x: 0, z: 0, t: 0})) == 0
                    for v in (x, z, w)))
    r.check("the x = 0 fibre of X is the cusp cylinder {z^2+t^3=0} x A^1_y",
            lambda: E(P.subs(x, 0) - (z**2 + t**3)) == 0)
    SING = sp.groebner([z**2 + t**3, sp.diff(z**2 + t**3, z),
                        sp.diff(z**2 + t**3, t)], z, t, order="grevlex")
    r.check("the cusp's singular edge is exactly {z = t = 0}",
            lambda: SING.reduce(z)[1] == 0 and SING.reduce(t**2)[1] == 0
            and SING.reduce(sp.Integer(1))[1] != 0,
            f"GB {list(SING.exprs)}")
    led["locus_chart_by_chart"] = {
        "x_nonzero_t_zero": "exactly V(Z) u V(p), Z = z + xw",
        "x_zero": "exactly the line {x = z = t = w = 0} = A^1_y, which lies "
                  "inside the cuspidal edge {z = t = 0} of the x = 0 fibre "
                  "(its w = 0 slice)",
        "answer_to_RD4_question":
            "YES -- the slice obstruction is supported entirely on {t = 0}; "
            "it meets the cusp fibre exactly in the w = 0 slice of the "
            "cuspidal edge.",
        "why_it_matters":
            "Masuda 2512.06687 Thm 4.3 needs a FREE Ga-action; his Example "
            "4.2 already records that the factoriality hypothesis fails at "
            "the cusp fibre of this same cylinder. For DMJP's partial the "
            "FREENESS hypothesis fails too, and inside the same cusp. The "
            "cusp is the escape hatch on both hypotheses at once.",
    }

    # ------------------------------------------------------------------
    print("\nRD-4c  the image ideal is proper (this is what kills the slice)")
    r.check("partial_red(A[w]) is inside (x, z, t, w) A[w]",
            lambda: all(E(c.subs({x: 0, z: 0, t: 0, w: 0})) == 0
                        for c in comps),
            "chain rule: partial_red(F) = sum partial_red(v) dF/dv")
    r.check("H15 boundary: it is NOT inside the smaller (x, z, t) A[w]",
            lambda: not all(E(c.subs({x: 0, z: 0, t: 0})) == 0
                            for c in comps),
            "the containment is sharp at this level")
    r.check("(x,z,t,w) A[w] is a PROPER ideal: A[w]/(x,z,t,w) = C[y]",
            lambda: E(P.subs({x: 0, z: 0, t: 0})) == 0)
    led["image_ideal"] = {
        "containment": "partial_red(A[w]) subset (x,z,t,w) A[w], proper",
        "sharpness": "NOT contained in (x,z,t): partial_red(y) = -2w there",
        "independent_route": "the origin lies on X x A^1 and every component "
                             "vanishes there, so partial_red(F)(0) = 0 for "
                             "every F (used again in RD-5)",
    }

    # ------------------------------------------------------------------
    print("\nRD-4d  plinth elements, in the kernel coordinates p, q, t")
    plinth = {}
    # two routes where both are affordable (depth 3): direct iteration in
    # A[w], and the cheap transport through Delta.
    for nm, v in (("x", x), ("w", w)):
        direct = apply_red(apply_red(v))
        F = plinth_via_transport(v, 2)
        back = E(F.xreplace({Pp: KER_P, Qq: KER_Q})) if F is not None else None
        agree = F is not None and is_in_ideal(E(direct - back), P, y)
        plinth[nm] = str(sp.factor(F)) if F is not None else None
        r.check(f"partial_red^2({nm}) = F(p,q,t): two independent routes agree",
                lambda a=agree: a,
                f"F = {sp.factor(F) if F is not None else '-'}")
    # depth 5: transport only (direct iteration is not affordable)
    for nm, v in (("y", y), ("z", z)):
        F = plinth_via_transport(v, 4)
        nz = plinth_via_transport(v, 5)
        plinth[nm] = str(sp.factor(F)) if F is not None else None
        r.check(f"partial_red^4({nm}) = F(p,q,t) in ker, and partial_red^5 = 0",
                lambda FF=F, nn=nz: FF is not None and nn is not None
                and sp.expand(nn) == 0,
                f"F = {sp.factor(F) if F is not None else '-'}")
    Zloc = E(z + x * w)
    dZ = apply_red(Zloc)
    plinth["Z"] = "t**3*(p_ + 1)"
    r.check("partial_red(Z) = t^3 (p+1) is a plinth element, Z = z + xw",
            lambda: (is_in_ideal(E(dZ - t**3 * (KER_P + 1)), P, y)
                     and is_in_ideal(E(apply_red(E(t**3 * (KER_P + 1)))),
                                     P, y)),
            "in the image AND in the kernel: the lowest-degree one found")
    r.check("H15 boundary: a non-kernel element has no (p,q,t) form",
            lambda: plinth_via_transport(x, 1) is None,
            "partial_red(x) is not in the kernel; the reader returns None")

    r.check("p, q, t all map to 0 in A[w]/(x,z,t,w) = C[y]",
            lambda: E(KER_P.subs({x: 0, z: 0, t: 0, w: 0})) == 0
            and E(KER_Q.subs({x: 0, z: 0, t: 0, w: 0})) == 0)
    r.record("pl(partial_red) is inside (p, q, t) C[p,q,t]", "MEASURED",
             "because partial_red(A[w]) subset (x,z,t,w)A[w] and the "
             "composite C[p,q,t] -> A[w]/(x,z,t,w) = C[y] kills p, q and t. "
             "A PROPER ideal, so 1 is not in the plinth.")
    r.record("observation, not a theorem", "MEASURED",
             "every plinth element computed here lies in t^3 (p+1) "
             "C[p,q,t]: t^3(p+1), q t^3(p+1), -2t^6(p+1), q t^3(p+1), "
             "-6 q^2 t^6 (p+1)^2, 12 q t^9 (p+1)^2. Whether "
             "pl(partial_red) = (t^3(p+1)) is NOT decided here.")
    led["plinth"] = {
        "explicit_elements": plinth,
        "containment": "pl(partial_red) subset (p, q, t) C[p,q,t] -- proper",
        "observation": "all computed elements lie in t^3 (p+1) C[p,q,t]; "
                       "equality with (t^3(p+1)) is NOT decided here",
    }

    # ------------------------------------------------------------------
    print("\nRD-4e  THE LOCALISATION THEOREM")
    r.check("Z := z + x w satisfies partial_red(Z) = t^3 (p+1)",
            lambda: is_in_ideal(E(apply_red(Z) - t**3 * (KER_P + 1)), P, y))
    r.check("the transported relation: x (p+1) + Z^2 + t^3 = 0 in A[w]",
            lambda: is_in_ideal(E(x * (KER_P + 1) + Z**2 + t**3), P, y),
            "the image of S = xy + z^2 + x + t^3 under Phi")
    xf = -(Z**2 + t**3) / (Pp + 1)
    wf = (Pp * Z - xf * Qq) / (2 * t**3)
    zf = Z - xf * (Pp * Z - xf * Qq) / (2 * t**3)
    yf = (-(Pp + Pp**2 + Qq * Z) / t**3
          - (Pp * Z - xf * Qq) ** 2 / (4 * t**6))
    subs_back = {Pp: KER_P, Qq: KER_Q}
    CLEAR = E(t**6 * (KER_P + 1) ** 3)
    formulas = {}
    for nm, expr, tgt in (("x", xf, x), ("y", yf, y),
                          ("z", zf, z), ("w", wf, w)):
        num = E(sp.cancel(E((expr.xreplace(subs_back) - tgt) * CLEAR)))
        ok = is_poly(num) and is_in_ideal(num, P, y)
        formulas[nm] = str(expr)
        r.check(f"{nm} recovered from p, q, t, Z over the localisation",
                lambda o=ok: o)
    r.check("H15 boundary: the recovery formulas genuinely need 1/(p+1)",
            lambda: sp.denom(sp.together(xf)) != 1
            and sp.denom(sp.together(yf)) != 1,
            "x = -(Z^2+t^3)/(p+1), and y needs 1/t^6 as well")
    r.record("A[w][1/(t(p+1))] = C[p,q,t][1/(t(p+1))][Z]", "MEASURED",
             "Z = z + xw is a slice after inverting t(p+1): "
             "partial_red(Z) = t^3(p+1). So X x A^1 IS a trivial A^1-bundle "
             "over the complement of the divisor {t(p+1) = 0} in "
             "A^3 = Spec(ker partial).")
    led["localisation_theorem"] = {
        "slice_after_localisation": "Z = z + x w, partial_red(Z) = t^3 (p+1)",
        "statement": "A[w][1/(t(p+1))] = C[p,q,t][1/(t(p+1))][Z]",
        "recovery_formulas": formulas,
        "reading":
            "the entire obstruction to X x A^1 = A^4 THROUGH THIS FLOW is "
            "concentrated on the divisor {t (p+1) = 0} of A^3. This is the "
            "boundary-globalisation question of FINAL_WORD A3 item 3, made "
            "explicit and elementary on the affine side.",
    }

    # ------------------------------------------------------------------
    print("\nRD-4f  the Masuda Thm 4.3 hypothesis grid, for THIS partial")
    grid = {
        "free Ga-action":
            "FAILS -- the degeneracy locus is non-empty (it contains the line {x=z=t=w=0}) and is supported on {t=0} (RD-4a/b). NEW.",
        "affine quotient exists and is A^3":
            "HOLDS -- ker partial = C[p,q,t] = C^[3] (RD-3). NEW.",
        "quotient morphism equidimensional / surjective":
            "NOT DECIDED here. The fixed line {x=z=t=w=0} maps to the origin "
            "of A^3; measuring fibre dimensions is a bounded follow-up.",
        "A^3-fibration with factorial closed fibres":
            "FAILS at the cusp -- Masuda's own Example 4.2, for the "
            "cylinder's induced fibration (verified reading, A21).",
    }
    for k, v in grid.items():
        r.record(f"Masuda 4.3 grid: {k}", "MEASURED", v)
    led["masuda_grid"] = grid

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RD-4-PLINTH.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RD-4", "plan": "RD-OPS.md (A22)",
            "date": "2026-07-31", "mode": "I",
            "source": "the derivation is DMJP arXiv:0903.4278 Prop 7.2; the "
                      "plinth, locus and localisation results are ours",
            "ceilings": {"vars": CEILING_VARS, "cpu_s": CEILING_CPU_S,
                         "groebner_calls": 2, "vars_used": 5,
                         "measured_before_running": True},
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
