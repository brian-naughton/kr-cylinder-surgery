#!/usr/bin/env python3
"""RG-1 — the MQ4 diagnostics (RG-OPS §4, authorised at A23).

(a) Is pl(partial_red) = (t^3 (p+1)) exactly?
(b) Masuda's remaining hypotheses for pi : X x A^1 -> A^3 = Spec C[p,q,t]:
    equidimensionality and surjectivity.

The derivation is DMJP's (arXiv:0903.4278 Prop 7.2); the diagnostics are
ours. Every carried-over fact is re-verified in process first (A24
re-load rule). Masuda's hypothesis list: arXiv:2512.06687 Thm 4.3.

DEGREE CAPS ARE DECLARED IN THE LEDGER BEFORE ANY SEARCH RUNS, and
cap-bounded non-findings are recorded MEASURED, never as negatives.

Ledger: runs_synthesis/RG-1-DIAGNOSTICS.json. Mode I.
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
from rg_common import (KGENS, LGENS, Pp, Qq, Tt, ZZ, H_L, TO_L, in_C_pqt,
                       krull_dim, order_along, to_L, verify_carryover,
                       z_degree, z_split)

CEILING_VARS, CEILING_CPU_S = 60, 30 * 60

#: DECLARED BEFORE THE SEARCH (RG-OPS §2)
CAPS = {
    "plinth_search_total_degree_in_xyztw": 4,
    "plinth_search_basis_size": None,      # filled in below, before running
    "fibre_dimension_variables": 5,
    "groebner_variable_ceiling": CEILING_VARS,
    "groebner_cpu_ceiling_s": CEILING_CPU_S,
}


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "MEASURED", "REFUTED", "NOT-ATTEMPTED")

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
        print(f"  {name:<54s} {outcome:<11s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<54s} {outcome:<11s}         [{note}]", flush=True)

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
    for a in range(deg + 1):
        for b in range(deg + 1 - a):
            for c in range(deg + 1 - a - b):
                for d in range(deg + 1 - a - b - c):
                    for e in range(deg + 1 - a - b - c - d):
                        out.append((a, b, c, d, e))
    out.sort()
    return [x**a * y**b * z**c * t**d * w**e for a, b, c, d, e in out]


def main() -> int:
    r = Runner()
    led: dict = {}

    print("\nRG-0  carry-over, re-verified IN PROCESS (A24 re-load rule)")
    red = verify_carryover(r.check)

    def d_red(f):
        return E(sum(red[v] * sp.diff(f, v) for v in GENS))

    # ------------------------------------------------------------------
    print("\nRG-1a  partial_red in the good coordinates")
    # On L, partial_red = t^3 (p+1) d/dZ. Read off its action on the
    # natural generators; every value re-verified against A[w].
    action = {}
    for nm, eL, eA in (("Z", ZZ, E(z + x * w)),
                       ("x", TO_L[x], x),
                       ("w", TO_L[w], w),
                       ("H", H_L, E(-(y + w**2)))):
        got = sp.cancel(E(Tt**3 * (Pp + 1) * sp.diff(eL, ZZ)))
        action[nm] = str(sp.factor(got))
        r.check(f"partial_red({nm}) = t^3(p+1) d/dZ applied: {sp.factor(got)}",
                lambda g=got, ea=eA: is_in_ideal(
                    E(d_red(ea) - _back(g)), P, y),
                "L-side and A[w]-side agree")
    r.check("H = -(y + w^2) really lies in A[w] and has Z-degree 1",
            lambda: z_degree(to_L(E(-(y + w**2)))) == 1
            and sp.cancel(to_L(E(-(y + w**2))) - H_L) == 0,
            "H = (p + p^2 + q Z)/t^3")
    led["partial_red_in_good_coordinates"] = {
        "formula": "on L, partial_red = t^3 (p+1) d/dZ",
        "action": action,
        "note": "p, q, t are killed; these four are the moving generators",
    }

    # ------------------------------------------------------------------
    print("\nRG-1a  MQ4(a): is pl(partial_red) = (t^3 (p+1))?")
    wit = E(y + w**2)
    dwit = d_red(wit)
    r.check("WITNESS: partial_red(y + w^2) = -q(p+1)   [A[w] side]",
            lambda: is_in_ideal(E(dwit + KER_Q * (KER_P + 1)), P, y))
    r.check("WITNESS: the same, read on the L side",
            lambda: sp.cancel(E(Tt**3 * (Pp + 1) * sp.diff(to_L(wit), ZZ)
                                + Qq * (Pp + 1))) == 0)
    r.check("y + w^2 is a local slice: partial_red^2(y + w^2) = 0",
            lambda: is_in_ideal(E(d_red(dwit)), P, y),
            "so -q(p+1) really is in the plinth")
    G_t3p = sp.groebner([Tt**3 * (Pp + 1)], *KGENS, order="grevlex")
    r.check("q(p+1) is NOT in the ideal (t^3(p+1)) of C[p,q,t]",
            lambda: G_t3p.reduce(E(Qq * (Pp + 1)))[1] != 0,
            "every element of (t^3(p+1)) dies at t=0; q(p+1) does not")
    r.check("H15 boundary: t^3(p+1) IS in it -- the membership test passes",
            lambda: G_t3p.reduce(E(Tt**3 * (Pp + 1)))[1] == 0)
    r.record("MQ4(a): pl(partial_red) = (t^3(p+1)) is REFUTED", "REFUTED",
             "pl contains q(p+1) = -partial_red(y + w^2), which is not in "
             "(t^3(p+1)). The R∂ observation that all five elements found "
             "there lay in t^3(p+1)C[p,q,t] was an artefact of computing "
             "only iterated images partial_red^k(generator); the local "
             "slice y + w^2 is not of that form. So pl STRICTLY CONTAINS "
             "(t^3(p+1)), and contains (p+1)(t^3, q).")
    G_known = sp.groebner([Tt**3 * (Pp + 1), Qq * (Pp + 1)], *KGENS,
                          order="grevlex")
    r.check("H15 boundary: (p+1) itself is NOT in (p+1)(t^3,q)",
            lambda: G_known.reduce(E(Pp + 1))[1] != 0,
            "consistent with RD-4's pl inside (p,q,t)")
    led["mq4a"] = {
        "verdict": "REFUTED",
        "witness": "partial_red(y + w^2) = -q(p+1); y + w^2 = -H is a "
                   "local slice (partial_red^2 = 0)",
        "known_containment": "pl contains (p+1)(t^3, q), strictly bigger "
                             "than (t^3(p+1))",
        "why_R-partial_missed_it": "R∂ sampled only iterated images "
                                   "partial_red^k(v) on the five "
                                   "generators; the local slice y+w^2 is "
                                   "not among them. Logged, not reworded.",
    }

    # ------------------------------------------------------------------
    D = CAPS["plinth_search_total_degree_in_xyztw"]
    mons = monomials(D)
    CAPS["plinth_search_basis_size"] = len(mons)
    print(f"\nRG-1a  cap-bounded hunt for MORE plinth (declared cap: total "
          f"degree <= {D} in x,y,z,t,w; {len(mons)} basis monomials)")
    r.record("DECLARED CAPS (before the search ran)", "MEASURED",
             json.dumps(CAPS, sort_keys=True))

    # f = sum c_m m ; partial_red(f) in C[p,q,t]  <=>  to_L(f) has
    # Z-degree <= 1 and t^3(p+1)*(its Z^1 coefficient) is polynomial.
    cs = sp.symbols(f"c0:{len(mons)}")
    cols = [z_split(to_L(m)) for m in mons]
    maxz = max(len(c) for c in cols)
    # Z-degree >= 2 must vanish: linear in the c_i over Q after clearing
    eqs = []
    for j in range(2, maxz):
        expr = sum((ci * col[j]) for ci, col in zip(cs, cols)
                   if j < len(col))
        num, _ = sp.fraction(sp.cancel(sp.together(E(expr))))
        if E(num) != 0:
            eqs += [E(c) for c in sp.Poly(E(num), *LGENS).coeffs()]
    M, rhs = sp.linear_eq_to_matrix(eqs, cs)
    ns = M.nullspace()
    pl_vals = []
    for vec in ns:
        fL = sp.cancel(sum(vec[i] * to_L(mons[i]) for i in range(len(mons))))
        if sp.cancel(fL) == 0:
            continue
        val = sp.cancel(E(Tt**3 * (Pp + 1) * sp.diff(fL, ZZ)))
        if val != 0 and in_C_pqt(val):
            pl_vals.append(sp.factor(val))
    r.record("nullspace dimension of the Z-degree<=1 condition", "MEASURED",
             f"{len(ns)} of {len(mons)} basis monomials survive")
    pl_vals = sorted({str(v) for v in pl_vals})
    r.record(f"plinth values found at the declared cap ({len(pl_vals)})",
             "MEASURED", "; ".join(pl_vals) if pl_vals else "none")
    # do they all lie in (p+1)(t^3, q)?
    allin = all(G_known.reduce(E(sp.sympify(v)))[1] == 0 for v in pl_vals)
    r.check("every plinth value found lies in (p+1)(t^3, q)",
            lambda: allin,
            "no generator beyond the two known ones appeared at this cap")
    r.record("MQ4(a) residue", "MEASURED",
             "pl(partial_red) CONTAINS (p+1)(t^3, q); at the declared cap "
             "nothing outside that ideal appeared. Whether the containment "
             "is equality is NOT decided -- a cap-bounded non-finding, not "
             "a negative.")
    led["mq4a_search"] = {"caps": dict(CAPS), "values_found": pl_vals,
                          "all_inside_(p+1)(t^3,q)": bool(allin)}

    # ------------------------------------------------------------------
    print("\nRG-1b  MQ4(b): Masuda's equidimensionality and surjectivity")
    r.record("elimination measurement before any Groebner call", "MEASURED",
             f"fibres are cut in {CAPS['fibre_dimension_variables']} "
             f"variables against the frozen ceiling of {CEILING_VARS} "
             f"unknowns / {CEILING_CPU_S//60} CPU-min; H7 alarm armed.")

    SAMPLES = (
        ("generic          (2,3,1)", (2, 3, 1)),
        ("origin           (0,0,0)", (0, 0, 0)),
        ("t=0, generic     (5,7,0)", (5, 7, 0)),
        ("t=0, q=0         (2,0,0)", (2, 0, 0)),
        ("p=-1, t!=0      (-1,3,1)", (-1, 3, 1)),
        ("p=-1, t=0       (-1,3,0)", (-1, 3, 0)),
        ("p=-1, q=0, t=0  (-1,0,0)", (-1, 0, 0)),
        ("q=0, t!=0        (2,0,1)", (2, 0, 1)),
    )
    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        fibres = {}
        for label, (p0, q0, t0) in SAMPLES:
            gens = [P, E(KER_P - p0), E(KER_Q - q0), E(t - t0)]
            fibres[label] = krull_dim(gens, GENS)
            print(f"    fibre over {label:<24s} dim = {fibres[label]}")
        # is q forced to vanish on {p = -1, t = 0}?
        Gpm1 = sp.groebner([P, E(KER_P + 1), t], *GENS, order="grevlex")
        qpow = None
        for k in range(1, 7):
            if Gpm1.reduce(E(KER_Q**k))[1] == 0:
                qpow = k
                break
        xpow = any(Gpm1.reduce(E(x**k))[1] == 0 for k in range(1, 7))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    r.record("fibre dimensions over the sampled points", "MEASURED",
             json.dumps({k: int(v) for k, v in fibres.items()},
                        sort_keys=True))

    r.check("the generic fibre is 1-dimensional (an A^1, per RD-4)",
            lambda: fibres["generic          (2,3,1)"] == 1)
    r.check("SURJECTIVITY FAILS: the fibre over (-1,3,0) is EMPTY",
            lambda: fibres["p=-1, t=0       (-1,3,0)"] == -1,
            "a point of A^3 with no preimage in X x A^1")
    r.check("the missing locus: q vanishes identically on {p=-1, t=0}",
            lambda: qpow is not None, f"q^{qpow} reduces to 0 there")
    r.check("H15 boundary: x does NOT vanish on {p=-1,t=0} -- not everything does",
            lambda: not xpow,
            "the radical test discriminates")
    r.check("EQUIDIMENSIONALITY FAILS: the fibre over the origin is a SURFACE",
            lambda: fibres["origin           (0,0,0)"] == 2,
            "dim 2 against the generic dim 1")
    r.check("H15 boundary: the dimension routine returns -1 on an empty ideal",
            lambda: krull_dim([x, E(x - 1)], GENS) == -1)
    r.check("H15 boundary: and returns 1 on a fibre that really is an A^1",
            lambda: fibres["t=0, generic     (5,7,0)"] == 1,
            "so dim 2 and dim -1 above are findings, not artefacts")

    jump = sorted(k for k, v in fibres.items() if v == 2)
    empty = sorted(k for k, v in fibres.items() if v == -1)
    led["mq4b_fibres"] = {k: int(v) for k, v in fibres.items()}
    led["mq4b"] = {
        "equidimensional": False,
        "surjective": False,
        "jump_locus_samples": jump,
        "empty_fibre_samples": empty,
        "missing_locus":
            "the image misses (at least) the punctured line "
            "{p=-1, t=0, q!=0} -- q lies in the radical of (P,p+1,t) -- "
            "AND {t=0, q=0, p not in {0,-1}}; see the t=0 table below for "
            "the exact pattern. Everything missing lies over {t = 0}.",
        "jump_locus":
            "the fibre jumps from dimension 1 to dimension 2 over "
            f"{jump} -- both sampled points have q = t = 0.",
        "image_dense": None,
    }

    old = signal.signal(signal.SIGALRM,
                        lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(CEILING_CPU_S)
    try:
        A_, B_, C_ = sp.symbols("A_ B_ C_")
        Gelim = sp.groebner([P, E(KER_P - A_), E(KER_Q - B_), E(t - C_)],
                            x, y, z, t, w, A_, B_, C_, order="lex")
        elim = [g for g in Gelim.exprs
                if not (set(g.free_symbols) & {x, y, z, t, w})]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    r.check("the image is nonetheless DENSE: the eliminant is zero",
            lambda: elim == [],
            "so pi misses only a lower-dimensional set")
    led["mq4b"]["image_dense"] = (elim == [])

    print("\nRG-1b-bis  the t = 0 slice, and WHY it behaves that way")
    # the master relation, certified directly in A[w]
    REL = E(KER_Q * (z + x * w) + KER_P + KER_P**2 + t**3 * (y + w**2))
    r.check("IDENTITY in A[w]:  q Z + p + p^2 + t^3 (y + w^2) = 0",
            lambda: is_in_ideal(REL, P, y),
            "Z = z+xw; this is the H-relation, rearranged")
    r.check("H15 boundary: the same expression with q -> q+1 is NOT in (P)",
            lambda: not is_in_ideal(E(REL + (z + x * w)), P, y))
    # the (p,q) table over t = 0
    tbl = {}
    for p0 in (-2, -1, 0, 1, 2, 5):
        for q0 in (0, 1, 3):
            tbl[f"p={p0},q={q0}"] = krull_dim(
                [E(x**2 * y + z**2 + x), E(KER_P - p0), E(KER_Q - q0)],
                (x, y, z, w))
    r.record("fibre dimensions over the t = 0 plane", "MEASURED",
             json.dumps(tbl, sort_keys=True))
    r.check("over t=0 with q != 0: dim 1 everywhere EXCEPT p = -1 (empty)",
            lambda: all(tbl[f"p={p0},q={q0}"] == (-1 if p0 == -1 else 1)
                        for p0 in (-2, -1, 0, 1, 2, 5) for q0 in (1, 3)))
    r.check("over t=0 with q = 0: EMPTY except p = 0 and p = -1 (dim 2)",
            lambda: all(tbl[f"p={p0},q=0"] == (2 if p0 in (0, -1) else -1)
                        for p0 in (-2, -1, 0, 1, 2, 5)))
    r.record("and that is exactly what the identity predicts", "MEASURED",
             "setting t = 0 in q Z + p + p^2 + t^3(y+w^2) = 0 leaves "
             "q Z = -p(1+p). So q = 0 FORCES p(1+p) = 0 -- the exact "
             "pattern above, with the two exceptional points p = 0 and "
             "p = -1 carrying the dimension jump.")
    led["t0_slice"] = {
        "identity": "q Z + p + p^2 + t^3 (y + w^2) = 0 in A[w]",
        "restricted_to_t0": "q Z = -p(1+p)",
        "table": {k: int(v) for k, v in tbl.items()},
        "image_over_t0": "{q != 0, p != -1} together with the two points "
                         "(p,q) = (0,0) and (-1,0)",
        "jump_locus": "exactly (p,q,t) = (0,0,0) and (-1,0,0), the two "
                      "roots of p(1+p) -- fibre dimension 2",
    }

    print("\nRG-1b-ter  WHICH boundary component misbehaves")
    r.check("over {t != 0} every sampled fibre is a 1-dimensional A^1,"
            " including p = -1",
            lambda: all(fibres[k] == 1 for k in fibres if "t!=0" in k
                        or k.startswith("generic")))
    r.record("the {p+1 = 0} component is TAME; {t = 0} carries everything",
             "MEASURED",
             "RG-OPS §1 asked this and said decide, don't assume. DECIDED: "
             "over t != 0 the fibres are A^1 even along p = -1, so the "
             "(p+1) component is well behaved on its own. Every failure -- "
             "non-surjectivity, the dimension jump, the degeneracy locus "
             "(RD-4) -- lives over {t = 0}. p = -1 only bites in "
             "COMBINATION with t = 0.")
    led["which_component"] = {
        "t=0": "carries the degeneracy locus, the empty fibres and the "
               "dimension jump",
        "p+1=0": "TAME on its own -- A^1 fibres over t != 0; only "
                 "misbehaves where it meets {t = 0}",
    }

    print("\nRG-1c  the Masuda Thm 4.3 table for this pi, completed")
    grid = {
        "free Ga-action":
            "FAILS -- degeneracy locus non-empty, contains {x=z=t=w=0} "
            "(RD-4, re-verified in process here).",
        "affine quotient exists and equals A^3":
            "HOLDS -- ker partial = C[p,q,t] = C^[3] (RD-3).",
        "quotient morphism surjective":
            "FAILS -- the punctured line {p=-1, t=0, q!=0} has empty "
            "fibre (q lies in the radical of (P,p+1,t)). The image is "
            "dense but not all of A^3. NEW at this visit.",
        "quotient morphism equidimensional":
            "FAILS -- the fibre over the origin (and over (-1,0,0)) is a "
            "SURFACE, against the generic A^1. NEW at this visit.",
        "A^3-fibration with factorial closed fibres":
            "FAILS at the cusp -- Masuda's own Example 4.2 (verified "
            "reading, A21).",
    }
    for k, v in grid.items():
        r.record(f"Masuda 4.3: {k}", "MEASURED", v)
    led["masuda_table"] = grid
    led["masuda_reading"] = (
        "The table is now complete, and it is much worse than R∂ left it. "
        "Only ONE of Masuda Thm 4.3's hypotheses holds for this pi: the "
        "affine quotient is A^3. FOUR fail -- freeness, surjectivity, "
        "equidimensionality, and factoriality of the closed fibres. R∂ "
        "recorded surjectivity and equidimensionality as NOT DECIDED and "
        "they have now decided AGAINST. Masuda's theorem is not merely "
        "inapplicable to this partial through one hypothesis; this "
        "Ga-action is far from his hypotheses. Mode I: this says nothing "
        "about X x A^1 = A^4 -- only that THIS action cannot be the "
        "vehicle for THAT theorem.")

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RG-1-DIAGNOSTICS.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RG-1", "plan": "RG-OPS.md (A24)",
            "date": "2026-07-31", "mode": "I",
            "source": "derivation DMJP arXiv:0903.4278 Prop 7.2; "
                      "hypothesis list Masuda arXiv:2512.06687 Thm 4.3; "
                      "diagnostics ours",
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


_BACK = {Pp: KER_P, Qq: KER_Q, Tt: t, ZZ: E(z + x * w)}


def _back(expr):
    return E(sp.cancel(expr).xreplace(_BACK))


if __name__ == "__main__":
    sys.exit(main())
