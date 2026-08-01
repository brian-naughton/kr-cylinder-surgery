#!/usr/bin/env python3
"""RD-1 — transcribe and verify the published objects of DMJP §7.

SOURCE (transcribed directly, 2026-07-31, from the arXiv PDF of
Dubouloz--Moser-Jauslin--Poloni, "Inequivalent embeddings of the
Koras-Russell cubic threefold", arXiv:0903.4278v1, §7, pages 11-12;
pypdf layout-mode extraction cross-read against default-mode
extraction).  EVERY object below is theirs.  No novelty is claimed on
the derivation, the conjugacy, or the corollary -- see W2 / RD-OPS §1.

This runner verifies, by machine and in exact rational arithmetic:

  * the paper's own eight printed identities (Prop 7.1);
  * that Delta is an LND of C[x,y,z,t^{+-1},w] killing S;
  * the four components of ``partial`` = Phi o Delta o Psi, that they
    are POLYNOMIAL (the paper asserts this), and the printed value
    ``partial(x) = -2 t^6 (z + x w)``;
  * that ``partial`` preserves the ideal (P) -- both M0 twin routes;
  * local nilpotence on generators in the faithful localisation model
    A[w] --> C[x^{+-1},z,t,w], with the nilpotency DEPTHS recorded;
  * H15 boundaries: a rival reading of the source and a perturbed
    derivation, both of which must FAIL.

Ledger: runs_synthesis/RD-1-TRANSCRIBE.json.  Mode I.
"""

from __future__ import annotations

import json
import os
import sys
import time

import sympy as sp

from checker import is_in_ideal
from checker_twin import member as member_twin

x, y, z, t, w = sp.symbols("x y z t w")
GENS = (x, y, z, t, w)

E = sp.expand

# --- the two hypersurfaces (DMJP §7) -------------------------------------
P = x**2 * y + z**2 + x + t**3          # Koras-Russell cubic
S = x * y + z**2 + x + t**3             # the auxiliary threefold X_1

# --- Prop 7.1: the conjugating endomorphisms (DMJP, verbatim) ------------
PHI = {
    x: x,
    y: x * y - x * w**2 - 2 * z * w,
    z: z + x * w,
    t: t,
    w: 2 * w + y * z + 3 * x * y * w - 3 * z * w**2 - x * w**3,
}

PSI = {
    x: x,
    y: -(y + y**2 + w * z) / t**3 - (y * z - x * w) ** 2 / (4 * t**6),
    z: z - x * (y * z - x * w) / (2 * t**3),
    t: t,
    w: (y * z - x * w) / (2 * t**3),
}


def ap(hom: dict, expr: sp.Expr) -> sp.Expr:
    """Apply a ring endomorphism given on the generators.

    ``xreplace`` is used rather than ``subs``: the keys are atoms, and
    xreplace is exact and SIMULTANEOUS (H1 -- never substitute into an
    expanded expression with a composite pattern; here every pattern is
    a bare Symbol).
    """
    return E(expr.xreplace(hom))


# --- Prop 7.2: the triangular LND on the localised auxiliary cylinder ----
def delta(f: sp.Expr) -> sp.Expr:
    """Delta = t^6 ( -2 z d/dx + (y+1) d/dz )  (DMJP Prop 7.2)."""
    return E(t**6 * (-2 * z * sp.diff(f, x) + (y + 1) * sp.diff(f, z)))


def t_min_exp(e: sp.Expr, pad: int = 40) -> int | None:
    """Least exponent of t occurring in a Laurent expression (None if 0)."""
    e = E(e)
    if e == 0:
        return None
    q = E(e * t**pad)
    p = sp.Poly(q, *GENS)          # raises if q is not polynomial
    return min(m[3] for m in p.monoms()) - pad


# --- the faithful localisation model  A[w] --> C[x^{+-1},z,t,w] ----------
# P is linear in y with leading coefficient x^2, so in A = C[x,y,z,t]/(P)
# we have y = -(z^2 + x + t^3)/x^2 and A embeds in C[x^{+-1},z,t].
IOTA_Y = -(z**2 + x + t**3) / x**2
IOTA = {y: IOTA_Y}


def iota(e: sp.Expr) -> sp.Expr:
    return E(e.xreplace(IOTA))


def _fresh(path):
    if os.path.exists(path):
        print(f"ERROR: refusing to overwrite existing ledger: {path}")
        sys.exit(1)
    return path


class Runner:
    CLASSES = ("PASS", "FAIL", "ERRATUM", "NOT-ATTEMPTED")

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
        print(f"  {name:<46s} {outcome:<13s} {dt:6.2f}s"
              + (f"  [{note}]" if note else ""), flush=True)
        return bool(cond)

    def record(self, name, outcome, note):
        assert outcome in self.CLASSES
        self.results.append((name, outcome, 0.0))
        self.notes[name] = note
        print(f"  {name:<46s} {outcome:<13s}         [{note}]", flush=True)

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

    print("\nRD-1a  Prop 7.1 -- the paper's own printed identities")
    r.check("7.1-i   Phi(S) = P", lambda: E(ap(PHI, S) - P) == 0)
    r.check("7.1-ii  Psi(P) = (1 - x y t^-3) S",
            lambda: E(ap(PSI, P) - (1 - x * y / t**3) * S) == 0)

    comps = [
        ("7.1-iii PsiPhi(z) = z", ap(PSI, ap(PHI, z)), z),
        ("7.1-iv  PsiPhi(y) = y - (y/t^3) S",
         ap(PSI, ap(PHI, y)), y - y / t**3 * S),
        ("7.1-v   PsiPhi(w) = w + ((xyw-y^2z-t^3w)/t^6) S",
         ap(PSI, ap(PHI, w)),
         w + (x * y * w - y**2 * z - t**3 * w) / t**6 * S),
        ("7.1-vii PhiPsi(z) = z + (xw/t^3) P",
         ap(PHI, ap(PSI, z)), z + x * w / t**3 * P),
        ("7.1-viii PhiPsi(y) = y - ((y-w^2)/t^3)P - (w^2/t^6)P^2",
         ap(PHI, ap(PSI, y)),
         y - (y - w**2) / t**3 * P - w**2 / t**6 * P**2),
    ]
    for name, got, tgt in comps:
        r.check(name, lambda g=got, s=tgt: E(g - s) == 0)

    # The one printed identity that does NOT hold as printed.  The
    # transcription was re-read from the source in layout mode before
    # this was concluded (RD-OPS: re-transcribe, do not patch); the
    # remaining seven identities over-determine every component of Phi
    # and Psi, so the objects are certified and the defect is in the
    # printed correction term.
    phipsi_w = ap(PHI, ap(PSI, w))
    as_printed = E(phipsi_w - (w - P / t**3)) == 0
    corrected = E(phipsi_w - (w - w * P / t**3)) == 0
    r.check("7.1-vi(as printed) PhiPsi(w) = w - (1/t^3) P  FAILS",
            (not as_printed) and corrected,
            "true value w - (w/t^3) P; numerator w missing in print")
    r.record("7.1-vi  ERRATUM in DMJP Prop 7.1", "ERRATUM",
             "printed 'PhiPsi(w) = w - (1/t^3)P'; verified value is "
             "w - (w/t^3)P. Both are = w mod (P), so Prop 7.1's "
             "conclusion is UNAFFECTED. Typo, not a mathematical error.")
    led["prop_7_1"] = {
        "identities_verified_verbatim": 7,
        "identities_failing_as_printed": 1,
        "erratum": "PhiPsi(w): printed 1/t^3, true w/t^3; = w mod (P)",
        "direction_of_the_algebra_maps":
            "Phi(S)=P so Phi induces R/(S)=Bt[w] --> R/(P)=At[w]; "
            "Psi(P)=(1-xy t^-3)S in (S) so Psi induces At[w] --> Bt[w]. "
            "Hence ker(partial | At[w]) = PHI(ker(Delta | Bt[w])).",
    }

    print("\nRD-1b  Prop 7.2 -- Delta")
    r.check("Delta(S) = 0", lambda: E(delta(S)) == 0)
    r.check("Delta(P) != 0 (boundary, H15)", lambda: E(delta(P)) != 0,
            "Delta descends to Bt[w] only")
    dep = {}
    for v in GENS:
        e, k = v, 0
        while E(e) != 0 and k < 20:
            e = delta(e)
            k += 1
        dep[str(v)] = k
    r.check("Delta locally nilpotent on all 5 generators",
            lambda: all(v < 20 for v in dep.values()), f"depths {dep}")
    led["delta"] = {"nilpotency_depths": dep,
                    "definition": "t^6(-2 z d/dx + (y+1) d/dz)"}

    print("\nRD-1c  the components of partial = Phi o Delta o Psi")
    d_of = {v: ap(PHI, delta(ap(PSI, v))) for v in GENS}
    r.check("(Phi Delta Psi)(t) = 0  (t is in the kernel)",
            lambda: E(d_of[t]) == 0)
    tmins = {str(v): t_min_exp(d_of[v]) for v in GENS}
    r.check("all four components are POLYNOMIAL (no negative t-powers)",
            lambda: all(m is None or m >= 0 for m in tmins.values()),
            f"min t-exponents {tmins}")
    r.check("partial(x) = -2 t^6 (z + x w)   [printed in the proof]",
            lambda: E(d_of[x] - (-2 * t**6 * (z + x * w))) == 0)

    dx, dy, dz, dw = (E(d_of[v]) for v in (x, y, z, w))
    degs = {n: int(sp.Poly(e, *GENS).total_degree())
            for n, e in (("x", dx), ("y", dy), ("z", dz), ("w", dw))}
    terms = {n: len(sp.Poly(e, *GENS).monoms())
             for n, e in (("x", dx), ("y", dy), ("z", dz), ("w", dw))}
    led["partial_components"] = {
        "partial_x": sp.srepr(dx), "partial_y": sp.srepr(dy),
        "partial_z": sp.srepr(dz), "partial_w": sp.srepr(dw),
        "partial_x_str": str(sp.factor(dx)),
        "partial_z_str": str(sp.factor(dz)),
        "partial_w_str": str(sp.factor(dw)),
        "partial_y_str": str(sp.factor(dy)),
        "total_degrees": degs, "term_counts": terms,
        "min_t_exponents": {k: (None if v is None else int(v))
                            for k, v in tmins.items()},
    }
    print(f"    partial(x) = {sp.factor(dx)}")
    print(f"    partial(z) = {sp.factor(dz)}")
    print(f"    partial(w) = {sp.factor(dw)}")
    print(f"    partial(y) : {terms['y']} terms, total degree {degs['y']}")

    print("\nRD-1d  partial preserves the ideal (P) -- both M0 twin routes")
    dP = E(dx * sp.diff(P, x) + dy * sp.diff(P, y)
           + dz * sp.diff(P, z) + E(d_of[t]) * sp.diff(P, t))
    r.check("twin A (localisation-substitution): partial(P) in (P)",
            lambda: is_in_ideal(dP, P, y))
    r.check("twin B (pseudo-division):           partial(P) in (P)",
            lambda: member_twin(dP, P, y))
    q = sp.cancel(dP / P)
    r.check("cofactor partial(P)/P is polynomial",
            lambda: sp.Poly(E(q), *GENS) is not None, f"= {sp.factor(q)}")
    led["ideal_preservation"] = {"cofactor": str(sp.factor(q)),
                                 "twin_A": "PASS", "twin_B": "PASS"}

    print("\nRD-1e  the faithful localisation model, and local nilpotence")
    # D = the same derivation read in C[x^{+-1},z,t,w] via y -> IOTA_Y.
    def D(f: sp.Expr) -> sp.Expr:
        return E(iota(dx) * sp.diff(f, x) + iota(dz) * sp.diff(f, z)
                 + iota(dw) * sp.diff(f, w))

    r.check("model is consistent: D(iota y) = iota(partial y)",
            lambda: E(sp.cancel(D(IOTA_Y) - iota(dy))) == 0,
            "independent second proof of ideal preservation")

    depths, caps = {}, {}
    for nm, e0 in (("x", x), ("y", IOTA_Y), ("z", z), ("t", t), ("w", w)):
        e, k = e0, 0
        while E(sp.cancel(e)) != 0 and k < 12:
            e = D(e)
            k += 1
        depths[nm] = k
        caps[nm] = (k < 12)
    r.check("partial locally nilpotent on all generators of A[w]",
            lambda: all(caps.values()), f"depths {depths}")
    led["local_nilpotence"] = {
        "model": "A[w] --> C[x^{+-1},z,t,w], y |-> -(z^2+x+t^3)/x^2",
        "depths": depths,
        "convention": "depth k means partial^k(v) = 0, partial^{k-1}(v) != 0",
    }

    print("\nRD-1f  H15 boundaries -- checks that must FAIL")
    # (i) the rival reading of the source line "Psi(z) = z - 1/(2t^3) x(yz-xw)":
    #     denominator 2t^3 x instead of numerator factor x.
    PSI_RIVAL = dict(PSI)
    PSI_RIVAL[z] = z - (y * z - x * w) / (2 * t**3 * x)
    r.check("H15(i) rival reading of Psi(z) breaks PsiPhi(z) = z",
            lambda: E(ap(PSI_RIVAL, ap(PHI, z)) - z) != 0,
            "the identity gate discriminates the two readings")
    # (ii) a perturbed derivation: partial(x) += x.  Must lose nilpotence.
    def D_pert(f: sp.Expr) -> sp.Expr:
        return E((iota(dx) + x) * sp.diff(f, x) + iota(dz) * sp.diff(f, z)
                 + iota(dw) * sp.diff(f, w))
    e, k = x, 0
    while E(sp.cancel(e)) != 0 and k < 12:
        e = D_pert(e)
        k += 1
    r.check("H15(ii) perturbed partial (+x d/dx) is NOT nilpotent on x",
            lambda: k >= 12, f"survived {k} iterations (cap 12)")
    # (iii) the same perturbation must also break ideal preservation.
    dP_pert = E(dP + x * sp.diff(P, x))
    r.check("H15(iii) perturbed partial does not preserve (P)",
            lambda: not is_in_ideal(dP_pert, P, y))
    led["h15_boundaries"] = {
        "rival_Psi_z_reading": "FAILS PsiPhi(z)=z as required",
        "perturbed_partial_nilpotence": f"survived cap 12 (k={k})",
        "perturbed_partial_ideal": "not in (P) as required",
    }

    print("\n" + "-" * 72)
    print(f"tally: {r.tally}   total {len(r.results)} checks")
    path = _fresh(os.path.join("runs_synthesis", "RD-1-TRANSCRIBE.json"))
    with open(path, "w") as fh:
        json.dump({
            "block": "RD-1", "plan": "RD-OPS.md (A22)",
            "date": "2026-07-31", "mode": "I",
            "source": {
                "arxiv": "0903.4278v1",
                "authors": "Dubouloz, Moser-Jauslin, Poloni",
                "title": "Inequivalent embeddings of the Koras-Russell "
                         "cubic threefold",
                "section": "7", "pages": "11-12",
                "extraction": "pypdf 6.14.2, layout + default modes, "
                              "cross-read 2026-07-31",
                "attribution": "Phi, Psi, Delta, partial, partial_1, "
                               "partial_2 are DMJP's. No novelty claimed "
                               "on the derivation (W2).",
            },
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
