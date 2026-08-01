#!/usr/bin/env python3
"""Independent twin certificate checker for the Koras-Russell cylinder problem.

This module provides exact (rational-arithmetic) certificate checks for the
cylinder ring of a "y-linear" generator

    f = c(x, z, t) * y + d(x, z, t)   in   Q[x, y, z, t]

with the *precondition* gcd(c, d) = 1 in Q[x, z, t].  Under this precondition
the ideal (f) is prime: f is linear (degree 1) in y and primitive in y (its two
coefficients c and d share no common factor), hence irreducible in
Q[x, z, t][y] = Q[x, y, z, t] by Gauss's lemma, so (f) is prime.

The default generator is the Koras-Russell cubic

    f = x + x**2 * y + z**2 + t**3   =>   c = x**2,  d = x + z**2 + t**3.

All arithmetic here is exact symbolic/rational; no floating point is ever used.
"""

from __future__ import annotations

import sympy as sp
from sympy import Poly, Symbol, expand, sympify

# Canonical symbols for the cylinder ring Q[x, y, z, t, w].
x, y, z, t, w = sp.symbols("x y z t w")

# Canonical coordinate symbols for the coordinate ring Q[a, b, c, d] of C^4.
a, b, c, d = sp.symbols("a b c d")

# Default generator: the Koras-Russell cubic (does not involve w).
DEFAULT_F = x + x**2 * y + z**2 + t**3


def _poly_gens(*exprs: sp.Expr, main: Symbol) -> list[Symbol]:
    """Collect generators for a Poly, placing ``main`` first.

    Args:
        *exprs: Sympy expressions whose free symbols are pooled.
        main: The distinguished (main) variable, forced to index 0 so that
            pseudo-division proceeds with respect to it.

    Returns:
        Ordered list of generator symbols with ``main`` first and the remaining
        symbols sorted by name for determinism.
    """
    others = set()
    for e in exprs:
        others |= set(sympify(e).free_symbols)
    others.discard(main)
    ordered = [main] + sorted(others, key=lambda s: s.name)
    return ordered


def member(g: sp.Expr, f: sp.Expr, y_sym: Symbol) -> bool:
    """Decide whether ``g`` lies in the ideal (f) via pseudo-division in y.

    The test uses the pseudo-remainder mechanism rather than substitution.
    Viewing g and f as polynomials in ``y_sym`` over Q[x, z, t, w], the
    pseudo-division identity is

        c**k * g = q * f + r ,   deg_y(r) < deg_y(f) = 1 ,

    where c = lc_y(f) is the leading coefficient of f in y and r = prem(g, f, y)
    is the pseudo-remainder.  We return True iff r vanishes identically.

    Validity (given the module precondition gcd(c, d) = 1, so (f) is prime):
      * If r == 0 then c**k * g = q * f in (f).  Since (f) is prime and
        c = lc_y(f) is not a multiple of f (deg_y(c) = 0 < 1 = deg_y(f), and
        c != 0), c is not in (f); primeness of (f) then forces g in (f).
      * Conversely, if g in (f) then g = q * f, whence
        c**k * g = (c**k * q) * f and the unique pseudo-remainder r is 0.
    Hence r == 0  <=>  g in (f).

    The final decision is exact polynomial vanishing (Poly.is_zero); no
    numerical evaluation is performed.

    Args:
        g: Candidate polynomial (sympy expression) over Q[x, y, z, t, w].
        f: The y-linear generator c*y + d with gcd(c, d) = 1.
        y_sym: The variable in which the pseudo-division is carried out.

    Returns:
        True iff ``g`` is a member of the ideal (f).
    """
    g = sympify(g)
    f = sympify(f)
    gens = _poly_gens(g, f, main=y_sym)
    gp = Poly(g, *gens)
    fp = Poly(f, *gens)
    remainder = gp.prem(fp)
    return remainder.is_zero


def certify_iso(
    A4_to_cyl: list[sp.Expr],
    cyl_to_A4: list[sp.Expr],
    f: sp.Expr,
) -> list[tuple[str, bool]]:
    """Verify a complete isomorphism certificate cylinder <-> C^4.

    The map data describe a candidate ring isomorphism between the coordinate
    ring Q[a, b, c, d] of affine 4-space and the cylinder ring
    Q[x, y, z, t, w]/(f):

      * ``cyl_to_A4`` = [A, B, C, D]: four polynomials in x, y, z, t, w giving
        the pullbacks of the coordinates a, b, c, d (cylinder -> C^4).
      * ``A4_to_cyl`` = [Gx, Gy, Gz, Gt, Gw]: five polynomials in a, b, c, d
        giving the pullbacks of x, y, z, t, w (C^4 -> cylinder).

    Two ring maps are mutually inverse iff they compose to the identity on
    generators on both sides.  Because a ring isomorphism of coordinate rings
    is exactly an isomorphism of the affine varieties, the following purely
    algebraic identities constitute a COMPLETE certificate -- no Jacobian rank
    or smoothness side-conditions are required.

    Checks performed (each returned as (name, passed)):
      (i)  f(Gx, Gy, Gz, Gt) == 0 identically in Q[a, b, c, d]: the image of
           the cylinder lands on the hypersurface (well-definedness of the map
           into the cylinder ring).  The ideal upstairs (in a, b, c, d) is (0),
           so this is an exact equality.
      (ii) For P in {A, B, C, D}: P(Gx, Gy, Gz, Gt, Gw) equals exactly the
           corresponding coordinate a, b, c, d.  These are exact polynomial
           identities in Q[a, b, c, d] (its ideal is (0)).
      (iii) For each coordinate x, y, z, t, w with pullback G in
           {Gx, Gy, Gz, Gt, Gw}: G(A, B, C, D) equals the coordinate modulo
           (f), i.e. member(G(A, B, C, D) - coord, f, y) is True.

    Args:
        A4_to_cyl: [Gx, Gy, Gz, Gt, Gw], polynomials in a, b, c, d.
        cyl_to_A4: [A, B, C, D], polynomials in x, y, z, t, w.
        f: The y-linear generator of the cylinder relation.

    Returns:
        List of (check_name, passed) tuples; the certificate holds iff every
        entry is True.
    """
    Gx, Gy, Gz, Gt, Gw = (sympify(e) for e in A4_to_cyl)
    A_, B_, C_, D_ = (sympify(e) for e in cyl_to_A4)
    f = sympify(f)

    results: list[tuple[str, bool]] = []

    # (i) f pulled back to Q[a,b,c,d] must vanish identically.
    f_pulled = f.subs({x: Gx, y: Gy, z: Gz, t: Gt}, simultaneous=True)
    results.append(("(i) f(Gx,Gy,Gz,Gt) == 0", expand(f_pulled) == 0))

    # (ii) cyl -> A4 -> cyl composed the C^4 way: A(G..) == a, etc. (exact).
    sub_to_cyl = {x: Gx, y: Gy, z: Gz, t: Gt, w: Gw}
    for name, poly, target in (
        ("A", A_, a),
        ("B", B_, b),
        ("C", C_, c),
        ("D", D_, d),
    ):
        composed = poly.subs(sub_to_cyl, simultaneous=True)
        ok = expand(composed - target) == 0
        results.append((f"(ii) {name}(G..) == {target}", ok))

    # (iii) A4 -> cyl -> A4 composed the cylinder way: G(A,B,C,D) == coord mod f.
    sub_to_a4 = {a: A_, b: B_, c: C_, d: D_}
    for name, poly, coord in (
        ("Gx", Gx, x),
        ("Gy", Gy, y),
        ("Gz", Gz, z),
        ("Gt", Gt, t),
        ("Gw", Gw, w),
    ):
        composed = poly.subs(sub_to_a4, simultaneous=True)
        ok = member(composed - coord, f, y)
        results.append((f"(iii) {name}(A,B,C,D) == {coord} mod (f)", ok))

    return results
