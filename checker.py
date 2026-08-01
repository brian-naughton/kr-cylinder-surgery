#!/usr/bin/env python3
"""Primary certificate checker for the Koras-Russell cylinder isomorphism problem.

Setting
-------
Let ``f = c(x, z, t) * y + d(x, z, t)`` be a *y-linear* generator in the
polynomial ring ``Q[x, y, z, t]`` with ``gcd(c, d) = 1`` in ``Q[x, z, t]``.
The default ``f`` is the Koras-Russell cubic

    f = x + x**2 * y + z**2 + t**3      (so c = x**2, d = x + z**2 + t**3).

The *cylinder ring* is ``Q[x, y, z, t, w] / (f)`` (note ``f`` does not involve
``w``; ``w`` is a free cylinder coordinate).

Why the localisation-substitution test is valid
------------------------------------------------
Because ``f`` is linear in ``y`` with content ``gcd(c, d) = 1``, the ideal
``(f)`` is prime and the quotient embeds into the localisation of the ring at
``c``. Over the fraction field ``K = Q(x, z, t)`` the polynomial ``f = c*y + d``
factors (up to the unit ``c``) as ``y - (-d/c)``, so a polynomial ``g`` lies in
``(f)`` in ``K[y, w]`` iff it vanishes under ``y -> -d/c``. Gauss's lemma,
applied because ``f`` is primitive (``gcd(c, d) = 1``), upgrades divisibility in
``K[y, w]`` to divisibility in ``Q[x, z, t][y, w]``. Hence ``g in (f)`` iff the
polynomial obtained from ``g`` by ``y -> -d/c`` and clearing the ``c``
denominators is *identically zero*. All arithmetic below is exact and rational.
"""

import sympy

# Standard cylinder coordinate symbols. ``f`` and the cylinder-side maps
# (cyl_to_A4) are expressed in these; ``f`` never involves ``w``.
X, Y, Z, T, W = sympy.symbols("x y z t w")

# Standard C^4 coordinate symbols. The A4-side maps (A4_to_cyl) are expressed
# in these; note ``c`` and ``d`` here are C^4 coordinates, unrelated to the
# generator components c(x, z, t), d(x, z, t) used internally by is_in_ideal.
A, B, C, D = sympy.symbols("a b c d")

# Default generator: the Koras-Russell cubic.
KR_F = X + X**2 * Y + Z**2 + T**3


def is_in_ideal(g: sympy.Expr, f: sympy.Expr, y_sym: sympy.Symbol) -> bool:
    """Decide whether ``g`` lies in the ideal ``(f)`` via localisation-substitution.

    The generator ``f`` must be linear in ``y_sym`` (``f = c*y + d`` with ``c, d``
    free of ``y_sym``) and primitive (``gcd(c, d) = 1`` in the remaining
    variables). Under these preconditions ``g in (f)`` iff substituting
    ``y_sym -> -d/c`` into ``g`` and multiplying by ``c**deg_y(g)`` (to clear the
    ``c`` denominators) yields the identically-zero polynomial. The final vanishing
    test is exact polynomial equality -- never numerical.

    Args:
        g: Polynomial (sympy expression) in the cylinder variables to test.
        f: The y-linear generator ``c*y + d`` of the ideal.
        y_sym: The distinguished variable in which ``f`` is linear.

    Returns:
        True iff ``g`` is a member of the ideal ``(f)``.
    """
    g = sympy.expand(g)
    if g == 0:
        return True

    c_coeff = sympy.expand(f.coeff(y_sym, 1))
    d_coeff = sympy.expand(f.coeff(y_sym, 0))

    deg = sympy.degree(g, gen=y_sym)
    deg = int(deg) if deg >= 0 else 0

    substituted = g.subs(y_sym, -d_coeff / c_coeff)
    cleared = sympy.cancel(substituted * c_coeff**deg)
    return sympy.expand(cleared) == 0


def check_iso(
    A4_to_cyl: list[sympy.Expr],
    cyl_to_A4: list[sympy.Expr],
    f: sympy.Expr,
) -> list[tuple[str, bool]]:
    """Verify a complete ring isomorphism ``Q[a,b,c,d] ~= Q[x,y,z,t,w]/(f)``.

    This is a COMPLETE isomorphism certificate: mutual-inverse ring homomorphisms
    between the coordinate ring of ``C^4`` and the cylinder ring pin down an
    isomorphism outright, so no Jacobian or smoothness side-conditions are needed.

    Args:
        A4_to_cyl: ``[Gx, Gy, Gz, Gt, Gw]`` -- five polynomials in ``a, b, c, d``,
            the pullbacks of ``x, y, z, t, w`` under the C^4 -> cylinder map.
        cyl_to_A4: ``[A, B, C, D]`` -- four polynomials in ``x, y, z, t, w``, the
            pullbacks of ``a, b, c, d`` under the cylinder -> C^4 map.
        f: The y-linear generator of the cylinder ideal.

    Returns:
        A named PASS/FAIL result for every check:
          - "f(G) == 0": ``f(Gx, Gy, Gz, Gt) = 0`` identically in ``Q[a,b,c,d]``.
          - "A4->cyl->A4: {a,b,c,d}": each C^4 pullback composed with the cylinder
            pullbacks returns the corresponding C^4 coordinate exactly.
          - "cyl->A4->cyl: {x,y,z,t,w}": each cylinder coordinate, pulled back and
            pushed forward, returns the coordinate modulo ``(f)``.
    """
    g_x, g_y, g_z, g_t, g_w = A4_to_cyl
    map_a, map_b, map_c, map_d = cyl_to_A4

    results: list[tuple[str, bool]] = []

    # (i) The image of C^4 lands on the hypersurface f = 0.
    on_hypersurface = sympy.expand(
        f.subs({X: g_x, Y: g_y, Z: g_z, T: g_t})
    ) == 0
    results.append(("f(G) == 0", on_hypersurface))

    # (ii) A4 -> cyl -> A4 is the identity on a, b, c, d (ideal on this side is 0).
    cyl_subs = {X: g_x, Y: g_y, Z: g_z, T: g_t, W: g_w}
    for name, expr, target in (
        ("A4->cyl->A4: a", map_a, A),
        ("A4->cyl->A4: b", map_b, B),
        ("A4->cyl->A4: c", map_c, C),
        ("A4->cyl->A4: d", map_d, D),
    ):
        recovered = sympy.expand(expr.subs(cyl_subs))
        results.append((name, recovered - target == 0))

    # (iii) cyl -> A4 -> cyl is the identity on x, y, z, t, w modulo (f).
    a4_subs = {A: map_a, B: map_b, C: map_c, D: map_d}
    for name, expr, target in (
        ("cyl->A4->cyl: x", g_x, X),
        ("cyl->A4->cyl: y", g_y, Y),
        ("cyl->A4->cyl: z", g_z, Z),
        ("cyl->A4->cyl: t", g_t, T),
        ("cyl->A4->cyl: w", g_w, W),
    ):
        recovered = sympy.expand(expr.subs(a4_subs))
        results.append((name, is_in_ideal(recovered - target, f, Y)))

    return results
