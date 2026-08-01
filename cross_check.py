#!/usr/bin/env python3
"""Cross-examination of the twin membership/iso checkers (M0 gate).

`checker.py` (localisation-substitution) and `checker_twin.py`
(pseudo-division) were implemented independently from the same written
spec, with no shared code, by different agents. This script — written by a
third party to both — subjects them to a common battery and demands exact
agreement, plus correctness wherever ground truth is known.

Battery:
  1. Membership agreement on structured samples for the Koras-Russell
     f = x + x^2 y + z^2 + t^3: multiples q*f (known IN), perturbed
     multiples q*f + r (known OUT), y-free nonzero polynomials (known OUT),
     0 (known IN), and high-degree stress cases.
  2. Same battery for the graph generator f' = y - (x^3 + z^2 t).
  3. Full iso-certificate agreement on the known-true graph isomorphism
     and on two deliberately broken variants (both checkers must flag the
     SAME broken pipeline as failing overall).

Exit 0 iff every agreement and every ground-truth check holds.
"""

import random
import sys

import sympy as sp

import checker
import checker_twin

X, Y, Z, T, W = sp.symbols('x y z t w')
A, B, C, D = sp.symbols('a b c d')

F_KR = X + X**2 * Y + Z**2 + T**3
F_GRAPH = Y - (X**3 + Z**2 * T)


def random_poly(rng: random.Random, degree: int = 3, terms: int = 6) -> sp.Expr:
    """Small random integer polynomial in x, y, z, t, w."""
    gens = [X, Y, Z, T, W]
    out = sp.Integer(0)
    for _ in range(terms):
        m = sp.Integer(rng.randint(-4, 4))
        for _ in range(rng.randint(0, degree)):
            m *= rng.choice(gens)
        out += m
    return sp.expand(out)


def main() -> int:
    rng = random.Random(20260722)
    checks: list[tuple[str, bool]] = []

    for f, tag in ((F_KR, 'KR'), (F_GRAPH, 'graph')):
        for i in range(6):
            q = random_poly(rng)
            g = sp.expand(q * f)
            r1 = checker.is_in_ideal(g, f, Y)
            r2 = checker_twin.member(g, f, Y)
            checks.append((f'[{tag}] q{i}*f: twins agree', r1 == r2))
            checks.append((f'[{tag}] q{i}*f: verdict IN', r1 is True))
        for r_extra in (sp.Integer(1), X, Y, W, X + Z**2 + T**3, X**5 * W**2):
            q = random_poly(rng)
            g = sp.expand(q * f + r_extra)
            r1 = checker.is_in_ideal(g, f, Y)
            r2 = checker_twin.member(g, f, Y)
            checks.append((f'[{tag}] q*f+{r_extra}: twins agree', r1 == r2))
            checks.append((f'[{tag}] q*f+{r_extra}: verdict OUT', r1 is False))
        checks.append((f'[{tag}] 0 in ideal (both)',
                       checker.is_in_ideal(sp.Integer(0), f, Y)
                       and checker_twin.member(sp.Integer(0), f, Y)))

    # stress: high-degree multiple with large coefficients
    q_big = sp.expand((X**4 - 3 * W**3 * Y**2 + 7 * T**5 - Z**6 + 11)**2)
    g_big = sp.expand(q_big * F_KR)
    checks.append(('stress q_big*f: twins agree IN',
                   checker.is_in_ideal(g_big, F_KR, Y) is True
                   and checker_twin.member(g_big, F_KR, Y) is True))
    g_big_off = sp.expand(g_big + X * W - 1)
    checks.append(('stress q_big*f + xw - 1: twins agree OUT',
                   checker.is_in_ideal(g_big_off, F_KR, Y) is False
                   and checker_twin.member(g_big_off, F_KR, Y) is False))

    # full certificate: known-true graph iso and two broken variants
    good_a4 = [A, A**3 + B**2 * C, B, C, D]
    good_cyl = [X, Z, T, W]
    bad_a4 = [A, A**3 + B**2 * C + 1, B, C, D]
    bad_cyl = [X + Z, Z, T, W]

    def overall(results: list) -> bool:
        return all(ok for _, ok in results)

    for name, a4, cyl, expect in (
            ('true graph iso', good_a4, good_cyl, True),
            ('broken Gy', bad_a4, good_cyl, False),
            ('broken A', good_a4, bad_cyl, False)):
        v1 = overall(checker.check_iso(a4, cyl, F_GRAPH))
        v2 = overall(checker_twin.certify_iso(a4, cyl, F_GRAPH))
        checks.append((f'cert [{name}]: twins agree', v1 == v2))
        checks.append((f'cert [{name}]: verdict {"PASS" if expect else "FAIL"}',
                       v1 is expect))

    width = max(len(n) for n, _ in checks)
    ok = True
    for name, good in checks:
        ok &= good
        print(f'{name:<{width}}  {"PASS" if good else "FAIL"}')
    print()
    n_agree = sum(1 for n, _ in checks if 'agree' in n)
    print(f'{len(checks)} checks ({n_agree} twin-agreement).')
    print('TWIN CHECKERS CROSS-VALIDATED.' if ok else 'CROSS-CHECK FAILURE.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
