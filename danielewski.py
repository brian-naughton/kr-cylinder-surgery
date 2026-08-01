#!/usr/bin/env python3
"""M0.5 true-positive control: explicit Danielewski isomorphism, certified.

W1 = {f1 = x*y1 - z1^2 + 1 = 0} and W2 = {f2 = x^2*y2 - z2^2 + 1 = 0} are
the classical Danielewski surfaces: W1 and W2 are non-isomorphic, yet their
cylinders are isomorphic. The maps below were derived in-session from first
principles (both surfaces are Ga-torsors over the affine line with doubled
origin; each admits a polynomial section over the other; trivialising the
torsor fibre product both ways yields the isomorphism):

  Phi*: O(W2 x C) -> O(W1 x C)          Psi*: O(W1 x C) -> O(W2 x C)
    x  |-> x                              x  |-> x
    z2 |-> z1(3 - z1^2)/2 + x^2 u         z1 |-> z2 + x v
    y2 |-> y1^2(x y1 - 3)/4               y1 |-> x y2 + 2 z2 v + x v^2
          + z1(3 - z1^2) u + x^2 u^2      u  |-> z2 y2/2 + (3/2) x y2 v
    v  |-> z1 y1 / 2 - x u                       + (3/2) z2 v^2 + x v^3 / 2

Certificate (complete for an iso of the two hypersurface cylinders):
  (W1) f2(Phi*x, Phi*y2, Phi*z2) in (f1)      [well-defined]
  (W2) f1(Psi*x, Psi*y1, Psi*z1) in (f2)      [well-defined]
  (C1) Phi* then Psi* fixes x, y1, z1, u mod (f1)   [Psi o Phi = id]
  (C2) Psi* then Phi* fixes x, y2, z2, v mod (f2)   [Phi o Psi = id]
Every membership is checked through BOTH independent twin checkers
(checker.is_in_ideal and checker_twin.member) per project discipline.

Exit 0 iff every check passes under both twins.
"""

import sys

import sympy as sp

import checker
import checker_twin

X = sp.Symbol('x')
Y1, Z1, U = sp.symbols('y1 z1 u')
Y2, Z2, V = sp.symbols('y2 z2 v')

F1 = X * Y1 - Z1**2 + 1          # W1: linear in y1, c = x,  gcd(x, z1^2-1) = 1
F2 = X**2 * Y2 - Z2**2 + 1       # W2: linear in y2, c = x^2, gcd likewise 1

PHI = {  # pullbacks of (x, y2, z2, v) into O(W1 x C)
    X: X,
    Z2: Z1 * (3 - Z1**2) / 2 + X**2 * U,
    Y2: Y1**2 * (X * Y1 - 3) / 4 + Z1 * (3 - Z1**2) * U + X**2 * U**2,
    V: Z1 * Y1 / 2 - X * U,
}
PSI = {  # pullbacks of (x, y1, z1, u) into O(W2 x C)
    X: X,
    Z1: Z2 + X * V,
    Y1: X * Y2 + 2 * Z2 * V + X * V**2,
    U: Z2 * Y2 / 2 + sp.Rational(3, 2) * X * Y2 * V
       + sp.Rational(3, 2) * Z2 * V**2 + X * V**3 / 2,
}


def both_member(g: sp.Expr, f: sp.Expr, y_sym: sp.Symbol) -> tuple[bool, bool]:
    g = sp.expand(sp.together(g) * 1)
    num, den = sp.fraction(sp.together(g))
    # clear the constant denominator (maps have /2, /4 coefficients)
    assert den.is_number, f'non-constant denominator: {den}'
    gg = sp.expand(num)
    return (checker.is_in_ideal(gg, f, y_sym), checker_twin.member(gg, f, y_sym))


def main() -> int:
    checks: list[tuple[str, bool]] = []

    def record(name: str, g: sp.Expr, f: sp.Expr, y_sym: sp.Symbol) -> None:
        r1, r2 = both_member(g, f, y_sym)
        checks.append((f'{name} [primary]', r1))
        checks.append((f'{name} [twin]', r2))

    record('(W1) f2 o Phi in (f1)', F2.subs(PHI, simultaneous=True), F1, Y1)
    record('(W2) f1 o Psi in (f2)', F1.subs(PSI, simultaneous=True), F2, Y2)

    # (C1): Psi o Phi = id_{W1xC}  <=>  (Psi o Phi)* = Phi* o Psi* = id on
    # O(W1 x C): for each generator g of O(W1 x C), take Psi*(g) (a
    # polynomial in x, y2, z2, v), then apply Phi* to that; result must be
    # congruent to g mod (f1).
    for var in (X, Y1, Z1, U):
        step = PSI[var] if var in PSI else var
        comp = step.subs(PHI, simultaneous=True)
        record(f'(C1) PhiPsi fixes {var} mod (f1)', comp - var, F1, Y1)

    # (C2): Phi o Psi = id_{W2xC}  <=>  Psi* o Phi* = id on O(W2xC).
    for var in (X, Y2, Z2, V):
        step = PHI[var] if var in PHI else var
        comp = step.subs(PSI, simultaneous=True)
        record(f'(C2) PsiPhi fixes {var} mod (f2)', comp - var, F2, Y2)

    width = max(len(n) for n, _ in checks)
    ok = True
    for name, good in checks:
        ok &= good
        print(f'{name:<{width}}  {"PASS" if good else "FAIL"}')
    print()
    print('DANIELEWSKI ISO CERTIFIED (both twins).' if ok
          else 'CERTIFICATE FAILURE.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
