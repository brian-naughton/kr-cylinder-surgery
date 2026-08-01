#!/usr/bin/env python3
"""Shared machinery for the RF visit (RF-OPS.md, master A30) — the final
bounded generator-tracking sprint.

THE PICTURE (A28/A30, verified at source). Y_1 = Q_{(3,1), z^2-1/4} in
the Asok--Dubouloz--Ostvaer family (arXiv:2112.08241 Ex. 4.2.5),
A^1-equivalent to P^1 smash P^1, complex realisation S^4. The commuting
square's smooth route is  X x A^1 (contractible) --> Y_1 --> A^4, and
the (p+1)-arrow sigma_4 is an algebraic surgery killing Y_1's sphere.

THE FIBRATION THAT ORGANISES EVERYTHING. Let pi : Y_1 --> A^2_{t,q} be
the projection. Its fibre over (t,q) != (0,0) is a plane A^2; over the
origin it is the DOUBLED FIBRE

    F_0    = {p = 0,  t = q = 0}  =~ A^2_{Z,H}
    F_{-1} = {p = -1, t = q = 0}  =~ A^2_{Z,H}

which is the support datum A30 makes binding. The two-chart cover is
V_0 = Y_1 \\ F_{-1} and V_{-1} = Y_1 \\ F_0, with intersection
W = pi^{-1}(A^2 \\ {0}).

DIVISION OF LABOUR (RF-OPS §3, binding). Every ALGEBRAIC identity below
is machine-certified. The genuinely TOPOLOGICAL steps are written as
lemmas over those certified identities, at integral singular /
Borel-Moore level, each citing precisely. Nothing topological is
asserted beyond what the certified algebra plus standard integral
topology supports.

ATTRIBUTION: ADO arXiv:2112.08241 (the family, its A^1-homotopy type,
Prop 4.3.1); DMJP arXiv:0903.4278 §7 (p, q, Z); Dubouloz--Fasel
arXiv:1512.01933 and HKO arXiv:1409.1293 (A^1-contractibility of KR
threefolds of the first kind); DPO arXiv:1805.08959 (framework —
EXTENDED, never quoted for the result, per A30).
"""

from __future__ import annotations

import sympy as sp

from checker import is_in_ideal
from rd_common import E, GENS, KER_P, KER_Q, P, x, y, z, t, w
from rg_common import Pp, Qq, Tt, ZZ, TO_L, H_L, krull_dim, verify_carryover
from rm_common import R1, Xx, in_ideal, verify_rg_carryover
from rs_common import (CUSP_CLASS, Hh, L, QSURF, W as WEQ, Y_GENS, chi,
                       verify_rm_carryover)

#: the shear coordinates on A^5 carrying X x A^1 (RS-2)
XS, TS, ZS, WS, HS = sp.symbols("Xs Ts Zs Ws Hs")
SH_GENS = (XS, TS, ZS, WS, HS)
SHEAR = {x: XS, y: E(-HS - WS**2), z: E(ZS - XS * WS), t: TS, w: WS}

#: P and the invariants, read in the shear coordinates
P_SH = E(P.xreplace(SHEAR))                       # Xs(1-XsHs-2WsZs)+Zs^2+Ts^3
P1_SH = E(KER_P.xreplace(SHEAR) + 1)              # p + 1
Q_SH = E(KER_Q.xreplace(SHEAR))                   # q
#: the E_4 constraint: p + 1 = 0
CON = E(XS * HS + 2 * WS * ZS - 1)

#: the two components of the doubled fibre, as ideals in Y_1's ambient A^5
F0_IDEAL = [WEQ, Pp, Tt, Qq]                      # p = 0, t = q = 0
Fm1_IDEAL = [WEQ, E(Pp + 1), Tt, Qq]              # p = -1, t = q = 0


def fibre_eq(t0, q0):
    """The equation cutting pi^{-1}(t0,q0) inside A^3_{p,Z,H}."""
    return E(WEQ.xreplace({Tt: t0, Qq: q0}))


def verify_rs_carryover(check):
    """Re-verify the RS facts the RF blocks lean on (A24 re-load rule)."""
    check("RS carry-over: [Y_1] = L^4 + L^2 and chi_c(Y_1) = 2",
          lambda: chi(sp.expand(L**4 + L**2)) == 2)
    check("RS carry-over: Y_1 = V(W) is SMOOTH",
          lambda: krull_dim([WEQ] + [sp.diff(WEQ, g) for g in Y_GENS],
                            Y_GENS) == -1)
    check("RS carry-over: the shear identities for P, p+1 and q",
          lambda: E(P_SH - (XS * (1 - XS * HS - 2 * WS * ZS)
                            + ZS**2 + TS**3)) == 0
          and E(P1_SH - (1 - XS * HS - 2 * WS * ZS)) == 0
          and E(Q_SH - (2 * WS - 2 * HS * XS * WS - 4 * WS**2 * ZS
                        - HS * ZS)) == 0)
    check("RS carry-over: on E_4 the image coordinate is q = -H Z",
          lambda: in_ideal(E(Q_SH + HS * ZS), [CON], SH_GENS))
    check("RS carry-over: E_4 lies over the cusp Z^2 + t^3 = 0",
          lambda: in_ideal(E(P_SH - (ZS**2 + TS**3)), [CON], SH_GENS))
    return True
