"""Exact Riemann solver for the 1D Euler equations of an ideal gas.

Follows E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
Dynamics*, 3rd ed., Chapter 4: Newton iteration for the star-region
pressure, then self-similar sampling of the wave pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class State:
    rho: float
    u: float
    p: float


def _f(p, s: State, gamma):
    """Toro's pressure function f_K and its derivative."""
    c = np.sqrt(gamma * s.p / s.rho)
    if p > s.p:  # shock
        a = 2.0 / ((gamma + 1.0) * s.rho)
        b = (gamma - 1.0) / (gamma + 1.0) * s.p
        q = np.sqrt(a / (p + b))
        return (p - s.p) * q, q * (1.0 - 0.5 * (p - s.p) / (b + p))
    # rarefaction
    r = p / s.p
    f = 2.0 * c / (gamma - 1.0) * (r ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    df = r ** (-(gamma + 1.0) / (2.0 * gamma)) / (s.rho * c)
    return f, df


def star_state(left: State, right: State, gamma: float = 1.4, tol: float = 1e-12):
    """Return (p*, u*) in the star region."""
    cl = np.sqrt(gamma * left.p / left.rho)
    cr = np.sqrt(gamma * right.p / right.rho)
    if 2.0 * (cl + cr) / (gamma - 1.0) <= right.u - left.u:
        raise ValueError("initial data generate a vacuum")

    # primitive-variable (PVRS) initial guess, Toro eq. 4.47
    p = 0.5 * (left.p + right.p) - 0.125 * (right.u - left.u) * (left.rho + right.rho) * (cl + cr)
    p = max(tol, p)
    du = right.u - left.u
    for _ in range(100):
        fl, dfl = _f(p, left, gamma)
        fr, dfr = _f(p, right, gamma)
        p_new = max(tol, p - (fl + fr + du) / (dfl + dfr))
        if 2.0 * abs(p_new - p) / (p_new + p) < tol:
            p = p_new
            break
        p = p_new
    fl, _ = _f(p, left, gamma)
    fr, _ = _f(p, right, gamma)
    return p, 0.5 * (left.u + right.u) + 0.5 * (fr - fl)


def sample(x, t, left: State, right: State, x0: float = 0.5, gamma: float = 1.4):
    """Exact (rho, u, p) at positions x and time t > 0."""
    ps, us = star_state(left, right, gamma)
    g = gamma
    s_all = (np.asarray(x, dtype=float) - x0) / t
    rho = np.empty_like(s_all)
    u = np.empty_like(s_all)
    p = np.empty_like(s_all)

    for k, s in enumerate(s_all):
        if s <= us:  # left of the contact
            st, sign = left, 1.0
        else:
            st, sign = right, -1.0
        c = np.sqrt(g * st.p / st.rho)
        # work in a frame where the wave of interest is the "left" one (sign trick)
        un = sign * st.u
        usn = sign * us
        sn = sign * s
        if ps > st.p:  # shock
            ss = un - c * np.sqrt((g + 1) / (2 * g) * ps / st.p + (g - 1) / (2 * g))
            if sn <= ss:
                rho[k], u[k], p[k] = st.rho, st.u, st.p
            else:
                r = ps / st.p
                rho[k] = st.rho * (r + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * r + 1)
                u[k], p[k] = us, ps
        else:  # rarefaction
            cs = c * (ps / st.p) ** ((g - 1) / (2 * g))
            head, tail = un - c, usn - cs
            if sn <= head:
                rho[k], u[k], p[k] = st.rho, st.u, st.p
            elif sn >= tail:
                rho[k] = st.rho * (ps / st.p) ** (1 / g)
                u[k], p[k] = us, ps
            else:  # inside the fan
                fac = 2 / (g + 1) + (g - 1) / ((g + 1) * c) * (un - sn)
                rho[k] = st.rho * fac ** (2 / (g - 1))
                u[k] = sign * 2 / (g + 1) * (c + (g - 1) / 2 * un + sn)
                p[k] = st.p * fac ** (2 * g / (g - 1))
    return rho, u, p
