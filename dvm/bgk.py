"""Discrete velocity method (DVM) for the BGK model of the Boltzmann equation.

Units: m = k = 1, so temperature T is the variance of each velocity component.

    df/dt + v . grad_x f = nu (f_eq - f),   nu = n T / mu(T),   mu(T) = mu_ref sqrt(T) (hard-sphere-like)

Two reduced problems (Chu reduction: integrate out the velocity components that the flow
does not depend on, carrying their energy in a second distribution psi):

* 1D shock tube:  f(x, v_x)  ->  phi = int f dv_y dv_z,   psi = int (v_y^2 + v_z^2) f dv_y dv_z
* 1D Couette:     f(y, v_x, v_y)  ->  phi = int f dv_z,   psi = int v_z^2 f dv_z

Velocity space uses Gauss-Hermite quadrature, so moments of Maxwellians are integrated exactly.
"""

from __future__ import annotations

import numpy as np

# The hard-sphere viscosity written with the hard-sphere mean free path lambda at n = T = 1:
# mu = 5/(16 d^2) sqrt(1/pi) and lambda = 1/(sqrt(2) pi d^2)  =>  mu = (5 sqrt(2 pi) / 16) lambda.
MU_PER_LAMBDA = 5 * np.sqrt(2 * np.pi) / 16


def gauss_hermite(n, scale=1.0):
    """Nodes/weights for int g(v) dv on (-inf, inf), with the weight exp(-v^2 / (2 scale^2)) absorbed."""
    x, w = np.polynomial.hermite.hermgauss(n)
    v = np.sqrt(2) * scale * x
    return v, w * np.sqrt(2) * scale * np.exp(x**2)


# ---------------------------------------------------------------- 1D shock tube (velocity v_x)


def uniform_velocities(n, vmax):
    """Uniform grid with trapezoidal weights: spectrally accurate for Maxwellians that decay by +-vmax."""
    v = np.linspace(-vmax, vmax, n)
    w = np.full(n, v[1] - v[0])
    w[[0, -1]] *= 0.5
    return v, w


class ShockTube:
    def __init__(self, kn, nx=400, nv=161, vmax=7.0, x0=0.0, x1=1.0):
        self.kn = kn
        self.x = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        self.dx = (x1 - x0) / nx
        self.v, self.w = uniform_velocities(nv, vmax)
        self.mu_ref = MU_PER_LAMBDA * kn * (x1 - x0)

    def equilibrium(self, n, u, T):
        v = self.v[None, :]
        phi = n[:, None] / np.sqrt(2 * np.pi * T[:, None]) * np.exp(-((v - u[:, None]) ** 2) / (2 * T[:, None]))
        return phi, 2 * T[:, None] * phi  # psi carries the two transverse degrees of freedom

    def moments(self, phi, psi):
        n = phi @ self.w
        u = (phi @ (self.w * self.v)) / n
        E = 0.5 * ((phi @ (self.w * self.v**2)) + psi @ self.w)  # total energy density
        T = (2 * E / n - u**2) / 3
        return n, u, T

    def _transport(self, f):
        """-v df/dx with second-order upwind (MUSCL, minmod) and zero-gradient ends."""
        g = np.pad(f, ((2, 2), (0, 0)), mode="edge")
        d = np.diff(g, axis=0)
        slope = np.where(d[:-1] * d[1:] > 0, np.sign(d[1:]) * np.minimum(np.abs(d[:-1]), np.abs(d[1:])), 0.0)
        left = g[1:-1] + 0.5 * slope  # value at right face of cells 1..nx+2 (padded indexing)
        right = g[1:-1] - 0.5 * slope
        vpos = self.v > 0
        face = np.where(vpos[None, :], left[:-1], right[1:])  # upwind value at faces between cells
        flux = self.v[None, :] * face
        return -(flux[1:] - flux[:-1]) / self.dx

    def run(self, n0, u0, T0, t_end, cfl=0.5, callback=None):
        phi, psi = self.equilibrium(n0, u0, T0)
        dt0 = cfl * self.dx / np.abs(self.v).max()
        t = 0.0
        while t < t_end - 1e-14:
            dt = min(dt0, t_end - t)
            # explicit transport, then implicit BGK relaxation (moments are conserved by relaxation,
            # so the post-transport moments define f_eq exactly: an asymptotic-preserving IMEX step)
            phi = phi + dt * self._transport(phi)
            psi = psi + dt * self._transport(psi)
            n, u, T = self.moments(phi, psi)
            nu = (n * T / (self.mu_ref * np.sqrt(T)))[:, None]
            pe, se = self.equilibrium(n, u, T)
            phi = (phi + dt * nu * pe) / (1 + dt * nu)
            psi = (psi + dt * nu * se) / (1 + dt * nu)
            t += dt
            if callback is not None:
                callback(t, phi, psi)
        return self.moments(phi, psi)


def free_molecular_shock_tube(x, t, nL, TL, nR, TR, x0=0.5):
    """Collisionless solution: each side's Maxwellian streams freely (density only)."""
    from scipy.special import erfc

    return 0.5 * nL * erfc((x - x0) / (np.sqrt(2 * TL) * t)) + 0.5 * nR * erfc(-(x - x0) / (np.sqrt(2 * TR) * t))


# ---------------------------------------------------------------- planar Couette flow (v_x, v_y)


class Couette:
    """Steady Couette flow between diffuse walls y = 0 (u = -U/2) and y = H (u = +U/2), T_w = 1."""

    def __init__(self, kn, U=0.4, H=1.0, ny=100, nv=28, vscale=1.0):
        self.kn, self.U, self.H = kn, U, H
        self.y = (np.arange(ny) + 0.5) * H / ny
        self.dy = H / ny
        v, w = gauss_hermite(nv, vscale)
        self.vx, self.vy = np.meshgrid(v, v, indexing="ij")
        self.w = np.outer(w, w)
        self.mu_ref = MU_PER_LAMBDA * kn * H

    def equilibrium(self, n, u, T):
        """Reduced Maxwellian on (ny, nv, nv): phi and psi = T phi (one hidden degree of freedom)."""
        n, u, T = (np.asarray(a)[:, None, None] for a in (n, u, T))
        phi = n / (2 * np.pi * T) * np.exp(-((self.vx - u) ** 2 + self.vy**2) / (2 * T))
        return phi, T * phi

    def moments(self, phi, psi):
        W = self.w
        n = (phi * W).sum(axis=(1, 2))
        u = (phi * W * self.vx).sum(axis=(1, 2)) / n
        v = (phi * W * self.vy).sum(axis=(1, 2)) / n
        E = 0.5 * ((phi * W * (self.vx**2 + self.vy**2)).sum(axis=(1, 2)) + (psi * W).sum(axis=(1, 2)))
        T = (2 * E / n - u**2 - v**2) / 3
        pxy = (phi * W * (self.vx - u[:, None, None]) * (self.vy - v[:, None, None])).sum(axis=(1, 2))
        return n, u, T, pxy

    def _wall(self, phi_in, vy_sign, u_wall):
        """Diffuse reflection: emitted Maxwellian at T_w = 1 whose density balances the incoming flux."""
        up = self.vy * vy_sign > 0
        flux_in = (phi_in * self.w * np.abs(self.vy) * ~up).sum()
        pe, se = self.equilibrium([1.0], [u_wall], [1.0])
        pe, se = pe[0], se[0]
        flux_unit = (pe * self.w * np.abs(self.vy) * up).sum()
        nw = flux_in / flux_unit
        return nw * pe, nw * se, up

    def solve(self, tol=1e-9, max_iter=20_000):
        """Source iteration: sweep characteristics with exact exponential integration per cell,
        then update the BGK equilibrium from the new moments."""
        ny = len(self.y)
        n = np.ones(ny)
        u = self.U * (self.y / self.H - 0.5)
        T = np.ones(ny)
        phi, psi = self.equilibrium(n, u, T)
        vy_abs = np.abs(self.vy)
        up = self.vy > 0
        for it in range(1, max_iter + 1):
            nu = n * T / (self.mu_ref * np.sqrt(T))
            pe, se = self.equilibrium(n, u, T)
            kappa = nu[:, None, None] * self.dy / vy_abs  # optical thickness of one cell
            att = np.exp(-kappa)
            avg = -np.expm1(-kappa) / kappa  # (1 - e^-kappa) / kappa, -> 1 as kappa -> 0
            new_phi = np.zeros_like(phi)
            new_psi = np.zeros_like(psi)
            # upward-moving particles start at the bottom wall, downward-moving ones at the top wall
            sweeps = ((up, range(ny), self._wall(phi[0], +1, -self.U / 2)),
                      (~up, range(ny - 1, -1, -1), self._wall(phi[-1], -1, +self.U / 2)))
            for mask, cells, (in_p, in_s, _) in sweeps:
                for j in cells:
                    # exact solution of v_y df/dy = nu (f_eq - f) across the cell (f_eq frozen):
                    # cell average and outflow
                    new_phi[j] = np.where(mask, pe[j] + (in_p - pe[j]) * avg[j], new_phi[j])
                    new_psi[j] = np.where(mask, se[j] + (in_s - se[j]) * avg[j], new_psi[j])
                    in_p = np.where(mask, in_p * att[j] + pe[j] * (1 - att[j]), 0)
                    in_s = np.where(mask, in_s * att[j] + se[j] * (1 - att[j]), 0)
            # the total mass between the plates is fixed (diffuse walls have zero net flux);
            # renormalise so that round-off and iteration error cannot make it drift
            scale = 1.0 / ((new_phi * self.w).sum(axis=(1, 2)).mean())
            new_phi *= scale
            new_psi *= scale
            n_new, u_new, T_new, _ = self.moments(new_phi, new_psi)
            change = max(np.abs(u_new - u).max(), np.abs(T_new - T).max(), np.abs(n_new - n).max())
            phi, psi, n, u, T = new_phi, new_psi, n_new, u_new, T_new
            if change < tol:
                break
        n, u, T, pxy = self.moments(phi, psi)
        return n, u, T, pxy, it
