"""DVM-BGK studies: shock tube from continuum to free-molecular, Couette flow vs DSMC."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from dvm import Couette, ShockTube, free_molecular_shock_tube
from dvm.euler_exact import State, sample

NX, T_END = 400, 0.15
X = (np.arange(NX) + 0.5) / NX
N0 = np.where(X < 0.5, 1.0, 0.125)
TT0 = np.where(X < 0.5, 1.0, 0.8)  # p_L = 1, p_R = 0.1 (Sod pressures), monatomic gas
KNS_TUBE = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 100.0]


def shock_tube():
    euler = sample(X, T_END, State(1.0, 0.0, 1.0), State(0.125, 0.0, 0.1), gamma=5 / 3)
    fm = free_molecular_shock_tube(X, T_END, 1.0, 1.0, 0.125, 0.8)
    print("Shock tube, t = 0.15: L1 density error against the two limits")
    print("| Kn | vs Euler (exact Riemann) | vs free-molecular (exact) |\n|---|---|---|")
    res = {}
    for kn in KNS_TUBE:
        s = ShockTube(kn, nx=NX)
        n, u, T = s.run(N0, 0 * X, TT0, T_END)
        res[kn] = (n, u, T)
        print(f"| {kn:g} | {np.abs(n - euler[0]).mean():.4f} | {np.abs(n - fm).mean():.4f} |")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(KNS_TUBE)))
    pe = euler[2]
    for ax, idx, name, ex in ((axes[0], 0, "density n", euler[0]), (axes[1], 1, "velocity u", euler[1]),
                              (axes[2], 2, "temperature T", pe / euler[0])):
        ax.plot(X, ex, "k-", lw=2.5, alpha=0.3, label="Euler (exact, γ = 5/3)")
        if idx == 0:
            ax.plot(X, fm, "k--", lw=1.5, label="free-molecular (exact)")
        for c, kn in zip(colors, KNS_TUBE):
            ax.plot(X, res[kn][idx], color=c, lw=1.2, label=f"DVM-BGK, Kn = {kn:g}")
        ax.set(xlabel="x", title=name, xlim=(0.15, 0.85))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sod shock tube with the BGK equation: continuum (Kn → 0) to free-molecular (Kn → ∞), t = 0.15")
    fig.tight_layout()
    fig.savefig("docs/shock_tube.png", dpi=120)
    plt.close(fig)

    # velocity-grid convergence in the free-molecular limit (ray effect)
    print("\nFree-molecular limit (Kn = 100), velocity-grid refinement:")
    for nv in (21, 41, 81, 161, 321):
        n, _, _ = ShockTube(100.0, nx=NX, nv=nv).run(N0, 0 * X, TT0, T_END)
        print(f"  nv = {nv:3d}: L1 error {np.abs(n - fm).mean():.2e}")
    return res


def shock_tube_gif():
    kns = [1e-4, 1e-2, 1.0, 100.0]
    snaps = {kn: [] for kn in kns}
    times = np.linspace(0, T_END, 46)[1:]
    for kn in kns:
        s = ShockTube(kn, nx=NX)
        k = [0]

        def cb(t, phi, psi, s=s, kn=kn, k=k):
            if k[0] < len(times) and t >= times[k[0]] - 1e-12:
                snaps[kn].append(s.moments(phi, psi)[0])
                k[0] += 1

        s.run(N0, 0 * X, TT0, T_END, callback=cb)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(kns)))
    lines = [ax.plot(X, N0, color=c, label=f"Kn = {kn:g}")[0] for c, kn in zip(colors, kns)]
    ax.set(xlabel="x", ylabel="n", ylim=(0, 1.08), title="")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    title = ax.set_title("")

    def update(i):
        for ln, kn in zip(lines, kns):
            ln.set_ydata(snaps[kn][i])
        title.set_text(f"Density, BGK shock tube, t = {times[i]:.3f}")
        return lines

    fig.tight_layout()
    FuncAnimation(fig, update, frames=len(times)).save("docs/shock_tube.gif", writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)


def couette_study():
    dsmc = np.loadtxt("data/dsmc_couette.csv", delimiter=",", skiprows=2)
    tau_fm = 0.4 / np.sqrt(2 * np.pi)
    kns = np.logspace(-1.3, 1.3, 14)
    rows = []
    print("\nCouette flow (U = 0.4): DVM-BGK vs hard-sphere DSMC")
    print("| Kn | DVM-BGK τ/τ_fm | DSMC τ/τ_fm | difference | iterations |\n|---|---|---|---|---|")
    for kn in dsmc[:, 0]:
        t0 = time.perf_counter()
        n, u, T, pxy, it = Couette(kn).solve(tol=1e-9)
        d = dsmc[dsmc[:, 0] == kn, 1][0]
        v = abs(pxy.mean()) / tau_fm
        print(f"| {kn:g} | {v:.4f} | {d:.4f} | {100 * (v / d - 1):+.1f} % | {it} |")
    for kn in kns:
        rows.append(abs(Couette(kn).solve(tol=1e-9)[3].mean()) / tau_fm)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogx(kns, rows, "C0-", lw=2, label="DVM-BGK (this repo)")
    ax.semilogx(dsmc[:, 0], dsmc[:, 1], "o", color="C3", ms=7, label="DSMC, hard spheres (dsmc-rarefied-gas)")
    ax.axhline(1, color="grey", lw=1, label="free-molecular")
    ax.set(xlabel="Kn = λ / H", ylabel="τ / τ_fm", ylim=(0, 1.05), title="Couette shear stress: kinetic vs particle method")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/couette_vs_dsmc.png", dpi=120)
    plt.close(fig)

    # the velocity distribution next to the wall: discontinuous at v_y = 0 in rarefied flow
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, kn in zip(axes, (1.0, 0.1)):
        c = Couette(kn, nv=40, vscale=0.9)
        n, u, T, pxy, _ = c.solve(tol=1e-9)
        # rebuild the distribution at the first cell with one more sweep's worth of data
        phi = _near_wall_distribution(c, n, u, T)
        dev = phi - c.equilibrium(n[:1], u[:1], T[:1])[0][0]
        m = np.abs(dev).max()
        im = ax.contourf(c.vx, c.vy, dev, levels=np.linspace(-m, m, 25), cmap="RdBu_r")
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", xlabel="v_x", ylabel="v_y",
               title=f"f − f_eq in the first cell, Kn = {kn:g}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Non-equilibrium near the bottom wall: a jump across v_y = 0 (wall-emitted vs incoming molecules)")
    fig.tight_layout()
    fig.savefig("docs/distribution.png", dpi=120)
    plt.close(fig)


def _near_wall_distribution(c, n, u, T):
    """Distribution in the first cell from one characteristic sweep with the converged moments."""
    nu = n * T / (c.mu_ref * np.sqrt(T))
    pe, _ = c.equilibrium(n, u, T)
    kappa = nu[:, None, None] * c.dy / np.abs(c.vy)
    att, avg = np.exp(-kappa), -np.expm1(-kappa) / kappa
    up = c.vy > 0
    # downward-moving molecules: sweep from the top wall to the first cell
    # (top-wall emission density is estimated from the local equilibrium: fine for visualisation)
    cur, _, _ = c._wall(pe[-1], -1, +c.U / 2)
    for j in range(len(c.y) - 1, 0, -1):
        cur = np.where(~up, cur * att[j] + pe[j] * (1 - att[j]), 0)
    down_val = pe[0] + (cur - pe[0]) * avg[0]
    bottom, _, _ = c._wall(pe[0], +1, -c.U / 2)
    up_val = pe[0] + (bottom - pe[0]) * avg[0]
    return np.where(up, up_val, down_val)


def main():
    shock_tube()
    shock_tube_gif()
    couette_study()


if __name__ == "__main__":
    main()
