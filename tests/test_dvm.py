import numpy as np
import pytest

from dvm import Couette, ShockTube, free_molecular_shock_tube, gauss_hermite
from dvm.bgk import uniform_velocities
from dvm.euler_exact import State, sample

NX = 200
X = (np.arange(NX) + 0.5) / NX
N0 = np.where(X < 0.5, 1.0, 0.125)
T0 = np.where(X < 0.5, 1.0, 0.8)


@pytest.mark.parametrize("grid", ["hermite", "uniform"])
def test_velocity_quadrature_integrates_maxwellian_moments(grid):
    v, w = gauss_hermite(40, 1.0) if grid == "hermite" else uniform_velocities(161, 7.0)
    M = np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi)
    assert np.allclose([w @ M, w @ (v**2 * M), w @ (v**4 * M)], [1, 1, 3], atol=1e-10)


def test_shock_tube_conserves_mass():
    n, _, _ = ShockTube(0.01, nx=NX).run(N0, 0 * X, T0, 0.1)
    assert n.mean() == pytest.approx(N0.mean(), rel=1e-6)


def test_continuum_limit_is_euler():
    n, _, _ = ShockTube(1e-4, nx=NX).run(N0, 0 * X, T0, 0.15)
    re, _, _ = sample(X, 0.15, State(1.0, 0.0, 1.0), State(0.125, 0.0, 0.1), gamma=5 / 3)
    assert np.abs(n - re).mean() < 0.01


def test_free_molecular_limit_is_exact():
    n, _, _ = ShockTube(100.0, nx=NX).run(N0, 0 * X, T0, 0.15)
    assert np.abs(n - free_molecular_shock_tube(X, 0.15, 1.0, 1.0, 0.125, 0.8)).mean() < 1e-3


def test_couette_stress_uniform_and_near_free_molecular():
    n, u, T, pxy, it = Couette(10.0, ny=50).solve(tol=1e-9)
    assert pxy.std() < 1e-6 * abs(pxy.mean())  # momentum conservation
    assert abs(pxy.mean()) / (0.4 / np.sqrt(2 * np.pi)) == pytest.approx(0.944, abs=0.01)
    assert n.mean() == pytest.approx(1.0, rel=1e-12)
