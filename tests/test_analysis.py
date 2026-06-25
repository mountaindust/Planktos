'''Tests for Environment flow-analysis diagnostics: 2D vorticity and FTLE.

All cases use analytic velocity fields whose answers are known in closed form:
  * Vorticity of a linear field is exact under the np.gradient stencil (machine
    precision) on both uniform and non-uniform grids.
  * FTLE of a uniform flow is zero (no stretching); FTLE of a simple shear has a
    closed-form value from the Cauchy-Green eigenvalues.
Self-contained: flow arrays are built by hand. Folded in from test_vorticity.py
and the vorticity case of test_flow_points.py.
'''

import numpy as np
import pytest

import planktos


# --------------------------------------------------------------------------- #
#                            2D vorticity                                      #
# --------------------------------------------------------------------------- #

def _make_envir(x, y, vx, vy):
    '''2D Environment with a hand-built static flow. flow_points[i] indexes axis i
    of the flow arrays (the convention get_2D_vorticity uses; the bare flow=
    constructor path stores them transposed, so set them explicitly).'''
    envir = planktos.Environment(Lx=float(x[-1]), Ly=float(y[-1]), flow=[vx, vy])
    envir.flow_points = (x, y)
    return envir


@pytest.fixture(params=['uniform', 'nonuniform'])
def grid(request):
    nx, ny = 15, 11
    if request.param == 'uniform':
        x = np.linspace(0.0, 10.0, nx); y = np.linspace(0.0, 8.0, ny)
    else:
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0.0, 10.0, nx)); x[0] = 0.0; x[-1] = 10.0
        y = np.sort(rng.uniform(0.0, 8.0, ny)); y[0] = 0.0; y[-1] = 8.0
    X, Y = np.meshgrid(x, y, indexing='ij')
    return x, y, X, Y


def test_vorticity_solid_body_rotation(grid):
    '''v = (-y, x): vorticity = dv_y/dx - dv_x/dy = 1 - (-1) = 2 everywhere.'''
    x, y, X, Y = grid
    vort = _make_envir(x, y, -Y, X).get_2D_vorticity()
    assert vort.shape == X.shape
    assert np.allclose(vort, 2.0, atol=1e-10)


def test_vorticity_shear(grid):
    '''v = (a*y, 0): vorticity = -a everywhere.'''
    x, y, X, Y = grid
    a = 1.7
    assert np.allclose(_make_envir(x, y, a * Y, np.zeros_like(Y)).get_2D_vorticity(),
                       -a, atol=1e-10)


def test_vorticity_general_linear(grid):
    '''v = (a*y, b*x): vorticity = b - a everywhere.'''
    x, y, X, Y = grid
    a, b = 3.0, 2.0
    assert np.allclose(_make_envir(x, y, a * Y, b * X).get_2D_vorticity(),
                       b - a, atol=1e-10)


def test_vorticity_on_nonsquare_constructor_flow():
    '''get_2D_vorticity must work on a non-square flow built via the constructor
    (regression: it previously raised due to swapped flow_points).'''
    nx, ny = 12, 9
    x = np.linspace(0, 10, nx); y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[-Y, X])   # solid-body rotation
    vort = envir.get_2D_vorticity()
    assert vort.shape == (nx, ny)
    assert np.allclose(vort, 2.0, atol=1e-10)


# --------------------------------------------------------------------------- #
#                                  FTLE                                        #
# --------------------------------------------------------------------------- #

def test_FTLE_uniform_flow_is_zero():
    # A uniform flow translates every particle identically: no stretching.
    envir = planktos.Environment(Lx=10, Ly=10,
                                 flow=[np.full((11, 11), 1.0), np.full((11, 11), 0.5)])
    envir.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05)
    assert np.nanmax(np.abs(envir.FTLE_largest)) < 1e-8


def test_FTLE_simple_shear_closed_form():
    # u = (a*y, 0). Flow-map gradient F = [[1, aT],[0,1]], Cauchy-Green C = F^T F,
    # largest eigenvalue lam = (2+(aT)^2 + sqrt((2+(aT)^2)^2 - 4))/2, and the
    # largest FTLE = ln(sqrt(lam))/T -- spatially constant for a linear field.
    a, T = 1.0, 1.0
    n = 21
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[a * Y, np.zeros_like(Y)])
    envir.calculate_FTLE(grid_dim=(8, 8), T=T, dt=0.05)

    aT = a * T
    lam = (2 + aT**2 + np.sqrt((2 + aT**2)**2 - 4)) / 2
    expected = np.log(np.sqrt(lam)) / T                  # = ln(golden ratio) ~ 0.4812
    assert np.nanmax(envir.FTLE_largest) == pytest.approx(expected, abs=1e-3)


@pytest.mark.xfail(strict=True, reason="BUG-FTLE-BACKWARD: backward-time FTLE is "
                   "not implemented. calculate_FTLE integrates only forward "
                   "(while current_time < T from current_time=t0), so T<0 runs no "
                   "steps and raises IndexError on empty pos_history. The "
                   "documented workaround -- negating FTLE_smallest -- is also "
                   "mathematically wrong: for incompressible flow FTLE_smallest is "
                   "identically -FTLE_largest (no independent backward info); in "
                   "general the smallest forward-time exponent is a contraction "
                   "rate, not the backward-time FTLE. A correct fix integrates the "
                   "flow map backward in time.")
def test_backward_FTLE_produces_a_field():
    a, n = 1.0, 21
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[a * Y, np.zeros_like(Y)])
    envir.calculate_FTLE(grid_dim=(8, 8), t0=0, T=-1.0, dt=0.05)
    assert np.isfinite(np.asarray(envir.FTLE_largest)).any(), "no backward FTLE field produced"
