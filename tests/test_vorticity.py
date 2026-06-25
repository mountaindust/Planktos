'''Tests for Environment.get_2D_vorticity (vectorized via np.gradient).

Uses linear velocity fields whose analytic vorticity is constant and known. The
finite-difference stencil is exact for linear fields, so these pass to machine
precision on BOTH uniform and non-uniform grids. Self-contained: flow arrays are
built by hand, no external data needed.
'''

import numpy as np
import pytest
import planktos


def _make_envir(x, y, vx, vy):
    '''2D Environment with a hand-built static flow. flow_points[i] is set to
    index axis i of the flow arrays (the convention the real data loaders and
    get_2D_vorticity use; the bare flow= constructor path stores them
    transposed).'''
    envir = planktos.Environment(Lx=float(x[-1]), Ly=float(y[-1]), flow=[vx, vy])
    envir.flow_points = (x, y)
    return envir


@pytest.fixture(params=['uniform', 'nonuniform'])
def grid(request):
    nx, ny = 15, 11
    if request.param == 'uniform':
        x = np.linspace(0.0, 10.0, nx)
        y = np.linspace(0.0, 8.0, ny)
    else:
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0.0, 10.0, nx)); x[0] = 0.0; x[-1] = 10.0
        y = np.sort(rng.uniform(0.0, 8.0, ny)); y[0] = 0.0; y[-1] = 8.0
    X, Y = np.meshgrid(x, y, indexing='ij')   # axis 0 -> x, axis 1 -> y
    return x, y, X, Y


def test_vorticity_solid_body_rotation(grid):
    '''v = (-y, x): vorticity = dv_y/dx - dv_x/dy = 1 - (-1) = 2 everywhere.'''
    x, y, X, Y = grid
    envir = _make_envir(x, y, -Y, X)
    vort = envir.get_2D_vorticity()
    assert vort.shape == X.shape
    assert np.allclose(vort, 2.0, atol=1e-10)


def test_vorticity_shear(grid):
    '''v = (a*y, 0): vorticity = -a everywhere.'''
    x, y, X, Y = grid
    a = 1.7
    envir = _make_envir(x, y, a * Y, np.zeros_like(Y))
    vort = envir.get_2D_vorticity()
    assert np.allclose(vort, -a, atol=1e-10)


def test_vorticity_general_linear(grid):
    '''v = (a*y, b*x): vorticity = b - a everywhere.'''
    x, y, X, Y = grid
    a, b = 3.0, 2.0
    envir = _make_envir(x, y, a * Y, b * X)
    vort = envir.get_2D_vorticity()
    assert np.allclose(vort, b - a, atol=1e-10)
