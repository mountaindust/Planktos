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
from planktos import motion


def _shear_FTLE(A, T=1.0):
    '''Largest FTLE of a shear flow whose accumulated shear is A: the flow-map
    gradient is [[1, A],[0,1]], so lam_max = (2+A^2 + sqrt((2+A^2)^2-4))/2 and
    FTLE = ln(sqrt(lam_max))/T.'''
    lam = (2 + A**2 + np.sqrt((2 + A**2)**2 - 4)) / 2
    return np.log(np.sqrt(lam)) / T


# --------------------------------------------------------------------------- #
#                            2D vorticity                                      #
# --------------------------------------------------------------------------- #

def _make_envir(x, y, vx, vy):
    '''2D Environment with a hand-built static flow. flow_points[i] indexes axis i
    of the flow arrays (the convention get_vorticity uses). On dyload flow_points
    lives on the FluidData object (envir.flow); set it explicitly because the
    constructor builds a uniform grid, while the non-uniform cases need the actual
    coordinates.'''
    envir = planktos.Environment(Lx=float(x[-1]), Ly=float(y[-1]), flow=[vx, vy])
    envir.flow.flow_points = (x, y)
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
    vort = _make_envir(x, y, -Y, X).get_vorticity()
    assert vort.shape == X.shape
    assert np.allclose(vort, 2.0, atol=1e-10)


def test_vorticity_shear(grid):
    '''v = (a*y, 0): vorticity = -a everywhere.'''
    x, y, X, Y = grid
    a = 1.7
    assert np.allclose(_make_envir(x, y, a * Y, np.zeros_like(Y)).get_vorticity(),
                       -a, atol=1e-10)


def test_vorticity_general_linear(grid):
    '''v = (a*y, b*x): vorticity = b - a everywhere.'''
    x, y, X, Y = grid
    a, b = 3.0, 2.0
    assert np.allclose(_make_envir(x, y, a * Y, b * X).get_vorticity(),
                       b - a, atol=1e-10)


def test_vorticity_on_nonsquare_constructor_flow():
    '''get_vorticity must work on a non-square flow built via the constructor
    (regression: it previously raised due to swapped flow_points).'''
    nx, ny = 12, 9
    x = np.linspace(0, 10, nx); y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[-Y, X])   # solid-body rotation
    vort = envir.get_vorticity()
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


# --------------------------------------------------------------------------- #
#                          backward-time FTLE                                  #
# --------------------------------------------------------------------------- #
# Backward FTLE is the forward integration of the reversed flow; FTLE_largest
# then holds the backward (attracting-LCS) field.

def test_backward_FTLE_uniform_flow_is_zero():
    envir = planktos.Environment(Lx=10, Ly=10,
                                 flow=[np.full((11, 11), 1.0), np.full((11, 11), 0.5)])
    envir.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05, backward=True)
    assert np.nanmax(np.abs(envir.FTLE_largest)) < 1e-8
    assert envir.FTLE_backward is True


def test_backward_FTLE_steady_shear_closed_form():
    # Steady shear is symmetric in time, so backward FTLE equals forward: ln(phi)/T.
    a, T, n = 1.0, 1.0, 21
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[a * Y, np.zeros_like(Y)])
    envir.calculate_FTLE(grid_dim=(8, 8), T=T, dt=0.05, backward=True)
    assert np.nanmax(envir.FTLE_largest) == pytest.approx(_shear_FTLE(a * T), abs=1e-3)


def test_FTLE_forward_vs_backward_differ_time_dependent_shear():
    # Time-dependent shear u = ((1+t)*y, 0) on flow_times spanning [-1, 1] so both
    # the forward [0,1] and backward [0,-1] real-time ranges are in-range. The
    # accumulated shear differs by direction, so forward and backward FTLE DIFFER:
    #   forward  A = \int_0^1 (1+t) dt = 1.5  -> ln(2)        ~ 0.693
    #   backward A = \int_{-1}^0 (1+t) dt = 0.5 -> ln(sqrt(lam(0.5))) ~ 0.248
    # Matching both closed forms (and that they differ) proves the backward path
    # genuinely integrates the reversed flow rather than re-deriving forward.
    n = 21
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    times = np.linspace(-1.0, 1.0, 9)
    u = np.stack([(1.0 + t) * Y for t in times])
    v = np.zeros_like(u)

    def envir():
        return planktos.Environment(Lx=10, Ly=10, flow=[u.copy(), v.copy()],
                                    flow_times=times.copy())

    ef = envir(); ef.calculate_FTLE(grid_dim=(8, 8), t0=0, T=1.0, dt=0.02)
    eb = envir(); eb.calculate_FTLE(grid_dim=(8, 8), t0=0, T=1.0, dt=0.02, backward=True)
    fwd = np.nanmax(ef.FTLE_largest); bwd = np.nanmax(eb.FTLE_largest)

    assert fwd == pytest.approx(_shear_FTLE(1.5), abs=1e-2)
    assert bwd == pytest.approx(_shear_FTLE(0.5), abs=1e-2)
    assert abs(fwd - bwd) > 0.3, "forward and backward FTLE should differ here"


def test_backward_FTLE_with_static_wall_runs():
    # Static immersed boundaries are respected in both time directions.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import _ib_harness as h
    n = 11
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[np.ones_like(Y), np.zeros_like(Y)])
    envir.ibmesh = h.wall_segments(20, 5.0)
    envir.max_meshpt_dist = h.max_meshpt_dist(envir.ibmesh)
    envir.calculate_FTLE(grid_dim=(10, 10), T=0.5, dt=0.05, backward=True)
    assert np.isfinite(np.asarray(envir.FTLE_largest)).any()


# --------------------------------------------------------------------------- #
#                          FTLE scope guards                                  #
# --------------------------------------------------------------------------- #

def test_backward_FTLE_rejects_non_tracer():
    # Reverse-time integration of dissipative inertial dynamics is ill-posed.
    n = 11
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[Y, np.zeros_like(Y)])
    with pytest.raises(NotImplementedError):
        envir.calculate_FTLE(grid_dim=(8, 8), T=1.0, dt=0.05, backward=True,
                             ode_gen=motion.inertial_particles, props={'R': 2/3, 'diam': 0.01})


@pytest.mark.parametrize('backward', [False, True])
def test_FTLE_rejects_moving_mesh(backward):
    n = 11
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[Y, np.zeros_like(Y)])
    envir.ibmesh = np.zeros((3, 5, 2, 2))            # 4D -> moving mesh
    envir.ibmesh_times = np.array([0.0, 0.5, 1.0])
    with pytest.raises(NotImplementedError):
        envir.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05, backward=backward)


def test_FTLE_rejects_nonpositive_extent():
    n = 11
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[Y, np.zeros_like(Y)])
    with pytest.raises(ValueError):
        envir.calculate_FTLE(grid_dim=(8, 8), t0=0, T=-1.0, dt=0.05, backward=True)
