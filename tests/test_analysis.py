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


def _full_stencil_values(envir, last_time, T):
    """FTLE values at grid points whose entire stencil integrated for the full T.

    calculate_FTLE normalizes by the time actually integrated, which is shorter
    than T wherever a stencil point left the domain early -- see the method's
    docstring. The closed forms below are all evaluated at s = T, so they apply
    only where nothing truncated. For a linear flow the field is spatially
    constant there, so both ends of the range are asserted rather than just the
    max: that pins every qualifying point instead of one lucky one.
    """
    lt = np.reshape(np.asarray(last_time), envir.FTLE_grid_dim)
    full = lt >= T - 1e-12
    stencil = np.zeros_like(full)
    stencil[1:-1, 1:-1] = (full[1:-1, 1:-1] & full[:-2, 1:-1] & full[2:, 1:-1]
                           & full[1:-1, :-2] & full[1:-1, 2:])
    keep = stencil & ~np.ma.getmaskarray(envir.FTLE_largest)
    assert keep.any(), "no grid point integrated for the full T"
    return np.asarray(envir.FTLE_largest)[keep]


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
    _, _, last_time = envir.calculate_FTLE(grid_dim=(8, 8), T=T, dt=0.05)

    aT = a * T
    lam = (2 + aT**2 + np.sqrt((2 + aT**2)**2 - 4)) / 2
    expected = np.log(np.sqrt(lam)) / T                  # = ln(golden ratio) ~ 0.4812
    vals = _full_stencil_values(envir, last_time, T)
    assert vals.max() == pytest.approx(expected, abs=1e-3)
    assert vals.min() == pytest.approx(expected, abs=1e-3)


def test_FTLE_normalizes_by_the_time_actually_integrated():
    # The lock on the normalization. u = a(x-1/2), v = -a(y-1/2) is pure strain:
    # the flow map over any elapsed time s has gradient diag(e^{as}, e^{-as}), so
    # the largest FTLE is ln(sqrt(e^{2as}))/s = a -- EXACTLY a, for every s. That
    # is what makes this field the right probe: a point whose stencil left the
    # domain early integrated for less than T, and must still report a.
    #
    # calculate_FTLE used to divide by T regardless of how long it had actually
    # integrated, so the stretching from one interval was normalized by another's
    # length. Truncated points came back at 0.34-0.72 of truth here, always low,
    # in a band 3-4 cells deep around the domain edge -- and in a through-flow
    # domain, where tracers leave continuously, that band is not a rim.
    a, T, n = 1.0, 0.2, 41
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g, indexing='ij')
    envir = planktos.Environment(Lx=1.0, Ly=1.0,
                                 flow=[a*(X - 0.5), -a*(Y - 0.5)])
    _, _, last_time = envir.calculate_FTLE(grid_dim=(31, 31), T=T, dt=0.001)

    F = envir.FTLE_largest
    unmasked = ~np.ma.getmaskarray(F)
    vals = np.asarray(F)[unmasked]
    assert vals.size > 800, "expected most of the grid to survive"
    assert np.abs(vals - a).max() < 1e-10

    # and specifically: the points that did NOT get the full T are also exact,
    # which is the half of the field the old normalization got wrong
    lt = np.reshape(np.asarray(last_time), envir.FTLE_grid_dim)
    truncated = unmasked & (lt < T - 1e-12)
    assert truncated.sum() > 50, "expected a meaningful number of early exits"
    assert np.abs(np.asarray(F)[truncated] - a).max() < 1e-10

    # FTLE_smallest is the contraction exponent: -a, by the same argument
    Fs = envir.FTLE_smallest
    small = np.asarray(Fs)[~np.ma.getmaskarray(Fs)]
    assert np.abs(small + a).max() < 1e-10


# --------------------------------------------------------------------------- #
#                    FTLE from a user-supplied Swarm                           #
# --------------------------------------------------------------------------- #
# calculate_FTLE(swrm=...) steps the Swarm's own apply_agent_model rather than
# running RK45, so it is a distinct code path from the tracer and ode_gen
# branches above -- and it had no coverage at all. Two regressions lived there:
# it read self.props_history (an Environment attribute that does not exist)
# where it meant the Swarm's, raising AttributeError on the very first step; and
# copy.copy leaves the copy's history lists aliased to the caller's Swarm, which
# the method's docstring promises not to alter.

class _AdvectSwarm(planktos.Swarm):
    '''Pure Euler advection by the local fluid velocity -- deterministic, unlike
    the default Brownian model, so the FTLE has a closed-form answer.'''

    def apply_agent_model(self, dt):
        return self.positions + self.get_fluid_drift()*dt


@pytest.mark.parametrize('store_prop_history', [False, True])
def test_FTLE_with_user_swarm_shear_closed_form(store_prop_history):
    # u = (a*y, 0): y never changes, so an Euler step is exact and the flow-map
    # gradient is the same [[1, aT],[0,1]] the RK45 path produces. Same closed
    # form as test_FTLE_simple_shear_closed_form, reached a different way.
    a, T, n = 1.0, 1.0, 21
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[a * Y, np.zeros_like(Y)])
    swrm = _AdvectSwarm(swarm_size=4, envir=envir, seed=1,
                        store_prop_history=store_prop_history)

    _, _, last_time = envir.calculate_FTLE(grid_dim=(8, 8), T=T, dt=0.05, swrm=swrm)
    vals = _full_stencil_values(envir, last_time, T)
    assert vals.max() == pytest.approx(_shear_FTLE(a * T), abs=1e-3)
    assert vals.min() == pytest.approx(_shear_FTLE(a * T), abs=1e-3)


@pytest.mark.parametrize('store_prop_history', [False, True])
def test_FTLE_with_user_swarm_leaves_it_unaltered(store_prop_history):
    # "The Swarm object itself will not be altered; a shallow copy will be
    # created" -- so the caller's histories must not collect the grid-sized
    # entries the FTLE integration generates.
    n = 11
    x = np.linspace(0, 10, n); y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[Y, np.zeros_like(Y)])
    swrm = _AdvectSwarm(swarm_size=4, envir=envir, seed=1,
                        store_prop_history=store_prop_history)
    swrm.move(0.1)                                  # one real step of its own

    n_pos = len(swrm.pos_history)
    n_vel = len(swrm.vel_history)
    n_props = None if swrm.props_history is None else len(swrm.props_history)

    envir.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05, swrm=swrm)

    assert len(swrm.pos_history) == n_pos
    assert len(swrm.vel_history) == n_vel
    if n_props is None:
        assert swrm.props_history is None
    else:
        assert len(swrm.props_history) == n_props
    assert swrm.positions.shape[0] == 4             # not re-gridded


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
    _, _, last_time = envir.calculate_FTLE(grid_dim=(8, 8), T=T, dt=0.05, backward=True)
    vals = _full_stencil_values(envir, last_time, T)
    assert vals.max() == pytest.approx(_shear_FTLE(a * T), abs=1e-3)
    assert vals.min() == pytest.approx(_shear_FTLE(a * T), abs=1e-3)


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

    ef = envir(); _, _, lt_f = ef.calculate_FTLE(grid_dim=(8, 8), t0=0, T=1.0, dt=0.02)
    eb = envir(); _, _, lt_b = eb.calculate_FTLE(grid_dim=(8, 8), t0=0, T=1.0,
                                                 dt=0.02, backward=True)
    fwd = _full_stencil_values(ef, lt_f, 1.0).max()
    bwd = _full_stencil_values(eb, lt_b, 1.0).max()

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


# --------------------------------------------------------------------------- #
#                    2D vorticity on a periodic grid                          #
# --------------------------------------------------------------------------- #
# periodic_dim=True means the upper grid edge duplicates the lower one, so the
# field continues past either end. np.gradient cannot know that and falls back
# to a one-sided difference there, which made the outermost ring of every
# vorticity plot wrong -- measured at 5-8% against IB2d's own Omega, where the
# interior matched it exactly. Periodic axes are now differenced across the wrap.


def _periodic_envir(n=33, L=2*np.pi, periodic=True):
    '''Taylor-Green on a periodic grid, endpoint duplicated as the contract asks.

    u = sin(x+a)cos(y+b), v = -cos(x+a)sin(y+b) => vorticity = 2 sin(x+a) sin(y+b)

    The phases matter: unshifted, that vorticity is identically zero along every
    edge of [0,L]^2, so an edge test against it passes no matter how the edge is
    differenced.
    '''
    a, b = 0.7, 0.3
    x = np.linspace(0.0, L, n)                       # x[-1] wraps to x[0]
    X, Y = np.meshgrid(x, x, indexing='ij')
    u = np.sin(X+a)*np.cos(Y+b)
    v = -np.cos(X+a)*np.sin(Y+b)
    envir = planktos.Environment(Lx=L, Ly=L, flow=[u, v], periodic_dim=periodic)
    envir.flow.flow_points = (x, x)
    return envir, 2*np.sin(X+a)*np.sin(Y+b)


def _edge_mask(shape):
    m = np.zeros(shape, bool)
    m[0] = m[-1] = True
    m[:, 0] = m[:, -1] = True
    return m


def test_vorticity_of_a_periodic_field_is_itself_periodic():
    # The exact invariant, needing no analytic answer: if the field wraps, its
    # curl wraps. A one-sided difference at the edge breaks this.
    envir, _ = _periodic_envir()
    vort = envir.get_vorticity()
    assert np.allclose(vort[0, :], vort[-1, :], atol=1e-12)
    assert np.allclose(vort[:, 0], vort[:, -1], atol=1e-12)


def test_vorticity_periodic_edge_is_as_accurate_as_the_interior():
    # The point of the fix. Central differencing is second order everywhere, so
    # the edge ring should be no worse than the bulk -- not merely closer than
    # it was.
    envir, exact = _periodic_envir()
    err = np.abs(envir.get_vorticity() - exact)
    edge = _edge_mask(err.shape)
    rms_edge = np.sqrt(np.mean(err[edge]**2))
    rms_interior = np.sqrt(np.mean(err[~edge]**2))
    assert rms_edge < 2*rms_interior


def test_vorticity_periodic_converges_at_second_order_including_the_edge():
    # Halving h must quarter the error on the edge ring too. Under the old
    # one-sided treatment the edge converged at first order and dominated.
    errs = []
    for n in (33, 65):
        envir, exact = _periodic_envir(n=n)
        err = np.abs(envir.get_vorticity() - exact)
        errs.append(np.sqrt(np.mean(err[_edge_mask(err.shape)]**2)))
    assert errs[0]/errs[1] > 3.5                     # 4 in the limit


def test_vorticity_non_periodic_is_unchanged():
    # Regression guard: the non-periodic path must still be exactly np.gradient,
    # so nothing that was not periodic moves.
    envir, _ = _periodic_envir(periodic=False)
    x = envir.flow.flow_points[0]
    u, v = envir.flow[0], envir.flow[1]
    expected = (np.gradient(v, x, axis=0) - np.gradient(u, x, axis=1))
    assert np.array_equal(envir.get_vorticity(), expected)


def test_vorticity_periodic_in_one_dimension_only():
    # Mixed periodicity: only the wrapped axis changes, and only at its own ends.
    envir, _ = _periodic_envir()
    envir.flow.periodic_dim = (True, False)
    vort = envir.get_vorticity()
    # x wraps, so the curl still wraps in x...
    assert np.allclose(vort[0, :], vort[-1, :], atol=1e-12)
    # ...but nothing forces agreement across the non-periodic y ends
    assert not np.allclose(vort[:, 0], vort[:, -1], atol=1e-12)


def test_vorticity_3d_periodic_axis_wraps():
    n, L = 17, 2*np.pi
    a = np.linspace(0.0, L, n)
    X, Y, Z = np.meshgrid(a, a, a, indexing='ij')
    u = np.sin(Y); v = np.sin(Z); w = np.sin(X)
    envir = planktos.Environment(Lx=L, Ly=L, Lz=L, flow=[u, v, w],
                                 periodic_dim=True)
    envir.flow.flow_points = (a, a, a)
    vort = envir.get_vorticity()
    for comp in vort:
        assert np.allclose(comp[0], comp[-1], atol=1e-12)
        assert np.allclose(comp[:, 0], comp[:, -1], atol=1e-12)
        assert np.allclose(comp[..., 0], comp[..., -1], atol=1e-12)


# ---- the helper itself ------------------------------------------------------

def test_spatial_gradient_non_periodic_defers_to_numpy():
    from planktos.fluid import _spatial_gradient
    rng = np.random.default_rng(0)
    f = rng.normal(size=(7, 5))
    x = np.sort(rng.uniform(0, 3, 7)); x[0] = 0.
    assert np.array_equal(_spatial_gradient(f, x, 0, periodic=False),
                          np.gradient(f, x, axis=0))


def test_spatial_gradient_periodic_handles_uneven_spacing():
    from planktos.fluid import _spatial_gradient
    # Non-uniform periodic axis. Checked against the unequal-spacing central
    # difference by hand at index 0, which is where the ghost point lands: its
    # left neighbour is x[-2] one period back, NOT x[-1] (the duplicate).
    # A constant-step formula, or a ghost taken from x[-1], gets this wrong.
    x = np.array([0., 0.5, 1.7, 3.0, 4.0])           # x[-1] wraps to x[0]
    rng = np.random.default_rng(3)
    f = rng.normal(size=(5, 3))
    f[-1] = f[0]                                     # the periodic contract
    g = _spatial_gradient(f, x, 0, periodic=True)

    period = x[-1] - x[0]
    hs = x[0] - (x[-2] - period)                     # back to the wrapped point
    hd = x[1] - x[0]
    expected = (-hd/(hs*(hs+hd))*f[-2] + (hd-hs)/(hs*hd)*f[0]
                + hs/(hd*(hs+hd))*f[1])
    assert np.allclose(g[0], expected)
    assert np.allclose(g[0], g[-1], atol=1e-12)      # wraps
    # the interior is untouched by any of this
    assert np.allclose(g[1:-1], np.gradient(f, x, axis=0)[1:-1])
