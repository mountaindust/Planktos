'''Pins the fluid-velocity-field interface contract of Environment.flow.

This module was written as the safety net required by
docs/notes/flow_field_interface.md §7.2: it fixed the *observable behavior* of
every consumer of Environment.flow before the FlowArray removal (§7.3), so that
the deletion could be shown to be behavior-preserving rather than merely assumed
so. It served that purpose -- FlowArray is gone and every assertion here passed
unchanged, including after the defensive np.asarray wrappers were stripped out --
and it now stands as the general contract test for the fluid interface. Keep it
green through the tiling (§9) and plotting (§8) work still to come.

Scope is deliberately the surfaces that had no direct coverage:
  * Environment.interpolate_flow / interpolate_temporal_flow — the per-move hot
    path, and the single most important thing not to break.
  * Swarm._calc_basic_stats — the fluid summary printed on every plot frame.
  * FluidData.fmin/fmax — the tuple contract (regression lock for §3.3).
  * 3D vorticity — a known-answer test the mvbnd overhaul deferred to this branch.
2D vorticity (test_analysis.py) and save_fluid round-trips (test_io_loaders.py)
are already pinned elsewhere and are not duplicated here.

Everything is analytic with a closed-form answer: fields linear in space are
reproduced exactly by linear spatial interpolation, and fields linear in time are
reproduced exactly by the cubic temporal spline, so the assertions are exact
rather than tolerance-tuned. No RNG, no external data, no file I/O.
'''

import numpy as np
import pytest

import planktos


# --------------------------------------------------------------------------- #
#                                  helpers                                     #
# --------------------------------------------------------------------------- #

def _linear_2d(nx=11, ny=9, Lx=10.0, Ly=8.0):
    '''Static 2D environment with u = x, v = 2y (exactly linear in space).'''
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=Lx, Ly=Ly, flow=[X.copy(), 2 * Y.copy()],
                                 x_bndry=('zero', 'zero'), y_bndry=('zero', 'zero'))
    return envir, X, Y


def _linear_3d(n=7, L=6.0):
    '''Static 3D environment with u = x, v = 2y, w = 3z.'''
    g = np.linspace(0, L, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    envir = planktos.Environment(Lx=L, Ly=L, Lz=L,
                                 flow=[X.copy(), 2 * Y.copy(), 3 * Z.copy()],
                                 x_bndry=('zero', 'zero'), y_bndry=('zero', 'zero'),
                                 z_bndry=('zero', 'zero'))
    return envir, X, Y, Z


# --------------------------------------------------------------------------- #
#                    interpolate_flow — static (the hot path)                  #
# --------------------------------------------------------------------------- #

def test_interpolate_flow_static_2d_exact_on_and_off_node():
    envir, _, _ = _linear_2d()
    # a mix of on-node, off-node, and domain-corner query points
    pts = np.array([[2.5, 3.0], [7.25, 1.5], [0.0, 8.0], [10.0, 0.0], [5.0, 4.0]])
    got = envir.interpolate_flow(pts)
    expected = np.stack([pts[:, 0], 2 * pts[:, 1]], axis=1)
    assert got.shape == (5, 2)
    assert np.allclose(got, expected)


def test_interpolate_flow_static_3d_exact():
    envir, _, _, _ = _linear_3d()
    pts = np.array([[1.0, 2.0, 3.0], [0.5, 5.5, 2.25], [6.0, 6.0, 6.0]])
    got = envir.interpolate_flow(pts)
    expected = np.stack([pts[:, 0], 2 * pts[:, 1], 3 * pts[:, 2]], axis=1)
    assert got.shape == (3, 3)
    assert np.allclose(got, expected)


def test_interpolate_flow_extrapolates_outside_domain():
    # fill_value=None + bounds_error=False => linear extrapolation, which the RK4
    # solvers depend on (see the splinef2d guard in interpolate_flow).
    envir, _, _ = _linear_2d()
    pts = np.array([[12.0, 2.0], [-1.5, 3.0]])
    got = envir.interpolate_flow(pts)
    assert np.allclose(got, np.array([[12.0, 4.0], [-1.5, 6.0]]))


def test_interpolate_flow_single_point_shape():
    envir, _, _ = _linear_2d()
    got = envir.interpolate_flow(np.array([[3.0, 4.0]]))
    assert got.shape == (1, 2)
    assert np.allclose(got, [[3.0, 8.0]])


def test_interpolate_flow_accepts_explicit_flow_argument():
    # The `flow=` argument bypasses the environment's own field. Used by the FTLE
    # machinery and by callers that have already interpolated in time.
    envir, X, Y = _linear_2d()
    other = [np.zeros_like(X), np.ones_like(Y)]
    got = envir.interpolate_flow(np.array([[3.0, 4.0]]), flow=other)
    assert np.allclose(got, [[0.0, 1.0]])


def test_interpolate_flow_honors_explicit_flow_points():
    # Regression (dyload-only): flow_points was accepted, documented, and then
    # unconditionally overwritten with the environment's grid, so a caller-supplied
    # grid was silently discarded. master carried the `if flow_points is None`
    # guard; it was lost when flow_points moved onto FluidData.
    #
    # The substitute field is deliberately the same *shape* as the environment's
    # grid but on twice the extent, so the bug returns a plausible wrong number
    # instead of raising a shape mismatch -- the dangerous failure mode.
    envir, _, _ = _linear_2d()                  # environment grid: x in [0,10], 11 pts
    xg = np.linspace(0, 20, 11)                 # supplied grid:    x in [0,20], 11 pts
    yg = np.linspace(0, 16, 9)
    Xg, Yg = np.meshgrid(xg, yg, indexing='ij')
    other = [Xg.copy(), np.zeros_like(Yg)]      # u = x on the *supplied* grid

    got = envir.interpolate_flow(np.array([[2.0, 3.0]]), flow=other,
                                 flow_points=(xg, yg))
    assert np.allclose(got, [[2.0, 0.0]])       # against the env grid this reads 4.0


def test_interpolate_flow_defaults_to_environment_flow_points():
    # The None default must still resolve to the environment's grid.
    envir, X, Y = _linear_2d()
    got = envir.interpolate_flow(np.array([[3.0, 4.0]]),
                                 flow=[X.copy(), 2 * Y.copy()], flow_points=None)
    assert np.allclose(got, [[3.0, 8.0]])


def test_interpolate_flow_periodic_wrap_uses_supplied_flow_points():
    # The periodic wrap (positions % flow_points[n][-1]) must wrap against the grid
    # actually being interpolated on, not unconditionally the environment's.
    # periodic_dim is fluid-level and independent of the agent boundary conditions,
    # so the defaults for bndry are fine here.
    x = np.linspace(0, 10.0, 11)
    y = np.linspace(0, 8.0, 9)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10.0, Ly=8.0,
                                 flow=[X.copy(), np.zeros_like(Y)],
                                 periodic_dim=True)
    xg = np.linspace(0, 20.0, 11)
    yg = np.linspace(0, 16.0, 9)
    Xg, Yg = np.meshgrid(xg, yg, indexing='ij')
    other = [Xg.copy(), np.zeros_like(Yg)]

    # x=22 wraps to 2 on the supplied grid (period 20) and reads u=2. Wrapping
    # against the environment grid (period 10) also gives 2, but then reads that
    # off the substitute array's coarser spacing -> 4.
    got = envir.interpolate_flow(np.array([[22.0, 3.0]]), flow=other,
                                 flow_points=(xg, yg))
    assert np.allclose(got, [[2.0, 0.0]])


# --------------------------------------------------------------------------- #
#                 interpolate_flow / temporal — time-varying                   #
# --------------------------------------------------------------------------- #

def _linear_in_time_2d():
    '''u(t,x,y) = t*x, v(t,x,y) = y. Linear in t, so the cubic temporal spline
    reproduces it exactly at every query time.'''
    nx, ny = 11, 9
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    times = np.array([0.0, 1.0, 2.0, 3.0])
    U = np.stack([t * X for t in times])
    V = np.stack([Y for _ in times])
    envir = planktos.Environment(Lx=10, Ly=8, flow=[U, V], flow_times=times,
                                 x_bndry=('zero', 'zero'), y_bndry=('zero', 'zero'))
    return envir


@pytest.mark.parametrize('t', [0.0, 0.5, 1.0, 1.75, 2.5, 3.0])
def test_interpolate_flow_time_varying_exact(t):
    envir = _linear_in_time_2d()
    pts = np.array([[4.0, 2.0], [7.5, 6.0]])
    got = envir.interpolate_flow(pts, time=t)
    expected = np.stack([t * pts[:, 0], pts[:, 1]], axis=1)
    assert np.allclose(got, expected)


def test_interpolate_flow_defaults_to_current_environment_time():
    envir = _linear_in_time_2d()
    envir.time = 2.0
    got = envir.interpolate_flow(np.array([[4.0, 2.0]]))
    assert np.allclose(got, [[8.0, 2.0]])


def test_interpolate_temporal_flow_returns_spatial_field():
    # Contract: a sequence of per-component arrays on the spatial grid, with the
    # time axis collapsed. This is what save_fluid and _calc_basic_stats consume.
    envir = _linear_in_time_2d()
    flow_t = envir.interpolate_temporal_flow(time=1.5)
    assert len(flow_t) == 2
    for comp in flow_t:
        assert comp.shape == (11, 9)
    x = np.linspace(0, 10, 11)
    expected_u = np.stack([1.5 * x for _ in range(9)], axis=1)
    assert np.allclose(flow_t[0], expected_u)


def test_interpolate_temporal_flow_at_data_times_reproduces_data():
    # For time-varying flow, flow[dim] is a temporal spline object rather than an
    # array; spline[idx] recovers the raw field at that data time. Pinning that
    # indexing contract matters because §6 of the refactor note keeps it.
    envir = _linear_in_time_2d()
    x = np.linspace(0, 10, 11)
    for idx, t in enumerate(envir.flow.flow_times):
        flow_t = envir.interpolate_temporal_flow(time=t)
        raw_u = envir.flow[0][idx]
        assert np.allclose(flow_t[0], raw_u)
        # and the raw data really is the analytic field u = t*x
        assert np.allclose(raw_u, np.stack([t * x for _ in range(9)], axis=1))


# --------------------------------------------------------------------------- #
#                          container contract (hat A)                          #
# --------------------------------------------------------------------------- #

def test_fluiddata_container_contract_static():
    envir, X, Y = _linear_2d()
    assert len(envir.flow) == 2
    assert envir.flow.flow_times is None
    assert envir.flow[0].shape == (11, 9)
    # iteration yields one entry per spatial component
    assert len([comp for comp in envir.flow]) == 2
    # np.array(flow) stacks the components along a new leading axis
    stacked = np.array(envir.flow)
    assert stacked.shape == (2, 11, 9)
    assert np.allclose(stacked[0], X)
    assert np.allclose(stacked[1], 2 * Y)


def test_flow_points_match_component_shape():
    envir, _, _ = _linear_2d()
    assert len(envir.flow.flow_points) == 2
    assert len(envir.flow.flow_points[0]) == 11
    assert len(envir.flow.flow_points[1]) == 9
    assert np.isclose(envir.flow.flow_points[0][-1], 10.0)
    assert np.isclose(envir.flow.flow_points[1][-1], 8.0)


# --------------------------------------------------------------------------- #
#                    fmin / fmax — tuple contract (§3.3 lock)                  #
# --------------------------------------------------------------------------- #

def test_fmin_fmax_are_reusable_tuples_static():
    # Regression lock: these were generator expressions, so unpacking worked
    # exactly once and subscripting raised TypeError. Plotting unpacks fmax on
    # every frame, and update_spline subscripts both on every window slide.
    envir, _, _ = _linear_2d()
    assert isinstance(envir.flow.fmin, tuple)
    assert isinstance(envir.flow.fmax, tuple)
    # unpack twice: a generator would be exhausted by the first
    max_u, max_v = envir.flow.fmax
    max_u2, max_v2 = envir.flow.fmax
    assert (max_u, max_v) == (max_u2, max_v2)
    # subscriptable, and numerically right
    assert np.isclose(envir.flow.fmax[0], 10.0)
    assert np.isclose(envir.flow.fmax[1], 16.0)
    assert np.isclose(envir.flow.fmin[0], 0.0)
    assert np.isclose(envir.flow.fmin[1], 0.0)


def test_fmin_fmax_are_reusable_tuples_time_varying():
    envir = _linear_in_time_2d()
    assert isinstance(envir.flow.fmin, tuple) and isinstance(envir.flow.fmax, tuple)
    a, b = envir.flow.fmax
    c, d = envir.flow.fmax
    assert (a, b) == (c, d)
    # u = t*x peaks at t=3, x=10
    assert np.isclose(envir.flow.fmax[0], 30.0)
    assert np.isclose(envir.flow.fmax[1], 8.0)


# --------------------------------------------------------------------------- #
#              _calc_basic_stats — the fluid summary shown on plots            #
# --------------------------------------------------------------------------- #

def test_calc_basic_stats_2d_known_answers():
    envir, X, Y = _linear_2d()
    swrm = planktos.Swarm(swarm_size=10, envir=envir, seed=1)
    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_swrm_vel = \
        swrm._calc_basic_stats(DIM3=False)

    u = X
    v = 2 * Y
    speed = np.sqrt(u ** 2 + v ** 2)
    assert np.isclose(perc_left, 100.0)
    assert np.isclose(avg_spd_x, u.mean())
    assert np.isclose(avg_spd_y, v.mean())
    assert np.isclose(avg_spd, speed.mean())
    # max_spd must be the max *speed*, not the max of any single component.
    # These differ here (max|u| = 10, max speed = sqrt(100+256)), which is what
    # makes this a real check rather than a coincidence.
    assert np.isclose(max_spd, speed.max())
    assert not np.isclose(max_spd, u.max())


def test_calc_basic_stats_3d_known_answers():
    envir, X, Y, Z = _linear_3d()
    swrm = planktos.Swarm(swarm_size=10, envir=envir, seed=1)
    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel = \
        swrm._calc_basic_stats(DIM3=True)

    u, v, w = X, 2 * Y, 3 * Z
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    assert np.isclose(perc_left, 100.0)
    assert np.isclose(avg_spd_x, u.mean())
    assert np.isclose(avg_spd_y, v.mean())
    assert np.isclose(avg_spd_z, w.mean())
    assert np.isclose(avg_spd, speed.mean())
    assert np.isclose(max_spd, speed.max())


def test_calc_basic_stats_time_varying_uses_requested_time_index():
    envir = _linear_in_time_2d()
    swrm = planktos.Swarm(swarm_size=10, envir=envir, seed=1)
    for _ in range(2):
        swrm.move(1.0)
    # pos_history holds the *previous* steps, so after two moves the valid
    # indices are 0 (t=0) and 1 (t=1); the current step is t_indx=None (t=2).
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 8, 9)
    v_field = np.stack([y for _ in range(11)], axis=0)

    # at t=0 the field is u = 0*x = 0, v = y
    stats0 = swrm._calc_basic_stats(DIM3=False, t_indx=0)
    assert np.isclose(stats0[3], 0.0)                   # avg_spd_x
    assert np.isclose(stats0[4], v_field.mean())        # avg_spd_y
    assert np.isclose(stats0[2], v_field.max())         # max_spd == max|v|

    # at t=1 the field is u = x, v = y
    stats1 = swrm._calc_basic_stats(DIM3=False, t_indx=1)
    u_field = np.stack([x for _ in range(9)], axis=1)
    assert np.isclose(stats1[3], u_field.mean())
    assert np.isclose(stats1[2], np.sqrt(u_field**2 + v_field**2).max())

    # u grows with time, so the max speed strictly increases
    assert stats1[2] > stats0[2]


def test_calc_basic_stats_returns_plain_scalars():
    # Consumers format these with '{:.1g}'.format(...) into plot titles, which
    # requires real scalars. Guards against a stray array-like leaking through.
    envir, _, _ = _linear_2d()
    swrm = planktos.Swarm(swarm_size=10, envir=envir, seed=1)
    stats = swrm._calc_basic_stats(DIM3=False)
    for value in stats[:5]:
        assert np.ndim(value) == 0
        assert '{:.1g}'.format(value)


# --------------------------------------------------------------------------- #
#              get_mean_fluid_speed / calculate_mag_gradient                   #
# --------------------------------------------------------------------------- #

def test_get_mean_fluid_speed_static_known_answer():
    envir, X, Y = _linear_2d()
    expected = np.sqrt(X ** 2 + (2 * Y) ** 2).mean()
    assert np.isclose(envir.get_mean_fluid_speed(), expected)


def test_calculate_mag_gradient_known_answer():
    # Choose the field so that the *magnitude* is exactly linear, making the
    # finite-difference gradient exact rather than approximate:
    #   u = 0.6*s, v = 0.8*s with s = x + 2y >= 0  =>  |u| = s,  grad|u| = (1, 2).
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 8, 9)
    X, Y = np.meshgrid(x, y, indexing='ij')
    s = X + 2 * Y
    envir = planktos.Environment(Lx=10, Ly=8, flow=[0.6 * s, 0.8 * s],
                                 x_bndry=('zero', 'zero'), y_bndry=('zero', 'zero'))
    envir.calculate_mag_gradient()
    assert envir.mag_grad is not None
    assert len(envir.mag_grad) == 2
    assert np.allclose(envir.mag_grad[0], 1.0)
    assert np.allclose(envir.mag_grad[1], 2.0)
    assert np.isclose(envir.mag_grad_time, envir.time)


# --------------------------------------------------------------------------- #
#          LinearSpline / INUM — the dynamic-loading temporal path             #
# --------------------------------------------------------------------------- #
#
# test_temporal_interp.py unit-tests fCubicSpline thoroughly but never touches
# LinearSpline, which is what every INUM (dynamic-loading) run interpolates
# with. Since §7.3 changes what *both* spline classes return, the linear path
# needs equivalent pinning. Environment(flow=...) hardcodes INUM=None, so these
# build FluidData directly -- the same approach test_temporal_interp.py takes.

@pytest.fixture
def linear_in_time_field():
    '''Flow that is exactly linear in time at every grid point, so linear and
    cubic temporal interpolation must agree to machine precision.'''
    T, nx, ny = 5, 4, 3
    t = np.linspace(0.0, 4.0, T)
    rng = np.random.default_rng(0)
    base = rng.uniform(-1, 1, (nx, ny))
    slope = rng.uniform(-1, 1, (nx, ny))
    data = np.stack([base + slope * tt for tt in t])
    fpoints = (np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    return t, data, base, slope, fpoints


def test_INUM_true_selects_linear_spline(linear_in_time_field):
    from planktos import fluid
    t, data, _, _, fpoints = linear_in_time_field
    fd = fluid.FluidData([data.copy(), data.copy()], fpoints,
                         flow_times=t.copy(), INUM=True)
    assert all(isinstance(s, fluid.LinearSpline) for s in fd._flow)
    # and the default stays cubic
    fd_default = fluid.FluidData([data.copy(), data.copy()], fpoints,
                                 flow_times=t.copy())
    assert all(isinstance(s, fluid.fCubicSpline) for s in fd_default._flow)


@pytest.mark.parametrize('query', [0.0, 0.5, 1.5, 2.75, 4.0])
def test_linear_spline_exact_on_linear_data(linear_in_time_field, query):
    from planktos import fluid
    t, data, base, slope, fpoints = linear_in_time_field
    fd = fluid.FluidData([data.copy(), data.copy()], fpoints,
                         flow_times=t.copy(), INUM=True)
    got = fd(query)[0]
    assert np.allclose(got, base + slope * query)


def test_linear_and_cubic_agree_on_linear_data(linear_in_time_field):
    # Both schemes are exact for data linear in time, so any disagreement means
    # one of the two paths is broken -- a cross-check that survives §7.3.
    from planktos import fluid
    t, data, _, _, fpoints = linear_in_time_field
    fd_lin = fluid.FluidData([data.copy(), data.copy()], fpoints,
                             flow_times=t.copy(), INUM=True)
    fd_cub = fluid.FluidData([data.copy(), data.copy()], fpoints,
                             flow_times=t.copy())
    for query in (0.3, 1.9, 3.6):
        assert np.allclose(fd_lin(query)[0],
                           fd_cub(query)[0])


def test_linear_spline_surface(linear_in_time_field):
    # The __call__/__getitem__/min/max/absmax/regenerate_data surface that §6 of
    # the refactor note commits to keeping.
    from planktos import fluid
    t, data, _, _, _ = linear_in_time_field
    sp = fluid.LinearSpline(t.copy(), data.copy())
    assert np.allclose(sp[2], data[2])
    assert np.isclose(sp.min(), data.min())
    assert np.isclose(sp.max(), data.max())
    assert np.isclose(sp.absmax(), np.abs(data).max())
    assert np.allclose(sp.regenerate_data(), data)
    assert sp.shape == data.shape


def test_linear_spline_derivative_is_the_slope(linear_in_time_field):
    # d/dt of a field linear in time is the slope, exactly. This feeds get_dudt
    # -> the material derivative -> the inertial particle models.
    from planktos import fluid
    t, data, _, slope, _ = linear_in_time_field
    sp = fluid.LinearSpline(t.copy(), data.copy())
    for query in (0.5, 2.5, 3.5):
        assert np.allclose(sp.derivative(query), slope)


def test_get_raw_loaded_data_round_trips(linear_in_time_field):
    # Regression lock: the old implementation branched on "is it an fCubicSpline"
    # and let the else-branch return LinearSpline *objects* rather than ndarrays,
    # so this raised TypeError on the entire dynamic-loading path.
    from planktos import fluid
    t, data, _, _, fpoints = linear_in_time_field

    for INUM in (None, True):
        fd = fluid.FluidData([data.copy(), data.copy()], fpoints,
                             flow_times=t.copy(), INUM=INUM)
        raw = fd.get_raw_loaded_data()
        assert len(raw) == 2
        for comp in raw:
            arr = comp
            assert arr.shape == data.shape
            assert np.allclose(arr, data)


def test_get_raw_loaded_data_static():
    envir, X, Y = _linear_2d()
    raw = envir.flow.get_raw_loaded_data()
    assert len(raw) == 2
    assert np.allclose(raw[0], X)
    assert np.allclose(raw[1], 2 * Y)


# --------------------------------------------------------------------------- #
#                    plotting-facing component operations                      #
# --------------------------------------------------------------------------- #

def test_quiver_style_strided_slice_and_transpose():
    # Swarm.plot/plot_all build quivers from flow[k][::M,::N].T. Pin it: this is
    # a derived-array operation, which was exactly the territory where the old
    # FlowArray view misbehaved, and the plotting smokes only assert "runs
    # without error".
    envir, X, Y = _linear_2d()
    M, N = 2, 3
    for k, truth in ((0, X), (1, 2 * Y)):
        sliced = envir.flow[k][::M, ::N]
        assert sliced.shape == truth[::M, ::N].shape
        assert np.allclose(sliced, truth[::M, ::N])
        assert np.allclose(envir.flow[k][::M, ::N].T, truth[::M, ::N].T)


def test_fshape_matches_component_shape():
    # fshape drives frame sizing in the plotting code, and its time-axis
    # handling was the subject of a real get_dudt bug (fshape vs fshape[1:]).
    envir, _, _ = _linear_2d()
    assert envir.flow.fshape == (11, 9)
    envir_t = _linear_in_time_2d()
    assert envir_t.flow.fshape == (4, 11, 9)
    assert envir_t.flow.fshape[1:] == (11, 9)


# --------------------------------------------------------------------------- #
#                        3D vorticity — known answer                           #
# --------------------------------------------------------------------------- #

def test_vorticity_3d_solid_body_rotation():
    # u = (-y, x, 0) => vorticity = (0, 0, 2) everywhere.
    n, L = 7, 6.0
    g = np.linspace(0, L, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    envir = planktos.Environment(Lx=L, Ly=L, Lz=L,
                                 flow=[-Y.copy(), X.copy(), np.zeros_like(Z)],
                                 x_bndry=('zero', 'zero'), y_bndry=('zero', 'zero'),
                                 z_bndry=('zero', 'zero'))
    vort = envir.get_vorticity()
    assert len(vort) == 3
    assert np.allclose(vort[0], 0.0)
    assert np.allclose(vort[1], 0.0)
    assert np.allclose(vort[2], 2.0)


def test_vorticity_3d_general_linear_field():
    # A general linear field has constant vorticity:
    #   u = a1 x + b1 y + c1 z,  v = a2 x + b2 y + c2 z,  w = a3 x + b3 y + c3 z
    #   curl = (b3 - c2, c1 - a3, a2 - b1)
    n, L = 7, 6.0
    g = np.linspace(0, L, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    a1, b1, c1 = 1.0, -2.0, 0.5
    a2, b2, c2 = 3.0, 0.25, -1.5
    a3, b3, c3 = -0.75, 2.0, 4.0
    envir = planktos.Environment(
        Lx=L, Ly=L, Lz=L,
        flow=[a1 * X + b1 * Y + c1 * Z,
              a2 * X + b2 * Y + c2 * Z,
              a3 * X + b3 * Y + c3 * Z],
        x_bndry=('zero', 'zero'), y_bndry=('zero', 'zero'),
        z_bndry=('zero', 'zero'))
    vort = envir.get_vorticity()
    assert np.allclose(vort[0], b3 - c2)
    assert np.allclose(vort[1], c1 - a3)
    assert np.allclose(vort[2], a2 - b1)


def test_vorticity_3d_shape_matches_grid():
    envir, _, _, _ = _linear_3d()
    vort = envir.get_vorticity()
    for comp in vort:
        assert comp.shape == (7, 7, 7)
