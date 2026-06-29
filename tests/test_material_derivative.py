'''Known-answer tests for the material derivative Du/Dt and the partial time
derivative du/dt of the fluid velocity at agent locations
(Swarm.get_DuDt / Swarm.get_dudt). Inertial particles and FTLE both depend on
these, but they had no closed-form coverage.

The material derivative is Du/Dt = u_t + (u . grad) u. For a velocity field that
is linear in space the spatial gradient is a constant, exact under the
np.gradient stencil (including the edge_order=2 boundaries), and the convective
term (u . grad)u is itself linear in space -- so the linear spatial interpolation
onto agent positions is exact to machine precision. Cubic-spline-in-time
interpolation reproduces a flow that is linear in time exactly, so its time
derivative is exact too. Every case below therefore has a closed-form answer.

Flow arrays are built by hand with indexing='ij' and flow_points (which live on
the FluidData object at Environment.flow on dyload) are set explicitly so that
axis i of each flow array is indexed by flow_points[i] -- the convention
np.gradient and interpolate_flow expect. The grids here are uniform, so this just
mirrors what the constructor builds; setting it explicitly keeps the helpers
correct for non-uniform grids too.
'''

import numpy as np

import planktos


# --------------------------------------------------------------------------- #
#                                 helpers                                      #
# --------------------------------------------------------------------------- #

def _envir_2d(x, y, vx, vy, flow_times=None):
    envir = planktos.Environment(Lx=float(x[-1]), Ly=float(y[-1]),
                                 flow=[vx, vy], flow_times=flow_times)
    envir.flow.flow_points = (x, y)
    return envir


def _envir_3d(x, y, z, vx, vy, vz):
    envir = planktos.Environment(Lx=float(x[-1]), Ly=float(y[-1]), Lz=float(z[-1]),
                                 flow=[vx, vy, vz])
    envir.flow.flow_points = (x, y, z)
    return envir


def _swarm(envir, pts):
    pts = np.asarray(pts, dtype=float)
    return planktos.Swarm(swarm_size=len(pts), envir=envir, init=pts, seed=1)


def _grid_2d(nx=15, ny=11):
    x = np.linspace(0.0, 10.0, nx)
    y = np.linspace(0.0, 8.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return x, y, X, Y


def _grid_3d(nx=11, ny=9, nz=13):
    x = np.linspace(0.0, 10.0, nx)
    y = np.linspace(0.0, 8.0, ny)
    z = np.linspace(0.0, 12.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return x, y, z, X, Y, Z


# interior query points, comfortably inside the domains above
PTS_2D = np.array([[3.0, 2.0], [7.0, 5.0], [5.0, 4.0], [2.0, 6.0]])
PTS_3D = np.array([[3.0, 2.0, 4.0], [7.0, 5.0, 2.0], [5.0, 4.0, 6.0]])


# --------------------------------------------------------------------------- #
#                       2D steady fields (convective term)                     #
# --------------------------------------------------------------------------- #

def test_DuDt_steady_straining_flow():
    # u = (a x, -a y): a divergence-free stagnation-point flow, steady in time.
    # u_t = 0 and (u . grad)u = (a^2 x, a^2 y), so Du/Dt = (a^2 x, a^2 y).
    a = 1.5
    x, y, X, Y = _grid_2d()
    swrm = _swarm(_envir_2d(x, y, a * X, -a * Y), PTS_2D)

    expected = np.column_stack([a**2 * PTS_2D[:, 0], a**2 * PTS_2D[:, 1]])
    assert np.allclose(swrm.get_DuDt(positions=PTS_2D), expected, atol=1e-9)
    # steady flow: the partial time derivative is exactly zero
    assert np.allclose(swrm.get_dudt(positions=PTS_2D), 0.0, atol=1e-9)


def test_DuDt_steady_shear_flow_convective_term_cancels():
    # u = (a y, 0): the classic shear example. u_t = 0 and (u . grad)u = 0, so
    # Du/Dt = 0 everywhere despite a nonzero velocity gradient (du_x/dy = a).
    a = 1.7
    x, y, X, Y = _grid_2d()
    swrm = _swarm(_envir_2d(x, y, a * Y, np.zeros_like(Y)), PTS_2D)
    assert np.allclose(swrm.get_DuDt(positions=PTS_2D), 0.0, atol=1e-9)


def test_DuDt_default_positions_uses_swarm_locations():
    # Exercise the default (self.positions) path, including the masked-array
    # handling: agents are seeded exactly at the query points.
    a = 2.0
    x, y, X, Y = _grid_2d()
    swrm = _swarm(_envir_2d(x, y, a * X, -a * Y), PTS_2D)
    expected = np.column_stack([a**2 * PTS_2D[:, 0], a**2 * PTS_2D[:, 1]])
    assert np.allclose(swrm.get_DuDt(), expected, atol=1e-9)


# --------------------------------------------------------------------------- #
#                  2D time-dependent field (the u_t term)                      #
# --------------------------------------------------------------------------- #

def test_dudt_and_DuDt_time_dependent_uniform_flow():
    # u = (b t, 0): uniform in space, linear in time. Cubic-spline-in-time
    # interpolation is exact for linear data, so du/dt = (b, 0) at every time.
    # The field is uniform in space so (u . grad)u = 0 and Du/Dt = (b, 0) too.
    b = 0.75
    x, y, X, Y = _grid_2d()
    times = np.linspace(0.0, 4.0, 5)
    u = np.stack([b * t * np.ones_like(X) for t in times])
    v = np.zeros_like(u)
    swrm = _swarm(_envir_2d(x, y, u, v, flow_times=times), PTS_2D)

    dudt = swrm.get_dudt(positions=PTS_2D)
    assert np.allclose(dudt[:, 0], b, atol=1e-9)
    assert np.allclose(dudt[:, 1], 0.0, atol=1e-9)

    DuDt = swrm.get_DuDt(positions=PTS_2D)
    expected = np.column_stack([np.full(len(PTS_2D), b), np.zeros(len(PTS_2D))])
    assert np.allclose(DuDt, expected, atol=1e-9)


# --------------------------------------------------------------------------- #
#                          3D steady field (convective term)                   #
# --------------------------------------------------------------------------- #

def test_DuDt_3d_steady_straining_flow():
    # u = (a x, a y, -2a z): divergence-free, steady. Component-wise,
    # (u . grad)u = (a^2 x, a^2 y, 4 a^2 z), and u_t = 0.
    a = 1.2
    x, y, z, X, Y, Z = _grid_3d()
    swrm = _swarm(_envir_3d(x, y, z, a * X, a * Y, -2 * a * Z), PTS_3D)
    expected = np.column_stack([a**2 * PTS_3D[:, 0],
                                a**2 * PTS_3D[:, 1],
                                4 * a**2 * PTS_3D[:, 2]])
    assert np.allclose(swrm.get_DuDt(positions=PTS_3D), expected, atol=1e-9)


# --------------------------------------------------------------------------- #
#                  du/dt time-boundary handling (regression)                   #
# --------------------------------------------------------------------------- #

def test_dudt_time_boundaries_and_extrapolation():
    # u = (b t, 0), times in [0, 4]. du/dt = b inside AND at the data endpoints
    # (the spline derivative is defined there); constant extrapolation gives
    # du/dt = 0 strictly beyond the ends. Pins two bugs in FluidData.get_dudt's
    # out-of-range branch: (1) it used <=/>= so it spuriously zeroed the
    # derivative *at* t0/tN, and (2) it built the zeros with the full fshape
    # (including the time axis) instead of a single-time NxD field, which made
    # calculate_DuDt raise a broadcast error at a boundary time.
    b = 0.6
    x, y, X, Y = _grid_2d()
    times = np.linspace(0.0, 4.0, 5)
    u = np.stack([b * t * np.ones_like(X) for t in times])
    v = np.zeros_like(u)
    swrm = _swarm(_envir_2d(x, y, u, v, flow_times=times), PTS_2D)

    # at the left (t=0) and right (t=4) data endpoints: du/dt = (b, 0)
    for t_end in (0.0, 4.0):
        d = swrm.get_dudt(positions=PTS_2D, time=t_end)
        assert d.shape == PTS_2D.shape
        assert np.allclose(d[:, 0], b, atol=1e-9)
        assert np.allclose(d[:, 1], 0.0, atol=1e-9)

    # strictly beyond the ends: constant extrapolation -> zero derivative,
    # and Du/Dt there is the convective term only (here zero, uniform flow)
    for t_out in (-1.0, 5.0):
        d = swrm.get_dudt(positions=PTS_2D, time=t_out)
        assert d.shape == PTS_2D.shape
        assert np.allclose(d, 0.0, atol=1e-9)
        assert np.allclose(swrm.get_DuDt(positions=PTS_2D, time=t_out), 0.0, atol=1e-9)
