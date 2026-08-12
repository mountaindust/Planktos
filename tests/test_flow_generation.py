'''Tests for analytic flow generation and flow-field manipulation on Environment.

Covers the self-contained flow generators (set_brinkman_flow,
set_two_layer_channel_flow, set_canopy_flow) plus the flow utility tile_domain and
the flow_points axis-ordering contract. No external data, no agents, no RNG;
resolutions are kept small so the whole module runs in well under a second.

On dyload the per-Environment fluid API (flow_times, flow_points, tile_flow) moved
onto the FluidData object at Environment.flow; the extend utility was removed.

Folded in from the former test_flow_points.py (axis ordering) and the flow halves
of test_framework.py (with the tautological move-loop bookkeeping asserts removed).
'''

import warnings

import numpy as np
import pytest

import planktos
from planktos.fluid import center_cell_regrid


# --------------------------------------------------------------------------- #
#                          Brinkman flow (analytic)                           #
# --------------------------------------------------------------------------- #

def test_brinkman_2D_static_profile():
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry=('zero', 'zero'),
                                 y_bndry=('noflux', 'zero'), rho=1000, mu=5000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.5, dpdx=0.22306, res=41)
    assert len(envir.L) == 2
    assert envir.flow.flow_times is None, "static flow must have flow_times None"
    assert envir.flow[0].shape == (41, 41)
    # Brinkman flow is unidirectional in x: top of the domain matches U, and the
    # transverse component is identically zero.
    assert np.isclose(envir.flow[0][20, -1], 0.5)
    assert np.allclose(envir.flow[1], 0.0)


def test_brinkman_2D_time_dependent():
    envir = planktos.Environment(rho=1000, mu=20000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.1 * np.arange(-2, 6),
                            dpdx=np.ones(8) * 0.22306, res=41, tspan=[0, 10])
    assert envir.flow.flow_times is not None and len(envir.flow.flow_times) == 8
    assert envir.flow[0].shape == (8, 41, 41)
    assert envir.flow.flow_times[0] == 0 and envir.flow.flow_times[-1] == 10
    # U ramps from -0.2 to 0.5: the top-of-domain flow follows that sign change.
    assert envir.flow[0][0, 20, -1] < 0
    assert np.isclose(envir.flow[0][-1, 20, -1], 0.5)


def test_brinkman_3D_static_shape():
    envir = planktos.Environment(Lx=20, Ly=20, Lz=20,
                                 z_bndry=('noflux', 'noflux'), rho=1000, mu=250000)
    envir.set_brinkman_flow(alpha=66, h_p=6, U=5, dpdx=0.22306, res=21)
    assert envir.flow.flow_times is None
    assert envir.flow[0].shape == (21, 21, 21)
    assert len(envir.L) == 3


# --------------------------------------------------------------------------- #
#                    tile_domain (temporarily unavailable)                    #
# --------------------------------------------------------------------------- #
#
# Tiling was implemented as a virtual ndarray view (FlowArray) reporting a tiled
# shape over a single stored tile. Modern scipy defeats that -- RegularGridInterp-
# olator calls np.asarray on any array-API object, discarding the virtual shape --
# so the tiled interpolation path was broken and untested, and FlowArray has been
# removed. Tiling returns later as a position-wrapping implementation covering 2D
# and 3D. See docs/notes/flow_field_interface.md.
#
# These pin the *interim* contract: a loud failure, and no partial mutation.

def test_tile_domain_raises_not_implemented():
    nx, ny = 21, 21
    x = np.linspace(0, 10, nx); y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(2 * np.pi * X / 10); v = np.cos(2 * np.pi * Y / 8)
    envir = planktos.Environment(Lx=10, Ly=8, flow=[u.copy(), v.copy()])

    with pytest.raises(NotImplementedError):
        envir.tile_domain(2, 2)


def test_tile_domain_leaves_environment_untouched():
    # A half-tiled environment (mesh/L updated, fluid not) would be worse than no
    # tiling at all, so the raise must happen before anything is mutated.
    nx, ny = 11, 11
    x = np.linspace(0, 10, nx); y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[X.copy(), Y.copy()])
    L_before = list(envir.L)
    shape_before = envir.flow[0].shape
    fp_before = [fp.copy() for fp in envir.flow.flow_points]

    with pytest.raises(NotImplementedError):
        envir.tile_domain(3, 2)

    assert envir.L == L_before
    assert envir.flow[0].shape == shape_before
    for before, after in zip(fp_before, envir.flow.flow_points):
        assert np.allclose(before, after)


def test_tile_flow_raises_on_fluiddata_directly():
    nx = 11
    x = np.linspace(0, 10, nx); y = np.linspace(0, 8, nx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[X.copy(), Y.copy()])
    with pytest.raises(NotImplementedError):
        envir.flow.tile_flow(2, 2)


# --------------------------------------------------------------------------- #
#                       extend (pad the flow domain)                          #
# --------------------------------------------------------------------------- #

@pytest.mark.skip(reason="Environment.extend was removed on dyload (extrapolation "
                         "is the intended replacement; see changelog/TODO.md). "
                         "Un-skip if extend is re-added.")
def test_extend_grows_domain_and_copies_edges():
    envir = planktos.Environment(Lx=10, Ly=10, rho=1000, mu=5000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.5, dpdx=0.22306, res=41)
    shape0 = envir.flow[0].shape
    envir.extend(x_minus=3, x_plus=2, y_minus=1, y_plus=5)
    # 5 new columns in x, 6 in y
    assert envir.flow[0].shape == (shape0[0] + 5, shape0[1] + 6)
    assert envir.flow[0].shape == envir.flow[1].shape
    # dx = 10/40 = 0.25, so L grows by 5*0.25 in x and 6*0.25 in y
    assert np.allclose(envir.L, [11.25, 11.5])
    # padded edges copy the original boundary values outward
    assert envir.flow[0][0, 20] == envir.flow[0][3, 20]
    assert envir.flow[0][-1, 20] == envir.flow[0][-3, 20]


# --------------------------------------------------------------------------- #
#                 flow_points axis ordering (regression)                      #
# --------------------------------------------------------------------------- #
# flow_points[i] must be the coordinate array for spatial axis i over length L[i].
# Non-square grids are used so a swapped axis<->coordinate pairing would show.

def test_flow_points_axis_order_static():
    nx, ny = 12, 9
    envir = planktos.Environment(Lx=10, Ly=8, flow=[np.zeros((nx, ny)), np.zeros((nx, ny))])
    assert len(envir.flow.flow_points[0]) == nx and len(envir.flow.flow_points[1]) == ny
    assert np.isclose(envir.flow.flow_points[0][-1], 10) and np.isclose(envir.flow.flow_points[1][-1], 8)


def test_flow_points_axis_order_time_dependent():
    T, nx, ny = 4, 12, 9
    envir = planktos.Environment(Lx=10, Ly=8,
                                 flow=[np.zeros((T, nx, ny)), np.zeros((T, nx, ny))],
                                 flow_times=[0.0, 1.0, 2.0, 3.0])
    assert len(envir.flow.flow_points[0]) == nx and len(envir.flow.flow_points[1]) == ny
    assert np.isclose(envir.flow.flow_points[0][-1], 10) and np.isclose(envir.flow.flow_points[1][-1], 8)


# --------------------------------------------------------------------------- #
#                       channel and canopy flows                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('dims', [
    dict(Lx=20, Ly=10),
    dict(Lx=20, Ly=30, Lz=10, z_bndry=('noflux', 'noflux')),
])
def test_two_layer_channel_flow_shape(dims):
    envir = planktos.Environment(rho=1000, mu=5000, **dims)
    envir.set_two_layer_channel_flow(a=1, h_p=1, Cd=0.25, S=0.1, res=31)
    assert envir.flow.flow_times is None
    assert len(envir.flow[0].shape) == len(envir.L)


def test_canopy_flow_2D_time_dependent_physics():
    envir = planktos.Environment(Lx=50, Ly=40, rho=1000, mu=1000)
    U_h = np.arange(-0.5, 1.2, 0.1); U_h[5] = 0
    envir.set_canopy_flow(h=15, a=1, U_h=U_h, tspan=[0, 20], res=41)
    assert envir.flow.flow_times[-1] == 20 and len(envir.flow.flow_times) == len(U_h)
    assert envir.flow[0].shape == (len(U_h), 41, 41)
    # physically: flow increases with height, increases over time, constant in x
    assert np.all(envir.flow[0][-1, :, -1] > envir.flow[0][-1, :, 20]), "should increase with height"
    assert np.all(envir.flow[0][-1, :, -1] > envir.flow[0][-2, :, -1]), "should increase over time"
    assert np.all(envir.flow[0][0, 0, -1] == envir.flow[0][0, -1, -1]), "should be constant in x"


def test_canopy_flow_3D_time_dependent_physics():
    envir = planktos.Environment(Lx=50, Ly=30, Lz=40, rho=1000, mu=1000)
    U_h = np.arange(-0.5, 1.2, 0.1); U_h[5] = 0
    envir.set_canopy_flow(h=15, a=1, U_h=U_h, tspan=[0, 20], res=31)
    assert len(envir.flow[0].shape) == 4   # (t, x, y, z)
    assert np.all(envir.flow[0][-1, :, :, -1] > envir.flow[0][-1, :, :, 15]), "increase with z"
    assert np.all(envir.flow[0][-1, :, :, -1] > envir.flow[0][-2, :, :, -1]), "increase over time"
    assert np.all(envir.flow[0][0, 0, :, -1] == envir.flow[0][0, -1, :, -1]), "constant in x"


# --------------------------------------------------------------------------- #
#                 flow periodicity at the domain edge (regression)             #
# --------------------------------------------------------------------------- #
# FluidData defaults to NON-periodic. Periodic interpolation wraps the upper grid
# edge to the lower edge (y=L -> y=0 via positions % L), so for a non-periodic
# shear u=(a*y, 0) the velocity at the exact upper edge must be the data value
# a*L, not the wrapped value u_x(y=0)=0. (This wraparound at y=L was what corrupted
# the FTLE boundary row; see tests/test_analysis.py.) Flow periodicity is
# independent of the agent boundary conditions.

def test_flow_non_periodic_by_default_at_upper_edge():
    a, n = 1.0, 11
    x = y = np.linspace(0, 10, n)
    Y = np.meshgrid(x, y, indexing='ij')[1]
    edge = np.array([[5.0, 10.0]])               # exact upper (y) edge

    envir = planktos.Environment(Lx=10, Ly=10, flow=[a * Y, np.zeros_like(Y)])
    assert envir.flow.periodic_dim == (False, False)
    swrm = planktos.Swarm(swarm_size=1, envir=envir, init=edge, seed=1)
    u_x = np.asarray(swrm.get_fluid_drift(0.0, swrm.positions))[0, 0]
    assert np.isclose(u_x, a * 10.0)             # data value at the edge, no wrap


def test_flow_periodic_dim_true_wraps_upper_edge():
    a, n = 1.0, 11
    x = y = np.linspace(0, 10, n)
    Y = np.meshgrid(x, y, indexing='ij')[1]
    edge = np.array([[5.0, 10.0]])

    envir = planktos.Environment(Lx=10, Ly=10, flow=[a * Y, np.zeros_like(Y)],
                                 periodic_dim=True)
    assert envir.flow.periodic_dim == (True, True)
    swrm = planktos.Swarm(swarm_size=1, envir=envir, init=edge, seed=1)
    u_x = np.asarray(swrm.get_fluid_drift(0.0, swrm.positions))[0, 0]
    assert np.isclose(u_x, 0.0)                  # y=L wraps to y=0 where u_x=0


# --------------------------------------------------------------------------- #
#              center_cell_regrid -- cell-centered data to the edges          #
# --------------------------------------------------------------------------- #
# Finite-volume solvers sample at cell centers, so the outermost samples sit
# half a cell inside the domain and the data reports it one cell narrower than
# it is. center_cell_regrid adds a grid plane at each end of each axis, at the
# domain boundary, and extends the field onto it.
#
# The field extension is multilinear, so anything linear in the coordinates is
# reproduced EXACTLY -- which is what most of these assert. Locating the
# boundary is the part that is only exact on a uniform grid; the stretched case
# is pinned separately, along with its warning.


def _centers(lo, hi, n):
    '''Cell centers of n uniform cells spanning [lo, hi].'''
    edges = np.linspace(lo, hi, n+1)
    return (edges[:-1] + edges[1:])/2


def test_regrid_recovers_a_uniform_2d_domain():
    # The whole point: raw centers span [0.125, 0.875] and report a domain of
    # 0.75 where the truth is 1.0. The added planes must land on 0 and 1.
    x, y = _centers(0., 1., 4), _centers(0., 2., 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    flow, pts = center_cell_regrid([X.copy(), Y.copy()], (x, y))
    assert np.isclose(pts[0][0], 0.) and np.isclose(pts[0][-1], 1.)
    assert np.isclose(pts[1][0], 0.) and np.isclose(pts[1][-1], 2.)
    assert np.array_equal(pts[0][1:-1], x) and np.array_equal(pts[1][1:-1], y)
    # u = x and v = y are linear, so the extension reproduces them exactly --
    # including at the four corners, which no single axis sweep reaches.
    NX, NY = np.meshgrid(pts[0], pts[1], indexing='ij')
    assert np.allclose(flow[0], NX)
    assert np.allclose(flow[1], NY)


def test_regrid_recovers_a_uniform_3d_domain():
    x, y, z = _centers(0., 1., 4), _centers(0., 1., 4), _centers(0., 2., 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # a general trilinear field, so the cross terms have to survive too
    f = [1 + 2*X - 3*Y + 0.5*Z + X*Y - 2*Y*Z + X*Y*Z, X - Z, np.zeros_like(X)]
    flow, pts = center_cell_regrid([a.copy() for a in f], (x, y, z))
    assert np.allclose([pts[d][0] for d in range(3)], [0., 0., 0.])
    assert np.allclose([pts[d][-1] for d in range(3)], [1., 1., 2.])
    NX, NY, NZ = np.meshgrid(*pts, indexing='ij')
    assert np.allclose(flow[0],
                       1 + 2*NX - 3*NY + 0.5*NZ + NX*NY - 2*NY*NZ + NX*NY*NZ)
    assert np.allclose(flow[1], NX - NZ)
    assert np.allclose(flow[2], 0.)


def test_regrid_handles_a_leading_time_axis():
    # Time is not a spatial axis: it must come through untouched, with the
    # extension applied independently at each timestep.
    x, y = _centers(0., 1., 4), _centers(0., 1., 3)
    X, Y = np.meshgrid(x, y, indexing='ij')
    t = np.array([0., 1., 4.])
    u = np.stack([tt*X for tt in t])                 # u = t*x
    v = np.stack([Y + tt for tt in t])               # v = y + t
    flow, pts = center_cell_regrid([u, v], (x, y))
    assert flow[0].shape == (3, 6, 5) and flow[1].shape == (3, 6, 5)
    NX, NY = np.meshgrid(pts[0], pts[1], indexing='ij')
    for k, tt in enumerate(t):
        assert np.allclose(flow[0][k], tt*NX)
        assert np.allclose(flow[1][k], NY + tt)


def test_regrid_on_a_rectilinear_grid():
    # The extension this needed: non-uniform interior spacing. The boundary is
    # half the distance to the neighboring center, and a field linear in the
    # coordinate is still exact.
    x = np.array([0.1, 0.2, 0.5, 1.1, 1.3])          # spacing .1 .3 .6 .2
    y = _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    with pytest.warns(UserWarning, match='not uniform'):
        flow, pts = center_cell_regrid([X.copy(), 2*X - 1], (x, y))
    assert np.isclose(pts[0][0], 0.1 - 0.05)         # half of the .1 spacing
    assert np.isclose(pts[0][-1], 1.3 + 0.1)         # half of the .2 spacing
    assert np.allclose(pts[1], np.concatenate(([0.], y, [1.])))
    NX, _ = np.meshgrid(pts[0], pts[1], indexing='ij')
    assert np.allclose(flow[0], NX)
    assert np.allclose(flow[1], 2*NX - 1)


def test_regrid_warns_only_where_the_grid_is_stretched():
    # Locating the boundary is exact on a uniform axis and a guess only on a
    # stretched one, so the warning has to name which.
    x = np.array([0., 1., 3.])                       # stretched
    y = _centers(0., 1., 3)                          # uniform
    X, _ = np.meshgrid(x, y, indexing='ij')
    with pytest.warns(UserWarning, match='not uniform') as rec:
        center_cell_regrid([X.copy(), X.copy()], (x, y))
    assert 'along x, so' in str(rec[0].message)      # x named, y not


def test_regrid_uniform_grid_does_not_warn():
    x, y = _centers(0., 1., 4), _centers(0., 1., 4)
    X, _ = np.meshgrid(x, y, indexing='ij')
    with warnings.catch_warnings():
        warnings.simplefilter('error')               # any warning fails
        center_cell_regrid([X.copy(), X.copy()], (x, y))


def test_regrid_periodic_axis_wraps_instead_of_extrapolating():
    # Both new planes of a periodic axis are the same physical location, so they
    # must hold the same values -- and those come from across the wrap, not from
    # running the interior gradient off the end.
    x = _centers(0., 1., 4)
    y = _centers(0., 1., 3)
    u = np.array([[10., 20., 30.], [1., 1., 1.], [2., 2., 2.], [50., 60., 70.]])
    flow, pts = center_cell_regrid([u.copy(), np.zeros_like(u)], (x, y),
                                   periodic_dim=[True, False])
    assert np.allclose(flow[0][0], flow[0][-1])      # same place, same value
    # uniform spacing, so the wrap value is the plain mean of the two ends
    assert np.allclose(flow[0][0][1:-1], (u[0] + u[-1])/2)
    # extrapolating instead would have run the 10 -> 1 gradient onward and
    # overshot above the first row; wrapping toward 50 cannot.
    assert flow[0][0][1] > u[0, 0]


def test_regrid_periodic_and_extrapolated_axes_commute():
    # Corners are reached by every sweep, so a corner of a mixed grid would
    # depend on the axis order unless the operations commute. They do.
    rng = np.random.default_rng(4)
    x, y, z = _centers(0., 1., 4), _centers(0., 1., 3), _centers(0., 1., 5)
    u = rng.normal(size=(4, 3, 5))
    fwd, pts = center_cell_regrid([u.copy()], (x, y, z),
                                  periodic_dim=[True, False, True])
    # the same problem with the axes reversed, then permuted back
    rev, rpts = center_cell_regrid([u.transpose(2, 1, 0).copy()], (z, y, x),
                                   periodic_dim=[True, False, True])
    assert np.allclose(fwd[0], rev[0].transpose(2, 1, 0))
    for d in range(3):
        assert np.allclose(pts[d], rpts[2-d])


def test_regrid_matches_the_openfoam_boundary_splice_geometry():
    # The step-2 preview, on the real fixture's geometry: OpenFOAMData recovers
    # a 6x6x7 grid over a 1 x 1 x 2 domain by splicing actual boundary patches
    # on. Regridding the interior alone has to agree on the GRID, and on a
    # linear field it agrees on the values too. That it agrees only for a linear
    # field is exactly why patches stay preferred where they exist: they carry
    # the boundary condition the solver applied, and the fixture's no-slip walls
    # are not the linear extension of the interior.
    x = y = np.array([0.125, 0.375, 0.625, 0.875])
    z = np.array([0.2, 0.6, 1.0, 1.4, 1.8])
    X, _, Z = np.meshgrid(x, y, z, indexing='ij')
    t = 3.0
    flow, pts = center_cell_regrid(                  # u = t, v = x, w = t*z
        [np.full_like(X, t), X.copy(), t*Z], (x, y, z))
    assert np.allclose(pts[0], [0., 0.125, 0.375, 0.625, 0.875, 1.])
    assert np.allclose(pts[2], [0., 0.2, 0.6, 1.0, 1.4, 1.8, 2.])
    assert flow[0].shape == (6, 6, 7)
    NX, _, NZ = np.meshgrid(*pts, indexing='ij')
    assert np.allclose(flow[0], t)
    assert np.allclose(flow[1], NX)
    assert np.allclose(flow[2], t*NZ)


def test_regrid_rejects_an_axis_it_cannot_locate_a_boundary_for():
    x, y = np.array([0.5]), _centers(0., 1., 3)
    u = np.zeros((1, 3))
    with pytest.raises(ValueError, match='At least two'):
        center_cell_regrid([u, u.copy()], (x, y))


def test_regrid_rejects_a_component_of_the_wrong_rank():
    x, y = _centers(0., 1., 4), _centers(0., 1., 3)
    bad = np.zeros((4, 3, 2, 2))
    with pytest.raises(ValueError, match='dimensions against'):
        center_cell_regrid([bad, bad.copy()], (x, y))


def test_regrid_does_not_modify_its_inputs():
    x, y = _centers(0., 1., 4), _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    u = [X.copy(), X.copy()]
    u_ref = [a.copy() for a in u]
    xy = (x.copy(), y.copy())
    center_cell_regrid(u, xy)
    for a, b in zip(u, u_ref):
        assert np.array_equal(a, b)
    assert np.array_equal(xy[0], x) and np.array_equal(xy[1], y)


# ---- bounds: the domain edge supplied rather than inferred ------------------
# Inferring the edge from the cell spacing is exact only on a uniform grid.
# A caller that knows the true extent -- OpenFOAMData does, for every face a
# boundary patch covers -- passes it instead, per end.


def test_regrid_uses_supplied_bounds():
    # Bounds put the new plane exactly where told, and the field is extended
    # that far rather than half a cell. The distances here are deliberately not
    # half-cell, so inference would give a different answer.
    x = _centers(0., 1., 4)                          # spacing .25, half-cell .125
    y = _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    flow, pts = center_cell_regrid([X.copy(), 3*X], (x, y),
                                   bounds=[(-1.0, 2.0), (None, None)])
    assert np.isclose(pts[0][0], -1.0) and np.isclose(pts[0][-1], 2.0)
    assert np.isclose(pts[1][0], 0.) and np.isclose(pts[1][-1], 1.)
    # u = x is linear, so extending to -1 and 2 must give exactly -1 and 2
    assert np.allclose(flow[0][0], -1.0)
    assert np.allclose(flow[0][-1], 2.0)
    assert np.allclose(flow[1][0], -3.0)
    assert np.allclose(flow[1][-1], 6.0)


def test_regrid_bounds_may_be_given_for_one_end_only():
    # The loader's own case: some faces covered by a patch, some not.
    x = _centers(0., 1., 4)
    y = _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    flow, pts = center_cell_regrid([X.copy(), X.copy()], (x, y),
                                   bounds=[(None, 3.0), (None, None)])
    assert np.isclose(pts[0][0], 0.)                 # inferred, half a cell out
    assert np.isclose(pts[0][-1], 3.0)               # supplied
    assert np.allclose(flow[0][0], 0.) and np.allclose(flow[0][-1], 3.0)


def test_regrid_supplied_bounds_suppress_the_stretch_warning():
    # The warning is about the inference being a guess. Supply the answer and
    # there is nothing to warn about, however stretched the grid is.
    x = np.array([0., 1., 3., 7.])
    y = _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        center_cell_regrid([X.copy(), X.copy()], (x, y),
                           bounds=[(-0.5, 9.0), (0., 1.)])


def test_regrid_still_warns_for_an_end_left_inferred_on_a_stretched_axis():
    x = np.array([0., 1., 3., 7.])
    y = _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    with pytest.warns(UserWarning, match='not uniform'):
        center_cell_regrid([X.copy(), X.copy()], (x, y),
                           bounds=[(-0.5, None), (0., 1.)])


def test_regrid_rejects_bounds_inside_the_cell_centers():
    # A domain edge that does not contain the cells it bounds is not a domain
    # edge; taken at face value it would make the grid non-monotone.
    x, y = _centers(0., 1., 4), _centers(0., 1., 3)
    X, _ = np.meshgrid(x, y, indexing='ij')
    with pytest.raises(ValueError, match='do not lie outside'):
        center_cell_regrid([X.copy(), X.copy()], (x, y),
                           bounds=[(0.5, 2.0), (None, None)])
