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

import numpy as np
import pytest

import planktos


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
    # transverse component is identically zero. (np.asarray on the transverse
    # component works around a FlowArray<->numpy interop bug where array-wide
    # np.allclose/np.isclose misread the buffer; see TODO.md Phase 0.)
    assert np.isclose(envir.flow[0][20, -1], 0.5)
    assert np.allclose(np.asarray(envir.flow[1]), 0.0)


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
#                       tile_flow (periodic replication)                      #
# --------------------------------------------------------------------------- #

def test_tile_flow_replicates_and_resizes():
    # A flow varying in BOTH x and y, so tiling is a non-trivial check.
    nx, ny = 21, 21
    x = np.linspace(0, 10, nx); y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(2 * np.pi * X / 10); v = np.cos(2 * np.pi * Y / 8)
    envir = planktos.Environment(Lx=10, Ly=8, flow=[u.copy(), v.copy()])

    envir.tile_domain(2, 2)
    assert envir.flow[0].shape == (2 * nx - 1, 2 * ny - 1)
    assert envir.L == [20, 16]
    assert np.isclose(envir.flow.flow_points[0][-1], 20) and np.isclose(envir.flow.flow_points[1][-1], 16)
    # periodic with period (n-1) in each tiled direction
    assert np.allclose(envir.flow[0][0, :], envir.flow[0][nx - 1, :])
    assert np.allclose(envir.flow[1][:, 0], envir.flow[1][:, ny - 1])


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
