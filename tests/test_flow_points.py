'''Regression tests for flow_points axis ordering in _set_flow_variables.

flow_points[i] must be the coordinate array for spatial axis i, over domain
length L[i] -- the convention the data loaders, interpolate_flow,
get_2D_vorticity, and plotting all assume. Previously _set_flow_variables zipped
the REVERSED flow shape against the (non-reversed) domain lengths, which swapped
the axis<->coordinate pairing on non-square grids (square grids hid the bug).
These tests use non-square grids so the pairing is actually exercised.
'''

import numpy as np
import planktos


def test_flow_points_axis_order_static():
    '''Static non-square flow: flow_points[i] matches axis i and spans L[i].'''
    nx, ny = 12, 9
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    envir = planktos.Environment(Lx=10, Ly=8, flow=[u, v])
    assert len(envir.flow_points[0]) == nx        # axis 0 -> x
    assert len(envir.flow_points[1]) == ny        # axis 1 -> y
    assert np.isclose(envir.flow_points[0][-1], 10)  # x spans Lx
    assert np.isclose(envir.flow_points[1][-1], 8)   # y spans Ly


def test_flow_points_axis_order_time_dependent():
    '''Time-varying non-square flow: leading time axis is excluded; spatial
    axes map to flow_points in order.'''
    T, nx, ny = 4, 12, 9
    u = np.zeros((T, nx, ny))
    v = np.zeros((T, nx, ny))
    envir = planktos.Environment(Lx=10, Ly=8, flow=[u, v],
                                 flow_times=[0.0, 1.0, 2.0, 3.0])
    assert len(envir.flow_points[0]) == nx
    assert len(envir.flow_points[1]) == ny
    assert np.isclose(envir.flow_points[0][-1], 10)
    assert np.isclose(envir.flow_points[1][-1], 8)


def test_vorticity_runs_on_nonsquare_constructor_flow():
    '''get_2D_vorticity must work on a non-square flow built via the
    constructor (it previously raised due to swapped flow_points).'''
    nx, ny = 12, 9
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[-Y, X])  # solid-body rotation
    vort = envir.get_2D_vorticity()
    assert vort.shape == (nx, ny)
    assert np.allclose(vort, 2.0, atol=1e-10)
