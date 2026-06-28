'''Tests for temporal interpolation of fluid data (planktos.fluid).

On dyload each flow array is replaced with an fCubicSpline (a CubicSpline subclass
that also remembers the original data's shape and extrema); FluidData.__init__ does
this internally when given time-varying data (mvbnd's standalone
create_temporal_interpolations helper was absorbed into FluidData). Cubic spline
interpolation is exact for polynomials up to degree 3, so feeding it a known
cubic-in-time gives a machine-precision known answer. Self-contained.
'''

import numpy as np
import pytest

from planktos import fluid


@pytest.fixture
def cubic_flow():
    '''A flow that is an exact, distinct cubic in time at every grid point.'''
    T, nx, ny = 6, 4, 3
    t = np.linspace(0.0, 5.0, T)
    rng = np.random.default_rng(0)
    c = [rng.uniform(-1, 1, (nx, ny)) for _ in range(3)] + [rng.uniform(-0.2, 0.2, (nx, ny))]

    def cubic(tt):
        return c[0] + c[1] * tt + c[2] * tt ** 2 + c[3] * tt ** 3

    flow = np.stack([cubic(tt) for tt in t])      # (T, nx, ny)
    return t, flow, cubic


def test_fluiddata_splines_time_varying_into_fcubicsplines(cubic_flow):
    # FluidData replaces each time-varying flow component with an fCubicSpline
    # (the dyload equivalent of mvbnd's create_temporal_interpolations).
    t, flow, _ = cubic_flow
    nx, ny = flow.shape[1], flow.shape[2]
    fpoints = (np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    fd = fluid.FluidData([flow.copy(), flow.copy()], fpoints, flow_times=t.copy())
    assert all(isinstance(s, fluid.fCubicSpline) for s in fd._flow)
    assert fd._flow[0].shape == flow.shape


def test_spline_reproduces_known_cubic(cubic_flow):
    t, flow, cubic = cubic_flow
    sp = fluid.fCubicSpline(t.copy(), flow.copy())
    for tt in (0.7, 2.3, 4.9):                        # off-node times
        assert np.allclose(sp(tt), cubic(tt), atol=1e-12)


def test_spline_interpolates_nodes_exactly(cubic_flow):
    t, flow, _ = cubic_flow
    sp = fluid.fCubicSpline(t, flow)
    for n in range(len(t)):
        assert np.allclose(sp(t[n]), flow[n])


def test_spline_indexing(cubic_flow):
    t, flow, _ = cubic_flow
    sp = fluid.fCubicSpline(t, flow)
    assert np.allclose(sp[2], flow[2])                # int index -> node value
    assert np.allclose(sp[1:4], flow[1:4])            # slice -> stacked nodes


def test_spline_regenerates_original_data(cubic_flow):
    t, flow, _ = cubic_flow
    sp = fluid.fCubicSpline(t, flow)
    assert np.allclose(sp.regenerate_data(), flow)


def test_spline_extrema(cubic_flow):
    t, flow, _ = cubic_flow
    sp = fluid.fCubicSpline(t, flow)
    assert np.isclose(sp.min(), flow.min())
    assert np.isclose(sp.max(), flow.max())
    assert np.isclose(sp.absmax(), np.abs(flow).max())


def test_spline_is_read_only(cubic_flow):
    t, flow, _ = cubic_flow
    sp = fluid.fCubicSpline(t, flow)
    with pytest.raises(RuntimeError):
        sp[0] = flow[0]
