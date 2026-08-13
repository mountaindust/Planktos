'''Plotting helpers that decide what a frame looks like without drawing one.

_vorticity_norm sets the colour limits of the RdBu vorticity backdrop. It is a
pure function, so unlike the rendering smokes in test_plotting_smoke.py these
stay in the fast run.

(On the dyload branch this section lives in test_frame_selection.py alongside the
frame-selection arithmetic, which master does not have. If the two branches are
ever merged, fold this file into that one rather than keeping both.)
'''

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from planktos._swarm import _vorticity_norm


# --------------------------------------------------------------------------- #
#            vorticity colour limits (the RdBu backdrop of plot_all)          #
# --------------------------------------------------------------------------- #
# RdBu is diverging and white at its midpoint, so limits that are not symmetric
# about zero tint the whole quiescent background. plot_all used to call
# ScalarMappable.autoscale() on every frame, which sets the limits to that
# frame's own min/max: asymmetric, and different every frame. The result was a
# background flashing light red and blue through a video -- and any clip the
# caller passed was silently discarded, since autoscale overwrites it.


def _zero_at_white(norm):
    '''RdBu puts white at the midpoint of the normalized range.'''
    return np.isclose(norm(0.0), 0.5)


def test_vorticity_clip_is_honored_exactly():
    norm = _vorticity_norm(np.array([[0.1, 0.2], [0.3, 0.4]]), clip=2.)
    assert (norm.vmin, norm.vmax) == (-2., 2.)
    assert norm.clip is True
    assert _zero_at_white(norm)


def test_vorticity_clip_is_not_rescaled_by_later_frames():
    # The old autoscale() call replaced the caller's clip with the frame's own
    # extremes, so passing clip did not actually stabilize a movie.
    norm = _vorticity_norm(np.zeros((2, 2)), clip=2.)
    for frame in (np.full((2, 2), 0.01), np.full((2, 2), 50.)):
        norm = _vorticity_norm(frame, clip=2., norm=norm)
        assert (norm.vmin, norm.vmax) == (-2., 2.)


def test_vorticity_limits_are_symmetric_so_zero_is_white():
    # A field that is strongly one-signed is the case that tinted the
    # background: min/max limits put zero far from the colormap midpoint.
    vort = np.array([[0.1, 0.2], [0.3, 5.0]])
    norm = _vorticity_norm(vort)
    assert np.isclose(norm.vmax, 5.0) and np.isclose(norm.vmin, -5.0)
    assert _zero_at_white(norm)


def test_vorticity_limits_grow_but_never_shrink():
    # The flashing fix. A quiet frame following a strong one must not pull the
    # scale back in, or everything drawn against it changes colour.
    norm = _vorticity_norm(np.array([[1.0]]))
    assert np.isclose(norm.vmax, 1.0)
    norm = _vorticity_norm(np.array([[4.0]]), norm=norm)      # grows
    assert np.isclose(norm.vmax, 4.0)
    norm = _vorticity_norm(np.array([[0.001]]), norm=norm)    # must not shrink
    assert np.isclose(norm.vmax, 4.0)
    assert _zero_at_white(norm)


def test_vorticity_all_zero_field_reads_as_white_not_saturated():
    # Limits of (0, 0) would send every cell to the bottom of the colormap, so a
    # motionless fluid would render solid blue rather than blank.
    norm = _vorticity_norm(np.zeros((3, 3)))
    assert norm.vmin < 0 < norm.vmax
    assert _zero_at_white(norm)


def test_vorticity_limits_ignore_non_finite_values():
    vort = np.array([[1.0, np.nan], [np.inf, 2.0]])
    norm = _vorticity_norm(vort)
    assert np.isclose(norm.vmax, 2.0)
    assert _zero_at_white(norm)


def test_vorticity_norm_is_grown_in_place():
    # plot_all hands the live norm back in each frame; replacing the object
    # instead of growing it would drop matplotlib's callbacks on the mappable.
    norm = _vorticity_norm(np.array([[1.0]]))
    same = _vorticity_norm(np.array([[9.0]]), norm=norm)
    assert same is norm and np.isclose(norm.vmax, 9.0)
