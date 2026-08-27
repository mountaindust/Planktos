'''Claim 2 -- a dynamically loaded run plots the way it always did.

    "I should be able to dynamically load fluid data and use plot and plot_all
     like I've always done, with the admitted downside that the fluid data will
     be streamed twice."

Nothing here records anything. That is the point: the unrecorded windowed path
is the fallback, and it has to be a *working* fallback, not merely one that does
not raise. Three things are being asked of it.

* **The picture is right.** A frame drawn while only part of the dataset is
  resident must be the same picture as the one drawn with all of it in memory.
  The comparison used throughout is against ``INUM=True`` -- the whole dataset,
  linear in time -- which is the same interpolation with the same weights over
  the same numbers, so the two must agree exactly and not merely closely.
* **The cost is one more pass, not many.** "Streamed twice" is a specific
  claim. A replay that slid the window backwards and forwards would still draw
  the right picture while costing far more than the user was told.
* **The run survives being drawn.** Replaying moves the window; the simulation
  has to be able to carry on afterwards as though it had not been.
'''

import re
import warnings

import numpy as np
import pytest

import planktos
from planktos import _frames

from _streaming import (IB2D_NDUMPS, LoadCounter, assert_same_run,
                        assert_unchanged, copy_ib2d, ib2d_envir, run, snapshot,
                        vtk3d_envir, walk_frames)


# A goal line for work in progress rather than a regression suite: these run
# whole simulations and are opt-in, via --runstreaming. The members also marked
# slow (the example scripts, the cross-version check, the movie renders) need
# --runslow as well.
pytestmark = pytest.mark.streaming

WINDOW = 4          # INUM: 5 resident time points out of 8
STEPS = 14
DT = 0.5            # the fixture's dumps sit at t = 0, 1, ... 7, so 14
                    #   steps of 0.5 walk the window across the whole series


def _windowed(tmp_path, name='src', INUM=WINDOW, n=5, seed=3):
    envir = ib2d_envir(copy_ib2d(tmp_path, name), INUM=INUM)
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed)
    return envir, swrm


def _swept(tmp_path, name='src', INUM=WINDOW, steps=STEPS):
    '''A finished windowed run whose window has crossed the whole series.'''

    envir, swrm = _windowed(tmp_path, name, INUM=INUM)
    run(swrm, steps, dt=DT)
    assert envir.flow.is_windowed, 'this test needs a sliding window'
    return envir, swrm


def _frame_times(swrm):
    source = _frames.FrameSource(swrm)
    frames = swrm._select_frames(fps=10, playback_rate=1)
    return [source.time(n) for n in frames]


# --------------------------------------------------------------------------- #
#                        the picture must be the same                         #
# --------------------------------------------------------------------------- #

def test_a_windowed_backdrop_equals_the_fully_resident_one(tmp_path):
    # Same data, same linear weights, so exact equality is the right bar.
    envir_w, swrm_w = _swept(tmp_path, 'win')
    envir_r, swrm_r = _windowed(tmp_path, 'res', INUM=True)
    run(swrm_r, STEPS, dt=DT)

    src_w = _frames.FrameSource(swrm_w, fluid='vort')
    src_r = _frames.FrameSource(swrm_r, fluid='vort')
    times = _frame_times(swrm_w)
    assert len(times) > 1
    for t in times:
        np.testing.assert_array_equal(
            src_w.vorticity(t), src_r.vorticity(t),
            err_msg='windowed vorticity differs from resident at t={}'.format(t))


def test_a_windowed_quiver_equals_the_fully_resident_one(tmp_path):
    envir_w, swrm_w = _swept(tmp_path, 'win')
    envir_r, swrm_r = _windowed(tmp_path, 'res', INUM=True)
    run(swrm_r, STEPS, dt=DT)

    src_w = _frames.FrameSource(swrm_w, fluid='quiver')
    src_r = _frames.FrameSource(swrm_r, fluid='quiver')
    src_w.resolve_strides((2, 2))
    src_r.resolve_strides((2, 2))
    for t in _frame_times(swrm_w):
        uw, vw = src_w.quiver(t)
        ur, vr = src_r.quiver(t)
        np.testing.assert_array_equal(uw, ur)
        np.testing.assert_array_equal(vw, vr)


def test_the_statistics_text_is_the_same_windowed_as_resident(tmp_path):
    # _calc_basic_stats reads the per-dump mean cache rather than the field, so
    # a window that has moved on must not change what the box says.
    envir_w, swrm_w = _swept(tmp_path, 'win')
    envir_r, swrm_r = _windowed(tmp_path, 'res', INUM=True)
    run(swrm_r, STEPS, dt=DT)
    for n in range(len(swrm_w.pos_history)):
        a = swrm_w._calc_basic_stats(DIM3=False, t_indx=n)
        b = swrm_r._calc_basic_stats(DIM3=False, t_indx=n)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- #
#                    the cost must be one more pass, not many                 #
# --------------------------------------------------------------------------- #

def test_a_windowed_replay_reads_each_dump_at_most_once(tmp_path):
    # "Streamed twice" is the promise. A replay that walked the window back and
    # forth would draw the right picture at several times the stated price.
    envir, swrm = _swept(tmp_path)
    with LoadCounter(envir) as loads:
        walk_frames(swrm, fluid='vort')
    dumps_read = sum(finish - start + 1 for start, finish in loads.calls)
    assert dumps_read <= IB2D_NDUMPS, (
        'replaying re-read {} dumps from a {}-dump series: {}'.format(
            dumps_read, IB2D_NDUMPS, loads.calls))


def test_a_replay_with_no_backdrop_costs_no_fluid_reads_at_all(tmp_path):
    # With fluid=None the only fluid a frame needs is the component means, and
    # those are cached per dump as the run loads them.
    envir, swrm = _swept(tmp_path)
    with LoadCounter(envir) as loads:
        walk_frames(swrm)
    assert len(loads) == 0, 'a backdrop-free replay read fluid: {}'.format(
        loads.calls)


def test_a_second_replay_costs_no_more_than_the_first(tmp_path):
    # The window ends where the first replay left it, so a second replay must
    # not pay to walk back to the beginning again... which it does, once per
    # replay. This is the "streamed twice" cost stated honestly: it is twice
    # per plot, not twice in total.
    envir, swrm = _swept(tmp_path)
    with LoadCounter(envir) as first:
        walk_frames(swrm, fluid='vort')
    with LoadCounter(envir) as second:
        walk_frames(swrm, fluid='vort')
    assert len(second) <= len(first) + 1


# --------------------------------------------------------------------------- #
#                   the run must survive having been drawn                    #
# --------------------------------------------------------------------------- #

def test_a_windowed_run_continues_identically_after_a_replay(tmp_path):
    # The window is left wherever the last frame put it. Continuing has to
    # reload forwards and give the same numbers as an uninterrupted run.
    _, straight = _windowed(tmp_path, 'a')
    run(straight, STEPS, dt=DT)

    _, interrupted = _windowed(tmp_path, 'b')
    run(interrupted, STEPS // 2, dt=DT)
    interrupted.plot_all(fluid='vort')
    walk_frames(interrupted, fluid='vort')
    interrupted.plot(t=0.0, fluid='quiver')
    run(interrupted, STEPS - STEPS // 2, dt=DT)

    assert_same_run(straight, interrupted)


def test_a_replay_does_not_disturb_the_simulation(tmp_path):
    envir, swrm = _swept(tmp_path)
    before = snapshot(swrm)
    swrm.plot_all(fluid='vort')
    walk_frames(swrm, fluid='vort')
    swrm.plot(t=1.0, fluid='vort')
    assert_unchanged(swrm, before, 'a windowed replay')


def test_drawing_the_first_frame_after_the_window_has_passed_it(tmp_path):
    # The window ends at the far end of the series; frame 0 is at the near end.
    # Nothing may extrapolate: the value must be the dump-0 curl.
    envir_w, swrm_w = _swept(tmp_path, 'win')
    envir_r, swrm_r = _windowed(tmp_path, 'res', INUM=True)
    run(swrm_r, STEPS, dt=DT)
    src_w = _frames.FrameSource(swrm_w, fluid='vort')
    src_r = _frames.FrameSource(swrm_r, fluid='vort')
    np.testing.assert_array_equal(src_w.vorticity(0.0), src_r.vorticity(0.0))


# --------------------------------------------------------------------------- #
#                        it has to say that it is doing it                    #
# --------------------------------------------------------------------------- #

def test_replaying_an_unrecorded_windowed_run_says_it_will_re_read(tmp_path):
    envir, swrm = _swept(tmp_path)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        swrm.plot_all(fluid='vort')
    said = [str(w.message) for w in log if 're-read' in str(w.message)]
    assert said, 'a windowed replay re-read the dataset without saying so'
    assert 'envir.record' in said[0], 'the warning does not name the remedy'


def test_the_warning_counts_dumps_that_exist(tmp_path):
    # The count is the whole content of the warning, so it has to be right: a
    # replay spanning the whole series reads every dump, and no more than exist.
    envir, swrm = _swept(tmp_path)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        swrm.plot_all(fluid='vort')
    said = [str(w.message) for w in log if 're-read' in str(w.message)][0]
    counts = re.search(r"about (\d+) of this dataset's (\d+) dumps", said)
    assert counts is not None, said
    spanned, total = (int(g) for g in counts.groups())
    assert total == IB2D_NDUMPS, said
    assert spanned == IB2D_NDUMPS, \
        'a whole-series replay spans {} of {} dumps: {}'.format(
            spanned, total, said)


# --------------------------------------------------------------------------- #
#                                   3D                                        #
# --------------------------------------------------------------------------- #

def test_a_windowed_3d_run_plots(tmp_path):
    envir = vtk3d_envir(INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=2)
    run(swrm, 12, dt=0.3)
    assert envir.flow.is_windowed
    with LoadCounter(envir) as loads:
        swrm.plot()
        swrm.plot_all()
        walk_frames(swrm)
    # 3D draws no fluid backdrop at all, so a replay has nothing to re-read.
    assert len(loads) == 0, '3D replay read fluid it cannot draw: {}'.format(
        loads.calls)


def test_a_windowed_3d_run_is_undisturbed_by_being_drawn(tmp_path):
    envir = vtk3d_envir(INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=2)
    run(swrm, 12, dt=0.3)
    before = snapshot(swrm)
    swrm.plot_all()
    walk_frames(swrm)
    assert_unchanged(swrm, before, 'a 3D replay')
