'''Animation frame selection: which recorded states become video frames.

plot_all lays frames down at a fixed interval of simulated time,
dt_frame = playback_rate/fps, and shows the recorded state nearest to each.
Swarm._select_frames is pure arithmetic over the time history, so these tests
run real (tiny) simulations but render nothing, and stay in the fast run. The
rendering side of plot_all is covered by test_plotting_smoke.py.

See docs/notes/flow_field_interface.md section 8.3.5.
'''

import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import planktos


def _run(dt=0.025, steps=48):
    '''A swarm advanced `steps` times, recording a state at each step.

    Its recorded times are 0, dt, ..., steps*dt: the position history plus the
    present state, so frame indices run 0..steps.
    '''
    swrm = planktos.Swarm(swarm_size=5, envir=planktos.Environment(), seed=1)
    for _ in range(steps):
        swrm.move(dt, silent=True)
    return swrm


# --------------------------------------------------------------------------- #
#                            the selection itself                             #
# --------------------------------------------------------------------------- #

def test_frames_are_spaced_by_playback_rate_over_fps():
    # dt_frame = 1/10 = 0.1 s = 4 timesteps at dt=0.025
    frames = _run(0.025, 48)._select_frames(fps=10, playback_rate=1)
    assert np.array_equal(frames, np.arange(0, 49, 4))


def test_playback_rate_sets_the_speed_at_fixed_fps():
    swrm = _run(0.025, 48)
    # twice as fast -> twice the simulated time per frame -> half the frames
    slow = swrm._select_frames(fps=10, playback_rate=1)
    fast = swrm._select_frames(fps=10, playback_rate=2)
    assert np.array_equal(fast, np.arange(0, 49, 8))
    assert len(fast) < len(slow)


def test_fps_sets_the_smoothness_at_fixed_playback_rate():
    swrm = _run(0.025, 48)
    coarse = swrm._select_frames(fps=5, playback_rate=1)
    fine = swrm._select_frames(fps=10, playback_rate=1)
    # both span the same simulated time; the finer rate uses more frames
    assert coarse[0] == fine[0] == 0
    assert coarse[-1] == fine[-1] == 48
    assert len(fine) == 2*len(coarse) - 1


def test_frames_index_the_position_history_and_the_present_state():
    swrm = _run(0.1, 4)
    frames = swrm._select_frames(fps=10, playback_rate=1)  # dt_frame == dt
    assert np.array_equal(frames, np.arange(5))
    assert frames[-1] == len(swrm.pos_history)   # the present state
    # (the time history accumulates dt, so it carries roundoff -- 0.3 arrives
    # as 0.30000000000000004. the selection must not care.)
    assert np.allclose(swrm.envir.time_history, [0.0, 0.1, 0.2, 0.3])


def test_first_and_last_states_are_always_frames():
    # 4.0 s of simulated time does not divide evenly by dt_frame = 3.0 s
    frames = _run(1.0, 4)._select_frames(fps=1, playback_rate=3)
    assert np.array_equal(frames, [0, 3, 4])


def test_frames_snap_to_the_nearest_recorded_state():
    # dt_frame = 1/30 s is 4/3 of dt, so frame k lands on round(4k/3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frames = _run(0.025, 12)._select_frames(fps=30, playback_rate=1)
    expected = np.unique(np.round(np.arange(10) * 4/3).astype(int))
    assert np.array_equal(frames[:len(expected)], expected)
    # every frame is a real recorded state, in order, no repeats
    assert np.all(np.diff(frames) > 0)
    assert frames[-1] == 12


def test_the_timestep_need_not_have_been_constant():
    # a run whose timestep changed partway through: states at 0, .1, ... .4,
    # then .6, .8, 1.0
    swrm = planktos.Swarm(swarm_size=5, envir=planktos.Environment(), seed=1)
    for _ in range(4):
        swrm.move(0.1, silent=True)
    for _ in range(3):
        swrm.move(0.2, silent=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frames = swrm._select_frames(fps=5, playback_rate=1)   # dt_frame=0.2
    times = np.append(swrm.envir.time_history, swrm.envir.time)
    assert np.allclose(times[frames], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


# --------------------------------------------------------------------------- #
#          the failure mode: asking for frames between recorded states        #
# --------------------------------------------------------------------------- #

def test_finer_than_the_recording_interval_clamps_and_warns():
    with pytest.warns(UserWarning, match='one frame per recorded state'):
        frames = _run(0.025, 20)._select_frames(fps=60, playback_rate=1)
    # clamped to every recorded state, and not one repeated
    assert np.array_equal(frames, np.arange(21))


def test_clamp_warning_reports_the_achieved_rate_and_the_usable_fps():
    with pytest.warns(UserWarning) as record:
        _run(0.025, 20)._select_frames(fps=60, playback_rate=1)
    msg = str(record[0].message)
    assert '1.5' in msg     # achieved playback rate: dt_state*fps = 0.025*60
    assert '40' in msg      # usable fps at the requested rate: 1/0.025


def test_frame_interval_equal_to_the_recording_interval_is_not_clamped():
    # the exact-fit case, and one where playback_rate/fps is inexact in binary
    # (0.075/3 != 0.025 to the last bit) -- roundoff must not trip the clamp
    swrm = _run(0.025, 48)
    for fps, rate in ((40, 1), (3, 0.075)):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            frames = swrm._select_frames(fps=fps, playback_rate=rate)
        assert np.array_equal(frames, np.arange(49))


def test_uneven_frame_spacing_warns():
    # dt_frame = 1.33*dt: frames alternate between 1 and 2 states apart
    with pytest.warns(UserWarning, match='slightly uneven'):
        _run(0.025, 48)._select_frames(fps=30, playback_rate=1)


def test_a_whole_multiple_of_the_recording_interval_does_not_warn():
    swrm = _run(0.025, 48)
    for fps in (10, 5, 4, 2):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            swrm._select_frames(fps=fps, playback_rate=1)


def test_large_non_integer_multiples_do_not_warn():
    # 10.5x the recording interval: the jitter is a twentieth of a frame, not
    # worth a warning. only small multiples are.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _run(0.025, 200)._select_frames(fps=40, playback_rate=10.5)


# --------------------------------------------------------------------------- #
#                              degenerate input                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kwargs', [{'fps': 0, 'playback_rate': 1},
                                    {'fps': -5, 'playback_rate': 1},
                                    {'fps': 10, 'playback_rate': 0},
                                    {'fps': 10, 'playback_rate': -1}])
def test_non_positive_rates_raise(kwargs):
    with pytest.raises(ValueError):
        _run(0.1, 4)._select_frames(**kwargs)


def test_a_run_that_has_not_moved_is_a_single_frame():
    swrm = _run(0.1, 0)
    assert np.array_equal(swrm._select_frames(fps=10, playback_rate=1), [0])


def test_dt_frame_longer_than_the_run_still_gives_both_ends():
    frames = _run(0.1, 4)._select_frames(fps=1, playback_rate=100)
    assert np.array_equal(frames, [0, 4])
