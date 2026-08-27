'''Claim 3 -- a recorded run replays without re-reading the fluid.

    "I should be able to run a simulation, streaming its results to disk. This
     goes to a structured, well documented directory. If I do not have all the
     fluid data sitting in RAM, I should then be able to use plot and plot_all
     to create figures and videos without restreaming the fluid data. If the
     fluid data is all sitting in RAM, it should be even faster."

Three separable promises, and they are tested separately.

* **The directory is a durable artifact.** It says what made it, it reads back
  in a process that never saw the simulation, and it survives being moved.
* **The picture is the right picture.** Cheap and wrong is not the deal. Every
  backdrop served from the archive is compared against the same run performed
  with the whole dataset resident, and must match exactly -- the stored dumps
  are blended with the same linear weights the velocity itself uses, so there
  is no rounding to allow for.
* **The cost is what was promised.** Zero velocity-dump loads on replay, and
  with the field resident, zero disk reads of any kind -- the archive is not
  even consulted for the backdrop, because the curl of a resident field is
  free and exact.

``plot_all()`` with no ``movie_filename`` draws **nothing** on Agg, so the
frame-level assertions go through ``walk_frames``, which is what ``animate``
does. The end-to-end movie renders are marked slow and need ffmpeg.
'''

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import planktos
from planktos import _frames

from _streaming import (IB2D_NDUMPS, REPO, LoadCounter, VorticityReadCounter,
                        assert_same_state, copy_ib2d, ib2d_envir, run,
                        vtk3d_envir, walk_frames)


# A goal line for work in progress rather than a regression suite: these run
# whole simulations and are opt-in, via --runstreaming. The members also marked
# slow (the example scripts, the cross-version check, the movie renders) need
# --runslow as well.
pytestmark = pytest.mark.streaming

WINDOW = 4
STEPS = 14
DT = 0.5            # dumps sit at t = 0, 1, ... 7; 14 steps of 0.5 cross them all


def _recorded(tmp_path, name='run', INUM=WINDOW, steps=STEPS, seed=3, n=5,
              with_vorticity=True, **kwargs):
    '''A finished recording, with the environment and swarm that made it.'''

    src = copy_ib2d(tmp_path, name + '_src', with_vorticity=with_vorticity)
    envir = ib2d_envir(src, INUM=INUM)
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed)
    with envir.record(str(tmp_path / name), **kwargs) as rec:
        run(swrm, steps, dt=DT)
    return rec, envir, swrm


def _resident_truth(tmp_path, name='truth', steps=STEPS, seed=3, n=5,
                    with_vorticity=True):
    '''The same run with the whole dataset in memory. Nothing is recorded.

    ``with_vorticity`` selects the same fixture the recorded run used -- the two
    ib2d fixtures carry different velocity fields, so mixing them compares two
    different simulations.
    '''

    envir = ib2d_envir(copy_ib2d(tmp_path, name + '_src',
                                 with_vorticity=with_vorticity), INUM=True)
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed)
    run(swrm, steps, dt=DT)
    return envir, swrm


# --------------------------------------------------------------------------- #
#                 the directory is a durable, self-describing thing           #
# --------------------------------------------------------------------------- #

def test_the_archive_directory_has_the_documented_shape(tmp_path):
    rec, envir, swrm = _recorded(tmp_path, fluid=('vort', 'quiver'))
    p = Path(rec.path)
    assert (p / 'meta.json').is_file()
    assert (p / 'grid.npz').is_file()
    assert (p / 'agents' / 'swarm00.json').is_file()
    assert sorted(f.name for f in (p / 'agents').glob('times_*.npy'))
    for short in ('pos', 'vel', 'mask'):
        assert list((p / 'agents').glob('swarm00_{}_*.npy'.format(short))), \
            'no {} chunks written'.format(short)
    assert (p / 'fluid' / 'dump_stats.npz').is_file()


def test_the_archive_says_what_made_it(tmp_path):
    # Without this the directory is a pile of numbers in an unnamed coordinate
    # system. It is also what a restart would have to read (claim 4).
    rec, envir, swrm = _recorded(tmp_path)
    meta = json.loads((Path(rec.path) / 'meta.json').read_text())
    assert meta['version'] >= 1
    assert tuple(meta['store']) == ('positions', 'velocities')
    prov = meta['provenance']
    assert prov['planktos_version'] == planktos.__version__
    assert prov['fluid']['loader'] == 'read_IB2d_fluid_data'
    assert Path(prov['fluid']['kwargs']['path']).name.endswith('_src')
    assert prov['environment']['L'] == list(envir.L)
    assert prov['environment']['units'] == envir.units


def test_the_archive_reads_back_in_a_process_that_never_saw_the_run(tmp_path):
    # The point of writing it down. Nothing in this process may be needed.
    rec, envir, swrm = _recorded(tmp_path)
    expected = tmp_path / 'expected.npy'
    np.save(expected, np.ma.filled(np.ma.stack(swrm.pos_history), np.nan))

    script = tmp_path / 'read_it.py'
    script.write_text(
        'import numpy as np, planktos\n'
        'run = planktos.load_run({archive!r})\n'
        'got = np.ma.filled(run.positions(0).asarray(), np.nan)\n'
        'want = np.load({expected!r})\n'
        'assert got.shape[0] >= want.shape[0], (got.shape, want.shape)\n'
        'np.testing.assert_array_equal(got[:want.shape[0]], want)\n'
        'print("captures", len(run.times))\n'.format(
            archive=str(rec.path), expected=str(expected)))
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_an_archive_that_holds_its_own_vorticity_survives_being_moved(tmp_path):
    # A directory the user can put somewhere and come back to. Only reachable
    # when the fluid source was unwritable, which is the case that has to be
    # self-contained -- a writable source keeps its own Omega series by design.
    src = copy_ib2d(tmp_path, 'src')                 # no Omega files
    envir = ib2d_envir(src, INUM=WINDOW)
    cls = type(envir.flow)
    original = cls.source_dir
    cls.source_dir = lambda self: tmp_path / 'read_only_mount'
    try:
        swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
        with envir.record(str(tmp_path / 'run'), fluid='vort') as rec:
            run(swrm, STEPS, dt=DT)
    finally:
        cls.source_dir = original

    meta = json.loads((Path(rec.path) / 'meta.json').read_text())
    assert meta['fluid']['vorticity'] == 'archive'
    moved = tmp_path / 'somewhere_else'
    shutil.move(str(rec.path), str(moved))
    envir._archive_path = moved
    fields = walk_frames(swrm, fluid='vort')
    assert len(fields) > 1
    assert all(np.isfinite(f).all() for f in fields)


def test_the_archive_can_be_deleted_after_being_plotted(tmp_path):
    # A memory map holds its file open, and on Windows that locks it. Anything
    # the render maps has to be released, or a user cannot tidy up after a plot.
    rec, envir, swrm = _recorded(tmp_path)
    walk_frames(swrm, fluid='vort')
    swrm.plot_all(fluid='vort')
    shutil.rmtree(rec.path)
    assert not Path(rec.path).exists()


# --------------------------------------------------------------------------- #
#                     the picture is the right picture                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('with_vorticity', [True, False],
                         ids=['solver_wrote_it', 'planktos_wrote_it'])
def test_a_recorded_backdrop_equals_the_fully_resident_truth(tmp_path,
                                                             with_vorticity):
    rec, envir, swrm = _recorded(tmp_path, with_vorticity=with_vorticity)
    _, truth = _resident_truth(tmp_path, with_vorticity=with_vorticity)

    got = walk_frames(swrm, fluid='vort')
    want = walk_frames(truth, fluid='vort')
    assert len(got) == len(want) > 1
    # The fixture's own Omega series is ascii, as IB2d writes it -- about 12
    # significant digits. That, not the blend, is the floor on agreement.
    for n, (a, b) in enumerate(zip(got, want)):
        scale = max(np.abs(b).max(), 1e-12)
        assert np.abs(a - b).max() < 1e-9 * scale, \
            'frame {} differs by {:g}'.format(n, np.abs(a - b).max())


def test_a_recorded_quiver_equals_the_fully_resident_truth(tmp_path):
    rec, envir, swrm = _recorded(tmp_path, fluid='quiver')
    _, truth = _resident_truth(tmp_path)

    stored = json.loads((Path(rec.path) / 'meta.json').read_text())
    strides = tuple(stored['fluid']['quiver_strides'])

    got = walk_frames(swrm, fluid='quiver', strides=strides)
    want = walk_frames(truth, fluid='quiver', strides=strides)
    assert len(got) == len(want) > 1
    for n, ((ua, va), (ub, vb)) in enumerate(zip(got, want)):
        np.testing.assert_allclose(ua, ub, rtol=1e-12, atol=1e-13,
                                   err_msg='frame {} u differs'.format(n))
        np.testing.assert_allclose(va, vb, rtol=1e-12, atol=1e-13,
                                   err_msg='frame {} v differs'.format(n))


def test_the_agent_states_drawn_are_the_ones_the_archive_holds(tmp_path):
    # Capture 0 is taken by record() itself, before any step, so the archive
    # holds every history state and then the present -- exactly the states a
    # plot can draw.
    rec, envir, swrm = _recorded(tmp_path)
    archive = planktos.load_run(rec.path)
    try:
        stored = archive.positions(0)
        assert len(stored) == len(swrm.pos_history) + 1
        for j in range(len(swrm.pos_history)):
            assert_same_state(stored[j], swrm.pos_history[j],
                              'capture {}'.format(j))
        assert_same_state(stored[-1], swrm.positions, 'the last capture')
        np.testing.assert_allclose(
            archive.times,
            list(envir.time_history) + [envir.time], rtol=0, atol=0)
    finally:
        archive.close()


@pytest.mark.parametrize('k', [1, 3])
def test_a_capture_schedule_keeps_the_archive_and_the_history_in_step(tmp_path, k):
    rec, envir, swrm = _recorded(tmp_path, capture_interval=k, steps=12)
    archive = planktos.load_run(rec.path)
    try:
        stored = archive.positions(0)
        assert len(stored) == len(swrm.pos_history) + 1
        for j in range(len(swrm.pos_history)):
            assert_same_state(stored[j], swrm.pos_history[j], 'capture {}'.format(j))
        assert_same_state(stored[-1], swrm.positions, 'the last capture')
    finally:
        archive.close()
    # and the frames a plot draws are those same states
    walk_frames(swrm, fluid='vort')


def test_a_recording_that_spans_several_chunks_reads_back_and_draws(tmp_path):
    # Everything else here fits in one chunk. The reader memory-maps chunks and
    # caches a handful of them, and a render walks captures in order across the
    # boundaries, so a run long enough to cross several is its own case.
    rec, envir, swrm = _recorded(tmp_path, chunk_size=4, steps=STEPS)
    assert len(list((Path(rec.path) / 'agents').glob('times_*.npy'))) > 2
    archive = planktos.load_run(rec.path)
    try:
        stored = archive.positions(0)
        for j in range(len(swrm.pos_history)):
            assert_same_state(stored[j], swrm.pos_history[j],
                              'capture {}'.format(j))
    finally:
        archive.close()
    assert len(walk_frames(swrm, fluid='vort')) > 1


def test_a_moving_immersed_boundary_is_recorded_and_drawn(tmp_path):
    # animate() interpolates the mesh at every frame, which is a fluid-adjacent
    # read the archive says nothing about. It must still work under a recording,
    # and it must not pull velocity dumps.
    src = copy_ib2d(tmp_path, 'src', with_vorticity=True)
    envir = ib2d_envir(src, INUM=WINDOW)
    envir.read_IB2d_mesh_data(str(REPO / 'tests' / 'fixtures' / 'lagspts_min'),
                              dt=0.1, print_dump=1, d_start=0)
    assert envir.ibmesh.ndim == 4, 'expected a moving mesh'
    swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run'), fluid='vort'):
        run(swrm, 4, dt=0.05)          # stay inside the mesh's own time series
    with LoadCounter(envir) as loads:
        walk_frames(swrm, fluid='vort')
    assert len(loads) == 0


def test_two_recorded_swarms_are_both_stored_and_both_draw(tmp_path):
    src = copy_ib2d(tmp_path, 'src', with_vorticity=True)
    envir = ib2d_envir(src, INUM=WINDOW)
    a = planktos.Swarm(swarm_size=4, envir=envir, seed=1)
    b = planktos.Swarm(swarm_size=6, envir=envir, seed=2)
    with envir.record(str(tmp_path / 'run')) as rec:
        for _ in range(STEPS):
            envir.move_swarms(DT, silent=True)

    archive = planktos.load_run(rec.path)
    try:
        assert [i for _, i in archive.swarms] == [0, 1]
        assert archive.positions(0).asarray().shape[1] == 4
        assert archive.positions(1).asarray().shape[1] == 6
    finally:
        archive.close()
    for s in (a, b):
        walk_frames(s, fluid='vort')


def test_frames_drawn_out_of_order_are_still_the_right_frames(tmp_path):
    # The stored-dump readers keep two slots and evict what is furthest from
    # what was just asked for, which is tuned for a monotone sweep. A caller
    # passing frames= can ask in any order; that may cost more reads, but it
    # must not change a single value.
    rec, envir, swrm = _recorded(tmp_path)
    n = len(swrm.pos_history)
    forward = list(range(0, n, 2))
    shuffled = forward[::-1][:3] + forward[:3] + forward[::-1]

    ordered = dict(zip(forward, walk_frames(swrm, fluid='vort',
                                            frames=forward)))
    jumbled = dict(zip(shuffled, walk_frames(swrm, fluid='vort',
                                             frames=shuffled)))
    for j in forward:
        np.testing.assert_array_equal(jumbled[j], ordered[j],
                                      err_msg='frame {} differs'.format(j))


def test_a_single_past_frame_can_be_drawn_from_the_archive(tmp_path):
    # plot(t=...) rather than plot_all: one frame, snapped to the nearest
    # capture, at a time the window has long since left behind.
    rec, envir, swrm = _recorded(tmp_path)
    _, truth = _resident_truth(tmp_path)
    source = _frames.FrameSource(swrm, fluid='vort')
    ref = _frames.FrameSource(truth, fluid='vort')
    for t in (0.4, 2.6, 5.1):
        j = source.capture_at(t)
        assert j == ref.capture_at(t)
        got, want = source.vorticity(source.time(j)), ref.vorticity(ref.time(j))
        scale = max(np.abs(want).max(), 1e-12)
        assert np.abs(got - want).max() < 1e-9 * scale
    with LoadCounter(envir) as loads:
        swrm.plot(t=2.6, fluid='vort')
    assert len(loads) == 0


def test_an_analysis_only_archive_still_draws_from_live_history(tmp_path):
    # store=('positions',) warns that the archive cannot be plotted from. The
    # backdrop still can be, because agent state comes from the Swarm and only
    # the fluid comes from disk.
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        rec, envir, swrm = _recorded(tmp_path, store=('positions',), steps=6)
    assert any('velocities' in str(w.message) for w in log)
    assert len(walk_frames(swrm, fluid='vort')) > 1


# --------------------------------------------------------------------------- #
#                        the cost is what was promised                        #
# --------------------------------------------------------------------------- #

def test_replaying_a_recorded_windowed_run_reads_no_velocity_dumps(tmp_path):
    rec, envir, swrm = _recorded(tmp_path, fluid=('vort', 'quiver'))
    with LoadCounter(envir) as loads:
        walk_frames(swrm, fluid='vort')
        walk_frames(swrm, fluid='quiver')
    assert len(loads) == 0, 'a recorded replay re-read the fluid: {}'.format(
        loads.calls)


def test_recording_and_replaying_costs_one_pass_over_the_fluid_in_total(tmp_path):
    # The headline, stated as a total rather than as two halves: taping the run
    # and then drawing it must cost exactly what running it cost.
    src = copy_ib2d(tmp_path, 'bare_src', with_vorticity=True)
    envir_bare = ib2d_envir(src, INUM=WINDOW)
    swrm_bare = planktos.Swarm(swarm_size=5, envir=envir_bare, seed=3)
    with LoadCounter(envir_bare) as bare:
        run(swrm_bare, STEPS, dt=DT)
    bare_calls = len(bare)

    src2 = copy_ib2d(tmp_path, 'taped_src', with_vorticity=True)
    envir = ib2d_envir(src2, INUM=WINDOW)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with LoadCounter(envir) as taped:
        with envir.record(str(tmp_path / 'run'), fluid='vort'):
            run(swrm, STEPS, dt=DT)
        walk_frames(swrm, fluid='vort')
    assert len(taped) == bare_calls, (
        'recording plus replay cost {} loads where the bare run cost {}'.format(
            len(taped), bare_calls))


def test_a_resident_recorded_run_touches_no_files_at_all(tmp_path):
    # The "even faster in RAM" half: with the whole field in memory the curl is
    # derived from it, so the archive is not consulted for the backdrop and no
    # stored vorticity is read either.
    rec, envir, swrm = _recorded(tmp_path, INUM=None)
    source = _frames.FrameSource(swrm, fluid='vort')
    assert source._vorticity_from == 'field'
    with LoadCounter(envir) as loads, VorticityReadCounter(envir) as reads:
        walk_frames(swrm, fluid='vort')
    assert len(loads) == 0 and len(reads) == 0, \
        'a resident replay read {} dumps and {} vorticity files'.format(
            len(loads), len(reads))


def test_a_windowed_replay_reads_each_stored_vorticity_dump_once(tmp_path):
    # A monotone sweep through a two-slot cache. More than one read per dump
    # means the cache is not doing its job and the replay is O(frames) in I/O.
    rec, envir, swrm = _recorded(tmp_path)
    with VorticityReadCounter(envir) as reads:
        walk_frames(swrm, fluid='vort')
    assert len(reads) == len(set(reads.reads)), \
        'a dump was read more than once: {}'.format(reads.reads)
    assert len(reads) <= IB2D_NDUMPS


def test_a_3d_recorded_run_carries_its_statistics_and_draws_free(tmp_path):
    envir = vtk3d_envir(INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=2)
    with envir.record(str(tmp_path / 'run3d')) as rec:
        run(swrm, 12, dt=0.3)
    stats = planktos.load_run(rec.path).dump_stats()
    assert stats is not None and stats['means'].shape[1] == 3
    with LoadCounter(envir) as loads:
        walk_frames(swrm)
    assert len(loads) == 0


# --------------------------------------------------------------------------- #
#                          the scales the render shares                       #
# --------------------------------------------------------------------------- #

def test_two_renders_of_the_same_run_share_a_colour_limit(tmp_path):
    rec, envir, swrm = _recorded(tmp_path)
    a = _frames.FrameSource(swrm, fluid='vort').vort_clip
    envir.flow(7.0)          # look at fluid unrelated to the recorded frames
    b = _frames.FrameSource(swrm, fluid='vort').vort_clip
    assert a is not None
    assert a == b


def test_two_renders_of_the_same_run_share_an_arrow_scale(tmp_path):
    rec, envir, swrm = _recorded(tmp_path, fluid='quiver', steps=6)
    a = _frames.FrameSource(swrm, fluid='quiver').quiver_scale
    envir.flow(7.0)          # fluid the recorded stretch of the run never used
    b = _frames.FrameSource(swrm, fluid='quiver').quiver_scale
    assert a == b, 'arrow scale moved from {} to {}'.format(a, b)
    # ...and fmax did move, so this is not two readings of one drifting number.
    assert not np.isclose(np.linalg.norm(np.array(envir.flow.fmax)), a)


def test_the_arrow_scale_is_the_one_the_archive_recorded(tmp_path):
    # The value the specification names, stated on its own so that the xfail
    # above says which of the two numbers is wanted.
    rec, envir, swrm = _recorded(tmp_path, fluid='quiver', steps=6)
    stats = planktos.load_run(rec.path).dump_stats()
    want = float(np.linalg.norm(np.nanmax(stats['vmax'], axis=0)))
    got = _frames.FrameSource(swrm, fluid='quiver').quiver_scale
    assert got == pytest.approx(want)


# --------------------------------------------------------------------------- #
#                     what happens when it cannot be done                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('recorded,asked', [('vort', 'quiver'),
                                            ('quiver', 'vort'),
                                            (None, 'vort')])
def test_a_backdrop_the_recording_lacks_is_refused_by_name(tmp_path, recorded,
                                                           asked):
    rec, envir, swrm = _recorded(tmp_path, fluid=recorded, steps=6)
    with pytest.raises(ValueError) as err:
        _frames.FrameSource(swrm, fluid=asked)
    assert asked in str(err.value)
    assert 'INUM=None' in str(err.value), 'the refusal does not name a remedy'


def test_an_archive_that_has_gone_missing_falls_back_with_a_warning(tmp_path):
    rec, envir, swrm = _recorded(tmp_path)
    shutil.rmtree(rec.path)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        _frames.FrameSource(swrm, fluid='vort')
    said = [str(w.message) for w in log if 'cannot be used' in str(w.message)]
    assert said, 'a vanished archive was passed over in silence'


def test_frames_outside_the_recorded_dumps_are_refused_before_any_are_drawn(
        tmp_path):
    src = copy_ib2d(tmp_path, 'src')
    envir = ib2d_envir(src, INUM=WINDOW)
    cls = type(envir.flow)
    original = cls.source_dir
    cls.source_dir = lambda self: tmp_path / 'read_only_mount'
    try:
        swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
        with envir.record(str(tmp_path / 'run'), fluid='vort'):
            run(swrm, 6, dt=DT)          # recording stops at t = 3
    finally:
        cls.source_dir = original
    run(swrm, 8, dt=DT)                  # the run carries on to t = 7

    # Refused while the source is being built, so no frame is ever drawn --
    # which for a long movie is the difference between a message and an hour.
    with pytest.raises(ValueError) as err:
        _frames.FrameSource(swrm, fluid='vort')
    assert 'record' in str(err.value).lower()
    assert 'INUM=None' in str(err.value), 'the refusal does not name a remedy'


# --------------------------------------------------------------------------- #
#                        end to end, with real pixels                         #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not on PATH')
@pytest.mark.parametrize('fluid', ['vort', 'quiver'])
def test_a_movie_of_a_recorded_windowed_run_reads_no_velocity_dumps(tmp_path,
                                                                    fluid):
    # The whole claim in one call: ffmpeg walks every frame for real.
    rec, envir, swrm = _recorded(tmp_path, fluid=('vort', 'quiver'))
    out = tmp_path / 'movie.mp4'
    with LoadCounter(envir) as loads:
        swrm.plot_all(movie_filename=str(out), fps=2, fluid=fluid)
    assert out.is_file() and out.stat().st_size > 0
    assert len(loads) == 0, 'rendering re-read the fluid: {}'.format(loads.calls)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not on PATH')
def test_record_can_render_the_movie_itself_when_the_block_ends(tmp_path):
    src = copy_ib2d(tmp_path, 'src', with_vorticity=True)
    envir = ib2d_envir(src, INUM=WINDOW)
    swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
    out = tmp_path / 'auto.mp4'
    with envir.record(str(tmp_path / 'run'), fluid='vort',
                      plot_all=dict(movie_filename=str(out), fps=2,
                                    fluid='vort')):
        run(swrm, STEPS, dt=DT)
    assert out.is_file() and out.stat().st_size > 0
