'''Drawing a run whose fluid was recorded -- component C of
docs/notes/run_persistence.md.

Where test_recording.py and test_fluid_recording.py pin what an archive holds,
this pins what a plot does with it. One rule decides most of it, and it is about
the fluid alone: a render must not read the fluid dataset without saying so.
Agent state is small and in memory, so it always comes from the Swarm's own
history; the backdrop does not, and under dynamic loading pulling it at every
frame re-reads the whole dataset.

So the headline is the same shape as the recording ones, a step further along:
once a run has been recorded, replaying it costs zero loader calls, where the
same replay unrecorded costs a full second pass. Around it sit the refusals that
keep that true and the global colour and arrow scales.

Most of this drives _frames.FrameSource rather than a figure, for the reason
test_frame_selection.py does: it is what decides a frame, and the source is
exactly what animate() calls. The tests that need pixels are marked slow.

Fixture: ib2d_fluid_vort_min, as in test_fluid_recording.py -- u dumps plus
IB2d's own Omega.####.vtk, with a vorticity that varies nonlinearly in time.
Copies with the Omega files dropped give the "solver printed none" regime, and
every test that writes does so into a temporary directory.
'''

import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import planktos
from planktos import _frames
from planktos._swarm import _vorticity_norm

FIXTURES = Path(__file__).parent / 'fixtures'
VORT_FIXTURE = FIXTURES / 'ib2d_fluid_vort_min'

IB2D_DT = 0.1
IB2D_PRINT_DUMP = 10
NDUMPS = 8


# --------------------------------------------------------------------------- #
#                                  helpers                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _envir(src, INUM=4):
    '''An IB2d environment on the fixture.

    Periodic agent boundaries so agents wrap and stay in the domain: once every
    agent has left, move() stops asking for fluid and the window stops sliding.
    '''

    envir = planktos.Environment(x_bndry='periodic', y_bndry='periodic')
    envir.read_IB2d_fluid_data(str(src), dt=IB2D_DT,
                               print_dump=IB2D_PRINT_DUMP, INUM=INUM)
    return envir


def _copy_fixture(tmp_path, name, with_vorticity=False):
    '''The fixture somewhere writable, with or without its Omega series.'''

    dest = tmp_path / name
    dest.mkdir(parents=True)
    for f in sorted(VORT_FIXTURE.glob('u.*.vtk')):
        shutil.copy(f, dest)
    if with_vorticity:
        for f in sorted(VORT_FIXTURE.glob('Omega.*.vtk')):
            shutil.copy(f, dest)
    return dest


def _sweep(envir, steps=14, dt=0.5, n=5, swrm=None):
    '''Run far enough that the window slides across the whole series.'''

    if swrm is None:
        swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=3)
    for _ in range(steps):
        swrm.move(dt, silent=True)
    return swrm


def _recorded_run(tmp_path, name='run', INUM=4, steps=14, **kwargs):
    '''A finished recording, and the environment and swarm that made it.'''

    src = _copy_fixture(tmp_path, name + '_src', with_vorticity=True)
    envir = _envir(src, INUM=INUM)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / name), **kwargs) as rec:
        _sweep(envir, steps=steps, swrm=swrm)
    return rec, envir, swrm


def _count_loads(envir):
    '''Watch load_dumpfiles on this environment's fluid. Returns (calls, undo).'''

    cls = type(envir.flow)
    original = cls.load_dumpfiles
    calls = []

    def counted(self, d_start, d_finish):
        calls.append((d_start, d_finish))
        return original(self, d_start, d_finish)

    cls.load_dumpfiles = counted
    return calls, lambda: setattr(cls, 'load_dumpfiles', original)


def _walk(swrm, source, frames, fluid):
    '''Read every frame the way animate does, and hand back the fields.'''

    out = []
    for n in frames:
        t = source.time(n)
        swrm._calc_basic_stats(DIM3=False, t_indx=n)
        out.append(source.vorticity(t) if fluid == 'vort' else source.quiver(t))
    return out


# --------------------------------------------------------------------------- #
#           the property the whole component exists for: free replay          #
# --------------------------------------------------------------------------- #

def test_replaying_a_recorded_run_costs_no_fluid_loads(tmp_path):
    # The headline, and the ordinary workflow: record inside a with block, plot
    # after it. Under a sliding window the velocity a backdrop needs is no
    # longer resident, so deriving it would drag load_dumpfiles behind it -- a
    # second streaming pass to draw a picture of data that was in memory once.
    rec, envir, swrm = _recorded_run(tmp_path, fluid=('vort', 'quiver'))

    calls, undo = _count_loads(envir)
    try:
        source = _frames.FrameSource(swrm, fluid='vort')
        frames = swrm._select_frames(fps=2, playback_rate=1)
        # Guard: a one-frame replay would pass this trivially, and so would a
        # dataset that never streamed in the first place.
        assert len(frames) > NDUMPS
        assert envir.flow.is_windowed
        _walk(swrm, source, frames, 'vort')
        _walk(swrm, _frames.FrameSource(swrm, fluid='quiver'), frames, 'quiver')
    finally:
        undo()

    assert calls == []


def test_replaying_an_unrecorded_run_costs_a_second_streaming_pass(tmp_path):
    # The counterpart, which is what makes the zero above mean something.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = _sweep(envir)

    calls, undo = _count_loads(envir)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source = _frames.FrameSource(swrm, fluid='vort')
            frames = swrm._select_frames(fps=2, playback_rate=1)
            _walk(swrm, source, frames, 'vort')
    finally:
        undo()

    assert len(calls) > 0


def test_the_archive_is_found_without_being_named(tmp_path):
    # A user knows they recorded a run; they should not also have to tell the
    # plot where it went. The Environment remembers, and keeps remembering after
    # the recording stops.
    rec, envir, swrm = _recorded_run(tmp_path)

    assert envir._recorder is None                  # the with block ended
    assert envir._archive_path == rec.path
    assert _frames.FrameSource(swrm).run is not None


def test_a_redirected_recording_is_the_one_remembered(tmp_path):
    # Recording into a non-empty directory writes to a timestamped sibling, and
    # that is the archive a later plot has to read.
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'occupied.txt').write_text('x')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rec, envir, swrm = _recorded_run(tmp_path, steps=2)

    assert rec.path != tmp_path / 'run'
    assert envir._archive_path == rec.path
    assert _frames.FrameSource(swrm).run.path == rec.path


def test_an_archive_describing_another_fluid_is_passed_over(tmp_path):
    # Loading new fluid after recording leaves the archive describing a run that
    # no longer matches, so it is reported and the field is used instead.
    rec, envir, swrm = _recorded_run(tmp_path, INUM=None, steps=4)
    other = _copy_fixture(tmp_path, 'other', with_vorticity=True)
    envir.read_IB2d_fluid_data(str(other), dt=IB2D_DT,
                               print_dump=IB2D_PRINT_DUMP, INUM=None)
    envir.L = [envir.L[0]*2, envir.L[1]]            # a different domain

    with pytest.warns(UserWarning, match='cannot be used for this plot'):
        source = _frames.FrameSource(swrm, fluid='vort')
    assert source.run is None


def test_a_render_with_no_archive_warns_that_it_will_re_read(tmp_path):
    # Plotting a dynamically loaded run that was never recorded still works. It
    # just costs a second pass, and nobody should find that out afterwards.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = _sweep(envir)

    source = _frames.FrameSource(swrm, fluid='vort')
    frames = swrm._select_frames(fps=2, playback_rate=1)
    with pytest.warns(UserWarning, match='re-read about'):
        source.warn_if_restreaming([source.time(n) for n in frames])


def test_a_single_frame_look_back_does_not_warn(tmp_path):
    # Two dumps is not a second pass, and a warning that fires on every plot is
    # one nobody reads.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = _sweep(envir)

    source = _frames.FrameSource(swrm, fluid='vort')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        source.warn_if_restreaming([envir.time])


def test_a_resident_field_never_warns(tmp_path):
    # Replay of an INUM=None run costs nothing and always did.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = _sweep(envir, steps=4)

    source = _frames.FrameSource(swrm, fluid='vort')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        source.warn_if_restreaming(list(source.times))


# --------------------------------------------------------------------------- #
#                    refusals: missing is not a fallback                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('recorded,asked', [
    (None, 'vort'),
    (None, 'quiver'),
    ('vort', 'quiver'),      # neither quantity may be derived from the other
    ('quiver', 'vort'),
])
def test_a_quantity_the_archive_lacks_is_refused(tmp_path, recorded, asked):
    # The only fallback available is re-reading the whole dataset, which is the
    # cost being avoided.
    rec, envir, swrm = _recorded_run(tmp_path, fluid=recorded)

    with pytest.raises(ValueError, match="no '{}' data".format(asked)):
        _frames.FrameSource(swrm, fluid=asked)


def test_a_resident_field_needs_nothing_from_the_archive(tmp_path):
    # What is available is decided by what is in memory, not by what was
    # recorded: a resident field gives the curl directly, exactly and free.
    rec, envir, swrm = _recorded_run(tmp_path, INUM=None, steps=4, fluid=None)

    source = _frames.FrameSource(swrm, fluid='vort')
    t = source.times[1]
    assert np.allclose(source.vorticity(t), envir.get_vorticity(time=t))


def test_a_3d_run_is_not_refused_a_backdrop_3d_never_draws(tmp_path):
    # fluid= is forced to None in 3D at record time, so asking for one at render
    # time has to be as silently meaningless there.
    x = y = z = np.linspace(0, 10, 6)
    Y = np.meshgrid(x, y, z, indexing='ij')[1]
    envir = planktos.Environment(Lx=10, Ly=10, Lz=10,
                                 flow=[0.1*Y, np.zeros_like(Y), np.zeros_like(Y)])
    envir.flow.flow_points = (x, y, z)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run')):
        for _ in range(3):
            swrm.move(0.1, silent=True)

    source = _frames.FrameSource(swrm, fluid='vort')
    assert source.vort_clip is None


# --------------------------------------------------------------------------- #
#              a backdrop off disk is the curl of the field in use            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('with_vorticity', [True, False])
def test_a_stored_backdrop_equals_the_live_curl(tmp_path, with_vorticity):
    # Both windowed regimes, sourced and written. The blend uses the two weights
    # LinearSpline uses and the curl is linear, so what a frame draws is the
    # curl of the field the agents moved through.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=with_vorticity)
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run')):
        _sweep(envir, swrm=swrm)

    # A resident copy computes the curl from the velocity directly, with no
    # per-dump file involved anywhere.
    ref = _envir(src, INUM=True)
    source = _frames.FrameSource(swrm, fluid='vort')

    for t in np.linspace(envir.flow.flow_times[0], envir.flow.flow_times[-1], 15):
        assert np.allclose(source.vorticity(t), ref.get_vorticity(time=t),
                           rtol=1e-8, atol=1e-10)


def test_stored_arrows_equal_the_strided_slice_of_the_velocity(tmp_path):
    # Subsampling is linear too, so blending strided slices gives the strided
    # slice of the blend.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run'), fluid='quiver',
                      quiver_shape=(3, 3)):
        _sweep(envir, swrm=swrm)

    ref = _envir(src, INUM=True)
    source = _frames.FrameSource(swrm, fluid='quiver')
    M, N = source.strides
    assert (M, N) != (1, 1)              # the fixture is coarse but not that coarse

    for t in np.linspace(envir.flow.flow_times[0], envir.flow.flow_times[-1], 9):
        u, v = source.quiver(t)
        field = ref.flow(t)
        assert np.allclose(u, field[0][::M, ::N])
        assert np.allclose(v, field[1][::M, ::N])


def test_stored_arrows_are_read_once_on_a_monotone_sweep(tmp_path):
    # A movie draws many frames per dump interval and consecutive frames share a
    # bracketing pair, so two slots are enough for one read per dump.
    rec, envir, swrm = _recorded_run(tmp_path, fluid='quiver')

    source = _frames.FrameSource(swrm, fluid='quiver')
    reads = []
    real = source.run.quiver
    source.run.quiver = lambda t_idx: (reads.append(int(t_idx)), real(t_idx))[1]

    for t in np.linspace(envir.flow.flow_times[0], envir.flow.flow_times[-1], 40):
        source.quiver(t)
        assert len(source._quiver_cache) <= 2

    assert sorted(set(reads)) == list(range(NDUMPS))
    assert len(reads) == len(set(reads))


# --------------------------------------------------------------------------- #
#                     the colour and arrow scales are global                  #
# --------------------------------------------------------------------------- #

def test_the_colour_limit_covers_the_whole_run(tmp_path):
    rec, envir, swrm = _recorded_run(tmp_path)

    with planktos.load_run(rec.path) as run:
        expected = float(run.dump_stats()['vort_absmax'])
    source = _frames.FrameSource(swrm, fluid='vort')
    assert np.isclose(source.vort_clip, expected)

    # Passed to _vorticity_norm it fixes the limits outright, so a quieter frame
    # drawn later cannot move them.
    norm = _vorticity_norm(np.zeros(1), source.vort_clip)
    assert np.isclose(norm.vmax, expected) and np.isclose(norm.vmin, -expected)
    quiet = _vorticity_norm(np.zeros((3, 3)), source.vort_clip, norm)
    assert np.isclose(quiet.vmax, expected)


def test_two_renders_of_different_stretches_share_a_colour_scale(tmp_path):
    # The defect this half of the work exists to fix, as a user sees it: drawing
    # the first few frames and drawing the whole run must put the same vorticity
    # at the same colour.
    rec, envir, swrm = _recorded_run(tmp_path, INUM=None, steps=8)

    def limits(source, frames):
        norm = _vorticity_norm(np.zeros(1), source.vort_clip)
        for n in frames:
            norm = _vorticity_norm(source.vorticity(source.time(n)),
                                   source.vort_clip, norm)
        return norm.vmin, norm.vmax

    def archived():
        return _frames.FrameSource(swrm, fluid='vort')

    n_states = archived().n_states
    assert limits(archived(), range(2)) == limits(archived(), range(n_states))

    # Unrecorded there is no global scale to be had, and the limits do move.
    envir._archive_path = None
    live = _frames.FrameSource(swrm, fluid='vort')
    assert live.vort_clip is None
    assert limits(live, range(2)) != limits(live, range(live.n_states))


def test_a_dump_the_run_never_reached_does_not_poison_the_scale(tmp_path):
    # A sliding window simply never loads the later dumps. The recorded extremum
    # runs over the dumps that did arrive, so a short run still has a scale --
    # where a per-dump array left holes that had to be reduced around.
    rec, envir, swrm = _recorded_run(tmp_path, steps=2)

    with planktos.load_run(rec.path) as run:
        stats = run.dump_stats()
        # The means still carry the hole, since they are read per dump.
        assert np.isnan(stats['means']).any()
        expected = float(stats['vort_absmax'])

    source = _frames.FrameSource(swrm, fluid='vort')
    assert np.isfinite(source.vort_clip)
    assert np.isclose(source.vort_clip, expected)


def test_an_explicit_clip_is_used_as_given(tmp_path):
    rec, envir, swrm = _recorded_run(tmp_path)

    source = _frames.FrameSource(swrm, fluid='vort', clip=0.25)
    assert source.vort_clip == 0.25


def test_the_arrow_scale_comes_from_the_recorded_extrema(tmp_path):
    rec, envir, swrm = _recorded_run(tmp_path, steps=4, fluid='quiver')

    with planktos.load_run(rec.path) as run:
        recorded = np.asarray(run.dump_stats()['vmax'])
    source = _frames.FrameSource(swrm, fluid='quiver')
    assert np.isclose(source.quiver_scale, np.linalg.norm(recorded))


def test_the_arrow_scale_holds_still_after_the_recording_stops(tmp_path):
    # fmax covers all data seen so far and goes on growing with any later fluid
    # access, so a scale taken from it would move between two renders of one
    # recorded run. The recorded extrema are fixed once the dump is read.
    rec, envir, swrm = _recorded_run(tmp_path, steps=4, fluid='quiver')

    before = _frames.FrameSource(swrm, fluid='quiver').quiver_scale
    envir.flow(float(envir.flow.flow_times[-1]))   # dumps the recording missed
    after = _frames.FrameSource(swrm, fluid='quiver').quiver_scale
    assert before == after
    # ...and fmax did move, so the two numbers are genuinely different sources.
    assert not np.isclose(np.linalg.norm(np.array(envir.flow.fmax)), before)


def test_without_a_recording_the_scales_are_the_ones_that_drift(tmp_path):
    # No recording, no fixed scale to be had: behaviour is what it always was.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = _sweep(envir, steps=4)

    source = _frames.FrameSource(swrm, fluid='quiver')
    assert source.vort_clip is None
    assert np.isclose(source.quiver_scale,
                      np.linalg.norm(np.array(envir.flow.fmax)))


def test_frames_past_the_recorded_dumps_are_refused(tmp_path):
    # A run that carried on after the recording stopped has frames whose
    # per-dump files were never written.
    src = _copy_fixture(tmp_path, 'src')          # no Omega: Planktos writes it
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run'), fluid='vort'):
        _sweep(envir, steps=3, swrm=swrm)
    _sweep(envir, steps=11, swrm=swrm)            # on past the recording

    with pytest.raises(ValueError, match='only part of the fluid series'):
        _frames.FrameSource(swrm, fluid='vort')


def test_the_re_read_warning_counts_dumps_that_exist(tmp_path):
    # The count is the number of dumps the interpolation actually reads, which
    # over the whole series is every dump and no more.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = _sweep(envir)

    source = _frames.FrameSource(swrm, fluid='vort')
    with pytest.warns(UserWarning, match='about {} of'.format(NDUMPS)):
        source.warn_if_restreaming(list(envir.flow.flow_times))


# --------------------------------------------------------------------------- #
#                   the quiver grid is fixed at record time                   #
# --------------------------------------------------------------------------- #

def test_stored_arrows_use_the_grid_they_were_recorded_on(tmp_path):
    # plot_all derives arrow density from the figure size and axis extent,
    # neither of which exists while a simulation is running.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run'), fluid='quiver',
                      quiver_shape=(2, 2)):
        _sweep(envir, swrm=swrm)

    source = _frames.FrameSource(swrm, fluid='quiver')
    stored = source.strides
    with pytest.warns(UserWarning, match='quiver grid is fixed'):
        assert source.resolve_strides((1, 1)) == stored


def test_a_figure_that_wanted_a_similar_grid_is_not_warned_at(tmp_path):
    # Rounding an arrow count against a grid lands a stride off by one
    # routinely, and a warning that fires on every plot is one nobody reads.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run'), fluid='quiver',
                      quiver_shape=(3, 3)):
        _sweep(envir, swrm=swrm)

    source = _frames.FrameSource(swrm, fluid='quiver')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert source.resolve_strides(source.strides) == source.strides


def test_with_the_field_resident_the_figure_chooses_the_grid(tmp_path):
    # Nothing is read from disk, so the grid stays a presentation choice like
    # every other figure parameter.
    rec, envir, swrm = _recorded_run(tmp_path, INUM=None, steps=4,
                                     fluid='quiver')

    source = _frames.FrameSource(swrm, fluid='quiver')
    assert source.strides is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert source.resolve_strides((2, 3)) == (2, 3)
    u, v = source.quiver(source.times[1])
    assert u.shape == envir.flow(source.times[1])[0][::2, ::3].shape


# --------------------------------------------------------------------------- #
#                       every frame, including the last                       #
# --------------------------------------------------------------------------- #

def test_the_last_state_is_the_present(tmp_path):
    # animate() draws every frame through one code path, which is only correct
    # if the last state really is the present.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = _sweep(envir, steps=4)

    source = _frames.FrameSource(swrm)
    last = source.n_states - 1
    assert np.allclose(source.positions(last), swrm.positions)
    assert np.allclose(source.velocities(last), swrm.velocities)
    assert np.isclose(source.time(last), envir.time)
    for a, b in zip(swrm._calc_basic_stats(DIM3=False, t_indx=last),
                    swrm._calc_basic_stats(DIM3=False)):
        assert np.allclose(a, b)


@pytest.mark.slow
def test_the_final_frame_draws_the_present_state(tmp_path):
    # Pins the frame the old code drew in a branch of its own: the artists it
    # leaves behind must show the present positions, the present time, and the
    # per-agent colours, which that branch applied only when a property history
    # had been kept.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = _sweep(envir, steps=4, n=6)
    # add_prop drops the shared color, which is correct -- the color is no
    # longer shared -- and is what the 2D placeholder scatter used to assume.
    swrm.add_prop('color', ['C{}'.format(i % 4) for i in range(swrm.N)])
    assert 'color' not in swrm.shared_props

    anim = _run_animation(swrm, fps=2, playback_rate=1)
    scat, time_text = anim['scat'], anim['time_text']

    last = len(swrm.pos_history)
    anim['animate'](last)
    assert np.allclose(scat.get_offsets(), swrm.positions)
    assert time_text.get_text() == 'time = {:.2f}'.format(envir.time)
    assert len(scat.get_facecolors()) == swrm.N


def _run_animation(swrm, **kwargs):
    '''Build plot_all's figure and hand back the animation function and artists.

    plt.show() is a no-op on Agg and FuncAnimation never runs its timer, so a
    frame is only drawn by asking for it.
    '''

    captured = {}
    real = matplotlib.animation.FuncAnimation

    def capture(fig, func, *args, **kw):
        captured['animate'] = func
        return real(fig, func, *args, **kw)

    matplotlib.animation.FuncAnimation = capture
    try:
        swrm.plot_all(**kwargs)
    finally:
        matplotlib.animation.FuncAnimation = real
    ax = plt.gcf().axes[0]
    captured['scat'] = [c for c in ax.collections
                        if isinstance(c, matplotlib.collections.PathCollection)][-1]
    captured['time_text'] = ax.texts[0]
    return captured


# --------------------------------------------------------------------------- #
#                     plot_all= on record(): the auto-render                  #
# --------------------------------------------------------------------------- #

def test_plot_all_on_record_is_refused_for_more_than_one_swarm(tmp_path):
    # plot_all is a Swarm method and joint multi-swarm plotting does not exist,
    # so this is refused before the run, and refusing leaves nothing recording.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    planktos.Swarm(swarm_size=3, envir=envir, seed=2)

    with pytest.raises(ValueError, match='renders one swarm'):
        envir.record(str(tmp_path / 'run'), plot_all=dict())
    assert envir._recorder is None
    assert not (tmp_path / 'run').exists()


def test_plot_all_on_record_must_be_a_dict_of_kwargs(tmp_path):
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    planktos.Swarm(swarm_size=3, envir=envir, seed=1)

    with pytest.raises(TypeError, match='dict of Swarm.plot_all'):
        envir.record(str(tmp_path / 'run'), plot_all='out.mp4')
    assert envir._recorder is None


def test_the_auto_render_passes_the_arguments_through(tmp_path):
    # Rendering is stubbed out here; what is asserted is that the dict reaches
    # plot_all untouched.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    seen = {}
    swrm.plot_all = lambda **kw: seen.update(kw)

    with envir.record(str(tmp_path / 'run'),
                      plot_all=dict(movie_filename='out.mkv', fluid='vort')):
        for _ in range(4):
            swrm.move(0.5, silent=True)

    assert seen == dict(movie_filename='out.mkv', fluid='vort')


def test_the_auto_render_fires_when_the_run_raises(tmp_path):
    # A crash is unexpected and the movie is diagnostic.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    fired = []
    swrm.plot_all = lambda **kw: fired.append(kw)

    with pytest.raises(RuntimeError, match='boom'):
        with envir.record(str(tmp_path / 'run'), plot_all=dict()):
            swrm.move(0.5, silent=True)
            raise RuntimeError('boom')
    assert len(fired) == 1


def test_the_auto_render_does_not_fire_on_a_keyboard_interrupt(tmp_path):
    # Ctrl-C asks for things to stop now; starting a ten-minute render then,
    # escapable only with a second Ctrl-C, is the opposite of that.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    fired = []
    swrm.plot_all = lambda **kw: fired.append(kw)

    with pytest.raises(KeyboardInterrupt):
        with envir.record(str(tmp_path / 'run'), plot_all=dict()) as rec:
            swrm.move(0.5, silent=True)
            raise KeyboardInterrupt
    assert fired == []
    # It still flushed, which is the half of the contract that does apply.
    assert len(planktos.load_run(rec.path).times) == 2


def test_a_failure_inside_the_auto_render_does_not_mask_the_runs_exception(tmp_path):
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)

    def explode(**kw):
        raise ValueError('no ffmpeg')
    swrm.plot_all = explode

    with pytest.warns(UserWarning, match='automatic plot_all= render failed'):
        with pytest.raises(RuntimeError, match='the run'):
            with envir.record(str(tmp_path / 'run'), plot_all=dict()):
                swrm.move(0.5, silent=True)
                raise RuntimeError('the run')


def test_a_swarm_joining_after_record_leaves_the_movie_unmade(tmp_path):
    # Rendering the first swarm would look complete while leaving the others
    # out. Everything needed is on disk either way.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    fired = []
    swrm.plot_all = lambda **kw: fired.append(kw)

    with pytest.warns(UserWarning, match='no movie was made'):
        with envir.record(str(tmp_path / 'run'), plot_all=dict()):
            planktos.Swarm(swarm_size=3, envir=envir, seed=2)
            envir.move_swarms(0.5, silent=True)
    assert fired == []


# --------------------------------------------------------------------------- #
#                       renders that need actual pixels                       #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize('fluid', ['vort', 'quiver'])
def test_a_single_frame_is_drawn_without_a_load(tmp_path, fluid):
    rec, envir, swrm = _recorded_run(tmp_path, fluid=('vort', 'quiver'))

    out = tmp_path / 'frame.png'
    calls, undo = _count_loads(envir)
    try:
        swrm.plot(t=3.0, filename=str(out), fluid=fluid)
    finally:
        undo()
    assert out.is_file()
    assert calls == []


@pytest.mark.slow
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize('fluid', ['vort', 'quiver'])
def test_a_movie_of_a_recorded_run_costs_no_fluid_loads(tmp_path, fluid):
    # The headline end to end, through the real animation function and a real
    # encoder, so every frame of the movie is actually drawn.
    rec, envir, swrm = _recorded_run(tmp_path, fluid=('vort', 'quiver'))

    out = tmp_path / 'movie.mp4'
    calls, undo = _count_loads(envir)
    try:
        swrm.plot_all(movie_filename=str(out), fluid=fluid, fps=2)
    finally:
        undo()
    assert out.is_file()
    assert calls == []


@pytest.mark.slow
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize('kw', [
    {'dist': 'hist'},
    {'downsamp': 3},
    {'frames': [0, 2, 4]},
])
def test_the_other_ways_of_drawing_a_frame_still_work(tmp_path, kw):
    # Every branch of animate() now reads through the source, so each is walked
    # at least once.
    rec, envir, swrm = _recorded_run(tmp_path, fluid=('vort', 'quiver'))

    out = tmp_path / 'movie.mp4'
    swrm.plot_all(movie_filename=str(out), fluid='vort', fps=2, **kw)
    assert out.is_file()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize('kw', [{'dist': 'density'}, {'dist': 'hist'},
                                {'downsamp': 3}])
def test_a_3d_run_renders(tmp_path, kw):
    # 3D draws nothing about the fluid, so what is exercised here is the agent
    # half of animate() and the statistics box.
    x = y = z = np.linspace(0, 10, 6)
    Y = np.meshgrid(x, y, z, indexing='ij')[1]
    envir = planktos.Environment(Lx=10, Ly=10, Lz=10,
                                 flow=[0.1*Y, np.zeros_like(Y), np.zeros_like(Y)])
    envir.flow.flow_points = (x, y, z)
    swrm = planktos.Swarm(swarm_size=8, envir=envir, seed=3)
    with envir.record(str(tmp_path / 'run')):
        for _ in range(6):
            swrm.move(0.1, silent=True)

    out = tmp_path / 'movie3d.mp4'
    swrm.plot_all(movie_filename=str(out), fps=5, playback_rate=0.5, **kw)
    assert out.is_file()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not on PATH")
def test_the_auto_render_writes_a_movie(tmp_path):
    # End to end through record(plot_all=...), with nothing stubbed.
    src = _copy_fixture(tmp_path, 'src', with_vorticity=True)
    envir = _envir(src, INUM=4)
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=3)
    out = tmp_path / 'auto.mp4'

    with envir.record(str(tmp_path / 'run'),
                      plot_all=dict(movie_filename=str(out), fluid='vort',
                                    fps=2)):
        for _ in range(14):
            swrm.move(0.5, silent=True)
    assert out.is_file()
