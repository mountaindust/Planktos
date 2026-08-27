'''Claim 1 -- an all-in-RAM run behaves exactly as it always did.

    "If I run any of the existing examples that create or load fluid flow and
     run a simulation, all in RAM, nothing should break or change."

The run-archive and plotting work is meant to be invisible to a user who never
records anything. Two things have to hold for that, and they are different
claims:

* **Nothing breaks.** The tutorials and the ordinary plotting calls still run.
  The example scripts are exercised as written, in a subprocess, because that
  is how the last three defects of this kind were found (TODO.md, Phase 0).
* **Nothing changes.** Drawing a picture of a run must not alter the run. That
  is stronger than "no exception": a plot that consumed a random number, slid a
  fluid window, or appended to a history would leave the *next* step different,
  and a user who plots midway through a notebook would silently get different
  physics from one who does not.

The second is what most of this module tests, and it is tested by doing the
thing twice and demanding bit-identical numbers -- not approximate ones. Two
runs of the same seeded arithmetic have no licence to differ at all.
'''

import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import planktos
from planktos import _frames

from _streaming import (REPO, FIXTURES, IB2D_NDUMPS, LoadCounter,
                        assert_same_run, assert_same_state, assert_unchanged,
                        brinkman_envir, copy_ib2d, ib2d_envir, needs_dir, run,
                        run_script, snapshot, vtk3d_envir, walk_frames)


# A goal line for work in progress rather than a regression suite: these run
# whole simulations and are opt-in, via --runstreaming. The members also marked
# slow (the example scripts, the cross-version check, the movie renders) need
# --runslow as well.
pytestmark = pytest.mark.streaming

BOX = FIXTURES / 'mesh_min' / 'box.vertex'


def _brinkman_world(n=12, seed=5, mesh=False):
    '''The tutorial world: analytic flow, in RAM, optionally with a boundary.

    Brinkman flow is **time-invariant**, which is a distinct regime from the
    loaded datasets -- ``flow_times`` is None and there is nothing to
    interpolate. Several of the plotting paths branch on it.
    '''

    envir = brinkman_envir()
    if mesh:
        envir.read_IB2d_mesh_data(str(BOX), method='adjacent')
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed)
    swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01
    return envir, swrm


def _ib2d_world(tmp_path, n=8, seed=5, name='src'):
    '''A time-varying loaded dataset held whole in memory: INUM=None.'''

    envir = ib2d_envir(copy_ib2d(tmp_path, name), INUM=None)
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed)
    return envir, swrm


# --------------------------------------------------------------------------- #
#      nothing changes: recording and plotting must not touch the physics     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('mesh', [False, True], ids=['open', 'with_ibmesh'])
def test_recording_leaves_an_in_ram_run_bit_identical(tmp_path, mesh):
    # The claim is not "close enough". Recording writes state out; it must not
    # read anything back into the simulation, so the two runs are the same
    # arithmetic on the same seed and must agree to the last bit.
    _, plain = _brinkman_world(mesh=mesh)
    run(plain, 20, dt=0.1)

    envir, taped = _brinkman_world(mesh=mesh)
    with envir.record(str(tmp_path / 'run')):
        run(taped, 20, dt=0.1)

    assert_same_run(plain, taped)


def test_recording_leaves_a_3d_run_bit_identical(tmp_path):
    # 3D records only the statistics sidecar, but it still hooks every step.
    envir = vtk3d_envir(INUM=None)
    plain = planktos.Swarm(swarm_size=8, envir=envir, seed=11)
    run(plain, 10, dt=0.3)

    envir2 = vtk3d_envir(INUM=None)
    taped = planktos.Swarm(swarm_size=8, envir=envir2, seed=11)
    with envir2.record(str(tmp_path / 'run3d')):
        run(taped, 10, dt=0.3)

    assert_same_run(plain, taped)


DRAW_CALLS = [
    pytest.param(lambda s: s.plot(), id='plot_now'),
    pytest.param(lambda s: s.plot(t=1.5), id='plot_at_t'),
    pytest.param(lambda s: s.plot(fluid='vort'), id='plot_vort'),
    pytest.param(lambda s: s.plot(fluid='quiver'), id='plot_quiver'),
    pytest.param(lambda s: s.plot_all(), id='plot_all'),
    pytest.param(lambda s: s.plot_all(fluid='vort'), id='plot_all_vort'),
    pytest.param(lambda s: s.plot_all(fluid='quiver'), id='plot_all_quiver'),
    pytest.param(lambda s: s.plot_all(frames=[0, 2, 4]), id='plot_all_frames'),
    pytest.param(lambda s: s.plot_all(downsamp=3), id='plot_all_downsamp'),
    # plot_all() renders nothing on Agg -- FuncAnimation only calls its
    # function on a draw event -- so the frames are walked explicitly too.
    pytest.param(lambda s: walk_frames(s), id='frames_plain'),
    pytest.param(lambda s: walk_frames(s, fluid='vort'), id='frames_vort'),
    pytest.param(lambda s: walk_frames(s, fluid='quiver'), id='frames_quiver'),
]


@pytest.mark.parametrize('call', DRAW_CALLS)
def test_drawing_a_picture_does_not_disturb_the_simulation(tmp_path, call):
    _, swrm = _ib2d_world(tmp_path)
    run(swrm, 10, dt=0.5)
    before = snapshot(swrm)
    call(swrm)
    assert_unchanged(swrm, before)


@pytest.mark.parametrize('call', [c for c in DRAW_CALLS
                                  if c.id != 'plot_all_vort'])
def test_drawing_a_time_invariant_flow_does_not_disturb_it_either(call):
    # plot_all_vort is left out here and pinned on its own below: it raises on
    # a time-invariant flow, so it never gets as far as disturbing anything.
    _, swrm = _brinkman_world()
    run(swrm, 10, dt=0.1)
    before = snapshot(swrm)
    call(swrm)
    assert_unchanged(swrm, before)


@pytest.mark.xfail(strict=True, reason=(
    "plot_all builds its vorticity placeholder from flow.fshape[1:], which "
    "drops the leading TIME axis. A time-invariant flow's fshape has no time "
    "axis, so this drops the x axis instead and hands pcolormesh a 1-D array. "
    "Every analytic flow (brinkman/channel/canopy) is time-invariant."))
def test_plot_all_draws_vorticity_over_a_time_invariant_flow():
    _, swrm = _brinkman_world()
    run(swrm, 6, dt=0.1)
    swrm.plot_all(fluid='vort')


def test_a_run_continues_identically_after_being_plotted(tmp_path):
    # The notebook workflow: run, look at it, keep going. If plotting perturbs
    # anything at all -- the RNG most of all -- the second half diverges.
    _, straight = _ib2d_world(tmp_path, name='a')
    run(straight, 14, dt=0.5)

    _, interrupted = _ib2d_world(tmp_path, name='b')
    run(interrupted, 7, dt=0.5)
    interrupted.plot()
    interrupted.plot_all(fluid='vort')
    walk_frames(interrupted, fluid='vort')
    interrupted.plot(t=1.5, fluid='quiver')
    run(interrupted, 7, dt=0.5)

    assert_same_run(straight, interrupted)


def test_a_recorded_run_continues_identically_after_a_mid_run_plot(tmp_path):
    # Same claim with a recorder attached, since a mid-run plot opens the
    # archive that is still being written to.
    _, straight = _ib2d_world(tmp_path, name='a')
    run(straight, 14, dt=0.5)

    envir, taped = _ib2d_world(tmp_path, name='b')
    with envir.record(str(tmp_path / 'run')):
        run(taped, 7, dt=0.5)
        envir.flush_recording()
        taped.plot_all(fluid='vort')
        walk_frames(taped, fluid='vort')
        run(taped, 7, dt=0.5)

    assert_same_run(straight, taped)


# --------------------------------------------------------------------------- #
#              nothing changes: an in-RAM render reads no files               #
# --------------------------------------------------------------------------- #

def test_an_in_ram_render_reads_no_fluid_from_disk(tmp_path):
    # INUM=None holds the whole dataset, so every frame is already in memory.
    # This is the "even faster" half of claim 3, stated for the unrecorded case.
    src = copy_ib2d(tmp_path)
    envir = ib2d_envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=6, envir=envir, seed=3)
    run(swrm, 14, dt=0.5)

    with LoadCounter(envir) as loads:
        walk_frames(swrm, fluid='vort')
        walk_frames(swrm, fluid='quiver')
        swrm.plot_all(fluid='vort')
        swrm.plot(fluid='vort')
    assert len(loads) == 0, 'a resident field was re-read from disk: {}'.format(
        loads.calls)


def test_an_in_ram_render_takes_the_backdrop_from_the_field(tmp_path):
    # Not merely "no I/O": the values drawn must be the curl of the field in
    # memory, exactly, with no archive in the picture.
    src = copy_ib2d(tmp_path)
    envir = ib2d_envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
    run(swrm, 14, dt=0.5)

    source = _frames.FrameSource(swrm, fluid='vort')
    assert source.run is None, 'an unrecorded run found an archive'
    for t in (0.0, 1.3, 2.5):
        np.testing.assert_array_equal(source.vorticity(t),
                                      envir.flow.get_vorticity(time=t))


def test_a_resident_field_never_warns_about_re_reading(tmp_path):
    src = copy_ib2d(tmp_path)
    envir = ib2d_envir(src, INUM=None)
    swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
    run(swrm, 14, dt=0.5)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        swrm.plot_all(fluid='vort')
    offenders = [str(w.message) for w in log if 're-read' in str(w.message)]
    assert not offenders, 'a resident field warned about re-reading: {}'.format(
        offenders)


# --------------------------------------------------------------------------- #
#                    nothing breaks: the ordinary shapes                      #
# --------------------------------------------------------------------------- #

def test_a_flow_free_environment_still_plots():
    # Environment() with no fluid at all -- the very first thing the 2D
    # tutorial builds, and the state much of the test suite runs in.
    envir = planktos.Environment()
    swrm = planktos.Swarm(swarm_size=10, envir=envir, seed=1)
    run(swrm, 5, dt=0.1)
    swrm.plot()
    swrm.plot_all()
    swrm.plot(fluid='vort')      # nothing to draw, must not raise
    swrm.plot_all(fluid='quiver')


def test_a_time_invariant_flow_draws_its_own_backdrop():
    # Brinkman has flow_times None. The backdrop must come from the field with
    # no reference to a time base that does not exist.
    envir, swrm = _brinkman_world()
    run(swrm, 6, dt=0.1)
    source = _frames.FrameSource(swrm, fluid='vort')
    np.testing.assert_array_equal(source.vorticity(envir.time),
                                  envir.flow.get_vorticity())
    source_q = _frames.FrameSource(swrm, fluid='quiver')
    source_q.resolve_strides((3, 3))
    u, v = source_q.quiver(envir.time)
    np.testing.assert_array_equal(u, envir.flow[0][::3, ::3])
    np.testing.assert_array_equal(v, envir.flow[1][::3, ::3])


def test_the_states_a_plot_can_draw_are_the_history_then_the_present():
    envir, swrm = _brinkman_world()
    run(swrm, 7, dt=0.1)
    source = _frames.FrameSource(swrm)
    assert source.n_states == len(swrm.pos_history) + 1
    for n in range(len(swrm.pos_history)):
        assert_same_state(source.positions(n), swrm.pos_history[n],
                          'state {}'.format(n))
    assert_same_state(source.positions(source.n_states - 1), swrm.positions,
                      'the last state is the present')
    assert source.time(source.n_states - 1) == envir.time


def test_a_3d_in_ram_run_plots():
    envir = vtk3d_envir(INUM=None)
    swrm = planktos.Swarm(swarm_size=6, envir=envir, seed=2)
    run(swrm, 6, dt=0.3)
    swrm.plot()
    swrm.plot_all()
    # fluid= is ignored in 3D rather than refused
    swrm.plot_all(fluid='vort')


def test_several_swarms_move_together_and_each_one_plots(tmp_path):
    envir = ib2d_envir(copy_ib2d(tmp_path), INUM=None)
    a = planktos.Swarm(swarm_size=8, envir=envir, seed=1)
    b = planktos.Swarm(swarm_size=8, envir=envir, seed=2)
    for _ in range(6):
        envir.move_swarms(0.5, silent=True)
    for s in (a, b):
        s.plot()
        s.plot_all(fluid='vort')
        walk_frames(s, fluid='vort')


def test_a_reset_environment_replots_from_the_start():
    envir, swrm = _brinkman_world()
    run(swrm, 6, dt=0.1)
    envir.reset(rm_swarms=True)
    swrm2 = planktos.Swarm(swarm_size=6, envir=envir, seed=9)
    run(swrm2, 4, dt=0.1)
    swrm2.plot_all()


def test_agents_that_have_all_left_the_domain_still_plot():
    # Zero boundaries and a strong drift: every agent is masked before the end.
    envir = planktos.Environment()
    swrm = planktos.Swarm(swarm_size=6, envir=envir, seed=4)
    swrm.shared_props['mu'] = np.array([40., 40.])
    run(swrm, 6, dt=0.5)
    assert np.ma.getmaskarray(swrm.positions).all(), 'expected an empty domain'
    swrm.plot()
    swrm.plot_all()


def test_property_history_is_used_for_headings_when_it_is_kept():
    envir = brinkman_envir()
    swrm = planktos.Swarm(swarm_size=6, envir=envir, seed=7,
                          store_prop_history=True)
    swrm.add_prop('angle', np.linspace(0, 1, 6))
    run(swrm, 5, dt=0.1)
    source = _frames.FrameSource(swrm)
    assert source.props(0) is not swrm.props or len(swrm.props_history) > 0
    swrm.plot_all()


# --------------------------------------------------------------------------- #
#        nothing changes: the same numbers the released line produces         #
# --------------------------------------------------------------------------- #

# "Nothing should change from when we started dyload" is answerable directly:
# check out master beside this tree and run the same simulations under both.
# The scenarios use only API that both branches have, so the comparison is of
# the physics and not of the fluid interface.

CROSS_VERSION_RUNNER = '''\
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
# An editable install puts a finder on sys.meta_path, which beats sys.path and
# would import the working tree's planktos into the worktree's process.
sys.meta_path = [f for f in sys.meta_path
                 if 'editable' not in getattr(f, '__name__', str(f)).lower()
                 and 'editable' not in getattr(type(f), '__module__', '').lower()]
sys.path.insert(0, here)
for m in [m for m in sys.modules if m == 'planktos' or m.startswith('planktos.')]:
    del sys.modules[m]

import matplotlib; matplotlib.use('Agg')
import numpy as np, planktos
assert os.path.abspath(planktos.__file__).startswith(here), planktos.__file__

FIXTURES, OUT = sys.argv[1], sys.argv[2]
out = {}

# a time-varying loaded 2D fluid, whole dataset resident, cubic in time
envir = planktos.Environment(x_bndry='periodic', y_bndry='periodic')
envir.read_IB2d_fluid_data(os.path.join(FIXTURES, 'ib2d_fluid_min'),
                           dt=0.1, print_dump=10)
swrm = planktos.Swarm(swarm_size=20, envir=envir, seed=5)
for _ in range(14):
    swrm.move(0.5, silent=True)
out['ib2d'] = np.ma.filled(swrm.positions, np.nan)

# analytic flow with an immersed boundary: collisions and sliding
e2 = planktos.Environment(rho=1000, mu=1000)
e2.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=51)
e2.read_IB2d_mesh_data(os.path.join(FIXTURES, 'mesh_min', 'box.vertex'),
                       method='adjacent')
s2 = planktos.Swarm(swarm_size=40, envir=e2, seed=9)
s2.shared_props['cov'] = s2.shared_props['cov'] * 0.2
for _ in range(30):
    s2.move(0.1, silent=True)
out['ibmesh'] = np.ma.filled(s2.positions, np.nan)

# 3D, loaded from vtk
e3 = planktos.Environment(x_bndry='periodic', y_bndry='periodic',
                          z_bndry='periodic')
try:
    e3.read_IBAMR3d_vtk_data(os.path.join(FIXTURES, 'vtk3d_min'),
                             title='IBAMR_db_')
except TypeError:
    e3.read_IBAMR3d_vtk_data(os.path.join(FIXTURES, 'vtk3d_min'))
s3 = planktos.Swarm(swarm_size=20, envir=e3, seed=7)
for _ in range(12):
    s3.move(0.3, silent=True)
out['vtk3d'] = np.ma.filled(s3.positions, np.nan)

np.savez(OUT, **out)
print('planktos', planktos.__version__)
'''


@pytest.mark.slow
@pytest.mark.parametrize('rev', ['master'])
def test_an_in_ram_run_gives_the_numbers_the_released_line_gives(tmp_path, rev):
    if shutil.which('git') is None:
        pytest.skip('git not on PATH')
    if subprocess.run(['git', 'rev-parse', '--verify', rev], cwd=str(REPO),
                      capture_output=True).returncode != 0:
        pytest.skip('no {} to compare against in this clone'.format(rev))

    worktree = tmp_path / 'other'
    made = subprocess.run(['git', 'worktree', 'add', '--detach',
                           str(worktree), rev], cwd=str(REPO),
                          capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip('could not create a worktree: ' + made.stderr.strip())
    try:
        results = {}
        for label, root in (('here', REPO), (rev, worktree)):
            script = Path(root) / '_cross_version_check.py'
            script.write_text(CROSS_VERSION_RUNNER)
            out = tmp_path / '{}.npz'.format(label)
            try:
                proc = subprocess.run(
                    [sys.executable, str(script), str(FIXTURES), str(out)],
                    cwd=str(root), capture_output=True, text=True,
                    env=dict(os.environ, MPLBACKEND='Agg'))
            finally:
                script.unlink(missing_ok=True)
            assert proc.returncode == 0, '{}:\n{}\n{}'.format(
                label, proc.stdout[-2000:], proc.stderr[-3000:])
            results[label] = dict(np.load(out))

        for scenario in results['here']:
            a, b = results['here'][scenario], results[rev][scenario]
            # NaN marks an agent that left; compare it as a value.
            np.testing.assert_array_equal(
                np.nan_to_num(a, nan=-1e30), np.nan_to_num(b, nan=-1e30),
                err_msg='{} differs from {}'.format(scenario, rev))
    finally:
        subprocess.run(['git', 'worktree', 'remove', '--force', str(worktree)],
                       cwd=str(REPO), capture_output=True)
        subprocess.run(['git', 'worktree', 'prune'], cwd=str(REPO),
                       capture_output=True)


# --------------------------------------------------------------------------- #
#           nothing breaks: the tutorials, run exactly as written             #
# --------------------------------------------------------------------------- #

# Only the examples that need no downloaded data. ex_IBAMR_ibmesh.py is
# excluded because it calls tile_domain, which raises NotImplementedError by
# design on this branch (run_persistence.md 9.3) -- its own header says so.
SELF_CONTAINED_EXAMPLES = [
    'basic_ex_2d.py',
    'basic_ex_3d.py',
    'ex_agent_behavior.py',
    'ex_ind_var.py',
    'ex_ode_gen.py',
    'ex_poisson_search.py',
]

# These need examples/ib2d_data, which is gitignored.
DATA_EXAMPLES = [
    'ex_vicsek_model_2d.py',
    'ex_vicsek_model_3d.py',
    'ex_produce_ftle_2d.py',
    'ex_ib2d_ibmesh.py',
    'ex_ib2d_sticky.py',
]


@pytest.mark.slow
@pytest.mark.parametrize('script', SELF_CONTAINED_EXAMPLES)
def test_a_self_contained_example_runs_to_completion(script, tmp_path):
    # Run as written, in a subprocess, from a copy of examples/ so that any
    # movie or figure lands in the temporary directory. Agg makes plt.show a
    # no-op, which is what a headless run of the tutorial does anyway.
    work = tmp_path / 'examples'
    shutil.copytree(REPO / 'examples', work,
                    ignore=shutil.ignore_patterns('*.mp4', 'results'))
    proc = run_script(work / script, cwd=work)
    assert proc.returncode == 0, \
        'examples/{} failed:\n{}\n{}'.format(script, proc.stdout[-3000:],
                                             proc.stderr[-4000:])


@pytest.mark.slow
@pytest.mark.parametrize('script', DATA_EXAMPLES)
def test_an_example_needing_downloaded_data_runs_to_completion(script, tmp_path):
    needs_dir(REPO / 'examples' / 'ib2d_data',
              'download the IB2d channel-flow dataset to run this example')
    work = tmp_path / 'examples'
    shutil.copytree(REPO / 'examples', work,
                    ignore=shutil.ignore_patterns('*.mp4', 'results'))
    proc = run_script(work / script, cwd=work)
    assert proc.returncode == 0, \
        'examples/{} failed:\n{}\n{}'.format(script, proc.stdout[-3000:],
                                             proc.stderr[-4000:])
