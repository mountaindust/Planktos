'''Tests for Environment.record and the capture hooks
(run_persistence.md section 2.1-2.3, build step A3a).

Where test_run_archive.py drives the writer with synthetic arrays, these drive
real runs and check that what lands on disk is what the run actually did. The
archive is read back with raw np.load, since the reader is A4.

The headline is test_recording_costs_no_extra_fluid_loads: recording a
dynamically-loaded run must cost *identically* many loader calls as the same run
without it. That single assertion is the property the whole design exists for --
an archive that re-streamed the fluid to write itself would be worse than no
archive at all.

At this step every environmental time step is captured. The capture schedule
(capture_interval) is A3b.
'''

import json
import warnings
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

import planktos
from planktos import archive, fluid

FIXTURES = Path(__file__).parent / 'fixtures'


# --------------------------------------------------------------------------- #
#                                  helpers                                     #
# --------------------------------------------------------------------------- #

def _envir(L=10.0):
    '''A still 2D environment; no flow interpolation, no RNG surprises.'''
    return planktos.Environment(Lx=L, Ly=L,
                                flow=[np.zeros((3, 3)), np.zeros((3, 3))])


def _swarm(envir, n=4, mu=(1.0, 0.0), seed=1):
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed,
                          init=np.full((n, 2), 2.0))
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array(mu, float)
    return swrm


def _chunks(agent_dir, pattern):
    '''Concatenate a swarm's chunk files back into one array, in index order.'''
    files = sorted(agent_dir.glob(pattern), key=archive._chunk_index_of)
    return np.concatenate([np.load(f, allow_pickle=False) for f in files])


def _stack(history):
    return np.stack([np.ma.getdata(entry) for entry in history])


# --------------------------------------------------------------------------- #
#              the headline: recording must not re-stream the fluid            #
# --------------------------------------------------------------------------- #

class _CountingSource(fluid.FluidData):
    '''A windowed FluidData whose "disk" is an in-memory array, counting loads.

    Mirrors tests/test_dynamic_loading.py's _InMemorySource. Kept here rather
    than imported so this module states its own premise.
    '''

    def __init__(self, full_flow, flow_points, flow_times, INUM):
        self._full = [np.asarray(f) for f in full_flow]
        self.load_calls = []
        self.d_start = 0
        self.d_finish = len(flow_times) - 1
        self.loaded_dump_bnds = (0, INUM)
        self.loaded_idx_bnds = (0, INUM)
        window = self.load_dumpfiles(0, INUM)
        super().__init__(window, flow_points, flow_times, INUM)

    def load_dumpfiles(self, d_start, d_finish):
        self.load_calls.append((d_start, d_finish))
        return [f[d_start:d_finish + 1].copy() for f in self._full]


def _streaming_envir(INUM=4, nt=21):
    t = np.linspace(0.0, 2.0, nt)
    x = np.linspace(0.0, 10.0, 6)
    y = np.linspace(0.0, 8.0, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.stack([np.sin(tt) * X for tt in t])
    v = np.stack([tt ** 2 * Y for tt in t])
    envir = planktos.Environment(Lx=10.0, Ly=8.0)
    envir.flow = _CountingSource([u, v], (x, y), t, INUM)
    return envir


def _drive(envir, steps=15, dt=0.1):
    swrm = planktos.Swarm(swarm_size=6, envir=envir, seed=3,
                          init=np.full((6, 2), 4.0))
    swrm.shared_props['cov'] = np.zeros((2, 2))
    for _ in range(steps):
        swrm.move(dt, silent=True)
    return swrm


def test_recording_costs_no_extra_fluid_loads(tmp_path):
    # The property the whole design exists for. An archive that re-streamed the
    # fluid in order to write itself would be worse than no archive.
    plain = _streaming_envir()
    _drive(plain)
    without = list(plain.flow.load_calls)

    recorded = _streaming_envir()
    with recorded.record(tmp_path / 'run'):
        _drive(recorded)
    with_recording = list(recorded.flow.load_calls)

    assert with_recording == without, (
        'recording changed the fluid loading pattern: {} loads while recording '
        'against {} without'.format(len(with_recording), len(without)))
    assert len(without) > 1, 'the window never slid; the test proves nothing'


def test_record_refuses_once_the_fluid_window_has_moved(tmp_path):
    # Dumps the window has passed are gone and are never re-reported, so a
    # recording started here would have holes in its fluid series -- refused now
    # rather than at render time, i.e. before the run instead of after it.
    envir = _streaming_envir()
    _drive(envir)
    assert envir.flow.loaded_idx_bnds[0] != 0, 'the window did not slide'
    with pytest.raises(RuntimeError, match='moved past the first dump'):
        envir.record(tmp_path / 'run')


def test_record_is_allowed_when_everything_is_in_memory(tmp_path):
    # With INUM=None there is no window to have moved, so a mid-run start is
    # permitted -- nothing about the fluid series can be missing.
    envir = _envir()
    swrm = _swarm(envir)
    for _ in range(3):
        swrm.move(0.1, silent=True)
    with envir.record(tmp_path / 'run') as rec:
        swrm.move(0.1, silent=True)
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 2


# --------------------------------------------------------------------------- #
#                    what lands on disk is what the run did                    #
# --------------------------------------------------------------------------- #

def test_the_archive_reproduces_the_histories_exactly(tmp_path):
    # Capture j is exactly full_pos_history[j] at (time_history + [time])[j] --
    # the same index convention the renderer already uses, so nothing has to be
    # translated at render time.
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', chunk_size=3) as rec:
        for _ in range(7):
            swrm.move(0.05, silent=True)

    agents = rec.path / 'agents'
    assert np.allclose(_chunks(agents, 'times_*.npy'),
                       envir.time_history + [envir.time])
    assert np.allclose(_chunks(agents, 'swarm00_pos_*.npy'),
                       _stack(swrm.full_pos_history))
    assert np.allclose(_chunks(agents, 'swarm00_vel_*.npy'),
                       _stack(swrm.full_vel_history))


def test_a_plain_move_captures_exactly_one_state_per_step(tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        for _ in range(5):
            swrm.move(0.1, silent=True)
    # capture 0 covers t0, then one per step
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 6


def test_move_swarms_captures_once_per_step_not_once_per_swarm(tmp_path):
    # A hook on Swarm.move would fire once per swarm, at a time that has not
    # advanced, with later swarms not yet moved. The trigger is the environment
    # time step.
    envir = _envir()
    a, b = _swarm(envir, seed=1), _swarm(envir, seed=2)
    with envir.record(tmp_path / 'run') as rec:
        for _ in range(4):
            envir.move_swarms(0.1, silent=True)
    agents = rec.path / 'agents'
    assert len(_chunks(agents, 'times_*.npy')) == 5
    for prefix in ('swarm00', 'swarm01'):
        assert len(_chunks(agents, prefix + '_pos_*.npy')) == 5


def test_a_masked_agent_is_recorded_as_masked(tmp_path):
    envir = _envir(L=1.0)
    swrm = _swarm(envir, n=3, mu=(2.0, 0.0))
    swrm.positions[0] = [0.95, 0.5]
    with envir.record(tmp_path / 'run') as rec:
        for _ in range(2):
            swrm.move(0.1, silent=True)
    mask = _chunks(rec.path / 'agents', 'swarm00_mask_*.npy')
    assert not mask[0].any(), 'nobody had left at t0'
    assert mask[-1, 0], 'the agent that left the domain is not marked'


def test_the_fingerprint_matches_the_environment_that_wrote_it(tmp_path):
    envir = _envir()
    _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        pass
    stored = dict(np.load(rec.path / 'grid.npz', allow_pickle=False))
    assert archive.compare_fingerprints(stored, archive.fingerprint_of(envir)) == []
    # and it would refuse a differently-gridded environment, naming the field
    other = planktos.Environment(Lx=99.0, Ly=10.0,
                                 flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    problems = archive.compare_fingerprints(stored, archive.fingerprint_of(other))
    assert any('L' in p for p in problems)


def test_a_flow_free_environment_records_without_a_fluid(tmp_path):
    envir = planktos.Environment(Lx=10, Ly=10)
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        swrm.move(0.1, silent=True)
    grid = dict(np.load(rec.path / 'grid.npz', allow_pickle=False))
    assert set(grid) == {'dimension', 'L', 'periodic_dim'}
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 2


def test_the_provenance_of_the_run_is_recorded(tmp_path):
    envir = planktos.Environment()
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        pass
    prov = json.loads((rec.path / 'meta.json').read_text())['provenance']
    assert prov['fluid']['loader'] == 'read_IB2d_fluid_data'
    assert prov['planktos_version'] == planktos.__version__
    assert prov['environment']['L'] == [float(v) for v in envir.L]
    assert prov['ibmesh'] is None


def test_the_environment_scalars_a_restart_needs_are_recorded(tmp_path):
    # A rebuilt Environment has to match the original attribute for attribute.
    # char_L and U are the ones that bite: motion.inertial_particles asserts
    # both, so an inertial run without them raises before it moves.
    envir = planktos.Environment(rho=1000, mu=0.001, char_L=0.5, U=0.2,
                                 ibmesh_color='firebrick')
    _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        pass
    env = json.loads((rec.path / 'meta.json').read_text())['provenance']['environment']
    for name in ('rho', 'mu', 'nu', 'char_L', 'U'):
        assert env[name] == pytest.approx(getattr(envir, name)), name
    assert env['ibmesh_color'] == 'firebrick'


def test_a_kinematic_only_environment_still_records_nu(tmp_path):
    # Environment(nu=...) alone leaves rho and mu None, so nu is the only one
    # of the three there is: recording rho and mu alone would lose it silently.
    envir = planktos.Environment(nu=1e-6)
    _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        pass
    env = json.loads((rec.path / 'meta.json').read_text())['provenance']['environment']
    assert env['rho'] is None and env['mu'] is None
    assert env['nu'] == pytest.approx(1e-6)


def test_the_default_ibmesh_color_is_recorded_as_resolved(tmp_path):
    # Recorded as resolved rather than as given, so a rebuilt mesh draws the
    # same in either dimension without the reader repeating the default.
    for kwargs, expected in ((dict(), 'k'), (dict(Lz=10), 'dimgrey')):
        envir = planktos.Environment(**kwargs)
        with envir.record(tmp_path / ('run_' + expected)) as rec:
            pass
        env = json.loads(
            (rec.path / 'meta.json').read_text())['provenance']['environment']
        assert env['ibmesh_color'] == expected


# --------------------------------------------------------------------------- #
#                                  lifecycle                                   #
# --------------------------------------------------------------------------- #

def test_recording_starts_immediately_without_a_with_block(tmp_path):
    # If the work happened in __enter__, a bare envir.record(path) would
    # silently record nothing -- a very expensive thing to discover after a
    # twelve-hour run.
    envir = _envir()
    swrm = _swarm(envir)
    rec = envir.record(tmp_path / 'run')
    for _ in range(3):
        swrm.move(0.1, silent=True)
    envir.stop_recording()
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 4


def test_flush_writes_without_ending_the_recording(tmp_path):
    # A mid-run notebook plot needs this: stopping instead would make the next
    # record() refuse the now-non-empty directory.
    envir = _envir()
    swrm = _swarm(envir)
    rec = envir.record(tmp_path / 'run', chunk_size=100)
    for _ in range(3):
        swrm.move(0.1, silent=True)
    envir.flush_recording()
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 4
    assert envir._recorder is rec, 'flush must not stop the recording'
    for _ in range(2):
        swrm.move(0.1, silent=True)
    envir.stop_recording()
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 6


def test_stop_recording_is_idempotent_and_safe_when_nothing_is_recording(tmp_path):
    envir = _envir()
    _swarm(envir)
    envir.stop_recording()                       # nothing recording: a no-op
    envir.record(tmp_path / 'run')
    envir.stop_recording()
    envir.stop_recording()
    assert envir._recorder is None


def test_a_non_empty_directory_redirects_and_the_handle_says_where(tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    target = tmp_path / 'run'
    with envir.record(target) as first:
        swrm.move(0.1, silent=True)
    with pytest.warns(UserWarning, match='already holds data'):
        second = envir.record(target)
    envir.stop_recording()
    assert second.path != first.path
    assert first.path == target


def test_a_raise_mid_run_leaves_a_readable_archive(tmp_path):
    # plot_all has never had a concept of a finished run, and neither does this:
    # a run that stopped at step 3 is a run of 3 steps.
    class _Exploding(planktos.Swarm):
        def apply_agent_model(self, dt):
            if len(self.pos_history) >= 3:
                raise ValueError('boom')
            return super().apply_agent_model(dt)

    envir = _envir()
    swrm = _Exploding(swarm_size=3, envir=envir, seed=1,
                      init=np.full((3, 2), 2.0))
    swrm.shared_props['cov'] = np.zeros((2, 2))
    rec = envir.record(tmp_path / 'run', chunk_size=2)
    with pytest.raises(ValueError, match='boom'):
        for _ in range(6):
            swrm.move(0.1, silent=True)
    envir.stop_recording()

    times = _chunks(rec.path / 'agents', 'times_*.npy')
    assert len(times) == 4                       # t0 plus the three good steps
    assert json.loads((rec.path / 'meta.json').read_text())['version'] == \
        archive.FORMAT_VERSION


# --------------------------------------------------------------------------- #
#                       a swarm that joins mid-recording                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('spelling', ['constructor', 'add_swarm'])
def test_a_swarm_joining_mid_recording_is_captured_from_that_point(tmp_path, spelling):
    # Both spellings must work. A hook on Environment.add_swarm would miss the
    # constructor -- which is what every example and most of this suite uses --
    # and record nothing for it, silently. The hook is on the two sites where a
    # Swarm appends itself to envir.swarms.
    envir = _envir()
    first = _swarm(envir, seed=1)
    with envir.record(tmp_path / 'run', chunk_size=100) as rec:
        for _ in range(4):
            first.move(0.1, silent=True)
        if spelling == 'constructor':
            late = planktos.Swarm(swarm_size=2, envir=envir, seed=9,
                                  init=np.full((2, 2), 3.0))
        else:
            late = envir.add_swarm(swarm_size=2, seed=9,
                                   init=np.full((2, 2), 3.0))
        late.shared_props['cov'] = np.zeros((2, 2))
        for _ in range(3):
            envir.move_swarms(0.1, silent=True)

    agents = rec.path / 'agents'
    entry = json.loads((agents / 'swarm01.json').read_text())
    assert entry['first_capture'] == 5, 'the late swarm should start after 5 captures'
    assert entry['N'] == 2
    # 8 captures overall; the late swarm contributes only the last 3
    assert len(_chunks(agents, 'times_*.npy')) == 8
    assert len(_chunks(agents, 'swarm00_pos_*.npy')) == 8
    assert len(_chunks(agents, 'swarm01_pos_*.npy')) == 3


# --------------------------------------------------------------------------- #
#                                 the refusals                                 #
# --------------------------------------------------------------------------- #

def test_a_second_record_on_a_recording_environment_raises(tmp_path):
    # There is one recorder per Environment by construction: the time-advance
    # hook finds it through a single reference, so a second could only replace
    # the first or run beside it, neither of which the hook can express.
    envir = _envir()
    _swarm(envir)
    envir.record(tmp_path / 'a')
    with pytest.raises(RuntimeError, match='already recording'):
        envir.record(tmp_path / 'b')
    envir.stop_recording()
    envir.record(tmp_path / 'b')                 # fine once the first has stopped
    envir.stop_recording()


def test_reset_while_recording_raises(tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run'):
        swrm.move(0.1, silent=True)
        with pytest.raises(RuntimeError, match='not allowed while recording'):
            envir.reset()
    envir.reset()                                # fine once stopped


@pytest.mark.parametrize('load', [
    lambda e: e.set_brinkman_flow(alpha=3, h_p=0.5, U=1, dpdx=1, res=11),
    lambda e: e.set_two_layer_channel_flow(a=0.5, h_p=0.5, Cd=1, S=1, res=11),
    lambda e: e.set_canopy_flow(h=0.5, a=1, u_star=1, res=11),
    lambda e: e.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                                     print_dump=10),
    lambda e: e.read_IBAMR3d_vtk_data(str(FIXTURES / 'vtk3d_min')),
])
def test_loading_new_fluid_while_recording_raises(tmp_path, load):
    # Every loader reassigns flow_points, flow_times and L, which would leave
    # the fingerprint already on disk describing a grid the run stopped using.
    envir = planktos.Environment(Lx=10, Ly=10, rho=1000, mu=1000,
                                 flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    _swarm(envir)
    with envir.record(tmp_path / 'run'):
        with pytest.raises(RuntimeError, match='not allowed while recording'):
            load(envir)


def test_a_hand_rolled_time_advance_warns_while_recording(tmp_path):
    # update_time=False exists for move_swarms to call. Reached any other way it
    # means the caller intends to advance the clock by hand, and a hand-rolled
    # advance fires no hook -- so the step happens but the archive never sees it.
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        with pytest.warns(UserWarning, match='will not be captured'):
            swrm.move(0.1, update_time=False, silent=True)
        envir.time += 0.1
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 1


def test_move_swarms_does_not_trip_the_hand_rolled_warning(tmp_path):
    envir = _envir()
    _swarm(envir, seed=1)
    _swarm(envir, seed=2)
    with envir.record(tmp_path / 'run'):
        with warnings.catch_warnings():
            warnings.simplefilter('error')       # any warning fails the test
            envir.move_swarms(0.1, silent=True)


def test_recording_without_velocities_warns_and_stores_only_positions(tmp_path):
    # Allowed -- the archive is analysis-only -- but said out loud at record()
    # time rather than discovered at render time, twelve hours later.
    envir = _envir()
    swrm = _swarm(envir)
    with pytest.warns(UserWarning, match='usable for analysis but not for plotting'):
        rec = envir.record(tmp_path / 'run', store=('positions',))
    swrm.move(0.1, silent=True)
    envir.stop_recording()
    agents = rec.path / 'agents'
    assert list(agents.glob('swarm00_vel_*.npy')) == []
    assert len(_chunks(agents, 'swarm00_pos_*.npy')) == 2
    assert json.loads((rec.path / 'meta.json').read_text())['store'] == ['positions']


def test_recording_without_positions_raises(tmp_path):
    envir = _envir()
    _swarm(envir)
    with pytest.raises(ValueError, match='positions must be stored'):
        envir.record(tmp_path / 'run', store=('velocities',))


def test_restricting_which_swarms_are_captured(tmp_path):
    envir = _envir()
    a = _swarm(envir, seed=1)
    _swarm(envir, seed=2)
    with envir.record(tmp_path / 'run', swarms=[a]) as rec:
        envir.move_swarms(0.1, silent=True)
    agents = rec.path / 'agents'
    assert (agents / 'swarm00.json').is_file()
    assert not (agents / 'swarm01.json').exists()


def test_ftle_does_not_fire_captures(tmp_path):
    # calculate_FTLE inlines its own move loop rather than calling move(), so it
    # cannot fire the hook -- which is what is wanted, since it would otherwise
    # write FTLE probe trajectories into the archive.
    L, a, n = 10.0, 0.4, 21
    x = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    envir = planktos.Environment(Lx=L, Ly=L, flow=[a * (X - L / 2), -a * (Y - L / 2)])
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run') as rec:
        swrm.move(0.1, silent=True)
        envir.calculate_FTLE(grid_dim=(6, 6), T=0.3, dt=0.05)
    assert len(_chunks(rec.path / 'agents', 'times_*.npy')) == 2


def test_every_fluid_setter_guards_against_loading_while_recording():
    # The failure this catches is a guard that is present in the source but is
    # not a statement -- e.g. sitting inside a docstring, where it reads exactly
    # like working code and does nothing. Several of these methods open their
    # docstring with a bare ''' on its own line, which is what made that
    # possible. Checked by parsing rather than by grepping, for the same reason.
    import ast
    import inspect

    guarded = set()
    tree = ast.parse(inspect.getsource(planktos._environment))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, 'attr', '')
                    == '_refuse_while_recording'):
                guarded.add(node.name)

    expected = {'set_brinkman_flow', 'set_two_layer_channel_flow',
                'set_canopy_flow', 'read_IB2d_fluid_data',
                'read_IBAMR3d_vtk_data', 'read_openfoam_vtk_data',
                'read_comsol_vtu_data', 'read_NetCDF_flow', 'reset'}
    assert expected <= guarded, \
        'unguarded: {}'.format(sorted(expected - guarded))


# --------------------------------------------------------------------------- #
#                        the capture schedule (A3b)                            #
# --------------------------------------------------------------------------- #
# capture_interval=k keeps state every k-th step, in the archive AND in
# time_history / pos_history / vel_history -- the same set of states, from one
# predicate, so there is no second notion of "a recorded state" anywhere.

def _wall_envir(span=(0.5, 9.5)):
    """A vertical wall at x=5 over the given y-span, finely meshed.

    Two spans are used below. A wall crossing the whole domain forces head-on
    collisions; a short one lets agents travel around its end over several
    steps, which is the case a stale movement start point turns into a chord
    straight through the wall.
    """
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux',
                                 flow=[np.zeros((5, 5)), np.zeros((5, 5))])
    y = np.linspace(span[0], span[1], 9)
    mesh = np.zeros((len(y) - 1, 2, 2))
    mesh[:, 0, 0] = mesh[:, 1, 0] = 5.0
    mesh[:, 0, 1], mesh[:, 1, 1] = y[:-1], y[1:]
    envir.ibmesh = mesh
    envir.max_meshpt_dist = float(
        np.linalg.norm(mesh[:, 0, :] - mesh[:, 1, :], axis=1).max())
    return envir


GEOMETRY = {
    # span,          initial positions,                     drift
    'across': ((0.5, 9.5), [[4., 3.], [4., 7.], [2., 5.], [4.5, 4.]], (1.0, 0.2)),
    'around': ((4.0, 6.0), [[4., 3.], [4., 5.], [4.2, 3.5], [3., 2.]], (0.9, 0.9)),
}


def _drive_into_wall(tmp_path, capture_interval, geometry, steps=6, name='run'):
    """Deterministic run against a wall, returning positions and collisions.

    The trajectory is collected from the live positions attribute rather than
    from history -- which capture_interval decimates, and which is exactly what
    must not be allowed to matter.
    """
    span, init, mu = GEOMETRY[geometry]
    envir = _wall_envir(span)
    swrm = planktos.Swarm(swarm_size=len(init), envir=envir, seed=1,
                          init=np.array(init, float))
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array(mu, float)

    traj, hits = [np.ma.filled(swrm.positions, np.nan)], []
    with envir.record(tmp_path / name, capture_interval=capture_interval):
        for _ in range(steps):
            swrm.move(1.0, silent=True)
            traj.append(np.ma.filled(swrm.positions, np.nan))
            hits.append(np.asarray(swrm.ib_collision_idx).copy())
    return np.stack(traj), np.stack(hits)


@pytest.mark.parametrize('geometry', ['across', 'around'])
@pytest.mark.parametrize('k', [2, 3, 5])
def test_a_capture_schedule_does_not_change_what_happens(k, geometry, tmp_path):
    # THE test for this step. What is recorded must not change the physics --
    # and the collision path is where it could, since the movement start point
    # used to be read straight out of pos_history (A0 decoupled it).
    every_step, every_hit = _drive_into_wall(tmp_path, 1, geometry, name='every')
    coarse, coarse_hit = _drive_into_wall(tmp_path, k, geometry, name='coarse')

    assert np.array_equal(every_step, coarse), (
        'capture_interval={} changed the {} trajectory; what is recorded must '
        'not change what happens'.format(k, geometry))
    assert np.array_equal(every_hit, coarse_hit)
    assert (every_hit >= 0).any(), 'no collision occurred; the test proves nothing'
    if geometry == 'across':
        # the wall spans the domain, so nothing may reach the far side. (The
        # 'around' wall is short on purpose -- passing its end is the legitimate
        # multi-step path a stale start point would chord straight through.)
        assert (every_step[-1, :, 0] < 5.0 + 1e-12).all(),             'an agent got through the wall'


@pytest.mark.parametrize('steps, k', [(6, 1), (6, 3), (7, 3), (10, 5), (9, 4)])
def test_the_histories_hold_exactly_the_captured_states(steps, k, tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', capture_interval=k) as rec:
        for _ in range(steps):
            swrm.move(0.1, silent=True)

    times = _chunks(rec.path / 'agents', 'times_*.npy')
    # every history is the same length, and holds only captured states
    assert len(envir.time_history) == len(swrm.pos_history) == len(swrm.vel_history)
    assert len(times) == steps // k + 1
    # capture j is exactly full_pos_history[j] -- no index translation anywhere
    full = _stack(swrm.full_pos_history)
    assert np.allclose(_chunks(rec.path / 'agents', 'swarm00_pos_*.npy'),
                       full[:len(times)])
    assert np.allclose(times, (envir.time_history + [envir.time])[:len(times)])


def test_time_history_holds_only_the_captured_times(tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', capture_interval=3):
        for _ in range(9):
            swrm.move(0.5, silent=True)
    # t0, t3, t6 -- the run reached t=4.5, and t9 is the live state
    assert np.allclose(envir.time_history, [0.0, 1.5, 3.0])


def test_a_coarse_schedule_looks_like_a_run_at_a_larger_dt(tmp_path):
    # The framing capture_interval is specified under: the archive should look
    # exactly like the same run performed at k*dt.
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', capture_interval=4) as rec:
        for _ in range(12):
            swrm.move(0.25, silent=True)
    times = _chunks(rec.path / 'agents', 'times_*.npy')
    assert np.allclose(np.diff(times), 1.0), 'captures should be 4*dt apart'


def test_move_swarms_keeps_every_swarm_in_step_under_a_schedule(tmp_path):
    envir = _envir()
    a, b = _swarm(envir, seed=1), _swarm(envir, seed=2)
    with envir.record(tmp_path / 'run', capture_interval=3) as rec:
        for _ in range(9):
            envir.move_swarms(0.1, silent=True)
    for swrm in (a, b):
        assert len(swrm.pos_history) == len(swrm.vel_history) \
            == len(envir.time_history)
    agents = rec.path / 'agents'
    assert len(_chunks(agents, 'times_*.npy')) == 4
    for prefix in ('swarm00', 'swarm01'):
        assert len(_chunks(agents, prefix + '_pos_*.npy')) == 4


def test_a_failed_step_under_a_schedule_leaves_the_histories_consistent(tmp_path):
    # move()'s failure handler appends envir.time to close the histories off
    # together. Under a schedule it must append only when the failed step was
    # one that appended to pos_history, or it does the opposite of its job.
    class _Exploding(planktos.Swarm):
        def apply_agent_model(self, dt):
            if self.envir._step_count == 4:
                raise ValueError('boom')
            return super().apply_agent_model(dt)

    envir = _envir()
    swrm = _Exploding(swarm_size=3, envir=envir, seed=1,
                      init=np.full((3, 2), 2.0))
    swrm.shared_props['cov'] = np.zeros((2, 2))
    with envir.record(tmp_path / 'run', capture_interval=3):
        with pytest.raises(ValueError, match='boom'):
            for _ in range(9):
                swrm.move(0.1, silent=True)
    assert len(envir.time_history) == len(swrm.pos_history) == len(swrm.vel_history)


def test_the_histories_go_back_to_every_step_once_recording_stops(tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', capture_interval=4):
        for _ in range(8):
            swrm.move(0.1, silent=True)
    assert len(swrm.pos_history) == 2
    for _ in range(3):
        swrm.move(0.1, silent=True)
    assert len(swrm.pos_history) == 5, 'gating outlived the recording'
    assert len(envir.time_history) == len(swrm.pos_history)


def test_captures_are_evenly_spaced_when_recording_starts_mid_run(tmp_path):
    # Counted from the step recording began at, not from step zero, so the
    # spacing is k*dt throughout rather than short at the front.
    envir = _envir()
    swrm = _swarm(envir)
    for _ in range(2):
        swrm.move(0.5, silent=True)
    with envir.record(tmp_path / 'run', capture_interval=3) as rec:
        for _ in range(6):
            swrm.move(0.5, silent=True)
    times = _chunks(rec.path / 'agents', 'times_*.npy')
    assert np.allclose(times, [1.0, 2.5, 4.0])


def test_a_capture_interval_below_one_is_refused(tmp_path):
    envir = _envir()
    _swarm(envir)
    with pytest.raises(ValueError, match='capture_interval'):
        envir.record(tmp_path / 'run', capture_interval=0)
    assert envir._recorder is None
    assert envir._capture_interval == 1
