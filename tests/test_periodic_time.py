'''Pins fluid and moving immersed meshes that repeat in time (``periodic_time``).

Data covering one cycle repeats indefinitely: the field at time t is the field at
the same phase of the cycle. The series may arrive closed, its last dump
repeating the first a period later, or open, one dump interval short of the
period; a closed series' last dump is dropped, so the two give the same field.

Covered here:

- the field repeats, in every regime (cubic, linear resident, sliding window),
  and the window gives the resident answer over several cycles in both
  directions, including du/dt and the per-dump means;
- open and closed series give the same field, and a cubic spline is smooth
  across the wrap;
- how the period is settled: stated, or inferred from the series, and the
  refusals and warnings when the series does not fit;
- the loaders and the Environment: a real IB2d fixture, restore from a run
  archive, the fingerprint, a moving mesh that repeats, and refusing a fluid and
  mesh with different periods;
- a simulation over several cycles equals the same simulation against the data
  copied end to end in memory, with no agent crossing the moving wall.

No RNG and no external data; the moving mesh is written to tmp_path.
'''

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import planktos
from planktos import fluid, _archive_format

FIXTURES = Path(__file__).parent / 'fixtures'

T = 1.0        # period of the analytic test field
N = 12         # dumps per cycle


def _field(t, x, y):
    '''One phase of a periodic 2D field: u, v at time t on the grid x, y.'''
    X, Y = np.meshgrid(x, y, indexing='ij')
    w = 2*np.pi*t/T
    return [np.sin(w)*(1 + 0.1*X) + 0.3*np.cos(2*w),
            np.cos(w)*Y*0.2 + 0.1*np.sin(3*w)]


def _series(closed, n=N, x=None, y=None):
    '''Times and per-component arrays for one cycle, closed or open.'''
    x = np.linspace(0, 2, 5) if x is None else x
    y = np.linspace(0, 1, 4) if y is None else y
    times = np.arange(n + closed)*T/n
    comps = [np.stack([_field(t, x, y)[c] for t in times]) for c in range(2)]
    return times, (x, y), comps


def _quiet(fn, *args, **kwargs):
    '''Call fn with its printed report silenced.'''
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class _Source(fluid.FluidData):
    '''A streaming FluidData whose "disk" is an in-memory array, structured as the
    real loaders are. Records every load, and refuses a dump outside the series.'''

    def __init__(self, full, flow_points, flow_times, INUM, periodic_time):
        self._full = [np.asarray(f) for f in full]
        self.load_calls = []
        self.d_start = 0
        self.d_finish = len(flow_times) - 1
        last = min(INUM, self.d_finish)
        self.loaded_dump_bnds = (0, last)
        self.loaded_idx_bnds = (0, last)
        window = self.load_dumpfiles(0, last)
        super().__init__(window, flow_points, flow_times, INUM,
                         periodic_time=periodic_time)

    def load_dumpfiles(self, d_start, d_finish):
        assert 0 <= d_start <= d_finish <= len(self._full[0]) - 1
        self.load_calls.append((d_start, d_finish))
        return [f[d_start:d_finish + 1].copy() for f in self._full]


def _build(regime, closed, periodic_time=T):
    '''A periodic FluidData over the analytic cycle, in the named regime.'''
    times, points, comps = _series(closed)
    if regime == 'cubic':
        return _quiet(fluid.FluidData, comps, points, times, INUM=None,
                      periodic_time=periodic_time)
    if regime == 'linear':
        return _quiet(fluid.FluidData, comps, points, times, INUM=True,
                      periodic_time=periodic_time)
    return _quiet(_Source, comps, points, times, int(regime[len('window'):]),
                  periodic_time)


def _diff(a, b):
    return max(float(np.max(np.abs(np.asarray(p) - np.asarray(q))))
               for p, q in zip(a, b))


# Times over several cycles, in both directions, on and off the dumps and
#   at the wrap itself.
SWEEP = np.concatenate([np.linspace(0, 3.4, 157), np.linspace(3.4, -1.3, 61),
                        np.arange(-2, 4)*T, np.arange(N)*T/N + 2*T])


# --------------------------------------------------------------------------- #
#                            the field repeats                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('regime', ['cubic', 'linear', 'window3'])
@pytest.mark.parametrize('closed', [False, True])
def test_the_field_repeats_with_its_period(regime, closed):
    flow = _build(regime, closed)
    for t in np.linspace(-0.7, 0.99, 23):
        ref = flow(t)
        for k in (1, 2, 5):
            assert _diff(flow(t + k*T), ref) < 1e-12
        np.testing.assert_allclose(flow.get_mean_velocity(time=t + 3*T),
                                   flow.get_mean_velocity(time=t), atol=1e-12)


@pytest.mark.parametrize('regime', ['cubic', 'linear'])
def test_the_field_matches_the_data_at_every_dump_of_every_cycle(regime):
    times, points, comps = _series(closed=False)
    flow = _build(regime, closed=False)
    for k in (-1, 0, 3):
        for i, t in enumerate(times):
            assert _diff(flow(t + k*T), [c[i] for c in comps]) < 1e-12


def test_linear_interpolation_runs_from_the_last_dump_back_to_the_first():
    times, _, comps = _series(closed=False)
    flow = _build('linear', closed=False)
    for w in (0.25, 0.5, 0.75):
        t = times[-1] + w*(T - times[-1])
        expect = [(1 - w)*c[-1] + w*c[0] for c in comps]
        assert _diff(flow(t), expect) < 1e-12
        assert _diff(flow(t + 2*T), expect) < 1e-12


@pytest.mark.parametrize('regime', ['cubic', 'linear'])
@pytest.mark.parametrize('periodic_time', [True, T])
def test_closed_and_open_series_give_the_same_field(regime, periodic_time):
    open_ = _build(regime, closed=False, periodic_time=periodic_time)
    closed = _build(regime, closed=True, periodic_time=periodic_time)
    assert len(closed.flow_times) == len(open_.flow_times) == N
    assert closed.period == pytest.approx(T) and open_.period == pytest.approx(T)
    for t in SWEEP:
        assert _diff(closed(t), open_(t)) < 1e-12


def test_a_cubic_spline_is_smooth_across_the_wrap():
    # Periodic end conditions: u and du/dt agree on either side of the wrap.
    flow = _build('cubic', closed=False)
    eps = 1e-7
    assert _diff(flow(T - eps), flow(T + eps)) < 1e-5
    assert _diff(flow.get_dudt(time=T - eps), flow.get_dudt(time=T + eps)) < 1e-4
    # ...and du/dt is the analytic one, to the spline's accuracy.
    x, y = flow.flow_points
    h = 1e-6
    exact = [(a - b)/(2*h) for a, b in zip(_field(T + h, x, y),
                                           _field(T - h, x, y))]
    assert _diff(flow.get_dudt(time=T), exact) < 0.05*max(
        np.abs(e).max() for e in exact)


def test_time_points_of_a_cubic_spline_index_one_cycle():
    times, _, comps = _series(closed=True)
    flow = _build('cubic', closed=True)
    assert flow.fshape[0] == N
    np.testing.assert_allclose(flow[0][-1], comps[0][N - 1], atol=1e-12)
    assert len(flow.get_raw_loaded_data()[0]) == N


# --------------------------------------------------------------------------- #
#                  a sliding window gives the resident answer                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('INUM', [2, 3, 5, 10])
@pytest.mark.parametrize('closed', [False, True])
def test_a_window_matches_the_resident_field_over_several_cycles(INUM, closed):
    window = _build('window{}'.format(INUM), closed)
    resident = _build('linear', closed)
    assert window.is_windowed
    tol = 1e-13
    for t in SWEEP:
        assert _diff(window(t), resident(t)) < tol
        assert _diff(window.get_dudt(time=t), resident.get_dudt(time=t)) < 1e-11
        np.testing.assert_allclose(window.get_mean_velocity(time=t),
                                   resident.get_mean_velocity(time=t), atol=tol)
    # Every dump arrived, and each once in the per-dump means.
    assert not np.isnan(window.dump_means).any()


def test_a_window_reads_dump_zero_again_for_the_end_of_the_cycle():
    window = _build('window3', closed=False)
    n_opening = len(window.load_calls)
    window(T - 0.01)
    # The window now ends a period after the first dump, on dump 0 read again,
    #   which is not reported as resident a second time.
    assert window._flow[0].x[-1] == pytest.approx(T)
    assert window.load_calls[-1] == (0, 0)
    assert window.load_calls[n_opening:].count((0, 0)) == 1
    resident = [i for i, _ in window.iter_resident_dumps()]
    assert resident[-1] == N - 1 and 0 not in resident


def test_a_closed_series_never_loads_its_repeated_last_dump_after_the_check():
    window = _build('window3', closed=True)
    # The check at construction reads the last two dumps; nothing after it
    #   reaches dump N, which repeats dump 0.
    n_check = len(window.load_calls)
    for t in SWEEP:
        window(t)
    assert all(b < N for _, b in window.load_calls[n_check:])


def test_an_INUM_spanning_the_cycle_holds_it_all():
    # N dumps make N intervals around the cycle, so INUM=N leaves no window.
    times, points, comps = _series(closed=False)
    with pytest.warns(UserWarning, match='spans the entire dataset'):
        flow = _quiet(_Source, comps, points, times, N, T)
    assert not flow.is_windowed
    resident = _build('linear', closed=False)
    for t in SWEEP:
        assert _diff(flow(t), resident(t)) < 1e-13


def test_dumps_between_follows_the_wrap():
    flow = _build('linear', closed=False)
    dt = T/N
    assert flow._dumps_between(2*T + 3.5*dt, 2*T + 5.5*dt) == [3, 4, 5, 6]
    assert flow._dumps_between(T - 1.5*dt, T + 1.5*dt) == [0, 1, 2, 10, 11]
    assert flow._dumps_between(0.3, 0.3 + T) == list(range(N))


# --------------------------------------------------------------------------- #
#                         settling the period                                  #
# --------------------------------------------------------------------------- #

def _report(closed, periodic_time, n=N):
    times, points, comps = _series(closed, n=n)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        flow = fluid.FluidData(comps, points, times, INUM=True,
                               periodic_time=periodic_time)
    return flow, out.getvalue()


def test_true_infers_a_closed_series_and_drops_its_last_dump():
    flow, printed = _report(closed=True, periodic_time=True)
    assert flow.period == pytest.approx(T) and len(flow.flow_times) == N
    assert 'repeats the first' in printed and 'dropped' in printed


def test_true_infers_an_open_series_and_adds_one_dump_interval():
    flow, printed = _report(closed=False, periodic_time=True)
    assert flow.period == pytest.approx(T) and len(flow.flow_times) == N
    assert 'follows the last' in printed


def test_a_stated_period_decides_by_time_alone():
    # The same open series read as closed when told its span is the period,
    #   which leaves a jump of one dump interval's change at the wrap.
    times, points, comps = _series(closed=False)
    span = times[-1] - times[0]
    with pytest.warns(UserWarning, match='cycle jumps there'):
        flow = _quiet(fluid.FluidData, comps, points, times, INUM=True,
                      periodic_time=span)
    assert len(flow.flow_times) == N - 1 and flow.period == span


def test_true_refuses_ends_that_barely_change():
    # A cycle that settles at its ends: nothing moves near the wrap, so the
    #   series cannot show whether its last dump repeats the first.
    x, y = np.linspace(0, 1, 3), np.linspace(0, 1, 3)
    times = np.arange(20)/20
    pulse = np.where((times > 0.3) & (times < 0.7),
                     np.sin(np.pi*(times - 0.3)/0.4)**2, 0.0)
    comps = [np.stack([np.full((3, 3), p) for p in pulse]),
             np.zeros((20, 3, 3))]
    with pytest.raises(ValueError, match='barely changes'):
        fluid.FluidData(comps, (x, y), times, INUM=True, periodic_time=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        flow = _quiet(fluid.FluidData, comps, (x, y), times, INUM=True,
                      periodic_time=1.0)
    assert flow.period == 1.0 and len(flow.flow_times) == 20


def test_true_refuses_an_open_series_with_uneven_spacing():
    times, points, comps = _series(closed=False)
    times = times.copy()
    times[3] += 0.01
    with pytest.raises(ValueError, match='unevenly spaced'):
        fluid.FluidData(comps, points, times, INUM=True, periodic_time=True)


@pytest.mark.parametrize('bad', [0.5, 0.0, -1.0])
def test_a_period_shorter_than_the_series_or_not_positive_raises(bad):
    times, points, comps = _series(closed=False)
    with pytest.raises(ValueError):
        fluid.FluidData(comps, points, times, INUM=True, periodic_time=bad)


def test_time_invariant_flow_cannot_be_periodic():
    x, y = np.linspace(0, 1, 3), np.linspace(0, 1, 3)
    with pytest.raises(ValueError, match='time-invariant'):
        fluid.FluidData([np.zeros((3, 3))]*2, (x, y), periodic_time=1.0)


def _drifting(closed, drift):
    '''A cycle that has not settled: it drifts by `drift` dump intervals'
    change over the cycle, so its end no longer meets its start.'''
    times, points, comps = _series(closed)
    step = max(np.abs(c[1] - c[0]).max() for c in comps)
    ramp = drift*step*np.arange(len(times))/N
    return times, points, [c + ramp[:, None, None] for c in comps]


def test_a_closed_series_whose_ends_disagree_warns():
    times, points, comps = _drifting(closed=True, drift=1.0)
    with pytest.warns(UserWarning, match='cycle jumps there'):
        _quiet(fluid.FluidData, comps, points, times, INUM=True, periodic_time=T)


def test_an_open_series_that_has_not_settled_warns():
    times, points, comps = _drifting(closed=False, drift=5.0)
    with pytest.warns(UserWarning, match='may not have settled'):
        _quiet(fluid.FluidData, comps, points, times, INUM=True, periodic_time=T)


def test_a_wrap_interval_unlike_the_dump_spacing_warns():
    times, points, comps = _series(closed=False)
    with pytest.warns(UserWarning, match='back to its first is'):
        _quiet(fluid.FluidData, comps, points, times, INUM=True,
               periodic_time=1.5*T)


@pytest.mark.parametrize('closed', [False, True])
def test_a_series_that_fits_raises_no_warning(closed):
    times, points, comps = _series(closed)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _quiet(fluid.FluidData, comps, points, times, INUM=True, periodic_time=T)


# --------------------------------------------------------------------------- #
#                       loaders and the Environment                            #
# --------------------------------------------------------------------------- #

# ib2d_fluid_min: dumps 0..7, u = dump number, so u = 10 t at dt*print_dump = 0.1.
IB2D_PERIOD = 0.8


def _ib2d_envir(INUM=None):
    envir = planktos.Environment()
    with pytest.warns(UserWarning, match='may not have settled'):
        _quiet(envir.read_IB2d_fluid_data, str(FIXTURES / 'ib2d_fluid_min'),
               dt=0.01, print_dump=10, INUM=INUM, periodic_time=IB2D_PERIOD)
    return envir


@pytest.mark.parametrize('INUM', [True, 4])
def test_the_ib2d_loader_repeats_its_series(INUM):
    envir = _ib2d_envir(INUM)
    flow = envir.flow
    assert flow.period == IB2D_PERIOD
    for t, u in ((0.25, 2.5), (0.25 + 3*IB2D_PERIOD, 2.5), (0.75, 3.5),
                 (0.8 + 0.05, 0.5), (-0.05, 3.5)):
        assert flow(t)[0] == pytest.approx(np.full(flow.spatial_shape, u))


def test_restore_rebuilds_a_fluid_that_repeats(tmp_path):
    envir = _ib2d_envir()
    planktos.Swarm(swarm_size=3, envir=envir, init=np.array(
        [[1., 1.], [2., 2.], [3., 1.]]))
    with envir.record(tmp_path / 'run') as rec:
        for _ in range(3):
            envir.swarms[0].move(0.4, silent=True)
    run = planktos.load_run(rec.path)
    try:
        with pytest.warns(UserWarning, match='may not have settled'):
            rebuilt, _ = _quiet(run.restore)
    finally:
        run.close()
    assert rebuilt.flow.period == IB2D_PERIOD
    assert _diff(rebuilt.flow(1.0), envir.flow(1.0)) == 0


def test_the_fingerprint_names_the_period_only_when_the_fluid_repeats():
    periodic = _archive_format.fingerprint_of(_ib2d_envir())
    assert float(periodic['period']) == IB2D_PERIOD
    envir = planktos.Environment()
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    assert 'period' not in _archive_format.fingerprint_of(envir)


def test_the_environment_constructor_takes_periodic_time():
    times, points, comps = _series(closed=False)
    envir = _quiet(planktos.Environment, Lx=2, Ly=1, flow=comps,
                   flow_times=times, periodic_time=T)
    assert envir.flow.period == T
    assert envir._fluid_provenance['kwargs']['periodic_time'] == T


# --------------------------------------------------------------------------- #
#                         a moving mesh that repeats                           #
# --------------------------------------------------------------------------- #

WALL_Y = (0.5, 9.5)


def _wall_x(t):
    return 5.0 + 1.5*np.sin(2*np.pi*t/T)


def _write_wall(directory, frames=N, x_of=_wall_x):
    '''lagsPts files for a vertical wall of 10 vertices oscillating in x, one
    frame every T/N.'''
    directory.mkdir(parents=True, exist_ok=True)
    ys = np.linspace(*WALL_Y, 10)
    for k in range(frames):
        pts = np.column_stack([np.full(10, x_of(k*T/N)), ys, np.zeros(10)])
        grid = pv.PolyData(pts).cast_to_unstructured_grid()
        grid.save(str(directory / 'lagsPts.{:04d}.vtk'.format(k)), binary=False)
    return directory


def _load_wall(envir, directory, periodic_time=T):
    _quiet(envir.read_IB2d_mesh_data, str(directory), dt=T/N, print_dump=1,
           periodic_time=periodic_time)


@pytest.mark.parametrize('closed', [False, True])
def test_a_moving_mesh_repeats_with_its_period(tmp_path, closed):
    envir = planktos.Environment(Lx=10, Ly=10)
    _load_wall(envir, _write_wall(tmp_path / 'wall', frames=N + closed))
    assert envir.ibmesh_period == pytest.approx(T)
    assert envir.ibmesh.shape[0] == N
    for t in np.linspace(-0.4, 0.99, 17):
        ref = envir.interpolate_temporal_mesh(time=t)
        for k in (1, 4):
            np.testing.assert_allclose(
                envir.interpolate_temporal_mesh(time=t + k*T), ref, atol=1e-12)
    # From the last frame back to the first across the wrap.
    t_last = (N - 1)*T/N
    mid = envir.interpolate_temporal_mesh(time=t_last + 0.5*(T - t_last))
    np.testing.assert_allclose(mid[:, :, 0],
                               0.5*(_wall_x(t_last) + _wall_x(0)), atol=1e-12)


def test_a_static_mesh_cannot_repeat(tmp_path):
    envir = planktos.Environment(Lx=10, Ly=10)
    with pytest.raises(ValueError, match='static'):
        envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min' / 'box.vertex'),
                                  periodic_time=T)


def test_a_fluid_and_mesh_with_different_periods_are_refused(tmp_path):
    times, points, comps = _series(closed=False)
    envir = _quiet(planktos.Environment, Lx=10, Ly=10, flow=comps,
                   flow_times=times*2, periodic_time=2*T)
    with pytest.raises(ValueError, match='repeats every'):
        _load_wall(envir, _write_wall(tmp_path / 'wall'))
    assert envir.ibmesh is None and envir.ibmesh_period is None


def test_a_fluid_that_repeats_with_a_mesh_that_does_not_warns(tmp_path):
    times, points, comps = _series(closed=False)
    envir = _quiet(planktos.Environment, Lx=10, Ly=10, flow=comps,
                   flow_times=times, periodic_time=T)
    with pytest.warns(UserWarning, match='holds its last state'):
        _load_wall(envir, _write_wall(tmp_path / 'wall'), periodic_time=None)


# --------------------------------------------------------------------------- #
#             a run over several cycles == the data repeated in memory         #
# --------------------------------------------------------------------------- #

CYCLES = 3


def _uniform_flow(t, x, y):
    X, _ = np.meshgrid(x, y, indexing='ij')
    return [np.full(X.shape, 1.0 + 0.8*np.sin(2*np.pi*t/T)), np.zeros(X.shape)]


def _run(envir):
    '''Deterministic agents drifting with the fluid into the moving wall.'''
    init = np.column_stack([np.linspace(2.5, 3.4, 8), np.linspace(1, 9, 8)])
    swrm = planktos.Swarm(swarm_size=8, envir=envir, init=init)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    walls, hits = [], []
    for _ in range(int(round(CYCLES*T/0.01))):
        swrm.move(0.01, silent=True)
        walls.append(envir.interpolate_temporal_mesh()[0, 0, 0])
        hits.append(np.count_nonzero(swrm.ib_collision_idx >= 0))
    return swrm, np.array(walls), np.array(hits)


def test_a_run_over_several_cycles_matches_the_data_repeated(tmp_path):
    x, y = np.linspace(0, 10, 11), np.linspace(0, 10, 11)
    times = np.arange(N)*T/N
    comps = [np.stack([_uniform_flow(t, x, y)[c] for t in times])
             for c in range(2)]
    periodic = _quiet(planktos.Environment, Lx=10, Ly=10, x_bndry='noflux',
                      y_bndry='noflux', flow=comps, flow_times=times,
                      periodic_time=T)
    _load_wall(periodic, _write_wall(tmp_path / 'wall'))
    # Linear in time in both, so the repeated copy is the same field.
    periodic.flow = _quiet(fluid.FluidData, comps, periodic.flow.flow_points,
                           times, INUM=True, periodic_time=T)

    n_all = CYCLES*N + 1
    all_times = np.arange(n_all)*T/N
    repeated = planktos.Environment(
        Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux',
        flow=[np.concatenate([c]*CYCLES + [c[:1]]) for c in comps],
        flow_times=all_times)
    repeated.flow = fluid.FluidData(
        [np.concatenate([c]*CYCLES + [c[:1]]) for c in comps],
        repeated.flow.flow_points, all_times, INUM=True)
    _load_wall(repeated, _write_wall(tmp_path / 'long', frames=n_all,
                                     x_of=lambda t: _wall_x(t % T)),
               periodic_time=None)

    a, walls, hits = _run(periodic)
    b, _, _ = _run(repeated)
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-9)
    for pa, pb in zip(a.pos_history, b.pos_history):
        np.testing.assert_allclose(pa, pb, atol=1e-9)
    # The wall met the agents in every cycle, and none crossed it.
    assert all(cycle.sum() > 0 for cycle in np.split(hits, CYCLES))
    xs = np.array([p[:, 0] for p in a.pos_history[1:]] + [a.positions[:, 0]])
    assert np.all(xs <= walls[:, None])
