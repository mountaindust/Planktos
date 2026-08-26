'''Tests for the fluid half of a run archive -- component B of
docs/notes/run_persistence.md (sections 3.3 to 3.6, build step B).

Where test_recording.py pins the *agent* half, this pins what an archive stores
about the fluid, and the property that matters is the same one: a render must
never re-stream the dataset to draw a picture of it. Three things are stored, and
each has its own rule because the regimes genuinely differ.

**Vorticity is not cached -- it is sourced, by regime** (section 3.3). Which of
three things happens is decided by whether a sliding window is in use and by
whether the source already ships the field:

    whole field resident      nothing written; recompute at render
    windowed, source has it   read the source's per-dump field
    windowed, source has not  Planktos writes one per dump as it lands

The headline test here is that the middle and bottom rows both reproduce
``envir.get_vorticity(time=t)`` exactly, and agree with each other. That is
section 3.2's linearity argument made executable: ``LinearSpline`` evaluates as a
weighted sum of two adjacent nodal fields and the curl is linear, so blending
per-dump vorticity with the same two weights *is* the curl of the blend. For the
top row there is nothing to compare, so what is asserted instead is that **no
vorticity file was written anywhere** -- the regime's whole content, and something
that would otherwise fail silently by costing disk nobody asked for.

Fixture: ``ib2d_fluid_vort_min`` carries ``u`` dumps *and* IB2d's own
``Omega.####.vtk``, with a velocity field whose vorticity varies nonlinearly in
time -- so a reader that took the nearest dump instead of blending the two
bracketing ones fails here, where against a steady field it would pass.
Regenerate it with tests/fixtures/_gen_fixtures.py.

The no-vorticity cases copy that same fixture and drop its ``Omega`` files,
rather than using a second fixture: the sourced and the written regimes have to be
compared *on the same data*, and they cannot be if each has its own field. It also
means every test that writes does so into a temporary directory, so nothing here
can modify a committed fixture -- which the middle regime must not do, and which
one of the tests below checks.
'''

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import planktos
from planktos import archive

FIXTURES = Path(__file__).parent / 'fixtures'
VORT_FIXTURE = FIXTURES / 'ib2d_fluid_vort_min'

IB2D_DT = 0.1
IB2D_PRINT_DUMP = 10
NDUMPS = 8


# --------------------------------------------------------------------------- #
#                                  helpers                                    #
# --------------------------------------------------------------------------- #

def _envir(src, INUM=4):
    '''An IB2d environment on one of the fixtures.

    Periodic agent boundaries so that agents wrap and stay in the domain: once
    every agent has left, ``move`` stops asking for fluid and the window stops
    sliding, which would quietly turn a streaming test into a resident one.
    '''

    envir = planktos.Environment(x_bndry='periodic', y_bndry='periodic')
    envir.read_IB2d_fluid_data(str(src), dt=IB2D_DT,
                              print_dump=IB2D_PRINT_DUMP, INUM=INUM)
    return envir


def _copy_fixture(tmp_path, name, dumps=()):
    '''Copy the fixture somewhere writable, with only `dumps` of its Omega series.

    Defaults to no Omega at all, which is the "solver printed none" case; pass a
    set of dump numbers for the complete or partial cases.
    '''

    dest = tmp_path / name
    dest.mkdir(parents=True)
    for f in sorted(VORT_FIXTURE.glob('u.*.vtk')):
        shutil.copy(f, dest)
    for f in sorted(VORT_FIXTURE.glob('Omega.*.vtk')):
        if int(f.name[6:10]) in dumps:
            shutil.copy(f, dest)
    return dest


def _sweep(envir, steps=14, dt=0.5, n=3):
    '''Run far enough that the window slides across the whole series.'''

    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=3)
    for _ in range(steps):
        swrm.move(dt, silent=True)
    return swrm


def _record_and_sweep(envir, path, steps=14, **kwargs):
    with envir.record(str(path), **kwargs) as rec:
        _sweep(envir, steps=steps)
    return rec


def _meta(rec):
    return json.loads((rec.path / 'meta.json').read_text())


def _omega_files(directory):
    return sorted(p.name for p in Path(directory).glob('Omega.*.vtk'))


# --------------------------------------------------------------------------- #
#         the three regimes of section 3.3, and which one gets chosen         #
# --------------------------------------------------------------------------- #

# INUM=None splines the resident dataset cubically and INUM=True linearly. The
# regime is decided by what is *resident*, not by which spline class is in use,
# so both must land in the top row.
@pytest.mark.parametrize('INUM', [None, True])
def test_a_resident_field_writes_no_vorticity_anywhere(tmp_path, INUM):
    # Nothing to compare a blend against here, so what the regime *is* has to be
    # asserted directly: with the whole field in memory, recomputing is cheaper
    # than reading a field back, so writing would buy negative performance.
    src = _copy_fixture(tmp_path, 'resident')
    envir = _envir(src, INUM=INUM)
    rec = _record_and_sweep(envir, tmp_path / 'archive')

    assert _meta(rec)['fluid']['vorticity'] == 'recomputed'
    assert _omega_files(src) == []
    assert _omega_files(rec.path / 'fluid') == []
    # dump_stats is the one thing that is always written: a few floats per dump,
    # and what lets the plot statistics come off an archive at all.
    assert sorted(p.name for p in (rec.path / 'fluid').iterdir()) \
        == ['dump_stats.npz']


def test_a_source_that_ships_vorticity_is_read_and_not_rewritten(tmp_path):
    # The middle row. The fixture is used in place rather than copied, which is
    # itself part of the assertion: this regime must not write into it.
    before = _omega_files(VORT_FIXTURE)
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive')

    meta = _meta(rec)['fluid']
    assert meta['vorticity'] == 'source'
    assert Path(meta['vorticity_dir']) == VORT_FIXTURE.resolve()
    assert _omega_files(VORT_FIXTURE) == before
    assert _omega_files(rec.path / 'fluid') == []


def test_a_source_without_vorticity_gets_a_complete_series_written(tmp_path):
    # The bottom row. Written beside the velocity dumps, in the source's own
    # naming, so the source becomes indistinguishable from one whose solver had
    # printed vorticity all along.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive')

    meta = _meta(rec)['fluid']
    assert meta['vorticity'] == 'source'
    assert Path(meta['vorticity_dir']) == src.resolve()
    assert _omega_files(src) == ['Omega.{:04d}.vtk'.format(k)
                                 for k in range(NDUMPS)]
    assert _omega_files(rec.path / 'fluid') == []
    # ...and IB2d's own reader must be able to read it, which is the whole point
    # of writing vtk rather than .npy.
    from planktos import _dataio
    back = _dataio.read_2DEulerian_Data_From_vtk(src, '0003', 'Omega')
    assert back.shape == (5, 6)          # [y,x], endpoint omitted as IB2d does


def test_a_partial_series_in_the_source_is_left_untouched(tmp_path):
    # A file that is already there is the solver's own field, which section 3.3
    # measures as the better one where the two differ. Nothing Planktos writes
    # may land on top of it -- and because a partial series sends the write to
    # the archive instead, that holds for the whole directory and not just the
    # dumps that have one.
    src = _copy_fixture(tmp_path, 'partial', dumps={2})
    keep = src / 'Omega.0002.vtk'
    original = keep.read_bytes()
    envir = _envir(src, INUM=4)
    with pytest.warns(UserWarning, match='only part of the dump range'):
        rec = _record_and_sweep(envir, tmp_path / 'archive')

    assert keep.read_bytes() == original
    assert _omega_files(src) == ['Omega.0002.vtk']
    assert len(_omega_files(rec.path / 'fluid')) == NDUMPS


def test_the_writer_refuses_to_overwrite_a_dump_it_finds(tmp_path):
    # Defence in depth. probe_stored_vorticity is what normally keeps this from
    # arising -- a directory that has the field is read from, and a partial one
    # is written past -- so this guard is unreachable through record() today. It
    # is asserted directly rather than dropped because it makes _write_vorticity
    # correct standing alone, independent of a policy decided elsewhere.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    with envir.record(str(tmp_path / 'archive')) as rec:
        writer = rec._fluid
        target = src / 'Omega.0003.vtk'
        planted = b'not a vtk file at all'
        target.write_bytes(planted)
        writer._write_vorticity(3, np.zeros(envir.flow.fshape[1:]))
        assert target.read_bytes() == planted


def test_a_partial_stored_series_is_not_used(tmp_path):
    # Serving one dump's field for another's is a plausible-looking wrong answer
    # (section 2.8), so a source covering part of the range is refused as a
    # source. The existing files are left alone and a complete, homogeneous
    # series is written elsewhere -- mixing the two would be the very thing
    # being avoided.
    src = _copy_fixture(tmp_path, 'partial', dumps={0, 1, 2})
    envir = _envir(src, INUM=4)
    with pytest.warns(UserWarning, match='only part of the dump range'):
        rec = _record_and_sweep(envir, tmp_path / 'archive')

    assert _meta(rec)['fluid']['vorticity'] == 'archive'
    assert _omega_files(src) == ['Omega.{:04d}.vtk'.format(k) for k in range(3)]
    assert _omega_files(rec.path / 'fluid') == [
        'Omega.{:04d}.vtk'.format(k) for k in range(NDUMPS)]


def test_vorticity_falls_back_to_the_archive_when_the_source_is_unwritable(
        tmp_path, monkeypatch):
    # Read-only mounts and shared datasets are normal, so this is a designed-for
    # case rather than an error. meta.json records which of the two happened, so
    # a reader knows where to look.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    monkeypatch.setattr(type(envir.flow), 'source_dir',
                        lambda self: tmp_path / 'does_not_exist')
    rec = _record_and_sweep(envir, tmp_path / 'archive')

    meta = _meta(rec)['fluid']
    assert meta['vorticity'] == 'archive'
    assert meta['vorticity_dir'] is None
    assert _omega_files(src) == []
    assert _omega_files(rec.path / 'fluid') == [
        'Omega.{:04d}.vtk'.format(k) for k in range(NDUMPS)]


def test_writable_source_dir_tests_rather_than_assumes(tmp_path):
    # Discovering this at the first dump would mean discovering it after the run
    # has started, so it is probed at record() time.
    envir = _envir(VORT_FIXTURE, INUM=4)
    assert archive._writable_source_dir(envir.flow) == VORT_FIXTURE.resolve()

    class _NoDir:
        def source_dir(self):
            return tmp_path / 'nope'
    assert archive._writable_source_dir(_NoDir()) is None

    class _NoSource:
        def source_dir(self):
            return None
    assert archive._writable_source_dir(_NoSource()) is None
    # nothing left behind by the probe itself
    assert not (VORT_FIXTURE / '.planktos_write_probe').exists()


# --------------------------------------------------------------------------- #
#      the headline: a blended per-dump field IS the curl of the velocity     #
# --------------------------------------------------------------------------- #
# Section 3.2 made executable. Times are chosen off the dump cadence so the
# blend does real work; the fixture's vorticity is nonlinear in time, so a
# nearest-dump reader fails these.

BLEND_TIMES = (0.25, 2.3, 4.5, 5.75, 6.9)


def test_sourced_vorticity_blended_equals_the_live_curl(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')

    for t in BLEND_TIMES:
        live = envir.get_vorticity(time=t)
        disk = envir.flow.get_stored_vorticity(t)
        scale = np.abs(live).max()
        # The fixture's Omega is ascii, as IB2d writes it: ~12 significant
        # digits, which is the floor on agreement here and not slack in the
        # blend. The written case below is exact for the same computation.
        assert np.abs(live - disk).max() < 1e-9 * scale


def test_written_vorticity_blended_equals_the_live_curl_exactly(tmp_path):
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')

    for t in BLEND_TIMES:
        live = envir.get_vorticity(time=t)
        disk = envir.flow.get_stored_vorticity(t)
        # Binary vtk, so this is round-off on the blend arithmetic alone.
        assert np.allclose(live, disk, rtol=0, atol=1e-12 * np.abs(live).max())


def test_sourced_and_written_vorticity_agree(tmp_path):
    # The two paths must not merely each be self-consistent: they are the same
    # field and have to come out the same.
    sourced = _envir(VORT_FIXTURE, INUM=4)
    _record_and_sweep(sourced, tmp_path / 'a_sourced')

    src = _copy_fixture(tmp_path, 'novort')
    written = _envir(src, INUM=4)
    _record_and_sweep(written, tmp_path / 'a_written')

    for t in BLEND_TIMES:
        a = sourced.flow.get_stored_vorticity(t)
        b = written.flow.get_stored_vorticity(t)
        assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()


def test_the_blend_is_not_the_nearest_dump(tmp_path):
    # Without this the two tests above would pass against a reader that snapped
    # to a dump, since a midpoint of a slowly varying field is close to both
    # ends. It is not close: the fixture's vorticity carries a t**2 term.
    envir = _envir(VORT_FIXTURE, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')

    mid = envir.flow.get_stored_vorticity(2.5)
    lo = envir.flow.get_stored_vorticity(2.0)
    hi = envir.flow.get_stored_vorticity(3.0)
    assert np.abs(mid - lo).max() > 0.1 * np.abs(lo).max()
    assert np.abs(mid - hi).max() > 0.1 * np.abs(hi).max()
    assert np.allclose(mid, 0.5*(lo + hi))


def test_a_time_outside_the_data_is_clamped_not_extrapolated(tmp_path):
    # Matching FluidData.__call__, which holds the velocity constant outside its
    # own bounds. A vorticity that extrapolated where the velocity does not
    # would stop being the curl of the field in use.
    envir = _envir(VORT_FIXTURE, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')

    assert np.array_equal(envir.flow.get_stored_vorticity(-5.0),
                          envir.flow.get_stored_vorticity(0.0))
    assert np.array_equal(envir.flow.get_stored_vorticity(99.0),
                          envir.flow.get_stored_vorticity(7.0))


def test_a_blended_read_is_refused_under_cubic_splining(tmp_path):
    # Not-a-knot weights are global, so applying them from per-dump files would
    # mean holding the whole series -- the memory cost this branch exists to
    # avoid. That regime has the field resident and calls get_vorticity instead.
    envir = _envir(VORT_FIXTURE, INUM=None)
    envir.flow.vorticity_path = VORT_FIXTURE
    with pytest.raises(RuntimeError, match='cubically in time'):
        envir.flow.get_stored_vorticity(2.3)


def test_reading_vorticity_with_no_location_established_raises(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    assert envir.flow.vorticity_path is None
    with pytest.raises(RuntimeError, match='no per-dump vorticity location'):
        envir.flow.read_dump_vorticity(0)


def test_a_missing_dump_file_names_itself(tmp_path):
    # A series Planktos wrote covers only the dumps the run actually loaded, so
    # asking for one outside that is a real case. It must say which file.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive', steps=2)   # sweeps 0-4 only
    with pytest.raises(FileNotFoundError, match='Omega.0007.vtk'):
        envir.flow.read_dump_vorticity(7)


# --------------------------------------------------------------------------- #
#                   the observer, and its two hard requirements               #
# --------------------------------------------------------------------------- #

def test_the_observer_sees_every_dump_exactly_once_on_a_forward_sweep(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    seen = []
    envir.flow.add_dump_observer(lambda i, f: seen.extend(
        range(i, i + len(f[0]))))
    _sweep(envir)
    # The opening window is not reported through the observer -- it landed before
    # anything could register -- so what this pins is the slides: dumps 5, 6, 7
    # arrive once each, in order, and the two holdovers are not re-reported.
    assert seen == [5, 6, 7]


def test_the_jump_to_start_reload_writes_nothing_twice(tmp_path, monkeypatch):
    # update_spline's jump-to-start branch reloads the opening window and
    # re-reports dumps that have already been seen. An observer that wrote
    # unconditionally would duplicate the work; the sidecar would be rewritten
    # with the same numbers and every file rewritten with the same bytes, so the
    # only way to see it is to watch the writes themselves.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    with envir.record(str(tmp_path / 'archive')) as rec:
        _sweep(envir)                       # slide to the end
        assert rec._fluid._written == set(range(NDUMPS))
        calls = []
        monkeypatch.setattr(type(envir.flow), 'write_dump_vorticity',
                            lambda self, t_idx, vort, path: calls.append(t_idx))
        envir.flow(0.0)                     # jump back to the start
    assert envir.flow.loaded_idx_bnds[0] == 0      # the jump really happened
    assert calls == []
    assert _omega_files(src) == ['Omega.{:04d}.vtk'.format(k)
                                for k in range(NDUMPS)]


def test_stopping_a_recording_unhooks_the_observer(tmp_path):
    # An observer that outlived its recording would keep writing into a source
    # directory after the archive it belonged to was closed.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    envir.record(str(tmp_path / 'archive'))
    swrm = planktos.Swarm(swarm_size=3, envir=envir, seed=3)
    for _ in range(2):                      # resident window only
        swrm.move(0.5, silent=True)
    envir.stop_recording()
    assert envir.flow._dump_observers == []
    before = _omega_files(src)
    assert len(before) == 5                 # just the opening window
    for _ in range(12):                     # now slide, with nothing recording
        swrm.move(0.5, silent=True)
    assert envir.flow.loaded_idx_bnds[1] == NDUMPS - 1
    assert _omega_files(src) == before


def test_time_invariant_flow_is_captured_once(tmp_path):
    # A steady field has no dumps to arrive, so the observer never fires and the
    # writer has to take it at construction instead.
    envir = planktos.Environment(Lx=10, Ly=10, rho=1000, mu=1000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=0.22, res=9)
    swrm = planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    with envir.record(str(tmp_path / 'archive')) as rec:
        for _ in range(3):
            swrm.move(0.1, silent=True)

    stats = np.load(rec.path / 'fluid' / 'dump_stats.npz')
    assert stats['means'].shape == (1, 2)
    assert not np.isnan(stats['means']).any()
    assert np.isclose(stats['means'][0, 0], np.mean(envir.flow[0]))
    assert 'flow_times' not in stats.files
    # Resident by definition, so nothing is written.
    assert _meta(rec)['fluid']['vorticity'] == 'recomputed'


# --------------------------------------------------------------------------- #
#                        the two-slot read cache                              #
# --------------------------------------------------------------------------- #

def test_the_read_cache_holds_two_dumps_and_no_more(tmp_path):
    # Holding more field data than the interpolation needs is the thing this
    # whole component exists to avoid, so the bound is part of the design and
    # not an implementation detail.
    envir = _envir(VORT_FIXTURE, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')
    for t in np.linspace(0.1, 6.9, 40):
        envir.flow.get_stored_vorticity(float(t))
        assert len(envir.flow._vort_cache) <= 2


@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_a_monotone_sweep_reads_each_dump_once(tmp_path, direction, monkeypatch):
    # A movie renders many frames per dump interval, so the naive path re-reads
    # two files per frame. Two slots reduce that to one read per dump interval
    # -- in either direction, which is why eviction is by distance from the
    # request rather than by insertion order.
    envir = _envir(VORT_FIXTURE, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')

    reads = []
    original = type(envir.flow).read_dump_vorticity

    def counted(self, t_idx):
        reads.append(t_idx)
        return original(self, t_idx)

    monkeypatch.setattr(type(envir.flow), 'read_dump_vorticity', counted)
    frames = np.linspace(0.05, 6.95, 60)
    if direction == 'backward':
        frames = frames[::-1]
    for t in frames:
        envir.flow.get_stored_vorticity(float(t))

    # 8 dumps: one read each, plus the one the first frame needs on its far side.
    assert len(reads) <= NDUMPS + 1
    assert sorted(set(reads)) == list(range(NDUMPS))
    assert len(reads) == len(set(reads))        # nothing read twice


# --------------------------------------------------------------------------- #
#                    the per-dump statistics sidecar                         #
# --------------------------------------------------------------------------- #

def test_dump_stats_records_means_extrema_and_the_vorticity_scale(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive')
    stats = np.load(rec.path / 'fluid' / 'dump_stats.npz')

    assert sorted(stats.files) == ['flow_times', 'means', 'vmax', 'vmin',
                                  'vort_absmax']
    assert stats['means'].shape == (NDUMPS, 2)
    assert np.array_equal(stats['flow_times'], envir.flow.flow_times)
    assert not np.isnan(stats['means']).any()      # the sweep covered them all

    # Every row must be the reduction of that dump, not of a neighbour.
    for t_idx in range(NDUMPS):
        field = envir.flow(float(envir.flow.flow_times[t_idx]))
        for n in range(2):
            assert np.isclose(stats['means'][t_idx, n], np.mean(field[n]))
            assert np.isclose(stats['vmin'][t_idx, n], np.min(field[n]))
            assert np.isclose(stats['vmax'][t_idx, n], np.max(field[n]))
        vort = envir.get_vorticity(t_n=t_idx)
        assert np.isclose(stats['vort_absmax'][t_idx], np.abs(vort).max())


def test_a_dump_that_never_loaded_is_nan_rather_than_zero(tmp_path):
    # Under a sliding window a short run simply never sees the later dumps. NaN
    # is the honest record; a zero would be indistinguishable from a still fluid
    # and would drag a global colour scale with it.
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive', steps=2)
    stats = np.load(rec.path / 'fluid' / 'dump_stats.npz')

    assert not np.isnan(stats['means'][:5]).any()   # the opening window
    assert np.isnan(stats['means'][5:]).all()
    assert np.isnan(stats['vort_absmax'][5:]).all()


def test_dump_stats_is_written_in_3d_where_no_backdrop_is_drawn(tmp_path):
    # Section 0.2: the statistics are the entire 3D deliverable of this
    # component. Everything else here is 2D-only, so this is the one thing a 3D
    # archive must still carry.
    envir = planktos.Environment()
    envir.read_IBAMR3d_vtk_data(str(FIXTURES / 'vtk3d_min'),
                                title='IBAMR_db_', INUM=4)
    swrm = planktos.Swarm(swarm_size=3, envir=envir, seed=2)
    with envir.record(str(tmp_path / 'archive')) as rec:
        for _ in range(6):
            swrm.move(0.5, silent=True)

    meta = _meta(rec)['fluid']
    assert meta['quantities'] == []              # fluid= forced to None in 3D
    assert 'vorticity' not in meta
    stats = np.load(rec.path / 'fluid' / 'dump_stats.npz')
    assert 'vort_absmax' not in stats.files      # no 3D vorticity is written
    assert stats['means'].shape == (8, 3)
    assert not np.isnan(stats['means'][:5]).any()
    assert _omega_files(rec.path / 'fluid') == []


# --------------------------------------------------------------------------- #
#                                  quiver                                     #
# --------------------------------------------------------------------------- #

def test_quiver_is_opt_in(tmp_path):
    # A second full-cadence array on disk for a backdrop most runs never draw.
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive')
    assert list((rec.path / 'fluid').glob('quiver_*.npy')) == []
    assert 'quiver_shape' not in _meta(rec)['fluid']


def test_quiver_stores_the_strided_velocity_per_dump(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive',
                            fluid=('vort', 'quiver'), quiver_shape=(4, 3))
    meta = _meta(rec)['fluid']
    M, N = meta['quiver_strides']

    files = sorted((rec.path / 'fluid').glob('quiver_*.npy'))
    assert len(files) == NDUMPS
    for t_idx in range(NDUMPS):
        arrows = np.load(rec.path / 'fluid'
                         / 'quiver_{:05d}.npy'.format(t_idx))
        field = envir.flow(float(envir.flow.flow_times[t_idx]))
        # Exactly the slice plot_all draws, so nothing is resampled at render.
        assert np.allclose(arrows[0], field[0][::M, ::N])
        assert np.allclose(arrows[1], field[1][::M, ::N])


def test_quiver_strides_resolve_the_requested_arrow_grid(tmp_path):
    # The grid is fixed at record time because the figure size plot_all normally
    # derives it from does not exist while a simulation is running.
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive',
                            fluid='quiver', quiver_shape=(4, 3))
    meta = _meta(rec)['fluid']
    assert meta['quiver_shape'] == [4, 3]
    assert meta['quiver_strides'] == [2, 2]          # a 7 x 6 grid
    assert meta['quiver_grid'] == [4, 3]
    arrows = np.load(rec.path / 'fluid' / 'quiver_00000.npy')
    assert arrows.shape == (2, 4, 3)


def test_quiver_strides_never_drop_below_one():
    # Asking for more arrows than there are grid points cannot be honoured, and
    # a stride of zero would raise deep inside a slice.
    points = (np.linspace(0, 1, 5), np.linspace(0, 1, 4))
    assert archive._quiver_strides(points, (60, 60)) == (1, 1)
    assert archive._quiver_strides(points, (1, 1)) == (5, 4)


def test_a_resident_field_still_stores_a_quiver(tmp_path):
    # Quiver is not a solver quantity, so no source can ship it and the "already
    # available" reasoning that keeps vorticity off disk does not apply.
    src = _copy_fixture(tmp_path, 'resident')
    envir = _envir(src, INUM=None)
    rec = _record_and_sweep(envir, tmp_path / 'archive', fluid='quiver')
    assert len(list((rec.path / 'fluid').glob('quiver_*.npy'))) == NDUMPS
    assert _omega_files(src) == []
    assert _omega_files(rec.path / 'fluid') == []


# --------------------------------------------------------------------------- #
#                     what fluid= accepts, and what it forces                 #
# --------------------------------------------------------------------------- #

def test_fluid_is_forced_to_none_on_a_flow_free_environment(tmp_path):
    # An analytic or flow-free run has no vorticity to record, and defaulting to
    # 'vort' there would fail on several of the examples and much of the suite.
    envir = planktos.Environment(Lx=10, Ly=10)
    swrm = planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    with envir.record(str(tmp_path / 'archive')) as rec:
        for _ in range(3):
            swrm.move(0.1, silent=True)

    assert _meta(rec)['fluid'] is None
    assert list((rec.path / 'fluid').iterdir()) == []


def test_forcing_is_silent(tmp_path):
    # Neither case is an error, and in neither is there anything else the caller
    # could have meant -- so neither warns.
    envir = planktos.Environment(Lx=10, Ly=10)
    planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        envir.record(str(tmp_path / 'archive')).stop()


@pytest.mark.parametrize('bad', ['vorticity', 'vort ', ('vort', 'speed'), 0])
def test_an_unknown_fluid_quantity_is_refused(tmp_path, bad):
    envir = _envir(VORT_FIXTURE, INUM=4)
    planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    with pytest.raises((ValueError, TypeError)):
        envir.record(str(tmp_path / 'archive'), fluid=bad)
    # A refusal must leave nothing recording, exactly as the other record()
    # refusals do.
    assert envir._recorder is None


def test_a_tuple_of_both_quantities_records_both(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive',
                            fluid=('vort', 'quiver'))
    meta = _meta(rec)['fluid']
    assert meta['quantities'] == ['vort', 'quiver']
    assert meta['vorticity'] == 'source'
    assert len(list((rec.path / 'fluid').glob('quiver_*.npy'))) == NDUMPS


def test_fluid_none_records_only_the_statistics(tmp_path):
    # The statistics are not optional: the plot box shows the component means in
    # every regime, so they are what makes an archive renderable at all.
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive', fluid=None)
    assert _meta(rec)['fluid']['quantities'] == []
    assert sorted(p.name for p in (rec.path / 'fluid').iterdir()) \
        == ['dump_stats.npz']


# --------------------------------------------------------------------------- #
#         the property the whole component exists for: no extra loads         #
# --------------------------------------------------------------------------- #

def test_recording_the_fluid_costs_no_extra_loads(tmp_path, monkeypatch):
    # The fluid counterpart of test_recording.py's headline. Deriving a dump's
    # vorticity as it lands is +0.4% on a streaming sweep; loading a dump again
    # to derive it would be the whole cost of streaming, twice. So the loader
    # call sequence with recording must be *identical* to the one without.
    def loads(record):
        src = _copy_fixture(tmp_path, 'src_rec' if record else 'src_plain')
        envir = _envir(src, INUM=4)
        calls = []
        original = type(envir.flow).load_dumpfiles

        def counted(self, d_start, d_finish):
            calls.append((d_start, d_finish))
            return original(self, d_start, d_finish)

        with monkeypatch.context() as patch:
            patch.setattr(type(envir.flow), 'load_dumpfiles', counted)
            if record:
                with envir.record(str(tmp_path / 'archive')):
                    _sweep(envir)
            else:
                _sweep(envir)
        return calls, envir

    with_rec, envir = loads(True)
    without, _ = loads(False)
    # Guard: a dataset that never streamed would pass this trivially.
    assert len(with_rec) > 0
    assert envir.flow.loaded_idx_bnds[1] == NDUMPS - 1
    assert with_rec == without


# --------------------------------------------------------------------------- #
#              reading the fluid half back through the archive                #
# --------------------------------------------------------------------------- #
# What component C consumes. Enough surface here that the fluid half is not
# write-only; the render-time refusals of section 2.8 belong with the rendering.

def test_the_reader_serves_the_per_dump_statistics(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive')
    with planktos.load_run(rec.path) as run:
        stats = run.dump_stats()
        assert np.allclose(stats['means'], envir.flow.dump_means, equal_nan=True)
        assert np.allclose(stats['flow_times'], envir.flow.flow_times)
        # The global colour scale C2 sets comes off this, and must be reduced
        # with nanmax: a run that stopped partway leaves NaN behind.
        assert np.isfinite(np.nanmax(stats['vort_absmax']))


def test_the_reader_reports_no_fluid_when_none_was_recorded(tmp_path):
    envir = planktos.Environment(Lx=10, Ly=10)
    swrm = planktos.Swarm(swarm_size=3, envir=envir, seed=1)
    with envir.record(str(tmp_path / 'archive')) as rec:
        for _ in range(3):
            swrm.move(0.1, silent=True)
    with planktos.load_run(rec.path) as run:
        assert run.dump_stats() is None
        assert run.meta['fluid'] is None


def test_the_reader_serves_stored_quiver_arrows(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive',
                            fluid=('vort', 'quiver'), quiver_shape=(4, 3))
    with planktos.load_run(rec.path) as run:
        M, N = run.meta['fluid']['quiver_strides']
        arrows = run.quiver(3)
        field = envir.flow(float(envir.flow.flow_times[3]))
        assert np.allclose(arrows[0], field[0][::M, ::N])


def test_asking_for_a_quiver_that_was_not_recorded_names_the_cause(tmp_path):
    # Section 2.8: a quantity that was not recorded is a hard refusal naming what
    # is absent, not a silent fall back to re-streaming the fluid.
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive')
    with planktos.load_run(rec.path) as run:
        with pytest.raises(ValueError, match='no quiver data'):
            run.quiver(0)


def test_asking_for_a_quiver_dump_that_never_loaded_is_refused(tmp_path):
    envir = _envir(VORT_FIXTURE, INUM=4)
    rec = _record_and_sweep(envir, tmp_path / 'archive', steps=2,
                            fluid='quiver')
    with planktos.load_run(rec.path) as run:
        run.quiver(0)                       # in the opening window
        with pytest.raises(ValueError, match='no quiver for fluid dump 7'):
            run.quiver(7)


# --------------------------------------------------------------------------- #
#          a written vorticity file appears complete or not at all            #
# --------------------------------------------------------------------------- #
# The per-source vtk writers name their own files, so they cannot be handed a
# temporary name the way _atomic_write does it. They get a temporary directory
# instead. Without that, a kill partway through a write leaves a truncated vtk,
# which raises on read -- worse than a missing one, since the missing case is
# ordinary under dynamic loading and already handled.

# The dump both failed-write tests target, and the file such a write leaves.
TRUNCATED_IDX = 6
TRUNCATED = 'Omega.{:04d}.vtk'.format(TRUNCATED_IDX)


def _truncating_writer(name):
    """A write_dump_vorticity that writes part of a file and then fails, which
    is what a kill mid-write looks like from the outside."""

    def die_partway(self, t_idx, vort, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / name).write_bytes(b'# vtk truncated')
        raise OSError('disk full')

    return die_partway


def test_a_failed_vorticity_write_leaves_no_truncated_file(tmp_path, monkeypatch):
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    with envir.record(str(tmp_path / 'archive')) as rec:
        monkeypatch.setattr(type(envir.flow), 'write_dump_vorticity',
                            _truncating_writer(TRUNCATED))
        with pytest.raises(OSError):
            rec._fluid._write_vorticity(6, np.zeros(envir.flow.fshape[1:]))

        # The truncated file is not where a reader looks for the series...
        assert not (src / TRUNCATED).exists()
        assert _omega_files(src) == ['Omega.{:04d}.vtk'.format(k)
                                    for k in range(5)]
        # ...it is in the staging directory, and the probe's own glob does not
        # recurse into it, so a truncated file can never enter the series.
        assert (src / archive.TMP_DIRNAME / TRUNCATED).is_file()
        assert TRUNCATED not in [f.name for f
                                        in src.glob('Omega.*.vtk')]


def test_the_staging_directory_is_cleared_when_recording_stops(tmp_path):
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    with envir.record(str(tmp_path / 'archive')):
        _sweep(envir)
        assert (src / archive.TMP_DIRNAME).is_dir()      # in use during the run
    assert not (src / archive.TMP_DIRNAME).exists()
    # Nothing but the series itself is left in the source directory.
    assert [p.name for p in src.iterdir() if p.is_dir()] == []
    assert len(_omega_files(src)) == NDUMPS


def test_staging_is_cleared_even_after_a_failed_write(tmp_path, monkeypatch):
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    rec = envir.record(str(tmp_path / 'archive'))
    monkeypatch.setattr(type(envir.flow), 'write_dump_vorticity',
                        _truncating_writer(TRUNCATED))
    with pytest.raises(OSError):
        rec._fluid._write_vorticity(6, np.zeros(envir.flow.fshape[1:]))
    monkeypatch.undo()
    envir.stop_recording()
    assert not (src / archive.TMP_DIRNAME).exists()


def test_a_written_series_is_complete_and_readable(tmp_path):
    # The property the staging exists to protect: every file a run wrote is a
    # whole file, and the reader that must parse them all succeeds on every one.
    src = _copy_fixture(tmp_path, 'novort')
    envir = _envir(src, INUM=4)
    _record_and_sweep(envir, tmp_path / 'archive')

    assert envir.flow.probe_stored_vorticity()[0] == 'complete'
    for t_idx in range(NDUMPS):
        field = envir.flow.read_dump_vorticity(t_idx)
        assert field.shape == envir.flow.fshape[1:]
        assert np.isfinite(field).all()
