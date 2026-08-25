'''Tests for the run-archive writer and the on-disk format
(planktos/archive.py, run_persistence.md section 2, build step A2).

Everything here drives the writer directly with synthetic captures -- plain
arrays, no Environment, no Swarm, no fluid -- and reads the results back with
raw np.load and json.load rather than through a reader of our own. That is
deliberate: a round-trip through our own code can be self-consistently wrong,
where reading the bytes pins the format itself. The reader gets its own tests
when it exists (A4).

The three properties under test, in order of how much they matter:

  * the archive is valid with nothing having run at the end;
  * no partially-written file is ever visible;
  * a chunk index means the same thing for every swarm, including one that
    joined halfway through.
'''

import json
import warnings
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

from planktos import archive


# --------------------------------------------------------------------------- #
#                                  helpers                                     #
# --------------------------------------------------------------------------- #

def _fingerprint(dimension=2, L=(10.0, 5.0), n_dumps=4, npts=(5, 3)):
    flow_points = tuple(np.linspace(0, L[d], npts[d]) for d in range(dimension))
    return archive.build_fingerprint(dimension, L, flow_points=flow_points,
                                     flow_times=np.linspace(0, 1, n_dumps),
                                     periodic_dim=False)


def _writer(tmp_path, chunk_size=100, meta=None, name='run', **kwargs):
    return archive._ArchiveWriter(tmp_path / name, _fingerprint(), meta=meta,
                                  chunk_size=chunk_size, **kwargs)


def _capture(N=3, D=2, value=0.0, masked=()):
    '''One swarm's named arrays for a capture, filled with recognizable values.

    Velocities are the negated positions so a mix-up between the two files is
    visible rather than plausible.
    '''
    pos = ma.masked_array(np.full((N, D), value, dtype=float),
                          mask=np.zeros((N, D), dtype=bool))
    vel = ma.masked_array(np.full((N, D), -value, dtype=float),
                          mask=np.zeros((N, D), dtype=bool))
    for row in masked:
        pos[row, :] = ma.masked
        vel[row, :] = ma.masked
    return {'positions': pos, 'velocities': vel}


def _run(writer, n_captures, swarm=0, N=3, D=2, start=0):
    '''Feed a contiguous run of captures; returns the values used.'''
    for j in range(start, start + n_captures):
        writer.add_capture(j, 0.1 * j, {swarm: _capture(N, D, value=float(j))})
    return [float(j) for j in range(start, start + n_captures)]


def _read(path):
    return np.load(path, allow_pickle=False)


def _chunk_files(agent_dir, prefix):
    return sorted(agent_dir.glob(prefix + '_*.npy'),
                  key=lambda p: archive._chunk_index_of(p))


# --------------------------------------------------------------------------- #
#                                fingerprint                                   #
# --------------------------------------------------------------------------- #

def test_the_fingerprint_round_trips_through_grid_npz(tmp_path):
    w = _writer(tmp_path)
    stored = dict(np.load(w.path / 'grid.npz', allow_pickle=False))
    assert archive.compare_fingerprints(stored, _fingerprint()) == []


def test_the_fingerprint_holds_the_grid_the_timeline_and_periodicity():
    fp = _fingerprint()
    assert set(fp) == {'dimension', 'L', 'periodic_dim',
                       'flow_points_0', 'flow_points_1', 'flow_times'}


def test_a_flow_free_environment_fingerprints_on_dimension_and_domain_alone():
    fp = archive.build_fingerprint(2, (10.0, 5.0))
    assert set(fp) == {'dimension', 'L', 'periodic_dim'}
    assert archive.fingerprint_summary(fp)['grid_shape'] is None
    assert archive.fingerprint_summary(fp)['n_dumps'] == 0


def test_periodic_dim_is_broadcast_to_one_flag_per_dimension():
    assert archive.build_fingerprint(3, (1, 1, 1))['periodic_dim'].tolist() == \
        [False, False, False]
    assert archive.build_fingerprint(2, (1, 1), periodic_dim=(True, False)
                                    )['periodic_dim'].tolist() == [True, False]


def test_a_mismatched_domain_size_is_refused_at_construction():
    with pytest.raises(ValueError, match='L has'):
        archive.build_fingerprint(3, (1.0, 2.0))


def test_a_mismatched_axis_count_is_refused_at_construction():
    with pytest.raises(ValueError, match='flow_points has'):
        archive.build_fingerprint(3, (1., 1., 1.), flow_points=(np.zeros(3),))


def test_every_field_of_the_fingerprint_is_load_bearing():
    # Change any one of the five and the archive must stop matching, or that
    # field is not earning its place in the fingerprint.
    base = _fingerprint()
    for changed in (_fingerprint(L=(10.0, 6.0)),
                    _fingerprint(n_dumps=5),
                    _fingerprint(npts=(6, 3)),
                    archive.build_fingerprint(
                        2, (10.0, 5.0),
                        flow_points=(np.linspace(0, 10, 5), np.linspace(0, 5, 3)),
                        flow_times=np.linspace(0, 1, 4), periodic_dim=True)):
        assert archive.compare_fingerprints(base, changed) != []
    assert archive.compare_fingerprints(base, _fingerprint()) == []


def test_a_corrupted_grid_file_is_caught_by_the_container(tmp_path):
    # grid.npz is a zip, and numpy verifies a CRC32 per member on read. This is
    # why the archive carries no checksum of its own: it would duplicate an
    # integrity check the format already performs.
    w = _writer(tmp_path)
    raw = bytearray((w.path / 'grid.npz').read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (w.path / 'grid.npz').write_bytes(bytes(raw))
    with pytest.raises(Exception):
        dict(np.load(w.path / 'grid.npz', allow_pickle=False))


def test_comparing_fingerprints_names_what_differs_not_just_that_it_does():
    # A refusal has to be actionable. A digest could only ever say a != b.
    problems = archive.compare_fingerprints(_fingerprint(n_dumps=6),
                                            _fingerprint(n_dumps=9))
    assert len(problems) == 1
    assert 'flow_times' in problems[0]
    assert '6 values' in problems[0] and '9 values' in problems[0]


def test_comparing_fingerprints_reports_a_field_present_on_only_one_side():
    problems = archive.compare_fingerprints(archive.build_fingerprint(2, (10., 5.)),
                                            _fingerprint())
    assert any('absent from the archive' in p for p in problems)


def test_comparing_fingerprints_catches_a_shifted_grid_of_the_same_shape():
    shifted = archive.build_fingerprint(
        2, (10.0, 5.0),
        flow_points=(np.linspace(0, 10, 5) + 1.0, np.linspace(0, 5, 3)),
        flow_times=np.linspace(0, 1, 4))
    problems = archive.compare_fingerprints(_fingerprint(), shifted)
    assert len(problems) == 1 and 'flow_points_0' in problems[0]


# --------------------------------------------------------------------------- #
#                                 metadata                                     #
# --------------------------------------------------------------------------- #

def test_metadata_is_written_at_construction_before_any_capture(tmp_path):
    w = _writer(tmp_path)
    meta = json.loads((w.path / 'meta.json').read_text())
    assert meta['version'] == archive.FORMAT_VERSION
    assert meta['dtype'] == 'float64'
    assert meta['chunk_size'] == 100
    assert meta['grid']['dimension'] == 2
    assert meta['grid']['grid_shape'] == [5, 3]
    assert meta['grid']['n_dumps'] == 4


def test_metadata_is_never_rewritten_during_the_run(tmp_path):
    # The rule that makes the single write survivable: anything that grows
    # lives in a file that grows. Nothing here may touch meta.json again.
    w = _writer(tmp_path, chunk_size=2)
    w.add_swarm(0, 'organism', 3, 2, 0)
    before = (w.path / 'meta.json').read_bytes()
    _run(w, 7)
    w.add_swarm(1, 'organism', 4, 2, 7)
    w.add_capture(7, 0.7, {0: _capture(3, 2), 1: _capture(4, 2)})
    w.flush()
    assert (w.path / 'meta.json').read_bytes() == before


def test_caller_metadata_is_carried_through(tmp_path):
    w = _writer(tmp_path, meta={'provenance': {'loader': 'read_IB2d_fluid_data'},
                                'fluid': 'vort', 'quiver_shape': [60, 60]})
    meta = json.loads((w.path / 'meta.json').read_text())
    assert meta['provenance']['loader'] == 'read_IB2d_fluid_data'
    assert meta['fluid'] == 'vort'


def test_metadata_that_is_not_strict_json_is_refused_at_the_start(tmp_path):
    # Better to fail while starting a twelve-hour run than while finishing one.
    with pytest.raises(ValueError):
        _writer(tmp_path, meta={'bad': float('nan')})


def test_the_swarm_roster_lives_in_per_swarm_sidecars(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'krill', 5, 3, 0)
    entry = json.loads((w.path / 'agents' / 'swarm00.json').read_text())
    assert entry == {'index': 0, 'name': 'krill', 'N': 5, 'D': 3,
                     'first_capture': 0}
    assert 'swarms' not in json.loads((w.path / 'meta.json').read_text())


def test_two_swarms_may_share_a_name_because_files_are_keyed_by_index(tmp_path):
    # 'organism' is the default name for every Swarm, so a filename built from
    # the name would silently overwrite.
    w = _writer(tmp_path)
    w.add_swarm(0, 'organism', 3, 2, 0)
    w.add_swarm(1, 'organism', 3, 2, 0)
    names = {json.loads(p.read_text())['name']
             for p in (w.path / 'agents').glob('swarm*.json')}
    assert names == {'organism'}
    assert len(list((w.path / 'agents').glob('swarm*.json'))) == 2


def test_reusing_a_swarm_index_is_refused(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'a', 3, 2, 0)
    with pytest.raises(ValueError, match='already recorded'):
        w.add_swarm(0, 'b', 3, 2, 0)


# --------------------------------------------------------------------------- #
#                                 directory                                    #
# --------------------------------------------------------------------------- #

def test_a_missing_directory_is_created_with_its_parents(tmp_path):
    target = tmp_path / 'deep' / 'deeper' / 'run'
    resolved = archive._resolve_archive_path(target)
    assert resolved == target and resolved.is_dir()


def test_an_existing_empty_directory_is_used_as_is(tmp_path):
    target = tmp_path / 'run'
    target.mkdir()
    assert archive._resolve_archive_path(target) == target


def test_a_non_empty_directory_redirects_to_a_timestamped_sibling(tmp_path):
    target = tmp_path / 'run'
    target.mkdir()
    (target / 'meta.json').write_text('{}')
    with pytest.warns(UserWarning, match='already holds data'):
        resolved = archive._resolve_archive_path(target)
    assert resolved != target
    assert resolved.name.startswith('run_') and resolved.is_dir()
    # the previous run is untouched
    assert (target / 'meta.json').read_text() == '{}'


def test_the_writer_exposes_the_directory_it_actually_chose(tmp_path):
    target = tmp_path / 'run'
    target.mkdir()
    (target / 'stale.npy').write_bytes(b'x')
    with pytest.warns(UserWarning):
        w = archive._ArchiveWriter(target, _fingerprint())
    # Without this, a later load_run('run/') would quietly read the earlier run.
    assert w.path != target
    assert (w.path / 'meta.json').is_file()
    assert not (target / 'meta.json').exists()


def test_two_redirects_in_the_same_second_do_not_collide(tmp_path):
    target = tmp_path / 'run'
    target.mkdir()
    (target / 'x').write_bytes(b'x')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = archive._resolve_archive_path(target)
        (first / 'x').write_bytes(b'x')
        second = archive._resolve_archive_path(target)
    assert first != second


def test_a_chunk_size_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match='chunk_size'):
        _writer(tmp_path, chunk_size=0)


# --------------------------------------------------------------------------- #
#                             chunk boundaries                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('n_captures, chunk_size, want_chunks, want_rows', [
    (1, 1, 1, [1]),                       # exactly one chunk, one row
    (2, 1, 2, [1, 1]),                    # one chunk plus one
    (4, 4, 1, [4]),                       # exactly one full chunk
    (5, 4, 2, [4, 1]),                    # one full chunk plus one row
    (3, 4, 1, [3]),                       # a partial chunk only
    (9, 4, 3, [4, 4, 1]),
])
def test_chunk_boundaries(tmp_path, n_captures, chunk_size, want_chunks, want_rows):
    w = _writer(tmp_path, chunk_size=chunk_size)
    w.add_swarm(0, 'organism', 3, 2, 0)
    values = _run(w, n_captures)
    w.close()

    agents = w.path / 'agents'
    chunks = _chunk_files(agents, 'swarm00_pos')
    assert len(chunks) == want_chunks
    assert [len(_read(p)) for p in chunks] == want_rows
    assert [len(_read(p)) for p in _chunk_files(agents, 'times')] == want_rows

    # and the captures come back in order, unbroken across the boundaries
    joined = np.concatenate([_read(p) for p in chunks])
    assert joined[:, 0, 0].tolist() == values


def test_the_times_series_is_the_authority_for_the_capture_time_base(tmp_path):
    w = _writer(tmp_path, chunk_size=3)
    w.add_swarm(0, 'organism', 2, 2, 0)
    _run(w, 7, N=2)
    w.close()
    times = np.concatenate([_read(p)
                            for p in _chunk_files(w.path / 'agents', 'times')])
    assert times == pytest.approx([0.1 * j for j in range(7)])
    # nothing summarizes it anywhere else
    meta = json.loads((w.path / 'meta.json').read_text())
    assert 'times' not in meta and 'n_captures' not in meta


def test_positions_and_velocities_round_trip(tmp_path):
    w = _writer(tmp_path, chunk_size=4)
    w.add_swarm(0, 'organism', 3, 2, 0)
    _run(w, 6)
    w.close()
    agents = w.path / 'agents'
    pos = np.concatenate([_read(p) for p in _chunk_files(agents, 'swarm00_pos')])
    vel = np.concatenate([_read(p) for p in _chunk_files(agents, 'swarm00_vel')])
    assert pos.shape == (6, 3, 2) and vel.shape == (6, 3, 2)
    assert np.array_equal(vel, -pos)


def test_a_masked_agent_round_trips_as_masked(tmp_path):
    w = _writer(tmp_path, chunk_size=8)
    w.add_swarm(0, 'organism', 4, 2, 0)
    for j in range(3):
        # agent 2 leaves the domain at capture 1 and stays gone
        gone = (2,) if j >= 1 else ()
        w.add_capture(j, float(j), {0: _capture(4, 2, float(j), masked=gone)})
    w.close()
    mask = _read(w.path / 'agents' / 'swarm00_mask_0000.npy')
    assert mask.dtype == np.bool_ and mask.shape == (3, 4)
    assert mask[0].tolist() == [False] * 4
    assert mask[1].tolist() == [False, False, True, False]
    assert mask[2].tolist() == [False, False, True, False]


def test_a_partially_masked_row_is_refused_rather_than_flattened(tmp_path):
    # A masked row means the agent left the domain -- agents leave whole rows.
    # A half-masked row means an invariant broke upstream, and storing a
    # row-reduced mask would hide that.
    w = _writer(tmp_path)
    w.add_swarm(0, 'organism', 2, 2, 0)
    named = _capture(2, 2)
    named['positions'][1, 0] = ma.masked
    with pytest.raises(ValueError, match='partially masked'):
        w.add_capture(0, 0.0, {0: named})


def test_a_wrongly_shaped_capture_is_refused(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'organism', 3, 2, 0)
    with pytest.raises(ValueError, match='expected'):
        w.add_capture(0, 0.0, {0: _capture(2, 2)})


# --------------------------------------------------------------------------- #
#                          contiguity and the swarm set                        #
# --------------------------------------------------------------------------- #

def test_a_gap_in_the_capture_series_is_refused(tmp_path):
    # A gap here would become a gap on disk, which the reader could only read
    # as a lost file.
    w = _writer(tmp_path)
    w.add_swarm(0, 'organism', 2, 2, 0)
    w.add_capture(0, 0.0, {0: _capture(2, 2)})
    with pytest.raises(ValueError, match='contiguous'):
        w.add_capture(2, 0.2, {0: _capture(2, 2)})


def test_a_capture_missing_an_active_swarm_is_refused(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'a', 2, 2, 0)
    w.add_swarm(1, 'b', 2, 2, 0)
    with pytest.raises(ValueError, match='expected'):
        w.add_capture(0, 0.0, {0: _capture(2, 2)})


def test_a_capture_naming_a_swarm_that_has_not_started_is_refused(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'a', 2, 2, 0)
    w.add_swarm(1, 'b', 2, 2, 5)
    with pytest.raises(ValueError, match='expected'):
        w.add_capture(0, 0.0, {0: _capture(2, 2), 1: _capture(2, 2)})


def test_capturing_after_close_is_refused(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'a', 2, 2, 0)
    w.close()
    with pytest.raises(RuntimeError, match='closed'):
        w.add_capture(0, 0.0, {0: _capture(2, 2)})


# --------------------------------------------------------------------------- #
#                           a swarm added mid-run                              #
# --------------------------------------------------------------------------- #

def test_a_late_swarm_gets_a_short_first_chunk_aligned_on_the_global_index(tmp_path):
    # Chunks are keyed on the global capture index, not on each swarm's own row
    # count, so a late swarm needs no second indexing scheme: its first chunk is
    # simply short at the front, and first_capture resolves the offset.
    w = _writer(tmp_path, chunk_size=4)
    w.add_swarm(0, 'early', 2, 2, 0)
    _run(w, 6, N=2)                              # captures 0-5
    w.add_swarm(1, 'late', 3, 2, 6)
    for j in range(6, 12):
        w.add_capture(j, 0.1 * j, {0: _capture(2, 2, float(j)),
                                   1: _capture(3, 2, float(j))})
    w.close()

    agents = w.path / 'agents'
    # chunk 1 covers captures 4-7; the late swarm contributes only 6 and 7
    assert len(_read(agents / 'swarm00_pos_0001.npy')) == 4
    assert len(_read(agents / 'swarm01_pos_0001.npy')) == 2
    assert len(_read(agents / 'times_0001.npy')) == 4
    # chunk 2 covers 8-11 and both swarms are present for all of it
    assert len(_read(agents / 'swarm00_pos_0002.npy')) == 4
    assert len(_read(agents / 'swarm01_pos_0002.npy')) == 4
    # its sidecar carries the offset
    assert json.loads((agents / 'swarm01.json').read_text())['first_capture'] == 6


def test_a_late_swarm_writes_no_file_for_a_chunk_it_missed_entirely(tmp_path):
    # A zero-row file would be a lie about first_capture; absence is the honest
    # record, and the reader resolves the offset from the sidecar.
    w = _writer(tmp_path, chunk_size=4)
    w.add_swarm(0, 'early', 2, 2, 0)
    _run(w, 4, N=2)                              # chunk 0: captures 0-3
    w.add_swarm(1, 'late', 2, 2, 4)
    for j in range(4, 8):
        w.add_capture(j, 0.1 * j, {0: _capture(2, 2), 1: _capture(2, 2)})
    w.close()
    agents = w.path / 'agents'
    assert not (agents / 'swarm01_pos_0000.npy').exists()
    assert (agents / 'swarm01_pos_0001.npy').is_file()


# --------------------------------------------------------------------------- #
#                        crash validity and atomicity                          #
# --------------------------------------------------------------------------- #

def test_an_archive_is_readable_with_no_finalizer_having_run(tmp_path):
    # A hard kill defeats __exit__, close(), atexit and __del__ alike. The most
    # any of them can save is one unflushed chunk; everything already written
    # must stand on its own. Nothing below calls flush() or close().
    w = _writer(tmp_path, chunk_size=3)
    w.add_swarm(0, 'organism', 2, 2, 0)
    _run(w, 8, N=2)
    del w                                        # no close, no flush

    path = next(tmp_path.glob('run*'))
    meta = json.loads((path / 'meta.json').read_text())
    assert meta['version'] == archive.FORMAT_VERSION
    stored = dict(np.load(path / 'grid.npz', allow_pickle=False))
    assert archive.compare_fingerprints(stored, _fingerprint()) == []

    agents = path / 'agents'
    chunks = _chunk_files(agents, 'swarm00_pos')
    # 8 captures at chunk_size 3: chunks 0 and 1 are complete, chunk 2 held the
    # two captures that were still buffered and is simply not there.
    assert [archive._chunk_index_of(p) for p in chunks] == [0, 1]
    assert [len(_read(p)) for p in chunks] == [3, 3]
    joined = np.concatenate([_read(p) for p in chunks])
    assert joined[:, 0, 0].tolist() == [0., 1., 2., 3., 4., 5.]


def test_no_partially_written_file_is_ever_visible(tmp_path, monkeypatch):
    # Without temp-then-replace, a kill during np.save leaves a truncated .npy
    # that raises on read -- so one unlucky moment would cost the whole archive
    # rather than one buffer.
    w = _writer(tmp_path, chunk_size=2)
    w.add_swarm(0, 'organism', 2, 2, 0)
    # A chunk is written when the *next* one starts, so captures 0-2 are what
    # it takes to land chunk 0. Capture 4 is then what triggers chunk 1.
    _run(w, 3, N=2)

    real_save = np.save

    def die_partway(fobj, arr, **kwargs):
        fobj.write(b'\x93NUMPY truncated garbage')
        raise OSError('disk full')

    monkeypatch.setattr(np, 'save', die_partway)
    with pytest.raises(OSError):
        _run(w, 2, N=2, start=3)                  # chunk 1 dies mid-write
    monkeypatch.setattr(np, 'save', real_save)

    agents = w.path / 'agents'
    assert (agents / 'times_0000.npy').is_file()
    assert not (agents / 'times_0001.npy').exists(), 'a truncated chunk appeared'
    assert list(agents.glob('*' + archive.TMP_SUFFIX)) == [], \
        'a temporary file was left behind'
    # and what did land is still readable
    assert len(_read(agents / 'swarm00_pos_0000.npy')) == 2


def test_a_failed_write_leaves_no_temporary_file(tmp_path):
    target = tmp_path / 'thing.npy'
    with pytest.raises(ValueError):
        archive._atomic_write(target, lambda f: (_ for _ in ()).throw(ValueError()))
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_flush_writes_a_partial_chunk_and_keeps_recording(tmp_path):
    # A mid-run plot needs a flush that does not end the recording, or the next
    # record() would refuse the now-non-empty directory.
    w = _writer(tmp_path, chunk_size=5)
    w.add_swarm(0, 'organism', 2, 2, 0)
    _run(w, 3, N=2)
    w.flush()
    partial = _read(w.path / 'agents' / 'swarm00_pos_0000.npy')
    assert len(partial) == 3

    _run(w, 2, N=2, start=3)                     # completes the chunk
    w.close()
    full = _read(w.path / 'agents' / 'swarm00_pos_0000.npy')
    assert len(full) == 5
    assert full[:, 0, 0].tolist() == [0., 1., 2., 3., 4.]


def test_flush_is_idempotent(tmp_path):
    w = _writer(tmp_path, chunk_size=5)
    w.add_swarm(0, 'organism', 2, 2, 0)
    _run(w, 3, N=2)
    w.flush()
    first = (w.path / 'agents' / 'swarm00_pos_0000.npy').read_bytes()
    w.flush()
    w.flush()
    assert (w.path / 'agents' / 'swarm00_pos_0000.npy').read_bytes() == first


def test_close_is_idempotent(tmp_path):
    w = _writer(tmp_path, chunk_size=5)
    w.add_swarm(0, 'organism', 2, 2, 0)
    _run(w, 2, N=2)
    w.close()
    w.close()
    assert len(_read(w.path / 'agents' / 'swarm00_pos_0000.npy')) == 2


def test_closing_a_recording_that_captured_nothing_leaves_a_valid_archive(tmp_path):
    w = _writer(tmp_path)
    w.add_swarm(0, 'organism', 2, 2, 0)
    w.close()
    assert json.loads((w.path / 'meta.json').read_text())['version'] == \
        archive.FORMAT_VERSION
    assert list((w.path / 'agents').glob('*.npy')) == []


# --------------------------------------------------------------------------- #
#                           chunk index recovery                               #
# --------------------------------------------------------------------------- #

def test_a_chunk_index_survives_the_round_trip_through_its_filename():
    for index in (0, 7, 42, 9999, 10_000, 123_456):
        assert archive._chunk_index_of(archive._chunk_name('swarm00_pos', index)) \
            == index


def test_a_non_chunk_filename_yields_no_index():
    assert archive._chunk_index_of('swarm00.json') is None
    assert archive._chunk_index_of('meta.json') is None


def test_numeric_ordering_is_not_lexical_ordering_past_four_digits():
    # This is why chunk discovery parses an integer instead of sorting names:
    # %04d grows a fifth digit at chunk 10000 and lexical order breaks, which
    # would silently assemble a run out of order. The same mistake has already
    # been paid for once on this branch, in the OpenFOAM dump directories.
    names = [archive._chunk_name('times', i) for i in (9999, 10_000, 10_001)]
    assert sorted(names) != names, 'the hazard this guards against is gone'
    assert sorted(names, key=archive._chunk_index_of) == names
