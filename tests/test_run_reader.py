'''Tests for reading a run archive back
(planktos.archive.RunArchive / load_run, run_persistence.md section 2.7-2.8,
build step A4).

test_run_archive.py pins the bytes on disk and test_recording.py pins what a run
writes; this pins what comes back out. The two properties that matter most:

  * answering a question about one capture must not read the whole archive --
    that is what lets a run larger than memory stay usable, and half the reason
    the format is chunked;
  * an archive that does not describe this environment is refused outright,
    because reading positions against the wrong grid gives a plausible picture
    that is silently false.
'''

import json

import numpy as np
import numpy.ma as ma
import pytest

import planktos
from planktos import archive


# --------------------------------------------------------------------------- #
#                                  helpers                                     #
# --------------------------------------------------------------------------- #

def _envir(L=10.0):
    return planktos.Environment(Lx=L, Ly=L,
                                flow=[np.zeros((3, 3)), np.zeros((3, 3))])


def _swarm(envir, n=4, mu=(1.0, 0.0), seed=1, init=2.0):
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=seed,
                          init=np.full((n, 2), init))
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array(mu, float)
    return swrm


def _recorded(tmp_path, steps=7, chunk_size=3, **kwargs):
    '''A finished archive plus the environment and swarm that made it.'''
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', chunk_size=chunk_size, **kwargs) as rec:
        for _ in range(steps):
            swrm.move(0.5, silent=True)
    return archive.load_run(rec.path), envir, swrm


def _stack(history):
    return np.stack([np.ma.getdata(entry) for entry in history])


# --------------------------------------------------------------------------- #
#                        what comes back is what went in                       #
# --------------------------------------------------------------------------- #

def test_the_archive_reads_back_as_the_run_that_wrote_it(tmp_path):
    run, envir, swrm = _recorded(tmp_path)
    assert np.allclose(run.times, envir.time_history + [envir.time])
    assert np.allclose(run.positions(0).asarray(), _stack(swrm.full_pos_history))
    assert np.allclose(run.velocities(0).asarray(), _stack(swrm.full_vel_history))
    assert run.swarms == [('organism', 0)]
    assert run.store == ('positions', 'velocities')


def test_load_run_is_the_public_entry_point(tmp_path):
    run, _, _ = _recorded(tmp_path)
    reopened = archive.load_run(run.path)
    assert isinstance(reopened, archive.RunArchive)
    assert np.allclose(reopened.times, run.times)


def test_the_metadata_and_fingerprint_come_back(tmp_path):
    run, envir, _ = _recorded(tmp_path)
    assert run.meta['version'] == archive.FORMAT_VERSION
    assert run.meta['provenance']['planktos_version'] == planktos.__version__
    assert archive.compare_fingerprints(run.grid,
                                        archive.fingerprint_of(envir)) == []


def test_a_masked_agent_reads_back_masked(tmp_path):
    envir = _envir(L=1.0)
    swrm = _swarm(envir, n=3, mu=(2.0, 0.0), init=0.5)
    swrm.positions[0] = [0.95, 0.5]
    with envir.record(tmp_path / 'run') as rec:
        for _ in range(2):
            swrm.move(0.1, silent=True)
    run = archive.load_run(rec.path)
    pos = run.positions(0)
    assert not ma.getmaskarray(pos[0]).any(), 'nobody had left at t0'
    assert ma.getmaskarray(pos[-1])[0].all(), 'the agent that left is not masked'
    assert not ma.getmaskarray(pos[-1])[1:].any()


# --------------------------------------------------------------------------- #
#                        resolving by time, not by index                       #
# --------------------------------------------------------------------------- #

def test_capture_at_snaps_to_the_nearest_capture(tmp_path):
    run, _, _ = _recorded(tmp_path, steps=6)      # t = 0, 0.5, ... 3.0
    assert run.capture_at(0.0) == 0
    assert run.capture_at(1.4) == 3               # 1.5 is nearer than 1.0
    assert run.capture_at(1.6) == 3
    assert run.capture_at(99.0) == len(run.times) - 1, 'should clamp to the end'
    assert run.capture_at(-99.0) == 0


def test_capture_at_never_interpolates(tmp_path):
    # Blending positions across a domain wrap or a boundary slide would invent
    # a trajectory that never happened, so state is snapped and the API returns
    # an index rather than a value.
    run, _, _ = _recorded(tmp_path)
    j = run.capture_at(1.2)
    assert isinstance(j, int)
    assert run.times[j] in run.times


def test_a_swarm_added_mid_run_is_front_padded_and_aligned_to_the_time_base(tmp_path):
    envir = _envir()
    first = _swarm(envir, seed=1)
    with envir.record(tmp_path / 'run', chunk_size=3) as rec:
        for _ in range(4):
            first.move(0.5, silent=True)
        late = _swarm(envir, n=2, seed=9, init=3.0)
        for _ in range(3):
            envir.move_swarms(0.5, silent=True)

    run = archive.load_run(rec.path)
    late_series = run.positions(1)
    # every series is the same length and lines up with times index for index
    assert late_series.shape == (len(run.times), 2, 2)
    assert run.positions(0).shape[0] == len(run.times)

    first_capture = json.loads(
        (run.path / 'agents' / 'swarm01.json').read_text())['first_capture']
    assert first_capture == 5
    for j in range(first_capture):
        assert ma.getmaskarray(late_series[j]).all(), \
            'capture {} is before the swarm joined and should be masked'.format(j)
    for j in range(first_capture, len(run.times)):
        assert not ma.getmaskarray(late_series[j]).any()
    # and the real rows are the tail of the swarm's own history, in order.
    # Its state at creation is NOT among them: it was built after the capture
    # taken at that moment, so its first captured state is the one after its
    # first move. first_capture says exactly that, which is why a consumer
    # resolves by time rather than assuming index j means history entry j.
    n_real = len(run.times) - first_capture
    assert np.allclose(late_series[first_capture:].data,
                       _stack(late.full_pos_history)[-n_real:])


# --------------------------------------------------------------------------- #
#                     addressing swarms: index, and names                      #
# --------------------------------------------------------------------------- #

def test_two_swarms_sharing_the_default_name_are_reachable_by_index(tmp_path):
    envir = _envir()
    _swarm(envir, seed=1)
    _swarm(envir, seed=2)
    with envir.record(tmp_path / 'run') as rec:
        envir.move_swarms(0.5, silent=True)
    run = archive.load_run(rec.path)

    assert run.swarms == [('organism', 0), ('organism', 1)]
    assert run.positions(0).shape[0] == run.positions(1).shape[0]
    # 'organism' is the default name for every Swarm, so it does not identify
    # one -- refused rather than silently picking the first
    with pytest.raises(KeyError, match='names 2 swarms'):
        run.positions('organism')


def test_a_unique_name_addresses_its_swarm(tmp_path):
    envir = _envir()
    krill = _swarm(envir, seed=1)
    krill.shared_props['name'] = 'krill'
    with envir.record(tmp_path / 'run') as rec:
        krill.move(0.5, silent=True)
    run = archive.load_run(rec.path)
    assert run.swarms == [('krill', 0)]
    assert np.allclose(run.positions('krill').asarray(), run.positions(0).asarray())


def test_an_unknown_swarm_is_refused(tmp_path):
    run, _, _ = _recorded(tmp_path)
    with pytest.raises(KeyError, match='no swarm 5'):
        run.positions(5)
    with pytest.raises(KeyError, match="no swarm named"):
        run.positions('nobody')


# --------------------------------------------------------------------------- #
#            reading one capture must not read the whole archive               #
# --------------------------------------------------------------------------- #

def test_one_capture_reads_one_chunk(tmp_path, monkeypatch):
    # The property that makes an archive larger than memory usable. 20 captures
    # in chunks of 2 is ten chunks per array; asking for one capture must touch
    # exactly the chunk it lives in.
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', chunk_size=2) as rec:
        for _ in range(19):
            swrm.move(0.5, silent=True)
    run = archive.load_run(rec.path)

    opened = []
    real_load = np.load

    def watched(path, *args, **kwargs):
        opened.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', watched)
    capture = run.positions(0)[13]
    monkeypatch.setattr(np, 'load', real_load)

    assert capture.shape == (4, 2)
    # one positions chunk and its mask, and nothing else
    assert len(opened) == 2, 'read {} files for one capture: {}'.format(
        len(opened), opened)
    assert all('_0006.npy' in f for f in opened), opened


def test_chunks_are_memory_mapped_not_read(tmp_path):
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    run.positions(0)[4]
    assert any(isinstance(v, np.memmap) for v in run._cache.values()), \
        'chunks should be memory-mapped, not read into memory'


def test_the_open_chunk_cache_stays_bounded(tmp_path):
    # A memmap holds a file descriptor, so caching every chunk of a long run
    # would exhaust them.
    envir = _envir()
    swrm = _swarm(envir)
    with envir.record(tmp_path / 'run', chunk_size=1) as rec:
        for _ in range(40):
            swrm.move(0.1, silent=True)
    run = archive.load_run(rec.path)
    series = run.positions(0)
    for j in range(len(run.times)):
        series[j]
    assert len(run._cache) <= run.CACHE_SIZE


def test_a_capture_series_is_not_an_ndarray(tmp_path):
    # It hands back real arrays and never claims to be one. FlowArray is why:
    # something that pretends to be an array it is not gets np.asarray()d into
    # the wrong buffer, silently.
    run, _, _ = _recorded(tmp_path)
    series = run.positions(0)
    assert not isinstance(series, np.ndarray)
    assert isinstance(series[0], ma.MaskedArray)
    assert isinstance(series.asarray(), ma.MaskedArray)


def test_capture_series_indexing(tmp_path):
    run, _, swrm = _recorded(tmp_path, steps=7, chunk_size=3)
    series = run.positions(0)
    full = _stack(swrm.full_pos_history)

    assert len(series) == len(run.times) == 8
    assert np.allclose(series[0], full[0])
    assert np.allclose(series[-1], full[-1]), 'negative indexing'
    assert np.allclose(series[2:6], full[2:6]), 'a slice spanning chunks'
    assert series[3:3].shape == (0, 4, 2), 'an empty slice'
    assert np.allclose(np.stack([c for c in series]), full), 'iteration'
    with pytest.raises(IndexError):
        series[len(series)]


# --------------------------------------------------------------------------- #
#                                  refusals                                    #
# --------------------------------------------------------------------------- #

def test_a_missing_middle_chunk_is_refused_rather_than_short_read(tmp_path):
    # Chunks are written in order, so a hard kill costs the LAST buffer and
    # never a middle one -- a gap is a lost or corrupt file, and reading around
    # it would hand back a run with a hole nobody was told about.
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    (run.path / 'agents' / 'times_0001.npy').unlink()
    with pytest.raises(ValueError, match='missing time chunk'):
        archive.load_run(run.path)


def test_a_missing_swarm_chunk_is_refused(tmp_path):
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    (run.path / 'agents' / 'swarm00_pos_0001.npy').unlink()
    with pytest.raises(ValueError, match='missing chunk'):
        archive.load_run(run.path)


def test_a_chunk_with_the_wrong_number_of_rows_is_refused(tmp_path):
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    target = run.path / 'agents' / 'swarm00_pos_0001.npy'
    np.save(target, np.load(target)[:1])           # truncate it by hand
    with pytest.raises(ValueError, match='rows where the capture count implies'):
        archive.load_run(run.path)


def test_an_array_that_was_not_stored_is_refused_by_name(tmp_path):
    envir = _envir()
    swrm = _swarm(envir)
    with pytest.warns(UserWarning):
        rec = envir.record(tmp_path / 'run', store=('positions',))
    swrm.move(0.5, silent=True)
    envir.stop_recording()
    run = archive.load_run(rec.path)
    assert run.store == ('positions',)
    with pytest.raises(ValueError, match="does not hold 'velocities'"):
        run.velocities(0)


def test_a_future_format_version_is_refused(tmp_path):
    run, _, _ = _recorded(tmp_path)
    meta = json.loads((run.path / 'meta.json').read_text())
    meta['version'] = archive.FORMAT_VERSION + 1
    (run.path / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='format version'):
        archive.load_run(run.path)


def test_a_directory_that_is_not_an_archive_is_refused(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError, match='no meta.json'):
        archive.load_run(tmp_path / 'empty')
    with pytest.raises(FileNotFoundError, match='no archive directory'):
        archive.load_run(tmp_path / 'nowhere')


# --------------------------------------------------------------------------- #
#                    validating against an Environment                         #
# --------------------------------------------------------------------------- #

def test_a_differently_gridded_environment_is_refused_naming_both_sides(tmp_path):
    run, _, _ = _recorded(tmp_path)
    other = planktos.Environment(Lx=99.0, Ly=10.0,
                                 flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    with pytest.raises(ValueError) as excinfo:
        run.check_against(other)
    message = str(excinfo.value)
    assert 'does not match' in message
    assert 'L:' in message, 'the message should name the field that differs'
    assert 'archive was recorded against' in message


def test_the_matching_environment_passes(tmp_path):
    run, envir, _ = _recorded(tmp_path)
    run.check_against(envir)                       # no raise, no warning


def test_a_different_source_on_the_same_grid_warns_rather_than_refusing(tmp_path):
    # Replotting a run whose script moved directories must not be refused; a
    # different simulation sharing a mesh and a cadence should still say so.
    run, envir, _ = _recorded(tmp_path)
    envir._fluid_provenance = {'loader': 'read_IB2d_fluid_data',
                               'kwargs': {'path': 'somewhere_else'}}
    with pytest.warns(UserWarning, match='grid for grid'):
        run.check_against(envir)


def test_reading_never_writes(tmp_path):
    # A reader that mutates the thing it reads is the wrong shape -- and it
    # must never flush a recording on the writer's behalf.
    run, _, _ = _recorded(tmp_path)
    before = {f: f.stat().st_mtime_ns for f in run.path.rglob('*') if f.is_file()}
    run.positions(0).asarray()
    run.velocities(0)[2]
    run.capture_at(1.0)
    after = {f: f.stat().st_mtime_ns for f in run.path.rglob('*') if f.is_file()}
    assert before == after, 'reading modified the archive'


@pytest.mark.parametrize('victim', ['pos', 'vel', 'mask'])
def test_a_missing_chunk_of_any_stored_array_is_refused(tmp_path, victim):
    # Validation used to check only the first stored array, so a missing
    # velocity or mask chunk passed load_run and surfaced later as a bare
    # FileNotFoundError from the middle of a read. Every array is as fatal as
    # the positions one.
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    run.close()
    (run.path / 'agents' / 'swarm00_{}_0001.npy'.format(victim)).unlink()
    with pytest.raises(ValueError, match='missing chunk'):
        archive.load_run(run.path)


@pytest.mark.parametrize('victim', ['pos', 'vel', 'mask'])
def test_a_short_chunk_of_any_stored_array_is_refused(tmp_path, victim):
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    run.close()
    target = run.path / 'agents' / 'swarm00_{}_0001.npy'.format(victim)
    np.save(target, np.load(target)[:1])
    with pytest.raises(ValueError, match='rows where the capture count implies'):
        archive.load_run(run.path)


def test_close_releases_the_files_the_archive_had_open(tmp_path):
    # Reading leaves chunks memory-mapped, and a memory map holds its file
    # open -- which on Windows locks it, so an archive that has been read
    # cannot be deleted or moved until it is closed.
    run, _, _ = _recorded(tmp_path, steps=11, chunk_size=3)
    run.positions(0)[4]
    assert run._cache, 'nothing was mapped; the test proves nothing'
    run.close()
    assert not run._cache
    # the proof: the files can now be removed
    for f in (run.path / 'agents').glob('*.npy'):
        f.unlink()


def test_close_is_idempotent_and_the_archive_still_reads(tmp_path):
    run, _, swrm = _recorded(tmp_path, steps=7, chunk_size=3)
    run.positions(0)[2]
    run.close()
    run.close()
    assert np.allclose(run.positions(0)[2],
                       np.ma.getdata(swrm.full_pos_history[2]))


def test_the_archive_is_a_context_manager(tmp_path):
    run, _, _ = _recorded(tmp_path, steps=7, chunk_size=3)
    run.close()
    with archive.load_run(run.path) as reopened:
        assert len(reopened.positions(0)) == len(reopened.times)
    assert not reopened._cache


@pytest.mark.parametrize('t', [float('nan'), float('inf'), -float('inf')])
def test_capture_at_refuses_a_non_finite_time(tmp_path, t):
    # |times - t| is uniform for a non-finite t, so argmin returns capture 0 --
    # an answer that looks like a real one. Infinity in particular reads as
    # "the end" and would silently have given the beginning.
    run, _, _ = _recorded(tmp_path, steps=5)
    with pytest.raises(ValueError, match='non-finite time'):
        run.capture_at(t)
    run.close()


@pytest.mark.parametrize('slicing, expect', [
    (slice(0, 8, 2), 4), (slice(None, None, -1), 8), (slice(5, 99), 3),
    (slice(-3, None), 3), (slice(3, 3), 0),
])
def test_capture_series_slicing_matches_the_history(tmp_path, slicing, expect):
    run, _, swrm = _recorded(tmp_path, steps=7, chunk_size=3)
    full = _stack(swrm.full_pos_history)
    got = run.positions(0)[slicing]
    assert got.shape[0] == expect
    if expect:
        assert np.allclose(got, full[slicing])
    run.close()


def test_an_archive_can_be_read_while_it_is_still_recording(tmp_path):
    # What flush_recording exists for: a mid-run plot must not have to stop the
    # recording, since the next record() would then refuse the directory.
    envir = _envir()
    swrm = _swarm(envir)
    rec = envir.record(tmp_path / 'run', chunk_size=100)
    for _ in range(5):
        swrm.move(0.5, silent=True)
    envir.flush_recording()
    with archive.load_run(rec.path) as partway:
        assert len(partway.times) == 6
    for _ in range(5):
        swrm.move(0.5, silent=True)
    envir.flush_recording()
    with archive.load_run(rec.path) as later:
        assert len(later.times) == 11
    envir.stop_recording()


def test_an_archive_killed_before_its_first_flush_still_opens(tmp_path):
    # The sidecar is written when the swarm joins, but no chunk has landed yet.
    # An archive of nothing is still a valid archive of nothing.
    envir = _envir()
    swrm = _swarm(envir)
    rec = envir.record(tmp_path / 'run', chunk_size=1000)
    for _ in range(4):
        swrm.move(0.5, silent=True)
    envir._recorder = None                        # a kill: no flush, no close
    envir._capture_interval = 1

    run = archive.load_run(rec.path)
    assert len(run.times) == 0
    assert run.swarms == [('organism', 0)]
    assert run.positions(0).shape == (0, 4, 2)
    with pytest.raises(ValueError, match='no captures'):
        run.capture_at(0.0)


def test_a_swarm_joining_exactly_on_a_chunk_boundary(tmp_path):
    envir = _envir()
    first = _swarm(envir, seed=1)
    with envir.record(tmp_path / 'run', chunk_size=3) as rec:
        for _ in range(2):
            first.move(0.5, silent=True)          # captures 0,1,2 -> next is 3
        _swarm(envir, n=2, seed=9, init=4.0)
        for _ in range(4):
            envir.move_swarms(0.5, silent=True)
    with archive.load_run(rec.path) as run:
        entry = json.loads(
            (run.path / 'agents' / 'swarm01.json').read_text())
        assert entry['first_capture'] == 3, 'should start a chunk, not split one'
        late = run.positions(1)
        assert late.shape == (len(run.times), 2, 2)
        assert ma.getmaskarray(late[2]).all()
        assert not ma.getmaskarray(late[3]).any()


def test_a_capture_schedule_and_a_late_swarm_together(tmp_path):
    envir = _envir()
    first = _swarm(envir, seed=1)
    with envir.record(tmp_path / 'run', capture_interval=3, chunk_size=2) as rec:
        for _ in range(6):
            first.move(0.5, silent=True)
        _swarm(envir, n=2, seed=9, init=4.0)
        for _ in range(6):
            envir.move_swarms(0.5, silent=True)
    with archive.load_run(rec.path) as run:
        assert np.allclose(run.times, [0.0, 1.5, 3.0, 4.5, 6.0])
        assert run.positions(1).shape == (5, 2, 2)
        assert ma.getmaskarray(run.positions(1)[2]).all()
        assert not ma.getmaskarray(run.positions(1)[-1]).any()
