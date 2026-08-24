'''
Run archives: append-only, crash-valid capture of agent state during a run.

Planktos holds an entire run in memory and can only write it out at the end.
An archive streams it to disk instead, as the run proceeds -- the mirror image
of what dynamic fluid loading does for the velocity field. Same architecture,
opposite direction, same reason: the thing is too big to hold at once.

This module owns the on-disk format. The design note is
``docs/notes/run_persistence.md``; section 2 is the archive and section 2.3 the
schema. What follows is what a reader of *this file* needs.

The layout::

    run_archive/
      meta.json                 written once, never rewritten
      grid.npz                  the fingerprint: dimension, L, flow_points,
                                flow_times, periodic_dim
      agents/
        swarm00.json            name, N, D, first_capture
        swarm00_pos_0000.npy    (rows, N, D)
        swarm00_vel_0000.npy    (rows, N, D)
        swarm00_mask_0000.npy   (rows, N) bool
        times_0000.npy          (rows,) shared across swarms
      fluid/                    component B; nothing here writes it

Three properties hold the format together, and every design choice below serves
one of them:

**It is valid with nothing having run at the end.** A hard kill -- HPC walltime,
OOM, node failure -- is a ``SIGKILL``, which defeats ``__exit__``, ``close()``,
``atexit`` and ``__del__`` alike. So metadata is written when recording starts,
every file appears atomically, and the reader reconstructs the timeline by
scanning what is on disk rather than trusting a recorded count. No finalizer is
load-bearing: the most any of them can save is one unflushed chunk. This is what
makes an archive worth having for cluster work at all, since the runs most
likely to be killed are exactly the ones most expensive to repeat.

**What accumulates lives in files that accumulate.** ``meta.json`` holds only
what is known when recording starts and never changes; the capture times, the
agent arrays and the per-swarm roster live in their own files and are discovered
by scanning. That is what lets ``meta.json`` be a single write that is never
touched again -- including when a swarm joins an hour into the run.

**Chunks are keyed on a global capture index**, not on each swarm's own row
count, so a swarm added mid-run needs no second indexing scheme: its first chunk
is simply short at the front.

Public surface: ``RunArchive`` and ``load_run`` for reading (not built yet), and
the three ``*fingerprint*`` functions, which are what a refusal message is
assembled from and are therefore worth having to hand when diagnosing one.
Everything else is underscored and is format machinery.

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.ma as ma

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

# Bumped when the on-disk layout changes in a way a previous reader would get
#   wrong. A reader refuses a version it does not know.
FORMAT_VERSION = 1

# Agent arrays are float64 throughout. This is recorded in the metadata rather
#   than assumed, so that a later single-precision option cannot silently change
#   what an existing archive means.
DTYPE = np.dtype('float64')

# Suffix for a file being written. A kill during the write leaves one of these
#   behind rather than a truncated real file; readers must ignore them.
TMP_SUFFIX = '.partial'

# Filename index width. Four digits at the default chunk size covers a million
#   captures. It is presentation only -- indices are parsed as integers and
#   sorted numerically, never lexically, because %04d simply grows a fifth digit
#   at chunk 10000 and lexical order would then put _10000 before _9999. This
#   branch has paid for that mistake once already, in the OpenFOAM dump
#   directories.
INDEX_WIDTH = 4



def _atomic_write(path, write_fn):
    '''Write a file so that it appears complete or not at all.

    Writes to a temporary name in the same directory, flushes it to disk, and
    then renames it into place -- ``os.replace`` is atomic on POSIX, and on
    Windows for a destination on the same volume. Without this the crash
    guarantee would be wrong in a way worse than a missing file: a kill during
    ``np.save`` leaves a **truncated .npy** that raises on read, so one unlucky
    moment would cost the whole archive rather than one buffer.

    The flush is a real ``fsync``, not just a buffer flush. ``os.replace`` alone
    is enough to survive process death, which is the common case, but a node
    failure or power loss can take the page cache with it -- and those are
    exactly the runs an archive exists for. The cost is one sync per chunk, so
    once per ``chunk_size`` captures, which is negligible against the physics of
    that many steps.

    Parameters
    ----------
    path : Path
        final destination
    write_fn : callable
        called with an open binary file object; writes the contents
    '''

    tmp = path.with_name(path.name + TMP_SUFFIX)
    try:
        with open(tmp, 'wb') as fobj:
            write_fn(fobj)
            fobj.flush()
            os.fsync(fobj.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Leave nothing half-written behind. BaseException so that a
        #   KeyboardInterrupt lands here too -- interrupting a long run is the
        #   most common way one ends.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _save_npy(path, array):
    '''np.save, atomically.'''

    _atomic_write(path, lambda fobj: np.save(fobj, array, allow_pickle=False))


def _save_json(path, obj):
    '''json.dump, atomically, and refusing anything that is not strict JSON.

    ``allow_nan=False`` is deliberate: Python's json module emits bare ``NaN``
    and ``Infinity`` by default, which no other tool will read back. Failing
    here, at the start of a run, beats writing a metadata file that turns out to
    be unparsable at the end of one.
    '''

    text = json.dumps(obj, indent=2, allow_nan=False, sort_keys=True)
    _atomic_write(path, lambda fobj: fobj.write(text.encode('utf-8')))


def _chunk_name(prefix, index):
    '''e.g. ("swarm00_pos", 7) -> "swarm00_pos_0007.npy".'''

    return '{}_{:0{}d}.npy'.format(prefix, index, INDEX_WIDTH)


def _chunk_index_of(name):
    '''Recover the chunk index from a chunk filename.

    The inverse of :func:`_chunk_name`, and the reason chunk discovery is safe:
    a reader parses this integer and sorts on it, rather than sorting filenames.
    Zero-padding makes the two agree only up to chunk 9999 -- at 10000 the name
    grows a fifth digit and lexical order puts ``_10000`` before ``_9999``,
    silently assembling a run out of order. Padding is for humans; the parse is
    for correctness.

    Parameters
    ----------
    name : str or Path

    Returns
    -------
    int, or None if this is not a chunk file
    '''

    stem = Path(name).stem
    _, _, tail = stem.rpartition('_')
    try:
        return int(tail)
    except ValueError:
        return None


def _swarm_prefix(index):
    '''e.g. 3 -> "swarm03". Two digits by convention, more if ever needed.'''

    return 'swarm{:02d}'.format(index)



#############################################################################
#                                                                           #
#                              FINGERPRINT                                  #
#                                                                           #
#############################################################################

# The fingerprint answers one question -- is this archive about the same
#   coordinate system and timeline as this environment? -- and a mismatch is a
#   hard refusal, because the stored arrays are not interpretable otherwise.
#   It deliberately does NOT answer "did the same thing produce it": that is the
#   provenance record's job (planktos/_provenance.py), and a mismatch there is a
#   warning. Keeping the two apart gets both realistic cases right. Replotting a
#   run whose script moved directories should not be refused; a different
#   simulation that happens to share a mesh and a cadence should say so out loud.
#
#   Be plain about what it does not catch: two runs on the same grid at the same
#   timestamps fingerprint identically, and nothing cheap catches a dataset
#   regenerated in place at the same path. It bounds the damage; it does not
#   eliminate it.

def build_fingerprint(dimension, L, flow_points=None, flow_times=None,
                      periodic_dim=None):
    '''Assemble the arrays that identify a fluid dataset and domain.

    These are what goes into ``grid.npz``. They are small -- a few hundred
    floats even for a large 3D grid, one float per dump for the timeline -- and
    the archive has to store the axes anyway so that it can plot without
    touching fluid. The fingerprint is therefore not a new stored artifact; it
    is a comparison over a file that had to exist.

    ``periodic_dim`` is part of it because it changes the vorticity computed in
    the outermost ring, so a stored vorticity field recorded under a different
    setting is a different field.

    Parameters
    ----------
    dimension : int
        2 or 3
    L : array_like
        domain size per dimension
    flow_points : tuple of ndarray, optional
        per-axis grid coordinates. None for a flow-free environment, in which
        case the fingerprint is dimension and L alone -- nothing becomes
        optional, it just gets smaller.
    flow_times : ndarray, optional
        the fluid time base, covering the whole dump series (not the resident
        window -- see FluidData's guard on that)
    periodic_dim : bool or sequence of bool, optional
        broadcast to one flag per dimension

    Returns
    -------
    dict of ndarray, ready for np.savez
    '''

    dimension = int(dimension)
    arrays = {'dimension': np.array(dimension, dtype=np.int64),
              'L': np.asarray(L, dtype=np.float64)}

    if arrays['L'].shape != (dimension,):
        raise ValueError('L has {} entries but the domain is {}D'.format(
            arrays['L'].size, dimension))

    if periodic_dim is None:
        periodic_dim = False
    arrays['periodic_dim'] = np.broadcast_to(
        np.asarray(periodic_dim, dtype=bool), (dimension,)).copy()

    if flow_points is not None:
        if len(flow_points) != dimension:
            raise ValueError(
                'flow_points has {} axes but the domain is {}D'.format(
                    len(flow_points), dimension))
        # One key per axis: the axes have different lengths, so they cannot go
        #   into an npz as a single array without object dtype, and object
        #   arrays would drag pickling into a format meant to be readable by
        #   anything that can open a .npy.
        for axis, points in enumerate(flow_points):
            arrays['flow_points_{}'.format(axis)] = np.asarray(
                points, dtype=np.float64)

    if flow_times is not None:
        arrays['flow_times'] = np.asarray(flow_times, dtype=np.float64)

    return arrays


def fingerprint_summary(arrays):
    '''A json-safe précis of the fingerprint, for ``meta.json``.

    This is the only thing in ``meta.json`` that says what fluid the archive was
    recorded against, so it duplicates ``grid.npz`` deliberately and by the same
    argument the provenance record makes: this is the file someone opens to see
    what a run *was*, and a summary that needs a second file loaded to be
    legible is worse at exactly the job it exists for. Both are written once, in
    the same call, from one source, so they cannot drift within a run.

    It is a *description*, never the match test -- that is
    :func:`compare_fingerprints`, which reads the arrays so it can say what
    differs.
    '''

    dimension = int(arrays['dimension'])
    summary = {'dimension': dimension,
               'L': [float(v) for v in arrays['L']],
               'periodic_dim': [bool(v) for v in arrays['periodic_dim']]}

    axes = ['flow_points_{}'.format(n) for n in range(dimension)]
    if all(key in arrays for key in axes):
        summary['grid_shape'] = [int(arrays[key].size) for key in axes]
    else:
        summary['grid_shape'] = None

    if 'flow_times' in arrays and arrays['flow_times'].size:
        times = arrays['flow_times']
        summary['n_dumps'] = int(times.size)
        summary['time_span'] = [float(times[0]), float(times[-1])]
    else:
        summary['n_dumps'] = 0
        summary['time_span'] = None

    return summary


def compare_fingerprints(stored, current):
    '''Return a list of human-readable differences; empty if they match.

    Comparison is exact -- shape, dtype and values -- because a rebuilt
    environment re-runs the same loader over the same files and gets
    bit-identical arrays. ``flow_points``, ``flow_times`` and ``L`` are built in
    each loader's ``__init__`` and are never reassigned by ``load_dumpfiles`` or
    ``update_spline``, so a windowed run does not drift from the values recorded
    when it started.

    The differences are returned rather than raised so that the caller can put
    them in a message alongside both sides' provenance, which is what makes a
    refusal actionable instead of merely correct.

    Parameters
    ----------
    stored, current : dict of ndarray
        as returned by :func:`build_fingerprint`

    Returns
    -------
    list of str
    '''

    problems = []
    for key in sorted(set(stored) | set(current)):
        if key not in stored:
            problems.append('{}: absent from the archive, present here'.format(key))
            continue
        if key not in current:
            problems.append('{}: recorded in the archive, absent here'.format(key))
            continue
        a, b = np.asarray(stored[key]), np.asarray(current[key])
        # array_equal is False rather than an error on mismatched shapes, so one
        #   branch covers both -- and both get described, since "shape (6,) vs
        #   (9,)" tells the reader less than "6 dump times spanning 0 to 1.4".
        if not np.array_equal(a, b):
            problems.append('{}: archive has {}, this environment {}'.format(
                key, _describe(a), _describe(b)))
    return problems


def _describe(array):
    '''A short description of an array, for a difference message.'''

    array = np.asarray(array)
    if array.ndim == 0:
        return repr(array.item())
    if array.size <= 4:
        return repr([v.item() for v in array.ravel()])
    return '{} values spanning {!r} to {!r}'.format(
        array.size, array.ravel()[0].item(), array.ravel()[-1].item())



#############################################################################
#                                                                           #
#                             DIRECTORY                                     #
#                                                                           #
#############################################################################

def _resolve_archive_path(path):
    '''Choose the directory to record into, and create it.

    Overwriting a previous run's data is never the right default, and refusing
    outright would strand a long job that was ready to start. So a non-empty
    directory is left alone and a timestamped sibling is used instead. The
    redirect is never silent: it warns, naming the path actually chosen, and the
    writer exposes it as ``.path``. Without that, a later
    ``load_run('run_archive/')`` would quietly read the *previous* run.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Path
        the created directory, which may not be the one asked for
    '''

    path = Path(path)
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path

    stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    candidate = path.with_name('{}_{}'.format(path.name, stamp))
    # Two archives started in the same second would collide; walk forward until
    #   an unused name is found rather than silently sharing one.
    suffix = 0
    while candidate.exists() and any(candidate.iterdir()):
        suffix += 1
        candidate = path.with_name('{}_{}_{}'.format(path.name, stamp, suffix))

    candidate.mkdir(parents=True, exist_ok=True)
    warnings.warn(
        "{} already holds data, so this run is being recorded to {} instead. "
        "Read it back with that path -- load_run('{}') would open the earlier "
        "run.".format(path, candidate, path), UserWarning, stacklevel=3)
    return candidate



#############################################################################
#                                                                           #
#                               WRITER                                      #
#                                                                           #
#############################################################################

class _ArchiveWriter:
    '''Writes agent captures to an archive directory as a run proceeds.

    This is the low-level half of recording: it is handed data and writes it,
    and knows nothing about Environments, Swarms, time steps or hooks. What
    drives it is ``Environment.record``. Keeping the split means the format can
    be tested without running a simulation, and a change to the capture schedule
    cannot reach into the format.

    Metadata is written by ``__init__`` -- when recording *starts*, per the
    crash-validity rule -- so constructing one of these creates the directory
    and commits the archive's identity. Nothing after that point is required for
    what is already on disk to be readable.

    Parameters
    ----------
    path : str or Path
        directory to record into. Created if missing; if it exists and is
        non-empty, a timestamped sibling is used instead and a warning names it.
        The resolved directory is available afterwards as ``.path``.
    fingerprint : dict of ndarray
        from :func:`build_fingerprint`
    meta : dict, optional
        extra archive-level metadata to record: the provenance block, which
        fluid quantity the render will need, the quiver grid. Must be strict
        JSON, and must be fixed for the life of the recording -- anything that
        accumulates belongs in a file that accumulates.
    chunk_size : int, default=100
        captures buffered before a chunk is written

    Attributes
    ----------
    path : Path
        the directory actually being written to
    '''

    def __init__(self, path, fingerprint, meta=None, chunk_size=100):
        if int(chunk_size) < 1:
            raise ValueError('chunk_size must be at least 1')
        self.chunk_size = int(chunk_size)
        self.path = _resolve_archive_path(path)
        self.agent_dir = self.path / 'agents'
        self.agent_dir.mkdir(exist_ok=True)
        # Component B writes here. Created now so that the layout is complete
        #   from the start and B never has to wonder whether it exists.
        (self.path / 'fluid').mkdir(exist_ok=True)

        self._fingerprint = dict(fingerprint)
        _atomic_write(self.path / 'grid.npz',
                      lambda fobj: np.savez(fobj, **self._fingerprint))

        record = dict(meta) if meta else {}
        record.update(version=FORMAT_VERSION,
                      dtype=str(DTYPE),
                      chunk_size=self.chunk_size,
                      grid=fingerprint_summary(self._fingerprint))
        _save_json(self.path / 'meta.json', record)
        self.meta = record

        # index -> {'name', 'N', 'D', 'first_capture'}
        self._swarms = {}
        # buffers for the chunk currently being filled
        self._chunk = None              # which chunk index that is
        self._times = []
        self._pos = {}
        self._vel = {}
        self._mask = {}
        self._next_capture = None       # the capture index expected next
        self._closed = False


    ####################   recording   ####################

    def add_swarm(self, index, name, N, D, first_capture):
        '''Register a swarm, writing its sidecar immediately.

        The roster lives in per-swarm files rather than in ``meta.json`` so that
        a swarm joining mid-run is an ordinary case: nothing already written is
        touched, and early and late swarms are discovered by the same scan. The
        only thing that distinguishes them is ``first_capture``.

        Parameters
        ----------
        index : int
            position in the recorder's swarm list; fixed for the run
        name : str
            the Swarm's name. Not used in filenames -- the default name is
            'organism' for every Swarm, so two swarms in one environment collide
            by name and a file built from it would silently overwrite.
        N, D : int
            agent count and spatial dimension
        first_capture : int
            global capture index at which this swarm starts. 0 for every swarm
            present when recording began.
        '''

        index = int(index)
        if index in self._swarms:
            raise ValueError('swarm index {} is already recorded'.format(index))
        entry = {'index': index, 'name': str(name), 'N': int(N), 'D': int(D),
                 'first_capture': int(first_capture)}
        self._swarms[index] = entry
        _save_json(self.agent_dir / (_swarm_prefix(index) + '.json'), entry)
        self._pos[index] = []
        self._vel[index] = []
        self._mask[index] = []


    def add_capture(self, capture_index, time, arrays):
        '''Buffer one capture, writing a chunk when one fills.

        Parameters
        ----------
        capture_index : int
            the global capture index. Must be the next one expected: captures
            are a contiguous series, and a gap here would silently become a gap
            on disk that the reader could only interpret as a lost file.
        time : float
            simulated time of this capture
        arrays : dict
            swarm index -> (positions, velocities), each an ``N x D`` masked
            array. Exactly the swarms whose ``first_capture`` has been reached
            must be present.
        '''

        if self._closed:
            raise RuntimeError('this archive writer has been closed')

        capture_index = int(capture_index)
        if self._next_capture is None:
            self._next_capture = capture_index
        if capture_index != self._next_capture:
            raise ValueError(
                'expected capture {}, got {}; captures must be contiguous'.format(
                    self._next_capture, capture_index))

        expected = {idx for idx, entry in self._swarms.items()
                    if entry['first_capture'] <= capture_index}
        if set(arrays) != expected:
            raise ValueError(
                'capture {} covers swarms {}, but {} were expected'.format(
                    capture_index, sorted(arrays), sorted(expected)))

        chunk = capture_index // self.chunk_size
        if self._chunk is None:
            self._chunk = chunk
        elif chunk != self._chunk:
            self._write_chunk()
            self._chunk = chunk

        self._times.append(float(time))
        for idx, (positions, velocities) in arrays.items():
            entry = self._swarms[idx]
            pos_data, row_mask = self._split(positions, entry, 'positions')
            vel_data, _ = self._split(velocities, entry, 'velocities')
            self._pos[idx].append(pos_data)
            self._vel[idx].append(vel_data)
            self._mask[idx].append(row_mask)

        self._next_capture += 1


    @staticmethod
    def _split(array, entry, label):
        '''Separate an N x D masked array into plain data and a row mask.

        A masked row means the agent has left the domain -- agents leave whole
        rows, never single coordinates -- so the mask is stored per row rather
        than per element. A partially masked row would therefore lose
        information on the way to disk, and is refused rather than flattened,
        since it means an invariant broke upstream and quietly rounding it off
        would hide that.
        '''

        expected = (entry['N'], entry['D'])
        if np.shape(array) != expected:
            raise ValueError('swarm {} {} has shape {}, expected {}'.format(
                entry['index'], label, np.shape(array), expected))

        mask = ma.getmaskarray(array)
        any_masked = mask.any(axis=1)
        if not np.array_equal(any_masked, mask.all(axis=1)):
            raise ValueError(
                'swarm {} has a partially masked {} row; a masked row means the '
                'agent left the domain, so rows are masked whole'.format(
                    entry['index'], label))

        return np.ma.getdata(array).astype(DTYPE, copy=True), any_masked


    def flush(self):
        '''Write everything buffered, and keep recording.

        Separate from :meth:`close` on purpose: a mid-run plot needs a flush
        that does not end the recording. Calling it repeatedly is harmless --
        the partial chunk is rewritten in place, atomically, growing as more
        captures arrive.
        '''

        if self._chunk is not None and self._times:
            self._write_chunk(keep=True)


    def close(self):
        '''Flush and stop accepting captures.

        Nothing here is load-bearing for correctness. If this is never reached
        -- a hard kill, an interpreter torn down -- the archive on disk is still
        valid, and the most that is lost is the captures buffered since the last
        chunk boundary or flush.
        '''

        if not self._closed:
            self.flush()
            self._closed = True


    ####################   internals   ####################

    def _write_chunk(self, keep=False):
        '''Write the open chunk's buffers to disk.

        Parameters
        ----------
        keep : bool, default=False
            if True, leave the buffers in place so recording continues into the
            same chunk (a flush). If False, the chunk is complete and the
            buffers are cleared for the next one.
        '''

        index = self._chunk
        _save_npy(self.agent_dir / _chunk_name('times', index),
                  np.asarray(self._times, dtype=DTYPE))

        for idx in sorted(self._pos):
            if not self._pos[idx]:
                # A swarm that joined after this chunk started contributes
                #   nothing to it. Writing a zero-row file would be a lie about
                #   its first_capture; absence is the honest record, and the
                #   reader resolves the offset from the sidecar.
                continue
            prefix = _swarm_prefix(idx)
            _save_npy(self.agent_dir / _chunk_name(prefix + '_pos', index),
                      np.stack(self._pos[idx]))
            _save_npy(self.agent_dir / _chunk_name(prefix + '_vel', index),
                      np.stack(self._vel[idx]))
            _save_npy(self.agent_dir / _chunk_name(prefix + '_mask', index),
                      np.stack(self._mask[idx]))

        if not keep:
            self._times = []
            for idx in self._pos:
                self._pos[idx] = []
                self._vel[idx] = []
                self._mask[idx] = []
