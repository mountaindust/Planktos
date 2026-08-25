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

Public surface: ``RunArchive`` and ``load_run`` for reading, ``CaptureSeries``
(what a read hands back), ``RunRecorder`` (what ``Environment.record`` hands
back), and the three ``*fingerprint*`` functions, which are what a refusal
message is assembled from and are therefore worth having to hand when
diagnosing one. Everything else is underscored and is format machinery.

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

import planktos
from . import _provenance

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

# Per-agent arrays the archive can store, mapped to the short name used in
#   filenames -- the same shorthand Swarm already uses for its histories. The
#   mask is not in here: it is derived from positions and always written, since
#   it is how a reader knows an agent left the domain. 'accelerations' is a
#   reserved slot, wired through but not yet offered by Environment.record.
STORABLE = {'positions': 'pos', 'velocities': 'vel', 'accelerations': 'acc'}



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

def _describe_source(provenance):
    """A short phrase naming what produced a fluid, for a refusal message.

    Takes a provenance record (planktos/_provenance.py) directly, so that both
    sides of a comparison -- one out of an archive's metadata, one off a live
    Environment -- describe themselves the same way.
    """

    if not provenance:
        return 'an unrecorded source'
    if provenance.get('loader') is None:
        return provenance.get('note', 'arrays supplied directly')
    kwargs = provenance.get('kwargs') or {}
    path = kwargs.get('path') or kwargs.get('filename')
    if path is None:
        return '{}(...)'.format(provenance['loader'])
    return "{}(path={!r}, ...)".format(provenance['loader'], path)


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

    def __init__(self, path, fingerprint, meta=None, chunk_size=100,
                 store=('positions', 'velocities')):
        if int(chunk_size) < 1:
            raise ValueError('chunk_size must be at least 1')
        self.store = tuple(store)
        unknown = [name for name in self.store if name not in STORABLE]
        if unknown:
            raise ValueError('cannot store {}; known arrays are {}'.format(
                unknown, sorted(STORABLE)))
        if 'positions' not in self.store:
            raise ValueError(
                'positions must be stored: nothing consumes an archive without '
                'them, in or out of plotting')
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
                      store=list(self.store),
                      grid=fingerprint_summary(self._fingerprint))
        _save_json(self.path / 'meta.json', record)

        # index -> {'name', 'N', 'D', 'first_capture'}
        self._swarms = {}
        # buffers for the chunk currently being filled
        self._chunk = None              # which chunk index that is
        self._times = []
        # swarm index -> {'positions': [...], 'velocities': [...], 'mask': [...]}
        self._buffers = {}
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
        self._buffers[index] = {name: [] for name in self.store}
        self._buffers[index]['mask'] = []


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
            swarm index -> {name: ``N x D`` masked array}, whose names must be
            exactly this writer's ``store``. Exactly the swarms whose
            ``first_capture`` has been reached must be present.
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
        for idx, named in arrays.items():
            entry = self._swarms[idx]
            if set(named) != set(self.store):
                raise ValueError(
                    'swarm {} supplied {}, but this archive stores {}'.format(
                        idx, sorted(named), sorted(self.store)))

            # Positions first: one mask is stored per capture, and it is theirs.
            #   A masked row means the agent is not in the domain, which is a
            #   fact about the agent rather than about any one array, so every
            #   other array must agree -- and a disagreement is refused rather
            #   than silently dropped, since the format has nowhere to put it.
            reference = None
            for name in ['positions'] + [n for n in self.store
                                         if n != 'positions']:
                data, row_mask = self._split(named[name], entry, name)
                if reference is None:
                    reference = row_mask
                    self._buffers[idx]['mask'].append(row_mask)
                elif not np.array_equal(row_mask, reference):
                    raise ValueError(
                        'swarm {} has {} masked differently from its positions. '
                        'One mask is stored per capture, so a disagreement '
                        'cannot be represented and would be lost.'.format(
                            idx, name))
                self._buffers[idx][name].append(data)

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

        for idx in sorted(self._buffers):
            buffers = self._buffers[idx]
            if not buffers['mask']:
                # A swarm that joined after this chunk started contributes
                #   nothing to it. Writing a zero-row file would be a lie about
                #   its first_capture; absence is the honest record, and the
                #   reader resolves the offset from the sidecar.
                continue
            prefix = _swarm_prefix(idx)
            for name in self.store:
                _save_npy(
                    self.agent_dir / _chunk_name(
                        '{}_{}'.format(prefix, STORABLE[name]), index),
                    np.stack(buffers[name]))
            _save_npy(self.agent_dir / _chunk_name(prefix + '_mask', index),
                      np.stack(buffers['mask']))

        if not keep:
            self._times = []
            for buffers in self._buffers.values():
                for name in buffers:
                    buffers[name] = []



#############################################################################
#                                                                           #
#                              RECORDER                                     #
#                                                                           #
#############################################################################

class RunRecorder:
    '''Captures agent state to an archive as a run proceeds.

    Returned by :meth:`planktos.Environment.record`, which is the only way to
    make one -- a recorder is environment-scoped by construction. It hooks the
    environment's time advance, its metadata is environment state, and the
    fluid half of the output belongs to the environment rather than to any
    swarm.

    Recording is **live as soon as this exists**. That is the ``open()`` model:
    the call does the work and ``with`` only adds the guaranteed close.

    ::

        with envir.record('run_archive/') as rec:
            for _ in range(steps):
                swrm.move(dt)

    Works without a ``with`` block too, since a ``with`` cannot span notebook 
    cells and interactive exploration -- run 200 steps, plot, run 800 more -- is 
    a supported Planktos workflow. ``envir.flush_recording()`` and 
    ``envir.stop_recording()`` live on the Environment for the same reason.

    Attributes
    ----------
    path : Path
        the directory being written to. **Not necessarily the one asked for**: 
        recording into a non-empty directory redirects to a timestamped sibling
        (with a warning), and this is what says where the data actually went.
    '''

    def __init__(self, envir, path, swarms=None, store=('positions', 'velocities'),
                 chunk_size=100, meta=None):
        self.envir = envir
        # Given an explicit list, capture exactly those. Given none, capture
        #   whatever the environment holds -- including swarms that join later.
        self._track_all = swarms is None
        if swarms is None:
            swarms = list(envir.swarms)
        # Recorded swarms are keyed by their position in this list, fixed for
        #   the run; a swarm joining later gets the next index (see _swarm_added).
        self._swarms = list(swarms)
        self._store = tuple(store)
        self._stopped = False

        record_meta = dict(meta) if meta else {}
        record_meta['provenance'] = {
            'planktos_version': planktos.__version__,
            'environment': {'L': [float(v) for v in envir.L],
                            'units': envir.units,
                            'bndry': [list(b) for b in envir.bndry],
                            'rho': _provenance.jsonable(envir.rho),
                            'mu': _provenance.jsonable(envir.mu)},
            'fluid': envir._fluid_provenance,
            'ibmesh': envir._ibmesh_provenance}

        self._writer = _ArchiveWriter(path, fingerprint_of(envir),
                                      meta=record_meta, chunk_size=chunk_size,
                                      store=self._store)
        self.path = self._writer.path

        self._n_captures = 0
        for index, swarm in enumerate(self._swarms):
            self._register(index, swarm, first_capture=0)
        # Capture 0 covers t0, so that capture j is exactly full_pos_history[j].
        self._capture()


    ####################   lifecycle   ####################

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False


    def flush(self):
        '''Write buffered captures to disk. Keeps recording.'''

        self._writer.flush()


    def stop(self):
        '''Flush, then unhook. Idempotent.'''

        if self._stopped:
            return
        self._stopped = True
        self._writer.close()
        if self.envir._recorder is self:
            self.envir._recorder = None
            # Nothing is being gated any more; history goes back to every step.
            self.envir._capture_interval = 1


    ####################   hooks   ####################

    def _register(self, index, swarm, first_capture):
        # The name lives in shared_props, and defaults to 'organism' for every
        #   Swarm -- which is exactly why files are keyed by index instead.
        name = swarm.shared_props.get('name', 'organism')
        self._writer.add_swarm(index, name, swarm.N, swarm.positions.shape[1],
                               first_capture)


    def _sync_swarms(self):
        '''Pick up any swarm that has joined the environment since last capture.

        **Swarms are discovered here rather than notified from ``Swarm``**, and
        the reason is that a swarm's existence only matters at a capture. Three
        things follow, all of them why this is the right moment rather than a
        convenient one:

        - There is no hook in ``Swarm`` at all, so nothing has to know which of
          the several ways of building one the user reached for. (An earlier
          design hooked ``Environment.add_swarm``, which the usual spelling
          ``planktos.Swarm(envir=envir)`` bypasses entirely; the next hooked
          both sites where a Swarm appends itself, which fires *partway through*
          ``Swarm.__init__``, before ``shared_props`` exists.)
        - **A swarm that comes and goes between two captures is never seen.**
          ``calculate_FTLE`` builds a grid of probe agents on the environment
          and pops it again, so without this it would write a sidecar for its
          own scratch swarm and then expect it in every later capture. FTLE
          needs to know nothing about recording for that to come out right.
        - ``first_capture`` is the capture index the swarm actually starts at,
          by construction, so indices correspond across every swarm with no
          per-swarm time base and no second indexing scheme.
        '''

        if not self._track_all:
            return
        known = {id(swarm) for swarm in self._swarms}
        for swarm in self.envir.swarms:
            if id(swarm) not in known:
                index = len(self._swarms)
                self._swarms.append(swarm)
                self._register(index, swarm, first_capture=self._n_captures)


    def _capture(self):
        '''Record one state for every swarm. Called by the environment's hook.'''

        if self._stopped:
            return
        self._sync_swarms()
        arrays = {}
        for index, swarm in enumerate(self._swarms):
            named = {}
            for name in self._store:
                named[name] = getattr(swarm, name)
            arrays[index] = named
        # Live attributes are read, not the histories: the archive does not
        #   depend on history existing, only on the two agreeing about when a
        #   state is recorded.
        self._writer.add_capture(self._n_captures, self.envir.time, arrays)
        self._n_captures += 1



def fingerprint_of(envir):
    '''Build an Environment's fingerprint (§2.3).

    A flow-free environment fingerprints on dimension and domain alone --
    nothing becomes optional, it just gets smaller.
    '''

    dimension = len(envir.L)
    if envir.flow is None:
        return build_fingerprint(dimension, envir.L)
    return build_fingerprint(dimension, envir.L,
                             flow_points=envir.flow.flow_points,
                             flow_times=envir.flow.flow_times,
                             periodic_dim=envir.flow.periodic_dim)


#############################################################################
#                                                                           #
#                                READER                                     #
#                                                                           #
#############################################################################

class CaptureSeries:
    '''One per agent-array of one swarm, across a whole run, read on demand.

    Returned by :meth:`RunArchive.positions` and :meth:`RunArchive.velocities`.
    Indexing it gives an ordinary masked array -- ``series[j]`` is one capture,
    ``series[a:b]`` a span of them -- and **only the chunks an index touches are
    read**. That is what lets an archive larger than memory stay usable, which
    is half the reason the format is chunked at all.

    It is a sequence, not an ndarray, and deliberately so. This branch has
    already learned what happens to something that pretends to be an array it is
    not: ``FlowArray`` overrode ``.shape`` and ``__getitem__`` so that scipy and
    matplotlib would treat one tile as a whole tiled grid, and modern scipy
    defeated it by calling ``np.asarray`` on anything array-like, silently
    getting the wrong buffer. So this hands back real arrays and never claims to
    be one; :meth:`asarray` materializes the lot when that is what you want, and
    says so in its name.

    A swarm that joined partway through a recording is **front-padded with
    fully-masked rows** up to its first capture, so every series is
    ``len(archive.times)`` long and lines up with ``archive.times`` index for
    index. A masked row already means "this agent is not in the domain"
    everywhere in Planktos; "not yet in the run" is the same statement.

    Attributes
    ----------
    shape : tuple
        ``(n_captures, N, D)``, without reading anything
    '''

    def __init__(self, archive, swarm_index, name):
        self._archive = archive
        self._index = swarm_index
        self._name = name
        entry = archive._by_index[swarm_index]
        self._first = entry['first_capture']
        self._N, self._D = entry['N'], entry['D']
        self.shape = (len(archive.times), self._N, self._D)


    def __len__(self):
        return self.shape[0]


    def __repr__(self):
        return '<CaptureSeries {} of swarm {}, shape {}>'.format(
            self._name, self._index, self.shape)


    def __iter__(self):
        for j in range(len(self)):
            yield self[j]


    def __getitem__(self, key):
        '''A capture, or a span of them, as a masked array.'''

        if isinstance(key, slice):
            rows = range(*key.indices(len(self)))
            if not rows:
                return ma.masked_array(np.empty((0, self._N, self._D), DTYPE),
                                       mask=np.empty((0, self._N, self._D), bool))
            return ma.stack([self._read_capture(j) for j in rows])

        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError('capture {} out of range for {} captures'.format(
                key, len(self)))
        return self._read_capture(key)


    def _read_capture(self, j):
        '''One capture, reading only the chunk it falls in.

        Named apart from ``RunRecorder._capture``, which is the opposite
        direction: that one writes a capture, this one reads one back.
        '''

        if j < self._first:
            # Before this swarm joined the run. Fully masked, which is already
            #   what a masked row means everywhere else in Planktos.
            return ma.masked_array(np.zeros((self._N, self._D), DTYPE),
                                   mask=np.ones((self._N, self._D), bool))

        chunk, offset = self._archive._locate(j, self._first)
        data = self._archive._chunk(self._index, self._name, chunk)[offset]
        mask = self._archive._chunk(self._index, 'mask', chunk)[offset]
        # The stored mask is per row -- agents leave whole rows -- so broadcast
        #   it back across the coordinates on the way out.
        return ma.masked_array(np.array(data, dtype=DTYPE),
                               mask=np.repeat(mask[:, None], self._D, axis=1))


    def asarray(self):
        '''Materialize the whole series as one masked array.

        ⚠️ Reads every chunk and holds the result in memory: this is the call
        that an archive larger than RAM cannot afford. Index or iterate instead
        when that matters.
        '''

        return self[:]



class RunArchive:
    '''A finished (or still-running) archive of agent state, opened for reading.

    Get one from :func:`planktos.load_run`. The archive is **read-only**: it
    never writes, and never flushes a recording on the writer's behalf, because
    a reader that mutates the thing it is reading is the wrong shape.

    ::

        run = planktos.load_run('run_archive/')
        run.times                  # capture times, the archive's own time base
        run.swarms                 # [('organism', 0), ('organism', 1)]
        run.positions(0)[run.capture_at(3.4)]     # where they were at t=3.4

    **Address swarms by index; names are a convenience.** The default ``Swarm``
    name is ``'organism'`` for every swarm, so two swarms in one environment
    collide by name by default -- ``run.positions('organism')`` therefore raises
    when the name is not unique, rather than picking one. ``run.swarms`` lists
    both name and index so a caller can see the collision instead of guessing.

    **Resolve by time, not by index into someone else's list.** A swarm added
    mid-run starts at a nonzero capture, and a recording started after t=0 has
    its capture 0 partway into the run; matching on :attr:`times` is right in
    every one of those cases, where assuming archive index *j* equals history
    index *j* is right only in the common one, and fails silently when it is
    not.

    **Agent state is snapped, never interpolated.** :meth:`capture_at` returns
    the index of the nearest capture and nothing blends between them.
    Interpolating positions across a domain wrap or an immersed-boundary slide
    would invent trajectories that never happened. Temporal interpolation
    belongs to the fluid, where the field is smooth.

    Attributes
    ----------
    path : Path
        the archive directory
    meta : dict
        everything ``meta.json`` holds, provenance included
    times : ndarray
        ``(n_captures,)`` capture times -- the archive's time base
    swarms : list of (str, int)
        ``(name, index)`` for every recorded swarm, in index order
    grid : dict of ndarray
        the fingerprint: dimension, L, flow_points, flow_times, periodic_dim
    store : tuple of str
        which per-agent arrays this archive holds
    '''

    # How many chunk files to keep open at once. A memmap holds a file
    #   descriptor, so caching every chunk of a long run would exhaust them;
    #   a handful is enough for the access pattern that matters, which is a
    #   monotone sweep (a render walking frames in order).
    CACHE_SIZE = 8

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError('no archive directory at {}'.format(self.path))

        meta_file = self.path / 'meta.json'
        if not meta_file.is_file():
            raise FileNotFoundError(
                '{} is not a Planktos run archive: no meta.json'.format(self.path))
        self.meta = json.loads(meta_file.read_text())

        version = self.meta.get('version')
        if version is None or version > FORMAT_VERSION:
            raise ValueError(
                'this archive is format version {}, and this Planktos reads up '
                'to version {}. Upgrade Planktos to read it.'.format(
                    version, FORMAT_VERSION))

        self.store = tuple(self.meta.get('store', ('positions', 'velocities')))
        self._chunk_size = int(self.meta['chunk_size'])
        self.grid = dict(np.load(self.path / 'grid.npz', allow_pickle=False))

        self._agent_dir = self.path / 'agents'
        self._entries = self._read_roster()
        self._by_index = {e['index']: e for e in self._entries}
        self.swarms = [(e['name'], e['index']) for e in self._entries]
        self._time_chunks = self._chunk_indices('times')
        self.times = self._read_times()
        self._cache = {}
        self._validate_chunks()


    def __repr__(self):
        span = '' if not len(self.times) else ', t={:g} to {:g}'.format(
            self.times[0], self.times[-1])
        return '<RunArchive {}: {} captures, {} swarm(s){}>'.format(
            self.path.name, len(self.times), len(self.swarms), span)


    def close(self):
        '''Release the memory-mapped chunk files this archive holds open.

        Reading leaves up to ``CACHE_SIZE`` chunks mapped, and a memory map
        holds its file open. On Windows that **locks the file**, so an archive
        that has been read cannot be deleted or moved until it is closed. Not
        needed to read correctly, and not needed at all on POSIX; needed to
        tidy up afterwards.

        Idempotent, and the archive stays usable -- a later read simply maps
        what it needs again.
        '''

        # A memmap closes its file when the last reference to it goes, so
        #   dropping the cache is the whole of it. (`del` on a loop variable
        #   would not be: it rebinds a name, it does not release the entry.)
        self._cache.clear()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


    ####################   the time base   ####################

    def capture_at(self, t):
        '''Index of the capture nearest in time to ``t``.

        Snapped, never interpolated -- see the class docstring. Ties go to the
        earlier capture.

        Parameters
        ----------
        t : float

        Returns
        -------
        int
        '''

        if not len(self.times):
            raise ValueError('this archive holds no captures')
        if not np.isfinite(t):
            # |times - t| is uniform for a non-finite t, so argmin would return
            #   capture 0 -- an answer that looks like a real one. Infinity in
            #   particular reads as "the end" and would silently give the
            #   beginning.
            raise ValueError(
                'cannot find the capture nearest t={}: a non-finite time is '
                'not nearer to one capture than another'.format(t))
        return int(np.argmin(np.abs(self.times - t)))


    ####################   agent state   ####################

    def positions(self, swarm=0):
        '''The position series of one swarm. See :class:`CaptureSeries`.'''

        return self.array('positions', swarm)


    def velocities(self, swarm=0):
        '''The velocity series of one swarm. See :class:`CaptureSeries`.'''

        return self.array('velocities', swarm)


    def array(self, name, swarm=0):
        '''One named per-agent series of one swarm.

        Parameters
        ----------
        name : str
            'positions', 'velocities', or whatever else this archive stored
        swarm : int or str, default=0
            swarm index, or name if it is unique

        Returns
        -------
        CaptureSeries
        '''

        if name not in self.store:
            raise ValueError(
                "this archive does not hold '{}': it was recorded with "
                "store={}. Re-record with that array included, or work from "
                "what is here.".format(name, list(self.store)))
        return CaptureSeries(self, self._resolve_swarm(swarm), name)


    ####################   validation against an Environment   ####################

    def check_against(self, envir):
        '''Raise unless this archive describes the same domain and fluid.

        The stored positions are bare numbers; nothing in them says what
        coordinate system they are in. Reading them against a different grid
        gives a plausible picture that is silently wrong, so a mismatch is a 
        hard refusal rather than a warning.

        A **provenance** difference is not a mismatch and only warns: replotting
        a run whose script moved directories is not be refused, while a
        different simulation that happens to share a mesh and a cadence is loud.

        Parameters
        ----------
        envir : Environment
        '''

        problems = compare_fingerprints(self.grid, fingerprint_of(envir))
        recorded = (self.meta.get('provenance') or {}).get('fluid')
        current = envir._fluid_provenance

        if problems:
            raise ValueError(
                'this archive does not match this Environment:\n  {}\n'
                'The archive was recorded against {}; this environment\'s fluid '
                'is {}.'.format('\n  '.join(problems),
                                _describe_source(recorded),
                                _describe_source(current)))

        if recorded != current:
            warnings.warn(
                'this archive matches this Environment grid for grid, but was '
                'recorded against {} where this environment\'s fluid is {}. '
                'Two runs on the same mesh at the same cadence are '
                'indistinguishable by grid alone.'.format(
                    _describe_source(recorded), _describe_source(current)),
                UserWarning)


    ####################   internals   ####################

    def _read_roster(self):
        '''The recorded swarms, from their sidecars.

        Scanned rather than read out of meta.json, which is written once at the
        start and so cannot know about a swarm that joined an hour into the run.
        '''

        entries = []
        for sidecar in self._agent_dir.glob('swarm*.json'):
            entries.append(json.loads(sidecar.read_text()))
        entries.sort(key=lambda e: e['index'])
        if not entries:
            raise ValueError(
                '{} holds no swarms; nothing was recorded'.format(self.path))
        return entries


    def _resolve_swarm(self, swarm):
        '''Turn an index or a name into an index, refusing an ambiguous name.'''

        if isinstance(swarm, (int, np.integer)):
            if swarm not in self._by_index:
                raise KeyError('no swarm {} in this archive; it holds {}'.format(
                    swarm, self.swarms))
            return int(swarm)

        matches = [e['index'] for e in self._entries if e['name'] == swarm]
        if not matches:
            raise KeyError("no swarm named '{}' in this archive; it holds "
                           "{}".format(swarm, self.swarms))
        if len(matches) > 1:
            raise KeyError(
                "'{}' names {} swarms in this archive (indices {}), so it does "
                "not identify one. The default Swarm name is 'organism' for "
                "every swarm, which is why files are keyed by index -- address "
                "it by index.".format(swarm, len(matches), matches))
        return matches[0]


    def _chunk_indices(self, prefix):
        '''Chunk indices present for a file prefix, in numeric order.

        Parsed and sorted as integers rather than sorted as names: zero-padding
        agrees with numeric order only up to chunk 9999, past which a lexical
        sort would silently assemble the run out of order.
        '''

        found = []
        for f in self._agent_dir.glob(prefix + '_*.npy'):
            index = _chunk_index_of(f)
            if index is not None:
                found.append(index)
        return sorted(found)


    def _read_times(self):
        '''The capture time base, assembled from the times chunks.

        Small -- one float per capture -- so this is the one thing read whole.
        '''

        if not self._time_chunks:
            return np.empty(0, dtype=DTYPE)
        return np.concatenate([
            np.load(self._agent_dir / _chunk_name('times', i), allow_pickle=False)
            for i in self._time_chunks])


    def _validate_chunks(self):
        '''Refuse an archive whose chunks do not add up.

        Chunks are written in order, so a hard kill costs the *last* buffer and
        never a middle one -- a gap therefore means a lost or corrupt file, not
        an interrupted run, and is refused rather than silently short-read.
        '''

        time_chunks = self._time_chunks
        expected = list(range(len(time_chunks)))
        if time_chunks != expected:
            missing = sorted(set(expected) - set(time_chunks)) or ['(out of order)']
            raise ValueError(
                'this archive is missing time chunk(s) {}. Chunks are written '
                'in order, so a gap is a lost or corrupt file rather than an '
                'interrupted run.'.format(missing))

        n = len(self.times)
        # Every stored array AND the mask, not just the first: a missing
        #   velocity or mask chunk is exactly as fatal as a missing position
        #   one, and checking only positions let it through to surface later as
        #   a bare FileNotFoundError from the middle of a read.
        for entry in self._entries:
            first_chunk = entry['first_capture'] // self._chunk_size
            want = [i for i in range(len(time_chunks)) if i >= first_chunk]
            for short in [STORABLE[name] for name in self.store] + ['mask']:
                prefix = '{}_{}'.format(_swarm_prefix(entry['index']), short)
                got = self._chunk_indices(prefix)
                if got != want:
                    raise ValueError(
                        'swarm {} is missing chunk(s) {} of {}'.format(
                            entry['index'], sorted(set(want) - set(got)), prefix))
                for i in got:
                    rows = len(np.load(self._agent_dir / _chunk_name(prefix, i),
                                       mmap_mode='r'))
                    lo = max(entry['first_capture'], i * self._chunk_size)
                    hi = min(n, (i + 1) * self._chunk_size)
                    if rows != hi - lo:
                        raise ValueError(
                            'swarm {} chunk {} of {} holds {} rows where the '
                            'capture count implies {}; this archive is '
                            'inconsistent'.format(entry['index'], i, prefix,
                                                  rows, hi - lo))


    def _locate(self, capture, first_capture):
        '''Which chunk a global capture index falls in, and where inside it.

        Chunks are keyed on the *global* index, so a swarm that joined mid-run
        needs no second indexing scheme: its first chunk is simply short at the
        front, and its own first capture resolves the offset.
        '''

        chunk = capture // self._chunk_size
        start = max(first_capture, chunk * self._chunk_size)
        return chunk, capture - start


    def _chunk(self, swarm_index, name, chunk):
        '''A memmapped chunk file, from a small cache of open ones.'''

        key = (swarm_index, name, chunk)
        if key not in self._cache:
            short = 'mask' if name == 'mask' else STORABLE[name]
            path = self._agent_dir / _chunk_name(
                '{}_{}'.format(_swarm_prefix(swarm_index), short), chunk)
            if len(self._cache) >= self.CACHE_SIZE:
                # Oldest out. Dicts keep insertion order, so this is FIFO --
                #   which is what a monotone sweep wants.
                del self._cache[next(iter(self._cache))]
            self._cache[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        return self._cache[key]



def load_run(path):
    '''Open a Planktos run archive for reading.

    Parameters
    ----------
    path : str or Path
        an archive directory written by :meth:`planktos.Environment.record`

    Returns
    -------
    RunArchive

    See Also
    --------
    planktos.Environment.record : write one
    '''

    return RunArchive(path)
