'''
Writing a run archive. Reading one is ``planktos.archive``.

Writing is reached only through ``Environment.record``, which builds a
``RunRecorder``. The two writers under it are separate because their cadences
differ: agent state is captured per time step, fluid quantities per fluid dump.

``RunRecorder``
    the handle ``record`` returns, and the only public writing surface. Hooks the
    environment's time advance, discovers swarms at capture time, and owns the
    two writers below. There is one per Environment.
``_ArchiveWriter``
    the agent half, and the format itself. Writes ``meta.json``, ``grid.npz``
    and the per-swarm sidecars up front, then buffers ``chunk_size`` captures at
    a time and writes a chunk. It is handed arrays and knows nothing about
    Environments, Swarms or time steps, so the format can be exercised without
    running a simulation. The directory it writes into comes from
    ``_resolve_archive_path``, which redirects a non-empty target to a
    timestamped sibling rather than overwriting it -- unless ``_appendable``
    says the target is an archive this run continues, in which case it is
    written into as it stands.
``_plan_fluid``
    decides what the fluid half will hold -- which quantities, and where
    vorticity will come from -- *before* the archive directory is resolved, since
    that answer goes into the write-once ``meta.json``.
``_FluidWriter``
    carries that plan out. Driven by an observer registered on the ``FluidData``
    rather than by the step hook, so it sees each dump as it lands and never
    causes a fluid load of its own.

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import copy
import hashlib
import os
import warnings
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd

import planktos
from . import _provenance
# Aliased so that the module is not shadowed by the `fluid=` parameter that
#   _plan_fluid and _normalize_fluid take -- same word, two different things.
from . import fluid as _fluid
from ._archive_format import (DTYPE, FORMAT_VERSION, INDEX_WIDTH, SERIES,
                              STORABLE, TMP_DIRNAME, _arrayable, _as_json,
                              _atomic_write, _checkpoint_names, _chunk_name,
                              _quiver_name, _save_json, _save_npy,
                              _split_props, _swarm_prefix, fingerprint_of,
                              fingerprint_summary)

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"



#############################################################################
#                                                                           #
#                           WHERE A RECORDING GOES                          #
#                                                                           #
#############################################################################

def _resolve_archive_path(path):
    '''Choose the directory to record into, and create it.

    A non-empty directory is left alone and a timestamped sibling used instead.
    The redirect warns, naming the path actually chosen, and the writer exposes
    it as ``.path``.

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

    # Redirect rather than overwrite or refuse: overwriting loses a previous
    #   run, and refusing strands a long job that was ready to start. The
    #   warning is what keeps a later load_run(path) from quietly reading the
    #   earlier run instead.
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


# Where vorticity lives is settled once per archive and then followed, so it is
#   not part of what an append has to match.
_FLUID_DERIVED = ('vorticity', 'vorticity_dir')


def _fluid_request(meta):
    '''The part of a fluid plan that says what was *asked for*, not what follows.'''

    if not meta:
        return meta
    return {k: v for k, v in meta.items() if k not in _FLUID_DERIVED}


def _appendable(path, envir, store, chunk_size, capture_interval, fluid_meta):
    """The archive at ``path`` if this recording continues it, else None.

    Appending is decided from a checkable fact rather than a remembered one:
    **the archive's last capture is exactly where the Environment now is**, and
    the recording is being made the same way.

    Returns
    -------
    RunArchive or None
        open, and the writer's to close

    Raises
    ------
    ValueError
        where the archive lines up in time but was recorded with a different
        ``store``, ``chunk_size``, ``capture_interval`` or set of fluid
        quantities. Quietly starting a second archive is the confusing outcome
        there; a changed ``capture_interval`` makes the timeline unevenly spaced
        half way through; and since ``meta.json`` is not rewritten, a changed
        fluid plan would leave the archive describing one thing and holding
        another.
    """

    # Imported here because archive imports this module, to re-export
    #   RunRecorder.
    from .archive import RunArchive

    path = Path(path)
    if not (path / 'meta.json').is_file():
        # Not an archive at all. A non-empty directory still redirects.
        return None
    try:
        archive = RunArchive(path)
    except (ValueError, FileNotFoundError):
        # Unreadable or inconsistent, so not something to continue -- and not
        #   this call's business to complain about.
        return None

    # A checkable fact beats remembering a restore three ways: it is verifiable
    #   from state; it fails safe, since restoring and then running before
    #   recording leaves the clock past the last capture and a separate archive
    #   is written rather than a series with a hole in it; and it picks up
    #   stop_recording(), a look at the data, and a second record().
    if not len(archive.times) or envir.time != archive.times[-1]:
        archive.close()
        return None

    try:
        archive.check_against(envir)
    except ValueError as err:
        # A different world -- a moved fluid, or one that could not be replayed
        #   at all -- so this is not a continuation of that run and the
        #   configuration of the recording is beside the point. Said out loud,
        #   then handled the way any non-empty directory is.
        warnings.warn(
            '{} ends where this run now is, but does not describe this '
            'Environment, so it is being left alone rather than continued: '
            '{}'.format(path, err), UserWarning)
        archive.close()
        return None

    recorded = archive.meta.get('capture_interval')
    problems = []
    # meta.json is not rewritten, so what this call asks of the fluid has to be
    #   what the archive already records -- new dumps written on a different
    #   quiver grid beside old ones, say. Compared over the *request* only:
    #   where vorticity lives is a derived fact, and a windowed recording that
    #   wrote a partial series into its source changed it, so recomputing the
    #   plan here would refuse the very append that recording was leading to.
    if _as_json(_fluid_request(archive.meta.get('fluid'))) !=             _as_json(_fluid_request(fluid_meta)):
        problems.append('fluid: archive recorded {}, this call {}'.format(
            _fluid_request(archive.meta.get('fluid')),
            _fluid_request(fluid_meta)))
    if tuple(store) != tuple(archive.store):
        problems.append('store: archive has {}, this call {}'.format(
            list(archive.store), list(store)))
    if int(chunk_size) != int(archive.meta['chunk_size']):
        problems.append('chunk_size: archive has {}, this call {}'.format(
            archive.meta['chunk_size'], chunk_size))
    if recorded is not None and int(capture_interval) != int(recorded):
        problems.append('capture_interval: archive has {}, this call {}'.format(
            recorded, capture_interval))
    if problems:
        archive.close()
        raise ValueError(
            '{} ends exactly where this run now is, so this recording would '
            'continue it -- but it was recorded differently: {}. Match those, '
            'or record to a different directory.'.format(
                path, '; '.join(problems)))
    if recorded is None:
        # Written before the interval was recorded, so nothing says the
        #   timeline would stay evenly spaced. Redirect, as it did then.
        archive.close()
        return None

    return archive



#############################################################################
#                                                                           #
#                               THE AGENT HALF                              #
#                                                                           #
#############################################################################

class _ArchiveWriter:
    '''Writes agent captures to an archive directory as a run proceeds.

    The low-level half of recording: it is handed data and writes it, and knows
    nothing about Environments, Swarms, time steps or hooks. ``RunRecorder``
    drives it.

    Metadata is written by ``__init__``, so constructing one of these creates the
    directory and commits the archive's identity. Nothing after that point is
    required for what is already on disk to be readable.

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
    append : RunArchive, optional
        an archive this recording continues rather than replaces, from
        :func:`_appendable`. Given one, ``path`` is used as it stands, nothing
        already written is rewritten but the tail chunk, and the metadata and
        the grid are left alone. Closed here once it has been read back.

    Attributes
    ----------
    path : Path
        the directory actually being written to
    agent_dir : Path
        the ``agents/`` subdirectory, where every per-swarm file goes
    store : tuple of str
        what is kept for each capture, as given
    arrays : tuple of str
        the members of ``store`` that are ``N x D`` arrays, chunked together
    series : tuple of str
        the members of ``store`` that are not arrays -- 'props' and
        'shared_props' -- each written by its own machinery
    derive : bool
        True when velocities are not stored, so the speed statistics and the 2D
        heading angle are derived at capture time instead
    shared : bool
        True when a ``shared_props`` series is being kept
    chunk_size : int
        captures buffered before a chunk is written
    appending : bool
        whether this writer continues an existing archive rather than starting
        one
    '''

    def __init__(self, path, fingerprint, meta=None, chunk_size=100,
                 store=('positions',), append=None):
        if int(chunk_size) < 1:
            raise ValueError('chunk_size must be at least 1')
        self.store = tuple(store)
        unknown = [name for name in self.store
                   if name not in STORABLE and name not in SERIES]
        if unknown:
            raise ValueError('cannot store {}; what can be stored is {}'.format(
                unknown, sorted(set(STORABLE) | set(SERIES))))
        # The N x D arrays, which are chunked together, apart from the series
        #   that are not arrays at all.
        self.arrays = tuple(n for n in self.store if n in STORABLE)
        self.series = tuple(n for n in self.store if n in SERIES)
        if 'positions' not in self.store:
            raise ValueError(
                'positions must be stored: nothing consumes an archive without '
                'them, in or out of plotting')
        self.chunk_size = int(chunk_size)
        self.appending = append is not None
        # The whole point of appending is not to redirect, and the metadata and
        #   the grid are written once at the start of an archive and never
        #   rewritten -- which is as true of the second recording into one as of
        #   the first.
        self.path = (Path(append.path) if self.appending
                     else _resolve_archive_path(path))
        self.agent_dir = self.path / 'agents'
        self.agent_dir.mkdir(exist_ok=True)
        # Component B writes here. Created now so that the layout is complete
        #   from the start and B never has to wonder whether it exists.
        (self.path / 'fluid').mkdir(exist_ok=True)

        self._fingerprint = dict(fingerprint)
        if not self.appending:
            _atomic_write(self.path / 'grid.npz',
                          lambda fobj: np.savez(fobj, **self._fingerprint))

            record = dict(meta) if meta else {}
            record.update(version=FORMAT_VERSION,
                          dtype=str(DTYPE),
                          chunk_size=self.chunk_size,
                          store=list(self.store),
                          grid=fingerprint_summary(self._fingerprint))
            _save_json(self.path / 'meta.json', record)

        # What velocities were needed for, recorded exactly when velocities are
        #   not: the agent-speed statistics a plot prints, and the heading angle
        #   a 2D frame draws. Both are derived at capture time, when the
        #   velocity is in hand; neither is recoverable from stored positions
        #   (section 5.1), which is why they are stored rather than re-derived.
        self.derive = 'velocities' not in self.store
        # shared_props is O(T) and has nothing to do with whether the O(N.T)
        #   per-agent DataFrame was wanted, so it is its own token. It shares
        #   the sidecar the statistics already write on the same cadence.
        self.shared = 'shared_props' in self.series

        # index -> {'name', 'N', 'D', 'first_capture'}
        self._swarms = {}
        # buffers for the chunk currently being filled
        self._chunk = None              # which chunk index that is
        self._times = []
        # swarm index -> {'positions': [...], 'velocities': [...], 'mask': [...]}
        self._buffers = {}
        self._next_capture = None       # the capture index expected next
        # index -> {'avg_vel': [...], 'avg_spd': [...], 'std_spd': [...]}
        self._stats = {}
        # index -> {'columns': {key: [...]}, 'shapes': {key: shape},
        #           'n': int, 'dropped': set}
        self._shared = {}
        self._closed = False
        if self.appending:
            self._seed_from(append)
            append.close()


    @property
    def next_capture(self):
        '''The capture index this writer expects next. 0 for a new archive.'''

        return 0 if self._next_capture is None else self._next_capture


    ####################   recording   ####################

    def _seed_from(self, archive):
        '''Pick up an existing archive so this recording continues it.

        Everything the writer would otherwise build from capture 0: the roster,
        the capture counter, the tail chunk's buffers, and the whole of each
        per-swarm series. Nothing already on disk is rewritten except the tail
        chunk, which is the one piece that has to grow -- and the series files,
        which are rewritten whole by design and so must be read back first or
        an append silently keeps only the appended stretch.
        '''

        n = len(archive.times)
        self._next_capture = n
        # Uniformly the chunk the next capture falls in: when the tail chunk is
        #   full this is a new one and there is nothing to read back.
        self._chunk = n // self.chunk_size
        partial = n % self.chunk_size

        for entry in archive._entries:
            index = entry['index']
            self._swarms[index] = dict(entry)
            self._buffers[index] = {name: [] for name in self.arrays}
            self._buffers[index]['mask'] = []
            if 'props' in self.series:
                self._buffers[index]['props'] = []
            if self.derive and entry['D'] == 2:
                self._buffers[index]['angle'] = []
            if self.derive:
                self._stats[index] = {'avg_vel': [], 'avg_spd': [],
                                      'std_spd': []}
            if self.shared:
                self._shared[index] = {'columns': {}, 'shapes': {}, 'n': 0,
                                       'dropped': set()}
            self._seed_series(archive, index)

        if not partial:
            return
        self._times = [float(t) for t in archive.times[n - partial:]]
        for entry in archive._entries:
            self._seed_chunk(archive, entry, self._chunk)


    def _seed_chunk(self, archive, entry, chunk):
        '''Read one swarm's tail chunk back into the buffers it was written from.

        _validate_chunks requires chunk i to hold exactly the rows the capture
        count implies, so a short chunk left in the middle of a series is a
        refusal. Bounded by one chunk.
        '''

        index = entry['index']
        buffers = self._buffers[index]
        rows = [j for j in range(max(entry['first_capture'], chunk*self.chunk_size),
                                len(archive.times))]
        if not rows:
            # The swarm joined after this chunk started and has not been
            #   captured in it, which is the case _write_chunk skips writing.
            return
        for name in self.arrays:
            series = archive.array(name, index)
            for j in rows:
                buffers[name].append(np.asarray(ma.getdata(series[j]), DTYPE))
        mask = archive.positions(index)
        for j in rows:
            buffers['mask'].append(ma.getmaskarray(mask[j])[:, 0].copy())
        if 'angle' in buffers:
            angles = archive.angles(index)
            for j in rows:
                buffers['angle'].append(
                    np.asarray(ma.getdata(angles[j]), np.float32))
        if 'props' in buffers:
            frames = archive.props(index) or []
            first = entry['first_capture']
            for j in rows:
                buffers['props'].append(frames[j - first])


    def _seed_series(self, archive, index):
        '''Read one swarm's whole per-capture sidecar back into its accumulators.

        Whole, not the tail: the file is rewritten entire on every flush, so an
        append that starts these empty leaves an archive covering the appended
        stretch only -- and silently, since the file is well formed either way.
        '''

        if self.derive:
            stats = archive.agent_stats(index)
            if stats is None:
                self._missing_series(archive, index, 'speed statistics')
            else:
                acc = self._stats[index]
                acc['avg_vel'] = [np.asarray(v) for v in stats['avg_vel']]
                acc['avg_spd'] = [float(v) for v in stats['avg_spd']]
                acc['std_spd'] = [float(v) for v in stats['std_spd']]
        if not self.shared:
            return
        columns = archive.shared_props(index)
        if columns is None:
            self._missing_series(archive, index, 'shared_props series')
            return
        acc = self._shared[index]
        for name, column in columns.items():
            absent = ma.getmaskarray(column)
            values = ma.getdata(column)
            # None back in the padded slots, which is what _stack_shared reads;
            #   a fill value would come back as a real one.
            acc['columns'][name] = [
                None if np.all(absent[j]) else np.array(values[j])
                for j in range(len(column))]
            present = [v for v in acc['columns'][name] if v is not None]
            if present:
                acc['shapes'][name] = present[0].shape
        acc['n'] = len(archive.times) - archive.first_capture(index)


    @staticmethod
    def _missing_series(archive, index, what):
        '''Say so when a series this archive should hold is not there to seed.

        It is rewritten whole, so what cannot be read back cannot be carried
        forward: the file will end up covering the appended stretch only. That
        is well formed either way, which is exactly why it has to be said.
        '''

        warnings.warn(
            'swarm {} has no {} in {}, so it cannot be carried forward and the '
            'file will cover only the stretch being appended now.'.format(
                index, what, archive.path), UserWarning)


    def add_swarm(self, index, name, N, D, first_capture):
        '''Register a swarm, writing its sidecar immediately.

        The roster lives in per-swarm files rather than in ``meta.json``, so a
        swarm joining mid-run is an ordinary case: nothing already written is
        touched, and ``first_capture`` is all that distinguishes it.

        Parameters
        ----------
        index : int
            position in the recorder's swarm list; fixed for the run
        name : str
            the Swarm's name. Not used in filenames, since the default name is
            'organism' for every Swarm and two swarms would collide.
        N, D : int
            agent count and spatial dimension
        first_capture : int
            global capture index at which this swarm starts. 0 for every swarm
            present when recording began.
        '''

        index = int(index)
        if index in self._swarms:
            if not self.appending:
                raise ValueError(
                    'swarm index {} is already recorded'.format(index))
            # Continuing an archive, so the roster is validated rather than
            #   added to. The shape is what the chunk files are written at and
            #   the one thing that cannot differ; the name lives in
            #   shared_props and is free to have changed.
            entry = self._swarms[index]
            if (int(N), int(D)) != (entry['N'], entry['D']):
                raise ValueError(
                    'swarm {} is {} agents in {}D, where {} recorded it as {} '
                    'in {}D. An archive is chunked at a fixed shape, so it '
                    'cannot be continued at another one.'.format(
                        index, N, D, self.path, entry['N'], entry['D']))
            return
        entry = {'index': index, 'name': str(name), 'N': int(N), 'D': int(D),
                 'first_capture': int(first_capture)}
        self._swarms[index] = entry
        _save_json(self.agent_dir / (_swarm_prefix(index) + '.json'), entry)
        self._buffers[index] = {name: [] for name in self.arrays}
        self._buffers[index]['mask'] = []
        if 'props' in self.series:
            self._buffers[index]['props'] = []
        if self.derive:
            # Headings are 2D-only, since a 3D frame draws none.
            if int(D) == 2:
                self._buffers[index]['angle'] = []
            # Three floats per capture whatever N is, so this accumulates in
            #   memory and is rewritten whole rather than chunked.
            self._stats[index] = {'avg_vel': [], 'avg_spd': [], 'std_spd': []}
        if self.shared:
            self._shared[index] = {'columns': {}, 'shapes': {}, 'n': 0,
                                   'dropped': set()}


    def add_capture(self, capture_index, time, arrays, velocities=None,
                    props=None, shared=None):
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
        velocities : dict, optional
            swarm index -> the ``N x D`` masked velocities of that capture, for
            deriving what an archive without them still has to be able to draw.
            Required when ``derive`` is set, ignored otherwise.
        props : dict, optional
            swarm index -> that capture's props DataFrame. Required when
            ``store`` names 'props', ignored otherwise.
        shared : dict, optional
            swarm index -> that capture's ``shared_props`` dict. Required when
            ``store`` names 'shared_props', ignored otherwise.
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
            if set(named) != set(self.arrays):
                raise ValueError(
                    'swarm {} supplied {}, but this archive stores {}'.format(
                        idx, sorted(named), sorted(self.arrays)))

            # Positions first: one mask is stored per capture, and it is theirs.
            #   A masked row means the agent is not in the domain, which is a
            #   fact about the agent rather than about any one array, so every
            #   other array must agree -- and a disagreement is refused rather
            #   than silently dropped, since the format has nowhere to put it.
            reference = None
            for name in ['positions'] + [n for n in self.arrays
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

            if self.derive:
                self._derive(idx, entry, velocities[idx])
            if 'props' in self.series:
                self._buffers[idx]['props'].append(props[idx])
            if self.shared:
                self._record_shared(idx, shared[idx])

        self._next_capture += 1


    def _derive(self, idx, entry, velocity):
        '''Buffer the heading and the speed statistics of one capture.'''

        data = np.asarray(ma.getdata(velocity), dtype=DTYPE)
        if entry['D'] == 2:
            # Defined even at (0,0) by convention, and float32 because the
            #   error there is ~1e-7 rad against a heading marker whose
            #   shortest visible increment is nearer 1e-2.
            self._buffers[idx]['angle'].append(
                np.arctan2(data[:, 1], data[:, 0]).astype(np.float32))

        # Only agents still in the domain contribute, which is what
        #   _calc_basic_stats does with the same numbers.
        present = data[~ma.getmaskarray(velocity).any(axis=1)]
        if present.size == 0:
            present = np.zeros((1, entry['D']))
        speed = np.linalg.norm(present, axis=1)
        stats = self._stats[idx]
        stats['avg_vel'].append(present.mean(axis=0))
        stats['avg_spd'].append(float(speed.mean()))
        stats['std_spd'].append(float(speed.std()))


    def _record_shared(self, idx, shared):
        '''Buffer one capture of a swarm's ``shared_props``.

        A column per key, so the accumulator is O(T) whatever the swarm size.
        ``shared_props`` is an ordinary mutable dict, which gives three cases:
        a key that appears mid-run gets a padded slot for the captures before
        it, a key that vanishes gets one for the captures after, and a key
        whose value changes shape cannot be stacked at all and is refused by
        name.
        '''

        acc = self._shared[idx]
        columns, shapes, n = acc['columns'], acc['shapes'], acc['n']
        for name, value in shared.items():
            array = _arrayable(value)
            if array is None:
                # Warned and dropped, as the checkpoint drops the same value,
                #   and once per key rather than once per capture.
                if name not in acc['dropped']:
                    acc['dropped'].add(name)
                    warnings.warn(
                        "swarm {} shared_props '{}' cannot be stored without "
                        "pickle, so no series of it is being kept. Carry that "
                        "value in props instead if a restart needs "
                        "it.".format(idx, name), UserWarning)
                continue
            # Copied, not referenced: the columns outlive the capture, and
            #   np.asarray hands back the caller's own buffer for an ndarray,
            #   so a shared_props value edited in place would rewrite every
            #   capture already recorded. Same copy the props series makes.
            array = np.array(array)
            if name not in columns:
                columns[name] = [None] * n
                shapes[name] = array.shape
            elif array.shape != shapes[name]:
                raise ValueError(
                    "swarm {} shared_props '{}' changed shape from {} to {} "
                    "during the run. A series of it cannot be stacked, so it "
                    "is refused rather than written in a form that will not "
                    "read back.".format(idx, name, shapes[name], array.shape))
            columns[name].append(array)
        for column in columns.values():
            if len(column) == n:
                column.append(None)
        acc['n'] = n + 1


    def _stack_shared(self, idx, acc):
        '''One swarm's ``shared_props`` columns, stacked with their padding.

        ``present__<key>`` rides beside ``shared__<key>`` only for a key that
        was not there at every capture, so a ``shared_props`` whose membership
        never changes writes nothing extra.
        '''

        arrays = {}
        for name, column in acc['columns'].items():
            present = np.array([value is not None for value in column])
            try:
                stacked = np.stack([v for v in column if v is not None])
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "swarm {} shared_props '{}' cannot be stacked into a "
                    "series: {}".format(idx, name, err))
            if present.all():
                arrays['shared__' + name] = stacked
                continue
            whole = np.zeros((len(column),) + stacked.shape[1:], stacked.dtype)
            if whole.dtype.kind == 'f':
                # NaN in a padded slot, so raw np.load of the file reads it as
                #   absent rather than as a real zero.
                whole.fill(np.nan)
            whole[present] = stacked
            arrays['shared__' + name] = whole
            arrays['present__' + name] = present
        return arrays


    def _first_capture_of(self, idx, chunk):
        """The global capture index the given chunk starts at, for a swarm."""

        return max(self._swarms[idx]['first_capture'], chunk * self.chunk_size)


    def _write_props(self, prefix, index, frames, first_capture):
        """Write one chunk's props, long format: a row per agent per capture.

        Readable and exact, at pandas' default float format, which already
        writes the shortest text that reads back bit for bit. A column holding
        one ndarray per agent cannot go in text at all -- it renders as a broken
        multi-line row -- so those spill to their own .npy beside it, as the
        checkpoint does.
        """

        rows, spills = [], {}
        for offset, props in enumerate(frames):
            frame, arrays = _split_props(props)
            frame = frame.copy()
            frame.insert(0, 'agent', np.arange(len(frame)))
            frame.insert(0, 'capture', first_capture + offset)
            rows.append(frame)
            for name, array in arrays.items():
                spills.setdefault(name, []).append(array)

        table = pd.concat(rows, ignore_index=True)
        _atomic_write(
            self.agent_dir / '{}_props_{:0{}d}.csv'.format(
                prefix, index, INDEX_WIDTH),
            lambda fobj: fobj.write(table.to_csv(index=False).encode('utf-8')))
        for name, stacked in spills.items():
            _save_npy(self.agent_dir / _chunk_name(
                '{}_prop-{}'.format(prefix, name), index), np.stack(stacked))


    @staticmethod
    def _split(array, entry, label):
        '''Separate an N x D masked array into plain data and a row mask.

        A masked row means the agent has left the domain -- agents leave whole
        rows, never single coordinates -- so the mask is stored per row rather
        than per element, and a partially masked row is refused.
        '''

        expected = (entry['N'], entry['D'])
        if np.shape(array) != expected:
            raise ValueError('swarm {} {} has shape {}, expected {}'.format(
                entry['index'], label, np.shape(array), expected))

        mask = ma.getmaskarray(array)
        any_masked = mask.any(axis=1)
        # Refused rather than flattened: a partial row means an invariant broke
        #   upstream, and rounding it off on the way to disk would hide that.
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


    def _write_series(self):
        '''Rewrite each swarm's per-capture sidecar, whole.

        Holds the speed statistics and any ``shared_props`` series. Called from
        :meth:`_write_chunk`, so it lands on the same boundary as everything
        else and covers exactly the captures the chunks do.
        '''

        # A few values per capture whatever the swarm size -- a few tens of kB
        #   over a long run, small enough to rewrite atomically rather than
        #   chunk, as dump_stats.npz does. The accumulators grow per capture and
        #   a chunk closes before the capture that rolled it over is buffered,
        #   which is what makes the coverage line up.
        for idx in sorted(set(self._stats) | set(self._shared)):
            arrays = {}
            stats = self._stats.get(idx)
            if stats and stats['avg_spd']:
                arrays.update(
                    avg_vel=np.stack(stats['avg_vel']).astype(DTYPE),
                    avg_spd=np.asarray(stats['avg_spd'], dtype=DTYPE),
                    std_spd=np.asarray(stats['std_spd'], dtype=DTYPE))
            acc = self._shared.get(idx)
            if acc and acc['n']:
                arrays.update(self._stack_shared(idx, acc))
            if not arrays:
                continue
            _atomic_write(
                self.agent_dir / (_swarm_prefix(idx) + '_series.npz'),
                lambda fobj, a=arrays: np.savez(fobj, **a))


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


    def write_checkpoint(self, index, capture, time, state):
        '''Rewrite one swarm's checkpoint, whole.

        Where the chunk files are append-only history, this is the one latest
        state plus everything history cannot give you -- the random stream, the
        properties, the class whose ``apply_agent_model`` is the behavior. Each
        of the three files is replaced atomically.

        Parameters
        ----------
        index : int
            swarm index, as registered with :meth:`add_swarm`
        capture : int
            the capture index this state was taken at
        time : float
            the environment time at that capture
        state : dict
            ``positions``, ``velocities``, ``accelerations``,
            ``ib_collision_idx``, ``props``, ``shared_props``, ``ib_condition``
            and ``swarm_class``
        '''

        npz_name, csv_name, json_name = _checkpoint_names(index)
        arrays, dropped = {}, []

        # The mask is per row -- agents leave whole rows -- and is positions'.
        positions = state['positions']
        arrays['mask'] = np.asarray(
            ma.getmaskarray(positions)[:, 0], dtype=bool, order='C')
        # order='C' throughout: .npy records the memory order in its header, so
        #   the same values in a Fortran-ordered array write different bytes,
        #   and a Swarm's arrays are C-ordered fresh but can come back either
        #   way from a restore. One recording of a run should not differ from
        #   another for that. Not ascontiguousarray, which promotes a 0-d array
        #   to shape (1,) and so would turn a scalar shared prop into a
        #   one-element array on the way back.
        for name in ('positions', 'velocities', 'accelerations'):
            value = state.get(name)
            if value is not None:
                arrays[name] = np.asarray(ma.getdata(value), dtype=DTYPE,
                                          order='C')
        if state.get('ib_collision_idx') is not None:
            arrays['ib_collision_idx'] = np.asarray(
                state['ib_collision_idx'], dtype=np.int64, order='C')

        # shared_props follows Swarm.save_data's precedent of an npz: it is a
        #   mixture of scalars and arrays, and npz takes both without pickle.
        for name, value in (state.get('shared_props') or {}).items():
            array = _arrayable(value)
            if array is None:
                dropped.append(name)
            else:
                arrays['shared__' + name] = np.asarray(array, order='C')
        if dropped:
            warnings.warn(
                'shared_props {} cannot be stored without pickle and are not '
                'being checkpointed. A restart will come back without '
                'them.'.format(sorted(dropped)), UserWarning)

        frame, prop_arrays = _split_props(state['props'])
        for name, array in prop_arrays.items():
            arrays['prop__' + name] = np.asarray(array, order='C')

        _atomic_write(self.agent_dir / npz_name,
                      lambda fobj: np.savez(fobj, **arrays))
        # Default float format, not '%.17g': pandas already writes the shortest
        #   representation that reads back exactly. Readers must pass
        #   float_precision='round_trip', which is where the loss would be.
        #   A swarm with no scalar props writes no file at all -- an empty one
        #   has not even a header row, and pandas cannot parse that back.
        if len(frame.columns):
            _atomic_write(self.agent_dir / csv_name,
                          lambda fobj: fobj.write(
                              frame.to_csv(index=False).encode('utf-8')))
        _save_json(self.agent_dir / json_name, {
            'index': int(index),
            'capture': int(capture),
            'time': float(time),
            'ib_condition': state.get('ib_condition'),
            'swarm_class': state.get('swarm_class'),
            'props_csv': list(frame.columns),
            'props_npz': sorted(prop_arrays),
            'shared_props': sorted(
                name for name in (state.get('shared_props') or {})
                if name not in dropped),
            'shared_props_dropped': sorted(dropped),
            'rndState': _provenance.jsonable(state.get('rndState'))})


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
            for name in self.arrays:
                _save_npy(
                    self.agent_dir / _chunk_name(
                        '{}_{}'.format(prefix, STORABLE[name]), index),
                    np.stack(buffers[name]))
            if 'props' in buffers:
                self._write_props(prefix, index, buffers['props'],
                                  self._first_capture_of(idx, index))
            _save_npy(self.agent_dir / _chunk_name(prefix + '_mask', index),
                      np.stack(buffers['mask']))
            if 'angle' in buffers:
                _save_npy(self.agent_dir / _chunk_name(prefix + '_ang', index),
                          np.stack(buffers['angle']))

        self._write_series()

        if not keep:
            self._times = []
            for buffers in self._buffers.values():
                for name in buffers:
                    buffers[name] = []



#############################################################################
#                                                                           #
#                               THE FLUID HALF                              #
#                                                                           #
#############################################################################

# The fluid half of an archive (docs/notes/run_persistence.md section 3). What
#   it exists for: never re-stream a 100 GB dataset to draw a picture of it. A
#   render needs three things from the fluid, and each is handled by its own rule
#   rather than by one mechanism, because the regimes genuinely differ:
#
#   frame statistics  the spatial mean of each velocity component, per dump.
#                     A few floats. Always written, in 2D and 3D alike -- this
#                     is the only part of component B a 3D run gets, and it is
#                     what lets the statistics box be drawn from an archive.
#   vorticity         sourced by regime (section 3.3): recomputed when the whole
#                     field is resident, read from the source when it ships one,
#                     written per dump only when neither holds. 2D only.
#   quiver            a downsampled subsample of the velocity, opt-in, written
#                     per dump when asked for. 2D only.
#
# Nothing here holds field data between dumps. A dump arrives, what is wanted
#   from it is derived and written, and it goes out of scope again -- the same
#   discipline the velocity window itself keeps.

# Target arrow grid for a stored quiver, if the caller does not say. Resolved
#   against the fluid grid into integer strides at record() time, because the
#   figure size that plot_all normally derives them from does not exist while a
#   simulation is running.
QUIVER_SHAPE = (60, 60)

# Which quantities `fluid=` may ask for.
FLUID_QUANTITIES = ('vort', 'quiver')


# What _plan_fluid decides and _FluidWriter carries out. A named structure
#   rather than a dict, so absence-means-something is not knowledge the writer
#   has to hold, and `vorticity_dir` means one thing: where per-dump vorticity
#   lives, or None for the archive's own fluid/.
_FluidPlan = namedtuple('_FluidPlan', 'quantities quiver_strides '
                                      'write_vorticity vorticity_dir meta')


class _FluidWriter:
    '''Derives and writes per-dump fluid quantities as dumps arrive.

    Driven by an observer registered on the ``FluidData``, so every derivation
    happens on data that is already resident and no dump is ever loaded on this
    class's behalf.

    Parameters
    ----------
    envir : Environment
    fluid_dir : Path
        the archive's ``fluid/`` directory
    plan : _FluidPlan
        from :func:`_plan_fluid`
    seed : dict, optional
        an existing ``dump_stats.npz``, when this recording continues an
        archive. The extrema combine by one ``max``; the per-dump means are
        filled in wherever this process has not loaded that dump; and the dumps
        it names are taken as already recorded, so their vorticity and quiver
        are not written a second time.
    '''

    # Dumps recorded before the statistics sidecar is rewritten. It is rewritten
    #   whole each time -- it is a few floats per dump, and one atomic replace is
    #   what keeps it always readable -- so doing that per arrival would cost
    #   O(n^2) bytes over a long series. Throttling leaves the same exposure to a
    #   kill that the agent chunks already have.
    STATS_INTERVAL = 100

    def __init__(self, envir, fluid_dir, plan, seed=None):
        self.dir = Path(fluid_dir)
        self.plan = plan
        self.quantities = plan.quantities
        self.flow = envir.flow
        self._written = set()

        # Running extrema over the dumps seen, not one row per dump: both are
        #   read once, reduced over the whole run, and never indexed by dump.
        #   NaN is the starting value rather than zero so that "no dump has
        #   arrived" stays distinguishable from "the field is genuinely still",
        #   which _vorticity_norm draws differently. Means are not here at all:
        #   they are interpolated per frame, so they have to stay per dump, and
        #   FluidData caches them already.
        ncomp = len(self.flow)
        self._vmax = np.full(ncomp, np.nan)
        self._vort_absmax = np.nan
        self._unwritten = 0

        # flow_times is fixed for the object's life, so convert it once rather
        #   than on every sidecar rewrite.
        self._flow_times = (None if self.flow.flow_times is None
                            else np.asarray(self.flow.flow_times, dtype=DTYPE))

        # The digest each written vorticity file carries in its stamp, per dump,
        #   so that a render can tell when a later recording has rewritten one
        #   from a different field. '' is a dump not yet seen, or one whose file
        #   Planktos did not write. Only a writing regime has files to vouch for,
        #   and it is always windowed, so flow_times is never None here.
        self._digests = (np.full(len(self.flow.flow_times), '', dtype='U32')
                         if plan.write_vorticity else None)
        self._warned_stale = False

        # Continuing an archive: everything the earlier recording derived, so
        #   that what it wrote is neither lost nor written again. Before the
        #   resident sweep below, which must see the seeded _written.
        self._seed_means = None
        if seed is not None:
            self._seed(seed)

        # Only one of the three regimes writes: a source that already ships the
        #   field is read, and a resident field is recomputed at render. This is
        #   the single place the fluid is told where its per-dump vorticity is,
        #   whichever regime applies.
        self._writes_vorticity = plan.write_vorticity
        self._vort_dir = Path(plan.vorticity_dir or self.dir)
        if 'vort' in self.quantities and self.flow.is_windowed:
            self.flow.vorticity_path = self._vort_dir

        # Sweep first, register second: nothing is hooked while anything can
        #   fail, so a construction that raises leaves no observer behind.
        for t_idx, field in self.flow.iter_resident_dumps():
            self._record_dump(t_idx, field)
        if self.flow.flow_times is not None:
            # A time-invariant field has no dumps to arrive, and the sweep above
            #   already took all of it.
            self.flow.add_dump_observer(self._observe)
        self.flush()


    def _seed(self, stats):
        '''Pick up what an earlier recording into this archive derived.'''

        stored = stats.get('means')
        if stored is not None and stored.shape == self.flow.dump_means.shape:
            self._seed_means = np.asarray(stored, dtype=DTYPE)
            # A row that is not NaN is a dump the earlier recording had in
            #   memory, which is exactly what _written means.
            self._written = {int(i) for i in
                             np.flatnonzero(~np.isnan(stored).any(axis=1))}
        vmax = stats.get('vmax')
        if vmax is not None and np.shape(vmax) == np.shape(self._vmax):
            self._vmax = np.fmax(self._vmax, vmax)
        absmax = stats.get('vort_absmax')
        if absmax is not None:
            self._vort_absmax = float(np.fmax(self._vort_absmax, absmax))
        digests = stats.get('vort_digest')
        if (self._digests is not None and digests is not None
                and digests.shape == self._digests.shape):
            self._digests = digests.astype(self._digests.dtype)


    ####################   lifecycle   ####################

    def stop(self):
        '''Unhook from the fluid and write the sidecar a last time.'''

        self.flow.remove_dump_observer(self._observe)
        self.flush()
        self._clear_staging()


    def flush(self):
        '''Rewrite the per-dump statistics sidecar.'''

        if not self._unwritten:
            return
        means = self.flow.dump_means
        if self._seed_means is not None:
            # A dump this process never loaded keeps the mean the earlier
            #   recording took of it; both are the mean of the same dump.
            means = np.where(np.isnan(means), self._seed_means, means)
        arrays = {'means': means, 'vmax': self._vmax}
        if 'vort' in self.quantities:
            arrays['vort_absmax'] = np.asarray(self._vort_absmax)
        if self._digests is not None:
            arrays['vort_digest'] = self._digests
        if self._flow_times is not None:
            arrays['flow_times'] = self._flow_times
        _atomic_write(self.dir / 'dump_stats.npz',
                      lambda fobj: np.savez(fobj, **arrays))
        self._unwritten = 0


    ####################   the observer   ####################

    def _observe(self, idx_start, flow):
        '''One or more dumps just landed in memory, starting at ``idx_start``.'''

        for i in range(len(flow[0])):
            self._record_dump(idx_start + i, [f[i] for f in flow])
        if self._unwritten >= self.STATS_INTERVAL:
            self.flush()


    def _record_dump(self, t_idx, field):
        '''Derive and write everything wanted from one dump.

        Parameters
        ----------
        t_idx : int
            index into ``flow_times``
        field : list of ndarrays
            one velocity component, single-time, indexed [x,y(,z)]
        '''

        if t_idx in self._written:
            # A re-report, which the jump-to-start slide does by design.
            return
        self._written.add(t_idx)
        self._unwritten += 1

        # np.fmax over np.maximum: it takes the number rather than propagating
        #   the NaN that marks "nothing seen yet".
        self._vmax = np.fmax(self._vmax, [np.max(f) for f in field])

        if 'vort' in self.quantities:
            # From the raw arrays, never through get_vorticity(time=), which calls
            #   the FluidData and can therefore trigger a load.
            vort = _fluid._vorticity_from_field(field, self.flow.flow_points,
                                                self.flow.periodic_dim)
            self._vort_absmax = float(
                np.fmax(self._vort_absmax, np.max(np.abs(vort))))
            if self._writes_vorticity:
                self._digests[t_idx] = self._write_vorticity(
                    t_idx, vort, _fluid._vorticity_digest(vort))

        if 'quiver' in self.quantities:
            self._write_quiver(t_idx, field)


    ####################   the two writes   ####################

    def _write_vorticity(self, t_idx, vort, digest):
        '''Write one dump's vorticity, unless an up-to-date file is there.

        A file Planktos stamped with ``digest`` is left as it is, and one it
        stamped with any other digest is rewritten, with a ``UserWarning`` the
        first time per recording. A file without the stamp is never touched.
        The file appears complete or not at all, like every other file in an
        archive.

        Returns
        -------
        str
            the digest the file in place now carries, or '' for an unstamped one
        '''

        name = self.flow.vorticity_filename(t_idx)
        target = self._vort_dir / name
        if target.exists():
            stamp = _fluid._read_vorticity_stamp(target)
            # Unstamped is the solver's own field, which is at least as good as
            #   a recomputation and better at the domain edge.
            if stamp is None:
                return ''
            if stamp[1] == digest:
                return digest
            self._note_stale(stamp[0])

        # The per-source writer names its own file, so it cannot be given a
        #   temporary name -- it gets a temporary directory on the same
        #   filesystem instead, and the result is moved into place.
        staging = self._staging_dir()
        staging.mkdir(exist_ok=True)
        self.flow.write_dump_vorticity(t_idx, vort, staging)
        # fsync before the rename, for the same reason _atomic_write does it:
        #   os.replace survives process death on its own, but not the loss of
        #   the page cache to a node failure.
        with open(staging / name, 'rb+') as fobj:
            os.fsync(fobj.fileno())
        os.replace(staging / name, target)
        return digest


    def _note_stale(self, version):
        '''Say, once per recording, that stamped vorticity files are being
        rewritten, and which Planktos version wrote them if it is not this one.'''

        if self._warned_stale:
            return
        self._warned_stale = True
        warnings.warn(
            "rewriting the vorticity in {}: those files were computed from a "
            "different velocity field (another vel_conv, vec_name or "
            "periodic_dim, or changed source data). A recording that relied on "
            "them can no longer draw its vorticity backdrop.".format(
                self._vort_dir), UserWarning)
        if version != planktos.__version__:
            print("The vorticity files being replaced were written by Planktos "
                  "{}; this is Planktos {}.".format(version,
                                                    planktos.__version__))


    def _write_quiver(self, t_idx, field):
        '''Write one dump's downsampled velocity, as .npy in the archive.'''

        # .npy rather than vtk: a subsample chosen at record time is not a
        #   quantity any solver writes or any other tool would want.
        M, N = self.plan.quiver_strides
        arrows = np.stack([np.asarray(f)[::M, ::N] for f in field])
        _save_npy(self.dir / _quiver_name(t_idx), arrows)


    def _staging_dir(self):
        '''Where a per-dump field is written before being moved into place.'''

        return self._vort_dir / TMP_DIRNAME


    def _clear_staging(self):
        '''Remove the staging directory, and anything a failed write left in it.

        Best-effort: an archive is not made invalid by a leftover, and raising
        here would mask whatever went wrong upstream.
        '''

        staging = self._staging_dir()
        try:
            for leftover in staging.iterdir():
                leftover.unlink()
            staging.rmdir()
        except OSError:
            pass


def _plan_fluid(envir, fluid, quiver_shape):
    """Decide what the fluid half of an archive will contain, before it exists.

    Parameters
    ----------
    envir : Environment
    fluid : str, tuple of str, or None
    quiver_shape : tuple of 2 int

    Returns
    -------
    _FluidPlan, or None if the environment has no fluid at all. Its ``meta``
    field is the block that goes into meta.json.
    """

    # Separate from _FluidWriter because the answer goes into meta.json, which is
    #   written before the archive directory is resolved and never rewritten.
    flow = envir.flow
    if flow is None:
        return None
    # A plan is returned for any fluid at all, including in 3D where quantities
    #   is empty: the statistics sidecar is what lets the plot statistics box be
    #   served from an archive, and it is a few floats per dump either way.
    quantities = _normalize_fluid(fluid, envir)
    meta = {'quantities': list(quantities)}

    strides = None
    if 'quiver' in quantities:
        # Normalized here rather than left to jsonable, which would record an
        #   ndarray's shape instead of its values -- right for a data array, and
        #   useless for a two-element parameter.
        quiver_shape = tuple(int(v) for v in quiver_shape)
        strides = _quiver_strides(flow.flow_points, quiver_shape)
        meta['quiver_shape'] = quiver_shape
        meta['quiver_strides'] = strides
        meta['quiver_grid'] = [len(flow.flow_points[d][::strides[d]])
                               for d in range(len(strides))]

    state, directory, write = 'recomputed', None, False
    if 'vort' in quantities:
        state, directory, write = _plan_vorticity(flow)
        meta['vorticity'] = state
        # The archive case records no path: it is the archive's own fluid/
        #   directory, which a reader already knows.
        meta['vorticity_dir'] = directory

    return _FluidPlan(quantities=quantities, quiver_strides=strides,
                      write_vorticity=write, vorticity_dir=directory,
                      meta=meta)


def _normalize_fluid(fluid, envir):
    '''Resolve the ``fluid=`` argument to a tuple of quantities.

    Empty in 3D, where no fluid backdrop is drawn. A flow-free environment is
    handled by :func:`_plan_fluid`, which records nothing fluid at all.
    '''

    if fluid is None:
        return ()
    if isinstance(fluid, str):
        fluid = (fluid,)
    fluid = tuple(fluid)
    unknown = [q for q in fluid if q not in FLUID_QUANTITIES]
    if unknown:
        raise ValueError(
            'cannot record fluid quantity {}; known quantities are {}, a tuple '
            'of them, or None'.format(unknown, list(FLUID_QUANTITIES)))
    # Silent rather than a warning: there is nothing else the caller could have
    #   meant, and 'vort' is the default, so it would fire on every 3D run.
    if len(envir.L) == 3:
        return ()
    return fluid


def _quiver_strides(flow_points, quiver_shape):
    '''Integer strides that subsample a fluid grid to about ``quiver_shape``.

    At least 1 in each direction, since more arrows than grid points cannot be
    honoured.
    '''

    # max(1, ...) on both: a stride of 0 would raise deep inside a slice.
    return tuple(max(1, int(round(len(flow_points[d]) / max(1, quiver_shape[d]))))
                 for d in range(len(flow_points)))


def _plan_vorticity(flow):
    """Which of the three vorticity regimes applies, flat: one return each.

    Returns
    -------
    state : {'recomputed', 'source', 'archive'}
        what a reader should do, and where it should look
    directory : Path or None
        where the per-dump field lives; None means the archive's own ``fluid/``
    write : bool
        whether Planktos has to produce the field itself
    """

    if not flow.is_windowed:
        # The whole field is in memory, so a render recomputes the curl -- which
        #   is cheaper than reading one back, and writes nothing.
        return 'recomputed', None, False

    state, directory = flow.probe_stored_vorticity()
    if state == 'complete':
        return 'source', Path(directory).resolve(), False

    if state == 'partial':
        # Write elsewhere rather than filling the gaps in place, so that what a
        #   render reads is all from one source.
        warnings.warn(
            "{} carries a '{}' field for only part of the dump range, so it "
            "cannot be used -- serving one dump's field for another's would be "
            "a plausible-looking wrong answer. Planktos will write a complete "
            "series into the archive instead.".format(
                directory, flow.VORTICITY_TITLE), UserWarning, stacklevel=5)
        return 'archive', None, True

    target = _writable_source_dir(flow)
    if target is None:
        return 'archive', None, True
    return 'source', target, True


def _writable_source_dir(flow):
    '''The source's own fluid directory, if a file can actually be created there.

    Returns None for a read-only mount or a source that came from arrays.
    '''

    directory = flow.source_dir()
    if directory is None or not directory.is_dir():
        return None
    # Probed rather than assumed, and probed now: discovering it at the first
    #   dump would mean discovering it after the run has started.
    probe = directory / '.planktos_write_probe'
    try:
        with open(probe, 'wb'):
            pass
        probe.unlink()
    except OSError:
        return None
    # Resolved: this goes into meta.json for a reader that may open it from a
    #   different working directory.
    return directory.resolve()


def _replan_as_recorded(plan, recorded):
    '''``plan`` with the archive's settled vorticity placement put back.

    Everything else in a plan is the caller's request, which :func:`_appendable`
    has already checked matches.
    '''

    if not recorded or 'vorticity' not in recorded:
        return plan
    directory = recorded.get('vorticity_dir')
    directory = None if directory is None else Path(directory)
    meta = dict(plan.meta)
    meta['vorticity'] = recorded['vorticity']
    meta['vorticity_dir'] = recorded.get('vorticity_dir')
    # Any regime but 'recomputed' means the series lives on disk, so this
    #   stretch goes on filling it in. Writing is per dump and never touches an
    #   unstamped file, so a series the source already shipped whole simply
    #   skips every one.
    return plan._replace(write_vorticity=recorded['vorticity'] != 'recomputed',
                         vorticity_dir=directory, meta=meta)



#############################################################################
#                                                                           #
#                                THE RECORDER                               #
#                                                                           #
#############################################################################

class RunRecorder:
    '''Captures agent state to an archive as a run proceeds.

    Returned by :meth:`planktos.Environment.record`, which is the only way to
    make one: a recorder is environment-scoped, hooking the environment's time
    advance, and there is one per Environment.

    Recording is **live as soon as this exists** -- the call does the work, and
    ``with`` only adds the guaranteed close.

    ::

        with envir.record('run_archive/') as rec:
            for _ in range(steps):
                swrm.move(dt)

    Works without a ``with`` block too, since a ``with`` cannot span notebook 
    cells: ``envir.flush_recording()`` and ``envir.stop_recording()`` reach the 
    active recorder without a variable having to survive across cells.

    Attributes
    ----------
    path : Path
        the directory being written to. **Not necessarily the one asked for**: 
        recording into a non-empty directory redirects to a timestamped sibling
        (with a warning), and this is what says where the data actually went.
    appending : bool
        whether this recording is continuing the archive already at ``path``
        rather than starting one
    '''

    def __init__(self, envir, path, swarms=None, store=('positions',),
                 chunk_size=100, fluid='vort', quiver_shape=QUIVER_SHAPE,
                 plot_all=None, meta=None, capture_interval=1):
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
        self._plot_all = _check_plot_all(plot_all, self._swarms)

        record_meta = dict(meta) if meta else {}
        # Recorded because a later recording has to be able to tell whether
        #   continuing this archive would leave its timeline evenly spaced.
        record_meta['capture_interval'] = int(capture_interval)
        record_meta['provenance'] = {
            'planktos_version': planktos.__version__,
            'environment': _environment_record(envir),
            'fluid': envir._fluid_provenance,
            'ibmesh': envir._ibmesh_provenance}
        # Compared against at stop(), which warns if the run went on to change
        #   what restore() rebuilds from.
        self._state_at_start = _restorable_state(envir)

        # Decided before the archive exists, because it goes into meta.json --
        #   written once when recording starts and never rewritten.
        plan = _plan_fluid(envir, fluid, quiver_shape)
        # Through jsonable rather than converting each field by hand: it handles
        #   Path, numpy scalars and containers, so a quiver_shape given as an
        #   ndarray cannot reach _save_json unconverted.
        record_meta['fluid'] = (None if plan is None
                                else _provenance.jsonable(plan.meta))

        # Decided from state rather than from a remembered restore, and
        #   before the writer exists, since it is what settles where the writer
        #   writes at all.
        append = _appendable(path, envir, self._store, chunk_size,
                             capture_interval, record_meta['fluid'])
        self.appending = append is not None
        seed_fluid = None if append is None else append.dump_stats()
        if self.appending and plan is not None:
            # meta.json is not rewritten, so the archive's own answer to where
            #   vorticity lives is the one on disk and the one this has to keep
            #   using -- including when this recording would have chosen
            #   differently, which is what a partial series written by the
            #   earlier stretch makes it do.
            plan = _replan_as_recorded(plan, append.meta.get('fluid'))
            record_meta['fluid'] = _provenance.jsonable(plan.meta)
        self._writer = _ArchiveWriter(path, fingerprint_of(envir),
                                      meta=record_meta, chunk_size=chunk_size,
                                      store=self._store, append=append)
        self.path = self._writer.path

        # Now that the directory is settled, hook the fluid. This also sweeps
        #   whatever is already resident, so a run whose fluid never slides is
        #   fully recorded by the time record() returns.
        self._fluid = None
        if plan is not None:
            self._fluid = _FluidWriter(envir, self.path / 'fluid', plan,
                                       seed=seed_fluid)

        self._n_captures = self._writer.next_capture
        for index, swarm in enumerate(self._swarms):
            self._register(index, swarm, first_capture=self._n_captures)
        if self.appending:
            # Every swarm the archive holds has to go on being captured: a
            #   series that stops half way is a refusal rather than a silent
            #   gap (section 2.8), and the writer would otherwise raise about
            #   mismatched swarm sets at the first step instead of here.
            absent = sorted(set(self._writer._swarms) - set(
                range(len(self._swarms))))
            if absent:
                raise ValueError(
                    'swarm(s) {} are in {} but are not being recorded now, so '
                    'continuing it would leave their series stopping part way '
                    'through. Record every swarm the archive holds, or record '
                    'to a different directory.'.format(absent, self.path))
            # The archive's last capture is this state -- that is what decided
            #   the append -- so taking one now would duplicate it at the same
            #   timestamp. The next one lands a capture interval from here,
            #   which is the spacing the series already has.
            return
        # Capture 0 covers t0, so that capture j is exactly full_pos_history[j].
        self._capture()
        self._checkpoint()


    ####################   lifecycle   ####################

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self._auto_plot(exc_type)
        return False


    def _auto_plot(self, exc_type):
        '''Render what ``plot_all=`` asked for, on a clean exit or an exception.'''

        if self._plot_all is None:
            return
        # A crash leaves a diagnostic movie; a Ctrl-C asks for things to stop.
        #   Both have already flushed, in stop().
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            return
        if len(self._swarms) != 1:
            # A swarm joined after record() checked. Everything is on disk, so
            #   say what happened and leave the render to the user.
            warnings.warn(
                'no movie was made: plot_all= renders one swarm and this '
                'recording ended up covering {}. The archive at {} is '
                'complete -- render by hand with '
                'swrm.plot_all(...).'.format(len(self._swarms), self.path),
                UserWarning)
            return
        try:
            self._swarms[0].plot_all(**self._plot_all)
        except Exception as err:
            # Caught so a rendering failure cannot mask the run's own
            #   exception, which may be on its way up right now.
            warnings.warn(
                'the automatic plot_all= render failed -- {}: {}. The archive '
                'at {} is complete, so it can be rendered by hand with '
                'swrm.plot_all(...).'.format(type(err).__name__, err,
                                             self.path), UserWarning)


    def flush(self):
        '''Write buffered captures to disk. Keeps recording.'''

        self._writer.flush()
        self._checkpoint()
        if self._fluid is not None:
            self._fluid.flush()


    def _checkpoint(self):
        '''Rewrite every recorded swarm's checkpoint at the latest capture.

        Written on the same cadence as a chunk flush, which is what bounds how
        stale it can be: a hard kill costs the captures buffered since the last
        chunk boundary, and the checkpoint is never older than that same
        boundary. It holds positions and velocities itself rather than pointing
        at a capture index, so it does not depend on the chunk that kill took.
        '''

        if self._n_captures == 0:
            return
        for index, swarm in enumerate(self._swarms):
            klass = type(swarm)
            self._writer.write_checkpoint(
                index, self._n_captures - 1, self.envir.time,
                {'positions': swarm.positions,
                 'velocities': swarm.velocities,
                 'accelerations': swarm.accelerations,
                 'ib_collision_idx': swarm.ib_collision_idx,
                 'props': swarm.props,
                 'shared_props': swarm.shared_props,
                 'ib_condition': swarm.ib_condition,
                 # A name only, the way a fluid loader is recorded: the class
                 #   cannot be rebuilt from here, and apply_agent_model IS the
                 #   behavior, so a reader must say plainly when it cannot be
                 #   imported rather than hand back a plain Swarm.
                 'swarm_class': '{}.{}'.format(klass.__module__,
                                               klass.__qualname__),
                 'rndState': swarm.rndState.bit_generator.state})


    def stop(self):
        '''Flush, then unhook. Idempotent.'''

        if self._stopped:
            return
        self._stopped = True
        if self._fluid is not None:
            # Unhook from the fluid first: were the agent writer to raise while
            #   closing, the observer would otherwise outlive the recording and
            #   keep writing into the source's own directory.
            self._fluid.stop()
        self._checkpoint()
        self._writer.close()
        if self.envir._recorder is self:
            self.envir._recorder = None
            # Nothing is being gated any more; history goes back to every step.
            self.envir._capture_interval = 1
        self._warn_if_environment_changed()


    def _warn_if_environment_changed(self):
        '''Warn, naming what changed, if the Environment no longer matches the
        record restore() will rebuild it from.'''

        now = _restorable_state(self.envir)
        changed = []
        for key, value in now.items():
            if value != self._state_at_start[key]:
                name = _RESTORABLE_NAMES.get(key, key)
                if name not in changed:
                    changed.append(name)
        if changed:
            warnings.warn(
                'the Environment changed while recording to {} ({}). The '
                'archive records it as it was when recording started, so '
                'RunArchive.restore() rebuilds that version, not this '
                'one.'.format(self.path, ', '.join(changed)), UserWarning)


    ####################   hooks   ####################

    def _register(self, index, swarm, first_capture):
        # The name lives in shared_props, and defaults to 'organism' for every
        #   Swarm -- which is exactly why files are keyed by index instead.
        name = swarm.shared_props.get('name', 'organism')
        self._writer.add_swarm(index, name, swarm.N, swarm.positions.shape[1],
                               first_capture)


    def _sync_swarms(self):
        '''Pick up any swarm that has joined the environment since last capture.

        **Swarms are discovered here rather than notified from ``Swarm``**,
        which has two consequences worth knowing:

        - **A swarm that comes and goes between two captures is never seen.**
          ``calculate_FTLE`` builds a grid of probe agents on the environment
          and pops it again, so it needs to know nothing about recording.
        - ``first_capture`` is the capture index the swarm actually starts at,
          by construction, so indices correspond across every swarm with no
          per-swarm time base and no second indexing scheme.
        '''

        # A swarm's existence only matters at a capture, so discovering them
        #   here needs no hook in Swarm, and nothing has to know which of the
        #   several ways of building one the user reached for.
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
        arrays, velocities, props, shared = {}, {}, {}, {}
        for index, swarm in enumerate(self._swarms):
            named = {}
            for name in self._writer.arrays:
                named[name] = getattr(swarm, name)
            arrays[index] = named
            if self._writer.derive:
                velocities[index] = swarm.velocities
            if 'props' in self._writer.series:
                # Copied, not referenced: the buffer outlives the capture, and
                #   an agent model that edits props in place would otherwise
                #   rewrite every capture still waiting to be flushed. This is
                #   the same copy move() makes for props_history.
                props[index] = swarm.props.copy()
            if self._writer.shared:
                # The dict is copied here and each value where it is buffered,
                #   for the same reason props are.
                shared[index] = dict(swarm.shared_props)
        # Live attributes are read, not the histories: the archive does not
        #   depend on history existing, only on the two agreeing about when a
        #   state is recorded.
        self._writer.add_capture(self._n_captures, self.envir.time, arrays,
                                 velocities, props, shared)
        self._n_captures += 1
        # On the chunk boundary, so the checkpoint is never staler than the
        #   captures a hard kill would cost anyway.
        if self._n_captures % self._writer.chunk_size == 0:
            self._checkpoint()


def _check_plot_all(plot_all, swarms):
    '''Validate ``plot_all=``, at ``record()`` rather than at the end of the run.'''

    # plot_all is a Swarm method, and joint multi-swarm plotting is issue #49.
    if plot_all is None:
        return None
    if not isinstance(plot_all, dict):
        raise TypeError(
            'plot_all= takes a dict of Swarm.plot_all keyword arguments, e.g. '
            "dict(movie_filename='out.mkv', fluid='vort'), not {}.".format(
                type(plot_all).__name__))
    if len(swarms) != 1:
        raise ValueError(
            'plot_all= renders one swarm, and this recording covers {}. Name '
            'the one to render with swarms=[...], or leave plot_all= out and '
            'render afterwards with swrm.plot_all(...).'.format(len(swarms)))
    return dict(plot_all)


def _environment_record(envir):
    '''The Environment's own parameters, as ``meta.json`` records them.'''

    return {'L': [float(v) for v in envir.L],
            'units': envir.units,
            'bndry': [list(b) for b in envir.bndry],
            'rho': _provenance.jsonable(envir.rho),
            'mu': _provenance.jsonable(envir.mu),
            # nu is derivable as mu/rho from every construction but
            #   Environment(nu=...), where rho and mu are both None and it is
            #   the only one of the three there is.
            'nu': _provenance.jsonable(envir.nu),
            # motion.inertial_particles asserts both of these, so an inertial
            #   run that loses them cannot be restarted at all -- it raises
            #   before it moves.
            'char_L': _provenance.jsonable(envir.char_L),
            'U': _provenance.jsonable(envir.U),
            # Resolved rather than as given: 'k' in 2D and 'dimgrey' in 3D
            #   when the caller named none.
            'ibmesh_color': _provenance.jsonable(envir.ibmesh_color),
            # They are functions, so they cannot be recorded. Whether there
            #   were any can, which is what lets a reboot say they are missing
            #   only when they are.
            'has_plot_structs': bool(envir.plot_structs)}


# How _restorable_state's keys are named in a warning.
_RESTORABLE_NAMES = {'has_plot_structs': 'plot_structs',
                     'ibmesh_data': 'ibmesh'}


def _restorable_state(envir):
    '''What RunArchive.restore() rebuilds an Environment from, in a form that
    compares with ``==``: the parameters and the fluid and mesh provenance
    ``meta.json`` records, plus the shape and a digest of the mesh arrays.'''

    # Modifiers append to a provenance record in place, so a later comparison
    #   needs copies. The digest is what notices an ibmesh assigned directly,
    #   which records no provenance at all.
    mesh = None
    if envir.ibmesh is not None:
        digest = hashlib.sha1(np.ascontiguousarray(envir.ibmesh).tobytes())
        if envir.ibmesh_times is not None:
            digest.update(np.ascontiguousarray(envir.ibmesh_times).tobytes())
        mesh = (envir.ibmesh.shape, digest.hexdigest())
    return {**_environment_record(envir),
            'fluid': copy.deepcopy(envir._fluid_provenance),
            'ibmesh': copy.deepcopy(envir._ibmesh_provenance),
            'ibmesh_data': mesh}
