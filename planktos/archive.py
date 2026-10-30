'''
Run archives: append-only, crash-valid capture of a run as it proceeds.

Planktos otherwise holds an entire run in memory and can only write it out at
the end. An archive streams agent state to disk instead, along with what a later
plot needs from the fluid -- the mirror image of what dynamic fluid loading does
for the velocity field.

This module owns the on-disk format::

    run_archive/
      meta.json                 written once, never rewritten
      grid.npz                  the fingerprint: dimension, L, flow_points,
                                flow_times, periodic_dim
      agents/
        swarm00.json            name, N, D, first_capture
        times_0000.npy          (rows,) shared across swarms
        swarm00_pos_0000.npy    (rows, N, D); _vel and _acc likewise, when
                                ``store`` names them
        swarm00_mask_0000.npy   (rows, N) bool
        swarm00_ang_0000.npy    (rows, N) 2D heading, written exactly when
                                velocities are not
        swarm00_series.npz      the per-swarm per-capture sidecar: the speed
                                statistics when velocities are not stored,
                                shared__<key> when ``store`` names
                                'shared_props', or both
        swarm00_props_0000.csv  the props series, when ``store`` names it; a
                                column holding one array per agent spills to
                                swarm00_prop-<name>_0000.npy
        checkpoint00.npz        the latest state, with _props.csv and
                                _meta.json beside it
      fluid/
        dump_stats.npz          per-dump component means, plus the run's
                                velocity and vorticity extrema, and the digest
                                stamped into each vorticity file written
        quiver_00042.npy        downsampled velocity, if requested
        Omega.0042.vtk          vorticity, only if written and the source
                                directory could not take it

Three properties hold the format together:

- **An archive is valid with no finalizer having run.** Metadata is written when
  recording starts, every file appears atomically, and a reader reconstructs the
  timeline by scanning what is on disk rather than trusting a recorded count.
  A hard kill costs at most one unflushed chunk.
- **What accumulates lives in files that accumulate.** ``meta.json`` holds only
  what is known when recording starts and never changes.
- **Chunks are keyed on a global capture index**, not on each swarm's own row
  count, so a swarm added mid-run needs no second indexing scheme: its first
  chunk is simply short at the front.

**How it fits together.** This module is the reading side. Writing is reached
only through ``Environment.record``, which builds a ``RunRecorder``, and lives
in ``planktos/_recording.py``. The constants, file naming, atomic writes and
fingerprint both sides share are in ``planktos/_archive_format.py``.

``load_run`` / ``RunArchive``
    open an archive, validate it (format version, chunk contiguity, row counts
    against the capture count) and serve it. ``RunArchive.check_against``
    compares it against a live Environment; ``dump_stats`` and ``quiver`` are
    the fluid half; ``restore`` rebuilds the Environment and Swarms.
``CaptureSeries``
    what a request for an agent array hands back. Lazy and memory-mapped:
    indexing one capture opens only the chunk it lives in, so an archive larger
    than RAM stays readable.

Public surface is ``RunRecorder`` (re-exported here from ``_recording``),
``load_run``, ``RunArchive`` and ``CaptureSeries``.

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

import json
import warnings
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd

import planktos
from ._archive_format import (DTYPE, FORMAT_VERSION, STORABLE,
                              _checkpoint_names, _chunk_index_of, _chunk_name,
                              _describe_source, _import_swarm_class,
                              _quiver_name, _same_source, _swarm_prefix,
                              _unwrap, compare_fingerprints, fingerprint_of)
from ._recording import RunRecorder

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

__all__ = ['RunRecorder', 'RunArchive', 'CaptureSeries', 'load_run']


class CaptureSeries:
    '''One per agent-array of one swarm, across a whole run, read on demand.

    Returned by :meth:`RunArchive.positions` and :meth:`RunArchive.velocities`.
    Indexing it gives an ordinary masked array -- ``series[j]`` is one capture,
    ``series[a:b]`` a span of them -- and **only the chunks an index touches are
    read**, so an archive larger than memory stays usable.

    It is a sequence, not an ndarray, and never claims to be one;
    :meth:`asarray` materializes the lot when that is what you want, and says so
    in its name.

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
        # A heading is one number per agent, not a vector, so it is the one
        #   series whose captures are 1-D.
        self._per_agent = name == 'ang'
        self.shape = ((len(archive.times), self._N) if self._per_agent
                      else (len(archive.times), self._N, self._D))


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

        shape = self.shape[1:]
        if j < self._first:
            # Before this swarm joined the run. Fully masked, which is already
            #   what a masked row means everywhere else in Planktos.
            return ma.masked_array(np.zeros(shape, DTYPE),
                                   mask=np.ones(shape, bool))

        chunk, offset = self._archive._locate(j, self._first)
        data = self._archive._chunk(self._index, self._name, chunk)[offset]
        mask = self._archive._chunk(self._index, 'mask', chunk)[offset]
        # The stored mask is per row -- agents leave whole rows -- so it applies
        #   as it stands to a per-agent series, and broadcasts across the
        #   coordinates of a vector one.
        if not self._per_agent:
            mask = np.repeat(mask[:, None], self._D, axis=1)
        return ma.masked_array(np.array(data, dtype=DTYPE), mask=mask)


    def asarray(self):
        '''Materialize the whole series as one masked array.

        ⚠️ Reads every chunk and holds the result in memory: this is the call
        that an archive larger than RAM cannot afford. Index or iterate instead
        when that matters.
        '''

        return self[:]


class RunArchive:
    '''A finished (or still-running) archive of a run, opened for reading.

    Get one from :func:`planktos.load_run`. The archive is **read-only**: it
    never writes, and never flushes a recording on the writer's behalf.

    ::

        run = planktos.load_run('run_archive/')
        run.times                  # capture times, the archive's own time base
        run.swarms                 # [('organism', 0), ('organism', 1)]
        run.positions(0)[run.capture_at(3.4)]     # where they were at t=3.4

    Three rules for using it:

    - **Address swarms by index; names are a convenience.** The default ``Swarm``
      name is ``'organism'`` for every swarm, so ``run.positions('organism')``
      raises when the name is not unique rather than picking one.
      :attr:`swarms` lists name and index together.
    - **Resolve by time, not by index into someone else's list.** A swarm added
      mid-run starts at a nonzero capture, and a recording started after t=0 has
      its capture 0 partway into the run. Matching on :attr:`times` is right in
      all of those cases.
    - **Agent state is snapped, never interpolated.** :meth:`capture_at` gives
      the nearest capture and nothing blends between them, since interpolating
      across a domain wrap or a boundary slide would invent trajectories that
      never happened. Temporal interpolation belongs to the fluid.

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
        what this archive holds: the per-agent arrays, and any opt-in series

    See Also
    --------
    dump_stats : the per-dump fluid statistics, if any fluid was recorded
    quiver : one dump's stored quiver arrows
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

        # Recorded by every archive, beside version and chunk_size, and what
        #   the chunk scan turns into the list of files that must be present.
        if 'store' not in self.meta:
            raise ValueError(
                "{} has no 'store' in its meta.json, so what its chunk files "
                "hold cannot be known.".format(self.path))
        self.store = tuple(self.meta['store'])
        self._chunk_size = int(self.meta['chunk_size'])
        self.grid = dict(np.load(self.path / 'grid.npz', allow_pickle=False))

        self._agent_dir = self.path / 'agents'
        self._entries = self._read_roster()
        self._by_index = {e['index']: e for e in self._entries}
        self.swarms = [(e['name'], e['index']) for e in self._entries]
        self._time_chunks = self._chunk_indices('times')
        self.times = self._read_times()
        self._cache = {}
        self._dump_stats = None
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

        Raises
        ------
        ValueError
            if the archive holds no captures, or ``t`` is not finite
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

        Raises
        ------
        ValueError
            if this archive does not hold ``name``
        '''

        if name not in self.store:
            raise ValueError(
                "this archive does not hold '{}': it was recorded with "
                "store={}. Re-record with that array included, or work from "
                "what is here.".format(name, list(self.store)))
        return CaptureSeries(self, self._resolve_swarm(swarm), name)


    def checkpoint(self, swarm=0):
        """One swarm's latest state, as a dict, or None if none was written.

        Where :meth:`positions` and :meth:`velocities` give append-only history,
        this is the one state the recording ended at plus everything history
        cannot supply: the random stream, the properties, the immersed-boundary
        condition, and the name of the Swarm subclass whose ``apply_agent_model``
        is the behavior.

        Parameters
        ----------
        swarm : int or str, default=0
            index, or name when unambiguous

        Returns
        -------
        dict
            ``capture`` and ``time`` say where in the run the state is from.
            ``positions``, ``velocities`` and ``accelerations`` are masked
            arrays; ``props`` is a DataFrame; ``shared_props`` a dict;
            ``rndState`` the generator state dict; plus ``ib_collision_idx``,
            ``ib_condition`` and ``swarm_class``. None when the archive holds no
            checkpoint for this swarm.
        """

        entry = self._by_index[self._resolve_swarm(swarm)]
        npz_name, csv_name, json_name = _checkpoint_names(entry['index'])
        json_file = self._agent_dir / json_name
        if not json_file.is_file():
            return None
        state = json.loads(json_file.read_text())

        with np.load(self._agent_dir / npz_name, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        # One mask per state, and it is positions': a masked row means the agent
        #   is out of the domain, which is a fact about the agent.
        row_mask = arrays.pop('mask')
        mask = np.broadcast_to(row_mask[:, None],
                               arrays['positions'].shape).copy()
        for name in ('positions', 'velocities', 'accelerations'):
            if name in arrays:
                state[name] = ma.masked_array(arrays.pop(name), mask=mask)
        state['ib_collision_idx'] = arrays.pop('ib_collision_idx', None)

        state['shared_props'] = {
            name[len('shared__'):]: _unwrap(value)
            for name, value in arrays.items() if name.startswith('shared__')}

        # float_precision='round_trip' is not optional: pandas' default csv
        #   parser is where a float64 loses its last bits, not the writer.
        #   The manifest says whether there is a file at all, rather than the
        #   read having to tell an absent one from an empty one.
        if state['props_csv']:
            frame = pd.read_csv(self._agent_dir / csv_name,
                                float_precision='round_trip')
        else:
            frame = pd.DataFrame(index=range(len(row_mask)))
        for name, value in arrays.items():
            if name.startswith('prop__'):
                frame[name[len('prop__'):]] = list(value)
        state['props'] = frame
        return state


    def props(self, swarm=0):
        """One swarm's props at every capture, or None if they were not stored.

        Written only for a recording that asked for them with
        ``store=(..., 'props')``, since a series of a DataFrame scales with run
        length where the checkpoint's single copy does not.

        Parameters
        ----------
        swarm : int or str, default=0
            index, or name when unambiguous

        Returns
        -------
        list of DataFrame, one per capture and aligned to ``times``, or None.
        A column that appeared partway through the run is NaN before it existed.
        """

        index = self._resolve_swarm(swarm)
        prefix = _swarm_prefix(index)
        files = sorted(self._agent_dir.glob(prefix + '_props_*.csv'),
                       key=_chunk_index_of)
        if not files:
            return None
        # float_precision='round_trip' is not optional: pandas' default parser
        #   is where a float64 loses its last bits, not the writer.
        table = pd.concat([pd.read_csv(f, float_precision='round_trip')
                           for f in files], ignore_index=True)

        spilled = {}
        for path in self._agent_dir.glob(prefix + '_prop-*_*.npy'):
            name = path.name[len(prefix) + len('_prop-'):].rsplit('_', 1)[0]
            spilled.setdefault(name, []).append(path)
        for name, paths in spilled.items():
            spilled[name] = np.concatenate(
                [np.load(p, allow_pickle=False)
                 for p in sorted(paths, key=_chunk_index_of)])

        frames = []
        for j, (_, block) in enumerate(table.groupby('capture', sort=True)):
            frame = block.drop(columns=['capture', 'agent']).reset_index(
                drop=True)
            for name, stacked in spilled.items():
                frame[name] = list(stacked[j])
            frames.append(frame)
        return frames


    def agent_stats(self, swarm=0):
        """One swarm's per-capture speed statistics, or None if not recorded.

        Written only when ``store`` omitted velocities, since that is the case
        where a plot cannot work them out for itself. Three values per capture
        whatever the swarm size, over the agents still in the domain.

        Parameters
        ----------
        swarm : int or str, default=0
            index, or name when unambiguous

        Returns
        -------
        dict of ndarray keyed 'avg_vel' ``(n_captures, D)``, 'avg_spd' and
        'std_spd' ``(n_captures,)``, or None when velocities were stored and
        this was therefore not.

        Notes
        -----
        Indexed from the swarm's own first capture, not from the archive's --
        see :meth:`shared_props`.
        """

        series = self._series(self._resolve_swarm(swarm))
        stats = {name: series[name] for name in
                 ('avg_vel', 'avg_spd', 'std_spd') if name in series}
        return stats or None


    def shared_props(self, swarm=0):
        """One swarm's ``shared_props`` at every capture, or None if not kept.

        Written only for a recording that asked with
        ``store=(..., 'shared_props')``. O(T) whatever the swarm size, which is
        why it is a separate opt-in from the per-agent ``'props'`` series.

        Parameters
        ----------
        swarm : int or str, default=0
            index, or name when unambiguous

        Returns
        -------
        dict of masked array, keyed by property name, each ``(n_captures, ...)``
        with a capture masked where the key was not in ``shared_props`` then --
        it may appear or vanish mid-run. None when no series was written.

        Notes
        -----
        ⚠️ Indexed from the swarm's own first capture, not from the archive's.
        For a swarm that joined mid-run these are offset against
        :attr:`times` and against :meth:`positions`, which is front-padded to
        the archive's index. So are :meth:`props` and :meth:`agent_stats`.
        """

        index = self._resolve_swarm(swarm)
        series = self._series(index)
        columns = {}
        for name, value in series.items():
            if not name.startswith('shared__'):
                continue
            name = name[len('shared__'):]
            present = series.get('present__' + name)
            if present is None:
                # Written only for a key that was ever absent, so no entry
                #   means it was there at every capture.
                mask = False
            else:
                mask = np.broadcast_to(
                    ~present.reshape((-1,) + (1,) * (value.ndim - 1)),
                    value.shape)
            columns[name] = ma.masked_array(value, mask=mask)
        return columns or None


    def first_capture(self, swarm=0):
        """The capture index at which one swarm joined the recording.

        0 for every swarm the environment already held when recording started.
        It is the offset between the archive's index and a per-swarm series':
        :meth:`positions` and :meth:`angles` are front-padded to the archive's,
        where :meth:`props`, :meth:`agent_stats` and :meth:`shared_props` start
        at this capture.

        Parameters
        ----------
        swarm : int or str, default=0
            index, or name when unambiguous

        Returns
        -------
        int
        """

        return int(self._by_index[self._resolve_swarm(swarm)]['first_capture'])


    def _series(self, index):
        """The per-swarm per-capture sidecar, as a dict of ndarray. May be {}."""

        key = ('series', index)
        if key not in self._cache:
            path = self._agent_dir / (_swarm_prefix(index) + '_series.npz')
            if not path.is_file():
                self._cache[key] = {}
            else:
                with np.load(path, allow_pickle=False) as data:
                    self._cache[key] = {k: data[k] for k in data.files}
        return self._cache[key]


    def _shared_at(self, index, offset):
        """One swarm's ``shared_props`` as a dict, at its own capture ``offset``.

        None where no series was written; otherwise the keys that were in the
        dict at that capture, with the values they held.
        """

        columns = self.shared_props(index)
        if columns is None:
            return None
        at = {}
        for name, column in columns.items():
            if np.all(ma.getmaskarray(column)[offset]):
                continue
            at[name] = _unwrap(np.array(ma.getdata(column)[offset]))
        return at


    def angles(self, swarm=0):
        """One swarm's per-capture heading angles, or None if not recorded.

        Radians, as ``arctan2(v_y, v_x)``. Written only in 2D and only when
        ``store`` omitted velocities, since a 3D frame draws no heading markers
        and a stored velocity gives the angle directly.

        Returns
        -------
        CaptureSeries of ``(N,)`` masked arrays, or None.
        """

        index = self._resolve_swarm(swarm)
        if not self._chunk_indices('{}_ang'.format(_swarm_prefix(index))):
            return None
        return CaptureSeries(self, index, 'ang')


    ####################   the fluid half   ####################

    def dump_stats(self):
        """The recorded fluid statistics, or None if no fluid was recorded.

        ``means`` is the spatial mean of each velocity component **per dump**,
        which the plot statistics interpolate between; NaN there marks a dump the
        run never loaded. ``vmax`` (per component) and, in 2D when vorticity was
        recorded, ``vort_absmax`` are single extrema **over the whole run**, which
        is what fixes a colour limit and an arrow scale that two renders of the
        same run agree on. NaN in those means no dump ever arrived.
        ``vort_digest``, when Planktos wrote the vorticity series, is the digest
        stamped into each dump's file, '' where it wrote none.

        Returns
        -------
        dict of ndarray, keyed 'means' and 'vmax', and where present
        'vort_absmax', 'vort_digest' and 'flow_times'. None when the recording
        had no fluid.
        """

        stats_file = self.path / 'fluid' / 'dump_stats.npz'
        if not stats_file.is_file():
            return None
        if self._dump_stats is None:
            with np.load(stats_file, allow_pickle=False) as data:
                self._dump_stats = {k: data[k] for k in data.files}
        return self._dump_stats


    def quiver(self, t_idx):
        """One dump's stored quiver arrows, as ``(ncomp, nx_sub, ny_sub)``.

        The downsampled velocity, on the grid fixed when recording started;
        ``meta['fluid']['quiver_strides']`` are the strides it was taken with.

        Parameters
        ----------
        t_idx : int
            index into the fluid's own ``flow_times``, not into :attr:`times`

        Returns
        -------
        ndarray
        """

        fluid_meta = self.meta.get('fluid') or {}
        if 'quiver' not in (fluid_meta.get('quantities') or []):
            raise ValueError(
                "this archive holds no quiver data: it was recorded with "
                "fluid={}. Quiver is opt-in -- re-record with fluid='quiver' or "
                "fluid=('vort','quiver'), or plot without an "
                "archive.".format(fluid_meta.get('quantities')))
        f = self.path / 'fluid' / _quiver_name(t_idx)
        if not f.is_file():
            raise ValueError(
                'this archive has no quiver for fluid dump {}. Under dynamic '
                'loading only dumps the run actually reached have one.'.format(
                    t_idx))
        return np.load(f, allow_pickle=False)


    ####################   validation against an Environment   ####################

    def check_against(self, envir):
        '''Raise unless this archive describes the same domain and fluid.

        A grid or timeline mismatch is a hard refusal. A **provenance**
        difference only warns, and ``INUM`` is left out of that comparison
        entirely, since it says how much of the dataset is held in memory at
        once rather than what the dataset is.

        Parameters
        ----------
        envir : Environment
        '''

        # A hard refusal because the stored positions are bare numbers: nothing
        #   in them says what coordinate system they are in, so reading them
        #   against a different grid gives a plausible picture that is silently
        #   wrong. A run whose script merely moved directories is what the
        #   provenance warning covers instead.
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

        if not _same_source(recorded, current):
            warnings.warn(
                'this archive matches this Environment grid for grid, but was '
                'recorded against {} where this environment\'s fluid is {}. '
                'Two runs on the same mesh at the same cadence are '
                'indistinguishable by grid alone.'.format(
                    _describe_source(recorded), _describe_source(current)),
                UserWarning)


    ####################   restoring   ####################

    def restore(self, history=True, capture=None):
        """Rebuild the Environment and Swarms this archive recorded.

        The run is picked up where it left off: same positions, same properties,
        same random stream, same clock. Nothing about the fluid or the mesh is
        deserialized -- the loader calls the archive recorded are replayed, so
        the data is re-read from wherever it lives.

        ``capture=j`` rebuilds at capture *j* rather than at the end of the run.
        Only what the recording stored per capture comes from *j*; the rest --
        ``shared_props`` and the random stream among them -- is the run's final
        state, and a notice printed at the call says which is which. The
        continuation past *j* is therefore stochastically different from the
        original run, not bit-identical to it.

        Parameters
        ----------
        history : bool, default=True
            fill each Swarm's ``pos_history`` and ``vel_history`` from the
            archive, and the Environment's ``time_history`` with them. False
            leaves all three empty, which costs nothing to memory but leaves
            ``plot_all`` a single frame and agent statistics counting against
            the restored state rather than the original swarm.
        capture : int, optional
            the capture to rebuild at; the last one by default. Negative
            indexes from the end, and :meth:`capture_at` turns a time into an
            index.

        Returns
        -------
        envir : Environment
        swarms : list of Swarm
            in archive index order, already attached to ``envir``. A swarm that
            had not joined the run by ``capture`` is left out, with a warning.

        Raises
        ------
        ValueError
            if the archive holds no checkpoint, if a fluid or mesh loader
            cannot be replayed, or if a Swarm's class cannot be imported
        IndexError
            if ``capture`` is outside the archive

        Notes
        -----
        ``Environment.plot_structs`` are function handles and are not recorded;
        a run that had them is restored without them, with a warning.

        Recording a restored run **continues the archive it came from**, since
        restoring leaves the clock exactly at that archive's last capture and
        that is what ``record()`` checks. The resumed stretch is appended to the
        same directory rather than written beside it, provided ``store``,
        ``chunk_size`` and ``capture_interval`` match what the archive records
        and the fingerprint still describes this Environment.
        """

        at_capture = capture is not None
        j = self._resolve_capture(capture)
        if at_capture:
            # Printed rather than warned, in the shape of record()'s store=
            #   notice: it is a statement of what this rebuild could and could
            #   not take from capture j, not a suspicion that anything is wrong.
            print(self._restore_notice(j))

        envir = self._restore_environment()
        swarms = []
        for entry in self._entries:
            if entry['first_capture'] > j:
                warnings.warn(
                    'swarm {} joined the run at capture {}, so at capture {} '
                    'it did not exist yet and has not been restored. The '
                    'swarms handed back are the ones the run held at that '
                    'point.'.format(entry['index'], entry['first_capture'], j),
                    UserWarning)
                continue
            swarms.append(self._restore_swarm(envir, entry['index'], history,
                                              j, at_capture))
        # Capture j is full_pos_history[j], so the capture rebuilt at is the
        #   present and everything before it is history. Both lists must come
        #   out the same length or every frame a plot draws is misaligned,
        #   silently.
        envir.time = float(self.times[j])
        envir.time_history = ([float(t) for t in self.times[:j]] if history
                              else [])
        # The same link record() leaves behind, so a plot of a restored run
        #   draws its fluid backdrop from what that run wrote instead of reading
        #   the dataset again. Frames past what the recording covers are refused
        #   at the point they are asked for, as they are for any archive.
        envir._archive_path = self.path
        return envir, swarms


    def _resolve_capture(self, capture):
        '''The capture index a restore should rebuild at. None is the last.'''

        n = len(self.times)
        if not n:
            raise ValueError(
                '{} holds no captures, so there is nothing to restore '
                'to.'.format(self.path))
        if capture is None:
            return n - 1
        j = int(capture)
        if j < 0:
            j += n
        if not 0 <= j < n:
            raise IndexError(
                'capture {} is outside the {} captures in {}'.format(
                    capture, n, self.path))
        return j


    def _restore_notice(self, capture):
        '''What a rebuild at ``capture`` takes from there, and what it cannot.

        The per-capture series are the whole of what capture j can supply; the
        rest of a swarm's state lives in the checkpoint, which is the end of the
        run.
        '''

        recorded = [name for name in ('positions', 'velocities',
                                      'accelerations', 'props', 'shared_props')
                    if name in self.store]
        head = ('Restoring at capture {} of {} (t={:g}). Recorded per capture: '
                '{}.'.format(capture, len(self.times), self.times[capture],
                             ', '.join(recorded)))
        if capture == len(self.times) - 1:
            return (head + ' That is the end of the run, so nothing else is'
                    ' being substituted for it.')
        substituted = [name for name in ('velocities', 'props',
                                         'shared_props')
                       if name not in self.store]
        tail = ('if it varied during the run, this resumes with its final value'
                if len(substituted) == 1 else
                'if any of them varied during the run, this resumes with their '
                'final values')
        return '{} Taken from the end of the run instead: {} -- {}.'.format(
            head, ', '.join(substituted), tail)


    def _restore_environment(self):
        '''The Environment half: scalars from provenance, data by replay.'''

        prov = self.meta.get('provenance') or {}
        env = dict(prov.get('environment') or {})
        L = env.get('L')
        if not L:
            raise ValueError(
                '{} records no domain size, so no Environment can be rebuilt '
                'from it.'.format(self.path))
        kwargs = {'Lx': L[0], 'Ly': L[1]}
        if len(L) == 3:
            kwargs['Lz'] = L[2]
        for name in ('units', 'rho', 'mu', 'nu', 'char_L', 'U', 'ibmesh_color'):
            if env.get(name) is not None:
                kwargs[name] = env[name]
        for axis, name in zip(env.get('bndry') or [],
                              ('x_bndry', 'y_bndry', 'z_bndry')):
            # The pair, not one end of it: a domain periodic on one side only
            #   would otherwise come back wrong.
            kwargs[name] = list(axis)

        envir = planktos.Environment(**kwargs)
        # A mesh loaded while no fluid was present came out of its loader
        #   unshifted, and any shift_ibmesh_to_match_LLC came after the fluid.
        #   Replaying the loader first and its modifiers after the fluid puts
        #   each call back where it was made.
        mesh = prov.get('ibmesh')
        mesh_first = mesh is not None and mesh.get('loaded_before') == 'flow'
        if mesh_first:
            self._replay(envir, mesh, 'ibmesh', modifiers=False)
        replayed = self._replay(envir, prov.get('fluid'), 'fluid')
        if mesh_first:
            self._replay_modifiers(envir, mesh)
        else:
            self._replay(envir, mesh, 'ibmesh')
        if replayed:
            # The fingerprint is what says the rebuilt world is the same world.
            #   Skipped where the fluid could not be replayed at all, since the
            #   mismatch is then the thing the warning above already named and
            #   an error would only restate it as a refusal.
            self.check_against(envir)
        if env.get('has_plot_structs'):
            warnings.warn(
                'the recorded run drew extra structures through '
                'Environment.plot_structs. Those are function handles, so they '
                'are not in the archive and this Environment does not have '
                'them: add them again before plotting.', UserWarning)
        return envir


    def _replay(self, envir, record, what, modifiers=True):
        '''Re-run one recorded loader call, and any modifier that followed it
        unless ``modifiers`` is False.

        True when the record was replayed or there was nothing to replay; False
        when it named no loader and the data could not be rebuilt.
        '''

        if record is None:
            return True
        loader = record.get('loader')
        if loader is None:
            warnings.warn(
                'the {} was supplied to Environment() as arrays in the process '
                'that recorded this run, so there is no loader call to replay '
                'and none has been made. Set it yourself before using this '
                'Environment.'.format(what), UserWarning)
            return False
        for prior in record.get('preceded_by') or []:
            self._replay(envir, prior, what)

        kwargs = dict(record.get('kwargs') or {})
        unrecordable = sorted(name for name, value in kwargs.items()
                              if isinstance(value, dict))
        if unrecordable:
            raise ValueError(
                '{} cannot be replayed: {} was called with {}, which the '
                'provenance record could not store as {} -- an array or an '
                'object has no text form to replay. Rebuild this Environment '
                'by hand.'.format(what, loader, unrecordable,
                                  'values' if len(unrecordable) > 1 else 'a value'))
        try:
            getattr(envir, loader)(**kwargs)
        except AttributeError:
            raise ValueError(
                '{} cannot be replayed: this Planktos has no '
                'Environment.{}. The archive was written by version {}.'.format(
                    what, loader,
                    (self.meta.get('provenance') or {}).get('planktos_version')))
        except Exception as err:
            raise ValueError(
                '{} cannot be replayed: {}({}) raised {}: {}. The data it read '
                'has to still be where it was.'.format(
                    what, loader, ', '.join(sorted(kwargs)),
                    type(err).__name__, err))
        if modifiers:
            self._replay_modifiers(envir, record)
        return True


    def _replay_modifiers(self, envir, record):
        '''Re-run, in order, the modifiers recorded against one loader call.'''

        # Every modifier in Planktos is deterministic given the loaded data and
        #   takes no arguments, so replaying by name reproduces the result.
        for name in record.get('modified_by') or []:
            getattr(envir, name)()


    def _restore_swarm(self, envir, index, history, capture, at_capture):
        '''One Swarm, rebuilt at capture ``capture`` and reattached to ``envir``.

        ``at_capture`` says the caller named that capture, which is what decides
        whether a stored series or the checkpoint supplies each piece of state.
        '''

        state = self.checkpoint(index)
        if state is None:
            raise ValueError(
                'swarm {} has no checkpoint in {}, so its properties and '
                'random stream were never written and it cannot be restored. '
                'Archives recorded before checkpoints existed are readable but '
                'not restartable.'.format(index, self.path))

        klass = _import_swarm_class(state['swarm_class'])
        # The checkpoint is the end of the run, and is the whole answer unless
        #   the caller named a capture -- then everything the recording kept per
        #   capture comes from there instead, and the rest still comes here.
        first = self._by_index[index]['first_capture']
        frames = (self.props(index) if 'props' in self.store
                  and (history or at_capture) else None)
        positions = (self.positions(index)[capture] if at_capture
                     else state['positions'])
        shared = None
        if at_capture and 'shared_props' in self.store:
            # Replaced wholesale rather than merged: a key the run deleted
            #   before capture j is absent there, and must not come back from
            #   the checkpoint's copy of the final dict.
            shared = self._shared_at(index, capture - first)
            if shared is None:
                # store= says there should be one, so its absence contradicts
                #   the notice already printed and has to be said out loud.
                warnings.warn(
                    'swarm {} has no shared_props series in {}, though it was '
                    'recorded with store={}. Its shared_props are coming from '
                    'the end of the run instead of from capture {}.'.format(
                        index, self.path, list(self.store), capture),
                    UserWarning)
        if shared is None:
            shared = dict(state['shared_props'])
        swarm = klass(swarm_size=positions.shape[0], envir=envir,
                      init=np.array(ma.getdata(positions)),
                      ib_condition=state['ib_condition'],
                      name=shared.get('name', 'organism'),
                      color=shared.get('color', 'darkgreen'))
        # Assigned after construction, so a subclass __init__ that sets its own
        #   defaults does not win over what the run actually had.
        swarm.positions = ma.copy(positions)
        for name in ('velocities', 'accelerations'):
            if at_capture and name in self.store:
                setattr(swarm, name, ma.copy(self.array(name, index)[capture]))
            elif state.get(name) is not None:
                setattr(swarm, name, ma.copy(state[name]))
        if state.get('ib_collision_idx') is not None:
            swarm.ib_collision_idx = np.array(state['ib_collision_idx'])
        if at_capture and frames is not None:
            swarm.props = frames[capture - first].copy(deep=True)
        else:
            swarm.props = state['props'].copy(deep=True)
        swarm.shared_props = shared
        swarm.rndState.bit_generator.state = state['rndState']

        if history:
            swarm.pos_history = [ma.copy(state_j) for state_j
                                 in self.positions(index)[:capture]]
            if 'velocities' in self.store:
                swarm.vel_history = [ma.copy(state_j) for state_j
                                     in self.velocities(index)[:capture]]
            if 'props' in self.store:
                # A list rather than None is what store_prop_history means, so
                #   the restored swarm goes on keeping one as the original did.
                if frames is None:
                    warnings.warn(
                        'swarm {} has no props series in {}, though it was '
                        'recorded with store={}. It is being restored without '
                        'a props history.'.format(index, self.path,
                                                  list(self.store)),
                        UserWarning)
                    frames = []
                # Padded at the front to the same length as pos_history, which
                #   is itself padded to the archive's index. move() appends to
                #   both, so two lists of different lengths would stay
                #   misaligned for the rest of the run. Two cases need it: a
                #   swarm that joined part-way, whose series starts at its own
                #   first capture, and a swarm with no props at all, whose
                #   DataFrame has no rows and so writes no frames.
                recorded = frames[:capture - first]
                pad = (frames[0] if frames else swarm.props).copy(deep=True)
                # One shared frame rather than a copy per state: no plot can
                #   read it -- every agent is masked out of the positions for
                #   exactly those states -- and copying costs O(N) apiece to
                #   describe a swarm that was not there.
                swarm.props_history = ([pad] * (capture - len(recorded))
                                       + recorded)
            # Left empty otherwise, which is what tells a plot to read the
            #   statistics and headings the recording derived instead. Filling
            #   it with masked rows to keep the two lists the same length would
            #   read as "every agent has left" and print zero speeds.
        return swarm


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

        Parsed and sorted as integers; :func:`_chunk_index_of` says why.
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
            shorts = [STORABLE[name] for name in self.store
                      if name in STORABLE] + ['mask']
            # Written only when velocities were not, so its absence is ordinary
            #   and only its being partial is a fault.
            if self._chunk_indices('{}_ang'.format(
                    _swarm_prefix(entry['index']))):
                shorts.append('ang')
            for short in shorts:
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

            # The sidecar is not chunked -- it is rewritten whole on the same
            #   boundary -- so it has no gaps to find, only a length to check.
            #   Absence is ordinary; a short one is the same fault a short
            #   chunk is, and would read back as a series that quietly stops.
            want_rows = n - entry['first_capture']
            for name, value in self._series(entry['index']).items():
                if len(value) != want_rows:
                    raise ValueError(
                        "swarm {} series '{}' holds {} captures where the "
                        'capture count implies {}; this archive is '
                        'inconsistent'.format(entry['index'], name, len(value),
                                              want_rows))


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
            # 'mask' and 'ang' are already the on-disk shorthand; a stored
            #   array's name has to be translated to it.
            short = name if name in ('mask', 'ang') else STORABLE[name]
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
