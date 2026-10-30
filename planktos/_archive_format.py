'''
The run-archive format both sides share: constants, atomic file writes, file
naming, and the fingerprint.

The layout itself is specified in the ``planktos.archive`` module docstring.
``_recording`` writes through these helpers and ``archive`` reads through them,
so the two cannot disagree about a filename or a dtype.

``_atomic_write`` is what every file in an archive goes through, except a
per-dump fluid field, whose per-source vtk writer is staged and renamed instead.
The ``*fingerprint*`` functions decide whether an archive and an Environment
describe the same domain and timeline, and a refusal message is assembled from
them.

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import importlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import planktos

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

# Directory a per-dump fluid field is written into before being moved into place.
#   Needed because the per-source vtk writers build their own filenames and so
#   cannot be handed a temporary one; a temporary directory on the same
#   filesystem gets the same guarantee. Readers must ignore it, as they must
#   TMP_SUFFIX.
TMP_DIRNAME = '.planktos_partial'

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
#   it is how a reader knows an agent left the domain.
STORABLE = {'positions': 'pos', 'velocities': 'vel', 'accelerations': 'acc'}

# Series ``store`` can also name, which are not N x D arrays and so are written
#   by their own machinery rather than as chunked .npy. Each is opt-in and each
#   scales with run length, which is why none is on by default. 'props' is
#   O(N.T); 'shared_props' is O(T) -- a few values per capture whatever the
#   swarm size -- and rides in the per-swarm sidecar rather than adding a file.
SERIES = ('props', 'shared_props')



#############################################################################
#                                                                           #
#                                   FILES                                   #
#                                                                           #
#############################################################################

def _atomic_write(path, write_fn):
    '''Write a file so that it appears complete or not at all.

    Writes to a temporary name in the same directory, fsyncs it, and renames it
    into place -- ``os.replace`` is atomic on POSIX, and on Windows for a
    destination on the same volume.

    Parameters
    ----------
    path : Path
        final destination
    write_fn : callable
        called with an open binary file object; writes the contents
    '''

    # Under a temporary name because a kill partway through np.save leaves a
    #   truncated .npy that raises on read: one unlucky moment would cost the
    #   whole archive rather than one buffer.
    tmp = path.with_name(path.name + TMP_SUFFIX)
    try:
        with open(tmp, 'wb') as fobj:
            write_fn(fobj)
            fobj.flush()
            # A real fsync: os.replace survives process death on its own, but a
            #   node failure or power loss takes the page cache with it, and
            #   those are the runs an archive exists for. One sync per chunk.
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
    '''json.dump, atomically, refusing non-finite values, which are not JSON.'''

    # allow_nan=False: Python's json module emits bare NaN and Infinity, which
    #   no other tool reads back. Failing at the start of a run beats writing a
    #   metadata file that turns out to be unparsable at the end of one.
    text = json.dumps(obj, indent=2, allow_nan=False, sort_keys=True)
    _atomic_write(path, lambda fobj: fobj.write(text.encode('utf-8')))


def _as_json(obj):
    '''A canonical JSON string, for comparing what a round trip may have changed.'''

    return json.dumps(obj, sort_keys=True, default=str)



#############################################################################
#                                                                           #
#                                   NAMING                                  #
#                                                                           #
#############################################################################

def _chunk_name(prefix, index):
    '''e.g. ("swarm00_pos", 7) -> "swarm00_pos_0007.npy".'''

    return '{}_{:0{}d}.npy'.format(prefix, index, INDEX_WIDTH)


def _chunk_index_of(name):
    '''Recover the chunk index from a chunk filename.

    The inverse of :func:`_chunk_name`. A reader sorts on this integer, never on
    the filenames themselves.

    Parameters
    ----------
    name : str or Path

    Returns
    -------
    int, or None if this is not a chunk file
    '''

    # Zero-padding makes name order and index order agree only up to chunk 9999:
    #   at 10000 the name grows a fifth digit and lexical order puts _10000
    #   before _9999. Padding is for humans; the parse is for correctness.
    stem = Path(name).stem
    _, _, tail = stem.rpartition('_')
    try:
        return int(tail)
    except ValueError:
        return None


def _swarm_prefix(index):
    '''e.g. 3 -> "swarm03". Two digits by convention, more if ever needed.'''

    return 'swarm{:02d}'.format(index)


def _checkpoint_names(index):
    '''The three files one swarm's checkpoint is spread over.

    Returns ``(npz, csv, json)`` names for ``agents/``.

    Named for the role rather than with the ``swarmNN`` prefix, so that the
    roster scan -- which globs ``swarm*.json`` -- cannot pick a checkpoint up
    and try to read a swarm out of it.
    '''

    prefix = 'checkpoint{:02d}'.format(index)
    return prefix + '.npz', prefix + '_props.csv', prefix + '_meta.json'


def _quiver_name(t_idx):
    '''e.g. 42 -> "quiver_00042.npy", keyed on the fluid dump index.

    One definition, so the writer and the reader cannot disagree about it -- the
    same reason _chunk_name exists.
    '''

    return 'quiver_{:05d}.npy'.format(int(t_idx))



#############################################################################
#                                                                           #
#                                CONVERSIONS                                #
#                                                                           #
#############################################################################

def _split_props(props):
    '''Sort a props DataFrame into what csv can carry and what it cannot.

    Returns ``(frame, arrays)``: a DataFrame of the columns to write as text,
    and a dict of column name to stacked ndarray for the rest.
    '''

    # A props column may hold one ndarray per agent -- that is how a per-agent
    #   covariance is given, and Swarm.get_prop is built on np.stack for it.
    #   Such a column renders to csv as a broken multi-line row, so it goes to
    #   the npz. np.stack is what turns any column into a uniform (N, ...)
    #   array, which is exactly what an npz entry wants.
    frame, arrays = {}, {}
    for name in props.columns:
        column = props[name]
        try:
            stacked = np.stack(column.array, axis=0)
        except (ValueError, TypeError):
            # Ragged, or something np.stack cannot make an array of at all.
            #   Neither container can hold it, so say so rather than write a
            #   checkpoint that quietly lacks a property.
            warnings.warn(
                "props column '{}' holds values that cannot be stacked into "
                'one array, so it is not being checkpointed. A restart will '
                'come back without it.'.format(name), UserWarning)
            continue
        if stacked.dtype == object:
            warnings.warn(
                "props column '{}' has object dtype, which cannot be stored "
                'without pickle, so it is not being checkpointed. A restart '
                'will come back without it.'.format(name), UserWarning)
        elif stacked.ndim == 1:
            frame[name] = column.to_numpy()
        else:
            arrays[name] = stacked
    return pd.DataFrame(frame, index=props.index), arrays


def _unwrap(array):
    '''Give back a scalar or str where npz stored one as a 0-d array.'''

    return array.item() if array.ndim == 0 else array


def _arrayable(value):
    '''An ndarray for ``value``, or None if storing it would need pickle.'''

    try:
        array = np.asarray(value)
    except (ValueError, TypeError):
        return None
    return None if array.dtype == object else array


def _import_swarm_class(dotted):
    '''The Swarm class a checkpoint names, imported.

    Recorded as a name only, the way a fluid loader is: ``apply_agent_model``
    *is* the behavior, and it cannot be rebuilt from an archive.
    '''

    if not dotted:
        raise ValueError(
            'this checkpoint names no Swarm class, so there is no behavior to '
            'restore. Rebuild the Swarm by hand.')
    module_name, _, name = dotted.rpartition('.')
    try:
        module = importlib.import_module(module_name)
        klass = getattr(module, name)
    except (ImportError, AttributeError):
        raise ValueError(
            "the recorded run used the Swarm class '{}', which cannot be "
            'imported here. Its apply_agent_model is the behavior of the run, '
            'so restoring a plain Swarm in its place would silently be a '
            'different simulation. Put that class on the import path and try '
            'again.'.format(dotted))
    if not issubclass(klass, planktos.Swarm):
        raise ValueError(
            "'{}' is not a Swarm subclass; this archive does not describe a "
            'Planktos run.'.format(dotted))
    return klass



#############################################################################
#                                                                           #
#                                FINGERPRINT                                #
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

    These are what goes into ``grid.npz``. ``periodic_dim`` is part of it
    because it changes the vorticity computed in the outermost ring, so a stored
    vorticity field recorded under a different setting is a different field.

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

    # Small -- a few hundred floats even for a large 3D grid, one per dump for
    #   the timeline -- and the archive stores the axes anyway so that it can
    #   plot without touching fluid, so this is a comparison over a file that
    #   had to exist rather than a new artifact.
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

    A *description*, never the match test -- that is
    :func:`compare_fingerprints`, which reads the arrays so it can say what
    differs.
    '''

    # Duplicates grid.npz deliberately: meta.json is the file someone opens to
    #   see what a run was, and a summary needing a second file loaded to be
    #   legible is worse at that job. Both are written once, from one source.
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

    Comparison is exact -- shape, dtype and values.

    Parameters
    ----------
    stored, current : dict of ndarray
        as returned by :func:`build_fingerprint`

    Returns
    -------
    list of str
    '''

    # Exact is safe because a rebuilt environment re-runs the same loader over
    #   the same files: flow_points, flow_times and L are built in each loader's
    #   __init__ and never reassigned by load_dumpfiles or update_spline, so a
    #   windowed run keeps the values it recorded. Differences are returned
    #   rather than raised so the caller can name both sides' provenance beside
    #   them, which is what makes a refusal actionable.
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


def fingerprint_of(envir):
    '''Build an Environment's fingerprint.

    A flow-free environment fingerprints on dimension and domain alone.
    '''

    dimension = len(envir.L)
    if envir.flow is None:
        return build_fingerprint(dimension, envir.L)
    return build_fingerprint(dimension, envir.L,
                             flow_points=envir.flow.flow_points,
                             flow_times=envir.flow.flow_times,
                             periodic_dim=envir.flow.periodic_dim)


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


def _same_source(recorded, current):
    """Do these two provenance records describe the same fluid?

    Every field except ``INUM``, which describes how much of the dataset is held
    in memory at once rather than what the dataset is.
    """

    # Reloading a finished run's fluid at another INUM, or resident, is ordinary
    #   and leaves the data it describes the same.
    def comparable(record):
        if not record:
            return record
        kwargs = {k: v for k, v in (record.get('kwargs') or {}).items()
                  if k != 'INUM'}
        return dict(record, kwargs=kwargs)

    return comparable(recorded) == comparable(current)
