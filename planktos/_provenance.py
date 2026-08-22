'''
Provenance records: what produced an Environment's fluid and immersed mesh.

An Environment cannot be serialized -- doing so would mean writing out the
whole velocity field and the mesh, which is the volume of data the run archive
exists to avoid duplicating. But almost everything needed to *reconstruct* one
is knowable at the moment it is built: the loader that was called and the
arguments it was given. That information exists only during the call, so each
loader records its own, and whatever wants it later (the run archive's metadata
sidecar, an error message explaining why an archive does not match this
environment) reads what was left behind.

Three things this buys, in increasing order of ambition:

1. A validation message that names both sides -- "this archive was recorded
   against read_IB2d_fluid_data(path='leaf_data', ...); this environment's
   fluid is read_openfoam_vtk_data(...)" -- rather than a bare mismatch.
2. A self-describing dataset: months later the archive says what produced it
   without anyone having to find the script.
3. Reconstruction, and therefore restart: reload becomes "re-run the recorded
   loader calls", which is honest about its cost, since the fluid is re-read
   from its own files, which is where it lives anyway.

**A provenance record is not a guarantee.** Paths go stale, datasets move, and
an environment built by hand -- an analytic field assembled in a script, a
programmatically modified bndry, arrays handed straight to Environment() --
leaves a record that is accurate but not sufficient. The rule is to record what
can be recorded, mark the rest plainly, and never let a reader silently act on
a record it could not verify. That is why unrepresentable values become typed
markers here rather than being dropped or guessed at, and why an in-place
modifier appends its name (see note_modifier) instead of leaving a record that
describes a mesh which no longer exists.

Nothing here writes anything. Serialization belongs to whatever consumes these
records; jsonable() is the guarantee that it can.

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

import functools
import inspect
import math
from pathlib import Path

import numpy as np

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

# How deep to descend into nested containers before giving up and recording a
#   marker. Loader arguments are flat in practice; this only bounds the damage
#   from something pathological (or self-referential) being passed in.
MAX_DEPTH = 6

# Longest repr kept for a value that cannot be represented directly. Long
#   enough to identify what it was, short enough that a stray large object
#   cannot bloat the metadata.
MAX_REPR = 200


def _short_repr(value):
    '''repr(value), truncated, and never raising on a broken __repr__.'''

    try:
        text = repr(value)
    except Exception:
        return '<unreprable {}>'.format(type(value).__name__)
    if len(text) > MAX_REPR:
        return text[:MAX_REPR - 3] + '...'
    return text


def _marker(kind, value):
    '''A typed stand-in for a value that cannot be recorded as itself.

    Always a dict carrying the reason and a truncated repr, so a reader can
    tell "this was not recordable" from "this was recorded as null" -- a
    distinction the run archive's validation depends on.
    '''

    return {'unrecorded': kind, 'repr': _short_repr(value)}


def jsonable(value, _depth=0):
    '''Convert a loader argument into something json.dump can write.

    Scalars, strings and containers of them pass through as themselves. numpy
    scalars become their Python equivalents and Paths become strings, since
    both round-trip back into a loader unchanged. Everything else becomes a
    typed marker: an ndarray records its shape and dtype but not its contents
    (the contents are the data this whole design avoids duplicating), a
    callable records its name, and anything unrecognized records its type.

    Non-finite floats are markers rather than bare floats because ``NaN`` and
    ``Infinity`` are not valid JSON -- Python's json module emits them by
    default, but the result is a file other tools reject.

    Parameters
    ----------
    value : any
        the argument to convert

    Returns
    -------
    A value composed only of dict, list, str, int, float, bool and None.
    '''

    # The numpy scalar types come first, and the order is load-bearing:
    #   np.float64 *is* a subclass of float, so a plain isinstance(value, float)
    #   check ahead of this would pass it through as a numpy scalar and the
    #   promise made above -- plain Python types only -- would quietly not hold.
    #   np.bool_ and np.integer are the opposite case, subclassing neither bool
    #   nor int, so they need naming whichever way round this is written.
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):          # bool already returned above
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _marker('nonfinite float', value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {'ndarray': {'shape': list(value.shape), 'dtype': str(value.dtype)}}

    if _depth >= MAX_DEPTH:
        return _marker('nested too deeply', value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item, _depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item, _depth + 1) for key, item in value.items()}

    if callable(value):
        return {'callable': getattr(value, '__name__', None) or _short_repr(value)}
    return _marker(type(value).__name__, value)


def records_provenance(slot, preceded_by=None):
    '''Decorate a loader so that it records its own call.

    The record lands on the Environment as the named attribute, in the form
    ``{'loader': <method name>, 'kwargs': {...}}``, with defaults filled in so
    that replaying it reproduces the call whether or not the caller spelled
    every argument out.

    **The slot is cleared before the call and set after it returns.** A loader
    that raises partway can leave the fluid or the mesh in any state at all, so
    the honest record for a failed load is "unknown" -- and specifically not
    the record of whatever was loaded before it, which would now describe data
    that has been partly overwritten.

    Parameters
    ----------
    slot : string
        attribute on the Environment to write the record to, e.g.
        '_fluid_provenance'
    preceded_by : string, optional
        another provenance slot whose record is a prerequisite for this call.
        Its contents are folded in under 'preceded_by' so that replaying the
        record means replaying both calls in order. Used for NetCDF, where
        load_NetCDF opens the dataset and read_NetCDF_flow reads a field out of
        it -- neither call reconstructs the fluid on its own.
    '''

    def decorate(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            setattr(self, slot, None)
            result = method(self, *args, **kwargs)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop('self', None)
            record = {'loader': method.__name__,
                      'kwargs': {name: jsonable(value)
                                 for name, value in arguments.items()}}
            if preceded_by is not None:
                prior = getattr(self, preceded_by, None)
                if prior is not None:
                    record['preceded_by'] = [prior]
            setattr(self, slot, record)
            return result

        return wrapper

    return decorate


def note_modifier(slot):
    '''Decorate a method that alters already-loaded data in place.

    Appends the method's name to the record's 'modified_by' list. Without this
    a record would keep claiming that the mesh is exactly what the loader
    produced, and a reconstruction built from it would silently differ from the
    mesh the run actually used -- the one failure mode this module exists to
    prevent.

    A record noting a modifier is still useful: every modifier in Planktos
    today is deterministic given the loaded data, so replaying the loader and
    then the listed modifiers reproduces the mesh. What a reader must not do is
    replay the loader alone and assume it matches.
    '''

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            record = getattr(self, slot, None)
            if record is not None:
                record.setdefault('modified_by', []).append(method.__name__)
            return result

        return wrapper

    return decorate
