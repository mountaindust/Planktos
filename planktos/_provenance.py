'''
Provenance records: what produced an Environment's fluid and immersed mesh.

Each loader and analytic flow generator records its own call onto the
Environment, under ``_fluid_provenance`` or ``_ibmesh_provenance``. A record is
a dict holding the method's name and the arguments it was given, plus the names
of any methods that later altered the loaded data in place (see note_modifier).
Replaying those calls rebuilds the fluid or the mesh; when a run archive was
recorded against different data than the Environment it is being read against,
the mismatch message names the loader call on each side.

An argument that cannot be stored as itself -- an array, an object, a
non-finite float -- becomes a marker: a dict naming why it could not be stored,
with a truncated repr. Restoring a run raises on a record holding a marker
rather than calling the loader with something that is not what it was given.

Nothing here writes to disk. jsonable() is what makes a record safe to hand to
json.dump; where it goes is the caller's business.

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
#   marker. Loader arguments are flat in practice; the limit is what stops a
#   self-referential one from recursing forever.
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
    tell "this was not recordable" from "this was recorded as null".
    '''

    return {'unrecorded': kind, 'repr': _short_repr(value)}


def jsonable(value, _depth=0):
    '''Convert a loader argument into something json.dump can write.

    Scalars, strings and containers of them pass through as themselves. numpy
    scalars become their Python equivalents and Paths become strings, since
    both round-trip back into a loader unchanged. Everything else -- an
    ndarray, a callable, a non-finite float, an unrecognized type -- becomes a
    typed marker: an ndarray records its shape and dtype but not its contents,
    a callable its name, and anything else its type.

    Parameters
    ----------
    value : any
        the argument to convert

    Returns
    -------
    A value composed only of dict, list, str, int, float, bool and None.
    '''

    # The numpy scalar types come first, and the order is load-bearing:
    #   np.float64 is a subclass of float, so testing it here is what keeps it
    #   out of the plain-float branch below. np.bool_ and np.integer subclass
    #   neither bool nor int, so they need naming either way.
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
    # NaN and Infinity are not valid JSON. Python's json module writes them
    #   anyway, producing a file other tools reject, so they become markers.
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

    The slot is cleared before the call and set after it returns, so the record
    for a load that raised partway through is None.

    Parameters
    ----------
    slot : string
        attribute on the Environment to write the record to, e.g.
        '_fluid_provenance'
    preceded_by : string, optional
        another provenance slot whose record is a prerequisite for this call.
        Its contents are folded in under 'preceded_by', so replaying the record
        replays both calls in order. Used where two calls load one thing:
        load_NetCDF opens the dataset, read_NetCDF_flow reads a field out of it.
    '''

    def decorate(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Cleared first: a loader that raises partway leaves the fluid or
            #   mesh partly overwritten, and None is the accurate record of
            #   what is then in place.
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

    Appends the method's name to the record's 'modified_by' list. Every
    modifier in Planktos is deterministic given the loaded data, so replaying
    the loader and then the listed modifiers reproduces the mesh the run used.
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
