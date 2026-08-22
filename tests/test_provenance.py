'''Tests for provenance records -- what produced an Environment's fluid and
immersed mesh (planktos/_provenance.py, run_persistence.md section 2.6).

Two halves. The first unit-tests the conversion and the decorators against
throwaway objects, so a failure points at the mechanism. The second drives the
real loaders against the committed fixtures, since the thing most likely to go
wrong is a loader that was never wired up -- which no unit test can see.

The load-bearing property throughout: a record must be either replayable or
plainly marked as not replayable. A record that looks like a loader call but
is not one is worse than no record at all.
'''

import json
import math
from pathlib import Path

import numpy as np
import pytest

import planktos
from planktos import _provenance


FIXTURES = Path(__file__).parent / 'fixtures'


# --------------------------------------------------------------------------- #
#                       jsonable: what can be recorded                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('value', [None, True, False, 0, -3, 2.5, 'a string', ''])
def test_jsonable_passes_plain_scalars_through_unchanged(value):
    assert _provenance.jsonable(value) is value or _provenance.jsonable(value) == value


def test_jsonable_converts_numpy_scalars_to_python():
    # numpy scalars are not json-serializable, and np.bool_ is not a bool.
    for value, want in ((np.int64(7), 7), (np.float64(1.5), 1.5),
                        (np.bool_(True), True), (np.float32(0.5), 0.5)):
        got = _provenance.jsonable(value)
        assert got == want and type(got) is type(want)


def test_jsonable_converts_paths_to_strings():
    got = _provenance.jsonable(Path('a') / 'b')
    assert isinstance(got, str) and got.endswith('b')


def test_jsonable_records_an_array_shape_but_never_its_contents():
    # Writing the values would duplicate exactly the data the archive exists to
    # avoid duplicating, and would put a velocity field inside a metadata file.
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = _provenance.jsonable(arr)
    assert got == {'ndarray': {'shape': [3, 4], 'dtype': 'float32'}}
    assert '11' not in json.dumps(got), 'array contents leaked into the record'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'),
                                   np.float64('nan'), np.float64('inf')])
def test_jsonable_marks_non_finite_floats_rather_than_emitting_invalid_json(value):
    # json.dumps writes bare NaN/Infinity by default, which is not valid JSON
    # and which other tools reject.
    got = _provenance.jsonable(value)
    assert isinstance(got, dict) and 'unrecorded' in got
    json.dumps(got, allow_nan=False)          # would raise on a bare float


def test_jsonable_records_a_callable_by_name():
    got = _provenance.jsonable(planktos.motion.tracer_particles)
    assert got == {'callable': 'tracer_particles'}


def test_jsonable_marks_an_unrepresentable_value_instead_of_dropping_it():
    class Opaque:
        def __repr__(self):
            return 'Opaque(...)'
    got = _provenance.jsonable(Opaque())
    assert got['unrecorded'] == 'Opaque'
    assert 'Opaque(...)' in got['repr']


def test_jsonable_survives_a_broken_repr():
    class Hostile:
        def __repr__(self):
            raise ValueError('no repr for you')
    got = _provenance.jsonable(Hostile())
    assert 'Hostile' in got['repr']


def test_jsonable_truncates_a_very_long_repr():
    got = _provenance.jsonable(object.__new__(type('Big', (), {
        '__repr__': lambda self: 'x' * 10_000})))
    assert len(got['repr']) <= _provenance.MAX_REPR


def test_jsonable_recurses_into_containers_and_stops_at_a_bounded_depth():
    got = _provenance.jsonable({'a': (1, np.int64(2)), 'b': [Path('p')]})
    assert got == {'a': [1, 2], 'b': ['p']}

    deep = value = []
    for _ in range(_provenance.MAX_DEPTH + 5):
        inner = []
        value.append(inner)
        value = inner
    json.dumps(_provenance.jsonable(deep), allow_nan=False)   # terminates


def test_jsonable_output_is_always_strict_json():
    messy = {'path': Path('x'), 'arr': np.zeros(3), 'nan': float('nan'),
             'fn': len, 'tup': (1, 'two', None), 'np': np.int32(4)}
    json.dumps(_provenance.jsonable(messy), allow_nan=False)


# --------------------------------------------------------------------------- #
#                     the decorators, on throwaway objects                    #
# --------------------------------------------------------------------------- #

class _Fake:
    def __init__(self):
        self._slot = None
        self._other = None

    @_provenance.records_provenance('_slot')
    def load(self, path, size=3, flag=False):
        '''A docstring worth preserving.'''
        self.loaded = (path, size, flag)
        return 'returned'

    @_provenance.records_provenance('_slot')
    def failing_load(self, path):
        raise FileNotFoundError(path)

    @_provenance.records_provenance('_other')
    def open_container(self, filename):
        return None

    @_provenance.records_provenance('_slot', preceded_by='_other')
    def read_from_container(self, field):
        return None

    @_provenance.note_modifier('_slot')
    def tweak(self):
        return None


def test_records_the_loader_name_and_every_argument_including_defaults():
    obj = _Fake()
    assert obj.load('somewhere') == 'returned', 'the return value must pass through'
    assert obj._slot == {'loader': 'load',
                         'kwargs': {'path': 'somewhere', 'size': 3, 'flag': False}}


def test_records_arguments_however_they_were_spelled():
    # positional and keyword calls must produce the same record, or replaying
    # one would not reproduce the other
    a, b = _Fake(), _Fake()
    a.load('p', 9, True)
    b.load(flag=True, size=9, path='p')
    assert a._slot == b._slot


def test_a_failed_load_leaves_no_record_rather_than_a_stale_one():
    # A loader that raises partway can leave the fluid in any state, so the
    # honest record is "unknown" -- and specifically not the previous load's,
    # which would now describe data that has been partly overwritten.
    obj = _Fake()
    obj.load('good')
    assert obj._slot is not None
    with pytest.raises(FileNotFoundError):
        obj.failing_load('bad')
    assert obj._slot is None


def test_preceded_by_folds_in_the_prerequisite_call():
    obj = _Fake()
    obj.open_container('data.nc')
    obj.read_from_container('u')
    assert obj._slot['loader'] == 'read_from_container'
    assert obj._slot['preceded_by'] == [{'loader': 'open_container',
                                         'kwargs': {'filename': 'data.nc'}}]


def test_preceded_by_is_omitted_when_there_was_no_prerequisite():
    obj = _Fake()
    obj.read_from_container('u')
    assert 'preceded_by' not in obj._slot


def test_a_modifier_appends_its_name_and_does_not_invent_a_record():
    obj = _Fake()
    obj.tweak()
    assert obj._slot is None, 'a modifier must not manufacture provenance'
    obj.load('p')
    obj.tweak()
    obj.tweak()
    assert obj._slot['modified_by'] == ['tweak', 'tweak']


def test_the_decorator_preserves_the_signature_and_docstring():
    # Sphinx autodoc reads both off the wrapper; losing them would silently
    # empty out the API reference for every loader.
    import inspect
    assert _Fake.load.__doc__ == 'A docstring worth preserving.'
    assert _Fake.load.__name__ == 'load'
    assert list(inspect.signature(_Fake.load).parameters) == \
        ['self', 'path', 'size', 'flag']


# --------------------------------------------------------------------------- #
#                     the real Environment loaders                            #
# --------------------------------------------------------------------------- #

def test_a_bare_environment_has_both_slots_and_they_are_empty():
    envir = planktos.Environment()
    assert envir._fluid_provenance is None
    assert envir._ibmesh_provenance is None


def test_arrays_handed_to_the_constructor_are_marked_unreplayable():
    # This is the path most of the test suite uses, and it bypasses every
    # loader. The record must exist -- so the writer never meets a missing
    # attribute -- and must say plainly that there is no call to replay.
    X = np.zeros((3, 3))
    envir = planktos.Environment(flow=[X, X])
    record = envir._fluid_provenance
    assert record is not None
    assert record['loader'] is None, 'must not invent a loader name'
    assert 'directly' in record['note']
    assert record['kwargs']['flow'][0] == {'ndarray': {'shape': [3, 3],
                                                       'dtype': 'float64'}}


def test_direct_assignment_of_an_ibmesh_leaves_no_provenance():
    # envir.ibmesh = mesh is how the collision tests build geometry. There is
    # nothing to record, and inventing something would be worse than None.
    envir = planktos.Environment()
    envir.ibmesh = np.zeros((2, 2, 2))
    assert envir._ibmesh_provenance is None


def test_an_analytic_flow_generator_records_its_call():
    envir = planktos.Environment(Lx=10, Ly=10, rho=1000, mu=1000)
    envir.set_brinkman_flow(alpha=3, h_p=0.5, U=1, dpdx=1, res=11)
    record = envir._fluid_provenance
    assert record['loader'] == 'set_brinkman_flow'
    assert record['kwargs']['alpha'] == 3
    assert record['kwargs']['res'] == 11
    assert record['kwargs']['tspan'] is None, 'defaults must be filled in'


def test_the_ib2d_fluid_loader_records_its_call():
    envir = planktos.Environment()
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    record = envir._fluid_provenance
    assert record['loader'] == 'read_IB2d_fluid_data'
    assert record['kwargs']['dt'] == 0.01 and record['kwargs']['print_dump'] == 10
    assert record['kwargs']['INUM'] is None
    assert 'ib2d_fluid_min' in record['kwargs']['path']


def test_the_vtk3d_loader_records_its_call():
    envir = planktos.Environment()
    envir.read_IBAMR3d_vtk_data(str(FIXTURES / 'vtk3d_min'))
    assert envir._fluid_provenance['loader'] == 'read_IBAMR3d_vtk_data'


@pytest.mark.parametrize('fixture, kwargs', [
    ('mesh_min/box.vertex', {}),
    ('lagspts_min', {'dt': 0.01, 'print_dump': 1}),
])
def test_the_ib2d_mesh_loader_records_its_call(fixture, kwargs):
    envir = planktos.Environment()
    envir.read_IB2d_mesh_data(str(FIXTURES / fixture), **kwargs)
    record = envir._ibmesh_provenance
    assert record['loader'] == 'read_IB2d_mesh_data'
    assert record['kwargs']['method'] == 'adjacent', 'defaults must be filled in'
    assert envir._fluid_provenance is None, 'a mesh load must not touch the fluid'


def test_loading_a_fluid_does_not_disturb_the_mesh_record_or_the_reverse():
    envir = planktos.Environment()
    envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min/box.vertex'))
    mesh_record = dict(envir._ibmesh_provenance)
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    assert envir._ibmesh_provenance == mesh_record
    assert envir._fluid_provenance['loader'] == 'read_IB2d_fluid_data'


def test_a_failed_real_load_clears_the_record():
    envir = planktos.Environment()
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    with pytest.raises(Exception):
        envir.read_IB2d_fluid_data(str(FIXTURES / 'no_such_dir'), dt=0.01,
                                   print_dump=10)
    assert envir._fluid_provenance is None


def test_a_mesh_modifier_is_recorded_so_a_reader_cannot_replay_the_loader_alone():
    envir = planktos.Environment()
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min/box.vertex'))
    assert 'modified_by' not in envir._ibmesh_provenance
    envir.add_vertices_to_static_2D_ibmesh()
    assert envir._ibmesh_provenance['modified_by'] == \
        ['add_vertices_to_static_2D_ibmesh']


def test_the_llc_shift_is_recorded_as_a_modifier():
    # The mesh is loaded first and then translated to match the fluid, so a
    # reconstruction that replayed only the loader would place it wrongly.
    envir = planktos.Environment()
    envir.read_IB2d_mesh_data(str(FIXTURES / 'lagspts_min'), dt=0.01, print_dump=1)
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    envir.shift_ibmesh_to_match_LLC()
    assert envir._ibmesh_provenance['modified_by'] == ['shift_ibmesh_to_match_LLC']


def test_every_recorded_environment_serializes_as_strict_json():
    # meta.json is written with json.dump; a record that cannot go through it
    # would fail at the end of a long run, which is the worst possible moment.
    envir = planktos.Environment()
    envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                               print_dump=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min/box.vertex'))
    for record in (envir._fluid_provenance, envir._ibmesh_provenance):
        json.dumps(record, allow_nan=False)

    direct = planktos.Environment(flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    json.dumps(direct._fluid_provenance, allow_nan=False)


def test_a_loader_still_returns_what_it_returned_before():
    # every decorated method's return value passes through untouched
    envir = planktos.Environment()
    assert envir.read_IB2d_fluid_data(str(FIXTURES / 'ib2d_fluid_min'), dt=0.01,
                                      print_dump=10) is None
    assert envir.flow is not None


@pytest.mark.parametrize('name', [
    'set_brinkman_flow', 'set_two_layer_channel_flow', 'set_canopy_flow',
    'read_IB2d_fluid_data', 'read_IBAMR3d_vtk_data', 'read_openfoam_vtk_data',
    'read_comsol_vtu_data', 'read_NetCDF_flow',
    'read_stl_mesh_data', 'read_IB2d_mesh_data',
    'read_3D_vertex_data_to_convex_hull',
])
def test_every_fluid_and_mesh_entry_point_is_wired_up(name):
    # The failure this catches is a loader nobody remembered to decorate --
    # which produces no error, just an environment that cannot say what it is.
    # Checked structurally because several of these need data we do not commit.
    method = getattr(planktos.Environment, name)
    assert hasattr(method, '__wrapped__'), \
        '{} records no provenance; see planktos/_provenance.py'.format(name)


@pytest.mark.parametrize('name', ['read_IB2d_fluid_data', 'read_IB2d_mesh_data',
                                  'set_brinkman_flow'])
def test_decorated_loaders_keep_their_signature_and_docstring(name):
    import inspect
    method = getattr(planktos.Environment, name)
    assert method.__name__ == name
    assert method.__doc__ and len(method.__doc__) > 50
    assert 'self' in inspect.signature(method).parameters
