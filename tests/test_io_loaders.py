'''Tests for the data loaders/writers (planktos Environment + _dataio).

Two tiers:
  * Fixture-based mesh loaders (always run, fast): load the small committed
    fixtures in tests/fixtures/ -- in particular the *moving* immersed-boundary
    import (a directory of lagsPts.####.vtk), which had no automated coverage.
    Regenerate the fixtures with tests/fixtures/_gen_fixtures.py.
  * Data-gated fluid loaders: the in-repo IBAMR vtk (@vtk) and, when present, the
    COMSOL vtu (@vtu). These assert the loaded-flow contract and that domain
    boundary conditions are respected by a moving swarm.

The fluid *save* path (save_fluid) is currently broken on modern pyvista and is
pinned as a strict xfail.
'''

from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

import planktos
from planktos import _dataio

FIXTURES = Path(__file__).parent / 'fixtures'


# --------------------------------------------------------------------------- #
#            moving immersed boundary import (committed fixture)              #
# --------------------------------------------------------------------------- #

def test_moving_mesh_import():
    # 4-vertex chain (-> 3 segments) over 3 frames, translating +0.5 in x/frame.
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / 'lagspts_min'), dt=0.1, print_dump=1, d_start=0)
    assert envir.ibmesh.shape == (3, 3, 2, 2), "expected (T, N-1, 2, 2) moving mesh"
    assert np.allclose(envir.ibmesh_times, [0.0, 0.1, 0.2])
    # frame 0 first segment is the base chain; frame 2 has translated +1.0 in x
    assert np.allclose(envir.ibmesh[0, 0], [[1., 1.], [1., 2.]])
    assert np.allclose(envir.ibmesh[2, 0], [[2., 1.], [2., 2.]])
    # the boundary moves rigidly: every vertex shifts by +1.0 in x from frame 0->2
    assert np.allclose(envir.ibmesh[2, :, :, 0] - envir.ibmesh[0, :, :, 0], 1.0)
    assert np.allclose(envir.ibmesh[2, :, :, 1], envir.ibmesh[0, :, :, 1])   # y unchanged


# --------------------------------------------------------------------------- #
#              IB2d fluid import, scalar form (committed fixture)             #
# --------------------------------------------------------------------------- #
# IB2d writes velocity either as one vector file per dump (u.####.vtk) or as a
# pair of scalar files (uX/uY.####.vtk). The vector form is covered end to end in
# test_dynamic_loading.py; this pins the scalar branch, which is selected by a
# different filename test in IB2dData and read through a different branch of
# _dataio.read_vtk_Structured_Points.

def test_ib2d_fluid_scalar_form_matches_vector_form():
    from planktos import fluid
    scalar = fluid.IB2dData(str(FIXTURES / 'ib2d_fluid_scalar_min'),
                            dt=0.1, print_dump=10, d_start=0)
    vector = fluid.IB2dData(str(FIXTURES / 'ib2d_fluid_min'),
                            dt=0.1, print_dump=10, d_start=0, d_finish=2)
    assert scalar.vector_data is False and vector.vector_data is True
    assert scalar.fshape == vector.fshape == (3, 7, 6)
    assert np.allclose(scalar.flow_times, vector.flow_times)
    for q in (0.0, 1.5, 2.0):
        for a, b in zip(scalar(q), vector(q)):
            assert np.allclose(a, b)


# --------------------------------------------------------------------------- #
#                 TIME header scan (_dataio.read_vtk_time_only)               #
# --------------------------------------------------------------------------- #
# Exists so a dump series can be timestamped without being parsed: legacy VTK
# puts FIELD FieldData right after DATASET, so TIME is in the first few hundred
# bytes. That is what lets VTK3dData build the full timeline up front, which the
# windowed path requires -- see tests/test_dynamic_loading.py.

def test_read_vtk_time_only_matches_the_full_reader():
    files = sorted((FIXTURES / 'vtk3d_min').glob('IBAMR_db_*.vtk'))
    assert len(files) == 8
    peeked = [_dataio.read_vtk_time_only(str(f)) for f in files]
    full = [_dataio.read_vtk_Rectilinear_Grid_Vector(str(f))[2] for f in files]
    assert peeked == [0., 1., 2., 3., 4., 5., 6., 7.]
    assert np.allclose(peeked, full)


def test_read_vtk_time_only_returns_none_when_absent():
    # A file with no TIME field data. None means "not found in the header", which
    # callers must treat as "fall back to a full read", not "no time exists".
    assert _dataio.read_vtk_time_only(str(FIXTURES / 'lagspts_min' /
                                          'lagsPts.0000.vtk')) is None


def test_read_vtk_time_only_does_not_read_the_whole_file():
    # The point of the function: cost is independent of file size. Scanning far
    # less than the file still finds TIME, since it lives in the header.
    f = str(FIXTURES / 'vtk3d_min' / 'IBAMR_db_003.vtk')
    assert Path(f).stat().st_size > 512
    assert _dataio.read_vtk_time_only(f, nbytes=512) == 3.0


def test_read_vtk_time_only_rejects_a_truncated_value():
    # If the scan window ends mid-number the value would be silently wrong, so a
    # value line that is also the last line in the buffer is refused.
    f = str(FIXTURES / 'vtk3d_min' / 'IBAMR_db_003.vtk')
    with open(f, 'rb') as fh:
        head = fh.read(4096).decode('utf-8', errors='ignore')
    decl = next(i for i, ln in enumerate(head.splitlines())
                if ln.split()[:1] == ['TIME'])
    cutoff = len('\n'.join(head.splitlines()[:decl + 1])) + 2   # into the value line
    assert _dataio.read_vtk_time_only(f, nbytes=cutoff) is None


# --------------------------------------------------------------------------- #
#            static immersed boundary import (committed fixture)              #
# --------------------------------------------------------------------------- #

def test_static_vertex_import_adjacent():
    # Square corners; 'adjacent' connects successive vertices -> 3 open segments.
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min' / 'box.vertex'), method='adjacent')
    assert envir.ibmesh.shape == (3, 2, 2)
    assert envir.ibmesh_times is None
    assert np.allclose(envir.ibmesh[0], [[2., 2.], [4., 2.]])


def test_static_vertex_import_closed_with_add_idx():
    # add_idx_list adds the closing segment (vertex 3 -> 0), giving all 4 sides.
    # (The static 'adjacent' method, unlike the moving branch, ignores `periodic`.)
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min' / 'box.vertex'),
                              method='adjacent', add_idx_list=[(3, 0)])
    assert envir.ibmesh.shape == (4, 2, 2)
    assert np.allclose(envir.ibmesh[-1], [[2., 4.], [2., 2.]])   # last vertex -> first


# --------------------------------------------------------------------------- #
#          mesh import and the fluid coordinate frame                          #
# --------------------------------------------------------------------------- #
# The fluid loaders translate their data so its lower-left corner sits at the
# origin, recording the original corner in fluid_domain_LLC. Mesh coordinates
# arrive in that original frame, so every import path has to subtract the same
# offset or mesh and fluid end up silently offset from each other.
#
# Two bugs lived here. 'proximity' and 'hull' dereferenced self.flow with no
# None check, so importing a mesh into a fluid-free environment raised
# AttributeError -- even though that is a supported workflow (passing `res`
# explicitly is how you say you have no fluid grid to infer the connection
# radius from). And 'adjacent' and the moving-mesh branch never applied the
# shift at all. All four now go through one helper.
#
# The shift is a no-op for IB2d data, whose grids start at the origin, which is
# why the omission went unnoticed -- so these tests set fluid_domain_LLC
# explicitly rather than relying on a loader to produce a nonzero one.
#
# box.vertex is the four corners of a 2x2 square, so sides are 2.0 apart and
# diagonals 2*sqrt(2) = 2.83. The connection radius is res_factor*res = 0.501*res,
# and res=4.5 puts it at 2.25 -- catching the four sides, excluding the diagonals.

BOX = 'mesh_min/box.vertex'
BOX_RES = 4.5
LLC = (1.0, 0.5)

# method -> (extra kwargs, expected number of segments)
STATIC_METHODS = {'adjacent': ({}, 3),                  # open chain of 4 vertices
                  'proximity': ({'res': BOX_RES}, 4),   # the four sides
                  'hull': ({}, 4)}                      # hull of a square


def _shifted_envir():
    '''Environment whose fluid records a nonzero original lower-left corner.'''
    g = np.linspace(0, 10, 11)
    X, Y = np.meshgrid(g, g, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=10, flow=[np.zeros_like(X), np.zeros_like(Y)])
    envir.flow.fluid_domain_LLC = LLC
    return envir


@pytest.mark.parametrize('method', sorted(STATIC_METHODS))
def test_static_vertex_import_without_fluid(method):
    kwargs, n_seg = STATIC_METHODS[method]
    envir = planktos.Environment(Lx=10, Ly=10)
    assert envir.flow is None
    envir.read_IB2d_mesh_data(str(FIXTURES / BOX), method=method, **kwargs)
    assert envir.ibmesh.shape == (n_seg, 2, 2)
    # unshifted: there is no fluid frame to shift into
    assert np.isclose(envir.ibmesh[..., 0].min(), 2.0)
    assert np.isclose(envir.ibmesh[..., 1].min(), 2.0)


def test_static_vertex_import_proximity_segment_lengths():
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / BOX), method='proximity', res=BOX_RES)
    seg_len = np.linalg.norm(envir.ibmesh[:, 0, :] - envir.ibmesh[:, 1, :], axis=1)
    assert np.allclose(seg_len, 2.0)            # sides only, no diagonals


@pytest.mark.parametrize('method', sorted(STATIC_METHODS))
def test_static_vertex_import_shifts_into_the_fluid_frame(method):
    # Every static method must follow the fluid's translation. 'adjacent' used to
    # skip it, putting the mesh in a different frame from the fluid it is meant
    # to sit in -- with no error, just wrong geometry.
    kwargs, n_seg = STATIC_METHODS[method]
    envir = _shifted_envir()
    envir.read_IB2d_mesh_data(str(FIXTURES / BOX), method=method, **kwargs)
    assert envir.ibmesh.shape == (n_seg, 2, 2)
    assert np.isclose(envir.ibmesh[..., 0].min(), 2.0 - LLC[0])
    assert np.isclose(envir.ibmesh[..., 1].min(), 2.0 - LLC[1])


def test_moving_mesh_import_shifts_into_the_fluid_frame():
    # Same for the moving (directory of lagsPts.####.vtk) branch, which also used
    # to skip the shift. Unshifted, frame 0's first segment is [[1,1],[1,2]].
    envir = _shifted_envir()
    envir.read_IB2d_mesh_data(str(FIXTURES / 'lagspts_min'), dt=0.1, print_dump=1,
                              d_start=0)
    assert envir.ibmesh.shape == (3, 3, 2, 2)
    assert np.allclose(envir.ibmesh[0, 0],
                       [[1. - LLC[0], 1. - LLC[1]], [1. - LLC[0], 2. - LLC[1]]])
    # the shift is rigid: the +1.0 x-translation across frames is untouched
    assert np.allclose(envir.ibmesh[2, 0] - envir.ibmesh[0, 0],
                       [[1., 0.], [1., 0.]])


def test_static_vertex_import_proximity_without_res_still_requires_fluid():
    # res=None means "infer the radius from the fluid grid", which genuinely
    # needs a fluid. That guard is the one that should fire, with its message.
    envir = planktos.Environment(Lx=10, Ly=10)
    with pytest.raises(AssertionError, match='flow data'):
        envir.read_IB2d_mesh_data(str(FIXTURES / BOX), method='proximity')


# --------------------------------------------------------------------------- #
#            shift_ibmesh_to_match_LLC -- the other load ordering              #
# --------------------------------------------------------------------------- #
# The loaders shift a mesh into the fluid's frame only if a fluid is already
# present. Load the mesh first and there is nothing to shift into yet, so this
# method exists to apply the shift after the fact. It is public API and had no
# coverage and no callers.
#
# Regression: it indexed self.ibmesh[:,:,ii], which is the coordinate axis only
# for a static mesh. A moving mesh is (T,N,2,2), where axis 2 is the *endpoint*
# axis -- so it subtracted LLC[0] from both coordinates of every segment's first
# endpoint and LLC[1] from both of the second, shearing each segment instead of
# translating it. Silently: the shapes happen to line up in 2D.

def _fluid_with_llc():
    '''A FluidData recording a nonzero original lower-left corner.'''
    g = np.linspace(0, 10, 11)
    X, Y = np.meshgrid(g, g, indexing='ij')
    donor = planktos.Environment(Lx=10, Ly=10,
                                 flow=[np.zeros_like(X), np.zeros_like(Y)])
    donor.flow.fluid_domain_LLC = LLC
    return donor.flow


def test_shift_ibmesh_to_match_LLC_static():
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / BOX), method='adjacent')
    before = envir.ibmesh.copy()
    envir.flow = _fluid_with_llc()
    envir.shift_ibmesh_to_match_LLC()
    assert np.allclose(envir.ibmesh, before - np.array(LLC))


def test_shift_ibmesh_to_match_LLC_moving_translates_rigidly():
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.read_IB2d_mesh_data(str(FIXTURES / 'lagspts_min'), dt=0.1,
                              print_dump=1, d_start=0)
    before = envir.ibmesh.copy()
    assert before.shape == (3, 3, 2, 2)
    envir.flow = _fluid_with_llc()
    envir.shift_ibmesh_to_match_LLC()

    # every vertex moves by exactly -LLC, at every time
    assert np.allclose(envir.ibmesh, before - np.array(LLC))
    # and the segments keep their shape -- this is what shearing destroyed
    assert np.allclose(envir.ibmesh[..., 1, :] - envir.ibmesh[..., 0, :],
                       before[..., 1, :] - before[..., 0, :])


def test_shift_ibmesh_to_match_LLC_3d():
    # Static 3D triangles are (N,3,3). Built by hand to avoid the optional
    # numpy-stl dependency; only the array layout matters here.
    envir = planktos.Environment(Lx=10, Ly=10, Lz=10)
    envir.ibmesh = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                             [[1., 0., 0.], [1., 1., 0.], [0., 1., 2.]]])
    before = envir.ibmesh.copy()
    g = np.linspace(0, 10, 6)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    donor = planktos.Environment(Lx=10, Ly=10, Lz=10,
                                 flow=[np.zeros_like(X), np.zeros_like(Y),
                                       np.zeros_like(Z)])
    donor.flow.fluid_domain_LLC = (1.0, 0.5, 0.25)
    envir.flow = donor.flow
    envir.shift_ibmesh_to_match_LLC()
    assert np.allclose(envir.ibmesh, before - np.array([1.0, 0.5, 0.25]))


@pytest.mark.parametrize('kind', ['static', 'moving'])
def test_shift_ibmesh_to_match_LLC_matches_the_other_load_order(kind):
    # The contract, stated as an equivalence: loading the fluid first (so the
    # loader shifts the mesh) and loading the mesh first then calling this must
    # produce the same mesh.
    if kind == 'static':
        args, kwargs = (str(FIXTURES / BOX),), {'method': 'adjacent'}
    else:
        args = (str(FIXTURES / 'lagspts_min'),)
        kwargs = {'dt': 0.1, 'print_dump': 1, 'd_start': 0}

    fluid_first = planktos.Environment(Lx=10, Ly=10)
    fluid_first.flow = _fluid_with_llc()
    fluid_first.read_IB2d_mesh_data(*args, **kwargs)

    mesh_first = planktos.Environment(Lx=10, Ly=10)
    mesh_first.read_IB2d_mesh_data(*args, **kwargs)
    mesh_first.flow = _fluid_with_llc()
    mesh_first.shift_ibmesh_to_match_LLC()

    assert np.allclose(mesh_first.ibmesh, fluid_first.ibmesh)


def test_shift_ibmesh_to_match_LLC_requires_a_mesh():
    # Was a bare TypeError from subscripting None, unlike the two neighboring
    # guards which say what is missing.
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.flow = _fluid_with_llc()
    assert envir.ibmesh is None
    with pytest.raises(AssertionError, match='mesh'):
        envir.shift_ibmesh_to_match_LLC()


# --------------------------------------------------------------------------- #
#            IBAMR vtk fluid (in-repo data, vtk-gated)                        #
# --------------------------------------------------------------------------- #

IBAMR_PATH = 'tests/IBAMR_test_data'


def _assert_domain_bcs_respected(envir, sw):
    '''No agent ends outside the zero-bndry box, masked rows are fully masked,
    and the noflux z-faces are respected.'''
    for pos in sw.positions:
        if pos[0] is ma.masked:
            assert pos[1] is ma.masked and pos[2] is ma.masked, "all dims not masked"
            assert (pos.data[0] < 0 or pos.data[0] > envir.L[0] or
                    pos.data[1] < 0 or pos.data[1] > envir.L[1]), "unknown reason for mask"
        else:
            assert 0 <= pos[2] <= envir.L[2], "noflux not respected"
            assert 0 <= pos[0] <= envir.L[0] and 0 <= pos[1] <= envir.L[1], "zero bndry not respected"


@pytest.mark.vtk
def test_IBAMR_load_single_time():
    envir = planktos.Environment()
    envir.read_IBAMR3d_vtk_data(IBAMR_PATH, d_start=5, d_finish=None)
    envir.set_boundary_conditions(('zero', 'zero'), ('zero', 'zero'), ('noflux', 'noflux'))

    assert len(envir.L) == 3 and len(envir.bndry) == 3
    assert envir.flow.flow_times is None
    assert len(envir.flow) == 3 and len(envir.flow[0].shape) == 3
    assert envir.flow[0].shape == envir.flow[1].shape == envir.flow[2].shape
    assert [envir.flow.flow_points[d][0] for d in range(3)] == [0, 0, 0]
    assert [envir.flow.flow_points[d][-1] for d in range(3)] == envir.L
    assert [envir.flow[0].shape[d] for d in range(3)] == [len(envir.flow.flow_points[d]) for d in range(3)]
    assert envir.h_p is None and envir.time == 0.0 and envir.time_history == []

    envir.add_swarm(init='random')
    sw = envir.swarms[0]
    sw.shared_props['cov'] *= 0.001
    for _ in range(20):
        sw.move(0.1)
    _assert_domain_bcs_respected(envir, sw)


@pytest.mark.vtk
def test_IBAMR_load_time_series():
    envir = planktos.Environment()
    envir.read_IBAMR3d_vtk_data(IBAMR_PATH, d_start=3, d_finish=None)
    envir.set_boundary_conditions(('zero', 'zero'), ('zero', 'zero'), ('noflux', 'noflux'))

    assert len(envir.flow.flow_times) == 3
    assert envir.flow.flow_times[0] == 0 and envir.flow.flow_times[1] == 2 and envir.flow.flow_times[2] == 4
    assert len(envir.flow[0].shape) == 4                       # time + 3 space
    assert envir.flow[0].shape[0] == len(envir.flow.flow_times)
    assert [envir.flow[0].shape[d] for d in range(1, 4)] == [len(envir.flow.flow_points[d]) for d in range(3)]

    envir.add_swarm(init='random')
    sw = envir.swarms[0]
    sw.shared_props['cov'] *= 0.001
    for _ in range(10):
        sw.move(0.1)
    _assert_domain_bcs_respected(envir, sw)


@pytest.mark.vtk
def test_unstructured_grid_points_reader():
    points, bounds = _dataio.read_vtk_Unstructured_Grid_Points('tests/IBAMR_test_data/mesh_db.vtk')
    assert points.ndim == 2 and points.shape[1] == 3


# --------------------------------------------------------------------------- #
#            COMSOL vtu fluid (external data, vtu-gated)                      #
# --------------------------------------------------------------------------- #

@pytest.mark.vtu
def test_vtu_load():
    pathname = 'tests/data/comsol/vtu_test_data.txt'
    assert Path(pathname).is_file(), f"Comsol data {pathname} not found!"
    envir = planktos.Environment()
    envir.read_comsol_vtu_data(pathname, vel_conv=1000)
    envir.set_boundary_conditions(('zero', 'zero'), ('zero', 'zero'), ('noflux', 'noflux'))

    assert len(envir.L) == 3 and envir.flow.flow_times is None
    assert len(envir.flow) == 3 and len(envir.flow[0].shape) == 3
    assert [envir.flow.flow_points[d][-1] for d in range(3)] == envir.L
    assert envir.time == 0.0


# --------------------------------------------------------------------------- #
#            fluid / vorticity save round-trips                               #
# --------------------------------------------------------------------------- #
# Regression for BUG-SAVEFLUID (fixed): the writers no longer set the invalid
# .origin/.dimensions on a RectilinearGrid, the save methods pass coordinate
# arrays (flow_points) rather than domain lengths, and static flows are handled.
# Saved coordinates are origin-centered (lower-left corner at 0), per the
# Planktos convention. Round-trips go through _dataio.read_vtk_Rectilinear_Grid_Vector.

def _read_scalar_vtk(filename):
    '''Read a 2D scalar RectilinearGrid vtk back as an (nx, ny) array.'''
    import vtk
    from vtk.util import numpy_support
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(str(filename)); reader.ReadAllScalarsOn(); reader.Update()
    vd = reader.GetOutput()
    arr = numpy_support.vtk_to_numpy(vd.GetPointData().GetScalars())
    return arr.reshape(vd.GetDimensions()[::-1]).T.squeeze()


def test_save_fluid_static_2D_roundtrips(tmp_path):
    envir = planktos.Environment(Lx=10, Ly=8, rho=1000, mu=5000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.5, dpdx=0.22306, res=11)
    envir.save_fluid(str(tmp_path), 'flow')
    assert (tmp_path / 'flow.vtk').is_file()                      # single static file
    data, mesh, _ = _dataio.read_vtk_Rectilinear_Grid_Vector(str(tmp_path / 'flow.vtk'))
    assert np.allclose(data[0][:, :, 0], envir.flow[0])   # u, orientation preserved
    assert np.allclose(data[1][:, :, 0], envir.flow[1])   # v
    assert np.allclose(mesh[0], envir.flow.flow_points[0])            # origin-centered coords
    assert np.allclose(mesh[1], envir.flow.flow_points[1])
    assert mesh[0][0] == 0.0 and mesh[1][0] == 0.0


def test_save_fluid_time_varying_2D_roundtrips(tmp_path):
    envir = planktos.Environment(Lx=10, Ly=8, rho=1000, mu=20000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.1 * np.arange(-2, 6),
                            dpdx=np.ones(8) * 0.22306, res=11, tspan=[0, 10])
    envir.save_fluid(str(tmp_path), 'flow', time_history=False, flow_times=True)
    files = sorted(tmp_path.glob('flow_*.vtk'))
    assert len(files) == len(envir.flow.flow_times)                   # one file per flow time
    data, _, t = _dataio.read_vtk_Rectilinear_Grid_Vector(str(tmp_path / 'flow_0003.vtk'))
    expect = envir.interpolate_temporal_flow(time=envir.flow.flow_times[3])
    assert np.allclose(data[0][:, :, 0], expect[0])
    assert np.isclose(t, envir.flow.flow_times[3])


def test_save_fluid_static_3D_roundtrips(tmp_path):
    envir = planktos.Environment(Lx=20, Ly=20, Lz=20, rho=1000, mu=250000)
    envir.set_brinkman_flow(alpha=66, h_p=6, U=5, dpdx=0.22306, res=9)
    envir.save_fluid(str(tmp_path), 'flow3d')
    data, mesh, _ = _dataio.read_vtk_Rectilinear_Grid_Vector(str(tmp_path / 'flow3d.vtk'))
    assert all(np.allclose(data[i], envir.flow[i]) for i in range(3))
    assert all(np.allclose(mesh[i], envir.flow.flow_points[i]) for i in range(3))


def test_save_2D_vorticity_static_roundtrips(tmp_path):
    # Solid-body rotation v = (-y, x): vorticity = 2 everywhere.
    n = 15
    x = np.linspace(0, 10, n); y = np.linspace(0, 8, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[-Y, X])
    envir.flow.flow_points = (x, y)
    envir.save_2D_vorticity(str(tmp_path), 'vort')
    assert (tmp_path / 'vort.vtk').is_file()
    vort = _read_scalar_vtk(tmp_path / 'vort.vtk')
    assert vort.shape == (n, n)
    assert np.allclose(vort, 2.0, atol=1e-9)
