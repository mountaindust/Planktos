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
    assert envir.flow_times is None
    assert len(envir.flow) == 3 and len(envir.flow[0].shape) == 3
    assert envir.flow[0].shape == envir.flow[1].shape == envir.flow[2].shape
    assert [envir.flow_points[d][0] for d in range(3)] == [0, 0, 0]
    assert [envir.flow_points[d][-1] for d in range(3)] == envir.L
    assert [envir.flow[0].shape[d] for d in range(3)] == [len(envir.flow_points[d]) for d in range(3)]
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

    assert len(envir.flow_times) == 3
    assert envir.flow_times[0] == 0 and envir.flow_times[1] == 2 and envir.flow_times[2] == 4
    assert len(envir.flow[0].shape) == 4                       # time + 3 space
    assert envir.flow[0].shape[0] == len(envir.flow_times)
    assert [envir.flow[0].shape[d] for d in range(1, 4)] == [len(envir.flow_points[d]) for d in range(3)]

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

    assert len(envir.L) == 3 and envir.flow_times is None
    assert len(envir.flow) == 3 and len(envir.flow[0].shape) == 3
    assert [envir.flow_points[d][-1] for d in range(3)] == envir.L
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
    assert np.allclose(data[0][:, :, 0], envir.flow[0])          # u, orientation preserved
    assert np.allclose(data[1][:, :, 0], envir.flow[1])          # v
    assert np.allclose(mesh[0], envir.flow_points[0])            # origin-centered coords
    assert np.allclose(mesh[1], envir.flow_points[1])
    assert mesh[0][0] == 0.0 and mesh[1][0] == 0.0


def test_save_fluid_time_varying_2D_roundtrips(tmp_path):
    envir = planktos.Environment(Lx=10, Ly=8, rho=1000, mu=20000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.1 * np.arange(-2, 6),
                            dpdx=np.ones(8) * 0.22306, res=11, tspan=[0, 10])
    envir.save_fluid(str(tmp_path), 'flow', time_history=False, flow_times=True)
    files = sorted(tmp_path.glob('flow_*.vtk'))
    assert len(files) == len(envir.flow_times)                   # one file per flow time
    data, _, t = _dataio.read_vtk_Rectilinear_Grid_Vector(str(tmp_path / 'flow_0003.vtk'))
    expect = envir.interpolate_temporal_flow(time=envir.flow_times[3])
    assert np.allclose(data[0][:, :, 0], expect[0])
    assert np.isclose(t, envir.flow_times[3])


def test_save_fluid_static_3D_roundtrips(tmp_path):
    envir = planktos.Environment(Lx=20, Ly=20, Lz=20, rho=1000, mu=250000)
    envir.set_brinkman_flow(alpha=66, h_p=6, U=5, dpdx=0.22306, res=9)
    envir.save_fluid(str(tmp_path), 'flow3d')
    data, mesh, _ = _dataio.read_vtk_Rectilinear_Grid_Vector(str(tmp_path / 'flow3d.vtk'))
    assert all(np.allclose(data[i], envir.flow[i]) for i in range(3))
    assert all(np.allclose(mesh[i], envir.flow_points[i]) for i in range(3))


def test_save_2D_vorticity_static_roundtrips(tmp_path):
    # Solid-body rotation v = (-y, x): vorticity = 2 everywhere.
    n = 15
    x = np.linspace(0, 10, n); y = np.linspace(0, 8, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    envir = planktos.Environment(Lx=10, Ly=8, flow=[-Y, X])
    envir.flow_points = (x, y)
    envir.save_2D_vorticity(str(tmp_path), 'vort')
    assert (tmp_path / 'vort.vtk').is_file()
    vort = _read_scalar_vtk(tmp_path / 'vort.vtk')
    assert vort.shape == (n, n)
    assert np.allclose(vort, 2.0, atol=1e-9)
