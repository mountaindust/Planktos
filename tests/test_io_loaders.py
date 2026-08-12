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

import json
import re
import shutil
import warnings
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
#        VTK XML multiblock series: .vtm.series, .vtm, and cell data          #
# --------------------------------------------------------------------------- #
# The readers behind the OpenFOAM finite-volume path. The fixture is a miniature
# of the real oral-arm export (docs/notes/openfoam_oral_arm_dataset.md): a 4x4x5
# uniform box of hexahedra carrying CELL data, per-timestep .vtm manifests naming
# an interior .vtu plus inlet/outlet/walls .vtp patches, and a .vtm.series index.
# Eight dumps are declared; two are absent (an interior hole at t=5 and an end
# truncation at t=7), reproducing the truncated transfer the real data arrived in.
# Fields are analytic: U = (t, x, t*z), vorticity = (z, y, x), p = t + y, with U
# exactly zero on the no-slip walls.

OF_DIR = FIXTURES / 'openfoam_min'
OF_TIMES = [0., 1., 2., 3., 4., 5., 6., 7.]
OF_ABSENT_TIMES = [5., 7.]
# Interior cell centers -- inset half a cell from the domain, which is why the
# boundary patches have to be spliced back on to recover the full extent.
OF_XC = np.array([0.125, 0.375, 0.625, 0.875])
OF_ZC = np.array([0.2, 0.6, 1.0, 1.4, 1.8])
OF_DOMAIN = ((0., 1.), (0., 1.), (0., 2.))


def _of_internal(num):
    return _dataio.read_vtm_manifest(OF_DIR / f'case_min_{num}.vtm')[0]['internal']


def test_read_vtm_series_declares_the_whole_timeline():
    # The series is the timeline source for dynamic loading, so it has to report
    # every declared dump -- including ones whose data never arrived. Filtering
    # is the caller's policy; a reader that quietly dropped them would hide the
    # gap instead of surfacing it.
    files, times = _dataio.read_vtm_series(OF_DIR / 'case_min.vtm.series')
    assert len(files) == 8
    assert np.array_equal(times, OF_TIMES)
    assert [f.name for f in files] == [f'case_min_{n}.vtm'
                                       for n in (10, 20, 30, 40, 50, 60, 70, 80)]
    # resolved relative to the series file, not the working directory
    assert all(f.parent == OF_DIR for f in files)
    assert sum(f.is_file() for f in files) == 8


def test_read_vtm_series_reports_a_missing_time_as_nan(tmp_path):
    # NaN rather than an exception, so a caller can fall back to the per-file
    # TimeValue for just the entries the index failed to describe.
    p = tmp_path / 'x.vtm.series'
    p.write_text('{"file-series-version":"1.0","files":['
                 '{"name":"a.vtm","time":1.5},{"name":"b.vtm"}]}')
    files, times = _dataio.read_vtm_series(p)
    assert [f.name for f in files] == ['a.vtm', 'b.vtm']
    assert times[0] == 1.5 and np.isnan(times[1])


def test_read_vtm_series_rejects_an_index_with_no_files(tmp_path):
    p = tmp_path / 'x.vtm.series'
    p.write_text('{"file-series-version":"1.0"}')
    with pytest.raises(ValueError, match='files'):
        _dataio.read_vtm_series(p)


def test_read_vtm_manifest_flattens_the_block_nesting():
    # inlet/outlet/walls sit inside a <Block name='boundary'>; the nesting
    # carries nothing a loader needs, so it is flattened away.
    datasets, time = _dataio.read_vtm_manifest(OF_DIR / 'case_min_10.vtm')
    assert set(datasets) == {'internal', 'inlet', 'outlet', 'walls'}
    assert time == 0.
    assert datasets['internal'] == OF_DIR / 'case_min_10' / 'internal.vtu'
    assert datasets['walls'] == (OF_DIR / 'case_min_10' / 'boundary' /
                                 'walls.vtp')
    assert all(p.is_file() for p in datasets.values())


def test_read_vtm_manifest_time_agrees_with_the_series_index():
    # Two independent declarations of the same timeline. They have to agree, or
    # the fallback chain (series first, per-file TimeValue for gaps) would put a
    # dump at a different time depending on which source answered.
    files, times = _dataio.read_vtm_series(OF_DIR / 'case_min.vtm.series')
    assert [_dataio.read_vtm_manifest(f)[1] for f in files] == list(times)


def test_read_vtm_manifest_resolves_children_that_do_not_exist():
    # The enabling case for gap tolerance: an absent dump still has a manifest,
    # so its data path can be resolved and tested cheaply at construction rather
    # than discovered at the window slide that needs it.
    for num, t in zip((60, 80), OF_ABSENT_TIMES):
        datasets, time = _dataio.read_vtm_manifest(
            OF_DIR / f'case_min_{num}.vtm')
        assert time == t
        assert datasets['internal'].name == 'internal.vtu'
        assert not datasets['internal'].is_file()


def test_read_vtm_manifest_time_is_none_when_absent(tmp_path):
    p = tmp_path / 'x.vtm'
    p.write_text("<?xml version='1.0'?>\n<VTKFile type='vtkMultiBlockDataSet'>"
                 "<vtkMultiBlockDataSet>"
                 "<DataSet name='internal' file='d/internal.vtu' />"
                 "</vtkMultiBlockDataSet></VTKFile>")
    datasets, time = _dataio.read_vtm_manifest(p)
    assert time is None
    assert datasets == {'internal': tmp_path / 'd' / 'internal.vtu'}


def test_read_vtkxml_cell_data_reads_centers_not_corners():
    # Finite-volume fields live on cells. GetPoints() would return the 5x5x6=150
    # hexahedron corners, which is the wrong lattice to interpolate a cell field
    # on; the 4x4x5=80 cell centers are the right one.
    centers, data, time = _dataio.read_vtkxml_cell_data(_of_internal(10))
    assert centers.shape == (80, 3)
    assert data['U'].shape == (80, 3)
    assert time == 0.
    for d, expected in ((0, OF_XC), (1, OF_XC), (2, OF_ZC)):
        assert np.allclose(np.unique(np.round(centers[:, d], 9)), expected)
    # inset half a cell: the centers do not reach the domain edges
    for d, (lo, hi) in enumerate(OF_DOMAIN):
        assert centers[:, d].min() > lo and centers[:, d].max() < hi


def test_read_vtkxml_cell_data_cell_order_is_not_lexicographic():
    # Pins the fixture property that makes reordering testable at all. The real
    # export's cells are not lexicographic either, so a loader that reshapes
    # without a permutation scrambles the field silently. If a regenerated
    # fixture ever came out sorted, every downstream reordering test would start
    # passing for the wrong reason -- this is what catches that.
    centers, _, _ = _dataio.read_vtkxml_cell_data(_of_internal(10), arrays=())
    order = np.lexsort((centers[:, 0], centers[:, 1], centers[:, 2]))
    assert not np.array_equal(order, np.arange(len(centers)))


def test_read_vtkxml_cell_data_returns_the_analytic_fields():
    # Closed forms, evaluated in the file's own (scrambled) cell order -- so a
    # reader that returned centers and data in mismatched orders would fail here.
    for num, t in ((10, 0.), (40, 3.)):
        centers, data, _ = _dataio.read_vtkxml_cell_data(
            _of_internal(num), arrays=('U', 'vorticity', 'p'))
        x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
        assert np.allclose(data['U'], np.stack([np.full(80, t), x, t * z], 1))
        assert np.allclose(data['vorticity'], np.stack([z, y, x], 1))
        assert np.allclose(data['p'], t + y)
        assert data['p'].shape == (80,)      # scalar stays 1D


def test_read_vtkxml_cell_data_reads_only_the_requested_arrays():
    # An unstructured-grid file repeats its whole mesh every timestep, so the
    # fields that are not wanted are waste on top of waste. Velocity alone by
    # default; vorticity is one argument away, since solvers often ship it and
    # reading it beats regenerating it.
    _, default, _ = _dataio.read_vtkxml_cell_data(_of_internal(10))
    assert set(default) == {'U'}
    _, both, _ = _dataio.read_vtkxml_cell_data(_of_internal(10),
                                               arrays=('U', 'vorticity'))
    assert set(both) == {'U', 'vorticity'}
    assert np.array_equal(default['U'], both['U'])
    _, every, _ = _dataio.read_vtkxml_cell_data(_of_internal(10), arrays=None)
    assert set(every) == {'U', 'vorticity', 'p'}


def test_read_vtkxml_cell_data_warns_for_an_array_the_file_lacks():
    # Collaborators' export pipelines vary; asking for a field that is not there
    # should say so by name rather than return a quietly incomplete result.
    with pytest.warns(UserWarning, match='not found'):
        _, data, _ = _dataio.read_vtkxml_cell_data(_of_internal(10),
                                                   arrays=('U', 'nope'))
    assert set(data) == {'U'}


def test_read_vtkxml_cell_data_can_skip_the_cell_centers():
    # The mesh is static across the series, so the lattice is established once
    # and every later dump needs the field arrays only.
    centers, data, _ = _dataio.read_vtkxml_cell_data(
        _of_internal(20), load_cell_coordinates=False)
    assert centers is None
    assert data['U'].shape == (80, 3)


def test_read_vtkxml_cell_data_reads_the_polydata_patches():
    # The boundary patches are .vtp, not .vtu, and are what closes the half-cell
    # inset. inlet/outlet must land on the domain's z faces and share the
    # interior's in-plane lattice exactly, or the splice would need interpolation.
    datasets, _ = _dataio.read_vtm_manifest(OF_DIR / 'case_min_40.vtm')
    interior, _, _ = _dataio.read_vtkxml_cell_data(datasets['internal'],
                                                   arrays=())
    for name, plane in (('inlet', 0.), ('outlet', 2.)):
        centers, data, time = _dataio.read_vtkxml_cell_data(datasets[name])
        assert centers.shape == (16, 3) and time == 3.
        assert np.allclose(np.unique(centers[:, 2]), [plane])
        for d in (0, 1):
            assert np.array_equal(np.unique(centers[:, d]),
                                  np.unique(interior[:, d]))
    # walls is one PolyData holding all four lateral planes, and is exactly zero
    # (no-slip) -- which is what makes padding the shell with zeros exact.
    centers, data, _ = _dataio.read_vtkxml_cell_data(datasets['walls'])
    assert centers.shape == (4 * 4 * 5, 3)
    assert not data['U'].any()


def test_read_vtkxml_cell_data_raises_on_a_missing_file():
    # vtk's XML readers report a missing file on stderr and hand back an empty
    # dataset, which would surface much later as a confusing shape mismatch.
    with pytest.raises(FileNotFoundError):
        _dataio.read_vtkxml_cell_data(OF_DIR / 'case_min_60' / 'internal.vtu')


def test_read_vtkxml_cell_data_rejects_an_unsupported_extension(tmp_path):
    p = tmp_path / 'flow.vtk'
    p.write_text('not xml')
    with pytest.raises(ValueError, match='.vtu'):
        _dataio.read_vtkxml_cell_data(p)


# --------------------------------------------------------------------------- #
#           TimeValue from a VTK XML header, without parsing the file         #
# --------------------------------------------------------------------------- #
# The .vtu/.vtp counterpart of read_vtk_time_only, and the time source of last
# resort for an export with no index and no manifests. The saving is the whole
# point: an unstructured export repeats its entire mesh in every dump (51 MB per
# file in the reference dataset), so parsing the series to recover one float per
# dump would read gigabytes to answer a question the headers already answer.

def test_read_vtkxml_time_only_matches_the_full_read():
    # Same answer as the full parse, on both container types.
    for num, t in zip((10, 30, 70), (0., 2., 6.)):
        f = _of_internal(num)
        assert _dataio.read_vtkxml_time_only(f) == t
        assert _dataio.read_vtkxml_cell_data(
            f, arrays=(), load_cell_coordinates=False)[2] == t
    patch = OF_DIR / 'case_min_30' / 'boundary' / 'inlet.vtp'
    assert _dataio.read_vtkxml_time_only(patch) == 2.


def test_read_vtkxml_time_only_reads_an_ascii_array(tmp_path):
    # VTK writes TimeValue as ascii text in ascii data mode and as inline base64
    # in binary mode. Both are ordinary output, so both are decoded.
    import vtk
    src = vtk.vtkXMLUnstructuredGridReader()
    src.SetFileName(str(_of_internal(40)))
    src.Update()
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(tmp_path / 'ascii.vtu'))
    w.SetInputData(src.GetOutput())
    w.SetDataModeToAscii()
    w.Write()
    assert _dataio.read_vtkxml_time_only(tmp_path / 'ascii.vtu') == 3.


def test_read_vtkxml_time_only_declines_a_compressed_array(tmp_path):
    # A compressed inline array is not decoded here -- one slow path through the
    # full reader beats a second, barely-exercised decoder for a single float.
    # What matters is that declining reads as "unknown", never as a wrong time.
    import vtk
    src = vtk.vtkXMLUnstructuredGridReader()
    src.SetFileName(str(_of_internal(40)))
    src.Update()
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(tmp_path / 'zlib.vtu'))
    w.SetInputData(src.GetOutput())
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()
    assert _dataio.read_vtkxml_time_only(tmp_path / 'zlib.vtu') is None
    # ...and the fallback the loader takes does get it
    assert _dataio.read_vtkxml_cell_data(
        tmp_path / 'zlib.vtu', arrays=(), load_cell_coordinates=False)[2] == 3.


def test_read_vtkxml_time_only_returns_none_rather_than_half_an_answer():
    # A scan cut short by nbytes must report "unknown", not a truncated number.
    f = _of_internal(20)
    assert _dataio.read_vtkxml_time_only(f, nbytes=150) is None
    assert _dataio.read_vtkxml_time_only(f, nbytes=4096) == 1.


def test_read_vtkxml_time_only_returns_none_when_there_is_no_timevalue(tmp_path):
    p = tmp_path / 'untimed.vtu'
    p.write_text("<?xml version=\"1.0\"?>\n"
                 "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
                 "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
                 "  <UnstructuredGrid>\n"
                 "    <Piece NumberOfPoints=\"0\" NumberOfCells=\"0\"/>\n"
                 "  </UnstructuredGrid>\n</VTKFile>\n")
    assert _dataio.read_vtkxml_time_only(p) is None


# --------------------------------------------------------------------------- #
#            OpenFOAM finite-volume ingestion (fluid.OpenFOAMData)            #
# --------------------------------------------------------------------------- #
# Assembly of the same fixture into a Planktos grid. Three things happen here
# that no other loader does: the cells are reordered out of the file's scrambled
# order onto a lattice, the boundary patches are spliced onto the six faces to
# undo the half-cell inset, and dumps the series declares but which are absent
# are dropped from the timeline.
#
# The fixture's 4x4x5 interior assembles to 6x6x7. Its analytic fields make the
# result closed-form: U = (t, x, t*z) in the interior and on the inlet/outlet
# caps, and exactly 0 on the four lateral walls.

OF_GRID_X = np.array([0., 0.125, 0.375, 0.625, 0.875, 1.])
OF_GRID_Z = np.array([0., 0.2, 0.6, 1.0, 1.4, 1.8, 2.])
# t=5 and t=7 are absent, so the surviving timeline is 0,1,2,3,4,6
OF_KEPT = [0., 1., 2., 3., 4., 6.]


def _openfoam(**kwargs):
    from planktos import fluid
    with pytest.warns(UserWarning):        # absent dumps; uneven spacing
        return fluid.OpenFOAMData(str(OF_DIR), **kwargs)


def test_openfoam_grid_spans_the_full_domain():
    # The point of the boundary splice. Raw cell centers would report the domain
    # as 0.75 x 0.75 x 1.6 instead of the true 1 x 1 x 2, and every coordinate in
    # it would be shifted by half a cell.
    fd = _openfoam()
    assert fd.fshape == (6, 6, 6, 7)                   # (t, nx+2, ny+2, nz+2)
    assert np.allclose(fd.flow_points[0], OF_GRID_X)
    assert np.allclose(fd.flow_points[1], OF_GRID_X)
    assert np.allclose(fd.flow_points[2], OF_GRID_Z)
    assert np.allclose(fd.L, [1., 1., 2.])
    # rectilinear but NOT uniform: the outermost interval in each direction is
    # the half-cell from the outermost cell center out to the domain edge
    for d in (0, 1, 2):
        dg = np.diff(fd.flow_points[d])
        assert np.isclose(dg[0], dg[1]/2) and np.isclose(dg[-1], dg[1]/2)


def test_openfoam_drops_absent_dumps_and_warns():
    from planktos import fluid
    with pytest.warns(UserWarning, match='not on disk'):
        fd = fluid.OpenFOAMData(str(OF_DIR))
    # densely indexed over the dumps that exist, so d_start/d_finish line up with
    # flow_times and load_dumpfiles is never handed an absent filename
    assert fd.d_start == 0 and fd.d_finish == 5
    assert np.allclose(fd.flow_times, OF_KEPT)


def test_openfoam_warns_about_the_uneven_spacing_a_gap_leaves():
    # Separate from the missing-dump warning: interpolation error scales with the
    # dump interval, so the user should be told which stretch is degraded.
    from planktos import fluid
    with pytest.warns(UserWarning, match='not evenly spaced'):
        fluid.OpenFOAMData(str(OF_DIR))


def test_openfoam_interior_is_reordered_onto_the_lattice():
    # THE permutation test. The file's cell order is scrambled, so a loader that
    # reshaped without reordering would produce a field that is wrong everywhere
    # but still the right shape. u = t, v = x, w = t*z is asserted cell by cell.
    fd = _openfoam()
    X, _, Z = np.meshgrid(OF_GRID_X[1:-1], OF_GRID_X[1:-1], OF_GRID_Z[1:-1],
                          indexing='ij')
    for t in (0., 3.):
        u, v, w = fd(t)
        assert np.allclose(u[1:-1, 1:-1, 1:-1], t)
        assert np.allclose(v[1:-1, 1:-1, 1:-1], X)
        assert np.allclose(w[1:-1, 1:-1, 1:-1], t*Z)


def test_openfoam_faces_carry_their_own_patch_data():
    # Each face is filled from whatever patch covers it -- not assumed no-slip.
    # The lateral walls are zero here because the data says so; the inlet/outlet
    # caps carry the analytic field.
    fd = _openfoam()
    t = 3.
    u, v, w = fd(t)
    X, _ = np.meshgrid(OF_GRID_X[1:-1], OF_GRID_X[1:-1], indexing='ij')
    # inlet at z=0 and outlet at z=2, on the interior x/y lattice
    for k, zc in ((0, 0.), (-1, 2.)):
        assert np.allclose(u[1:-1, 1:-1, k], t)
        assert np.allclose(v[1:-1, 1:-1, k], X)
        assert np.allclose(w[1:-1, 1:-1, k], t*zc)
    # the four lateral walls
    for comp in (u, v, w):
        assert not comp[0, 1:-1, 1:-1].any()
        assert not comp[-1, 1:-1, 1:-1].any()
        assert not comp[1:-1, 0, 1:-1].any()
        assert not comp[1:-1, -1, 1:-1].any()


def test_openfoam_edges_and_corners_are_filled_from_adjoining_faces():
    # The 12 edges and 8 corners appear in no patch file. They are filled from
    # the faces that meet there, NOT assumed to be zero -- this fixture has an
    # inlet running into a no-slip wall, so the two sides genuinely disagree and
    # the fill is their average.
    fd = _openfoam()
    t = 1.
    u, v, w = fd(t)
    # edge where the x=0 wall meets the z=0 inlet: wall says 0, inlet says
    # (t, x_1, 0) at the first interior x center
    assert np.allclose(u[0, 1:-1, 0], t/2)
    assert np.allclose(v[0, 1:-1, 0], OF_GRID_X[1]/2)
    assert np.allclose(w[0, 1:-1, 0], 0.)
    # edge where two walls meet: both say 0, so no compromise is involved
    assert np.allclose(u[0, 0, 1:-1], 0.)
    # corner: the average of the three edges meeting at it, two of which are the
    # wall/inlet compromise above and one of which is wall/wall
    assert np.isclose(u[0, 0, 0], (t/2 + t/2 + 0.)/3)
    assert np.isclose(v[0, 0, 0], (OF_GRID_X[1]/2)*2/3)


def test_openfoam_warns_when_boundary_conditions_disagree_at_an_edge():
    # A real discontinuity in the boundary conditions, not a data error -- but
    # the fill there is a compromise and the user should know.
    from planktos import fluid
    with pytest.warns(UserWarning, match='disagree'):
        fluid.OpenFOAMData(str(OF_DIR))


def test_openfoam_rejects_a_mesh_that_is_not_a_lattice():
    # Planktos interpolates on a tensor-product grid, and an unstructured
    # container says nothing about the mesh inside it. Verified, not assumed.
    from planktos import fluid
    fd = _openfoam()
    good = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.)
                     for z in (0., 1.)])
    fd._build_lattice(good)                     # 2x2x2, fine
    bad = good.copy()
    bad[0, 0] = 0.5                             # 3 x-levels, 8 cells: not 3*2*2
    with pytest.raises(ValueError, match='not on a rectilinear grid'):
        fd._build_lattice(bad)


def _openfoam_manifest_without(tmp_path, drop):
    '''A copy of the fixture series whose manifests omit one boundary patch.'''
    lines = ["<?xml version='1.0'?>",
             "<VTKFile type='vtkMultiBlockDataSet' version='1.0'>",
             "  <vtkMultiBlockDataSet>"]
    entries = []
    for num in (10, 20, 30, 40, 50, 70):
        datasets, time = _dataio.read_vtm_manifest(OF_DIR / f'case_min_{num}.vtm')
        body = list(lines)
        for name, path in datasets.items():
            if name == drop:
                continue
            body.append(f"    <DataSet name='{name}' "
                        f"file='{path.resolve().as_posix()}' />")
        body += ["  </vtkMultiBlockDataSet>",
                 "  <FieldData>",
                 "    <DataArray type='Float32' Name='TimeValue' "
                 "NumberOfTuples='1' format='ascii'>",
                 f"{time}", "    </DataArray>", "  </FieldData>", "</VTKFile>"]
        (tmp_path / f'case_min_{num}.vtm').write_text('\n'.join(body))
        entries.append({'name': f'case_min_{num}.vtm', 'time': time})
    import json
    (tmp_path / 'case_min.vtm.series').write_text(json.dumps(
        {'file-series-version': '1.0', 'files': entries}))
    return tmp_path


def test_openfoam_requires_a_patch_on_every_face(tmp_path):
    # Without the walls patch the x and y faces have no data, so the domain would
    # come out half a cell short in both -- an error nothing downstream could
    # detect. Raise rather than hand back a quietly wrong domain.
    from planktos import fluid
    path = _openfoam_manifest_without(tmp_path, 'walls')
    with pytest.raises(RuntimeError, match='No boundary patch covers'):
        fluid.OpenFOAMData(str(path))


def test_openfoam_require_boundary_false_is_not_implemented_yet(tmp_path):
    from planktos import fluid
    path = _openfoam_manifest_without(tmp_path, 'inlet')
    with pytest.raises(NotImplementedError, match='require_boundary=False'):
        fluid.OpenFOAMData(str(path), require_boundary=False)


def test_openfoam_environment_reader():
    envir = planktos.Environment()
    with pytest.warns(UserWarning):
        envir.read_openfoam_vtk_data(str(OF_DIR))
    assert np.allclose(envir.L, [1., 1., 2.])
    assert envir.flow.fshape == (6, 6, 6, 7)
    # the fluid is reachable through the normal Environment interface, and
    # reads back (u, v, w) = (t, x, t*z) at an interior point
    got = envir.interpolate_flow(np.array([[0.5, 0.5, 1.0]]), time=2.0)
    assert np.allclose(np.squeeze(got), [2.0, 0.5, 2.0])


# --------------------------------------------------------------------------- #
#          OpenFOAM: rebuilding the timeline from an incomplete export        #
# --------------------------------------------------------------------------- #
# The .vtm.series index, the .vtm manifests, and the internal.vtu TimeValue are
# three independent declarations of the same timeline, so losing one of them
# need not be fatal. The loader tries them in that order and then falls back to
# unit steps.
#
# What is pinned here is the announcement as much as the recovery: a timeline
# quietly rebuilt from a worse source is the shape of the VTK3dData frozen-fluid
# bug, so every step past the first warns and records which source answered.
#
# Variants are derived from the committed fixture rather than committed
# separately, so they cannot drift from what they are a damaged copy of.

# Renaming the surviving dumps 10,20,30,40,50,70 -> 7,8,9,10,11,12 makes the
# lexical and numeric orderings of the directory names disagree: sorted as text,
# 10/11/12 come before 7/8/9. That is the real export's naming (case08_..._787
# through case08_..._1034, unpadded), and the reason the glob sorts numerically.
OF_RENUMBER = {10: 7, 20: 8, 30: 9, 40: 10, 50: 11, 70: 12}


def _strip_timevalue(path):
    '''Delete the FieldData block from a VTK XML file, in place.

    The fixture writes inline base64 with no appended binary section, so the
    file is ASCII throughout and the block can simply be cut out.
    '''
    path.write_text(re.sub(r'\s*<FieldData>.*?</FieldData>', '',
                           path.read_text(), flags=re.DOTALL))


def _set_manifest_time(path, t):
    '''Rewrite the TimeValue a .vtm manifest declares.'''
    path.write_text(re.sub(r"(Name='TimeValue'[^>]*>)\s*\S+\s*(</DataArray>)",
                           r'\g<1>' + '\n{}\n'.format(t) + r'\g<2>',
                           path.read_text()))


def _of_variant(tmp_path, index=True, manifests=True, timevalue=True,
                renumber=False):
    '''A copy of the committed fixture with pieces of the export taken away.'''
    root = tmp_path / 'VTK'
    shutil.copytree(OF_DIR, root)
    if not index:
        for p in root.glob('*.vtm.series'):
            p.unlink()
    if not manifests:
        for p in root.glob('*.vtm'):
            p.unlink()
    if not timevalue:
        for p in list(root.rglob('*.vtu')) + list(root.rglob('*.vtp')):
            _strip_timevalue(p)
    if renumber:
        # Only meaningful once the manifests are gone: they name the dump
        # directories, so renaming those would break the paths they resolve.
        assert not manifests and not index
        for old, new in OF_RENUMBER.items():
            (root / 'case_min_{}'.format(old)).rename(
                root / 'case_min_{}'.format(new))
    return root


def _assert_fixture_field(fd, times):
    '''The fixture's interior field is u = t, v = x, w = t*z.

    Asserting it against the *original* dump times, indexed by position in the
    rebuilt timeline, is what pins the dump ORDER: it is the one thing that
    distinguishes a correctly ordered series from a scrambled one carrying the
    same set of values.
    '''
    X, _, Z = np.meshgrid(OF_GRID_X[1:-1], OF_GRID_X[1:-1], OF_GRID_Z[1:-1],
                          indexing='ij')
    for k, t_orig in enumerate(times):
        u, v, w = fd(fd.flow_times[k])
        assert np.allclose(u[1:-1, 1:-1, 1:-1], t_orig)
        assert np.allclose(v[1:-1, 1:-1, 1:-1], X)
        assert np.allclose(w[1:-1, 1:-1, 1:-1], t_orig*Z)


def test_openfoam_records_the_primary_source_when_the_export_is_whole():
    # The baseline the fallbacks are measured against. No warning about the
    # source, because nothing was lost.
    fd = _openfoam()
    assert fd.dump_source == 'series'
    assert fd.time_source == 'the .vtm.series index'
    assert fd.series_path.name == 'case_min.vtm.series'


def test_openfoam_natural_sort_orders_unpadded_dump_numbers():
    # The trap the directory glob is sorted against, in the real export's own
    # names: unpadded, so 1008 sorts before 787 as text.
    from planktos import fluid
    names = ['case08_alpha2_1e8_{}'.format(n)
             for n in (787, 800, 917, 1008, 1034)]
    assert sorted(reversed(names)) != names            # lexically wrong...
    assert sorted(reversed(names), key=fluid.OpenFOAMData._natural_key) == names


# ---- 1: no .vtm.series index -> the .vtm manifests --------------------------

def test_openfoam_falls_back_to_the_manifests_when_the_index_is_gone(tmp_path):
    from planktos import fluid
    path = _of_variant(tmp_path, index=False)
    with pytest.warns(UserWarning, match='No .vtm.series index'):
        fd = fluid.OpenFOAMData(str(path))
    assert fd.dump_source == 'manifests'
    assert 'manifest' in fd.time_source
    assert fd.series_path is None
    # the same timeline and the same field the index would have produced
    assert np.allclose(fd.flow_times, OF_KEPT)
    _assert_fixture_field(fd, OF_KEPT)


def test_openfoam_manifests_still_report_the_dumps_that_never_arrived(tmp_path):
    # A manifest survives for each absent dump, so the gap is still declared and
    # still warned about -- the fallback loses the index, not the record.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False)
    with pytest.warns(UserWarning, match='not on disk'):
        fd = fluid.OpenFOAMData(str(path))
    assert len(fd.flow_times) == 6


def test_openfoam_fills_a_gap_in_the_index_from_the_manifests(tmp_path):
    # A partial failure of the primary source: the index is there but does not
    # time every entry. The manifest answers for just those entries, and says so.
    from planktos import fluid
    path = _of_variant(tmp_path)
    series = path / 'case_min.vtm.series'
    data = json.loads(series.read_text())
    del data['files'][2]['time']                       # case_min_30, t = 2
    series.write_text(json.dumps(data))
    with pytest.warns(UserWarning, match='without a usable time'):
        fd = fluid.OpenFOAMData(str(path))
    assert fd.dump_source == 'series'
    assert 'filling 1 gap' in fd.time_source
    assert np.allclose(fd.flow_times, OF_KEPT)


# ---- 2: no manifests either -> the dump directories -------------------------

def test_openfoam_falls_back_to_the_dump_directories(tmp_path):
    from planktos import fluid
    path = _of_variant(tmp_path, index=False, manifests=False)
    with pytest.warns(UserWarning, match='no .vtm manifests'):
        fd = fluid.OpenFOAMData(str(path))
    assert fd.dump_source == 'directories'
    assert 'internal.vtu' in fd.time_source
    assert np.allclose(fd.flow_times, OF_KEPT)
    _assert_fixture_field(fd, OF_KEPT)


def test_openfoam_directory_fallback_finds_the_boundary_patches(tmp_path):
    # Without a manifest naming them, the patches are found by their position:
    # the .vtp files in each dump's boundary/ subdirectory. They still have to
    # splice on, or the domain would come out half a cell short in every
    # direction -- so the recovered grid is the assertion.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False, manifests=False)
    with pytest.warns(UserWarning):
        fd = fluid.OpenFOAMData(str(path))
    assert fd.fshape == (6, 6, 6, 7)
    assert np.allclose(fd.flow_points[0], OF_GRID_X)
    assert np.allclose(fd.flow_points[2], OF_GRID_Z)
    assert np.allclose(fd.L, [1., 1., 2.])
    u, v, w = fd(3.)
    for comp in (u, v, w):                             # the no-slip walls
        assert not comp[0, 1:-1, 1:-1].any()
        assert not comp[1:-1, -1, 1:-1].any()


def test_openfoam_directory_fallback_cannot_know_a_dump_is_missing(tmp_path):
    # Worth pinning because it is a real loss, not an oversight. The absent
    # dumps are declared nowhere once the manifests are gone, so nothing can
    # report them missing; the widened interval is the only remaining trace, and
    # the uneven-spacing warning is the only thing that will mention it.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False, manifests=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        fd = fluid.OpenFOAMData(str(path))
    msgs = [str(w.message) for w in rec]
    assert not any('not on disk' in m for m in msgs)
    assert any('not evenly spaced' in m for m in msgs)
    assert np.allclose(fd.flow_times, OF_KEPT)


# ---- 3: no time information anywhere -> unit steps --------------------------

def test_openfoam_assumes_unit_steps_when_nothing_carries_a_time(tmp_path):
    from planktos import fluid
    path = _of_variant(tmp_path, index=False, manifests=False, timevalue=False,
                       renumber=True)
    with pytest.warns(UserWarning, match='assuming unit time steps'):
        fd = fluid.OpenFOAMData(str(path))
    assert fd.time_source == 'assumed unit steps'
    assert np.allclose(fd.flow_times, np.arange(6))
    # No uneven-spacing warning is possible here, and that is the point of the
    # loud one: unit steps have silently closed the gap the real timeline had.


def test_openfoam_untimed_dumps_are_ordered_numerically_not_lexically(tmp_path):
    # THE ordering test, and the only place the numeric sort is load-bearing:
    # with no times to sort by, the filename order IS the timeline. Sorted as
    # text the six dumps would come out 10,11,12,7,8,9 -- so u, which reads back
    # the dump's original time, would run 3,4,6,0,1,2 instead of 0,1,2,3,4,6.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False, manifests=False, timevalue=False,
                       renumber=True)
    with pytest.warns(UserWarning):
        fd = fluid.OpenFOAMData(str(path))
    _assert_fixture_field(fd, OF_KEPT)


def test_openfoam_refuses_a_series_that_is_only_partly_timed(tmp_path):
    # Deliberately NOT the unit-step fallback. Overwriting a mostly-real
    # timeline with indices would move every dump that did carry a time.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False, manifests=False)
    _strip_timevalue(path / 'case_min_30' / 'internal.vtu')
    with pytest.raises(RuntimeError, match='rest are timed'):
        fluid.OpenFOAMData(str(path))


# ---- the chain's own guard rails --------------------------------------------

def test_openfoam_rejects_a_timeline_that_does_not_advance(tmp_path):
    # Both splines divide by the interval between successive times, so a repeat
    # is not a degraded timeline but an unusable one.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False)
    _set_manifest_time(path / 'case_min_30.vtm', 1.0)  # already case_min_20's
    with pytest.raises(RuntimeError, match='not strictly increasing'):
        fluid.OpenFOAMData(str(path))


def test_openfoam_reorders_globbed_dumps_by_their_recovered_times(tmp_path):
    # For a globbed source the filename order was only ever a proxy; the times
    # are the real thing, and disagreeing with the names is worth saying.
    from planktos import fluid
    path = _of_variant(tmp_path, index=False)
    # t = 0 -> 5, so case_min_10 moves from the front of the series to between
    # case_min_50 (t = 4) and case_min_70 (t = 6).
    _set_manifest_time(path / 'case_min_10.vtm', 5.0)
    with pytest.warns(UserWarning, match='not in time order'):
        fd = fluid.OpenFOAMData(str(path))
    assert np.allclose(fd.flow_times, [0., 1., 2., 3., 4., 5.])
    # The field is baked into the files and does not move with the relabeling,
    # so u still reads back each dump's ORIGINAL time -- which is what shows the
    # dumps themselves were reordered, not merely their timestamps.
    _assert_fixture_field(fd, [1., 2., 3., 4., 0., 6.])


def test_openfoam_raises_when_there_is_nothing_to_read(tmp_path):
    from planktos import fluid
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError, match='No .vtm.series index'):
        fluid.OpenFOAMData(str(tmp_path / 'empty'))


# --------------------------------------------------------------------------- #
#              OpenFOAM: the mesh is assumed static across the series         #
# --------------------------------------------------------------------------- #
# The lattice and permutation are built once from the first dump, and every
# later dump reshaped through them. A changed cell COUNT is caught on every
# dump; a reordering at the SAME count is the dangerous one, since the reshape
# succeeds and every value lands in the wrong place. The second dump is checked
# against the mesh for it.
#
# Second dump only, deliberately; TODO.md item 5 records why, and what that
# leaves uncovered.


def _of_reorder(path, seed, drop=0):
    '''Rewrite a .vtu/.vtp with its cells in a different order, in place.

    Cell data is permuted with the cells, so the file describes exactly the same
    physical field -- only the order it is stored in changes. That is the shape
    of a series stitched together from two runs, and the case that a permutation
    built once from the first dump silently gets wrong.

    drop discards that many cells instead, giving the changed-count case.
    '''
    import pyvista as pv
    src = pv.read(path)
    n = src.n_cells
    order = np.random.default_rng(seed).permutation(n)
    if drop:
        order = order[:-drop]

    pts = np.asarray(src.points)
    if path.suffix == '.vtu':
        cells = src.cells.reshape(-1, 9)               # all hexahedra
        out = pv.UnstructuredGrid(cells[order].ravel(),
                                  np.full(len(order), pv.CellType.HEXAHEDRON),
                                  pts)
    else:
        faces = src.faces.reshape(-1, 5)               # all quads
        out = pv.PolyData(pts, faces[order].ravel())
    for k in src.cell_data:
        out.cell_data[k] = np.asarray(src.cell_data[k])[order]

    import vtk
    w = (vtk.vtkXMLUnstructuredGridWriter() if path.suffix == '.vtu'
         else vtk.vtkXMLPolyDataWriter())
    w.SetFileName(str(path))
    w.SetInputData(out)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToNone()
    w.SetHeaderTypeToUInt64()
    w.Write()


def test_openfoam_catches_a_reordered_second_dump(tmp_path):
    # THE check. Same cells, same field, different storage order -- so the count
    # check cannot see it and the reshape succeeds.
    from planktos import fluid
    path = _of_variant(tmp_path)
    _of_reorder(path / 'case_min_20' / 'internal.vtu', seed=0)
    with pytest.raises(RuntimeError, match='do not lie on the mesh'):
        fluid.OpenFOAMData(str(path))


def test_openfoam_catches_a_reordered_patch_on_the_second_dump(tmp_path):
    # The patches carry their own permutation, built the same way, and are
    # equally exposed.
    from planktos import fluid
    path = _of_variant(tmp_path)
    _of_reorder(path / 'case_min_20' / 'boundary' / 'inlet.vtp', seed=1)
    with pytest.raises(RuntimeError, match='do not lie on the mesh'):
        fluid.OpenFOAMData(str(path))


@pytest.mark.parametrize('INUM', [None, 4])
def test_openfoam_the_mesh_check_runs_on_the_windowed_path_too(tmp_path, INUM):
    # The check rides on the opening load, which is the whole series when it
    # fits and dumps 0..INUM when it does not. Dump 1 is in both, so streaming
    # must not skip it -- that would be the check quietly not running in the
    # exact configuration this branch exists for.
    from planktos import fluid
    path = _of_variant(tmp_path)
    _of_reorder(path / 'case_min_20' / 'internal.vtu', seed=0)
    with pytest.raises(RuntimeError, match='do not lie on the mesh'):
        fluid.OpenFOAMData(str(path), INUM=INUM)


def test_openfoam_the_mesh_check_costs_one_read_not_one_per_dump(tmp_path):
    # Cell coordinates are computed for dump 1 and no other. If a slide back to
    # the start re-verified, or every dump did, the cost would be per-dump and
    # the design decision silently undone.
    from planktos import fluid
    path = _of_variant(tmp_path)
    fd = fluid.OpenFOAMData(str(path), INUM=4)
    assert fd._mesh_verified
    seen = []
    real = _dataio.read_vtkxml_cell_data

    def spy(filename, arrays=('U',), load_cell_coordinates=True):
        seen.append(load_cell_coordinates)
        return real(filename, arrays, load_cell_coordinates)

    from planktos import fluid as fluid_mod
    fluid_mod._dataio.read_vtkxml_cell_data = spy
    try:
        fd.update_spline(fd.flow_times[-1])            # slide to the end
        fd.update_spline(fd.flow_times[0])             # and back to the start
    finally:
        fluid_mod._dataio.read_vtkxml_cell_data = real
    assert len(seen) > 0 and not any(seen)


def test_openfoam_a_changed_cell_count_raises_on_any_dump(tmp_path):
    # Not limited to the second dump: it costs one comparison, so every dump
    # gets it. case_min_30 is the third, past the coordinate check.
    from planktos import fluid
    path = _of_variant(tmp_path)
    _of_reorder(path / 'case_min_30' / 'internal.vtu', seed=0, drop=1)
    with pytest.raises(RuntimeError, match='cells; the mesh established'):
        fluid.OpenFOAMData(str(path))


def test_openfoam_a_changed_patch_cell_count_raises_on_any_dump(tmp_path):
    # A patch is indexed by a selection built at construction, so a shorter one
    # would take the wrong cells or raise a bare IndexError naming nothing.
    from planktos import fluid
    path = _of_variant(tmp_path)
    _of_reorder(path / 'case_min_30' / 'boundary' / 'inlet.vtp', seed=1, drop=1)
    with pytest.raises(RuntimeError, match='cells; the mesh established'):
        fluid.OpenFOAMData(str(path))


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
