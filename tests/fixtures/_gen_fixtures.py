'''Generate the tiny, committed fixtures used by the loader tests.

Run from the repository root to (re)create the fixture files:

    python tests/fixtures/_gen_fixtures.py

The outputs are deliberately small (a few vertices, a few time steps) and are
committed to the repo so the data loaders -- in particular the moving immersed
boundary import and the 3D windowed vtk path, neither of which had automated
coverage -- can be exercised everywhere without any external download. They are
normally gitignored by the global *.vtk / *.vertex rules; .gitignore has explicit
exceptions for tests/fixtures/.

Provenance lives here: edit this script and rerun to change a fixture, rather than
hand-editing the vtk files.
'''

import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

HERE = Path(__file__).parent

# Moving immersed boundary: a vertical open chain of 4 vertices (-> 3 segments)
# that translates +x by 0.5 each frame, over 3 frames. read_IB2d_mesh_data turns
# this into an ibmesh of shape (3, 3, 2, 2).
MOVING_BASE = np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
MOVING_FRAMES = 3
MOVING_DX = 0.5

# Static immersed boundary: the 4 corners of a square, as an IB2d .vertex file.
# 'adjacent' meshing -> 3 open segments; periodic=True closes it to 4.
STATIC_VERTS = np.array([[2., 2.], [4., 2.], [4., 4.], [2., 4.]])

# 3D rectilinear vtk series for VTK3dData. Eight dumps is the smallest series
# that lets a window of INUM=4 actually slide (5 points resident, 8 available),
# which is what the dynamic path needs to be exercised at all. Each file is ~2 kB.
#
# The velocity field is analytic so the tests can assert closed forms:
#     u = t        linear in t, uniform in space -> pins the timeline outright
#     v = x        steady, linear in x           -> pins spatial interpolation
#     w = t*z      varies in both
# TIME is written as field data (0, 1, ... 7), which is where VTK3dData reads the
# timeline from -- see _dataio.read_vtk_time_only.
VTK3D_SHAPE = (5, 4, 3)
VTK3D_EXTENT = (4.0, 3.0, 2.0)
VTK3D_NT = 8
VTK3D_TITLE = 'IBAMR_db_'


def write_moving_mesh(outdir=HERE / 'lagspts_min'):
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(MOVING_FRAMES):
        pts = MOVING_BASE.copy()
        pts[:, 0] += MOVING_DX * k
        pts3d = np.column_stack([pts, np.zeros(len(pts))])
        # read_vtk_Unstructured_Grid_Points expects a legacy UNSTRUCTURED_GRID of
        # singleton vertex points, so cast PolyData -> UnstructuredGrid.
        grid = pv.PolyData(pts3d).cast_to_unstructured_grid()
        grid.save(str(outdir / f'lagsPts.{k:04d}.vtk'), binary=False)
    return outdir


def write_static_vertex(outpath=HERE / 'mesh_min' / 'box.vertex'):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        f.write(f"{len(STATIC_VERTS)}\n")
        for x, y in STATIC_VERTS:
            f.write(f"{x} {y}\n")
    return outpath


# 2D IB2d fluid series for IB2dData -- the *reference* dynamic-loading path (2D is
# what has been exercised by hand), which had no automated coverage at all. IB2d
# writes structured-points vtk on a regular grid and omits the periodic endpoint in
# each direction, so Planktos wraps it back on: a 6x5 dump becomes a 7x6 field over
# a 6x5 domain.
#
#     u = t                steady in space -> u reads back the simulation time
#     v = sin(2*pi*x/Lx)   periodic in x, steady -> pins spatial layout and the wrap
#
# With dt=0.1 and print_dump=10 the timestamps come out as 0, 1, ... 7, matching u.
IB2D_SHAPE = (6, 5)          # grid points written per dump (endpoint omitted)
IB2D_NT = 8
IB2D_DT = 0.1
IB2D_PRINT_DUMP = 10
IB2D_SCALAR_NT = 3           # smaller series, only to cover the uX/uY read branch


def _ib2d_fields(t):
    nx, ny = IB2D_SHAPE
    X, _ = np.meshgrid(np.arange(nx) * 1.0, np.arange(ny) * 1.0, indexing='ij')
    return np.full(X.shape, t), np.sin(2 * np.pi * X / nx)


def _ib2d_image():
    nx, ny = IB2D_SHAPE
    return pv.ImageData(dimensions=(nx, ny, 1), spacing=(1., 1., 1.),
                        origin=(0., 0., 0.))


def _write_ib2d_velocity_dump(outdir, k, u, v):
    '''One u.####.vtk, as IB2d writes them.'''
    grid = _ib2d_image()
    # set_vectors, not grid['vel']: read_vtk_Structured_Points prefers scalars
    # when both are present, and would take the wrong branch.
    grid.point_data.set_vectors(
        np.stack([u.ravel(order='F'), v.ravel(order='F'),
                  np.zeros(u.size)], axis=1), 'vel')
    grid.save(str(outdir / f'u.{k:04d}.vtk'), binary=False)


def write_ib2d_fluid(outdir=HERE / 'ib2d_fluid_min'):
    '''Vector form: one u.####.vtk per dump.'''
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(IB2D_NT):
        _write_ib2d_velocity_dump(outdir, k, *_ib2d_fields(float(k)))
    return outdir


def write_ib2d_fluid_scalar(outdir=HERE / 'ib2d_fluid_scalar_min'):
    '''Scalar form: uX.####.vtk and uY.####.vtk per dump.'''
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(IB2D_SCALAR_NT):
        for name, field in zip(('uX', 'uY'), _ib2d_fields(float(k))):
            grid = _ib2d_image()
            grid[name] = field.ravel(order='F')
            grid.set_active_scalars(name)
            grid.save(str(outdir / f'{name}.{k:04d}.vtk'), binary=False)
    return outdir


# 2D IB2d fluid series that also ships VORTICITY -- Omega.####.vtk beside the
# u dumps, which is what IB2d writes when input2d asks for it. This is the fixture
# for run_persistence.md section 3.3: per-dump vorticity, sourced from disk under
# a sliding window instead of recomputed from a velocity field that is no longer
# resident.
#
# The velocity is chosen so vorticity VARIES IN TIME and is nonlinear in it:
#
#     u = sin(t) * sin(2*pi*y/Ly)
#     v = t**2   * sin(2*pi*x/Lx)
#
# so vort = dv/dx - du/dy carries a t**2 term and a sin(t) term. A reader that
# took the nearest dump instead of blending the two bracketing ones would pass
# against a steady field and fails against this one, which is the whole point.
# Both components are genuinely periodic on this grid (sin(2*pi) == sin(0)), so
# the wrap IB2d omits and Planktos restores is exact.
#
# Omega holds the CENTRAL-DIFFERENCE curl of the wrapped field, not the analytic
# one, and is written here by pyvista rather than through Planktos' own writer --
# deliberately, on both counts. What the tests using it assert is that blending
# per-dump fields with the interpolator's own weights reproduces the curl of the
# interpolated velocity (exact, by linearity); comparing a finite difference
# against an analytic derivative would instead measure the discretization, which
# is a different question with a nonzero answer. Section 3.3 records the separate
# empirical finding that for real IB2d data the solver's own Omega and Planktos'
# curl agree to 0.00%.
IB2D_VORT_NT = 8


def _ib2d_vort_fields(t):
    nx, ny = IB2D_SHAPE
    X, Y = np.meshgrid(np.arange(nx) * 1.0, np.arange(ny) * 1.0, indexing='ij')
    u = np.sin(t) * np.sin(2 * np.pi * Y / ny)
    v = t ** 2 * np.sin(2 * np.pi * X / nx)
    return u, v


def _ib2d_vorticity(t):
    """The curl IB2dData will compute, stripped back to IB2d's own grid.

    Built by wrapping the field exactly as the loader does, differencing on the
    wrapped grid (which is where the periodic wrap is what makes the outermost
    ring right), and then dropping the duplicated end lines again -- so what
    lands on disk is in the source's convention, one cell short in each
    direction, the same as u.
    """
    from planktos import fluid as _fluid
    nx, ny = IB2D_SHAPE
    u, v = _ib2d_vort_fields(t)
    grid = (np.arange(nx) * 1.0, np.arange(ny) * 1.0)
    flow, fpts, _ = _fluid._wrap_flow([u, v], grid, periodic_dim=(True, True))
    vort = _fluid._vorticity_from_field(flow, fpts, (True, True))
    return _fluid._unwrap_scalar(vort, (True, True))


def write_ib2d_fluid_with_vorticity(outdir=HERE / 'ib2d_fluid_vort_min'):
    """u.####.vtk plus Omega.####.vtk, as IB2d writes both: ascii structured
    points, the scalar array named for the quantity."""
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(IB2D_VORT_NT):
        t = float(k)
        _write_ib2d_velocity_dump(outdir, k, *_ib2d_vort_fields(t))

        omega = _ib2d_vorticity(t)
        ogrid = _ib2d_image()
        ogrid['Omega'] = omega.ravel(order='F')
        ogrid.set_active_scalars('Omega')
        ogrid.save(str(outdir / f'Omega.{k:04d}.vtk'), binary=False)
    return outdir


def vtk3d_grid():
    '''The coordinate arrays of the 3D fixture, for tests to assert against.'''
    return [np.linspace(0, VTK3D_EXTENT[d], VTK3D_SHAPE[d]) for d in range(3)]


def write_vtk3d_series(outdir=HERE / 'vtk3d_min'):
    outdir.mkdir(parents=True, exist_ok=True)
    x, y, z = vtk3d_grid()
    X, _, Z = np.meshgrid(x, y, z, indexing='ij')
    for k in range(VTK3D_NT):
        t = float(k)
        grid = pv.RectilinearGrid(x, y, z)
        u = np.full(X.shape, t)
        v = X.copy()
        w = t * Z
        # RectilinearGrid points run with x fastest, matching ravel(order='F')
        # over arrays built with indexing='ij'.
        grid['vel'] = np.stack([u.ravel(order='F'), v.ravel(order='F'),
                                w.ravel(order='F')], axis=1)
        grid.set_active_vectors('vel')
        # read_vtk_Rectilinear_Grid_Vector (and read_vtk_time_only) look for a
        # single-valued field-data array named TIME.
        grid.field_data['TIME'] = np.array([t])
        grid.save(str(outdir / f'{VTK3D_TITLE}{k:03d}.vtk'), binary=False)
    return outdir


# OpenFOAM-style multiblock series, for the .vtm.series / .vtm / cell-data readers.
# Structurally a miniature of the Phase 2 oral-arm export (see
# docs/notes/openfoam_oral_arm_dataset.md): a uniform Cartesian box of hexahedra
# carrying CELL data, wrapped in per-timestep .vtm manifests that name an interior
# .vtu plus inlet/outlet/walls .vtp boundary patches, indexed by a .vtm.series JSON.
#
# Three properties are deliberate, and the fixture is worth little without them:
#
#   * Cell order is SCRAMBLED by a fixed permutation. The real export's cells are
#     not lexicographic either, so a loader that reshapes without reordering must
#     fail here rather than pass by accident.
#   * Two declared dumps are ABSENT (one interior hole, one end truncation), with
#     their .vtm manifests still present -- exactly the truncated-transfer state of
#     the real data. Gap tolerance has to have something to find.
#   * Fields are analytic, so tests assert closed forms:
#         U         = (t, x, t*z)   u reads back the time, v pins spatial x
#         vorticity = (z, y, x)     steady, and distinct from U, so array
#                                   selection cannot silently return the wrong one
#         p         = t + y         a scalar, to pin the scalar branch
#     On the walls U is exactly zero (no-slip), as in the real export -- which is
#     what makes padding the lateral shell with zeros exact rather than an
#     approximation.
OF_CELLS = (4, 4, 5)                 # cells in x, y, z
OF_ORIGIN = (0., 0., 0.)
OF_EXTENT = (1., 1., 2.)             # full domain, walls included
OF_TITLE = 'case_min'
OF_DUMPS = (10, 20, 30, 40, 50, 60, 70, 80)     # dump numbers, as written
OF_TIMES = tuple(float(k) for k in range(len(OF_DUMPS)))
OF_ABSENT = (60, 80)                 # interior hole at t=5, end truncation at t=7
OF_SEED = 20260810                   # fixes the cell-order scramble


def openfoam_grid():
    '''Cell-center coordinate arrays of the interior, for tests to assert against.

    These are inset half a cell from the domain edges, which is the whole reason
    the boundary patches have to be spliced back on.
    '''
    return [OF_ORIGIN[d] + (np.arange(OF_CELLS[d]) + 0.5) *
            (OF_EXTENT[d] - OF_ORIGIN[d]) / OF_CELLS[d] for d in range(3)]


def _of_fields(centers, t):
    '''The analytic fields above, evaluated at an (N,3) array of cell centers.'''
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    U = np.stack([np.full(len(centers), t), x, t * z], axis=1)
    vort = np.stack([z, y, x], axis=1)
    return U, vort, t + y


def _of_interior():
    '''The interior hexahedral mesh, cells scrambled out of lexicographic order.

    Returns the pyvista grid and its cell centers in the scrambled file order.
    '''
    nx, ny, nz = OF_CELLS
    axes = [np.linspace(OF_ORIGIN[d], OF_EXTENT[d], OF_CELLS[d] + 1)
            for d in range(3)]
    # Corner points, x fastest -- the layout pid() below indexes into.
    P = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3,
                                                                    order='F')
    nxp, nyp = nx + 1, ny + 1

    def pid(i, j, k):
        return i + j * nxp + k * nxp * nyp

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # VTK_HEXAHEDRON node order: bottom face, then top face
                cells.append([8, pid(i, j, k), pid(i+1, j, k),
                              pid(i+1, j+1, k), pid(i, j+1, k),
                              pid(i, j, k+1), pid(i+1, j, k+1),
                              pid(i+1, j+1, k+1), pid(i, j+1, k+1)])
    cells = np.array(cells)
    perm = np.random.default_rng(OF_SEED).permutation(len(cells))
    cells = cells[perm]

    grid = pv.UnstructuredGrid(cells.ravel(),
                               np.full(len(cells), pv.CellType.HEXAHEDRON), P)
    return grid, np.asarray(grid.cell_centers().points)


def _of_patch(axis, side):
    '''One boundary patch as quads on the plane `axis` = min (side 0) / max (1).

    Cell centers land on the same in-plane lattice as the interior, which is what
    lets the caps splice on exactly instead of being interpolated.
    '''
    tan = [d for d in range(3) if d != axis]
    axes = [np.linspace(OF_ORIGIN[d], OF_EXTENT[d], OF_CELLS[d] + 1)
            for d in range(3)]
    a, b = axes[tan[0]], axes[tan[1]]
    plane = OF_ORIGIN[axis] if side == 0 else OF_EXTENT[axis]

    pts = np.zeros(((len(a)) * (len(b)), 3))
    A, B = np.meshgrid(a, b, indexing='ij')
    pts[:, tan[0]] = A.ravel(order='F')
    pts[:, tan[1]] = B.ravel(order='F')
    pts[:, axis] = plane

    na = len(a)
    faces = []
    for jb in range(len(b) - 1):
        for ia in range(na - 1):
            p = ia + jb * na
            faces.append([4, p, p + 1, p + 1 + na, p + na])
    return pv.PolyData(pts, np.array(faces).ravel())


def _of_boundary():
    '''inlet (z min), outlet (z max), and walls (the four lateral planes).

    walls is a single PolyData holding all four planes, as foamToVTK writes it.
    '''
    inlet = _of_patch(2, 0)
    outlet = _of_patch(2, 1)
    walls = _of_patch(0, 0) + _of_patch(0, 1) + _of_patch(1, 0) + _of_patch(1, 1)
    return inlet, outlet, walls


def _of_save(dataset, path):
    '''Write a .vtu/.vtp the way foamToVTK does: inline base64 with UInt64 size
    headers and no compressor.

    pyvista's save() gives no control here and defaults to zlib with UInt32
    headers. The difference is invisible through vtk's readers, but it is exactly
    what the planned direct-XML fast path for U depends on (note
    docs/notes/openfoam_oral_arm_dataset.md sec 7) -- and that would be developed
    against this fixture, so the fixture has to be honest about the byte layout.
    '''
    writer = (vtk.vtkXMLUnstructuredGridWriter() if path.suffix == '.vtu'
              else vtk.vtkXMLPolyDataWriter())
    writer.SetFileName(str(path))
    writer.SetInputData(dataset)
    writer.SetDataModeToBinary()        # inline base64, not an appended blob
    writer.SetCompressorTypeToNone()
    writer.SetHeaderTypeToUInt64()
    writer.Write()


def _write_vtm(path, dumpdir, time):
    '''The per-timestep manifest. Hand-written so the fixture pins the exact XML
    shape the reader parses -- the nested boundary Block and the TimeValue field
    data -- rather than whatever a writer happens to emit.'''
    path.write_text(
        "<?xml version='1.0'?>\n"
        "<VTKFile type='vtkMultiBlockDataSet' version='1.0' "
        "byte_order='LittleEndian' header_type='UInt64'>\n"
        "  <vtkMultiBlockDataSet>\n"
        f"    <DataSet name='internal' file='{dumpdir}/internal.vtu' />\n"
        "    <Block name='boundary'>\n"
        f"      <DataSet name='inlet' file='{dumpdir}/boundary/inlet.vtp' />\n"
        f"      <DataSet name='outlet' file='{dumpdir}/boundary/outlet.vtp' />\n"
        f"      <DataSet name='walls' file='{dumpdir}/boundary/walls.vtp' />\n"
        "    </Block>\n"
        "  </vtkMultiBlockDataSet>\n"
        "  <FieldData>\n"
        "    <DataArray type='Float32' Name='TimeValue' NumberOfTuples='1' "
        "format='ascii'>\n"
        f"{time}\n"
        "    </DataArray>\n"
        "  </FieldData>\n"
        "</VTKFile>\n")


def write_openfoam_series(outdir=HERE / 'openfoam_min'):
    outdir.mkdir(parents=True, exist_ok=True)
    interior, int_centers = _of_interior()
    inlet, outlet, walls = _of_boundary()

    for num, t in zip(OF_DUMPS, OF_TIMES):
        dumpdir = f'{OF_TITLE}_{num}'
        # The manifest is written for every declared dump, including the ones
        # whose data is missing -- that is the state a truncated transfer leaves.
        _write_vtm(outdir / (dumpdir + '.vtm'), dumpdir, t)
        if num in OF_ABSENT:
            continue

        U, vort, p = _of_fields(int_centers, t)
        interior.cell_data['U'] = U
        interior.cell_data['vorticity'] = vort
        interior.cell_data['p'] = p
        interior.field_data['TimeValue'] = np.array([t], dtype=np.float32)
        (outdir / dumpdir).mkdir(exist_ok=True)
        _of_save(interior, outdir / dumpdir / 'internal.vtu')

        bdir = outdir / dumpdir / 'boundary'
        bdir.mkdir(exist_ok=True)
        for name, patch in (('inlet', inlet), ('outlet', outlet),
                            ('walls', walls)):
            cen = np.asarray(patch.cell_centers().points)
            pU, pvort, pp = _of_fields(cen, t)
            if name == 'walls':
                pU = np.zeros_like(pU)      # no-slip, exactly zero
            patch.cell_data['U'] = pU
            patch.cell_data['vorticity'] = pvort
            patch.cell_data['p'] = pp
            patch.field_data['TimeValue'] = np.array([t], dtype=np.float32)
            _of_save(patch, bdir / (name + '.vtp'))

    (outdir / (OF_TITLE + '.vtm.series')).write_text(json.dumps(
        {'file-series-version': '1.0',
         'files': [{'name': f'{OF_TITLE}_{num}.vtm', 'time': t}
                   for num, t in zip(OF_DUMPS, OF_TIMES)]}, indent=2) + '\n')
    return outdir


if __name__ == '__main__':
    d = write_moving_mesh()
    print("wrote moving mesh ->", d, sorted(p.name for p in d.iterdir()))
    v = write_static_vertex()
    print("wrote static vertex ->", v)
    g = write_vtk3d_series()
    print("wrote 3D vtk series ->", g, sorted(p.name for p in g.iterdir()))
    f = write_ib2d_fluid()
    print("wrote IB2d fluid (vector) ->", f, sorted(p.name for p in f.iterdir()))
    s = write_ib2d_fluid_scalar()
    print("wrote IB2d fluid (scalar) ->", s, sorted(p.name for p in s.iterdir()))
    w = write_ib2d_fluid_with_vorticity()
    print("wrote IB2d fluid + vorticity ->", w,
          sorted(p.name for p in w.iterdir()))
    o = write_openfoam_series()
    print("wrote OpenFOAM-style series ->", o,
          sorted(p.name for p in o.iterdir()))
