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

from pathlib import Path

import numpy as np
import pyvista as pv

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


def write_ib2d_fluid(outdir=HERE / 'ib2d_fluid_min'):
    '''Vector form: one u.####.vtk per dump.'''
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(IB2D_NT):
        u, v = _ib2d_fields(float(k))
        grid = _ib2d_image()
        # set_vectors, not grid['vel']: read_vtk_Structured_Points prefers scalars
        # when both are present, and would take the wrong branch.
        grid.point_data.set_vectors(
            np.stack([u.ravel(order='F'), v.ravel(order='F'),
                      np.zeros(u.size)], axis=1), 'vel')
        grid.save(str(outdir / f'u.{k:04d}.vtk'), binary=False)
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
