'''Generate the tiny, committed IB2d-style fixtures used by test_io_loaders.py.

Run from the repository root to (re)create the fixture files:

    python tests/fixtures/_gen_fixtures.py

The outputs are deliberately small (a few vertices, a few time steps) and are
committed to the repo so the data loaders -- in particular the moving immersed
boundary import, which had no automated coverage -- can be exercised everywhere
without any external download. They are normally gitignored by the global *.vtk /
*.vertex rules; .gitignore has explicit exceptions for tests/fixtures/.

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


if __name__ == '__main__':
    d = write_moving_mesh()
    print("wrote moving mesh ->", d, sorted(p.name for p in d.iterdir()))
    v = write_static_vertex()
    print("wrote static vertex ->", v)
