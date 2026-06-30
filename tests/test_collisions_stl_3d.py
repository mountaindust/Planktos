'''End-to-end 3D immersed-boundary collision against a real STL file.

test_collisions_static_3d.py exercises the triangle-mesh project-and-slide by
calling _ibc directly on in-memory meshes. This covers the full pipeline instead:
write a small closed STL (a cube -- convex, so "no penetration" simply means
"never strictly inside the box"), load it with Environment.read_stl_mesh_data, and
drive a Swarm into it with Swarm.move(). STL files are gitignored, so the fixture
is generated at test time; this needs the optional numpy-stl dependency, so the
module skips if it is absent.

A head-on hit makes the projected slide vector zero, tripping a benign "invalid
value in divide" RuntimeWarning in _ibc (the result is still correct); filtered
here as in test_collisions_static_3d.py.
'''

import sys
from pathlib import Path

import numpy as np
import pytest

import planktos

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h

stlmesh = pytest.importorskip('stl.mesh')

pytestmark = pytest.mark.filterwarnings('ignore:invalid value encountered in')

CUBE_LO, CUBE_HI = 4.0, 6.0


def _write_cube_stl(path, lo=CUBE_LO, hi=CUBE_HI):
    '''Write the axis-aligned cube [lo,hi]^3 as 12 triangles to an STL file.
    Winding is irrelevant: the collision code derives the surface normal from the
    agent's approach direction, not from the stored facet normals.'''
    v = np.array([[lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
                  [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi]], float)
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 3, 7), (0, 7, 4),
             (1, 5, 6), (1, 6, 2), (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7)]
    data = np.zeros(len(faces), dtype=stlmesh.Mesh.dtype)
    for i, (a, b, c) in enumerate(faces):
        data['vectors'][i] = v[[a, b, c]]
    stlmesh.Mesh(data).save(str(path))


def _cube_envir(tmp_path):
    '''A 10x10x10 noflux domain with the cube STL loaded as the immersed mesh.'''
    envir = planktos.Environment(Lx=10, Ly=10, Lz=10,
                                 x_bndry=('noflux',) * 2, y_bndry=('noflux',) * 2,
                                 z_bndry=('noflux',) * 2,
                                 flow=[np.zeros((3, 3, 3)) for _ in range(3)])
    stl_path = tmp_path / 'cube.stl'
    _write_cube_stl(stl_path)
    envir.read_stl_mesh_data(str(stl_path))
    return envir


def _inside_cube(p, lo=CUBE_LO, hi=CUBE_HI, eps=1e-6):
    return bool(np.all(p > lo + eps) and np.all(p < hi - eps))


def _drift_swarm(envir, init, drift):
    swrm = h._ConstDriftSwarm(swarm_size=len(init), envir=envir,
                              init=np.asarray(init, float), seed=1)
    swrm._drift = np.asarray(drift, float)
    return swrm


def test_stl_loads_as_triangle_mesh(tmp_path):
    envir = _cube_envir(tmp_path)
    assert envir.ibmesh.shape == (12, 3, 3)            # 6 faces x 2 triangles
    assert envir.ibmesh.dtype == np.float64
    # the longest edge of a triangulated edge-2 cube face is its diagonal, 2*sqrt(2).
    # (numpy-stl stores vertices as float32, so use a float32-scale tolerance.)
    assert envir.max_meshpt_dist == pytest.approx(2.0 * np.sqrt(2), abs=1e-5)


def test_agents_do_not_penetrate_stl_cube(tmp_path):
    envir = _cube_envir(tmp_path)
    # agents aimed at the -x, +x, and -y faces, plus two that miss the cube
    init = np.array([[2., 5., 5.], [2., 4.5, 5.5], [8., 5., 5.],
                     [5., 2., 5.], [2., 2., 5.], [5., 8., 5.]])
    drift = np.array([[1., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
                      [0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
    swrm = _drift_swarm(envir, init, drift)
    for _ in range(6):
        swrm.move(1.0, ib_collisions='sliding', silent=True)

    pos = np.asarray(swrm.positions)
    for i, p in enumerate(pos):
        assert not _inside_cube(p), f"agent {i} penetrated the cube: {p}"
    # the loaded mesh is actually being used: pinned agents re-collide every step
    assert np.any(np.asarray(swrm.ib_collision_idx) != -1), "no collision recorded"
    # the three aimed agents were stopped at their respective faces
    assert pos[0, 0] <= CUBE_LO + 1e-4, "-x face not enforced"
    assert pos[2, 0] >= CUBE_HI - 1e-4, "+x face not enforced"
    assert pos[3, 1] <= CUBE_LO + 1e-4, "-y face not enforced"
    # the two agents that miss the cube advance freely past it
    assert pos[4, 0] > CUBE_HI and pos[5, 0] == pytest.approx(5.0)


def test_agent_slides_along_stl_face(tmp_path):
    envir = _cube_envir(tmp_path)
    # hits the -x face and slides +y while pinned to the face (x stays ~CUBE_LO)
    swrm = _drift_swarm(envir, [[2.0, 4.3, 5.0]], [[1.0, 0.2, 0.0]])
    y0 = float(swrm.positions[0, 1])
    for _ in range(6):
        swrm.move(1.0, ib_collisions='sliding', silent=True)

    p = np.asarray(swrm.positions[0])
    assert not _inside_cube(p)
    assert p[0] <= CUBE_LO + 1e-4, "did not stay on the -x face"
    assert p[1] > y0 + 0.5, "did not slide tangentially along the face"
