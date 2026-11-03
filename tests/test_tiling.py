'''Tiling: a tiled run equals the same run over data materialized to the tiled size.

The materialized "oracle" is built in ``_tiling_harness``, independently of the
code under test: fluid concatenated tile by tile, the grid extended point by point,
mesh copies translated in the documented order. Equality is bitwise, on the
dyadic geometry where every node and tile offset is exactly representable, with
the fluid periodic on every tiled axis (run_persistence.md 9.4.2).

The first section checks the oracle itself, against untiled code: that it is the
field it claims to be, and that wrapping a position onto the stored tile
reproduces it exactly. Then come mesh copying (_geom.tile_mesh), tile_domain and
the refusals around it, and last the tiled runs against the oracle: interpolation,
trajectories in every temporal regime, derived fields, FTLE, means, and immersed
meshes -- the array of cylinders, a moving mesh, a 3D mesh.
'''

import copy
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import planktos
from planktos import _geom, fluid, motion

import _ib_harness as H
import _tiling_harness as T


def _random_positions(L, counts, n=400, seed=0):
    '''Positions over the whole tiled domain, plus grid nodes and every seam.'''
    rng = np.random.default_rng(seed)
    big = np.array([L_*k for L_, k in zip(L, counts)])
    pts = rng.uniform(0, 1, (n, len(L))) * big
    # Exactly on grid nodes and on the seams, where a wrong wrap shows first.
    nodes = np.column_stack([
        rng.integers(0, k*(N - 1) + 1, 60) * (L_ / (N - 1))
        for L_, N, k in zip(L, T.DYADIC_N, counts)])
    seams = pts[:counts[0] + 1].copy()
    seams[:, 0] = np.arange(counts[0] + 1) * L[0]
    return np.concatenate([pts, nodes, seams])


# --------------------------------------------------------------------------- #
#                              the oracle itself                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('dim, counts', [(2, c) for c in T.TILINGS_2D]
                         + [(3, c) for c in T.TILINGS_3D])
def test_the_oracle_grid_is_the_evenly_spaced_tiled_grid(dim, counts):
    fp = T.dyadic_points(dim)
    tiled = T.tile_points(fp, counts)
    for n, (p, k) in enumerate(zip(tiled, counts)):
        N = T.DYADIC_N[n]
        h = T.DYADIC_L[n] / (N - 1)
        assert len(p) == k*(N - 1) + 1
        np.testing.assert_array_equal(p, np.arange(len(p)) * h)
        assert p[-1] == k * T.DYADIC_L[n]


@pytest.mark.parametrize('dim, counts', [(2, c) for c in T.TILINGS_2D]
                         + [(3, c) for c in T.TILINGS_3D])
@pytest.mark.parametrize('times', [None, (0.0, 0.25, 0.5, 0.75)])
def test_the_oracle_field_repeats_the_stored_tile(dim, counts, times):
    # Virtual index v along an axis holds stored index v mod (N-1).
    flow, _ = T.dyadic_flow(dim, times)
    for f in flow:
        big = T.tile_field(f, counts)
        lead = f.ndim - dim
        idx = np.ix_(*[np.arange(s) for s in big.shape[:lead]],
                     *[np.arange(s) % (N - 1)
                       for s, N in zip(big.shape[lead:], T.DYADIC_N)])
        np.testing.assert_array_equal(big, f[idx])


@pytest.mark.parametrize('mesh', ['static2d', 'static3d', 'moving'])
def test_the_oracle_mesh_is_copies_in_order_base_first(mesh):
    rng = np.random.default_rng(3)
    if mesh == 'static2d':
        L, counts, base = (1.0, 0.5), (3, 2), rng.uniform(0, 0.5, (4, 2, 2))
    elif mesh == 'static3d':
        L, counts = (1.0, 0.5, 0.25), (1, 2, 3)
        base = rng.uniform(0, 0.25, (5, 3, 3))
    else:
        L, counts, base = (1.0, 0.5), (3, 2), rng.uniform(0, 0.5, (3, 4, 2, 2))
    tiled = T.tile_mesh_copies(base, L, counts)
    N = base.shape[-3]
    offsets = T.tile_offsets(L, counts)
    assert len(offsets) == np.prod(counts)
    np.testing.assert_array_equal(offsets[0], 0)
    # x slowest: the second copy steps the last axis whose count exceeds 1.
    step = [a for a, k in enumerate(counts) if k > 1][-1]
    np.testing.assert_array_equal(offsets[1], np.eye(len(L))[step] * L[step])
    for t, off in enumerate(offsets):
        np.testing.assert_array_equal(tiled[..., t*N:(t + 1)*N, :, :],
                                      base + off)
    assert tiled.shape[-3] == N * len(offsets)


@pytest.mark.parametrize('dim, counts', [(2, c) for c in T.TILINGS_2D]
                         + [(3, c) for c in T.TILINGS_3D])
@pytest.mark.parametrize('times', [None, (0.0, 0.25, 0.5, 0.75)])
def test_wrapping_onto_the_stored_tile_reproduces_the_oracle_exactly(
        dim, counts, times):
    # The premise of position-wrapping tiling, checked on today's code: on the
    # dyadic geometry, the stored tile interpolated at a position wrapped onto
    # it is bit-identical to the materialized field interpolated at the
    # position itself. The untiled periodic environment wraps by % L.
    flow, fp = T.dyadic_flow(dim, times)
    ft = None if times is None else np.array(times)
    INUM = None if times is None else True
    stored = T.fluid_envir(flow, fp, ft, INUM=INUM)
    oracle = T.oracle_envir(flow, fp, counts, ft, INUM=INUM)
    pts = _random_positions(T.DYADIC_L[:dim], counts)
    for t in ([None] if times is None else [0.0, 0.3, 0.6125, 0.75]):
        a = stored.interpolate_flow(pts, time=t)
        b = oracle.interpolate_flow(pts, time=t)
        for ca, cb in zip(a, b):
            np.testing.assert_array_equal(ca, cb)


@pytest.mark.parametrize('counts', T.TILINGS_2D)
@pytest.mark.parametrize('times', [None, (0.0, 0.25, 0.5, 0.75)])
def test_the_stored_vorticity_tiled_is_the_oracle_vorticity_exactly(
        counts, times):
    # The periodic central difference at a seam reads across it, exactly as
    # the materialized grid's interior difference does.
    flow, fp = T.dyadic_flow(2, times)
    ft = None if times is None else np.array(times)
    INUM = None if times is None else True
    stored = T.fluid_envir(flow, fp, ft, INUM=INUM)
    oracle = T.oracle_envir(flow, fp, counts, ft, INUM=INUM)
    for t in ([None] if times is None else [0.0, 0.3]):
        np.testing.assert_array_equal(
            T.tile_field(stored.get_vorticity(time=t), counts),
            oracle.get_vorticity(time=t))


def test_an_untiled_oracle_is_the_stored_environment():
    flow, fp = T.dyadic_flow(2, (0.0, 0.25, 0.5, 0.75))
    ft = np.array((0.0, 0.25, 0.5, 0.75))
    runs = []
    for envir in (T.fluid_envir(flow, fp, ft, INUM=True),
                  T.oracle_envir(flow, fp, (1, 1), ft, INUM=True)):
        swrm = planktos.Swarm(swarm_size=20, envir=envir, seed=4)
        for _ in range(10):
            swrm.move(0.05, silent=True)
        runs.append(np.ma.filled(swrm.positions, np.nan))
    np.testing.assert_array_equal(*runs)


@pytest.mark.slow
def test_frame_capture_sees_every_frame_and_tells_them_apart():
    flow, fp = T.dyadic_flow(2, (0.0, 0.25, 0.5, 0.75))
    envir = T.fluid_envir(flow, fp, np.array((0.0, 0.25, 0.5, 0.75)), INUM=True)
    swrm = planktos.Swarm(swarm_size=20, envir=envir, seed=4)
    for _ in range(4):
        swrm.move(0.05, silent=True)
    shots = []
    for _ in range(2):
        with T.capture_movie_frames() as frames:
            swrm.plot_all(movie_filename='unused.mp4', frames=[0, 2, 4],
                          fluid='vort')
        shots.append(frames)
    assert len(shots[0]) == 3
    assert shots[0] == shots[1]
    assert len({f.digest for f in shots[0]}) == 3
    with T.capture_shown() as shown:
        swrm.plot(fluid='quiver', blocking=False)
    assert len(shown) == 1 and shown[0].array().shape[2] == 4


# --------------------------------------------------------------------------- #
#                     mesh copying (_geom.tile_mesh)                          #
# --------------------------------------------------------------------------- #

def _cylinder(center, r, n=24):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    return H.closed_polygon(np.column_stack([center[0] + r*np.cos(th),
                                             center[1] + r*np.sin(th)]))


def _random_mesh(kind, seed=3):
    '''Elements strictly inside the stored tile, so copies never meet.'''
    rng = np.random.default_rng(seed)
    if kind == 'static2d':
        return (1.0, 0.5), (3, 2), rng.uniform(0.05, 0.45, (4, 2, 2))
    if kind == 'static3d':
        return (1.0, 0.5, 0.25), (1, 2, 3), rng.uniform(0.02, 0.23, (5, 3, 3))
    return (1.0, 0.5), (3, 2), rng.uniform(0.05, 0.45, (3, 4, 2, 2))


@pytest.mark.parametrize('kind', ['static2d', 'static3d', 'moving'])
def test_tile_mesh_is_the_oracle_bitwise(kind):
    L, counts, base = _random_mesh(kind)
    tiled = _geom.tile_mesh(base, L, counts)
    oracle = T.tile_mesh_copies(base, L, counts)
    assert tiled.shape == oracle.shape
    np.testing.assert_array_equal(tiled, oracle)


@pytest.mark.parametrize('kind', ['static2d', 'static3d', 'moving'])
def test_tile_mesh_holds_each_copy_once_base_first(kind):
    # Regression: the pre-1.1 loop seeded the list with the base mesh and then
    # looped from copy (0, 0), so the base appeared twice; on a moving mesh it
    # offset [:, :, 0] -- both coordinates of each first vertex -- and
    # concatenated along time.
    L, counts, base = _random_mesh(kind)
    tiled = _geom.tile_mesh(base, L, counts)
    N = base.shape[-3]
    K = int(np.prod(counts))
    assert tiled.shape[-3] == K * N
    if kind == 'moving':
        assert tiled.shape[0] == base.shape[0]
    np.testing.assert_array_equal(tiled[..., :N, :, :], base)
    for t, off in enumerate(T.tile_offsets(L, counts)):
        for e in range(N):
            idx = t*N + e
            assert idx % N == e
            np.testing.assert_array_equal(tiled[..., idx, :, :],
                                          base[..., e, :, :] + off)


def test_tile_mesh_with_one_tile_is_the_mesh():
    L, _, base = _random_mesh('static2d')
    np.testing.assert_array_equal(_geom.tile_mesh(base, L, (1, 1)), base)


@pytest.mark.parametrize('counts', [(3,), (3, 2, 1), (0, 2), (2.5, 1), (-1, 1)])
def test_tile_mesh_refuses_bad_counts(counts):
    L, _, base = _random_mesh('static2d')
    with pytest.raises(ValueError):
        _geom.tile_mesh(base, L, counts)


# Along each tiled axis the mesh must lie strictly inside the tile.
#   L = (1, 0.5) in 2D.

def test_an_array_of_cylinders_tiles():
    base = _cylinder((0.5, 0.25), 0.15)
    assert _geom.tile_mesh(base, (1.0, 0.5), (3, 2)).shape == (6*24, 2, 2)


@pytest.mark.parametrize('seg, counts, axis', [
    ([[0.0, 0.1], [0.3, 0.4]], (2, 1), 'x'),    # a vertex on x = 0
    ([[0.6, 0.1], [1.0, 0.4]], (2, 1), 'x'),    # a vertex on x = L
    ([[0.9, 0.1], [1.1, 0.4]], (2, 1), 'x'),    # crossing x = L
    ([[0.2, 0.3], [0.4, 0.5]], (1, 2), 'y'),    # a vertex on y = L
    ([[0.2, -0.1], [0.4, 0.2]], (3, 3), 'y'),   # crossing y = 0
])
def test_a_mesh_reaching_a_tiled_face_is_refused(seg, counts, axis):
    mesh = np.array([[[0.2, 0.1], [0.3, 0.2]], seg])
    with pytest.raises(ValueError) as err:
        _geom.tile_mesh(mesh, (1.0, 0.5), counts)
    assert 'touch or cross its boundary' in str(err.value)
    assert ': 1 ({}).'.format(axis) in str(err.value)


def test_a_mesh_reaching_only_untiled_faces_is_accepted():
    wall = H.horizontal_wall(4, 0.25, 0.0, 1.0)
    assert _geom.tile_mesh(wall, (1.0, 0.5), (1, 3)).shape == (12, 2, 2)


def test_the_refusal_names_each_element_and_its_axes():
    mesh = np.array([[[0.2, 0.1], [0.3, 0.2]],
                     [[0.0, 0.2], [0.3, 0.5]],
                     [[0.5, 0.4], [0.6, 0.6]]])
    with pytest.raises(ValueError, match=r': 1 \(x, y\), 2 \(y\)\.'):
        _geom.tile_mesh(mesh, (1.0, 0.5), (2, 2))
    many = np.tile([[[0.0, 0.2], [0.3, 0.3]]], (12, 1, 1))
    with pytest.raises(ValueError, match=r'9 \(x\), and 2 more\.'):
        _geom.tile_mesh(many, (1.0, 0.5), (2, 1))


def test_a_moving_mesh_reaching_the_boundary_later_is_refused():
    # The first two frames stay inside; in the third the first segment
    # crosses x = L.
    frames = np.array([[[[0.6, 0.1], [0.8, 0.4]], [[0.2, 0.4], [0.3, 0.1]]],
                       [[[0.75, 0.1], [0.95, 0.4]], [[0.2, 0.4], [0.3, 0.1]]],
                       [[[0.9, 0.1], [1.1, 0.4]], [[0.2, 0.4], [0.3, 0.1]]]])
    _geom.tile_mesh(frames[:2], (1.0, 0.5), (2, 1))
    with pytest.raises(ValueError, match=r': 0 \(x\)\.'):
        _geom.tile_mesh(frames, (1.0, 0.5), (2, 1))


L3 = (1.0, 0.5, 0.25)


def _floor(z):
    '''Two triangles covering the tile's x-y extent at height z.'''
    c = [np.array([x, y, z]) for x, y in ((0, 0), (1, 0), (1, 0.5), (0, 0.5))]
    return np.array([[c[0], c[1], c[2]], [c[0], c[2], c[3]]])


def test_a_floor_tiles_along_z_only_in_3d():
    assert _geom.tile_mesh(_floor(0.1), L3, (1, 1, 3)).shape == (6, 3, 3)
    with pytest.raises(ValueError, match=r': 0 \(x, y\), 1 \(x, y\)\.'):
        _geom.tile_mesh(_floor(0.1), L3, (3, 2, 1))


def test_a_triangle_crossing_a_tiled_face_is_refused_in_3d():
    mesh = np.array([[[0.05, 0.1, 0.05], [0.05, 0.4, 0.05], [0.05, 0.25, 0.2]],
                     [[0.9, 0.2, 0.1], [1.2, 0.25, 0.1], [0.9, 0.3, 0.1]]])
    _geom.tile_mesh(mesh, L3, (1, 2, 3))
    with pytest.raises(ValueError, match=r': 1 \(x\)\.'):
        _geom.tile_mesh(mesh, L3, (2, 1, 1))


# --------------------------------------------------------------------------- #
#                         tile_domain and its guards                          #
# --------------------------------------------------------------------------- #

CASES = [(2, c) for c in T.TILINGS_2D] + [(3, c) for c in T.TILINGS_3D]
TIMES = np.arange(8) * 0.125
FIXTURES = Path(__file__).parent / 'fixtures'


def _tile(envir, counts):
    if len(counts) == 3:
        envir.tile_domain(counts[0], counts[1], z=counts[2])
    else:
        envir.tile_domain(*counts)
    return envir


class _MemoryDumps(fluid.FluidData):
    '''Fluid held in memory but served dump by dump through load_dumpfiles,
    so that an int INUM slides a window as it would over files.'''

    def __init__(self, flow, flow_points, flow_times, INUM):
        self._all = flow
        self.d_start, self.d_finish = 0, len(flow_times) - 1
        self.loaded_dump_bnds = (0, INUM)
        self.loaded_idx_bnds = (0, INUM)
        super().__init__([f[:INUM + 1] for f in flow], flow_points, flow_times,
                         INUM, periodic_dim=True)

    def load_dumpfiles(self, d_start, d_finish):
        return [f[d_start:d_finish + 1] for f in self._all]


def _pair(dim, counts, regime='static', mesh=None, ibmesh_times=None,
          bndry='periodic'):
    '''A tiled environment and its oracle, over the dyadic flow.

    ``regime`` is 'static', 'cubic', 'linear' or 'window'. A windowed run is
    set against an oracle holding everything, splined linearly as it is.
    '''

    times = None if regime == 'static' else TIMES
    flow, fp = T.dyadic_flow(dim, times)
    INUM = {'static': None, 'cubic': None, 'linear': True, 'window': 3}[regime]
    tiled = T.fluid_envir(flow, fp, times, None if regime == 'window' else INUM,
                          bndry=bndry)
    if regime == 'window':
        tiled.flow = _MemoryDumps(flow, fp, times, INUM)
    oracle = T.oracle_envir(flow, fp, counts, times,
                            True if regime == 'window' else INUM, bndry=bndry)
    if mesh is not None:
        T.set_mesh(tiled, mesh, ibmesh_times)
        T.set_mesh(oracle, T.tile_mesh_copies(mesh, T.DYADIC_L[:dim], counts),
                   ibmesh_times)
    _tile(tiled, counts)
    return tiled, oracle


def _ib2d_dyadic(tmp_path, INUM=None):
    src = tmp_path / 'ib2d_fluid_dyadic'
    shutil.copytree(FIXTURES / 'ib2d_fluid_dyadic', src)
    envir = planktos.Environment(x_bndry='periodic', y_bndry='periodic')
    envir.read_IB2d_fluid_data(str(src), dt=1/16, print_dump=4, INUM=INUM)
    return envir


@pytest.mark.parametrize('dim, counts', CASES)
def test_tile_domain_tiles_L_and_leaves_the_fluid_one_tile(dim, counts):
    flow, fp = T.dyadic_flow(dim, TIMES)
    envir = T.fluid_envir(flow, fp, TIMES, INUM=True)
    fshape = envir.flow.fshape
    _tile(envir, counts)
    assert envir.tiling == counts
    assert envir.flow.tiling == counts
    assert envir.L == [L*k for L, k in zip(T.DYADIC_L, counts)]
    assert envir.flow.fshape == fshape
    for a, b in zip(envir.flow.flow_points, fp):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(envir.flow.tiled_points, T.tile_points(fp, counts)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(envir.flow(0.25)[0], flow[0][2])


def test_tiling_leaves_the_fluids_own_L_alone(tmp_path):
    # The loader sets envir.L = flow.L, the same list.
    envir = _ib2d_dyadic(tmp_path)
    envir.tile_domain(3, 2)
    assert envir.L == [3.0, 1.0]
    assert envir.flow.L == [1.0, 0.5]


def test_an_untiled_environment_reports_one_tile_per_axis():
    flow, fp = T.dyadic_flow(3)
    envir = T.fluid_envir(flow, fp)
    assert envir.tiling == (1, 1, 1)
    assert envir.flow.tiling == (1, 1, 1)
    assert envir.flow.tiled_points is envir.flow.flow_points
    assert envir.flow.materialize_tiled(flow[0]) is flow[0]


def test_tiling_all_ones_changes_nothing():
    flow, fp = T.dyadic_flow(2)
    envir = T.fluid_envir(flow, fp)
    L = envir.L
    envir.tile_domain(1, 1)
    assert envir.L is L
    assert envir.tiling == (1, 1)
    envir.flow = envir.flow
    envir.tile_domain(3, 2)
    assert envir.tiling == (3, 2)


@pytest.mark.parametrize('kind', ['static2d', 'static3d', 'moving'])
def test_tile_domain_copies_the_mesh_into_every_tile(kind):
    L, counts, base = _random_mesh(kind)
    flow, fp = T.dyadic_flow(len(L))
    envir = T.fluid_envir(flow, fp)
    times = np.arange(3.0) if kind == 'moving' else None
    T.set_mesh(envir, base, times)
    dist = envir.max_meshpt_dist
    _tile(envir, counts)
    np.testing.assert_array_equal(envir.ibmesh,
                                  T.tile_mesh_copies(base, L, counts))
    assert envir.max_meshpt_dist == dist
    if kind == 'moving':
        np.testing.assert_array_equal(envir.ibmesh_times, times)


def test_tile_ibmesh_false_leaves_the_mesh_as_loaded():
    # Including one reaching the boundary, which only copying would refuse.
    mesh = np.concatenate([_cylinder((0.5, 0.25), 0.15),
                           H.wall_segments(2, 0.0, 0.0, 0.5)])
    flow, fp = T.dyadic_flow(2)
    envir = T.fluid_envir(flow, fp)
    T.set_mesh(envir, mesh)
    envir.tile_domain(3, 2, tile_ibmesh=False)
    np.testing.assert_array_equal(envir.ibmesh, mesh)
    assert envir.L == [3.0, 1.0]


def _snapshot(envir):
    return dict(L=list(envir.L), tiling=envir.tiling, flow=envir.flow,
                ibmesh=None if envir.ibmesh is None else envir.ibmesh.copy(),
                fluid_prov=copy.deepcopy(envir._fluid_provenance),
                mesh_prov=copy.deepcopy(envir._ibmesh_provenance))


def _assert_unchanged(envir, snap):
    assert envir.L == snap['L']
    assert envir.tiling == snap['tiling']
    assert envir.flow is snap['flow']
    np.testing.assert_array_equal(envir.ibmesh, snap['ibmesh'])
    assert envir._fluid_provenance == snap['fluid_prov']
    assert envir._ibmesh_provenance == snap['mesh_prov']


AFTER_TILING = {
    'analytic fluid': lambda e, src: e.set_brinkman_flow(
        alpha=66, h_p=0.1, U=0.5, dpdx=0.22306, res=11),
    'fluid file': lambda e, src: e.read_IB2d_fluid_data(
        str(src), dt=1/16, print_dump=4),
    'flow =': lambda e, src: setattr(e, 'flow', e.flow),
    'ibmesh =': lambda e, src: setattr(e, 'ibmesh', None),
    'mesh file': lambda e, src: e.read_IB2d_mesh_data(
        str(FIXTURES / 'mesh_min' / 'box.vertex')),
    'mesh modifier': lambda e, src: e.add_vertices_to_static_2D_ibmesh(),
    'mesh shift': lambda e, src: e.shift_ibmesh_to_match_LLC(),
    'tile again': lambda e, src: e.tile_domain(2, 1),
}


@pytest.mark.parametrize('action', list(AFTER_TILING))
def test_after_tiling_the_domain_cannot_change(action, tmp_path):
    src = tmp_path / 'ib2d'
    shutil.copytree(FIXTURES / 'ib2d_fluid_dyadic', src)
    flow, fp = T.dyadic_flow(2)
    envir = T.fluid_envir(flow, fp)
    envir.rho, envir.mu = 1000, 1e-3
    T.set_mesh(envir, _cylinder((0.5, 0.25), 0.15))
    envir.tile_domain(3, 2)
    snap = _snapshot(envir)
    match = 'already tiled' if action == 'tile again' else 'after tile_domain'
    with pytest.raises(ValueError, match=match):
        AFTER_TILING[action](envir, src)
    _assert_unchanged(envir, snap)


def _started(envir):
    swrm = planktos.Swarm(swarm_size=5, envir=envir, seed=1)
    swrm.move(0.05, silent=True)


REFUSED_TILINGS = {
    'z in 2D': (lambda e: e.tile_domain(2, 1, z=2), ValueError, 'only to a 3D'),
    'zero': (lambda e: e.tile_domain(0, 2), ValueError, 'at least 1'),
    'negative': (lambda e: e.tile_domain(2, -1), ValueError, 'at least 1'),
    'fractional': (lambda e: e.tile_domain(2.5, 1), ValueError, 'at least 1'),
    'run started': (lambda e: (_started(e), e.tile_domain(2, 1)), ValueError,
                    'before the run starts'),
    'mesh at the boundary': (lambda e: (T.set_mesh(e, H.wall_segments(
        2, 0.0, 0.1, 0.4)), e.tile_domain(2, 1)), ValueError, 'strictly inside'),
}


@pytest.mark.parametrize('case', list(REFUSED_TILINGS))
def test_a_refused_tiling_changes_nothing(case):
    flow, fp = T.dyadic_flow(2)
    envir = T.fluid_envir(flow, fp)
    action, exc, match = REFUSED_TILINGS[case]
    with pytest.raises(exc, match=match):
        action(envir)
    assert envir.L == [1.0, 0.5]
    assert envir.tiling == (1, 1)
    assert envir.flow.tiling == (1, 1)
    assert envir.ibmesh is None or envir.ibmesh.shape[0] == 2
    envir.flow = envir.flow


def test_tiling_while_recording_is_refused(tmp_path):
    flow, fp = T.dyadic_flow(2)
    envir = T.fluid_envir(flow, fp)
    planktos.Swarm(swarm_size=5, envir=envir, seed=1)
    with envir.record(str(tmp_path / 'run')):
        with pytest.raises(RuntimeError, match='while recording'):
            envir.tile_domain(2, 1)
    assert envir.L == [1.0, 0.5]
    assert envir.tiling == (1, 1)


def test_tiling_warns_where_the_fluid_is_not_flagged_periodic():
    flow, fp = T.dyadic_flow(2)
    envir = T.fluid_envir(flow, fp, periodic_dim=(True, False))
    with pytest.warns(UserWarning, match='along y, where the fluid is not flagged'):
        envir.tile_domain(3, 2)


def test_tiling_warns_where_periodic_edges_differ():
    flow, fp = T.dyadic_flow(2, TIMES)
    flow[1] = flow[1].copy()
    flow[1][5, -1, 3] += 0.01
    envir = T.fluid_envir(flow, fp, TIMES)
    with pytest.warns(UserWarning, match='along x, where the fluid is flagged '
                                         'periodic but its opposite edges differ'):
        envir.tile_domain(3, 2)


@pytest.mark.parametrize('counts, periodic', [((3, 2), True),
                                              ((5, 1), (True, False))])
def test_tiling_periodic_data_draws_no_warning(counts, periodic):
    flow, fp = T.dyadic_flow(2, TIMES)
    envir = T.fluid_envir(flow, fp, TIMES, periodic_dim=periodic)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        envir.tile_domain(*counts)


def test_non_periodic_fluid_repeats_the_stored_field_in_every_tile():
    # Inside each tile the velocity is the stored field at the position
    # wrapped onto the stored tile, so it jumps at the seams.
    fp = T.dyadic_points(2)
    X, Y = np.meshgrid(*fp, indexing='ij')
    flow = [X + 0.3*Y**2, X*Y]
    stored = T.fluid_envir(flow, fp, periodic_dim=False, bndry='zero')
    tiled = T.fluid_envir(flow, fp, periodic_dim=False, bndry='zero')
    with pytest.warns(UserWarning):
        tiled.tile_domain(3, 2)
    pts = _random_positions(T.DYADIC_L[:2], (3, 2))
    pts = pts[np.all(pts < [3.0, 1.0], axis=1)]
    np.testing.assert_array_equal(
        tiled.interpolate_flow(pts),
        stored.interpolate_flow(pts % np.array(T.DYADIC_L[:2])))
    # The tiled field reads a seam node from stored index 0, as the
    # simulation's wrap does: x = L here, and y = L at column 16.
    big = tiled.flow.materialize_tiled(flow[0])
    np.testing.assert_array_equal(big[31, :16], flow[0][31, :16])
    np.testing.assert_array_equal(big[32, :16], flow[0][0, :16])
    np.testing.assert_array_equal(big[31, 16], flow[0][31, 0])


# --------------------------------------------------------------------------- #
#                      a tiled run equals the oracle's                        #
# --------------------------------------------------------------------------- #

def _query_points(dim, counts, L):
    '''Inside and outside the tiled domain, on nodes and seams, and just below
    zero.'''
    rng = np.random.default_rng(2)
    outside = rng.uniform(-1.5, 2.5, (200, dim)) * np.array(L)
    return np.concatenate([_random_positions(T.DYADIC_L[:dim], counts), outside,
                           np.full((1, dim), -1e-17)])


@pytest.mark.parametrize('dim, counts', CASES)
@pytest.mark.parametrize('regime', ['static', 'cubic', 'linear', 'window'])
def test_a_tiled_fluid_interpolates_as_the_oracle_does(dim, counts, regime):
    tiled, oracle = _pair(dim, counts, regime)
    pts = _query_points(dim, counts, oracle.L)
    for t in ([None] if regime == 'static' else [0.0, 0.3, 0.8, 0.1, 0.875]):
        np.testing.assert_array_equal(tiled.interpolate_flow(pts, time=t),
                                      oracle.interpolate_flow(pts, time=t))


def _non_dyadic(counts):
    fp = (np.linspace(0, 0.7, 31), np.linspace(0, 0.45, 19))
    X, Y = np.meshgrid(*fp, indexing='ij')
    u = 0.4 + 0.2*np.sin(2*np.pi*X/0.7)*np.cos(2*np.pi*Y/0.45)
    v = 0.15*np.cos(2*np.pi*X/0.7)*np.sin(2*np.pi*Y/0.45)
    flow = [T.make_periodic(u, 2), T.make_periodic(v, 2)]
    tiled = T.fluid_envir(flow, fp)
    tiled.tile_domain(*counts)
    return tiled, T.oracle_envir(flow, fp, counts)


@pytest.mark.parametrize('counts', T.TILINGS_2D)
def test_on_a_non_dyadic_grid_the_oracle_agrees_to_roundoff(counts):
    tiled, oracle = _non_dyadic(counts)
    rng = np.random.default_rng(6)
    pts = rng.uniform(-0.5, 1.5, (300, 2)) * np.array(oracle.L)
    np.testing.assert_allclose(tiled.interpolate_flow(pts),
                               oracle.interpolate_flow(pts),
                               rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        tiled.flow.materialize_tiled(tiled.flow.get_vorticity()),
        oracle.flow.get_vorticity(), rtol=1e-12, atol=1e-11)
    tiled.calculate_mag_gradient()
    oracle.calculate_mag_gradient()
    for a, b in zip(tiled.mag_grad, oracle.mag_grad):
        np.testing.assert_allclose(tiled.flow.materialize_tiled(a), b,
                                   rtol=1e-12, atol=1e-11)


def _history(swrm, steps, dt):
    '''Positions, velocities, accelerations and collision indexes per step.'''
    hist = []
    for _ in range(steps):
        swrm.move(dt, silent=True)
        hist.append([np.ma.filled(a, np.nan).copy() for a in
                     (swrm.positions, swrm.velocities, swrm.accelerations)]
                    + [np.array(swrm.ib_collision_idx)])
    return hist


def _assert_same_history(a, b):
    assert len(a) == len(b)
    for step_a, step_b in zip(a, b):
        for x, y in zip(step_a, step_b):
            np.testing.assert_array_equal(x, y)


def _swarm(envir, cls=planktos.Swarm, n=40, seed=5, cov=1e-3, **kwargs):
    swrm = cls(swarm_size=n, envir=envir, seed=seed, **kwargs)
    swrm.shared_props['cov'] = swrm.shared_props['cov'] * cov
    return swrm


@pytest.mark.parametrize('dim, counts', CASES)
@pytest.mark.parametrize('regime', ['static', 'cubic', 'linear', 'window'])
def test_a_tiled_run_is_the_oracle_run_bitwise(dim, counts, regime):
    runs = [_history(_swarm(envir), 20, 0.05)
            for envir in _pair(dim, counts, regime)]
    _assert_same_history(*runs)


class _Inertial(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(
            self, dt, ode=motion.inertial_particles(self))


class _Tracer(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(
            self, dt, ode=motion.tracer_particles(self))


@pytest.mark.parametrize('cls', [_Inertial, _Tracer])
@pytest.mark.parametrize('dim, counts', [(2, (3, 2)), (3, (1, 2, 3))])
def test_inertial_and_tracer_runs_are_the_oracle_runs_bitwise(cls, dim, counts):
    runs = []
    for envir in _pair(dim, counts, 'cubic'):
        envir.rho, envir.mu = 1000, 1e-3
        envir.char_L, envir.U = 0.1, 0.5
        swrm = _swarm(envir, cls, n=20, seed=12, cov=1e-4)
        swrm.shared_props['R'] = 0.5
        swrm.shared_props['diam'] = 1e-3
        runs.append(_history(swrm, 15, 0.05))
    _assert_same_history(*runs)


def test_a_windowed_file_series_tiled_is_the_oracle_bitwise(tmp_path):
    # The IB2d loader slides its window over files; the oracle is the same
    # series read whole and materialized.
    counts = (3, 2)
    tiled = _ib2d_dyadic(tmp_path, INUM=3)
    tiled.tile_domain(*counts)
    whole = _ib2d_dyadic(tmp_path / 'whole', INUM=True)
    flow = [T.tile_field(f, counts) for f in whole.flow.get_raw_loaded_data()]
    oracle = T.fluid_envir(flow, T.tile_points(whole.flow.flow_points, counts),
                           whole.flow.flow_times, INUM=True)
    runs = [_history(_swarm(envir), 35, 0.05) for envir in (tiled, oracle)]
    _assert_same_history(*runs)


@pytest.mark.parametrize('counts', T.TILINGS_2D)
@pytest.mark.parametrize('t', [0.0, 0.3])
def test_derived_fields_tiled_are_the_oracle_fields_bitwise(counts, t):
    tiled, oracle = _pair(2, counts, 'cubic')
    tiled.time = oracle.time = t
    np.testing.assert_array_equal(
        tiled.flow.materialize_tiled(tiled.flow.get_vorticity(time=t)),
        oracle.flow.get_vorticity(time=t))
    for a, b in zip(tiled.flow.calculate_DuDt(time=t),
                    oracle.flow.calculate_DuDt(time=t)):
        np.testing.assert_array_equal(tiled.flow.materialize_tiled(a), b)
    tiled.calculate_mag_gradient()
    oracle.calculate_mag_gradient()
    for a, b in zip(tiled.mag_grad, oracle.mag_grad):
        np.testing.assert_array_equal(tiled.flow.materialize_tiled(a), b)


def test_ftle_on_the_default_grid_is_the_oracle_ftle_bitwise():
    tiled, oracle = _pair(2, (3, 2), 'cubic')
    for envir in (tiled, oracle):
        envir.calculate_FTLE(T=0.2, dt=0.05)
    assert tiled.FTLE_grid_dim == oracle.FTLE_grid_dim == (97, 33)
    for name in ('FTLE_largest', 'FTLE_smallest', 'FTLE_loc', 'FTLE_loc_end'):
        np.testing.assert_array_equal(
            np.ma.filled(getattr(tiled, name), np.nan),
            np.ma.filled(getattr(oracle, name), np.nan))


@pytest.mark.parametrize('dim, counts', CASES)
@pytest.mark.parametrize('regime', ['static', 'cubic', 'linear', 'window'])
def test_tiled_means_are_the_oracle_means_to_roundoff(dim, counts, regime):
    # The weighted sum runs in a different order from np.mean over the
    # materialized field.
    tiled, oracle = _pair(dim, counts, regime)
    for t in ([None] if regime == 'static' else [0.3, 0.8, 0.1, 0.875]):
        np.testing.assert_allclose(tiled.flow.get_mean_velocity(time=t),
                                   oracle.flow.get_mean_velocity(time=t),
                                   rtol=1e-13, atol=1e-15)
        if t is not None:
            tiled.time = oracle.time = t
        np.testing.assert_allclose(tiled.get_mean_fluid_speed(),
                                   oracle.get_mean_fluid_speed(), rtol=1e-13)


def test_an_array_of_cylinders_is_the_oracle_bitwise_and_never_entered():
    # The motivating case: one cylinder tiled (3, 2) against six placed by hand,
    # agents sliding around them for long enough that most meet a copy.
    counts = (3, 2)
    base = _cylinder((0.5, 0.25), 0.15)
    tiled, oracle = _pair(2, counts, 'linear', mesh=base)
    centers = np.array([[0.5 + i, 0.25 + 0.5*j]
                        for i in range(3) for j in range(2)])
    rng = np.random.default_rng(11)
    pts = rng.uniform(0, 1, (800, 2)) * np.array(oracle.L)
    near = np.linalg.norm(pts[:, None] - centers[None], axis=2).min(axis=1)
    init = pts[near > 0.17][:300]
    runs = []
    for envir in (tiled, oracle):
        swrm = _swarm(envir, n=len(init), seed=9, cov=5e-3, init=init.copy())
        runs.append(_history(swrm, 60, 0.02))
    _assert_same_history(*runs)

    N = len(base)
    hit = np.array([step[3] for step in runs[0]])
    on_copies = (hit >= N).sum()
    on_base = ((hit >= 0) & (hit < N)).sum()
    assert on_copies > 2*on_base > 0
    copies = [tiled.ibmesh[t*N:(t + 1)*N, 0] for t in range(6)]
    for step in runs[0]:
        pos = step[0]
        for c, poly in zip(centers, copies):
            close = np.linalg.norm(pos - c, axis=1) < 0.16
            for P in pos[close]:
                H.assert_outside_polygon(P, poly)


def test_a_moving_mesh_tiled_is_the_oracle_bitwise():
    # Two segments swinging inside the stored tile, in phase in every copy.
    frames = np.array([[[[0.3, 0.1], [0.4, 0.3]], [[0.6, 0.1], [0.7, 0.3]]],
                       [[[0.3, 0.1], [0.5, 0.3]], [[0.6, 0.1], [0.8, 0.3]]],
                       [[[0.3, 0.1], [0.6, 0.3]], [[0.6, 0.1], [0.9, 0.3]]]])
    times = np.array([0.0, 0.4, 0.8])
    runs = []
    for envir in _pair(2, (3, 2), 'cubic', mesh=frames, ibmesh_times=times):
        runs.append(_history(_swarm(envir, n=60, seed=3, cov=5e-3), 20, 0.04))
    _assert_same_history(*runs)
    assert (np.array([s[3] for s in runs[0]]) >= 2).any()


def test_a_3d_mesh_tiled_is_the_oracle_bitwise():
    box = H.extruded_polygon(np.array([[0.4, 0.2], [0.6, 0.2], [0.6, 0.3],
                                       [0.4, 0.3], [0.4, 0.2]]), 0.05, 0.2)
    runs = []
    for envir in _pair(3, (3, 2, 1), 'static', mesh=box):
        runs.append(_history(_swarm(envir, n=60, seed=4, cov=5e-3), 20, 0.05))
    _assert_same_history(*runs)
    assert (np.array([s[3] for s in runs[0]]) >= 0).any()
