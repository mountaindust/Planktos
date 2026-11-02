'''Tiling: a tiled run equals the same run over data materialized to the tiled size.

The materialized "oracle" is built in ``_tiling_harness``, independently of the
code under test: fluid concatenated tile by tile, the grid extended point by point,
mesh copies translated in the documented order. Equality is bitwise, on the
dyadic geometry where every node and tile offset is exactly representable, with
the fluid periodic on every tiled axis (run_persistence.md 9.4.2).

The first section checks the oracle itself, against today's untiled code: that
it is the field it claims to be, and that wrapping a position onto the stored
tile reproduces it exactly. Everything after that compares tiled runs with it.
'''

import numpy as np
import pytest

import planktos
from planktos import _geom

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
