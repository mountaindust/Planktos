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
