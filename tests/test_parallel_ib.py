'''Correctness tests for the optional pool-based parallelization of
immersed-boundary (IB) collision detection.

What is proven here:
  serial (pool=None) == threads == processes
plus the hard invariant that no agent ever penetrates to the far side of a
boundary. Each test computes both the serial and the parallel results fresh and
compares them, so no stored reference data is needed. Since pool=None reproduces
the unparallelized behavior, pinning the parallel paths to it is sufficient. The
deterministic scenarios are defined in _ib_harness.py.

These tests are self-contained (no external data) and small/fast by design.
'''

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pytest

# Ensure the scenario helper (a non-test module in this dir) is importable.
sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as scen

# These full-simulation parallelization checks are the suite's slowest tests
# (process spawn + GIL-bound ODE work on the moving mesh). The no-penetration
# property they assert is also covered cheaply by the direct unit tests in
# test_collisions_static/moving, so they are gated behind --runslow to keep the
# default run fast.
pytestmark = pytest.mark.slow


def _assert_traj_equal(a, b):
    '''Assert two trajectory dicts are bit-identical (masked entries -> NaN).'''
    assert np.array_equal(a['pos'], b['pos'], equal_nan=True), "positions differ"
    assert np.array_equal(a['vel'], b['vel'], equal_nan=True), "velocities differ"
    assert np.array_equal(a['ib'], b['ib']), "ib_collision_idx differ"


@pytest.fixture(scope='module')
def serial_results():
    '''Run every scenario once with pool=None; reused across tests.'''
    return {name: scen.SCENARIOS[name](pool=None) for name in scen.SCENARIOS}


@pytest.mark.parametrize('name', list(scen.SCENARIOS))
def test_thread_pool_matches_serial(name, serial_results):
    '''A ThreadPoolExecutor must produce bit-identical results to serial.'''
    with ThreadPoolExecutor(max_workers=4) as pool:
        traj = scen.SCENARIOS[name](pool=pool)
    _assert_traj_equal(serial_results[name], traj)


@pytest.mark.parametrize('name', list(scen.SCENARIOS))
def test_process_pool_matches_serial(name, serial_results):
    '''A ProcessPoolExecutor must produce bit-identical results to serial.'''
    with ProcessPoolExecutor(max_workers=4) as pool:
        traj = scen.SCENARIOS[name](pool=pool)
    _assert_traj_equal(serial_results[name], traj)


@pytest.mark.parametrize('name', list(scen.SCENARIOS))
def test_no_penetration(name, serial_results):
    '''No agent that started on the near side of the wall ends up on the far
    side, at any recorded step. This is the hard correctness invariant.'''
    traj = serial_results[name]
    x = traj['pos'][..., 0]           # x-coordinate, shape (steps+1, N)
    eps = 1e-6
    if name == 'static':
        wall = scen.STATIC['wall_x']
        assert np.nanmax(x) <= wall + eps
    else:
        cfg = scen.MOVING
        times = np.linspace(0.0, cfg['dt'] * cfg['K'], cfg['T'])
        xpos = np.linspace(cfg['wall_x0'], cfg['wall_x1'], cfg['T'])
        rec_times = np.arange(x.shape[0]) * cfg['dt']
        wall_at = np.interp(rec_times, times, xpos)   # constant extrapolation
        overshoot = np.nanmax(x - wall_at[:, None])
        assert overshoot <= eps
