'''Periodic domain boundary x immersed boundary interaction.

When an agent crosses a periodic domain boundary, Swarm._domain_BC_loop wraps it
to the opposite side and then re-runs the immersed-boundary collision check on the
re-entry trajectory (reconstructed from the far edge using the step velocity).
This is a subtle interaction -- an agent can wrap across the domain and immediately
meet a wall on the far side -- that had no coverage.

Each test drives a real Swarm.move() with zero diffusion and a constant drift, so
the single step is deterministic and the post-collision position is known. The
collision back-off places a stopped agent ~1e-5..1e-4 *inside* the boundary it
struck (never past it); POS_ATOL is sized for that. A head-on hit makes the
projected slide vector zero, tripping a benign "invalid value in divide"
RuntimeWarning in _ibc (the result is still correct); it is filtered here as in
test_collisions_static.py.
'''

import sys
from pathlib import Path

import numpy as np
import pytest

import planktos

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h
from _ib_harness import POS_ATOL

pytestmark = pytest.mark.filterwarnings('ignore:invalid value encountered in')


def _run(init, mu, wall_x=None, x_bndry='periodic', y_bndry='noflux',
         ib='sliding', dt=1.0):
    '''Advance a single agent one deterministic step: zero flow, constant drift
    mu (zero diffusion), and an optional vertical wall at x=wall_x spanning
    y in [1,9]. Returns (final_position, ib_collision_idx).'''
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry=x_bndry, y_bndry=y_bndry,
                                 flow=[np.zeros((5, 5)), np.zeros((5, 5))])
    if wall_x is not None:
        mesh = h.wall_segments(20, wall_x, y_lo=1.0, y_hi=9.0)
        envir.ibmesh = mesh
        envir.max_meshpt_dist = h.max_meshpt_dist(mesh)
    swrm = planktos.Swarm(swarm_size=1, envir=envir, init=np.array([init], float),
                          seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array(mu, float)
    swrm.move(dt, ib_collisions=ib, silent=True)
    return np.asarray(swrm.positions)[0], int(swrm.ib_collision_idx[0])


def test_periodic_wrap_without_wall_reenters_cleanly():
    # Control: drifting -4 from x=1 exits the left boundary and wraps to x=7,
    # with no immersed boundary present and no collision recorded.
    pos, idx = _run([1.0, 5.0], [-4.0, 0.0])
    assert np.allclose(pos, [7.0, 5.0], atol=POS_ATOL)
    assert idx == -1


@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_periodic_left_wrap_hits_far_wall_head_on(ib):
    # Exit the left boundary, wrap to the right, and immediately meet a wall at
    # x=8. The agent re-enters at x=10 moving -x and must stop on the +x side of
    # the wall (no penetration) rather than landing at the bare wrap point x=7.
    pos, idx = _run([1.0, 5.0], [-4.0, 0.0], wall_x=8.0, ib=ib)
    assert pos[0] >= 8.0 - POS_ATOL, "penetrated past the far-side wall"
    assert np.allclose(pos, [8.0, 5.0], atol=POS_ATOL)
    assert idx != -1, "the post-wrap IB collision was not recorded"


def test_periodic_left_wrap_then_slides_along_far_wall():
    # Diagonal drift (-4, 2): exit left, wrap to the right, strike the wall at
    # x=8 at an angle and slide up. The tangential (y) motion completes (5 -> 7)
    # while x is held at the wall -- the recursive slide runs on the wrapped path.
    pos, idx = _run([1.0, 5.0], [-4.0, 2.0], wall_x=8.0, ib='sliding')
    assert pos[0] >= 8.0 - POS_ATOL, "penetrated past the far-side wall"
    assert np.allclose(pos, [8.0, 7.0], atol=POS_ATOL)
    assert idx != -1


@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_periodic_right_wrap_hits_far_wall(ib):
    # Mirror image: drift +4 from x=9 exits the right boundary, wraps to the
    # left, and meets a wall at x=2. The agent re-enters at x=0 moving +x and
    # must stop on the -x side of the wall.
    pos, idx = _run([9.0, 5.0], [4.0, 0.0], wall_x=2.0, ib=ib)
    assert pos[0] <= 2.0 + POS_ATOL, "penetrated past the far-side wall"
    assert np.allclose(pos, [2.0, 5.0], atol=POS_ATOL)
    assert idx != -1
