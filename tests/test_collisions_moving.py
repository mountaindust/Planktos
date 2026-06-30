'''No-penetration and known-answer tests for *moving* immersed-boundary collisions
-- the core new capability of this branch.

These call planktos._ibc.apply_internal_moving_BC directly: the boundary is given
as two mesh snapshots (start_mesh, end_mesh) and deforms linearly across the step,
while the agent travels startpt->endpt. No Environment, Swarm, flow, or RNG, so the
result is an exact function of geometry and the tests are fast.

Two real defects were found while writing these and are pinned below as strict
xfails (they will flip to failures, prompting marker removal, once fixed):
  * BUG-STICKY-AXIS: _ibc.py:266 computes the contact parameter with
    max((x0-Q0x)/(Q1x-Q0x), (x1-Q0y)/(Q1y-Q0y)). For an axis-aligned (vertical or
    horizontal) moving element one denominator is 0, giving 0/0 -> NaN, and
    max(NaN, valid) returns NaN. Real IB2d meshes are essentially never perfectly
    axis-aligned, so the showcase example dodges it. The non-degenerate (tilted)
    sticky cases below pass and exercise the same code.
The second defect (BUG-ZEROLEN-SEG, _geom.py:79) is pinned in test_geom.py.

All non-degenerate cases here use a *moving agent* (startpt != endpt); a stationary
agent against a deforming mesh routes through BUG-ZEROLEN-SEG.
'''

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h
from _ib_harness import POS_ATOL

pytestmark = pytest.mark.filterwarnings('ignore:invalid value encountered in')


def _tilted_wall(M, x_at_y0, slope, y_lo=0.0, y_hi=10.0):
    '''A wall x = x_at_y0 + slope*y, split into M elements. A nonzero slope keeps
    elements off-axis so the sticky-moving code path is non-degenerate.'''
    ys = np.linspace(y_lo, y_hi, M + 1)
    mesh = np.zeros((M, 2, 2))
    mesh[:, 0, 0] = x_at_y0 + slope * ys[:-1]; mesh[:, 0, 1] = ys[:-1]
    mesh[:, 1, 0] = x_at_y0 + slope * ys[1:];  mesh[:, 1, 1] = ys[1:]
    return mesh


# --------------------------------------------------------------------------- #
#            translating vertical wall, sliding (axis-aligned OK)             #
# --------------------------------------------------------------------------- #

def test_translating_wall_catch_and_ride():
    # Wall x:5->6 over the step; fast agent (4.8,5)->(6.3,5) overtakes it at t=0.4
    # (x=5.4) and is then carried to the wall's final position x~6, never past it.
    start_mesh = h.wall_segments(20, 5.0); end_mesh = h.wall_segments(20, 6.0)
    start = np.array([4.8, 5.0]); end = np.array([6.3, 5.0])
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sliding')
    assert idx is not None
    assert newend[0] <= 6.0 + 1e-9, "rode past the wall's final position"
    assert np.allclose(newend, [6.0, 5.0], atol=POS_ATOL)


def test_translating_wall_diagonal_catch_slides_tangentially():
    # Same catch, but the agent also moves in +y; the tangential part completes.
    start_mesh = h.wall_segments(20, 5.0); end_mesh = h.wall_segments(20, 6.0)
    start = np.array([4.8, 5.0]); end = np.array([6.3, 5.6])
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sliding')
    assert newend[0] <= 6.0 + 1e-9
    assert np.allclose(newend, [6.0, 5.6], atol=POS_ATOL)


def test_receding_wall_drags_agent_no_penetration():
    # Wall recedes x:5->4 across the agent, which tries to move +x. It cannot pass
    # through, so it is dragged to just left of the wall's final position x~4.
    start_mesh = h.wall_segments(20, 5.0); end_mesh = h.wall_segments(20, 4.0)
    start = np.array([4.99, 5.0]); end = np.array([5.5, 5.0])
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sliding')
    assert idx is not None
    assert newend[0] <= 4.0 + 1e-9, "agent ended on the far side of the receding wall"
    assert np.allclose(newend, [4.0, 5.0], atol=POS_ATOL)


def test_agent_stays_behind_moving_wall_no_contact():
    # Agent (4.9,2)->(5.4,8) never catches the wall x:5->6 (always trails it).
    start_mesh = h.wall_segments(20, 5.0); end_mesh = h.wall_segments(20, 6.0)
    start = np.array([4.9, 2.0]); end = np.array([5.4, 8.0])
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sliding')
    assert idx is None and dx is None
    assert np.allclose(newend, end)


# --------------------------------------------------------------------------- #
#            tilted translating wall: sliding AND sticky (non-degenerate)     #
# --------------------------------------------------------------------------- #

def test_tilted_wall_sticky_stops_on_wall():
    # Wall x = 5+0.05y -> 6+0.05y. Agent (4.6,5)->(6.4,5) catches it; sticky stops
    # the agent on the wall at the final time, at x = 6 + 0.05*5 = 6.25.
    start_mesh = _tilted_wall(20, 5.0, 0.05); end_mesh = _tilted_wall(20, 6.0, 0.05)
    start = np.array([4.6, 5.0]); end = np.array([6.4, 5.0])
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sticky')
    assert not np.isnan(newend).any()
    assert newend[0] <= 6.25 + POS_ATOL, "penetrated past the tilted wall"
    assert np.allclose(newend, [6.25, 5.0], atol=POS_ATOL)


def test_tilted_wall_sliding_stays_on_near_side():
    # Sliding along the tilted moving wall: agent ends on (or just inside) the
    # near side of the wall's final position, never past it.
    end_mesh = _tilted_wall(20, 6.0, 0.05)
    start = np.array([4.6, 5.0]); end = np.array([6.4, 5.0])
    newend, dx, idx = h.call_moving(start, end, _tilted_wall(20, 5.0, 0.05),
                                    end_mesh, 'sliding')
    assert idx is not None
    # near side of the wall x = 6 + 0.05*y is where (x - 0.05*y) <= 6
    assert newend[0] - 0.05 * newend[1] <= 6.0 + POS_ATOL, "penetrated the tilted wall"


# --------------------------------------------------------------------------- #
#            axis-aligned sticky-moving walls (regression: BUG-STICKY-AXIS)    #
# --------------------------------------------------------------------------- #

def _horizontal_wall(M, y, x_lo=0.0, x_hi=10.0):
    xs = np.linspace(x_lo, x_hi, M + 1)
    mesh = np.zeros((M, 2, 2))
    mesh[:, 0, 0] = xs[:-1]; mesh[:, 0, 1] = y
    mesh[:, 1, 0] = xs[1:];  mesh[:, 1, 1] = y
    return mesh


@pytest.mark.parametrize('orient', ['vertical', 'horizontal'])
def test_sticky_moving_axis_aligned_wall_stops_on_wall(orient):
    # A perfectly axis-aligned moving element used to trip a 0/0 in the sticky
    # contact parameter and return NaN (BUG-STICKY-AXIS, now fixed). The agent
    # should stick to the wall and ride it to its final position.
    if orient == 'vertical':
        start_mesh = h.wall_segments(20, 5.0); end_mesh = h.wall_segments(20, 6.0)
        start = np.array([4.8, 5.0]); end = np.array([6.3, 5.0])
        expected, axis = [6.0, 5.0], 0
    else:
        start_mesh = _horizontal_wall(20, 5.0); end_mesh = _horizontal_wall(20, 6.0)
        start = np.array([5.0, 4.8]); end = np.array([5.0, 6.3])
        expected, axis = [5.0, 6.0], 1
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sticky')
    assert not np.isnan(newend).any()
    assert idx is not None
    assert newend[axis] <= 6.0 + POS_ATOL, "penetrated past the wall's final position"
    assert np.allclose(newend, expected, atol=POS_ATOL)


# --------------------------------------------------------------------------- #
#       golden multi-step trajectory (drift detector for moving collisions)    #
# --------------------------------------------------------------------------- #
# The cases above are single-step known answers. This pins a full deterministic
# multi-step Swarm.move() run through a translating wall (h.run_moving_golden) so
# any unintended change in the moving-collision behavior shows up as a diff. The
# baseline was generated from the trusted code and independently satisfies the
# no-penetration invariant (asserted separately below, so the lock is not purely
# circular). Regenerate GOLDEN_SLIDING only after a deliberate, reviewed change.

GOLDEN_SLIDING = np.array([
    [[4.00000000, 3.00000000], [4.00000000, 7.00000000], [2.00000000, 5.00000000], [6.00000000, 5.00000000]],
    [[5.00000000, 3.00000000], [4.80000000, 7.40000000], [2.30000000, 5.00000000], [6.20000000, 5.00000000]],
    [[6.00000000, 3.00000000], [5.60000000, 7.80000000], [2.60000000, 5.00000000], [6.40000000, 5.00000000]],
    [[6.49999000, 3.00000000], [6.40000000, 8.20000000], [2.90000000, 5.00000000], [6.60000000, 5.00000000]],
    [[6.99990000, 3.00000000], [6.99990000, 8.60000000], [3.20000000, 5.00000000], [7.00001000, 5.00000000]],
    [[7.49990000, 3.00000000], [7.49990000, 9.00000000], [3.50000000, 5.00000000], [7.50010000, 5.00000000]],
    [[7.99990000, 3.00000000], [7.99990000, 9.40000000], [3.80000000, 5.00000000], [8.00010000, 5.00000000]],
])


def test_moving_golden_trajectory_matches_baseline():
    traj = h.run_moving_golden('sliding')
    assert traj.shape == GOLDEN_SLIDING.shape
    assert np.allclose(traj, GOLDEN_SLIDING, atol=1e-6), \
        "moving-collision trajectory drifted from the pinned baseline"


def test_moving_golden_trajectory_no_penetration():
    # Independent of the pinned values: agents 0-2 start left of the wall and
    # must stay on the near side at every recorded step; agent 3 starts on the
    # far side and must stay there as the wall sweeps past it.
    cfg = h.GOLDEN_MOVING
    traj = h.run_moving_golden('sliding')
    started_left = np.array(cfg['init'])[:, 0] < cfg['wall_x0']
    for k in range(traj.shape[0]):
        wall = h.golden_moving_wall_x(k * cfg['dt'])
        x = traj[k, :, 0]
        assert np.all(x[started_left] <= wall + POS_ATOL), \
            f"near-side agent penetrated the wall at step {k}"
        assert np.all(x[~started_left] >= wall - POS_ATOL), \
            f"far-side agent penetrated the wall at step {k}"


def test_moving_golden_trajectory_is_deterministic():
    # No RNG/flow: two runs must be bit-for-bit identical.
    assert np.array_equal(h.run_moving_golden('sliding'),
                          h.run_moving_golden('sliding'))
