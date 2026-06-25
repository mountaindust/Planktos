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
#            known defect: axis-aligned sticky-moving returns NaN             #
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="BUG-STICKY-AXIS: _ibc.py:266 max() over a "
                   "0/0 NaN for axis-aligned moving elements yields NaN. Fix: pick "
                   "the contact-parameter component with nonzero denominator.")
def test_sticky_moving_axis_aligned_wall_should_not_nan():
    start_mesh = h.wall_segments(20, 5.0); end_mesh = h.wall_segments(20, 6.0)
    start = np.array([4.8, 5.0]); end = np.array([6.3, 5.0])
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sticky')
    assert not np.isnan(newend).any()
    assert newend[0] <= 6.0 + POS_ATOL
