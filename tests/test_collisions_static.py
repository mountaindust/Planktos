'''No-penetration and known-answer tests for static immersed-boundary collisions.

These call planktos._ibc.apply_internal_static_BC directly on a single agent
trajectory against a hand-built mesh -- no Environment, Swarm, flow, or RNG -- so
each post-collision position is an exact function of geometry alone and the tests
run in milliseconds.

The load-bearing invariant is no penetration: an agent that starts on one side of
a boundary never ends on the far side. The collision code places a stopped agent a
tiny epsilon (~1e-5) *inside* the boundary it struck; POS_ATOL (1e-4) is comfortably
larger, so "exact" position asserts use it.

A pure head-on (normal) impact makes the projected slide vector zero, which trips a
benign "invalid value in divide" RuntimeWarning in _ibc (the result is still
correct). It is filtered here; a defensive guard in _ibc would remove it.
'''

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h
from _ib_harness import POS_ATOL

pytestmark = pytest.mark.filterwarnings('ignore:invalid value encountered in')


# --------------------------------------------------------------------------- #
#            single vertical segment: exact known answers                     #
# --------------------------------------------------------------------------- #
# Wall at x=5 spanning y in [0,10] as 20 elements; agents approach from the left.

@pytest.fixture
def vwall():
    return h.wall_segments(20, 5.0)


@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_vertical_normal_hit_stops_at_wall(vwall, ib):
    # Head-on in +x. No tangential component, so both modes stop at the wall.
    start = np.array([4.9, 5.0]); end = np.array([5.3, 5.0])
    newend, dx, idx = h.call_static(start, end, vwall, ib)
    assert newend[0] <= 5.0 + 1e-9, "penetrated past the wall"
    assert np.allclose(newend, [5.0, 5.0], atol=POS_ATOL)
    assert idx is not None
    h.assert_not_penetrated_2D(start, newend, [5., 0.], [5., 10.])


def test_vertical_diagonal_hit_slides(vwall):
    # (4.9,5)->(5.3,5.4): hits x=5 at (5,5.1); remaining tangential 0.3 in y.
    start = np.array([4.9, 5.0]); end = np.array([5.3, 5.4])
    newend, dx, idx = h.call_static(start, end, vwall, 'sliding')
    assert newend[0] <= 5.0 + 1e-9
    assert np.allclose(newend, [5.0, 5.4], atol=POS_ATOL)


def test_vertical_diagonal_hit_sticky_stops_at_intersection(vwall):
    start = np.array([4.9, 5.0]); end = np.array([5.3, 5.4])
    newend, dx, idx = h.call_static(start, end, vwall, 'sticky')
    assert newend[0] <= 5.0 + 1e-9
    assert np.allclose(newend, [5.0, 5.1], atol=POS_ATOL)   # the intersection point


def test_no_collision_passes_through(vwall):
    # Motion entirely on the near side never reaches the wall.
    start = np.array([4.9, 5.0]); end = np.array([4.95, 5.2])
    newend, dx, idx = h.call_static(start, end, vwall, 'sliding')
    assert idx is None and dx is None
    assert np.allclose(newend, end)


def test_parallel_motion_along_wall_no_collision(vwall):
    start = np.array([4.9, 2.0]); end = np.array([4.9, 8.0])
    newend, dx, idx = h.call_static(start, end, vwall, 'sliding')
    assert idx is None
    assert np.allclose(newend, end)


# --------------------------------------------------------------------------- #
#                 horizontal segment: symmetry check                          #
# --------------------------------------------------------------------------- #

def _hwall(y, x0=0.0, x1=10.0, M=20):
    xs = np.linspace(x0, x1, M + 1)
    mesh = np.zeros((M, 2, 2))
    mesh[:, 0, 0] = xs[:-1]; mesh[:, 0, 1] = y
    mesh[:, 1, 0] = xs[1:];  mesh[:, 1, 1] = y
    return mesh


def test_horizontal_diagonal_hit_slides():
    # Mirror of the vertical diagonal case: slide in x along the wall at y=5.
    mesh = _hwall(5.0)
    start = np.array([5.0, 4.9]); end = np.array([5.4, 5.3])
    newend, dx, idx = h.call_static(start, end, mesh, 'sliding')
    assert newend[1] <= 5.0 + 1e-9, "penetrated above the wall"
    assert np.allclose(newend, [5.4, 5.0], atol=POS_ATOL)


# --------------------------------------------------------------------------- #
#                 diagonal (45 deg) segment                                   #
# --------------------------------------------------------------------------- #

def _diag_wall(M=28):
    t = np.linspace(0.0, 10.0, M + 1)
    pts = np.stack([t, t], axis=1)
    return np.stack([pts[:-1], pts[1:]], axis=1)


@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_diagonal_normal_approach_stops_on_line(ib):
    # Motion (3,-3) is purely normal to y=x, so there is no tangential slide:
    # the agent stops at the intersection (4,4) on the correct (x<=y) side.
    mesh = _diag_wall()
    start = np.array([3.0, 5.0]); end = np.array([6.0, 2.0])
    newend, dx, idx = h.call_static(start, end, mesh, ib)
    assert np.allclose(newend, [4.0, 4.0], atol=POS_ATOL)
    # started above the line (x-y<0); must not cross to x-y>0
    assert newend[0] - newend[1] <= POS_ATOL


# --------------------------------------------------------------------------- #
#                 convex joint (outer L corner)                               #
# --------------------------------------------------------------------------- #

def test_convex_corner_slides_along_arm_and_sticks_at_vertex():
    # L: (2,5)-(5,5) horizontal arm, (5,5)-(5,2) vertical arm; vertex (5,5).
    # Agent above-right pushing toward (4.5,4.5) is stopped by the top arm.
    mesh = h.polyline([[2., 5.], [5., 5.], [5., 2.]])
    start = np.array([6.0, 6.0]); end = np.array([4.5, 4.5])

    newend, dx, idx = h.call_static(start, end, mesh, 'sliding')
    assert newend[1] >= 5.0 - POS_ATOL, "penetrated below the top arm"
    assert np.allclose(newend, [4.5, 5.0], atol=POS_ATOL)     # slid along y=5 to x=4.5

    newend_s, _, _ = h.call_static(start, end, mesh, 'sticky')
    assert np.allclose(newend_s, [5.0, 5.0], atol=POS_ATOL)   # stops at the corner


# --------------------------------------------------------------------------- #
#                 concave joint (inner V vertex)                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_concave_vertex_no_penetration_no_infinite_bounce(ib):
    # V: (3,8)-(5,5)-(7,8); inner vertex (5,5). An agent dropping straight down
    # the centerline must stop at/above the vertex -- and the recursion must
    # terminate (a non-terminating concave bounce would hang, not return).
    mesh = h.polyline([[3., 8.], [5., 5.], [7., 8.]])
    start = np.array([5.0, 7.0]); end = np.array([5.0, 4.5])
    newend, dx, idx = h.call_static(start, end, mesh, ib)
    assert newend[1] >= 5.0 - POS_ATOL, "penetrated below the vertex"
    assert np.allclose(newend, [5.0, 5.0], atol=POS_ATOL)


# --------------------------------------------------------------------------- #
#                 multi-element slide & grazing                               #
# --------------------------------------------------------------------------- #

def test_slide_crosses_element_boundary(vwall):
    # The diagonal hit lands on element 10 even though contact spans several
    # small elements -- the reported idx is the struck element, and the slide
    # crosses element seams without penetrating.
    start = np.array([4.9, 5.0]); end = np.array([5.3, 5.4])
    newend, dx, idx = h.call_static(start, end, vwall, 'sliding')
    assert idx == 10
    assert newend[0] <= 5.0 + 1e-9


def test_grazing_hit_stays_outside(vwall):
    # Agent barely crosses the wall plane while moving mostly tangentially.
    start = np.array([4.99, 3.0]); end = np.array([5.02, 7.0])
    newend, dx, idx = h.call_static(start, end, vwall, 'sliding')
    assert idx is not None
    assert newend[0] <= 5.0 + 1e-9, "grazing hit penetrated"
    assert newend[1] == pytest.approx(7.0, abs=POS_ATOL)     # tangential motion completes
