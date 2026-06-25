'''Unit tests for planktos._geom -- the pure geometry workhorses underneath all
agent-boundary interaction.

Every function here is deterministic and side-effect free, so the tests use exact,
analytically-known answers and reconstruct the intersection point from the
returned parameter to prove internal consistency. No Environment, Swarm, flow, or
RNG is involved.

Return conventions (verified against the code):
  seg_intersect_2D(P0,P1,Q0,Q1)            -> None or (x, s_I, Q0, Q1, idx)
  seg_intersect_3D_triangles(...)          -> None or (x, s_I, Q0, Q1, Q2, idx)
  seg_intersect_2D_multilinear_poly(...)   -> None or (x, t_I, Q0_t, Q1_t, idx)
  seg_intersect_2D(..., get_all=True)      -> lazy zip of the above tuples
where x is the intersection point, s_I in [0,1] is the fraction of P0->P1
traveled, and idx indexes the struck element (None for a single element).
'''

import numpy as np
import pytest

from planktos import _geom


# --------------------------------------------------------------------------- #
#                       2D segment-segment intersection                       #
# --------------------------------------------------------------------------- #

def _recon_2D(x, s_I, P0, P1):
    '''The returned point must equal P0 + s_I*(P1-P0).'''
    assert np.allclose(x, np.asarray(P0) + s_I * (np.asarray(P1) - np.asarray(P0)))


def test_2D_parallel_returns_none():
    P0 = np.array([0.0, 0.0]); P1 = np.array([0.0, 1.0])
    Q0 = np.array([1.0, 0.0]); Q1 = np.array([1.0, 0.5])      # parallel, offset
    assert _geom.seg_intersect_2D(P0, P1, Q0, Q1) is None


def test_2D_no_intersection_returns_none():
    P0 = np.array([0.0, 0.0]); P1 = np.array([0.0, 1.0])
    Q0 = np.array([1.0, 0.0]); Q1 = np.array([0.5, 0.5])      # misses
    assert _geom.seg_intersect_2D(P0, P1, Q0, Q1) is None


def test_2D_single_intersection_known_point():
    # Vertical P crosses a slanted Q at (1,2). P spans y in [0,5], so s_I = 2/5.
    P0 = np.array([1.0, 0.0]); P1 = np.array([1.0, 5.0])
    Q0 = np.array([0.0, 2.0]); Q1 = np.array([3.0, 2.0])
    x, s_I, q0, q1, idx = _geom.seg_intersect_2D(P0, P1, Q0, Q1)
    assert np.allclose(x, [1.0, 2.0])
    assert np.isclose(s_I, 2.0 / 5.0)
    _recon_2D(x, s_I, P0, P1)


def test_2D_first_of_many_is_closest_in_s():
    # Four candidate segments; the true hit closest along P (smallest s_I) wins.
    P0 = np.array([1.0, 0.0]); P1 = np.array([1.0, 5.0])
    Q0 = np.array([[0., 4.], [0., 1.], [0., 2.], [10., 10.]])
    Q1 = np.array([[5., 4.], [0.5, 2.], [3., 2.], [11., 11.]])
    x, s_I, q0, q1, idx = _geom.seg_intersect_2D(P0, P1, Q0, Q1)
    assert np.allclose(x, [1.0, 2.0])
    assert np.isclose(s_I, 2.0 / 5.0)
    assert idx == 2                          # the [0,2]-[3,2] segment
    assert np.allclose(q0, [0., 2.]) and np.allclose(q1, [3., 2.])


def test_2D_get_all_returns_every_hit():
    P0 = np.array([1.0, 0.0]); P1 = np.array([1.0, 5.0])
    Q0 = np.array([[0., 4.], [0., 1.], [0., 2.], [10., 10.]])
    Q1 = np.array([[5., 4.], [0.5, 2.], [3., 2.], [11., 11.]])
    allhits = list(_geom.seg_intersect_2D(P0, P1, Q0, Q1, get_all=True))
    hit_idx = sorted(h[4] for h in allhits)
    assert hit_idx == [0, 2]                 # both crossing segments, not the misses
    # the scalar (non-get_all) call must return the smallest-s_I member
    scalar = _geom.seg_intersect_2D(P0, P1, Q0, Q1)
    assert np.isclose(scalar[1], min(h[1] for h in allhits))


def test_2D_grazing_endpoint_hit():
    # P just reaches the endpoint of Q. Should register as a hit at that endpoint.
    P0 = np.array([0.0, 0.0]); P1 = np.array([2.0, 0.0])
    Q0 = np.array([1.0, 0.0]); Q1 = np.array([1.0, 3.0])
    res = _geom.seg_intersect_2D(P0, P1, Q0, Q1)
    assert res is not None
    assert np.allclose(res[0], [1.0, 0.0])
    assert np.isclose(res[1], 0.5)


# ---- 2D routine operating on coplanar points embedded in 3D (z held fixed) ---

def test_2D_in_3D_coplanar_cases():
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.]); tri2 = np.array([1., 2., 1.])
    Q0 = np.array([tri0, tri1, tri2]); Q1 = np.array([tri1, tri2, tri0])
    # segment above the z=1 plane: no intersection with the in-plane edges
    assert _geom.seg_intersect_2D(np.array([1.1, 1.1, 1.]),
                                  np.array([1.2, 1.2, 1.]), Q0, Q1) is None
    # along an edge, reaching across it: intersection at the far vertex tri1
    res = _geom.seg_intersect_2D(np.array([1.1, 1., 1.]),
                                 np.array([2.1, 1., 1.]), Q0, Q1)
    assert res is not None and np.allclose(res[0], tri1)
    # off the edges, crossing the hypotenuse at (1.5,1.5,1)
    res = _geom.seg_intersect_2D(np.array([1.1, 1.1, 1.]),
                                 np.array([2.1, 2.1, 1.]), Q0, Q1)
    assert np.allclose(res[0], [1.5, 1.5, 1.0])


# --------------------------------------------------------------------------- #
#                     3D segment-triangle intersection                        #
# --------------------------------------------------------------------------- #

def test_3D_parallel_returns_none():
    # Segment in a plane parallel to the triangle (above and in-plane) -> None.
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.]); tri2 = np.array([1., 2., 1.])
    assert _geom.seg_intersect_3D_triangles(np.array([0.5, 0.5, 2.]),
                                            np.array([2.5, 2.5, 2.]), tri0, tri1, tri2) is None
    assert _geom.seg_intersect_3D_triangles(np.array([0.5, 0.5, 1.]),
                                            np.array([2.5, 2.5, 1.]), tri0, tri1, tri2) is None


def test_3D_no_hit_returns_none():
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.]); tri2 = np.array([1., 2., 1.])
    # crosses the plane but outside the triangle
    assert _geom.seg_intersect_3D_triangles(np.array([0.5, 0.5, 2.]),
                                            np.array([2.5, 2.5, 1.5]), tri0, tri1, tri2) is None


def test_3D_single_hit_known_point():
    P0 = np.array([0.5, 0.5, 0.5]); P1 = np.array([2.0, 2.0, 1.5])
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([4., 1., 1.]); tri2 = np.array([1., 4., 1.])
    x, s_I, q0, q1, q2, idx = _geom.seg_intersect_3D_triangles(P0, P1, tri0, tri1, tri2)
    assert np.allclose(x, [1.25, 1.25, 1.0])
    assert np.isclose(s_I, 0.5)
    _recon_2D(x, s_I, P0, P1)                 # reconstruction works in 3D too
    assert np.allclose(q0, tri0) and np.allclose(q1, tri1) and np.allclose(q2, tri2)


def test_3D_first_of_many_triangles():
    P0 = np.array([0.5, 0.5, 0.5]); P1 = np.array([2.0, 2.0, 1.5])
    tri0 = np.array([[1., 1., 1.2], [2.5, 2.5, 20.], [1., 1., 1.],
                     [6., 1., 10.], [1., 1., 1.1]])
    tri1 = np.array([[4., 1., 1.2], [4., 1., 1.], [4., 1., 1.],
                     [4., 1., 1.], [4., 1., 1.1]])
    tri2 = np.array([[1., 4., 1.2], [1., 4., 1.], [1., 4., 1.],
                     [8., 1., 1.], [1., 4., 1.1]])
    x, s_I, q0, q1, q2, idx = _geom.seg_intersect_3D_triangles(P0, P1, tri0, tri1, tri2)
    assert np.allclose(x, [1.25, 1.25, 1.0])
    assert np.isclose(s_I, 0.5)
    assert idx == 2                            # the z=1 triangle hit at s_I=0.5


# --------------------------------------------------------------------------- #
#         2D moving-mesh intersection (multilinear polynomial in t)           #
# --------------------------------------------------------------------------- #
# A segment that translates/deforms linearly over the step: endpoints go
# Q0->Q2 and Q1->Q3 as t runs 0->1. Used by the moving-boundary collision code.

def test_multilinear_translating_wall_closed_form():
    # Wall at x=5 (t=0) translating to x=6 (t=1): v_wall = 1 in x.
    # Agent (4.8,5)->(6.3,5): v_agent = 1.5 in x, initial gap delta = 0.2.
    # Closed form catch time t* = delta/(v_agent - v_wall) = 0.2/0.5 = 0.4,
    # at x = 4.8 + 0.4*1.5 = 5.4.
    Q0 = np.array([5., 0.]); Q1 = np.array([5., 10.])
    Q2 = np.array([6., 0.]); Q3 = np.array([6., 10.])
    P0 = np.array([4.8, 5.]); P1 = np.array([6.3, 5.])
    x, t_I, q0t, q1t, idx = _geom.seg_intersect_2D_multilinear_poly(P0, P1, Q0, Q1, Q2, Q3)
    assert np.isclose(t_I, 0.4)
    assert np.allclose(x, [5.4, 5.0])
    # the interpolated segment at t_I is the wall at x = 5.4
    assert np.allclose(q0t, [5.4, 0.0]) and np.allclose(q1t, [5.4, 10.0])


def test_multilinear_wall_outruns_agent_returns_none():
    # Wall x=5 -> x=7 (v=2) outruns a slow agent (4.8,5)->(5.3,5) (v=0.5).
    Q0 = np.array([5., 0.]); Q1 = np.array([5., 10.])
    Q2 = np.array([7., 0.]); Q3 = np.array([7., 10.])
    P0 = np.array([4.8, 5.]); P1 = np.array([5.3, 5.])
    assert _geom.seg_intersect_2D_multilinear_poly(P0, P1, Q0, Q1, Q2, Q3) is None


def test_multilinear_degenerates_to_static():
    # No mesh motion (Q0==Q2, Q1==Q3): reduces to a static wall at x=5.
    # Agent (4.8,5)->(5.6,5): catch at x=5, t = (5-4.8)/(5.6-4.8) = 0.25.
    Q0 = np.array([5., 0.]); Q1 = np.array([5., 10.])
    P0 = np.array([4.8, 5.]); P1 = np.array([5.6, 5.])
    x, t_I, q0t, q1t, idx = _geom.seg_intersect_2D_multilinear_poly(
        P0, P1, Q0, Q1, Q0.copy(), Q1.copy())
    assert np.isclose(t_I, 0.25)
    assert np.allclose(x, [5.0, 5.0])


# --------------------------------------------------------------------------- #
#                     closest-distance routines                               #
# --------------------------------------------------------------------------- #

def test_closest_dist_line_and_pts_is_segment_aware():
    # Segment along the x-axis from (0,0) to (10,0).
    start = np.array([0., 0.]); end = np.array([10., 0.])
    pts = np.array([[5., 0.],    # on the segment        -> 0
                    [5., 3.],    # perpendicular offset  -> 3
                    [-2., 0.],   # past the start cap     -> 2
                    [12., 0.],   # past the end cap       -> 2
                    [0., 4.]])   # off the start, in y    -> 4
    d = _geom.closest_dist_btwn_line_and_pts(start, end, pts)
    assert np.allclose(d, [0., 3., 2., 2., 4.])


def test_closest_dist_lines_and_pt():
    Q0 = np.array([[0., 0.], [0., 1.]])
    Q1 = np.array([[10., 0.], [10., 1.]])
    pt = np.array([5., 0.])
    assert np.allclose(_geom.closest_dist_btwn_lines_and_pt(Q0, Q1, pt), [0., 1.])


@pytest.mark.xfail(strict=True, reason="BUG-ZEROLEN-SEG: _geom.py:79 reads "
                   "`seg_lengths_2[~z_check] = seg_lengths_2` (a shape-mismatched "
                   "self-assignment) where it must be "
                   "`seg_lengths_2 = seg_lengths_2[~z_check]`. Any mix of "
                   "zero-length and normal segments raises ValueError. This routes "
                   "in from a stationary agent meeting a deforming (pinned-vertex) "
                   "moving mesh.")
def test_closest_dist_lines_and_pt_mixed_zero_length():
    # One zero-length segment (a point at (5,0)) and one real segment. Distances
    # to (5,5) should be [5, 1]; instead the buggy line raises ValueError.
    Q0 = np.array([[5., 0.], [0., 1.]])
    Q1 = np.array([[5., 0.], [10., 1.]])
    pt = np.array([5., 5.])
    d = _geom.closest_dist_btwn_lines_and_pt(Q0, Q1, pt)
    assert np.allclose(d, [5.0, 4.0])


def test_closest_dist_two_lines_3D_skew():
    # Line A along x at z=0; line B along y at z=2 crossing x=5: gap is 2.
    a0 = np.array([0., 0., 0.]); a1 = np.array([10., 0., 0.])
    b0 = np.array([[5., -5., 2.]]); b1 = np.array([[5., 5., 2.]])
    assert np.allclose(_geom.closest_dist_btwn_two_lines(a0, a1, b0, b1), [2.])


def test_closest_dist_two_lines_intersecting_is_zero():
    a0 = np.array([0., 0., 0.]); a1 = np.array([10., 0., 0.])
    b0 = np.array([[5., -5., 0.]]); b1 = np.array([[5., 5., 0.]])   # crosses A at (5,0,0)
    assert np.allclose(_geom.closest_dist_btwn_two_lines(a0, a1, b0, b1), [0.])


# --------------------------------------------------------------------------- #
#                          point-to-plane distance                            #
# --------------------------------------------------------------------------- #

def test_dist_point_to_plane_known_values():
    Q0 = np.array([0., 0., 0.]); n = np.array([0., 0., 1.])
    assert np.isclose(_geom.dist_point_to_plane(np.array([0., 0., 3.]), n, Q0), 3.0)
    assert np.isclose(_geom.dist_point_to_plane(np.array([7., -4., 3.]), n, Q0), 3.0)
    # unsigned: below the plane gives the same magnitude, and |normal| is divided out
    assert np.isclose(_geom.dist_point_to_plane(np.array([0., 0., -3.]),
                                                np.array([0., 0., 2.]), Q0), 3.0)


def test_dist_point_to_plane_on_plane_is_zero():
    Q0 = np.array([1., 2., 3.]); n = np.array([0., 1., 0.])
    assert np.isclose(_geom.dist_point_to_plane(np.array([9., 2., -4.]), n, Q0), 0.0)
