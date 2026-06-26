'''No-penetration and known-answer tests for *3D* static immersed-boundary
collisions -- the triangle-mesh project-and-slide path in
planktos._ibc.apply_internal_static_BC / _project_and_slide_static.

3D cannot be checked visually, so these were derived by computing the correct
post-collision position independently (plane projection, edge crossing, dihedral
joints) and asserting both the exact answer where known and the no-penetration
invariant against the *finite* triangles. Built from in-memory triangle meshes
(M,3,3); no Environment, Swarm, flow, or RNG.

Two of these (coplanar tiled surface, gentle concave fold) are regressions for a
fixed bug: agents used to stick at a shared edge instead of sliding onto the
adjacent triangle, because the recursion re-detected the just-crossed edge. The
3D edge test now keys on whether the projected end overshot the triangle (like the
2D branch) rather than on where the agent currently sits.
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
#            single horizontal triangle (z=1): exact known answers            #
# --------------------------------------------------------------------------- #
# Big triangle so slides stay within it; agents approach from below (z<1).

@pytest.fixture
def htri():
    return h.triangle([0., 0., 1.], [10., 0., 1.], [0., 10., 1.])


@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_normal_hit_stops_at_plane(htri, ib):
    # Head-on in +z: no in-plane component, so both modes stop just below z=1.
    start = np.array([2., 2., 0.9]); end = np.array([2., 2., 1.3])
    newend, dx, idx = h.call_static(start, end, htri, ib)
    assert newend[2] <= 1.0 + 1e-9, "penetrated the triangle plane"
    assert np.allclose(newend, [2., 2., 1.0], atol=POS_ATOL)
    assert idx is not None


def test_oblique_slide_matches_plane_projection(htri):
    # (2,2,0.9)->(5,2,1.3): hits z=1 at (2.75,2,1); the +x part slides in-plane,
    # landing at (5,2,~1) -- exactly the projection of the travel onto the plane.
    start = np.array([2., 2., 0.9]); end = np.array([5., 2., 1.3])
    newend, dx, idx = h.call_static(start, end, htri, 'sliding')
    assert newend[2] <= 1.0 + 1e-9
    assert np.allclose(newend, [5., 2., 1.0], atol=POS_ATOL)


def test_oblique_sticky_stops_at_intersection(htri):
    start = np.array([2., 2., 0.9]); end = np.array([5., 2., 1.3])
    newend, dx, idx = h.call_static(start, end, htri, 'sticky')
    assert np.allclose(newend, [2.75, 2., 1.0], atol=POS_ATOL)


def test_no_collision_passes_through(htri):
    start = np.array([2., 2., 0.9]); end = np.array([2.5, 2., 0.95])
    newend, dx, idx = h.call_static(start, end, htri, 'sliding')
    assert idx is None and dx is None
    assert np.allclose(newend, end)


def test_grazing_hit_stays_outside(htri):
    # Nearly tangent: barely dips below z=1 while moving mostly in +x.
    start = np.array([1., 1., 0.999]); end = np.array([6., 1., 1.001])
    newend, dx, idx = h.call_static(start, end, htri, 'sliding')
    assert idx is not None
    assert newend[2] <= 1.0 + POS_ATOL, "grazing hit penetrated"


def test_slide_off_finite_edge(htri):
    # Small triangle: the slide reaches the hypotenuse and the agent rounds the
    # finite edge, continuing on its original trajectory beyond the triangle.
    tri = h.triangle([0., 0., 1.], [2., 0., 1.], [0., 2., 1.])
    start = np.array([0.5, 0.5, 0.9]); end = np.array([3., 0.5, 1.3])
    newend, dx, idx = h.call_static(start, end, tri, 'sliding')
    assert newend[0] + newend[1] > 2.0, "did not round the finite edge"
    assert newend[2] >= 1.0 - POS_ATOL                    # over (not through) the edge
    assert np.allclose(newend, [3.0, 0.5, 1.24], atol=POS_ATOL)


# --------------------------------------------------------------------------- #
#                       convex dihedral (ridge)                               #
# --------------------------------------------------------------------------- #

def test_convex_ridge_crests_without_penetrating():
    # Right face x+z=2 and left face z=2+x meet at the ridge edge x=0,z=2.
    # An agent above the right face moving left crests the ridge and ends above
    # the left face (within its extent) -- it does not drop through.
    A = [0., 0., 2.]; B = [0., 10., 2.]
    mesh = np.array([[A, B, [2., 5., 0.]],      # right face
                     [A, B, [-2., 5., 0.]]])    # left face
    start = np.array([1., 5., 1.5]); end = np.array([-1., 5., 1.5])
    newend, dx, idx = h.call_static(start, end, mesh, 'sliding')
    assert newend[0] < 0.0, "did not cross the ridge"
    assert -2.0 <= newend[0] <= 0.0                       # within the left face extent
    assert newend[2] - newend[0] >= 2.0 - POS_ATOL, "dropped below the left face"


# --------------------------------------------------------------------------- #
#                       concave joints                                        #
# --------------------------------------------------------------------------- #

def test_concave_right_angle_corner_stops_no_penetration():
    # Floor (z=0, +x) and wall (x=0, +z) meeting along the y-axis. An agent in the
    # open quadrant driven at the corner stops at the edge, on the near side of
    # BOTH faces (x>=0 and z>=0).
    floor = [[0., 0., 0.], [0., 10., 0.], [8., 0., 0.]]
    wall = [[0., 0., 0.], [0., 10., 0.], [0., 0., 8.]]
    mesh = np.array([floor, wall])
    start = np.array([4., 5., 3.]); end = np.array([-2., 5., -1.])
    newend, dx, idx = h.call_static(start, end, mesh, 'sliding')
    assert newend[0] >= -POS_ATOL, "penetrated the wall (x<0)"
    assert newend[2] >= -POS_ATOL, "penetrated the floor (z<0)"
    assert np.allclose(newend, [0., 5., 0.], atol=POS_ATOL)


@pytest.mark.parametrize('end', [np.array([-0.3, 5., -1.]),      # gentle drive
                                 np.array([-0.5, 5., -6.])])     # hard-driven overshoot
def test_acute_groove_no_penetration(end):
    # A sharp (~40deg) concave V. However hard the agent is driven into it, it
    # ends on the open side of both faces (no penetration of either).
    ang = np.deg2rad(20)
    A = np.array([0., 0., 0.]); B = np.array([0., 10., 0.])
    Lf = np.array([-np.sin(ang), 5., np.cos(ang)]) * 4
    Rf = np.array([np.sin(ang), 5., np.cos(ang)]) * 4
    left = np.array([A, B, Lf]); right = np.array([A, B, Rf])
    start = np.array([0.2, 5., 4.])
    newend, dx, idx = h.call_static(start, end, np.array([left, right]), 'sliding')
    h.assert_not_penetrated_3D(start, newend, A, B, Lf)
    h.assert_not_penetrated_3D(start, newend, A, B, Rf)


# --------------------------------------------------------------------------- #
#        slide onto the adjacent triangle (regression: BUG-3D-SLIDE-STICK)     #
# --------------------------------------------------------------------------- #

def test_coplanar_tiled_surface_slides_across_internal_edge():
    # A flat z=0 square tiled by two triangles along the diagonal. An agent that
    # pierces one triangle must glide across the internal edge into the other,
    # not stick at it. (Previously it stuck at (2,2,0).)
    t1 = [[0., 0., 0.], [4., 0., 0.], [4., 4., 0.]]   # lower-right
    t2 = [[0., 0., 0.], [4., 4., 0.], [0., 4., 0.]]   # upper-left
    start = np.array([3., 1., 0.3]); end = np.array([1., 3., -0.5])
    newend, dx, idx = h.call_static(start, end, np.array([t1, t2]), 'sliding')
    assert newend[2] >= -POS_ATOL, "penetrated the flat surface"
    assert newend[1] - newend[0] > 1.0, "stuck at the internal diagonal edge"
    assert np.allclose(newend[:2], [1.0, 3.0], atol=POS_ATOL)   # slid to slide_pt in t2


@pytest.mark.parametrize('slope', [0.05, 0.2, 0.5])
def test_gentle_concave_fold_slides_onto_neighbor(slope):
    # A gentle concave valley (faces z = slope*|x|). An agent sliding down one
    # face crosses the valley edge and continues up the other face, ending on (or
    # above) its surface -- it neither sticks at the edge nor drops through.
    A = [0., 0., 0.]; B = [0., 10., 0.]
    left = np.array([A, B, [-4., 5., 4. * slope]])
    right = np.array([A, B, [4., 5., 4. * slope]])
    start = np.array([-3., 5., 2.]); end = np.array([3., 5., -1.])
    newend, dx, idx = h.call_static(start, end, np.array([left, right]), 'sliding')
    assert newend[0] > 0.5, "stuck near the valley edge instead of sliding onto the far face"
    # on or above the (finite) right face surface z = slope*x: no penetration
    assert newend[2] >= slope * newend[0] - POS_ATOL, "dropped through the far face"
