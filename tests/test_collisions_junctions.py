'''Immersed-boundary collision cases the existing suite structurally cannot reach.

Everything in test_collisions_static.py, test_collisions_static_3d.py and
test_collisions_moving.py builds meshes out of *chains*: walls, polylines,
L-corners, V-vertices, arcs, dihedrals, tiled surfaces, folds. In every one of
them a vertex joins at most two segments and an edge is shared by at most two
triangles. So when an agent slides off the end of an element there is always
exactly ONE candidate element to continue on.

The slide code has a separate branch for two or more candidates. Nothing in the
suite -- and nothing in the examples -- has ever executed it.

Real meshes do reach it. `read_IB2d_mesh_data(..., method='proximity')` joins
every pair of vertices within a radius, so a densely sampled boundary picks up
vertices of degree 4: examples/ib2d_data/channel.vertex meshes to 2700 segments
with 429 such vertices. An agent that slides off one of them has three
candidates.

These tests therefore assert **invariants rather than positions**. There is no
specification yet for which element a slide should continue onto when several
are available, so pinning an exact resolved point would be inventing one. What
must hold under any correct policy is asserted instead:

  * the routine returns rather than raising,
  * the result is a finite point,
  * the resolved displacement does not exceed the attempted displacement
    (sliding projects motion; it cannot add any),
  * an agent that started outside a closed obstacle does not end up inside it.

The last is the project's hard invariant, and the only one of the four that
constrains the geometry rather than merely the arithmetic.

The checks that apply to any geometry at all -- finiteness, motion not being
amplified, independence from placement and units, and what happens when a slide
outruns the stack -- are not about junctions and live in
test_collisions_invariants.py.
'''

import warnings

import numpy as np
import pytest

import _ib_harness as h


# --------------------------------------------------------------------------- #
#                     geometries reaching each k                              #
# --------------------------------------------------------------------------- #
# k is the number of candidate elements the slide code must choose among after
# it discards the element being left and the ones facing away from the agent.
# These were built by instrumenting the branch and reading off the k it saw, so
# the parametrization is measured, not assumed.

# A hub with n spokes. The agent slides in along the lower spoke and runs off
# the hub; the spokes on its own side of that element are the candidates.
STAR_HUB = (5.0, 5.0)
STAR_TIPS = [(5.0, 0.0), (5.0, 10.0), (1.0, 3.0), (2.0, 1.0), (0.5, 5.0)]
STAR_START, STAR_END = (4.0, 2.0), (6.0, 7.0)

# n_spokes -> k reached
STAR_K = {2: 1, 3: 2, 4: 3, 5: 4}

# A closed square with fins radiating from one corner into the half-plane the
# agent approaches through. Closed, so "inside" is well defined.
SQUARE = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
SQUARE_CORNER = (6.0, 4.0)
SQUARE_FINS = [(7.0, 3.2), (6.3, 2.6), (7.4, 3.8), (6.8, 2.4)]
SQUARE_START, SQUARE_END = (4.3, 3.7), (7.6, 4.35)

# n_fins -> k reached
SQUARE_K = {1: 1, 2: 2, 3: 3, 4: 4}

# Triangles sharing one non-manifold edge, like pages of a book. A closed
# surface shares every edge between exactly two triangles, so this cannot arise
# from a manifold mesh -- but nothing validates that a loaded mesh is manifold.
BOOK_E0, BOOK_E1 = (0.0, 0.0, 0.0), (0.0, 2.0, 0.0)
BOOK_TIPS = [(2.0, 1.0, 0.0), (-1.0, 1.0, 1.0), (-1.0, 1.0, 2.0),
             (-1.0, 1.0, 0.5), (-1.0, 1.0, 3.0)]
BOOK_START, BOOK_END = (1.0, 1.0, 0.6), (-0.6, 1.0, -0.4)

# n_triangles -> k reached
BOOK_K = {2: 1, 3: 2, 4: 3, 5: 4}


def _star(n_spokes):
    return h.star_2D(STAR_HUB, STAR_TIPS[:n_spokes])


def _square_with_fins(n_fins):
    return np.concatenate([h.closed_polygon(SQUARE),
                           h.star_2D(SQUARE_CORNER, SQUARE_FINS[:n_fins])])


def _book(n_triangles):
    return h.book_3D(BOOK_E0, BOOK_E1, BOOK_TIPS[:n_triangles])


# --------------------------------------------------------------------------- #
#            2D: a vertex where three or more segments meet                   #
# --------------------------------------------------------------------------- #

# Only the sliding path reaches the multi-candidate branch; sticky stops at the
# point of intersection and never chooses a next element, so it is unaffected.
# Both are exercised anyway, so that a future change which starts routing sticky
# collisions through the branch does not go unnoticed.

@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_star_junction_returns_a_finite_point(ib, n_spokes):
    newend, _, _ = h.call_static(STAR_START, STAR_END, _star(n_spokes),
                                 ib_collisions=ib)
    h.assert_finite(newend, f'{n_spokes}-spoke star ({ib})')


@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
def test_star_junction_does_not_amplify_motion(n_spokes):
    newend, _, _ = h.call_static(STAR_START, STAR_END, _star(n_spokes))
    h.assert_displacement_bounded(STAR_START, STAR_END, newend)


@pytest.mark.parametrize('n_fins', sorted(SQUARE_K))
@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_finned_corner_keeps_the_agent_outside(ib, n_fins):
    '''The hard invariant, on a closed obstacle so that it is well posed: the
    agent starts outside the square and must finish outside it, wherever along
    the boundary the slide leaves it.'''
    newend, _, _ = h.call_static(SQUARE_START, SQUARE_END,
                                 _square_with_fins(n_fins), ib_collisions=ib)
    h.assert_finite(newend, f'{n_fins}-fin corner ({ib})')
    h.assert_outside_polygon(newend, SQUARE, f'{n_fins}-fin corner ({ib})')


@pytest.mark.parametrize('n_fins', sorted(SQUARE_K))
def test_finned_corner_does_not_amplify_motion(n_fins):
    newend, _, _ = h.call_static(SQUARE_START, SQUARE_END,
                                 _square_with_fins(n_fins))
    h.assert_displacement_bounded(SQUARE_START, SQUARE_END, newend)


def test_star_junction_reports_a_valid_element_index():
    '''The returned index feeds Swarm.ib_collision_idx, which users read. It has
    to index the mesh that was passed in.'''
    mesh = _star(4)
    _, _, idx = h.call_static(STAR_START, STAR_END, mesh)
    assert idx is None or 0 <= int(idx) < mesh.shape[0], (
        f'element index {idx} out of range for a {mesh.shape[0]}-element mesh')


# --------------------------------------------------------------------------- #
#            3D: an edge shared by three or more triangles                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('n_tri', sorted(BOOK_K))
@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_nonmanifold_edge_returns_a_finite_point(ib, n_tri):
    newend, _, _ = h.call_static(BOOK_START, BOOK_END, _book(n_tri),
                                 ib_collisions=ib)
    h.assert_finite(newend, f'{n_tri}-triangle book ({ib})')


@pytest.mark.parametrize('n_tri', sorted(BOOK_K))
def test_nonmanifold_edge_does_not_amplify_motion(n_tri):
    newend, _, _ = h.call_static(BOOK_START, BOOK_END, _book(n_tri))
    h.assert_displacement_bounded(BOOK_START, BOOK_END, newend)


# --------------------------------------------------------------------------- #
#            3D: the candidate ranking must use the dihedral angle            #
# --------------------------------------------------------------------------- #
# In 2D the joint between two segments has one angle and there is nothing to
# confuse it with. In 3D the joint between two triangles is a *dihedral* about
# their shared edge, and a candidate triangle's third vertex can sit anywhere
# along that edge without changing the dihedral at all. Ranking candidates by
# the direction to that third vertex therefore mixes in a component running
# along the edge, which has nothing to do with how far the triangle is folded.
#
# The geometry below is built so the two readings disagree: the triangle that
# actually blocks the agent is NOT the one the third-vertex direction favors.

SPINE_0, SPINE_1 = (0.0, 0.0, 0.0), (0.0, 2.0, 0.0)


def _page(dihedral_deg, y_of_tip):
    '''Third vertex of a triangle hinged on the spine at the given dihedral
    angle from the z=0 plane, placed at a chosen position along the spine.
    Only the dihedral should matter to the ranking; y should not.'''
    a = np.radians(dihedral_deg)
    return (-np.cos(a), y_of_tip, np.sin(a))


def _dihedral_book():
    '''Page 0 in the z=0 plane, plus two candidates hinged on the spine.

    BLOCKER is folded further over the agent (60 deg vs 20 deg), so an agent
    hugging page 0 reaches it first and cannot get past it to the other one.
    SHADOWED is nearly flat but has its third vertex far along the spine, which
    inflates the third-vertex angle above the blocker's.
    '''
    return h.book_3D(SPINE_0, SPINE_1,
                     [(2.0, 1.0, 0.0),                  # page 0, the agent's
                      _page(60.0, 1.0),                 # BLOCKER, index 1
                      _page(20.0, 3.0)])                # SHADOWED, index 2


def test_3d_slide_cannot_pass_through_the_blocking_triangle():
    '''The invariant, stated without reference to which element gets chosen: the
    agent may not end up on the far side of the triangle that stands between it
    and everything else at that edge.
    '''
    mesh = _dihedral_book()
    blocker = mesh[1]
    start = np.array([1.0, 1.0, 0.6])
    end = np.array([-0.6, 1.0, -0.4])
    newend, _, _ = h.call_static(start, end, mesh)
    h.assert_finite(newend, 'dihedral book')
    h.assert_not_penetrated_3D(start, newend, blocker[0], blocker[1], blocker[2])


def test_3d_candidate_ranking_ignores_position_along_the_shared_edge():
    '''Sliding the shadowed triangle's third vertex along the spine changes no
    angle in the problem, so it must not change the outcome.'''
    start = np.array([1.0, 1.0, 0.6])
    end = np.array([-0.6, 1.0, -0.4])

    results = []
    for y_tip in (1.0, 3.0, 6.0):
        mesh = h.book_3D(SPINE_0, SPINE_1,
                         [(2.0, 1.0, 0.0), _page(60.0, 1.0), _page(20.0, y_tip)])
        newend, _, _ = h.call_static(start, end, mesh)
        results.append(np.asarray(newend, float))

    for other in results[1:]:
        assert np.allclose(other, results[0], atol=h.POS_ATOL), (
            f'moving a third vertex along the shared edge changed the result: '
            f'{results[0]} vs {other}')


# --------------------------------------------------------------------------- #
#            moving meshes: the same joints, on a translating boundary        #
# --------------------------------------------------------------------------- #
# _project_and_slide_moving is a separate implementation of the same idea, and
# it selects among adjacent elements the same way. Moving boundaries are 2D
# only.
#
# The invariants differ in one respect: a moving boundary does work on an agent,
# so the resolved displacement may exceed the attempted one. It cannot exceed
# the attempt plus the distance the boundary itself travelled, which is what
# assert_displacement_bounded_moving checks.
#
# Candidate counts here were measured the same way as the static ones: the
# translating star reaches k = 1, 2, 3, 4 for 2, 3, 4, 5 spokes, and does so for
# every mesh translation tried, including none.

MOVING_SHIFTS = {'still': (0.0, 0.0), 'advancing': (0.3, 0.0),
                 'receding': (-0.3, 0.0), 'sideways': (0.0, 0.4)}

def _moving_star(n_spokes, shift):
    m0 = _star(n_spokes)
    return m0, m0 + np.asarray(shift, dtype=float)


@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
@pytest.mark.parametrize('shift', sorted(MOVING_SHIFTS))
@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_moving_star_junction_returns_a_finite_point(ib, shift, n_spokes):
    m0, m1 = _moving_star(n_spokes, MOVING_SHIFTS[shift])
    newend, _, _ = h.call_moving(STAR_START, STAR_END, m0, m1, ib_collisions=ib)
    h.assert_finite(newend, f'{n_spokes}-spoke moving star ({shift}, {ib})')


@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
@pytest.mark.parametrize('shift', sorted(MOVING_SHIFTS))
def test_moving_star_junction_does_not_amplify_motion(shift, n_spokes):
    m0, m1 = _moving_star(n_spokes, MOVING_SHIFTS[shift])
    newend, _, _ = h.call_moving(STAR_START, STAR_END, m0, m1)
    h.assert_displacement_bounded_moving(STAR_START, STAR_END, newend, m0, m1)


@pytest.mark.parametrize('n_fins', sorted(SQUARE_K))
@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_moving_finned_corner_keeps_the_agent_outside(ib, n_fins):
    '''The hard invariant on a moving obstacle: the agent must finish outside
    the square *where the square ends up*, not where it started.'''
    shift = np.array([0.25, 0.0])
    m0 = _square_with_fins(n_fins)
    m1 = m0 + shift
    newend, _, _ = h.call_moving(SQUARE_START, SQUARE_END, m0, m1,
                                 ib_collisions=ib)
    h.assert_finite(newend, f'{n_fins}-fin moving corner ({ib})')
    h.assert_outside_polygon(newend, np.asarray(SQUARE) + shift,
                             f'{n_fins}-fin moving corner ({ib})')


@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
def test_moving_star_junction_is_rotation_equivariant(n_spokes):
    '''Rotating the whole problem -- agent, mesh at both ends of the step --
    must rotate the answer. As in the static case this is the only check that
    can see an answer that is wrong rather than absent.'''
    m0, m1 = _moving_star(n_spokes, (0.3, 0.0))
    base, _, _ = h.call_moving(STAR_START, STAR_END, m0, m1)

    T = h.rigid_2D(np.pi/2)
    moved, _, _ = h.call_moving(T(STAR_START), T(STAR_END), T(m0), T(m1))

    assert np.allclose(np.asarray(moved, float), T(base), atol=1e-3), (
        f'{n_spokes}-spoke moving star: rotated problem gave '
        f'{np.asarray(moved)}, expected {T(base)}')


@pytest.mark.parametrize('n_fins', sorted(SQUARE_K))
def test_moving_finned_corner_is_rotation_equivariant(n_fins):
    '''The finned corner is the geometry whose static counterpart exposed the
    two-candidate case: it returns a plausible answer that stops being plausible
    once the problem is rotated.'''
    shift = np.array([0.25, 0.0])
    m0 = _square_with_fins(n_fins)
    m1 = m0 + shift
    base, _, _ = h.call_moving(SQUARE_START, SQUARE_END, m0, m1)

    T = h.rigid_2D(np.pi/2)
    moved, _, _ = h.call_moving(T(SQUARE_START), T(SQUARE_END), T(m0), T(m1))

    assert np.allclose(np.asarray(moved, float), T(base), atol=1e-3), (
        f'{n_fins}-fin moving corner: rotated problem gave '
        f'{np.asarray(moved)}, expected {T(base)}')


def test_moving_star_junction_carries_the_agent_with_the_boundary():
    '''A translating junction should leave the agent at the joint's new
    position, not its old one -- the moving-boundary behavior the static code
    has no analogue for.'''
    still_m0, still_m1 = _moving_star(4, (0.0, 0.0))
    still, _, _ = h.call_moving(STAR_START, STAR_END, still_m0, still_m1)

    shift = np.array([0.3, 0.0])
    m0, m1 = _moving_star(4, shift)
    moved, _, _ = h.call_moving(STAR_START, STAR_END, m0, m1)

    assert np.allclose(np.asarray(moved, float),
                       np.asarray(still, float) + shift, atol=h.POS_ATOL), (
        f'agent at {np.asarray(moved)}; a boundary that translated by {shift} '
        f'should have carried it from {np.asarray(still)}')


# --------------------------------------------------------------------------- #
#                       degenerate mesh elements                              #
# --------------------------------------------------------------------------- #
# A duplicated vertex in a .vertex file produces a zero-length element under
# either meshing method: 'adjacent' joins the duplicate to itself, and
# 'proximity' sees a pair at distance zero, comfortably inside any radius.
#
# Such an element has no direction to slide along. Normalizing it yields NaN,
# and the consequences differ by path: the static slider returned NaN positions,
# while the moving solver did not return at all.
#
# The invariant is that a degenerate element contributes nothing, so adding one
# to a mesh must leave the answer exactly as it was.

def _with_zero_length_spoke(mesh, at):
    return np.concatenate([mesh, h.segment(at, at)])


@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
def test_degenerate_element_does_not_change_the_static_answer(n_spokes):
    sound = _star(n_spokes)
    degenerate = _with_zero_length_spoke(sound, STAR_HUB)

    base, _, _ = h.call_static(STAR_START, STAR_END, sound)
    got, _, _ = h.call_static(STAR_START, STAR_END, degenerate)

    h.assert_finite(got, f'{n_spokes}-spoke star + zero-length spoke')
    assert np.allclose(np.asarray(got, float), np.asarray(base, float),
                       atol=h.POS_ATOL), (
        f'a zero-length element changed the result: {base} -> {got}')


@pytest.mark.parametrize('n_spokes', sorted(STAR_K))
def test_degenerate_element_does_not_change_the_moving_answer(n_spokes):
    # Before this was guarded, the moving solver did not merely give a wrong
    # answer on this input -- it never returned. A regression would therefore
    # hang the suite here rather than fail it.
    shift = np.array([0.3, 0.0])
    sound = _star(n_spokes)
    degenerate = _with_zero_length_spoke(sound, STAR_HUB)

    base, _, _ = h.call_moving(STAR_START, STAR_END, sound, sound + shift)
    got, _, _ = h.call_moving(STAR_START, STAR_END,
                              degenerate, degenerate + shift)

    h.assert_finite(got, f'{n_spokes}-spoke moving star + zero-length spoke')
    assert np.allclose(np.asarray(got, float), np.asarray(base, float),
                       atol=h.POS_ATOL), (
        f'a zero-length element changed the result: {base} -> {got}')


def test_degenerate_triangle_does_not_change_the_3d_answer():
    '''The 3D counterpart: a zero-area sliver whose apex lies on the shared
    edge, which has no dihedral to be ranked by.'''
    sound = h.book_3D(BOOK_E0, BOOK_E1, BOOK_TIPS[:3])
    sliver = h.book_3D(BOOK_E0, BOOK_E1, BOOK_TIPS[:3] + [(0.0, 1.0, 0.0)])

    base, _, _ = h.call_static(BOOK_START, BOOK_END, sound)
    got, _, _ = h.call_static(BOOK_START, BOOK_END, sliver)

    h.assert_finite(got, 'book + zero-area sliver')
    assert np.allclose(np.asarray(got, float), np.asarray(base, float),
                       atol=h.POS_ATOL)


@pytest.mark.parametrize('geom', ['2d wall', '3d triangle'])
def test_head_on_hit_computes_no_invalid_values(geom):
    '''A hit with no tangential component has no sliding direction, so the
    projection onto the element is the zero vector. Normalizing it produced a
    NaN that happened to go unused -- correct only because the code path that
    would have consumed it is not taken for a head-on hit.
    '''
    if geom == '2d wall':
        mesh = h.wall_segments(4, 5.0, y_lo=0.0, y_hi=10.0)
        start, end = (4.0, 5.0), (6.0, 5.0)
    else:
        mesh = h.triangle((0.0, 0.0, 1.0), (4.0, 0.0, 1.0), (0.0, 4.0, 1.0))
        start, end = (1.0, 1.0, 2.0), (1.0, 1.0, 0.0)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        newend, _, _ = h.call_static(start, end, mesh)
    h.assert_finite(newend, f'head-on {geom}')


# --------------------------------------------------------------------------- #
#              rotation equivariance on the junction geometries               #
# --------------------------------------------------------------------------- #
# The geometry-independent invariants live in test_collisions_invariants.py.
# This one stays here: it needs the junction meshes built above, and it is the
# check that found the bug those meshes were written for.

JUNCTION_CASES = {
    'star k=2':   (lambda: _star(3), STAR_START, STAR_END),
    'star k=3':   (lambda: _star(4), STAR_START, STAR_END),
    'finned k=2': (lambda: _square_with_fins(2), SQUARE_START, SQUARE_END),
    'finned k=3': (lambda: _square_with_fins(3), SQUARE_START, SQUARE_END),
}


@pytest.mark.parametrize('name', sorted(JUNCTION_CASES))
def test_junction_answer_is_rotation_equivariant(name):
    '''Rotating the whole problem must rotate the answer.

    This is the only check in the suite that can see a wrong-but-finite answer.
    Two candidates in 2D returns a plausible position and satisfies every
    arithmetic invariant; normalizing candidate vectors against the wrong axis
    mixes the coordinate components, which a rotation of the whole problem then
    exposes.
    '''
    build, start, end = JUNCTION_CASES[name]
    mesh = build()
    base, _, _ = h.call_static(start, end, mesh)

    T = h.rigid_2D(np.pi/2)
    moved, _, _ = h.call_static(T(start), T(end), T(mesh))

    assert np.allclose(np.asarray(moved, float), T(base), atol=1e-3), (
        f'{name}: rotated problem gave {np.asarray(moved)}, '
        f'expected {T(base)}')
