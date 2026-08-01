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

The final section holds general sanity checks that apply to any geometry at all,
including the chain-shaped ones already covered: results must be finite, motion
must not be amplified, and the answer must not depend on where the problem sits
in space or on the units it is expressed in.
'''

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
# The xfail marks below record that split precisely: each one is a specific
# (geometry, collision mode) combination, and strict=True means the mark itself
# fails the suite the moment the underlying defect is fixed.

XF_MULTI = pytest.mark.xfail(strict=True, reason=(
    'BUG-IB-JUNCTION: three or more candidate elements at a joint. The slide '
    'code normalizes the candidate vectors against a wrongly-shaped array of '
    'norms, which raises for most candidate counts.'))

XF_SILENT = pytest.mark.xfail(strict=True, reason=(
    'BUG-IB-JUNCTION: exactly two candidates returns an answer rather than '
    'raising, but one computed from mis-normalized vectors, so it is not '
    'invariant under rotation of the whole problem.'))


def _spoke_params():
    '''(ib, n_spokes) with the sliding cases that reach k >= 3 marked.'''
    out = []
    for ib in ('sliding', 'sticky'):
        for n in sorted(STAR_K):
            marks = [XF_MULTI] if (ib == 'sliding' and STAR_K[n] >= 3) else []
            out.append(pytest.param(ib, n, marks=marks, id=f'{ib}-{n}spokes'))
    return out


def _fin_params():
    out = []
    for ib in ('sliding', 'sticky'):
        for n in sorted(SQUARE_K):
            marks = [XF_MULTI] if (ib == 'sliding' and SQUARE_K[n] >= 3) else []
            out.append(pytest.param(ib, n, marks=marks, id=f'{ib}-{n}fins'))
    return out


def _book_params():
    '''3D raises for every candidate count except k == 3, where the wrong
    normalization happens to be shape-compatible and returns silently.'''
    out = []
    for ib in ('sliding', 'sticky'):
        for n in sorted(BOOK_K):
            bad = ib == 'sliding' and BOOK_K[n] >= 2 and BOOK_K[n] != 3
            out.append(pytest.param(ib, n, marks=[XF_MULTI] if bad else [],
                                    id=f'{ib}-{n}tri'))
    return out


@pytest.mark.parametrize('ib,n_spokes', _spoke_params())
def test_star_junction_returns_a_finite_point(ib, n_spokes):
    newend, _, _ = h.call_static(STAR_START, STAR_END, _star(n_spokes),
                                 ib_collisions=ib)
    h.assert_finite(newend, f'{n_spokes}-spoke star ({ib})')


@pytest.mark.parametrize('n_spokes', [
    pytest.param(n, marks=[XF_MULTI] if STAR_K[n] >= 3 else [])
    for n in sorted(STAR_K)])
def test_star_junction_does_not_amplify_motion(n_spokes):
    newend, _, _ = h.call_static(STAR_START, STAR_END, _star(n_spokes))
    h.assert_displacement_bounded(STAR_START, STAR_END, newend)


@pytest.mark.parametrize('ib,n_fins', _fin_params())
def test_finned_corner_keeps_the_agent_outside(ib, n_fins):
    '''The hard invariant, on a closed obstacle so that it is well posed: the
    agent starts outside the square and must finish outside it, wherever along
    the boundary the slide leaves it.'''
    newend, _, _ = h.call_static(SQUARE_START, SQUARE_END,
                                 _square_with_fins(n_fins), ib_collisions=ib)
    h.assert_finite(newend, f'{n_fins}-fin corner ({ib})')
    h.assert_outside_polygon(newend, SQUARE, f'{n_fins}-fin corner ({ib})')


@pytest.mark.parametrize('n_fins', [
    pytest.param(n, marks=[XF_MULTI] if SQUARE_K[n] >= 3 else [])
    for n in sorted(SQUARE_K)])
def test_finned_corner_does_not_amplify_motion(n_fins):
    newend, _, _ = h.call_static(SQUARE_START, SQUARE_END,
                                 _square_with_fins(n_fins))
    h.assert_displacement_bounded(SQUARE_START, SQUARE_END, newend)


@XF_MULTI
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

@pytest.mark.parametrize('ib,n_tri', _book_params())
def test_nonmanifold_edge_returns_a_finite_point(ib, n_tri):
    newend, _, _ = h.call_static(BOOK_START, BOOK_END, _book(n_tri),
                                 ib_collisions=ib)
    h.assert_finite(newend, f'{n_tri}-triangle book ({ib})')


@pytest.mark.parametrize('n_tri', [
    pytest.param(n, marks=[XF_MULTI] if (BOOK_K[n] >= 2 and BOOK_K[n] != 3) else [])
    for n in sorted(BOOK_K)])
def test_nonmanifold_edge_does_not_amplify_motion(n_tri):
    newend, _, _ = h.call_static(BOOK_START, BOOK_END, _book(n_tri))
    h.assert_displacement_bounded(BOOK_START, BOOK_END, newend)


# --------------------------------------------------------------------------- #
#                 general sanity checks, any geometry                         #
# --------------------------------------------------------------------------- #
# These say nothing about junctions; they are properties every collision must
# have. The chain-shaped cases are included so that a failure here is clearly
# attributable to the property and not to the exotic geometry.

def _vwall():
    return h.wall_segments(4, 5.0, y_lo=0.0, y_hi=10.0)


def _lcorner():
    return h.polyline([(2.0, 5.0), (5.0, 5.0), (5.0, 9.0)])


def _vgroove():
    return h.polyline([(2.0, 8.0), (5.0, 5.0), (8.0, 8.0)])


# name -> (mesh builder, start, end)
GEOMETRIES = {
    'vertical wall':   (_vwall,   (4.0, 5.0), (6.0, 6.0)),
    'convex L corner': (_lcorner, (3.0, 3.0), (4.5, 6.5)),
    'concave V groove': (_vgroove, (5.0, 8.0), (5.0, 5.5)),
    'star k=2':        (lambda: _star(3), STAR_START, STAR_END),
    'star k=3':        (lambda: _star(4), STAR_START, STAR_END),
    'finned k=2':      (lambda: _square_with_fins(2), SQUARE_START, SQUARE_END),
    'finned k=3':      (lambda: _square_with_fins(3), SQUARE_START, SQUARE_END),
}

CHAINS = ['vertical wall', 'convex L corner', 'concave V groove']
JUNCTIONS = ['star k=2', 'star k=3', 'finned k=2', 'finned k=3']

# Geometries reaching three or more candidates raise outright.
ALL_GEOMETRIES = [pytest.param(n, marks=[XF_MULTI] if n.endswith('k=3') else [])
                  for n in CHAINS + JUNCTIONS]


@pytest.mark.parametrize('name', ALL_GEOMETRIES)
def test_result_is_finite(name):
    build, start, end = GEOMETRIES[name]
    newend, _, _ = h.call_static(start, end, build())
    h.assert_finite(newend, name)


@pytest.mark.parametrize('name', ALL_GEOMETRIES)
def test_motion_is_never_amplified(name):
    build, start, end = GEOMETRIES[name]
    newend, _, _ = h.call_static(start, end, build())
    h.assert_displacement_bounded(start, end, newend)


@pytest.mark.parametrize('name', CHAINS)
@pytest.mark.parametrize('theta,offset', [
    (0.0, (13.0, 7.0)),                 # translation only
    (np.pi/2, (0.0, 0.0)),              # quarter turn, still axis-aligned
    (0.37, (0.0, 0.0)),                 # rotation off the axes
    (0.37, (11.0, 4.0)),                # both
])
def test_answer_is_independent_of_position_and_orientation(name, theta, offset):
    '''Collision resolution is a geometric operation: rotating and translating
    the whole problem must rotate and translate the answer. A result that
    depends on the absolute placement means something is keying off the
    coordinate axes -- the failure mode behind the axis-aligned sticky-wall bug
    fixed in 1.0.0.

    Offsets keep every coordinate positive, so the epsilon back-off (derived
    from the largest coordinate) stays in the same decade and the comparison is
    not swamped by it.
    '''
    build, start, end = GEOMETRIES[name]
    mesh = build()
    base, _, _ = h.call_static(start, end, mesh)

    T = h.rigid_2D(theta, offset)
    moved, _, _ = h.call_static(T(start), T(end), T(mesh))

    assert np.allclose(np.asarray(moved, float), T(base), atol=1e-3), (
        f'{name}: moved problem gave {np.asarray(moved)}, '
        f'expected {T(base)}')


@pytest.mark.parametrize('name', [
    'star k=2',
    pytest.param('finned k=2', marks=XF_SILENT),
    pytest.param('star k=3', marks=XF_MULTI),
    pytest.param('finned k=3', marks=XF_MULTI),
])
def test_junction_answer_is_rotation_equivariant(name):
    '''The same property as above, on the junction geometries.

    This is the only check in the suite that can see a wrong-but-finite answer.
    Two candidates in 2D returns a plausible position and satisfies every
    arithmetic invariant; normalizing candidate vectors against the wrong axis
    mixes the coordinate components, which a rotation of the whole problem then
    exposes.
    '''
    build, start, end = GEOMETRIES[name]
    mesh = build()
    base, _, _ = h.call_static(start, end, mesh)

    T = h.rigid_2D(np.pi/2)
    moved, _, _ = h.call_static(T(start), T(end), T(mesh))

    assert np.allclose(np.asarray(moved, float), T(base), atol=1e-3), (
        f'{name}: rotated problem gave {np.asarray(moved)}, '
        f'expected {T(base)}')


@pytest.mark.parametrize('scale', [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4])
def test_no_penetration_at_any_scale(scale):
    '''The same collision expressed in different units must resolve the same
    way. Here only the weakest consequence is asserted -- the agent stays on the
    side of the wall it approached from.
    '''
    wall = h.wall_segments(4, 5.0*scale, y_lo=0.0, y_hi=10.0*scale)
    start = np.array([4.0, 5.0])*scale
    end = np.array([6.0, 6.0])*scale
    newend, _, _ = h.call_static(start, end, wall)
    h.assert_finite(newend, f'scale {scale:g}')
    h.assert_not_penetrated_2D(start, newend, wall[0, 0], wall[0, 1],
                               atol=1e-4*scale)
    # at large scales the back-off itself is big enough to fling the agent
    # further than it ever tried to travel
    h.assert_displacement_bounded(start, end, newend, atol=h.POS_ATOL*scale)


def _wall_gap_at_scale(scale):
    '''Resolve the same wall collision at a given scale; return the distance the
    agent was left short of the wall, as a fraction of the geometry.'''
    wall = h.wall_segments(4, 5.0*scale, y_lo=0.0, y_hi=10.0*scale)
    start = np.array([4.0, 5.0])*scale
    end = np.array([6.0, 6.0])*scale
    newend, _, _ = h.call_static(start, end, wall)
    return abs(np.asarray(newend, float)[0] - 5.0*scale)/scale


@pytest.mark.parametrize('scale', [0.01, 0.1, 10.0, 100.0, 1000.0, 1e4])
def test_backoff_does_not_grow_with_scale(scale):
    '''And the stronger consequence: the agent is left essentially *on* the
    boundary. The epsilon back-off exists only to keep round-off from putting it
    on the wrong side, so as a fraction of the geometry it must not depend on
    how large the geometry is.

    Stated against the scale=1 answer rather than an absolute tolerance, so the
    test asserts scale-invariance itself and not some chosen constant. The
    factor of 100 is slack for the back-off being rounded to a power of ten.
    '''
    reference = _wall_gap_at_scale(1.0)
    gap = _wall_gap_at_scale(scale)
    assert gap <= max(reference, 1e-12)*100, (
        f'scale {scale:g}: back-off is {gap:.3g} of the geometry, against '
        f'{reference:.3g} at scale 1 -- a factor of {gap/max(reference,1e-30):.3g}')


def test_negative_coordinates_do_not_produce_nan():
    '''A mesh sitting entirely at negative coordinates is legal -- Environment
    does not require the domain to start at the origin, meshes can be assigned
    directly, and shift_ibmesh_to_match_LLC can move one into negative space.
    '''
    wall = h.wall_segments(4, -5.0, y_lo=-20.0, y_hi=-10.0)
    start = np.array([-6.0, -15.0])
    end = np.array([-4.0, -14.0])
    newend, _, _ = h.call_static(start, end, wall)
    h.assert_finite(newend, 'all-negative coordinates')
    h.assert_not_penetrated_2D(start, newend, wall[0, 0], wall[0, 1])


def test_zero_length_movement_is_a_no_op():
    '''An agent that does not move cannot collide with anything.'''
    wall = _vwall()
    start = np.array([4.0, 5.0])
    newend, _, idx = h.call_static(start, start, wall)
    h.assert_finite(newend, 'zero-length movement')
    assert np.allclose(np.asarray(newend, float), start, atol=h.POS_ATOL)
