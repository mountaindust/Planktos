'''Unit tests for the small helpers and guard rails inside planktos._ibc.

The collision modules next door drive whole agent trajectories against a mesh.
These cover the pieces underneath that, plus the paths that exist to fail
cleanly:

  _boundary_eps        the back-off distance used to keep a stopped agent off
                       the boundary it struck
  _point_in_triangle   containment test used by the 3D slider to decide whether
                       a projected slide is still on its triangle
  _clear_of_joint      where a 2D slider leaves an agent near a joint: EPS from
                       both elements' lines
  make_ib_worker       the parallel dispatch layer's shared-data unpacking
  the 3D-moving guard  moving meshes are 2D only; 3D must say so rather than
                       compute something wrong

Everything here is deterministic and analytically known -- no Environment,
Swarm, flow, or RNG.
'''

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h
from planktos import _ibc


# --------------------------------------------------------------------------- #
#                    _boundary_eps: back-off scaled to the geometry           #
# --------------------------------------------------------------------------- #
# The back-off has to dominate double-precision round-off (~1e-16 relative) while
# staying physically negligible, so it is a fixed fraction of the largest
# coordinate in play rather than a fixed distance. It was once a constant, which
# is why a domain far from the origin could round an agent through a boundary.

@pytest.mark.parametrize('coords,expected', [
    (np.array([1.0, 0.0]), 1e-7),
    (np.array([5.0, 0.0]), 1e-6),
    (np.array([0.03, 0.0]), 1e-8),
    (np.array([-250.0, 3.0]), 1e-4),
])
def test_boundary_eps_tracks_coordinate_magnitude(coords, expected):
    assert _ibc._boundary_eps(coords) == pytest.approx(expected)


def test_boundary_eps_ignores_sign():
    '''Magnitude is what matters; a domain at negative coordinates must get the
    same back-off as its mirror image. Getting this wrong gave NaN back-offs at
    negative coordinates.'''
    assert (_ibc._boundary_eps(np.array([-5.0, -2.0])) ==
            _ibc._boundary_eps(np.array([5.0, 2.0])))


def test_boundary_eps_at_the_origin_falls_back_to_unit_scale():
    '''Every coordinate zero leaves no magnitude to scale to, and log10(0) is
    -inf. Fall back to a scale of 1 rather than propagate that.'''
    eps = _ibc._boundary_eps(np.zeros(2), np.zeros(2))
    assert np.isfinite(eps)
    assert eps == pytest.approx(1e-7)


def test_boundary_eps_spans_all_the_arrays_it_is_given():
    '''It is called with several arrays at once -- start point, end point, and
    the mesh -- and must scale to the largest coordinate among all of them, not
    just the first.'''
    small = np.array([0.5, 0.5])
    big = np.array([[[0.0, 0.0], [400.0, 0.0]]])
    assert (_ibc._boundary_eps(small, small, big) ==
            _ibc._boundary_eps(big))


# --------------------------------------------------------------------------- #
#                _point_in_triangle: containment for the 3D slider            #
# --------------------------------------------------------------------------- #

TRI_A = np.array([0.0, 0.0, 0.0])
TRI_B = np.array([1.0, 0.0, 0.0])
TRI_C = np.array([0.0, 1.0, 0.0])


@pytest.mark.parametrize('label,P,expected', [
    ('centroid', np.array([1/3, 1/3, 0.0]), True),
    ('well outside', np.array([1.0, 1.0, 0.0]), False),
    ('on an edge', np.array([0.5, 0.0, 0.0]), True),
    ('on a vertex', TRI_A, True),
])
def test_point_in_triangle(label, P, expected):
    # A point on the boundary counts as inside: the slider uses this to ask
    # whether a slide is still on its element, and a slide that has arrived
    # exactly at an edge has not left yet.
    # bool() because the barycentric path returns a numpy bool while the
    # degenerate branch returns a Python one; only truthiness is contracted.
    assert bool(_ibc._point_in_triangle(P, TRI_A, TRI_B, TRI_C)) is expected


@pytest.mark.parametrize('label,verts', [
    ('collinear vertices', (TRI_A, TRI_B, np.array([2.0, 0.0, 0.0]))),
    ('all three coincident', (TRI_A, TRI_A, TRI_A)),
])
def test_degenerate_triangle_contains_nothing(label, verts):
    '''A zero-area triangle has no interior and no normal to build barycentric
    coordinates from. Report no containment rather than divide by zero -- a
    duplicated vertex in a mesh file is enough to produce one.'''
    Q0, Q1, Q2 = verts
    # a point that lies on the degenerate "triangle" is still not contained
    assert bool(_ibc._point_in_triangle(np.array([0.5, 0.0, 0.0]),
                                        Q0, Q1, Q2)) is False


# --------------------------------------------------------------------------- #
#                  make_ib_worker: parallel dispatch unpacking                #
# --------------------------------------------------------------------------- #
# Swarm.apply_boundary_conditions packs the per-timestep mesh data into a tagged
# tuple once, then binds it into a one-argument callable so each agent's task
# stays small and picklable. These check the unpacking round-trips to the same
# answer the serial call gives, and that an unknown tag is refused.

def test_static_worker_matches_the_direct_call():
    mesh = h.wall_segments(20, 5.0)
    start = np.array([4.9, 5.0]); end = np.array([5.3, 5.4])
    shared = ('static', mesh, h.max_meshpt_dist(mesh), 'sliding')

    worker = _ibc.make_ib_worker(shared)
    agent_idx, result = worker((7, start, end))

    assert agent_idx == 7, 'the agent index must be echoed back for reordering'
    direct = _ibc.apply_internal_static_BC(start, end, mesh,
                                           h.max_meshpt_dist(mesh),
                                           ib_collisions='sliding')
    assert np.allclose(result[0], direct[0])
    assert result[2] == direct[2]


def test_moving_worker_matches_the_direct_call():
    start_mesh = h.wall_segments(20, 5.0)
    end_mesh = h.wall_segments(20, 6.0)
    start = np.array([4.8, 5.0]); end = np.array([6.3, 5.0])
    mmd = h.max_meshpt_dist(start_mesh)
    max_mov = float(np.linalg.norm(end_mesh - start_mesh, axis=-1).max())
    shared = ('moving', start_mesh, end_mesh, mmd, max_mov, 'sliding')

    worker = _ibc.make_ib_worker(shared)
    agent_idx, result = worker((3, start, end))

    assert agent_idx == 3
    direct = _ibc.apply_internal_moving_BC(start, end, start_mesh, end_mesh,
                                           mmd, max_mov, ib_collisions='sliding')
    assert np.allclose(result[0], direct[0])
    assert result[2] == direct[2]


def test_unknown_shared_tag_is_refused():
    '''The tag selects how the rest of the tuple is unpacked, so an unrecognized
    one cannot be guessed at -- it would unpack the wrong number of items or
    silently treat a moving mesh as static.'''
    with pytest.raises(ValueError, match='Unknown IB shared-data tag'):
        _ibc.make_ib_worker(('bogus', 1, 2, 3))


# --------------------------------------------------------------------------- #
#                    moving meshes are 2D only, and say so                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
def test_3d_moving_mesh_raises_not_implemented(ib):
    '''3D moving boundaries are planned but not built. The guard matters because
    the 2D intersection routine would otherwise be handed 3D input: it must
    refuse rather than compute a confidently wrong answer.'''
    start = np.zeros(3); end = np.ones(3)
    start_mesh = np.zeros((1, 3, 3)); end_mesh = np.ones((1, 3, 3))
    with pytest.raises(NotImplementedError, match='3D moving meshes'):
        _ibc.apply_internal_moving_BC(start, end, start_mesh, end_mesh,
                                      1.0, 1.0, ib_collisions=ib)


# --------------------------------------------------------------------------- #
#                 backing a point off near a 2D joint                         #
# --------------------------------------------------------------------------- #
# The element runs from the vertex V along +x, with the agent above it; the
# adjacent element leaves V at angle phi.

EPS_J = 1e-6
V_J = np.array([5.0, 5.0])
N_UP = np.array([0.0, 1.0])
C_U = np.array([1.0, 0.0])


def _joint(pt, phi):
    adj_u = np.array([np.cos(phi), np.sin(phi)])
    out = _ibc._clear_of_joint(np.asarray(pt, float), N_UP, V_J, C_U, adj_u,
                               EPS_J)
    n_adj = C_U - np.dot(C_U, adj_u)*adj_u
    n_adj = n_adj/np.linalg.norm(n_adj) if np.linalg.norm(n_adj) > 0 else N_UP
    return out, np.dot(out - V_J, N_UP), np.dot(out - V_J, n_adj)


@pytest.mark.parametrize('phi_deg', [90.0, 45.0, 135.0, 179.5])
def test_point_at_a_concave_joint_ends_EPS_from_both_lines(phi_deg):
    out, d_cur, d_adj = _joint(V_J, np.deg2rad(phi_deg))
    assert np.isclose(d_cur, EPS_J, rtol=1e-9) and         np.isclose(d_adj, EPS_J, rtol=1e-9), (d_cur, d_adj)


def test_right_angle_joint_puts_the_point_at_the_corner_of_the_clearance():
    out, _, _ = _joint(V_J, np.pi/2)
    assert np.allclose(out, V_J + EPS_J, rtol=0, atol=1e-15)


@pytest.mark.parametrize('phi', [np.pi, -0.5])
def test_collinear_or_convex_joint_backs_off_along_the_normal(phi):
    pt = V_J + [3e-7, 0.0]
    out, _, _ = _joint(pt, phi)
    assert np.array_equal(out, pt + EPS_J*N_UP)


def test_point_already_clear_of_the_adjacent_line_is_left_alone():
    pt = V_J + [5*EPS_J, 0.0]      # 45 degree joint: 3.5 EPS from the adjacent
    out, _, _ = _joint(pt, np.pi/4)
    assert np.array_equal(out, pt + EPS_J*N_UP)


def test_near_fold_stays_on_the_agent_side_of_both_lines():
    out, d_cur, d_adj = _joint(V_J, 1e-5)
    assert d_cur > 0 and d_adj > 0, (d_cur, d_adj)
    assert np.linalg.norm(out - V_J) <= 1e3*EPS_J
