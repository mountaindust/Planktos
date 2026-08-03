'''No-penetration and known-answer tests for *moving* immersed-boundary collisions
-- the core new capability of this branch.

These call planktos._ibc.apply_internal_moving_BC directly: the boundary is given
as two mesh snapshots (start_mesh, end_mesh) and deforms linearly across the step,
while the agent travels startpt->endpt. No Environment, Swarm, flow, or RNG, so the
result is an exact function of geometry and the tests are fast.

Three defects found while writing these have since been fixed, and the cases that
caught them are kept below as regressions: BUG-STICKY-AXIS (the sticky contact
parameter went 0/0 -> NaN on a perfectly axis-aligned moving element),
BUG-ZEROLEN-SEG (in _geom.py, also pinned in test_geom.py), and BUG-TCRIT (the
slider's critical times, final section). Most cases here use a *moving agent*
(startpt != endpt), which is what the second of those was about.
'''

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h
from _ib_harness import POS_ATOL
from planktos import _ibc

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
        start_mesh = h.horizontal_wall(20, 5.0); end_mesh = h.horizontal_wall(20, 6.0)
        start = np.array([5.0, 4.8]); end = np.array([5.0, 6.3])
        expected, axis = [5.0, 6.0], 1
    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sticky')
    assert not np.isnan(newend).any()
    assert idx is not None
    assert newend[axis] <= 6.0 + POS_ATOL, "penetrated past the wall's final position"
    assert np.allclose(newend, expected, atol=POS_ATOL)


@pytest.mark.parametrize('ib', ['sticky', 'sliding'])
def test_single_element_moving_mesh_reports_a_real_index(ib):
    '''A one-element mesh is the case that could plausibly yield a None index,
    and the sticky path subscripts the mesh with whatever comes back.

    _geom reports a None index only for the single-element *form* of its input,
    where the element arrives as bare 1D arrays and there is nothing to index.
    apply_internal_moving_BC always assembles (N,2) arrays, so N=1 still gets an
    ordinary index. Pinned because the sticky path depends on it.
    '''
    start_mesh = h.segment((0.0, 0.0), (1.0, 0.0))
    end_mesh = h.segment((0.0, 0.1), (1.0, 0.1))
    start = np.array([0.5, 0.5]); end = np.array([0.5, -0.2])

    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, ib)

    assert idx is not None, "index came back None; the sticky path subscripts with it"
    assert int(idx) == 0, f"expected element 0, got {idx!r}"
    # The agent drops straight onto a wall rising from y=0 to y=0.1, so it has no
    # tangential motion and both modes leave it riding the wall's final position.
    assert np.allclose(newend, [0.5, 0.1], atol=POS_ATOL), \
        f"expected the agent to ride the wall to y=0.1, got {newend}"


# --------------------------------------------------------------------------- #
#        sliding off a free end and resuming the original trajectory          #
# --------------------------------------------------------------------------- #

def test_moving_slide_off_free_end_resumes_original_trajectory():
    '''The moving counterpart of the static free-flight case: the agent runs off
    an end with no element attached and flies the rest of its original movement.

    Element (0,0)-(1,0) translating straight up to y=0.1 across the step. The
    agent goes (0.1,0.3) -> (1.3,-0.3), so vec = (1.2,-0.6) and it closes on the
    rising element at 0.7/step: contact at t=3/7, at x=0.6143. It then slides in
    +x at 1.2/step while being carried up with the element, and reaches the free
    end x=1 at t=3/4, by which time the element has risen to y=0.075. The last
    quarter of the movement is flown unobstructed:
        (1,0.075) + (1/4)*(1.2,-0.6) = (1.3,-0.075).
    '''
    start_mesh = h.segment((0.0, 0.0), (1.0, 0.0))
    end_mesh = h.segment((0.0, 0.1), (1.0, 0.1))
    start = np.array([0.1, 0.3]); end = np.array([1.3, -0.3])

    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sliding')

    assert idx == 0
    assert np.allclose(newend, [1.3, -0.075], atol=POS_ATOL), \
        f"expected the free-flight endpoint (1.3,-0.075), got {newend}"
    h.assert_displacement_bounded_moving(start, end, newend, start_mesh, end_mesh)


# --------------------------------------------------------------------------- #
#          the boundary turning out from under the agent (release branch)     #
# --------------------------------------------------------------------------- #
# _project_and_slide_moving has a branch for an element that turns away from a
# sliding agent faster than the agent presses into it. It solves for the time
# t_rot at which the two perpendicular speeds matched, releases the agent there,
# and lets it continue on its original trajectory. Nothing in the suite, and
# nothing in the examples, had ever reached it -- so the largest uncovered block
# in _ibc.py was also the one containing a root find.
#
# Reaching it needs an element pivoting about an interior point; see
# _ib_harness.pivoting_segment for why an endpoint pivot cannot do it. The agent
# makes contact on the half sweeping toward it, slides outward past the pivot
# onto the half sweeping away, and the recession rate grows with distance from
# the pivot until it overtakes the agent.
#
# _ibc calls optimize.root_scalar in exactly one place -- this branch -- so
# counting those calls is a faithful detector of whether it ran.

PIVOT_CENTER, PIVOT_LENGTH = (0.0, 0.0), 4.0

# (turn, start, end). Measured to enter the branch by the counter below, not
# assumed: the branch fires only in a window of turn rates, because a slower
# element never outruns the agent and a faster one is never touched at all.
PIVOT_CASES = [
    (0.2, (-1.0, 0.5), (2.0, -0.5)),
    (0.3, (-1.0, 0.25), (2.0, -0.5)),
    (0.3, (-2.0, 0.5), (2.5, -0.75)),
    (0.4, (-1.5, 0.5), (3.0, -1.0)),
]


@contextmanager
def release_branch_calls():
    '''Collect the release times found by the moving slider's rotation branch.

    Patches scipy.optimize.root_scalar, which _ibc reaches in that branch alone,
    and restores it afterwards even if the body raises.
    '''
    found = []
    real_root_scalar = optimize.root_scalar

    def counting(*args, **kwargs):
        sol = real_root_scalar(*args, **kwargs)
        found.append(sol.root)
        return sol

    _ibc.optimize.root_scalar = counting
    try:
        yield found
    finally:
        _ibc.optimize.root_scalar = real_root_scalar


@pytest.mark.parametrize('turn,start,end', PIVOT_CASES)
def test_pivoting_element_reaches_the_release_branch(turn, start, end):
    '''Assert the cases below really do reach the branch. Without this the rest
    of this section could go on passing while silently covering nothing, which
    is exactly how the branch stayed unreached in the first place.'''
    m0, m1 = h.pivoting_segment(PIVOT_CENTER, PIVOT_LENGTH, 0.0, -turn)
    with release_branch_calls() as released_at:
        h.call_moving(np.array(start), np.array(end), m0, m1)
    assert len(released_at) == 1, \
        f"release branch ran {len(released_at)} times, expected exactly once"
    assert 0.0 < released_at[0] < 1.0, \
        f"release time {released_at[0]} lies outside the step"


@pytest.mark.parametrize('turn,start,end', PIVOT_CASES)
def test_pivoting_element_release_obeys_the_invariants(turn, start, end):
    '''There is no independent specification for where a released agent lands,
    so assert what must hold under any correct release. Note that the usual
    no-penetration check does not apply to a lone pivoting element: the boundary
    sweeps across the agent's own start point, so "the side it started on" is not
    well defined. The closed-obstacle test below supplies that invariant instead.
    '''
    m0, m1 = h.pivoting_segment(PIVOT_CENTER, PIVOT_LENGTH, 0.0, -turn)
    start = np.array(start); end = np.array(end)

    newend, dx, idx = h.call_moving(start, end, m0, m1)

    assert idx is not None, "expected contact with the pivoting element"
    h.assert_finite(newend, 'released agent')
    h.assert_displacement_bounded_moving(start, end, newend, m0, m1)


@pytest.mark.parametrize('turn,start,end', PIVOT_CASES)
def test_pivoting_element_release_is_rotation_equivariant(turn, start, end):
    '''Rotating the whole problem must rotate the answer. As elsewhere in the
    suite this is the check that can see an answer which is wrong rather than
    absent -- the release time comes out of a root find, and nothing else here
    would notice it converging to the wrong root.'''
    m0, m1 = h.pivoting_segment(PIVOT_CENTER, PIVOT_LENGTH, 0.0, -turn)
    base, _, _ = h.call_moving(np.array(start), np.array(end), m0, m1)

    T = h.rigid_2D(0.7, (3.0, -2.0))
    moved, _, _ = h.call_moving(T(start), T(end), T(m0), T(m1))

    assert np.allclose(np.asarray(moved, float), T(base), atol=1e-5), \
        f"rotated problem gave {np.asarray(moved)}, expected {T(base)}"


def test_faster_turn_leaves_more_of_the_agents_movement_intact():
    '''The physical content of the release branch, as a monotone trend.

    A stationary element blocks the agent's approach entirely. One turning away
    fast enough is never touched, and the agent completes its movement. In
    between, the boundary gives way partway through the step -- which is what the
    release branch computes. A release at the wrong time would show up here as a
    non-monotone or reversed trend, which no single-case assertion can see.
    '''
    start = np.array([-1.0, 0.5]); end = np.array([2.0, -0.5])
    shortfall = []
    for turn in (0.0, 0.1, 0.2, 0.3):
        m0, m1 = h.pivoting_segment(PIVOT_CENTER, PIVOT_LENGTH, 0.0, -turn)
        newend, _, _ = h.call_moving(start, end, m0, m1)
        shortfall.append(float(np.linalg.norm(newend - end)))

    assert all(a > b for a, b in zip(shortfall, shortfall[1:])), \
        f"shortfall from the intended endpoint is not decreasing: {shortfall}"
    assert shortfall[0] > 0.4, \
        f"a stationary element should block most of the movement, got {shortfall[0]}"
    assert shortfall[-1] < POS_ATOL, \
        f"a fast enough turn should never touch the agent, got {shortfall[-1]}"


# A closed obstacle, so that "inside" is defined and the project's hard
# invariant can be asserted on a release. Each edge of a square turning about
# its centre has a normal velocity that varies linearly along it and changes
# sign at its midpoint, so the same mechanism as the lone pivoting segment
# applies -- the agent contacts the rising part of an edge and slides out onto
# the falling part.
TURNING_SQUARE = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
TURNING_SQUARE_TURN = 0.7
TURNING_SQUARE_START, TURNING_SQUARE_END = (-1.0, 1.1), (1.5, 0.8)


def test_release_from_a_turning_obstacle_keeps_the_agent_outside():
    '''The hard invariant, on a release: the agent must finish outside the
    obstacle *where the obstacle ends up*, not where it started.'''
    corners_end = h.rigid_2D(-TURNING_SQUARE_TURN)(TURNING_SQUARE)
    m0 = h.closed_polygon(TURNING_SQUARE)
    m1 = h.closed_polygon(corners_end)
    start = np.array(TURNING_SQUARE_START); end = np.array(TURNING_SQUARE_END)

    with release_branch_calls() as released_at:
        newend, dx, idx = h.call_moving(start, end, m0, m1)

    assert released_at, "expected this case to reach the release branch"
    assert idx == 2, f"expected contact with the top edge, got element {idx}"
    h.assert_finite(newend, 'agent released by turning square')
    h.assert_outside_polygon(newend, corners_end, 'agent released by turning square')
    h.assert_displacement_bounded_moving(start, end, newend, m0, m1)


# --------------------------------------------------------------------------- #
#             contact arriving at the very end of the step                    #
# --------------------------------------------------------------------------- #

def test_contact_at_the_very_end_of_the_step_just_backs_off():
    '''With essentially none of the step left to slide through, the slider skips
    the whole projection and simply places the agent off the boundary it just
    reached. Worth its own case because that shortcut returns before any of the
    machinery the other tests exercise.

    A wall rising from y=0 to y=0.1 and an agent dropping from y=0.5 to a hair
    under y=0.1 meet at t = 0.5/(0.5+1e-9), which is 1 to within 2e-9.
    '''
    start_mesh = h.segment((0.0, 0.0), (1.0, 0.0))
    end_mesh = h.segment((0.0, 0.1), (1.0, 0.1))
    start = np.array([0.5, 0.5])
    end = np.array([0.5, 0.1 - 1e-9])

    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, 'sliding')

    assert idx == 0, 'expected contact with the rising wall'
    assert np.allclose(newend, [0.5, 0.1], atol=POS_ATOL), \
        f'expected the agent left at the contact point (0.5,0.1), got {newend}'
    assert newend[1] >= 0.1, 'agent was left below the wall it just met'


# --------------------------------------------------------------------------- #
#              the travel reversal as the binding critical time               #
# --------------------------------------------------------------------------- #
# The slider re-checks two critical times when an agent is still on its element
# at the end of the step: the element's shortest moment, and where the agent's
# direction of travel along it reverses. The reversal is the binding one when it
# comes first, which needs the element to turn through the perpendicular to the
# agent's travel -- something no translating or gently tilting wall does.

def test_travel_reversal_can_be_the_binding_critical_time():
    '''An element pivoting from 60 to 120 degrees while the agent travels
    straight in +x. Its projection onto the agent's travel goes from +2 to -2,
    so the direction of sliding reverses partway through the step.

    There is no closed form for where the agent lands, so this asserts the
    invariants plus equivariance. As with the release branch, the usual
    no-penetration check does not apply to a lone pivoting element: it sweeps
    across the agent's own start point.
    '''
    m0, m1 = h.pivoting_segment((0.0, 0.0), 2.0,
                                np.radians(60), np.radians(120))
    start = np.array([-1.0, 0.5]); end = np.array([1.0, 0.5])
    vec = end - start

    # the reversal exists: the element's projection onto the travel flips sign
    assert (np.dot(vec, m0[0, 1] - m0[0, 0]) *
            np.dot(vec, m1[0, 1] - m1[0, 0])) < 0, \
        'this element does not turn through the perpendicular to the travel'

    newend, dx, idx = h.call_moving(start, end, m0, m1)

    assert idx is not None, 'expected contact with the pivoting element'
    h.assert_finite(newend, 'slide through a travel reversal')
    h.assert_displacement_bounded_moving(start, end, newend, m0, m1)

    T = h.rigid_2D(0.7, (3.0, -2.0))
    moved, _, _ = h.call_moving(T(start), T(end), T(m0), T(m1))
    assert np.allclose(np.asarray(moved, float), T(newend), atol=1e-5), \
        f'rotated problem gave {np.asarray(moved)}, expected {T(newend)}'


# --------------------------------------------------------------------------- #
#           a wedged agent on a moving joint, either vertex order             #
# --------------------------------------------------------------------------- #

def test_moving_wedge_rides_the_joint_whatever_the_vertex_order():
    '''An agent that slides into a moving joint it cannot advance past is left
    at that joint and carried with it.

    Which of the adjacent element's two vertices is the shared one is an
    accident of how the mesh file was written, and the slider has a branch per
    case. Both must give the same answer, so both are run here and compared:
    that is the only way to see the two branches disagree.

    The focal element is traversed left to right, so the agent leaves by its
    *second* vertex -- the other of a similar pair of branches, and the one no
    other moving case reaches.
    '''
    joint = np.array([5.0, 5.0])
    shift = np.array([0.05, 0.0])
    focal = h.segment((2.0, 5.0), tuple(joint))     # agent slides off its Q1
    ramp_tip = (5.6, 7.5)                           # leans back over the focal
    start = np.array([4.2, 6.4])                    # diving steeply into focal
    end = np.array([5.2, 4.2])                      # overshooting the joint

    results = {}
    for label, ramp in (('joint first', h.segment(tuple(joint), ramp_tip)),
                        ('joint second', h.segment(ramp_tip, tuple(joint)))):
        m0 = np.concatenate([focal, ramp])
        m1 = m0 + shift
        newend, dx, idx = h.call_moving(start, end, m0, m1)
        h.assert_finite(newend, f'moving wedge ({label})')
        # wedged at the joint, which has itself moved with the mesh
        assert np.allclose(newend, joint + shift, atol=POS_ATOL), \
            f'{label}: expected the agent at the joint {joint+shift}, got {newend}'
        # backed off it, not sitting exactly on it
        offset = float(np.linalg.norm(newend - (joint + shift)))
        assert 0 < offset < POS_ATOL, \
            f'{label}: agent left {offset:.3e} from the vertex'
        results[label] = newend

    assert np.allclose(results['joint first'], results['joint second'],
                       atol=1e-12), \
        ('the resolved position changed with the order the adjacent element\'s '
         f'vertices were stored: {results}')


# --------------------------------------------------------------------------- #
#       golden multi-step trajectory (drift detector for moving collisions)    #
# --------------------------------------------------------------------------- #
# The cases above are single-step known answers. This pins a full deterministic
# multi-step Swarm.move() run through a translating wall (h.run_moving_golden) so
# any unintended change in the moving-collision behavior shows up as a diff. The
# baseline was generated from the trusted code and independently satisfies the
# no-penetration invariant (asserted separately below, so the lock is not purely
# circular). Regenerate GOLDEN_SLIDING only after a deliberate, reviewed change.
#
# Regenerated once, when the boundary back-off was rescaled (_ibc._boundary_eps).
# Ten of the 56 entries moved, all of them agents resting *on* the wall and all
# by the difference between the old and new back-off: 6.49999 -> 6.499999 and so
# on. Contacts, ordering and wall positions were unchanged, and the independent
# no-penetration check below passed across the change.

GOLDEN_SLIDING = np.array([
    [[4.00000000, 3.00000000], [4.00000000, 7.00000000], [2.00000000, 5.00000000], [6.00000000, 5.00000000]],
    [[5.00000000, 3.00000000], [4.80000000, 7.40000000], [2.30000000, 5.00000000], [6.20000000, 5.00000000]],
    [[6.00000000, 3.00000000], [5.60000000, 7.80000000], [2.60000000, 5.00000000], [6.40000000, 5.00000000]],
    [[6.49999900, 3.00000000], [6.40000000, 8.20000000], [2.90000000, 5.00000000], [6.60000000, 5.00000000]],
    [[6.99999900, 3.00000000], [6.99999900, 8.60000000], [3.20000000, 5.00000000], [7.00000100, 5.00000000]],
    [[7.49999900, 3.00000000], [7.49999900, 9.00000000], [3.50000000, 5.00000000], [7.50000100, 5.00000000]],
    [[7.99999900, 3.00000000], [7.99999900, 9.40000000], [3.80000000, 5.00000000], [8.00000100, 5.00000000]],
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


# --------------------------------------------------------------------------- #
#        frame independence of the slide (regression: BUG-TCRIT)              #
# --------------------------------------------------------------------------- #
# When the end-of-step check does not find that a sliding agent ran off the end
# of its element, _project_and_slide_moving refines the verdict at two critical
# times during the step: where the element's interpolated length is stationary,
# and where the agent's direction of travel along it reverses.
#
# Both were computed wrongly (BUG-TCRIT, now fixed). The length-minimum time was
# written with a bare vector of ones standing in for Q_tI, which cannot appear in
# a rotation-invariant geometric quantity -- so the resolved position depended on
# how the problem happened to sit in the coordinate frame. Separately, both times
# were solved for in the normalized u = (t-t_I)/(1-t_I) but then compared against
# (t_I, 1) and used as times, which agree only when t_I = 0.
#
# The geometry below is one that exposed it: an element that both turns and
# changes length, contacted partway through the step. The failure was silent --
# no exception, just a wrong position -- so a rotation is what makes it visible.

TCRIT_START = np.array([0.99, 0.15])
TCRIT_END = np.array([-0.01, 2.06])
TCRIT_MESH_START = np.array([[[-0.52, -0.50], [0.98, 0.27]]])
TCRIT_MESH_END = np.array([[[-0.85, -0.50], [0.82, -0.35]]])

# Several angles rather than one: the frame dependence was continuous, but it
# only moved the resolved position when the wrong critical time flipped the
# went-past-the-end verdict, which 0.9 rad does for this geometry.
TCRIT_ANGLES = (0.0, 0.3, 0.6, 0.9, 1.2, np.pi/2, 2.0, 2.5, 3.0)


def test_moving_slide_answer_does_not_depend_on_frame_orientation():
    '''Rotating the whole problem -- agent and mesh at both ends of the step --
    must rotate the answer and nothing else. Physics does not know which way the
    axes point.'''
    base, _, _ = h.call_moving(TCRIT_START, TCRIT_END,
                               TCRIT_MESH_START, TCRIT_MESH_END)
    for theta in TCRIT_ANGLES:
        T = h.rigid_2D(theta, (5.0, -3.0))
        moved, _, _ = h.call_moving(T(TCRIT_START), T(TCRIT_END),
                                    T(TCRIT_MESH_START), T(TCRIT_MESH_END))
        assert np.allclose(np.asarray(moved, float), T(base), atol=1e-5), \
            (f'turning the problem by {theta:.4f} rad moved the answer: got '
             f'{np.asarray(moved)}, expected {T(base)}')
