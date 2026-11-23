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
import warnings
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
@pytest.mark.parametrize('motion', ['stationary', 'translating'])
def test_anti_diagonal_moving_element_returns_on_the_near_side(motion, ib):
    '''An element exactly parallel to (1,-1) made the slider's contact parameter
    0/0, and the NaN sent solve_ivp into a loop that never returned, for a
    stationary moving mesh as much as a moving one.

    The element runs (4,6)-(6,4), on the line x+y=10, and the agent comes from
    (3,3), where x+y=6. A stationary moving mesh must give the static answer.
    '''
    start_mesh = h.segment((4.0, 6.0), (6.0, 4.0))
    shift = 0.5 if motion == 'translating' else 0.0
    end_mesh = start_mesh + shift
    start = np.array([3.0, 3.0]); end = np.array([7.0, 7.5])

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh, ib)

    h.assert_finite(newend, 'agent on an anti-diagonal element')
    assert idx == 0
    line = 10.0 + 2*shift          # the element's final line is x+y = line
    assert newend.sum() < line, f'agent ended on the far side: {newend}'
    if motion == 'stationary' and ib == 'sliding':
        static_end = h.call_static(start, end, start_mesh, ib)[0]
        assert np.allclose(newend, static_end, rtol=0, atol=1e-10), \
            f'moving {newend} != static {static_end}'


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


def test_slide_path_stays_on_the_element_line_past_its_first_vertex(monkeypatch):
    '''The slider integrates the agent's path along a moving element with
    solve_ivp, and evaluates that path past the element's ends (the end-of-step
    position, the run-off time solve). The path must stay on the element's line
    there too. The ODE used |s| for the barycentric coordinate s, which has the
    wrong sign once the agent passes Q0 and carried the path up to 0.63 off the
    line on a turning, stretching element.

    The element turns and stretches from (5,5)-(6,5.2) at contact (t_I=0.1) to
    (5.3,5.4)-(6.1,6.3) at t=1. The agent meets it at s=0.25 travelling
    vec=(-1.5,0.3) and slides past Q0 at t~0.24. Every solve_ivp call the slider
    makes is recorded and its endpoint checked against the element at that time.
    '''
    t_I = 0.1
    Q0, Q1 = np.array([5.0, 5.0]), np.array([6.0, 5.2])
    Q0_end, Q1_end = np.array([5.3, 5.4]), np.array([6.1, 6.3])
    vec = np.array([-1.5, 0.3])
    contact = Q0 + 0.25*(Q1 - Q0)
    start = contact - t_I*vec; end = start + vec
    at = lambda A, B, t: A + (B - A)*(t - t_I)/(1 - t_I)  # vertex at time t
    start_mesh = np.array([[at(Q0, Q0_end, 0.0), at(Q1, Q1_end, 0.0)]])
    end_mesh = np.array([[Q0_end, Q1_end]])

    from scipy import integrate
    real_solve_ivp = integrate.solve_ivp
    ends = []
    def recording_solve_ivp(fun, t_span, y0, **kwargs):
        sol = real_solve_ivp(fun, t_span, y0, **kwargs)
        ends.append((sol.t[-1], sol.y[:, -1]))
        return sol
    monkeypatch.setattr(_ibc.integrate, 'solve_ivp', recording_solve_ivp)

    h.call_moving(start, end, start_mesh, end_mesh, 'sliding')

    assert ends, 'the slider made no solve_ivp calls; this test needs rewriting'
    s_min = np.inf
    for t, x in ends:
        A, B = at(Q0, Q0_end, t), at(Q1, Q1_end, t)
        s_min = min(s_min, np.dot(x - A, B - A)/np.dot(B - A, B - A))
        off_line = abs(h.signed_perp_dist_2D(x, A, B))
        assert off_line < 1e-6, \
            f'slide path is {off_line:.1e} off the element line at t={t:.4f}'
    assert s_min < -0.1, 'the recorded path never went well past Q0'


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
#        the agent's motion along a turning element reversing mid-step        #
# --------------------------------------------------------------------------- #
# Along the element, a sliding agent moves relative to each of its ends, and
# the slider samples the step where that relative motion reverses. A reversal
# needs the element to turn through the perpendicular to the agent's motion
# relative to that end -- something no translating or stretching wall does.

def test_slide_reversing_along_a_turning_element_obeys_the_invariants():
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
#             running off an element's end partway through a step             #
# --------------------------------------------------------------------------- #
# Each case is one moving element, where the agent runs off an end during the
# step; the answer is h.moving_slide_model's slide-then-fly end point, from an
# independent quadrature. Two kinds of defect are pinned:
#   - transient run-offs, where the agent is back over the element by the end
#     of the step. The slider kept the agent on the element: it sampled the
#     step at the element's shortest moment and where vec.Qvec = 0, which are
#     not where the agent's motion relative to Q0 or Q1 reverses, and only the
#     earlier of the two (issue #73).
#   - run-offs after the agent's motion relative to the crossed end reverses.
#     The run-off time solve was a least-squares minimization started at the
#     contact time, which stopped there at once; the agent was moved to the end
#     and re-contacted it, recursing until it ran out of stack.
#   - a step that runs off both ends, where the agent must leave by the end it
#     reaches first. The slider chose the end farther from its end-of-step
#     point, here the later one.
# A contact exactly at a vertex, with the agent already moving off it, is also
# pinned: the run-off time is then the contact time itself, with no root to
# solve for. Its coordinates are dyadic and the element is horizontal at the
# contact (t = 1/2, at Q1 = (5,5)), so every quantity is exact. A contact found
# at random lands within roundoff of the vertex, and which side it falls on
# depends on the BLAS: one such case hit the element on Windows (MKL) and
# missed it on Linux (OpenBLAS), failing CI.

RUN_OFF_CASES = {
    'transient, both old sample times wrong': (
        [5.48185439396559, 5.001240562879473],
        [5.598187591919126, 5.610750737676608],
        [[[5.0398356934162765, 5.058643828567173],
          [5.415451946499984, 5.417722055189501]]],
        [[[5.117062128881987, 4.842300607273068],
          [5.817834779007834, 4.6502040287223805]]]),
    'transient, only the later old sample time catches it': (
        [4.463249093224856, 5.0744359179837115],
        [1.542662452452522, 5.130766085533219],
        [[[4.539208075023466, 3.9443111210621273],
          [4.329561298306154, 5.315113757418108]]],
        [[[4.024687079163556, 2.9357957068718967],
          [5.4204924717743825, 4.119646023622436]]]),
    'reverses then runs off Q1, rigid 3.1 degree turn': (
        [3.4252301424372082, 4.622599411711805],
        [4.0983468559399725, 3.947551458443008],
        [[[4.685755540150286, 5.410270389837448],
          [3.4906651282976564, 4.5212653118299375]]],
        [[[4.82376811322203, 5.432219103512595],
          [3.678130526138604, 4.480333795144568]]]),
    'reverses then runs off Q0, 65 degree turn': (
        [4.688406805068963, 4.995292070648784],
        [2.6055203765254804, 4.968031204470195],
        [[[4.530374351297982, 4.968744788511176],
          [4.074155801961116, 6.264388876165749]]],
        [[[3.325910163742835, 4.865267316598745],
          [3.8512627443514815, 5.387254206990081]]]),
    'off both ends, Q1 first': (
        [4.826423222611346, 4.385226862581428],
        [4.794782640408741, 6.21269981988177],
        [[[4.053266714317672, 4.414190883687158],
          [5.097908469741366, 5.202621738878933]]],
        [[[4.708819406616773, 5.116342195930927],
          [4.394707548386113, 4.127231539847764]]]),
    'contact exactly at Q1, already moving off it': (
        [4.75, 5.5],
        [5.25, 4.5],
        [[[4.0, 5.25],
          [5.0, 4.75]]],
        [[[4.0, 4.75],
          [5.0, 5.25]]]),
}


@pytest.mark.parametrize('case', RUN_OFF_CASES)
def test_run_off_during_the_step_matches_the_slide_model(case):
    start, end, start_mesh, end_mesh = map(np.array, RUN_OFF_CASES[case])
    model = h.moving_slide_model(start, end, start_mesh, end_mesh)
    assert model['released'] and model['why'] == 'end', \
        'the case no longer runs off an end; it has stopped testing anything'

    newend, dx, idx = h.call_moving(start, end, start_mesh, end_mesh)

    assert idx == 0
    assert np.allclose(newend, model['final'], rtol=0, atol=1e-5), \
        f"slider ended at {newend}, model at {model['final']}"


def test_agent_left_on_a_stretching_element_is_backed_off_exactly_EPS():
    '''An agent that slides on a moving element all step ends EPS from it, on
    its own side. The integrated slide path drifted off the element's line by
    up to a quarter of EPS here (an element stretching twelvefold), eating into
    the back-off that keeps the agent off the solid side.'''
    start = np.array([4.42632755038969, 6.010368857103861])
    end = np.array([6.1153154103556595, 5.895001458201781])
    m0 = np.array([[[4.36605649895233, 5.944859928643714],
                    [4.997315499352104, 6.000727463412911]]])
    m1 = np.array([[[-1.2395922384918858, 5.202583210251121],
                    [6.467036333109455, 6.5411157582818245]]])
    assert not h.moving_slide_model(start, end, m0, m1)['released'], \
        'the agent no longer stays on the element; the case tests nothing'
    eps = _ibc._boundary_eps(start, end, m0, m1)

    newend, dx, idx = h.call_moving(start, end, m0, m1)

    # start is above the element (positive side) at contact
    backoff = h.signed_perp_dist_2D(newend, m1[0, 0], m1[0, 1])
    assert np.isclose(backoff, eps, rtol=1e-6, atol=0), \
        f'agent ended {backoff/eps:.6f} EPS from the element, not 1'


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
#        joints on a moving mesh: continuation and back-off                   #
# --------------------------------------------------------------------------- #
# A frictionless agent moves along a boundary relative to it, so for a mesh
# translating rigidly by v over a step
#     moving(P0 -> P1; mesh, mesh + v) == static(P0 -> P1 - v; mesh) + v.
# The slider decided whether the agent continues past a joint from its lab-frame
# movement. Where lab and relative motion along the next element disagreed, it
# stopped the agent at the joint, and for collinear elements left it exactly on
# the element line, from where a later step passed straight through.

def _galilean_pair(P0, P1, mesh, v, ib='sliding'):
    '''(moving result, static result + v, EPS) for one step.'''
    mmd = h.max_meshpt_dist(mesh)
    mov = _ibc.apply_internal_moving_BC(P0, P1, mesh, mesh + v, mmd,
                                        np.linalg.norm(v), ib_collisions=ib)
    sta = _ibc.apply_internal_static_BC(P0, P1 - v, mesh, mmd,
                                        ib_collisions=ib)
    eps = _ibc._boundary_eps(P0, P1, mesh, mesh + v)
    return mov, sta, eps


def test_continuing_past_a_moving_joint_is_judged_relative_to_the_joint():
    '''A wall of collinear elements sliding along itself faster than the agent:
    in the lab the agent moves forward along the wall, relative to it backward.
    It must continue past the joints as the static slider does in the wall's
    frame, and end EPS below the wall.'''
    wall = h.horizontal_wall(10, 5.0, 1.0, 9.0)
    v = np.array([0.9812818424145259, 0.11706805944363517])
    P0 = np.array([6.257728457107769, 4.327690434163696])
    P1 = np.array([6.587874175080389, 5.513467032286361])
    assert (P1 - P0)[0] > 0 > (P1 - P0 - v)[0], \
        'lab and relative motion along the wall no longer disagree'

    mov, sta, eps = _galilean_pair(P0, P1, wall, v)

    assert mov[2] is not None
    assert np.linalg.norm(mov[0] - (sta[0] + v)) <= 2*eps, \
        f'moving {mov[0]} vs static + v {sta[0] + v}'
    assert np.isclose(mov[0][1] - (5.0 + v[1]), -eps, rtol=1e-6, atol=0)


_c = np.array([5.0, 5.0])
GALILEAN_GEOMS = {
    'tilted segment': h.segment((3.6, 4.1), (6.3, 5.8)),
    'axis-aligned walls': np.concatenate([h.horizontal_wall(6, 5.0, 2.0, 8.0),
                                          h.wall_segments(6, 3.0, 5.0, 8.0)]),
    'convex L': h.polyline([(3, 7), (3, 4), (7, 4)]),
    'concave V': h.polyline([(3, 7), (5, 4), (7, 7)]),
    'acute wedge': h.polyline([(2.5, 5.5), (6.5, 5.0), (2.5, 4.3)]),
    'octagon': h.closed_polygon(
        [_c + 1.5*np.array([np.cos(a), np.sin(a)])
         for a in np.arange(8)*np.pi/4 + 0.2]),
    'polyline arc': h.polyline(
        [_c + 2*np.array([np.cos(a), np.sin(a)])
         for a in np.linspace(-0.3, np.pi + 0.3, 13)]),
    '5-spoke star': h.star_2D(
        _c, [_c + 2*np.array([np.cos(a), np.sin(a)])
             for a in np.arange(5)*2*np.pi/5 + 0.1]),
}


@pytest.mark.parametrize('ib', ['sliding', 'sticky'])
@pytest.mark.parametrize('geom', GALILEAN_GEOMS)
def test_rigidly_translating_mesh_matches_the_static_answer(geom, ib):
    '''Random steps against a mesh translating by up to the step length give the
    static answer in the mesh's frame, to the back-off: 2 EPS, since the sticky
    back-offs point in different directions (along the normal when moving,
    back along the path when static).'''
    mesh = GALILEAN_GEOMS[geom]
    rng = np.random.default_rng(list(GALILEAN_GEOMS).index(geom))
    contacts = 0
    for trial in range(400):
        P0 = _c + rng.uniform(-2.5, 2.5, 2)
        w = rng.uniform(-1.2, 1.2, 2)
        v = rng.uniform(-1, 1, 2)*np.linalg.norm(w)
        mov, sta, eps = _galilean_pair(P0, P0 + w, mesh, v, ib)
        if sta[2] is None:
            continue
        contacts += 1
        assert np.linalg.norm(mov[0] - (sta[0] + v)) <= 2*eps, \
            (f'trial {trial}: moving {mov[0]} vs static + v {sta[0] + v} '
             f'({np.linalg.norm(mov[0] - (sta[0] + v))/eps:.3g} EPS)')
    assert contacts > 25


def test_agents_pushed_into_a_wall_sliding_along_itself_never_cross():
    '''Agents drift up into a wall of collinear elements sliding along itself
    faster than they do. None may ever end above the wall.'''
    import planktos

    class Drift(planktos.Swarm):
        def apply_agent_model(self, dt):
            return self.positions + self.drift*dt

    wall = h.horizontal_wall(10, 5.0, 1.0, 9.0)
    V, U, dt, T, N = np.array([1.0, 0.0]), np.array([0.2, 0.5]), 0.1, 6.0, 40
    envir = planktos.Environment(Lx=30, Ly=30, x_bndry='noflux',
                                 y_bndry='noflux',
                                 flow=[np.zeros((5, 5)), np.zeros((5, 5))])
    envir.ibmesh = np.stack([wall, wall + V*T])
    envir.ibmesh_times = np.array([0.0, T])
    envir.max_meshpt_dist = 0.8
    rng = np.random.default_rng(1)
    init = np.c_[rng.uniform(2.0, 8.0, N), rng.uniform(4.55, 4.8, N)]
    s = Drift(swarm_size=N, envir=envir, init=init, seed=1)
    s.drift = np.tile(U, (N, 1))
    for _ in range(int(round(T/dt))):
        s.move(dt, silent=True)

    traj = np.stack([np.ma.filled(p, np.nan) for p in s.full_pos_history])
    t = dt*np.arange(traj.shape[0])[:, None]
    over = (traj[:, :, 0] > 1.0 + V[0]*t) & (traj[:, :, 0] < 9.0 + V[0]*t)
    crossed = (over & (traj[:, :, 1] >= 5.0)).any(axis=0)
    assert not crossed.any(), \
        f'{crossed.sum()} of {N} agents were on or above the wall at some step'


def test_rotating_segment_split_into_collinear_elements_matches_one_element():
    '''A straight segment pivoting about a point off its line, as one element and
    as four collinear ones, carries a sliding agent across a joint to the same
    place.'''
    import planktos

    class Drift(planktos.Swarm):
        def apply_agent_model(self, dt):
            return self.positions + self.drift*dt

    e = lambda th: np.array([np.cos(th), np.sin(th)])
    n = lambda th: np.array([-np.sin(th), np.cos(th)])
    C, h_off, half, omega, th0, T, dt = _c, 1.0, 3.0, 0.5, 0.3, 2.0, 0.05
    times = np.arange(0, T + dt/2, dt)

    def run(M):
        a = np.linspace(-half, half, M + 1)
        frames = []
        for t in times:
            th = th0 + omega*t
            pts = C + h_off*n(th) + a[:, None]*e(th)
            frames.append(np.stack([pts[:-1], pts[1:]], axis=1))
        envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux',
                                     y_bndry='noflux',
                                     flow=[np.zeros((5, 5)), np.zeros((5, 5))])
        envir.ibmesh = np.array(frames)
        envir.ibmesh_times = times
        envir.max_meshpt_dist = 2*half/M + 1e-9
        init = C + h_off*n(th0) + 1.0*e(th0) + 0.05*n(th0)
        s = Drift(swarm_size=1, envir=envir, init=init[None, :], seed=1)
        s.drift = (-0.3*n(th0 + 0.1))[None, :]
        for _ in range(int(round(T/dt))):
            s.move(dt, silent=True)
        return np.asarray(s.positions[0])

    one, four = run(1), run(4)
    th = th0 + omega*T
    along = np.dot(one - (C + h_off*n(th)), e(th))
    assert along > 1.5, 'the agent no longer crosses the joint at 1.5'
    assert np.linalg.norm(one - four) <= 2e-6, f'one element {one}, four {four}'


@pytest.mark.parametrize('motion', ['stationary', 'translating'])
def test_stay_put_at_a_nearly_collinear_moving_joint_backs_off_from_both(motion):
    '''An agent stopped at a joint turning 0.5 degrees toward it ends EPS from
    both elements' lines. Backing off along the bisector by EPS left it about
    0.004 EPS from each.'''
    V = np.array([5.0, 5.0])
    ang = np.deg2rad(0.5)
    mesh = np.concatenate([h.segment((4.0, 5.0), tuple(V)),
                           h.segment(tuple(V), tuple(V + [np.cos(ang),
                                                          np.sin(ang)]))])
    shift = np.array([0.0, 0.01]) if motion == 'translating' else np.zeros(2)
    start, end = np.array([4.994, 5.5]), np.array([5.002, 4.5])
    eps = _ibc._boundary_eps(start, end, mesh, mesh + shift)

    newend, dx, idx = h.call_moving(start, end, mesh, mesh + shift)

    m1 = mesh + shift
    assert np.linalg.norm(newend - (V + shift)) < POS_ATOL, 'not left at the joint'
    for k in (0, 1):
        d = h.signed_perp_dist_2D(newend, m1[k, 0], m1[k, 1])
        assert d >= eps*(1 - 1e-6), f'{d/eps:.4f} EPS from element {k}'


@pytest.mark.parametrize('over', [5e-7, -5e-7])
@pytest.mark.parametrize('motion', ['stationary', 'translating'])
def test_slide_ending_within_EPS_of_a_moving_concave_joint_clears_both(motion,
                                                                     over):
    '''A slide ending within EPS of a vertex where a wall rises (past it, or
    short of it) ends EPS from both. It was backed off from the element alone,
    up to EPS on the wrong side of the wall.'''
    mesh = np.array([[[4., 5.], [5., 5.]], [[5., 5.], [5., 6.]]])
    shift = np.array([0.0, 1e-3]) if motion == 'translating' else np.zeros(2)
    start, end = np.array([4.5, 5.5]), np.array([5 + over, 4.5])
    eps = _ibc._boundary_eps(start, end, mesh, mesh + shift)

    newend, dx, idx = h.call_moving(start, end, mesh, mesh + shift)

    m1 = mesh + shift
    for k in (0, 1):
        d = h.signed_perp_dist_2D(newend, m1[k, 0], m1[k, 1])
        assert d >= eps*(1 - 1e-6), f'{d/eps:.4f} EPS from element {k}'


def test_sticky_hit_just_short_of_an_acute_concave_joint_clears_both():
    '''A sticky hit 0.5 EPS short of a 45 degree concave joint ends EPS from both
    elements. Backed off normal to the struck element alone, it ended 0.35 EPS
    on the wrong side of the other.'''
    mesh = np.array([[[4., 5.], [5., 5.]], [[5., 5.], [4., 6.]]])
    start = np.array([4.0, 5.5])
    end = np.array([4.9999995, 5.0]) + 0.2*np.array([0.9999995, -0.5])
    eps = _ibc._boundary_eps(start, end, mesh, mesh)

    newend, dx, idx = h.call_moving(start, end, mesh, mesh, 'sticky')

    assert idx == 0
    for k in (0, 1):
        d = h.signed_perp_dist_2D(newend, mesh[k, 0], mesh[k, 1])
        assert d >= eps*(1 - 1e-6), f'{d/eps:.4f} EPS from element {k}'


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
# _project_and_slide_moving checks for a run-off during the step at critical
# times computed from the element's vectors at contact and at the end of the
# step. At the time of this defect the times were where the element's
# interpolated length is stationary and where vec.Qvec = 0.
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
