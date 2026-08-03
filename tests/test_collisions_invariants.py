'''Collision properties that hold for any geometry at all.

Everything in the other collision modules is tied to a particular shape: a wall,
a dihedral, a joint of degree four, a pivoting element. This module holds the
checks that are not -- the ones a correct collision must satisfy whatever it hit.

Two kinds live here:

  * **Invariants of the answer.** The result must be a real point, must not
    travel further than the agent tried to, must not depend on where the problem
    sits or which way the axes point, and must behave the same expressed in any
    units. These constrain the arithmetic rather than the geometry, so a failure
    is attributable to the property and not to an exotic mesh -- which is why the
    geometries used are deliberately the plain ones.

  * **What happens when a slide outruns the stack.** Recursion depth is set by
    the step length against the mesh spacing, so it is a property of the ratio
    and not of any particular shape. Nothing caps it; the requirement is that
    running out explains itself rather than surfacing as a bare RecursionError
    from wherever in the geometry code the stack gave way.

Rotation equivariance on the *junction* geometries stays in
test_collisions_junctions.py, where the geometry it needs is defined and where
the bug it guards was found.
'''

import sys

import numpy as np
import pytest

import _ib_harness as h


# --------------------------------------------------------------------------- #
#                 invariants of the resolved answer                           #
# --------------------------------------------------------------------------- #

def _vwall():
    return h.wall_segments(4, 5.0, y_lo=0.0, y_hi=10.0)


def _lcorner():
    return h.polyline([(2.0, 5.0), (5.0, 5.0), (5.0, 9.0)])


def _vgroove():
    return h.polyline([(2.0, 8.0), (5.0, 5.0), (8.0, 8.0)])


# name -> (mesh builder, start, end). Chain-shaped on purpose: see the module
# docstring. The junction geometries get the same checks, over a wider range of
# k and in both collision modes, in test_collisions_junctions.py.
GEOMETRIES = {
    'vertical wall':   (_vwall,   (4.0, 5.0), (6.0, 6.0)),
    'convex L corner': (_lcorner, (3.0, 3.0), (4.5, 6.5)),
    'concave V groove': (_vgroove, (5.0, 8.0), (5.0, 5.5)),
}


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_result_is_finite(name):
    build, start, end = GEOMETRIES[name]
    newend, _, _ = h.call_static(start, end, build())
    h.assert_finite(newend, name)


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_motion_is_never_amplified(name):
    build, start, end = GEOMETRIES[name]
    newend, _, _ = h.call_static(start, end, build())
    h.assert_displacement_bounded(start, end, newend)


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
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


# --------------------------------------------------------------------------- #
#                       running out of stack while sliding                    #
# --------------------------------------------------------------------------- #
# A sliding agent recurses once per element it crosses, so the depth is set by
# how far the step carries it along the boundary relative to the mesh spacing.
# Each transfer consumes a whole element, so this is bounded -- but by the step
# length over the element length, which an over-large dt pushes arbitrarily
# high. Nothing here caps the depth; the point is that when the stack does run
# out, the failure explains itself instead of surfacing as a bare RecursionError
# from wherever in the geometry code the stack happened to give way.
#
# For scale, an agent grazing the entire length of the 2700-element IB2d channel
# wall in one step recurses about 300 times, comfortably inside the default
# limit.

def _fine_wall(n_elements, x=5.0, y_hi=10.0):
    return h.wall_segments(n_elements, x, y_lo=0.0, y_hi=y_hi)


def test_ordinary_deep_slide_still_works():
    '''A long slide across many elements is legitimate and must not be limited
    by anything other than the interpreter's own stack.'''
    mesh = _fine_wall(400)
    start = np.array([4.9, 1.0])
    end = np.array([5.3, 7.0])              # ~240 elements
    newend, _, _ = h.call_static(start, end, mesh)
    h.assert_finite(newend, 'deep but legitimate slide')
    h.assert_not_penetrated_2D(start, newend, mesh[0, 0], mesh[0, 1])


def test_exhausting_the_stack_reports_the_cause():
    '''The re-raise names what ran out, why, and what to change -- and keeps the
    original RecursionError as the cause so the traceback still shows where.'''
    mesh = _fine_wall(4000)
    start = np.array([4.9, 0.5])
    end = np.array([5.3, 9.5])              # far more elements than the stack

    with pytest.raises(RuntimeError, match='sliding an agent') as excinfo:
        h.call_static(start, end, mesh)

    msg = str(excinfo.value)
    assert 'dt' in msg, f'should say what to change; got: {msg}'
    assert 'setrecursionlimit' in msg, f'should offer the escape hatch: {msg}'
    assert isinstance(excinfo.value.__cause__, RecursionError), \
        'the original RecursionError should be chained as the cause'


def test_stack_exhaustion_message_quantifies_the_mismatch():
    '''The useful number is the step length against the mesh spacing, since
    their ratio is what sets the recursion depth.'''
    mesh = _fine_wall(4000, y_hi=10.0)      # elements of 0.0025
    start = np.array([4.9, 0.5])
    end = np.array([5.3, 9.5])

    with pytest.raises(RuntimeError) as excinfo:
        h.call_static(start, end, mesh)
    msg = str(excinfo.value)
    assert '0.0025' in msg, f'should report the element scale; got: {msg}'
    assert 'elements in one step' in msg


def _stack_depth():
    '''Frames currently on the stack. Cheaper than inspect.stack(), which builds
    a full record per frame.'''
    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def test_moving_slide_exhausting_the_stack_reports_the_cause():
    '''The moving slider has its own re-raise, and needs its own case: a slide
    along a deforming mesh recurses through a different routine than the static
    one, so covering one says nothing about the other.

    The interpreter's real stack is not exhausted here. Every element the moving
    slider crosses costs an ODE solve, so running it out naturally takes seconds;
    the limit is lowered relative to the depth already in use instead. What is
    under test is the re-raise, not how many frames the interpreter starts with,
    and pinning it to a relative depth keeps the test independent of the platform
    stack size.
    '''
    # ~270 elements across the slide, comfortably more than the depth allowed
    start_mesh = h.wall_segments(300, 5.0, y_lo=0.0, y_hi=1.0)
    end_mesh = h.wall_segments(300, 5.02, y_lo=0.0, y_hi=1.0)
    start = np.array([4.995, 0.05])
    end = np.array([5.1, 0.95])

    original = sys.getrecursionlimit()
    sys.setrecursionlimit(_stack_depth() + 100)
    try:
        h.call_moving(start, end, start_mesh, end_mesh)
        raised = None
    except RuntimeError as err:
        # RecursionError subclasses RuntimeError, so this catches both the
        # explained failure and a bare one leaking through; they are told apart
        # below rather than by the except clause.
        raised = err
    finally:
        sys.setrecursionlimit(original)

    assert raised is not None, 'the slide should have run out of stack'
    assert not isinstance(raised, RecursionError), \
        f'bare RecursionError escaped instead of being explained: {raised}'
    msg = str(raised)
    assert 'sliding an agent' in msg, f'should name what ran out; got: {msg}'
    assert 'dt' in msg, f'should say what to change; got: {msg}'
    assert 'setrecursionlimit' in msg, f'should offer the escape hatch: {msg}'
    assert isinstance(raised.__cause__, RecursionError), \
        'the original RecursionError should be chained as the cause'
