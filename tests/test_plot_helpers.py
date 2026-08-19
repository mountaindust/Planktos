'''Plotting helpers that decide what a frame looks like without drawing one.

_vorticity_norm sets the colour limits of the RdBu vorticity backdrop, and
_calc_basic_stats produces the numbers printed in the corner of a frame. Both
are pure computation, so unlike the rendering smokes in test_plotting_smoke.py
these stay in the fast run.

(On the dyload branch the _vorticity_norm section lives in test_frame_selection.py
alongside the frame-selection arithmetic, which master does not have, and the
_calc_basic_stats section lives in test_flow_interface.py, which master also does
not have. If the branches are ever merged, fold this file into those rather than
keeping both.)
'''

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import planktos
from planktos._swarm import _vorticity_norm


# --------------------------------------------------------------------------- #
#            vorticity colour limits (the RdBu backdrop of plot_all)          #
# --------------------------------------------------------------------------- #
# RdBu is diverging and white at its midpoint, so limits that are not symmetric
# about zero tint the whole quiescent background. plot_all used to call
# ScalarMappable.autoscale() on every frame, which sets the limits to that
# frame's own min/max: asymmetric, and different every frame. The result was a
# background flashing light red and blue through a video -- and any clip the
# caller passed was silently discarded, since autoscale overwrites it.


def _zero_at_white(norm):
    '''RdBu puts white at the midpoint of the normalized range.'''
    return np.isclose(norm(0.0), 0.5)


def test_vorticity_clip_is_honored_exactly():
    norm = _vorticity_norm(np.array([[0.1, 0.2], [0.3, 0.4]]), clip=2.)
    assert (norm.vmin, norm.vmax) == (-2., 2.)
    assert norm.clip is True
    assert _zero_at_white(norm)


def test_vorticity_clip_is_not_rescaled_by_later_frames():
    # The old autoscale() call replaced the caller's clip with the frame's own
    # extremes, so passing clip did not actually stabilize a movie.
    norm = _vorticity_norm(np.zeros((2, 2)), clip=2.)
    for frame in (np.full((2, 2), 0.01), np.full((2, 2), 50.)):
        norm = _vorticity_norm(frame, clip=2., norm=norm)
        assert (norm.vmin, norm.vmax) == (-2., 2.)


def test_vorticity_limits_are_symmetric_so_zero_is_white():
    # A field that is strongly one-signed is the case that tinted the
    # background: min/max limits put zero far from the colormap midpoint.
    vort = np.array([[0.1, 0.2], [0.3, 5.0]])
    norm = _vorticity_norm(vort)
    assert np.isclose(norm.vmax, 5.0) and np.isclose(norm.vmin, -5.0)
    assert _zero_at_white(norm)


def test_vorticity_limits_grow_but_never_shrink():
    # The flashing fix. A quiet frame following a strong one must not pull the
    # scale back in, or everything drawn against it changes colour.
    norm = _vorticity_norm(np.array([[1.0]]))
    assert np.isclose(norm.vmax, 1.0)
    norm = _vorticity_norm(np.array([[4.0]]), norm=norm)      # grows
    assert np.isclose(norm.vmax, 4.0)
    norm = _vorticity_norm(np.array([[0.001]]), norm=norm)    # must not shrink
    assert np.isclose(norm.vmax, 4.0)
    assert _zero_at_white(norm)


def test_vorticity_all_zero_field_reads_as_white_not_saturated():
    # Limits of (0, 0) would send every cell to the bottom of the colormap, so a
    # motionless fluid would render solid blue rather than blank.
    norm = _vorticity_norm(np.zeros((3, 3)))
    assert norm.vmin < 0 < norm.vmax
    assert _zero_at_white(norm)


def test_vorticity_limits_ignore_non_finite_values():
    vort = np.array([[1.0, np.nan], [np.inf, 2.0]])
    norm = _vorticity_norm(vort)
    assert np.isclose(norm.vmax, 2.0)
    assert _zero_at_white(norm)


def test_vorticity_norm_is_grown_in_place():
    # plot_all hands the live norm back in each frame; replacing the object
    # instead of growing it would drop matplotlib's callbacks on the mappable.
    norm = _vorticity_norm(np.array([[1.0]]))
    same = _vorticity_norm(np.array([[9.0]]), norm=norm)
    assert same is norm and np.isclose(norm.vmax, 9.0)


# --------------------------------------------------------------------------- #
#          agent velocity statistics (the text printed on a frame)            #
# --------------------------------------------------------------------------- #
# _calc_basic_stats used to build the agent velocity by differencing consecutive
# pos_history entries. That is not the velocity the agents had: move() sets
# velocities from PRE-boundary-condition positions and apply_boundary_conditions
# then mutates positions, so the two part company for any agent that hit an
# immersed boundary or the domain edge -- and across a periodic wrap the
# difference of positions is nearly the whole domain. vel_history records the
# real thing and is what is read now. avg_swrm_vel is the last element of the
# returned tuple in every branch, so these index it from the end.


def _linear_2d(nx=11, ny=9, Lx=10.0, Ly=8.0):
    '''Static 2D environment with u = x, v = 2y (exactly linear in space).'''
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return planktos.Environment(Lx=Lx, Ly=Ly, flow=[X.copy(), 2 * Y.copy()],
                                x_bndry=('zero', 'zero'),
                                y_bndry=('zero', 'zero'))


def test_calc_basic_stats_agent_velocity_at_initial_time_is_the_recorded_drift():
    # t_indx=0 reports the velocity the agents actually had. Swarm.__init__ sets
    # that to the local fluid drift, so it is generally NOT zero -- an earlier
    # version reported the zero vector here on the grounds that velocity is
    # undefined before the first step. The recorded value is the truth.
    envir = _linear_2d()                                # u = x, v = 2y
    swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=1)
    # place the agents by hand so the drift is closed-form. The field is exactly
    # linear, so linear interpolation reproduces it exactly.
    pts = np.array([[1.0, 1.0], [2.0, 3.0], [7.0, 2.0], [4.0, 6.0]])
    swrm.positions[:, :] = pts
    swrm.velocities[:, :] = swrm.get_fluid_drift()
    swrm.move(0.1)

    drift = np.column_stack((pts[:, 0], 2 * pts[:, 1]))
    avg_swrm_vel = swrm._calc_basic_stats(DIM3=False, t_indx=0)[-1]
    assert np.allclose(avg_swrm_vel, drift.mean(axis=0))
    # emphatically not the zero the old convention reported
    assert np.linalg.norm(avg_swrm_vel) > 1.0


def test_calc_basic_stats_agent_velocity_at_initial_time_is_zero_without_flow():
    # With no fluid there is no drift to inherit, so Swarm.__init__ leaves the
    # initial velocities at zero and the t_indx=0 statistics are zero -- the same
    # numbers the retired convention produced, now for a reason that is true.
    envir = planktos.Environment(Lx=10, Ly=10)
    swrm = planktos.Swarm(swarm_size=6, envir=envir, seed=1)
    swrm.move(0.1)
    assert np.allclose(swrm._calc_basic_stats(DIM3=False, t_indx=0)[-1], 0.0)


def test_calc_basic_stats_velocity_survives_a_periodic_wrap():
    # An agent that wraps has a position difference of nearly the whole domain,
    # which as a velocity is enormous and fictitious; the recorded velocity is
    # the real one. This is the sharpest case of a defect that reaches every
    # agent that collides with anything.
    class _Rightward(planktos.Swarm):
        def apply_agent_model(self, dt):
            return self.positions + np.array([1.0, 0.0]) * dt

    envir = planktos.Environment(Lx=10, Ly=10, x_bndry=('periodic', 'periodic'))
    swrm = _Rightward(swarm_size=1, envir=envir, seed=1)
    swrm.positions[:, :] = [9.5, 5.0]
    swrm.velocities[:, :] = 0.0

    swrm.move(1.0)                      # 9.5 -> 10.5, wrapped back to 0.5
    assert swrm.positions[0, 0] < 1.0, 'the agent did not actually wrap'
    swrm.move(1.0)                      # 0.5 -> 1.5, no wrap

    # index 1 is the state just after the wrap, so the position difference
    # spanning it is -9: the value the old derivation would have reported.
    avg_swrm_vel = swrm._calc_basic_stats(DIM3=False, t_indx=1)[-1]
    assert np.allclose(avg_swrm_vel, [1.0, 0.0])
