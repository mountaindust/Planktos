'''Tests for the Swarm.move() lifecycle: history bookkeeping, the masked-array
contract for agents that leave the domain, the domain boundary conditions
(zero / noflux / periodic), property storage, and multi-swarm advancement.

Everything is driven by a constant drift with zero diffusion (cov = 0) so each
outcome is an exact, deterministic known answer -- no flow interpolation, no RNG.
'''

import numpy as np
import numpy.ma as ma
import pytest

import planktos


def _envir(bc, L=1.0):
    '''Square domain with the same boundary condition on all four sides, no flow.'''
    return planktos.Environment(Lx=L, Ly=L, x_bndry=(bc, bc), y_bndry=(bc, bc),
                                flow=[np.zeros((3, 3)), np.zeros((3, 3))])


def _drift_swarm(envir, init, mu=(2.0, 0.0), seed=1):
    swrm = planktos.Swarm(swarm_size=len(init), envir=envir,
                          init=np.asarray(init, float), seed=seed)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array(mu, float)
    return swrm


class _PerAgentDriftSwarm(planktos.Swarm):
    '''Each agent advances by its own fixed velocity (self._drift) each step,
    ignoring flow and diffusion. Lets one move() send different agents toward
    different faces so a single test can check every dimension at once.'''
    def apply_agent_model(self, dt):
        return self.positions + self._drift * dt


def _pa_swarm(envir, init, drift):
    swrm = _PerAgentDriftSwarm(swarm_size=len(init), envir=envir,
                               init=np.asarray(init, float), seed=1)
    swrm._drift = np.asarray(drift, float)
    return swrm


# --------------------------------------------------------------------------- #
#                          history bookkeeping                                #
# --------------------------------------------------------------------------- #

def test_move_records_history_and_time():
    envir = planktos.Environment(Lx=100, Ly=100, flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    swrm = _drift_swarm(envir, [[50., 50.]] * 3)
    for _ in range(5):
        swrm.move(0.2)
    assert len(swrm.pos_history) == 5
    assert len(swrm.full_pos_history) == 6           # history + current
    assert envir.time == pytest.approx(1.0)
    assert len(envir.time_history) == 5
    assert envir.time_history[0] == 0.0


# --------------------------------------------------------------------------- #
#                          domain boundary conditions                         #
# --------------------------------------------------------------------------- #

def test_zero_bc_masks_leavers_in_all_dims():
    # Agent 0 starts near the +x wall and is pushed out; agent 1 stays inside.
    envir = _envir('zero')
    swrm = _drift_swarm(envir, [[0.95, 0.5], [0.1, 0.5]])
    swrm.move(0.1)                                    # +0.2 in x: agent 0 -> 1.15 (out)
    assert np.all(swrm.positions.mask[0]), "leaver not fully masked"
    assert not np.any(swrm.positions.mask[1]), "interior agent wrongly masked"


def test_masked_agent_stays_masked():
    envir = _envir('zero')
    swrm = _drift_swarm(envir, [[0.95, 0.5], [0.5, 0.5]])
    swrm.move(0.1)
    assert np.all(swrm.positions.mask[0])
    swrm.shared_props['mu'] = np.array([-2.0, 0.0])   # reverse drift
    swrm.move(0.1)                                     # a live agent would re-enter
    assert np.all(swrm.positions.mask[0]), "masked agent re-entered the domain"


def test_noflux_bc_clips_to_boundary():
    envir = _envir('noflux')
    swrm = _drift_swarm(envir, [[0.9, 0.5]])
    swrm.move(0.1)                                     # 0.9 + 0.2 = 1.1 -> clipped to 1.0
    assert np.isclose(float(swrm.positions[0, 0]), 1.0)
    assert not np.any(swrm.positions.mask), "noflux agent was masked"


def test_periodic_bc_wraps():
    envir = _envir('periodic')
    swrm = _drift_swarm(envir, [[0.9, 0.5]])
    swrm.move(0.1)                                     # 1.1 -> wraps to 0.1
    assert np.isclose(float(swrm.positions[0, 0]), 0.1)
    assert not np.any(swrm.positions.mask)


# --------------------------------------------------------------------------- #
#                 3D and mixed-per-dimension boundary conditions              #
# --------------------------------------------------------------------------- #
# The cases above are 2D with the same condition on all sides. These cover all
# three conditions in 3D and mixed-per-dimension combinations -- previously only
# exercised indirectly via the IBAMR loader test. Each move sends one agent
# toward each face so every dimension is checked at once.

def _envir3d(bc, L=1.0):
    '''Cubic domain with the same condition on all six faces, no flow.'''
    return planktos.Environment(Lx=L, Ly=L, Lz=L, x_bndry=(bc, bc),
                                y_bndry=(bc, bc), z_bndry=(bc, bc),
                                flow=[np.zeros((3, 3, 3)) for _ in range(3)])


def test_3d_zero_bc_masks_leavers_in_each_dim():
    envir = _envir3d('zero')
    swrm = _pa_swarm(envir,
                     [[0.9, 0.5, 0.5], [0.5, 0.9, 0.5], [0.5, 0.5, 0.9], [0.5, 0.5, 0.5]],
                     [[2, 0, 0], [0, 2, 0], [0, 0, 2], [0, 0, 0]])
    swrm.move(0.1)                                     # each leaver overshoots its face
    mask = ma.getmaskarray(swrm.positions)
    assert np.all(mask[:3]), "a leaver (in x, y, or z) was not masked"
    assert not np.any(mask[3]), "interior agent wrongly masked"


def test_3d_noflux_bc_clips_in_each_dim():
    envir = _envir3d('noflux')
    swrm = _pa_swarm(envir,
                     [[0.9, 0.5, 0.5], [0.5, 0.9, 0.5], [0.5, 0.5, 0.9]],
                     [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    swrm.move(0.1)
    pos = np.asarray(swrm.positions)
    assert np.isclose(pos[0, 0], 1.0)                  # clipped to the +x face
    assert np.isclose(pos[1, 1], 1.0)                  # clipped to the +y face
    assert np.isclose(pos[2, 2], 1.0)                  # clipped to the +z face
    assert not np.any(ma.getmaskarray(swrm.positions)), "noflux agent was masked"


def test_3d_periodic_bc_wraps_in_each_dim():
    envir = _envir3d('periodic')
    # x, y exit the high face (1.1 -> 0.1); z exits the low face (-0.15 -> 0.85)
    swrm = _pa_swarm(envir,
                     [[0.9, 0.5, 0.5], [0.5, 0.9, 0.5], [0.5, 0.5, 0.05]],
                     [[2, 0, 0], [0, 2, 0], [0, 0, -2]])
    swrm.move(0.1)
    pos = np.asarray(swrm.positions)
    assert np.isclose(pos[0, 0], 0.1)
    assert np.isclose(pos[1, 1], 0.1)
    assert np.isclose(pos[2, 2], 0.85)
    assert not np.any(ma.getmaskarray(swrm.positions))


def test_3d_mixed_bc_applies_per_dimension():
    # periodic in x, noflux in y, zero in z -- one agent exits each high face.
    envir = planktos.Environment(Lx=1, Ly=1, Lz=1,
                                 x_bndry=('periodic', 'periodic'),
                                 y_bndry=('noflux', 'noflux'),
                                 z_bndry=('zero', 'zero'),
                                 flow=[np.zeros((3, 3, 3)) for _ in range(3)])
    swrm = _pa_swarm(envir,
                     [[0.9, 0.5, 0.5], [0.5, 0.9, 0.5], [0.5, 0.5, 0.9]],
                     [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    swrm.move(0.1)
    mask = ma.getmaskarray(swrm.positions)
    pos = np.asarray(swrm.positions)
    assert np.isclose(pos[0, 0], 0.1) and not mask[0, 0], "periodic-x did not wrap"
    assert np.isclose(pos[1, 1], 1.0) and not mask[1, 1], "noflux-y did not clip"
    assert np.all(mask[2]), "zero-z leaver was not masked"


def test_2d_mixed_bc_periodic_x_noflux_y():
    # The TODO's explicit example: periodic in x, noflux in y.
    envir = planktos.Environment(Lx=1, Ly=1,
                                 x_bndry=('periodic', 'periodic'),
                                 y_bndry=('noflux', 'noflux'),
                                 flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    swrm = _pa_swarm(envir, [[0.9, 0.5], [0.5, 0.9]], [[2, 0], [0, 2]])
    swrm.move(0.1)
    pos = np.asarray(swrm.positions)
    assert np.isclose(pos[0, 0], 0.1)                  # periodic-x wrap
    assert np.isclose(pos[1, 1], 1.0)                  # noflux-y clip
    assert not np.any(ma.getmaskarray(swrm.positions))


# --------------------------------------------------------------------------- #
#                          properties                                         #
# --------------------------------------------------------------------------- #

def test_individual_vs_shared_props():
    envir = planktos.Environment()
    swrm = envir.add_swarm(swarm_size=4)
    swrm.add_prop('mu', [np.array([1., 0.]) for _ in range(4)])   # per-agent
    assert 'mu' in swrm.props.columns and 'mu' not in swrm.shared_props
    assert swrm.get_prop('mu').ndim == 2                          # (N, 2)
    swrm.add_prop('mu', np.zeros(2), shared=True)                 # promote to shared
    assert 'mu' in swrm.shared_props and 'mu' not in swrm.props.columns


# --------------------------------------------------------------------------- #
#                          multiple swarms + reset                            #
# --------------------------------------------------------------------------- #

def test_move_swarms_and_reset():
    envir = planktos.Environment()
    a = envir.add_swarm(swarm_size=3)
    b = envir.add_swarm(swarm_size=3)
    assert len(envir.swarms) == 2
    for _ in range(4):
        envir.move_swarms(0.5)
    assert envir.time == pytest.approx(2.0)
    assert len(a.pos_history) == 4 and len(b.pos_history) == 4

    envir.reset()
    assert envir.time == 0.0
    assert len(envir.time_history) == 0
    assert len(a.pos_history) == 0
    assert len(a.full_pos_history) == 1               # just the (reset) current positions


def test_bare_move_refuses_when_the_environment_holds_more_than_one_swarm():
    # One Swarm cannot advance the environment clock while the others stand
    #   still. This used to warn and then freeze the others by appending to
    #   pos_history alone, which left vel_history and props_history behind for
    #   the rest of the session.
    envir = planktos.Environment()
    a = envir.add_swarm(swarm_size=3)
    envir.add_swarm(swarm_size=3)
    with pytest.raises(RuntimeError, match='move_swarms'):
        a.move(0.5)
    # nothing was moved, recorded, or advanced on the way to the raise
    assert envir.time == 0.0
    assert envir.time_history == []
    assert all(len(s.pos_history) == 0 for s in envir.swarms)


def test_move_swarms_keeps_every_history_in_step():
    # The freeze-append that used to run here grew pos_history without
    #   vel_history, so full_vel_history fell out of alignment with
    #   full_pos_history and every consumer pairing the two read the wrong
    #   entry or raised.
    envir = planktos.Environment()
    a = envir.add_swarm(swarm_size=3)
    b = envir.add_swarm(swarm_size=3)
    for _ in range(3):
        envir.move_swarms(0.5)
    for s in (a, b):
        assert len(s.pos_history) == len(s.vel_history) == len(envir.time_history)
        assert len(s.full_pos_history) == len(s.full_vel_history)
        # the two consumers that pair the histories by index
        s._calc_basic_stats(False, t_indx=2)
        np.arctan2(s.vel_history[2][:, 1], s.vel_history[2][:, 0])


def test_a_single_swarm_still_moves_itself():
    # the guard keys on there being more than one Swarm, so the ordinary
    #   one-Swarm workflow is untouched
    envir = planktos.Environment()
    swrm = envir.add_swarm(swarm_size=3)
    for _ in range(3):
        swrm.move(0.5)
    assert envir.time == pytest.approx(1.5)
    assert len(swrm.pos_history) == len(swrm.vel_history) == 3


# --------------------------------------------------------------------------- #
#                    a time step that fails partway through                   #
# --------------------------------------------------------------------------- #
# Boundary conditions are applied one agent at a time, so an exception out of
# the collision code leaves the step applied to some agents and not others.
# move() leaves that partial state alone for inspection, closes the histories
# off consistently, and marks the Environment by setting time to None.

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
import _ib_harness as _h
from planktos import _ibc


def _wall_swarm(N=6):
    '''N agents below a horizontal wall, drifting up into it. No diffusion.'''
    envir = planktos.Environment(Lx=10, Ly=10)
    envir.flow = None
    mesh = _h.horizontal_wall(4, 5.0, 0.0, 10.0)
    envir.ibmesh = mesh
    envir.max_meshpt_dist = _h.max_meshpt_dist(mesh)
    init = np.column_stack((np.linspace(2.0, 8.0, N), np.full(N, 4.0)))
    swrm = planktos.Swarm(swarm_size=N, envir=envir, init=init, seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array([0.0, 2.0])
    return envir, swrm


def _fail_on_third_agent(monkeypatch, exc=RuntimeError):
    real = _ibc.apply_internal_static_BC
    calls = [0]

    def flaky(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise exc('synthetic collision failure')
        return real(*args, **kwargs)

    monkeypatch.setattr(_ibc, 'apply_internal_static_BC', flaky)


def test_failed_step_closes_histories_and_marks_the_environment(monkeypatch):
    envir, swrm = _wall_swarm()
    for _ in range(2):
        swrm.move(0.5, silent=True)
    last_good = swrm.positions.copy()

    _fail_on_third_agent(monkeypatch)
    with pytest.raises(RuntimeError, match='envir.time has been set to None'):
        swrm.move(0.5, silent=True)

    # the histories stay a consistent record: the failed step contributed the
    # state as it was when the step began, at the time the step began.
    assert envir.time is None
    assert len(swrm.pos_history) == len(envir.time_history) == 3
    assert envir.time_history[-1] == pytest.approx(1.0)
    assert np.array_equal(np.asarray(swrm.pos_history[-1]), np.asarray(last_good))

    # the partial step itself is left alone, for inspection
    assert not np.array_equal(np.asarray(swrm.positions), np.asarray(last_good))


def test_interrupted_step_is_marked_and_the_interrupt_propagates(monkeypatch):
    '''Ctrl-C partway through a step leaves the same half-applied state an error
    does, so it gets marked the same way -- but it must still arrive as a
    KeyboardInterrupt, not wrapped into something an outer "except Exception"
    would swallow.'''
    envir, swrm = _wall_swarm()
    for _ in range(2):
        swrm.move(0.5, silent=True)
    last_good = swrm.positions.copy()

    _fail_on_third_agent(monkeypatch, exc=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        swrm.move(0.5, silent=True)
    monkeypatch.undo()

    assert envir.time is None
    assert len(swrm.pos_history) == len(envir.time_history) == 3
    assert envir.time_history[-1] == pytest.approx(1.0)
    assert np.array_equal(np.asarray(swrm.pos_history[-1]), np.asarray(last_good))

    # and the mark is enforced, exactly as on the error path
    with pytest.raises(RuntimeError, match='error state'):
        swrm.move(0.5, silent=True)


def test_error_state_blocks_moves_until_it_is_backed_out(monkeypatch):
    envir, swrm = _wall_swarm()
    for _ in range(2):
        swrm.move(0.5, silent=True)
    last_good = swrm.positions.copy()

    _fail_on_third_agent(monkeypatch)
    with pytest.raises(RuntimeError):
        swrm.move(0.5, silent=True)
    monkeypatch.undo()

    # moving on would advance from positions that may be inside the boundary,
    # and an agent left inside never intersects the mesh again.
    with pytest.raises(RuntimeError, match='error state'):
        swrm.move(0.5, silent=True)

    # the recovery the error message documents
    envir.time = envir.time_history.pop()
    swrm.positions = swrm.pos_history.pop()
    swrm.velocities = swrm.vel_history.pop()
    assert envir.time == pytest.approx(1.0)
    assert np.array_equal(np.asarray(swrm.positions), np.asarray(last_good))

    swrm.move(0.5, silent=True)
    assert envir.time == pytest.approx(1.5)
    assert len(swrm.pos_history) == len(envir.time_history) == 3


def test_frame_selection_in_the_error_state_plots_history_only(monkeypatch):
    envir, swrm = _wall_swarm()
    for _ in range(2):
        swrm.move(0.5, silent=True)
    _fail_on_third_agent(monkeypatch)
    with pytest.raises(RuntimeError):
        swrm.move(0.5, silent=True)

    with pytest.warns(UserWarning, match='failed partway through'):
        frames = swrm._select_frames(fps=2, playback_rate=1)

    # every frame is a recorded state; the incomplete step (drawn at index
    # len(pos_history)) must never appear.
    assert frames.max() == len(swrm.pos_history) - 1
    assert len(swrm.pos_history) == len(envir.time_history)


def test_reset_clears_every_history_not_just_positions():
    # reset() used to clear pos_history alone, leaving vel_history (and
    # props_history) behind. The lists are indexed together -- full_vel_history
    # lines up with full_pos_history -- so one survivor misaligns them for the
    # rest of the session, which reaches the plotted heading markers and the
    # agent-velocity statistics.
    envir = _envir('zero')
    swrm = planktos.Swarm(swarm_size=5, envir=envir, init=np.full((5, 2), 0.5),
                          seed=1, store_prop_history=True)
    for _ in range(3):
        swrm.move(0.1, silent=True)
    assert len(swrm.pos_history) == len(swrm.vel_history) == 3
    assert len(swrm.props_history) == 3

    envir.reset()

    assert envir.time == 0.0
    assert envir.time_history == []
    assert swrm.pos_history == []
    assert swrm.vel_history == []
    assert swrm.props_history == []

    # and they stay in step as the run continues past the reset
    for _ in range(2):
        swrm.move(0.1, silent=True)
    assert len(swrm.pos_history) == len(swrm.vel_history) == 2
    assert len(swrm.props_history) == 2
    assert len(envir.time_history) == 2


def test_reset_leaves_props_history_off_when_it_was_never_on():
    # store_prop_history=False means props_history is None, not an empty list.
    # Clearing it must not quietly turn the feature on.
    envir = _envir('zero')
    swrm = planktos.Swarm(swarm_size=3, envir=envir, init=np.full((3, 2), 0.5),
                          seed=1)
    swrm.move(0.1, silent=True)
    envir.reset()
    assert swrm.props_history is None


# --------------------------------------------------------------------------- #
#        the movement start point is control state, not the recording          #
# --------------------------------------------------------------------------- #
# apply_boundary_conditions tests the segment (start -> end) against every mesh
# element to decide whether an agent crossed a boundary. That start point used
# to be read out of pos_history[-1], which is a recording -- so anything that
# changed what got recorded, or how often, would have silently changed the
# physics. It now comes from Swarm._prev_positions, set by whichever loop just
# moved the agents.


class _HistoryHostileSwarm(planktos.Swarm):
    '''Poisons the position history immediately before the boundary stage.

    Every recorded entry becomes NaN while keeping the list's length and
    structure, so anything still reading a start point out of the recording
    gets NaN and produces a different trajectory -- while anything reading
    _prev_positions is untouched.
    '''
    def apply_boundary_conditions(self, dt, **kwargs):
        self.pos_history = [np.full_like(p, np.nan) for p in self.pos_history]
        return super().apply_boundary_conditions(dt, **kwargs)


def _wall_envir():
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux',
                                 flow=[np.zeros((5, 5)), np.zeros((5, 5))])
    # a vertical wall at x = 5, meshed finely enough to catch every crossing
    M, y = 20, np.linspace(0.5, 9.5, 21)
    mesh = np.zeros((M, 2, 2))
    mesh[:, 0, 0] = mesh[:, 1, 0] = 5.0
    mesh[:, 0, 1], mesh[:, 1, 1] = y[:-1], y[1:]
    envir.ibmesh = mesh
    envir.max_meshpt_dist = float(
        np.linalg.norm(mesh[:, 0, :] - mesh[:, 1, :], axis=1).max())
    return envir


def _drive_into_wall(cls, ib_collisions='sliding'):
    '''Four agents with fixed drift, driven into the wall for 6 steps.

    The trajectory is collected here, step by step, from the live positions
    attribute -- deliberately not from pos_history, which the hostile subclass
    poisons. Reading the answer out of the recording is exactly the coupling
    under test.
    '''
    envir = _wall_envir()
    init = np.array([[4.0, 3.0], [4.0, 7.0], [2.0, 5.0], [4.5, 4.0]])
    swrm = cls(swarm_size=4, envir=envir, init=init, seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array([1.0, 0.2])
    traj = [np.ma.filled(swrm.positions, np.nan)]
    hits = []
    for _ in range(6):
        swrm.move(1.0, ib_collisions=ib_collisions, silent=True)
        traj.append(np.ma.filled(swrm.positions, np.nan))
        hits.append(np.asarray(swrm.ib_collision_idx).copy())
    return np.stack(traj), np.stack(hits)


@pytest.mark.parametrize('ib_collisions', ['sliding', 'sticky'])
def test_collisions_do_not_read_the_position_history(ib_collisions):
    # The claim A0 makes: destroying the recording cannot change the physics.
    control, control_hits = _drive_into_wall(planktos.Swarm, ib_collisions)
    hostile, hostile_hits = _drive_into_wall(_HistoryHostileSwarm, ib_collisions)
    assert np.isfinite(control).all(), 'control run should not produce NaN'
    assert np.array_equal(control, hostile), (
        'trajectory changed when pos_history was poisoned, so the collision '
        'path is still reading its movement start point out of the recording')
    assert np.array_equal(control_hits, hostile_hits)
    # the test only means something if the wall was actually reached
    assert (control_hits >= 0).any(), 'no collision was detected; test proves nothing'
    assert (control[-1, :, 0] < 5.0 + 1e-12).all(), 'an agent got through the wall'


def test_prev_positions_is_the_history_entry_while_capture_is_every_step():
    # At capture_interval=1 the two are the same object, which is what makes
    # A0 a provable no-op. When the capture schedule lands (run_persistence.md
    # A3) this is the invariant that catches one append site being gated and
    # the other not.
    envir = _wall_envir()
    swrm = planktos.Swarm(swarm_size=3, envir=envir,
                          init=np.array([[4.0, 3.0], [4.0, 5.0], [4.0, 7.0]]),
                          seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array([1.0, 0.0])
    for _ in range(4):
        swrm.move(1.0, silent=True)
        assert swrm._prev_positions is swrm.pos_history[-1]


def test_prev_positions_is_set_before_the_first_step():
    # apply_boundary_conditions can be reached on step 1, when pos_history is
    # still empty, so the attribute has to exist from construction.
    envir = _wall_envir()
    init = np.array([[4.0, 5.0], [1.0, 1.0]])
    swrm = planktos.Swarm(swarm_size=2, envir=envir, init=init, seed=1)
    assert np.array_equal(np.ma.getdata(swrm._prev_positions), init)
    assert swrm.pos_history == []


def test_ftle_sets_the_start_point_in_its_own_move_loops():
    # calculate_FTLE inlines its own move loops rather than calling move(), so
    # both of them are edit sites for this decoupling. A miss shows up as an
    # AttributeError or a wrong field, not as a warning.
    L, n, a = 10.0, 21, 0.4
    x = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    envir = planktos.Environment(Lx=L, Ly=L, flow=[a * (X - L / 2), -a * (Y - L / 2)])
    envir.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05)     # RK45 loop
    assert np.isfinite(np.ma.filled(envir.FTLE_largest, np.nan)).any()

    envir2 = planktos.Environment(Lx=L, Ly=L, flow=[a * (X - L / 2), -a * (Y - L / 2)])
    swrm = planktos.Swarm(swarm_size=4, envir=envir2, seed=3)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    envir2.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05, swrm=swrm)  # discrete loop
    assert np.isfinite(np.ma.filled(envir2.FTLE_largest, np.nan)).any()


# --------------------------------------------------------------------------- #
#                  moving a Swarm from one Environment to another             #
# --------------------------------------------------------------------------- #
# Environment.add_swarm accepts a Swarm object as well as a size, which is how
# a Swarm built on its own is put into the Environment it belongs in, and how
# calculate_FTLE adds its working copy. It used to append to the new
# Environment without removing the swarm from the old one, leaving it in both.

def test_a_swarm_built_on_its_own_can_be_added_to_an_environment_after_the_fact():
    # The documented way round: create the two separately, then join them. A
    # Swarm with no Environment gets a default one, which it must then leave.
    envir = planktos.Environment(Lx=20, Ly=20)
    swrm = planktos.Swarm(swarm_size=5, init=np.full((5, 2), 3.0))
    default = swrm.envir
    assert default is not envir
    assert len(default.swarms) == 1

    envir.add_swarm(swrm)

    assert swrm.envir is envir
    assert envir.swarms == [swrm]
    assert default.swarms == [], 'the swarm is still in its default Environment'


def test_moving_a_swarm_between_environments_leaves_the_first_one():
    a = planktos.Environment()
    b = planktos.Environment()
    swrm = planktos.Swarm(swarm_size=3, envir=a, seed=1)

    b.add_swarm(swrm)

    assert swrm.envir is b
    assert [s is swrm for s in b.swarms] == [True]
    assert a.swarms == []


def test_the_old_environment_stops_moving_a_swarm_that_left():
    # The visible consequence of being in both lists: move_swarms on the old
    # Environment went on moving a swarm that had left it.
    a = planktos.Environment()
    b = planktos.Environment()
    swrm = planktos.Swarm(swarm_size=3, envir=a, seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    b.add_swarm(swrm)

    a.move_swarms(0.1, silent=True)
    assert len(swrm.pos_history) == 0, 'the old Environment still moved it'
    assert a.time == pytest.approx(0.1), 'but its own clock still advances'

    b.move_swarms(0.1, silent=True)
    assert len(swrm.pos_history) == 1


def test_ftle_with_a_user_swarm_still_works_during_a_recording(tmp_path):
    # calculate_FTLE adds a shallow copy of the caller's Swarm to this same
    # Environment and pops it again. The copy is not in envir.swarms to begin
    # with, so nothing is removed and the recording is untouched -- computing an
    # FTLE field partway through a recording must not be refused.
    L, a, n = 10.0, 0.4, 21
    x = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    envir = planktos.Environment(Lx=L, Ly=L,
                                 flow=[a * (X - L / 2), -a * (Y - L / 2)])
    swrm = planktos.Swarm(swarm_size=4, envir=envir, seed=3)
    swrm.shared_props['cov'] = np.zeros((2, 2))

    with envir.record(tmp_path / 'run'):
        swrm.move(0.1, silent=True)
        envir.calculate_FTLE(grid_dim=(6, 6), T=0.3, dt=0.05, swrm=swrm)
        swrm.move(0.1, silent=True)

    assert envir.swarms == [swrm], 'the FTLE working copy was not cleaned up'


def test_moving_a_recorded_swarm_out_while_recording_raises(tmp_path):
    # A swarm leaving mid-recording cannot be expressed in the archive: rows
    # are padded at the front for a late arrival, and there is no counterpart
    # for a departure.
    a = planktos.Environment(Lx=10, Ly=10,
                             flow=[np.zeros((3, 3)), np.zeros((3, 3))])
    b = planktos.Environment(Lx=10, Ly=10)
    swrm = planktos.Swarm(swarm_size=3, envir=a, seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))

    with a.record(tmp_path / 'run'):
        swrm.move(0.1, silent=True)
        with pytest.raises(RuntimeError, match='not allowed while recording'):
            b.add_swarm(swrm)
        assert [s is swrm for s in a.swarms] == [True], \
            'the swarm must not have been half-moved'
    b.add_swarm(swrm)                     # fine once the recording has stopped
    assert a.swarms == []
