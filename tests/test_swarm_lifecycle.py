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
