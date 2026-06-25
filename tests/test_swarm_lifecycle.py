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
