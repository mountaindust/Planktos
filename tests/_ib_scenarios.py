'''Shared, deterministic immersed-boundary scenarios for parallelization tests.

These build small static and moving IB collision scenarios using only the public
Planktos API, run a fixed number of steps, and return the full trajectory. The
same builders are used both to generate the pre-edit golden baseline
(gen_ib_baseline.py) and to verify, after the parallelization refactor, that the
serial (pool=None) and parallel paths reproduce that baseline exactly.

Determinism: agent stochasticity comes only from Swarm.rndState, seeded here. The
boundary-collision code itself draws no random numbers, so a fixed seed makes the
trajectories reproducible and identical regardless of whether the per-agent IB
work is run serially or through a worker pool.
'''

import numpy as np
import planktos


# ----- scenario parameters (changing these invalidates a saved baseline) -----

STATIC = dict(N=40, M=20, K=25, seed=54321, ic_seed=12345,
              mu=(2.0, 0.0), cov=0.01, dt=0.1, ib='sliding',
              x_lo=3.0, x_hi=4.8, wall_x=5.0)

MOVING = dict(N=40, M=20, K=25, T=6, seed=54321, ic_seed=12345,
              mu=(2.0, 0.0), cov=0.01, dt=0.1, ib='sliding',
              x_lo=3.0, x_hi=4.2, wall_x0=4.5, wall_x1=5.5)


def _wall_segments(M, x, y_lo=0.0, y_hi=10.0):
    '''A vertical wall at the given x, split into M line-segment elements.'''
    ys = np.linspace(y_lo, y_hi, M + 1)
    mesh = np.zeros((M, 2, 2))
    mesh[:, 0, 0] = x; mesh[:, 0, 1] = ys[:-1]
    mesh[:, 1, 0] = x; mesh[:, 1, 1] = ys[1:]
    return mesh


def _initial_conditions(N, ic_seed, x_lo, x_hi, y_lo=0.5, y_hi=9.5):
    rng = np.random.default_rng(ic_seed)
    ICs = np.zeros((N, 2))
    ICs[:, 0] = rng.uniform(x_lo, x_hi, N)
    ICs[:, 1] = rng.uniform(y_lo, y_hi, N)
    return ICs


def _trajectory(pos_history, vel_history, ib_history):
    '''Stack histories into plain ndarrays, masked entries -> NaN.'''
    pos = np.stack([np.ma.filled(p, np.nan) for p in pos_history])
    vel = np.stack([np.ma.filled(v, np.nan) for v in vel_history])
    ib = np.stack(ib_history)
    return dict(pos=pos, vel=vel, ib=ib)


def run_static(pool=None):
    '''Run the static-wall scenario. Returns dict of pos/vel/ib trajectories.'''
    cfg = STATIC
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux')
    mesh = _wall_segments(cfg['M'], cfg['wall_x'])
    envir.ibmesh = mesh
    envir.max_meshpt_dist = float(
        np.linalg.norm(mesh[:, 0, :] - mesh[:, 1, :], axis=1).max())

    ICs = _initial_conditions(cfg['N'], cfg['ic_seed'], cfg['x_lo'], cfg['x_hi'])
    kw = {} if pool is None else {'pool': pool}
    swrm = planktos.Swarm(swarm_size=cfg['N'], envir=envir, init=ICs,
                          seed=cfg['seed'], **kw)
    swrm.shared_props['mu'] = np.array(cfg['mu'])
    swrm.shared_props['cov'] = np.eye(2) * cfg['cov']

    ib_history = []
    for _ in range(cfg['K']):
        swrm.move(cfg['dt'], ib_collisions=cfg['ib'], silent=True)
        ib_history.append(np.asarray(swrm.ib_collision_idx).copy())
    return _trajectory(swrm.full_pos_history, swrm.full_vel_history, ib_history)


def run_moving(pool=None):
    '''Run the moving-wall scenario. Returns dict of pos/vel/ib trajectories.'''
    cfg = MOVING
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux')
    base = _wall_segments(cfg['M'], 0.0)  # x filled in per time below
    times = np.linspace(0.0, cfg['dt'] * cfg['K'], cfg['T'])
    xpos = np.linspace(cfg['wall_x0'], cfg['wall_x1'], cfg['T'])
    mesh = np.zeros((cfg['T'], cfg['M'], 2, 2))
    for ti in range(cfg['T']):
        mesh[ti] = base
        mesh[ti, :, 0, 0] = xpos[ti]
        mesh[ti, :, 1, 0] = xpos[ti]
    envir.ibmesh = mesh
    envir.ibmesh_times = times
    envir.max_meshpt_dist = float(
        np.linalg.norm(base[:, 0, :] - base[:, 1, :], axis=1).max())

    ICs = _initial_conditions(cfg['N'], cfg['ic_seed'], cfg['x_lo'], cfg['x_hi'])
    kw = {} if pool is None else {'pool': pool}
    swrm = planktos.Swarm(swarm_size=cfg['N'], envir=envir, init=ICs,
                          seed=cfg['seed'], **kw)
    swrm.shared_props['mu'] = np.array(cfg['mu'])
    swrm.shared_props['cov'] = np.eye(2) * cfg['cov']

    ib_history = []
    for _ in range(cfg['K']):
        swrm.move(cfg['dt'], ib_collisions=cfg['ib'], silent=True)
        ib_history.append(np.asarray(swrm.ib_collision_idx).copy())
    return _trajectory(swrm.full_pos_history, swrm.full_vel_history, ib_history)


# Map scenario name -> runner, for convenience in tests/generator.
SCENARIOS = {'static': run_static, 'moving': run_moving}
