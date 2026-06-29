'''Shared, deterministic immersed-boundary test harness.

Two roles:

1. Full-simulation scenarios (run_static / run_moving / SCENARIOS) that drive a
   small Swarm through a static or moving wall and return the full trajectory.
   These are consumed by test_parallel_ib.py to prove serial == threads ==
   processes and that the no-penetration invariant holds. Determinism: agent
   stochasticity comes only from Swarm.rndState, seeded here; the collision code
   itself draws no random numbers, so a fixed seed makes trajectories reproducible
   regardless of whether per-agent IB work runs serially or through a worker pool.

2. Single-trajectory primitives (mesh builders + geometry assertions) used by the
   test_collisions_static.py / test_collisions_moving.py unit tests, which call
   planktos._ibc.apply_internal_static_BC / apply_internal_moving_BC directly --
   no Environment, Swarm, flow, or RNG -- so the post-collision position is an
   exact, fast function of geometry alone.

The collision back-off is small (~1e-5): a stopped agent is placed just *inside*
the boundary it struck, never past it. POS_ATOL below is the position tolerance
used by the unit tests, chosen comfortably larger than that back-off.
'''

import numpy as np
import planktos
from planktos import _ibc


# Position tolerance for asserting exact post-collision locations. Larger than the
# code's ~1e-5 epsilon back-off, small enough to pin a real answer.
POS_ATOL = 1e-4


# --------------------------- mesh builders (2D) ----------------------------
# A 2D static mesh is an (M, 2, 2) array: M line segments, each [[x0,y0],[x1,y1]].

def wall_segments(M, x, y_lo=0.0, y_hi=10.0):
    '''A vertical wall at the given x, split into M line-segment elements.'''
    ys = np.linspace(y_lo, y_hi, M + 1)
    mesh = np.zeros((M, 2, 2))
    mesh[:, 0, 0] = x; mesh[:, 0, 1] = ys[:-1]
    mesh[:, 1, 0] = x; mesh[:, 1, 1] = ys[1:]
    return mesh


def segment(Q0, Q1):
    '''A single 2D segment as a (1, 2, 2) mesh.'''
    return np.array([[Q0, Q1]], dtype=float)


def polyline(points):
    '''Connect a sequence of points into a chain of segments: (len-1, 2, 2).
    Use to build convex (e.g. L-shape) and concave (e.g. V) multi-element joints.'''
    pts = np.asarray(points, dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def max_meshpt_dist(mesh):
    '''Longest edge of any element. Vertex-count agnostic: for 2D segments
    (M,2,2) this is |v0-v1|; for 3D triangles (M,3,3) it is the longest of the
    three edges (matching read_stl_mesh). Accepts a leading time axis too.'''
    m = np.asarray(mesh, dtype=float)
    nv = m.shape[-2]
    return max(float(np.linalg.norm(m[..., i, :] - m[..., j, :], axis=-1).max())
               for i in range(nv) for j in range(i + 1, nv))


# --------------------------- 3D triangle helpers ---------------------------
# A 3D static mesh is an (M, 3, 3) array: M triangles, each [[P0],[P1],[P2]].

def triangle(P0, P1, P2):
    '''A single triangle as a (1, 3, 3) mesh.'''
    return np.array([[P0, P1, P2]], dtype=float)


def tri_unit_normal(P0, P1, P2):
    '''Unit normal of the triangle (right-hand rule on P0->P1->P2).'''
    n = np.cross(np.asarray(P1, float) - P0, np.asarray(P2, float) - P1)
    return n / np.linalg.norm(n)


def signed_dist_to_plane(P, P0, n):
    '''Signed distance from P to the plane through P0 with unit normal n.'''
    return float(np.dot(np.asarray(P, float) - P0, n))


def assert_not_penetrated_3D(start, end, P0, P1, P2, atol=POS_ATOL):
    '''Given a trajectory that started off the plane of triangle (P0,P1,P2),
    assert the end point did not cross to the far side of that plane (it may sit
    on it within atol). Use for full-span faces the agent cannot go around.'''
    n = tri_unit_normal(P0, P1, P2)
    s0 = signed_dist_to_plane(start, P0, n)
    s1 = signed_dist_to_plane(end, P0, n)
    assert abs(s0) > atol, "start point lies on the plane; pick a start off it"
    assert np.sign(s1) == np.sign(s0) or abs(s1) <= atol, (
        f"penetration: start side {s0:.2e}, end side {s1:.2e}")


def translating_wall(M, x0, x1, T):
    '''A vertical wall translating in x from x0 to x1 over T time slices.
    Returns (mesh (T,M,2,2), times-agnostic xpositions). For direct calls to
    apply_internal_moving_BC, just take two slices as start_mesh / end_mesh.'''
    base = wall_segments(M, 0.0)
    xpos = np.linspace(x0, x1, T)
    mesh = np.zeros((T, M, 2, 2))
    for ti in range(T):
        mesh[ti] = base
        mesh[ti, :, 0, 0] = xpos[ti]
        mesh[ti, :, 1, 0] = xpos[ti]
    return mesh, xpos


# ----------------------- geometry assertion helpers ------------------------

def signed_perp_dist_2D(P, Q0, Q1):
    '''Signed perpendicular distance from point P to the infinite line through
    Q0->Q1. Sign indicates side (left of the directed line is positive).'''
    Q0 = np.asarray(Q0, float); Q1 = np.asarray(Q1, float); P = np.asarray(P, float)
    d = Q1 - Q0
    return np.cross(d, P - Q0) / np.linalg.norm(d)


def project_param_2D(P, Q0, Q1):
    '''Parameter s in P's projection onto segment Q0->Q1 (0 at Q0, 1 at Q1).'''
    Q0 = np.asarray(Q0, float); Q1 = np.asarray(Q1, float); P = np.asarray(P, float)
    d = Q1 - Q0
    return np.dot(P - Q0, d) / np.dot(d, d)


def assert_on_segment_2D(P, Q0, Q1, atol=POS_ATOL):
    '''Assert P lies on the (finite) segment Q0->Q1: ~zero perpendicular distance
    and projection parameter within [0,1].'''
    perp = signed_perp_dist_2D(P, Q0, Q1)
    assert abs(perp) <= atol, f"point {np.asarray(P)} off the segment line by {perp:.2e}"
    s = project_param_2D(P, Q0, Q1)
    assert -atol <= s <= 1 + atol, f"projection param {s:.4f} outside [0,1]"


def assert_not_penetrated_2D(start, end, Q0, Q1, atol=POS_ATOL):
    '''Given a trajectory that started strictly on one side of the infinite line
    through Q0->Q1, assert the end point did not pass to the far side (it may sit
    on the line within atol). This is the no-penetration invariant for a single
    full-span boundary the agent could not go around.'''
    s0 = signed_perp_dist_2D(start, Q0, Q1)
    s1 = signed_perp_dist_2D(end, Q0, Q1)
    assert abs(s0) > atol, "start point is on the boundary; pick a start off it"
    # end must remain on the start side, give-or-take the on-line tolerance
    assert np.sign(s1) == np.sign(s0) or abs(s1) <= atol, (
        f"penetration: start side dist {s0:.2e}, end side dist {s1:.2e}")


def call_static(start, end, mesh, ib_collisions='sliding'):
    '''Thin wrapper: call apply_internal_static_BC with max_meshpt_dist derived
    from the mesh. Returns (newend, dx, idx).'''
    return _ibc.apply_internal_static_BC(
        np.asarray(start, float), np.asarray(end, float),
        np.asarray(mesh, float), max_meshpt_dist(mesh),
        ib_collisions=ib_collisions)


def call_moving(start, end, start_mesh, end_mesh, ib_collisions='sliding'):
    '''Thin wrapper: call apply_internal_moving_BC with max_meshpt_dist and
    max_mov derived from the two mesh slices. Returns (newend, dx, idx).'''
    start_mesh = np.asarray(start_mesh, float)
    end_mesh = np.asarray(end_mesh, float)
    mmd = max(max_meshpt_dist(start_mesh), max_meshpt_dist(end_mesh))
    max_mov = float(np.linalg.norm(end_mesh - start_mesh, axis=-1).max())
    return _ibc.apply_internal_moving_BC(
        np.asarray(start, float), np.asarray(end, float),
        start_mesh, end_mesh, mmd, max_mov, ib_collisions=ib_collisions)


# ----- scenario parameters (changing these invalidates a saved baseline) -----

STATIC = dict(N=40, M=20, K=25, seed=54321, ic_seed=12345,
              mu=(2.0, 0.0), cov=0.01, dt=0.1, ib='sliding',
              x_lo=3.0, x_hi=4.8, wall_x=5.0)

MOVING = dict(N=40, M=20, K=25, T=6, seed=54321, ic_seed=12345,
              mu=(2.0, 0.0), cov=0.01, dt=0.1, ib='sliding',
              x_lo=3.0, x_hi=4.2, wall_x0=4.5, wall_x1=5.5)


# Back-compat alias for the full-simulation scenarios below.
_wall_segments = wall_segments


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


# ---- golden moving-boundary trajectory (deterministic; drift detector) -----
# A small, fully deterministic multi-step run through a translating wall, pinned
# as a baseline in test_collisions_moving.py. No RNG and no flow (agents advance
# by a fixed per-agent velocity), so the trajectory is reproducible to FP and any
# change is a genuine drift in the moving-collision code. The four agents cover:
#   0: catches the wall head-on and rides it,
#   1: catches the wall and slides tangentially (+y) along it,
#   2: trails the wall and never makes contact (control),
#   3: starts ahead of the wall and is swept/pushed by it from behind.
# Changing any value here invalidates the pinned baseline.

GOLDEN_MOVING = dict(
    M=20, n_steps=6, dt=1.0, wall_x0=5.0, wall_x1=8.0, y_lo=0.5, y_hi=9.5,
    init=[[4.0, 3.0], [4.0, 7.0], [2.0, 5.0], [6.0, 5.0]],
    drift=[[1.0, 0.0], [0.8, 0.4], [0.3, 0.0], [0.2, 0.0]],
)


class _ConstDriftSwarm(planktos.Swarm):
    '''Agents advance by a fixed per-agent velocity (self._drift) each step,
    ignoring flow and diffusion -- a deterministic driver for the golden lock.'''
    def apply_agent_model(self, dt):
        return self.positions + self._drift * dt


def golden_moving_wall_x(t):
    '''Analytic x-position of the translating golden wall at time t (linear in
    time between the two snapshots).'''
    c = GOLDEN_MOVING
    frac = t / (c['n_steps'] * c['dt'])
    return c['wall_x0'] + (c['wall_x1'] - c['wall_x0']) * frac


def run_moving_golden(ib_collisions='sliding'):
    '''Deterministic multi-step trajectory of GOLDEN_MOVING['init'] agents
    interacting with a wall translating wall_x0 -> wall_x1. Returns the
    (n_steps+1, N, 2) trajectory (masked entries -> NaN).'''
    c = GOLDEN_MOVING
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux',
                                 flow=[np.zeros((5, 5)), np.zeros((5, 5))])
    base = wall_segments(c['M'], 0.0, y_lo=c['y_lo'], y_hi=c['y_hi'])
    mesh = np.zeros((2, c['M'], 2, 2))
    for ti, xpos in enumerate((c['wall_x0'], c['wall_x1'])):
        mesh[ti] = base
        mesh[ti, :, 0, 0] = xpos
        mesh[ti, :, 1, 0] = xpos
    envir.ibmesh = mesh
    envir.ibmesh_times = np.array([0.0, c['n_steps'] * c['dt']])
    envir.max_meshpt_dist = max_meshpt_dist(base)

    swrm = _ConstDriftSwarm(swarm_size=len(c['init']), envir=envir,
                            init=np.array(c['init'], float), seed=1)
    swrm._drift = np.array(c['drift'], float)
    for _ in range(c['n_steps']):
        swrm.move(c['dt'], ib_collisions=ib_collisions, silent=True)
    return np.stack([np.ma.filled(p, np.nan) for p in swrm.full_pos_history])
