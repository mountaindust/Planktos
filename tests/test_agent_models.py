'''Tests for agent behavior models: the apply_agent_model / after_move extension
points and the planktos.motion equation-of-motion generators.

Determinism is enforced via the Swarm seed and, where a pure trajectory is wanted,
zero diffusion (cov = 0). Constant uniform flows give straight-line advection with
an exact known answer. The stochastic engine is checked statistically with a fixed
seed and many agents.
'''

import numpy as np
import pytest

import planktos
from planktos import motion


# ---- subclasses exercising the extension points -------------------------- #

class ConstDisplacementSwarm(planktos.Swarm):
    '''Override apply_agent_model to move +0.5 in x each step, ignoring flow.'''
    def apply_agent_model(self, dt):
        out = self.positions.copy()
        out[:, 0] = self.positions[:, 0] + 0.5
        return out


class AfterMoveSwarm(planktos.Swarm):
    def after_move(self, dt):
        self.after_called = getattr(self, 'after_called', 0) + 1


class TracerSwarm(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(self, dt, ode=motion.tracer_particles(self))


class HighReSwarm(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(self, dt, ode=motion.highRe_massive_drift(self))


class LowReSwarm(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(self, dt, ode=motion.inertial_particles(self))


def _still_envir(L=100.0):
    return planktos.Environment(Lx=L, Ly=L, flow=[np.zeros((3, 3)), np.zeros((3, 3))])


def _uniform_flow_envir(u, v, L=50.0):
    return planktos.Environment(Lx=L, Ly=L, flow=[np.full((5, 5), u), np.full((5, 5), v)])


# --------------------------------------------------------------------------- #
#                      apply_agent_model / after_move                         #
# --------------------------------------------------------------------------- #

def test_override_positions_are_used_and_velocity_finite_differenced():
    envir = _still_envir()
    swrm = ConstDisplacementSwarm(swarm_size=4, envir=envir,
                                  init=np.array([[10., 10.]] * 4), seed=1)
    swrm.move(1.0)
    assert np.allclose(np.asarray(swrm.positions[:, 0]), 10.5)
    assert np.allclose(np.asarray(swrm.positions[:, 1]), 10.0)
    # move() recomputes velocity by finite difference: dx/dt = 0.5/1.0
    assert np.allclose(np.asarray(swrm.velocities[:, 0]), 0.5)


def test_after_move_hook_runs_each_step():
    envir = _still_envir()
    swrm = AfterMoveSwarm(swarm_size=3, envir=envir, init=np.array([[10., 10.]] * 3), seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    for _ in range(3):
        swrm.move(0.1)
    assert swrm.after_called == 3


# --------------------------------------------------------------------------- #
#                      advection (zero-diffusion) known answers               #
# --------------------------------------------------------------------------- #

def test_euler_zero_diffusion_is_pure_advection():
    envir = _uniform_flow_envir(3.0, -1.0)
    swrm = planktos.Swarm(swarm_size=3, envir=envir, init=np.array([[25., 25.]] * 3), seed=3)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    p0 = swrm.positions.copy()
    swrm.move(0.5)
    assert np.allclose(np.asarray(swrm.positions - p0), [1.5, -0.5])


def test_tracer_particles_follow_flow():
    envir = _uniform_flow_envir(3.0, -1.0)
    swrm = TracerSwarm(swarm_size=3, envir=envir, init=np.array([[25., 25.]] * 3), seed=2)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    p0 = swrm.positions.copy()
    swrm.move(0.5)
    assert np.allclose(np.asarray(swrm.positions - p0), [1.5, -0.5])


def test_shared_mu_adds_constant_drift():
    envir = _still_envir()
    swrm = planktos.Swarm(swarm_size=3, envir=envir, init=np.array([[50., 50.]] * 3), seed=5)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    swrm.shared_props['mu'] = np.array([4.0, -2.0])
    p0 = swrm.positions.copy()
    swrm.move(0.25)
    assert np.allclose(np.asarray(swrm.positions - p0), [1.0, -0.5])


# --------------------------------------------------------------------------- #
#                      stochastic engine (seeded)                             #
# --------------------------------------------------------------------------- #

def test_brownian_is_seed_reproducible_and_seed_sensitive():
    envir = _still_envir()
    def run(seed):
        s = planktos.Swarm(swarm_size=50, envir=envir, init=np.array([[50., 50.]] * 50), seed=seed)
        s.shared_props['cov'] = np.eye(2) * 0.2
        for _ in range(5):
            s.move(0.1)
        return np.asarray(s.positions)
    assert np.array_equal(run(7), run(7)), "same seed must reproduce"
    assert not np.allclose(run(7), run(8)), "different seed must differ"


def test_brownian_diffusion_statistics():
    # Isotropic cov = D*I gives per-step displacement variance ~ D*dt, mean ~ 0.
    D, dt, N = 0.3, 0.1, 20000
    envir = _still_envir(L=1000.0)
    swrm = planktos.Swarm(swarm_size=N, envir=envir, init=np.array([[500., 500.]] * N), seed=42)
    swrm.shared_props['cov'] = np.eye(2) * D
    p0 = swrm.positions.copy()
    swrm.move(dt)
    disp = np.asarray(swrm.positions - p0)
    assert np.allclose(disp.mean(axis=0), 0.0, atol=2e-2)
    assert np.allclose(disp.var(axis=0) / dt, D, rtol=0.1)


# --------------------------------------------------------------------------- #
#                 massive-particle generators (deterministic smoke)           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('cls,kwargs', [
    (LowReSwarm, dict(diam=0.002, R=2 / 3)),
    (HighReSwarm, dict(diam=0.2, m=0.01, Cd=0.47, cross_sec=np.pi * 0.1 ** 2)),
])
def test_massive_particle_models_run_deterministically(cls, kwargs, dim):
    # Time-dependent Brinkman flow so the inertial/drag terms are non-trivial.
    # Both 2D and 3D: highRe_massive_drift previously hardcoded 3 spatial
    # components and raised in 2D (now uses a dimension-agnostic broadcast).
    def build():
        env_kw = dict(Lx=20, Ly=20, rho=1000, mu=1000, char_L=10)
        if dim == 3:
            env_kw['Lz'] = 20
        envir = planktos.Environment(**env_kw)
        U = 0.1 * np.arange(0, 8)
        envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(8) * 0.22306,
                                res=21, tspan=[0, 8])
        envir.U = U.max()
        swrm = cls(envir=envir, swarm_size=10, seed=11, **kwargs)
        for _ in range(4):
            swrm.move(0.2)
        return np.asarray(swrm.full_pos_history[-1])
    a = build(); b = build()
    assert a.shape[1] == dim
    assert np.isfinite(a[~np.isnan(a)]).all(), "produced non-finite positions"
    assert np.array_equal(a, b, equal_nan=True), "not reproducible under a fixed seed"


# --------------------------------------------------------------------------- #
#                 motion.RK45 solver (public deterministic ODE)               #
# --------------------------------------------------------------------------- #
# RK45 wraps scipy.integrate.solve_ivp and is part of the public motion API
# (used by Environment.calculate_FTLE and documented for custom agent models).
# Contract: y0 is a 2D (rows, D) array, fun(t, y) maps (rows, D) -> (rows, D),
# and RK45 returns the state at tf with the same shape. `rows` is arbitrary -- N
# for a first-order system, or 2N when stacking positions on velocities for a
# second-order one. Keyword args pass through to solve_ivp. These pin that
# contract with closed-form ODEs and no Swarm involved.

def test_rk45_linear_growth_matches_closed_form():
    # dy/dt = a*y  ->  y(tf) = y0*exp(a*tf), preserving the (N, D) shape.
    a, tf = 0.5, 2.0
    y0 = np.array([[1.0, 2.0], [3.0, -1.0]])
    y = motion.RK45(lambda t, x: a * x, 0.0, y0, tf, rtol=1e-8, atol=1e-10)
    assert y.shape == y0.shape
    assert np.allclose(y, y0 * np.exp(a * tf), atol=1e-6)


def test_rk45_constant_velocity_is_exact():
    # dy/dt = c  ->  y(tf) = y0 + c*tf (RK integrates a degree-1 polynomial exactly).
    c = np.array([[1.0, -2.0, 0.5]])
    y0 = np.array([[10.0, 10.0, 10.0]])
    y = motion.RK45(lambda t, x: np.broadcast_to(c, x.shape), 0.0, y0, 3.0)
    assert np.allclose(y, y0 + c * 3.0, atol=1e-8)


def test_rk45_second_order_system_uses_2NxD_state():
    # The inertial-particle / FTLE stacking: state = [positions; velocities] as a
    # 2NxD array, dx/dt = [vel; -w^2 pos] (a harmonic oscillator), no Swarm.
    w, N = 2.0, 2
    pos0 = np.array([[1.0], [0.5]]); vel0 = np.array([[0.0], [1.0]])
    state0 = np.concatenate((pos0, vel0))                  # (2N, D)

    def osc(t, s):
        n = s.shape[0] // 2
        return np.concatenate((s[n:], -w**2 * s[:n]))

    sf = motion.RK45(osc, 0.0, state0, 1.0, rtol=1e-9, atol=1e-12)
    assert sf.shape == state0.shape
    assert np.allclose(sf[:N], pos0 * np.cos(w) + (vel0 / w) * np.sin(w), atol=1e-6)
    assert np.allclose(sf[N:], -pos0 * w * np.sin(w) + vel0 * np.cos(w), atol=1e-6)


def test_rk45_tolerance_kwargs_passed_to_solver():
    # Tightening rtol/atol (forwarded to solve_ivp) reduces the integration error.
    a, tf = 1.0, 1.0
    y0 = np.array([[1.0]])
    exact = y0[0, 0] * np.exp(a * tf)
    loose = motion.RK45(lambda t, x: a * x, 0.0, y0, tf, rtol=1e-2, atol=1e-4)
    tight = motion.RK45(lambda t, x: a * x, 0.0, y0, tf, rtol=1e-10, atol=1e-12)
    assert abs(tight[0, 0] - exact) < abs(loose[0, 0] - exact)


def test_rk45_rejects_non_2d_initial_state():
    with pytest.raises(TypeError):
        motion.RK45(lambda t, x: x, 0.0, np.array([1.0, 2.0]), 1.0)
