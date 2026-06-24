'''Benchmark the optional pool-based parallelization of immersed-boundary
collision detection.

Times one simulation run (serial vs ThreadPoolExecutor vs ProcessPoolExecutor)
for a large static-mesh case and a moving-mesh case, and reports wall-clock and
speedup. This is a manual perf tool, NOT a pytest test (filename does not match
test_*), so it is not collected by the suite.

Usage (from repo root):
    PYTHONPATH=. python tests/bench_ib_parallel.py

The scenarios here are intentionally larger than (and independent of) the
correctness scenarios in _ib_scenarios.py, whose parameters are pinned to the
golden baseline and must not change.
'''

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import planktos


def _build(N, M, moving, T=8, K=10, dt=0.1):
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry='noflux', y_bndry='noflux')
    ys = np.linspace(0.0, 10.0, M + 1)
    base = np.zeros((M, 2, 2))
    base[:, 0, 1] = ys[:-1]; base[:, 1, 1] = ys[1:]
    if not moving:
        mesh = base.copy()
        mesh[:, 0, 0] = 5.0; mesh[:, 1, 0] = 5.0
        envir.ibmesh = mesh
        envir.max_meshpt_dist = float(
            np.linalg.norm(mesh[:, 0, :] - mesh[:, 1, :], axis=1).max())
    else:
        times = np.linspace(0.0, dt * K, T)
        xpos = np.linspace(4.5, 5.5, T)
        mesh = np.zeros((T, M, 2, 2))
        for ti in range(T):
            mesh[ti] = base
            mesh[ti, :, 0, 0] = xpos[ti]
            mesh[ti, :, 1, 0] = xpos[ti]
        envir.ibmesh = mesh
        envir.ibmesh_times = times
        envir.max_meshpt_dist = float(
            np.linalg.norm(base[:, 0, :] - base[:, 1, :], axis=1).max())
    rng = np.random.default_rng(0)
    ICs = np.zeros((N, 2))
    ICs[:, 0] = rng.uniform(3.0, 4.2, N)
    ICs[:, 1] = rng.uniform(0.5, 9.5, N)
    return envir, ICs


def _timed_run(N, M, K, moving, pool, dt=0.1):
    envir, ICs = _build(N, M, moving, K=K, dt=dt)
    kw = {} if pool is None else {'pool': pool}
    swrm = planktos.Swarm(swarm_size=N, envir=envir, init=ICs, seed=1, **kw)
    swrm.shared_props['mu'] = np.array([2.0, 0.0])
    swrm.shared_props['cov'] = np.eye(2) * 0.01
    t0 = time.perf_counter()
    for _ in range(K):
        swrm.move(dt, ib_collisions='sliding', silent=True)
    return time.perf_counter() - t0


def _bench(label, N, M, K, moving):
    cpu = os.cpu_count()
    print(f"\n=== {label}: N={N} agents, M={M} mesh elems, K={K} steps, "
          f"cores={cpu} ===")
    t_serial = _timed_run(N, M, K, moving, pool=None)
    print(f"  serial (pool=None)      : {t_serial:7.3f} s")
    with ThreadPoolExecutor(max_workers=cpu) as pool:
        t_thread = _timed_run(N, M, K, moving, pool=pool)
    print(f"  ThreadPoolExecutor      : {t_thread:7.3f} s  "
          f"({t_serial / t_thread:4.2f}x)")
    with ProcessPoolExecutor(max_workers=cpu) as pool:
        t_proc = _timed_run(N, M, K, moving, pool=pool)
    print(f"  ProcessPoolExecutor     : {t_proc:7.3f} s  "
          f"({t_serial / t_proc:4.2f}x)")


def main():
    # Static mesh: cheap per-agent work, so dominated by Python overhead.
    _bench("static mesh", N=4000, M=300, K=10, moving=False)
    # Moving mesh: expensive per-agent SciPy solves -> best parallel payoff.
    _bench("moving mesh", N=600, M=120, K=8, moving=True)


if __name__ == '__main__':
    main()
