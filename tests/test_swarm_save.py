'''Round-trip tests for the Swarm data-save methods: save_pos_to_csv, save_data,
and save_pos_to_vtk. Each writes files, reads them back with standard tools
(numpy / pandas / pyvista) and checks the recovered data matches the in-memory
swarm. Deterministic: a uniform flow with zero diffusion makes the trajectory an
exact closed form (position after k steps = init + (u*dt*k, 0)), so the saved
numbers are known.

save_pos_to_vtk(all=True) doubles as a regression guard -- it previously crashed
on any unmasked history step because the pos_history loop used `.mask` (which is
the scalar nomask when nothing is masked) instead of ma.getmaskarray.
'''

import numpy as np
import numpy.ma as ma
import pandas as pd
import pyvista as pv

import planktos


def _moved_swarm(steps=2, dt=1.0, u=2.0):
    '''A 4-agent swarm in a uniform +x flow with zero diffusion, advanced `steps`
    steps. The exact position of every agent after k steps is init + (u*dt*k, 0).'''
    envir = planktos.Environment(Lx=50, Ly=50,
                                 flow=[np.full((5, 5), u), np.zeros((5, 5))])
    init = np.array([[10., 10.], [10., 20.], [10., 30.], [10., 40.]])
    swrm = planktos.Swarm(swarm_size=4, envir=envir, init=init, seed=1)
    swrm.shared_props['cov'] = np.zeros((2, 2))
    for _ in range(steps):
        swrm.move(dt, silent=True)
    return swrm


# --------------------------------------------------------------------------- #
#                              save_pos_to_csv                                 #
# --------------------------------------------------------------------------- #

def test_save_pos_to_csv_roundtrips(tmp_path):
    swrm = _moved_swarm()
    N, D = 4, 2
    T = len(swrm.envir.time_history) + 1

    swrm.save_pos_to_csv(str(tmp_path / 'p'))
    arr = np.loadtxt(str(tmp_path / 'p.csv'), delimiter=',')
    assert arr.shape == (N + 1, (1 + D) * T)

    # header row: each time block is [cycle, time, time] (time repeated D times)
    header = arr[0].reshape(T, 1 + D)
    full_time = np.array([*swrm.envir.time_history, swrm.envir.time])
    assert np.array_equal(header[:, 0], np.arange(T))
    assert np.allclose(header[:, 1:], full_time[:, None])

    # body: per agent, per time block [mask, x, y] -- matches full_pos_history
    body = arr[1:].reshape(N, T, 1 + D)
    for k, pos in enumerate(swrm.full_pos_history):
        assert np.array_equal(body[:, k, 0].astype(bool), ma.getmaskarray(pos[:, 0]))
        assert np.allclose(body[:, k, 1:], pos.data)

    # the deterministic drift is recoverable: agent x advances by u*dt each step
    assert np.allclose(body[:, :, 1], 10.0 + 2.0 * np.arange(T)[None, :])
    assert np.allclose(body[:, :, 2], [[10.], [20.], [30.], [40.]])


def test_save_pos_to_csv_velocity_and_acceleration_roundtrip(tmp_path):
    swrm = _moved_swarm()
    swrm.save_pos_to_csv(str(tmp_path / 'p'), sv_vel=True, sv_accel=True)

    vel = np.loadtxt(str(tmp_path / 'p_vel.csv'), delimiter=',')
    acc = np.loadtxt(str(tmp_path / 'p_accel.csv'), delimiter=',')
    # each row is [mask, vx, vy] / [mask, ax, ay]
    assert np.allclose(vel[:, 1:], swrm.velocities.data)
    assert np.allclose(acc[:, 1:], swrm.accelerations.data)
    assert np.array_equal(vel[:, 0].astype(bool), ma.getmaskarray(swrm.velocities[:, 0]))
    # uniform flow + zero diffusion: velocity is exactly the drift (u, 0)
    assert np.allclose(vel[:, 1:], [2.0, 0.0])


# --------------------------------------------------------------------------- #
#                                 save_data                                    #
# --------------------------------------------------------------------------- #

def test_save_data_roundtrips_props_and_shared_props(tmp_path):
    swrm = _moved_swarm()
    swrm.add_prop('size', np.array([1., 2., 3., 4.]))      # per-agent -> props
    swrm.shared_props['mu'] = np.array([0.5, -0.5])        # shared -> npz

    swrm.save_data(str(tmp_path), 'run')
    assert (tmp_path / 'run.csv').is_file()
    assert (tmp_path / 'run_vel.csv').is_file()
    assert (tmp_path / 'run_accel.csv').is_file()

    props_back = pd.read_json(str(tmp_path / 'run_props.json'))
    assert np.allclose(props_back['size'].to_numpy(), [1., 2., 3., 4.])

    sp = np.load(str(tmp_path / 'run_shared_props.npz'), allow_pickle=True)
    assert np.allclose(sp['mu'], [0.5, -0.5])
    assert np.allclose(sp['cov'], np.zeros((2, 2)))


# --------------------------------------------------------------------------- #
#                              save_pos_to_vtk                                 #
# --------------------------------------------------------------------------- #

def test_save_pos_to_vtk_all_history_roundtrips(tmp_path):
    # Regression: all=True previously crashed on unmasked history steps.
    swrm = _moved_swarm()
    swrm.save_pos_to_vtk(str(tmp_path), 'pts', all=True)

    files = sorted(tmp_path.glob('pts_*.vtk'))
    assert len(files) == len(swrm.full_pos_history)
    for k, f in enumerate(files):
        m = pv.read(str(f))
        pos = swrm.full_pos_history[k]
        unmasked = pos[~ma.getmaskarray(pos[:, 0]), :].data
        assert m.n_points == unmasked.shape[0]
        pts = np.asarray(m.points)
        assert np.allclose(pts[:, :2], unmasked)           # 2D positions
        assert np.allclose(pts[:, 2], 0.0)                 # padded z
        assert int(np.asarray(m.field_data['CYCLE']).ravel()[0]) == k


def test_save_pos_to_vtk_excludes_masked_agents(tmp_path):
    swrm = _moved_swarm()
    swrm.positions[2, :] = ma.masked          # agent 2 has left the domain
    swrm.save_pos_to_vtk(str(tmp_path), 'pts', all=True)

    files = sorted(tmp_path.glob('pts_*.vtk'))
    # the unmasked history steps keep all 4 agents...
    assert pv.read(str(files[0])).n_points == 4
    # ...and the current step drops the masked one
    last = pv.read(str(files[-1]))
    assert last.n_points == 3
    expected = swrm.positions[~ma.getmaskarray(swrm.positions[:, 0]), :].data
    assert np.allclose(np.asarray(last.points)[:, :2], expected)
