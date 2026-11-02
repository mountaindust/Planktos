'''Everything an untiled run produces, saved for a bitwise comparison across revisions.

Run by ``test_tiling_baseline.py``, once against the working tree and once against
the commit the ``tiling`` branch started from (run_persistence.md 9.4.1). It is
copied into the root of the checkout under test and run from there, so that
``import planktos`` finds that checkout's package:

    python _tiling_baseline_check.py TESTS_DIR FIXTURES WORK OUT_NPZ

``WORK`` is wiped and refilled on every run, and both runs are given the same
``WORK``, so that paths an archive records come out identical. Only API that
exists at the branch point may be used here.

Scaffolding: deleted with ``test_tiling_baseline.py`` when ``tiling`` merges back
into ``dyload``.
'''

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS, FIXTURES, WORK, OUT = (Path(a).resolve() for a in sys.argv[1:5])
sys.path.append(str(TESTS))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import planktos
from planktos import motion

from _tiling_harness import capture_movie_frames, capture_shown

if Path(planktos.__file__).resolve().parent.parent != HERE:
    raise RuntimeError('imported planktos from {}, not from {}'.format(
        planktos.__file__, HERE))

warnings.simplefilter('ignore')

out = {}


def put(key, value):
    if key in out:
        raise KeyError('duplicate key ' + key)
    if isinstance(value, list) and value and hasattr(value[0], 'digest'):
        value = np.array([f.digest for f in value])
    elif isinstance(value, np.ma.MaskedArray):
        value = np.ma.filled(value.astype(float), np.nan)
    out[key] = np.asarray(value)


def fresh(name):
    path = WORK / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def copy_fixture(name):
    dest = WORK / 'src' / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(FIXTURES / name, dest)
    return dest


def tile_noop(envir):
    '''An all-ones tiling must change nothing. The branch point raises before
    mutating anything, so there it is skipped. tile_domain(1, 1) is a call both
    signatures accept, in 2D and 3D.'''
    try:
        envir.tile_domain(1, 1)
    except NotImplementedError:
        pass


# --------------------------------------------------------------------------- #
#                              what gets saved                                #
# --------------------------------------------------------------------------- #

def run(tag, swrm, steps, dt, envir=None):
    '''Step a swarm (or every swarm, through envir) and keep each step's state.'''
    swarms = [swrm] if envir is None else envir.swarms
    rec = {i: {'pos': [], 'vel': [], 'acc': [], 'ib': []}
           for i in range(len(swarms))}
    for _ in range(steps):
        if envir is None:
            swrm.move(dt, silent=True)
        else:
            envir.move_swarms(dt)
        for i, s in enumerate(swarms):
            rec[i]['pos'].append(np.ma.filled(s.positions.astype(float), np.nan))
            rec[i]['vel'].append(np.ma.filled(s.velocities.astype(float), np.nan))
            rec[i]['acc'].append(
                np.ma.filled(s.accelerations.astype(float), np.nan))
            rec[i]['ib'].append(np.asarray(s.ib_collision_idx))
    for i, s in enumerate(swarms):
        for k, v in rec[i].items():
            put('{}/s{}/{}'.format(tag, i, k), np.stack(v))
        props(tag + '/s{}'.format(i), s)


def props(tag, swrm):
    for col in swrm.props.columns:
        try:
            put('{}/props/{}'.format(tag, col),
                np.asarray(swrm.props[col].tolist(), dtype=float))
        except (TypeError, ValueError):
            pass


def accessors(tag, envir, swrm, pts, times=(None,)):
    '''Fluid values at fixed points, some outside the domain and on its edges.'''
    pts = np.asarray(pts, dtype=float)
    for t in times:
        tt = '' if t is None else '@{}'.format(t)
        put(tag + '/interp' + tt, envir.interpolate_flow(pts, time=t))
        put(tag + '/drift' + tt, swrm.get_fluid_drift(time=t, positions=pts))
        if envir.flow.flow_times is not None:
            put(tag + '/dudt' + tt, swrm.get_dudt(time=t, positions=pts))
            put(tag + '/mean' + tt, envir.flow.get_mean_velocity(
                time=envir.time if t is None else t))
        else:
            put(tag + '/mean' + tt, envir.flow.get_mean_velocity())
    put(tag + '/mag_grad', swrm.get_fluid_mag_gradient(positions=pts))
    put(tag + '/mean_speed', envir.get_mean_fluid_speed())


def edge_points(L):
    '''Inside, on every edge and corner, just outside, and far outside.'''
    L = np.asarray(L, dtype=float)
    fr = [-0.3, -1e-17, 0.0, 0.17, 0.5, 0.83, 1.0, 1.0 + 1e-12, 1.4, 2.6]
    D = len(L)
    rng = np.random.default_rng(11)
    pts = [np.array([fr[(3*i + 7*j) % len(fr)] for j in range(D)])*L
           for i in range(len(fr))]
    pts += list(rng.uniform(-0.5, 1.5, (12, D))*L)
    return np.array(pts)


def pixels(tag, swrm, fluids, frames=(0, 2, -1), times=(None,)):
    '''Swarm.plot and selected plot_all frames, for each fluid backdrop.'''
    n = len(swrm.envir.time_history) + 1
    fr = sorted({f % n for f in frames})
    for fl in fluids:
        name = 'none' if fl is None else fl
        for t in times:
            with capture_shown() as shown:
                swrm.plot(t=t, fluid=fl, blocking=False)
            put('{}/plot/{}@{}'.format(tag, name, t), shown)
        with capture_movie_frames() as movie:
            swrm.plot_all(movie_filename=str(WORK / 'unused.mp4'), frames=fr,
                          fluid=fl)
        put('{}/plot_all/{}'.format(tag, name), movie)


def envir_pixels(tag, envir, time=None):
    with capture_shown() as shown:
        envir.plot_flow(time=time)
    put(tag + '/plot_flow', shown)
    if len(envir.L) == 2:
        with capture_shown() as shown:
            envir.plot_2D_vort(t=time)
        put(tag + '/plot_2D_vort', shown)


def directory(tag, path):
    '''Every file under a directory: npz arrays, parsed json, raw bytes otherwise.'''
    path = Path(path)
    for f in sorted(p for p in path.rglob('*') if p.is_file()):
        rel = f.relative_to(path).as_posix()
        if f.suffix == '.npz':
            with np.load(f, allow_pickle=False) as z:
                for k in z.files:
                    put('{}/{}:{}'.format(tag, rel, k), z[k])
        elif f.suffix == '.json':
            put('{}/{}'.format(tag, rel),
                json.dumps(json.loads(f.read_text()), sort_keys=True))
        else:
            put('{}/{}'.format(tag, rel), np.frombuffer(f.read_bytes(), np.uint8))


def exports(tag, envir):
    path = fresh(tag.replace('/', '_') + '_export')
    envir.save_fluid(str(path), 'fluid')
    if len(envir.L) == 2:
        envir.save_2D_vorticity(str(path), 'vort')
    directory(tag + '/export', path)


def record_and_restore(tag, envir, swrm, steps, dt, fluid, more=5):
    '''Record a stretch of the run, render from the archive, restore, continue.'''
    arch = WORK / (tag.replace('/', '_') + '_archive')
    if arch.exists():
        shutil.rmtree(arch)
    with envir.record(str(arch), fluid=fluid):
        run(tag + '/recorded', swrm, steps, dt)
    directory(tag + '/archive', arch)
    fls = fluid if isinstance(fluid, tuple) else (fluid,)
    pixels(tag + '/archived', swrm, [f for f in fls if len(envir.L) == 2])
    _, swarms = planktos.load_run(str(arch)).restore()
    run(tag + '/restored', swarms[0], more, dt)


class InertialSwarm(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(
            self, dt, ode=motion.inertial_particles(self))


class TracerSwarm(planktos.Swarm):
    def apply_agent_model(self, dt):
        return motion.Euler_brownian_motion(
            self, dt, ode=motion.tracer_particles(self))


def swarm(envir, n, seed, cls=planktos.Swarm, cov=0.01, **kwargs):
    s = cls(swarm_size=n, envir=envir, seed=seed, **kwargs)
    s.shared_props['cov'] = s.shared_props['cov'] * cov
    return s


# --------------------------------------------------------------------------- #
#                                 scenarios                                   #
# --------------------------------------------------------------------------- #

def brinkman(tspan=None, bndry='zero', mesh=None):
    '''Analytic, non-periodic fluid; agents leave the domain under 'zero'.'''
    envir = planktos.Environment(rho=1000, mu=1000, x_bndry=bndry,
                                 y_bndry=bndry)
    if tspan is None:
        envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=51)
    else:
        envir.set_brinkman_flow(alpha=66, h_p=1.5, U=[0.5, 1, 1.5, 1],
                                dpdx=[0.5, 1, 1.5, 1], res=51, tspan=tspan)
    if mesh == 'static':
        envir.read_IB2d_mesh_data(str(FIXTURES / 'mesh_min' / 'box.vertex'),
                                  method='adjacent')
    elif mesh == 'moving':
        envir.read_IB2d_mesh_data(str(FIXTURES / 'lagspts_min'), dt=0.1,
                                  print_dump=1, d_start=0)
    tile_noop(envir)
    return envir


def scenario_brinkman_static():
    envir = brinkman()
    s = swarm(envir, 30, 5, cov=0.2)
    accessors('brink/pre', envir, s, edge_points(envir.L))
    run('brink', s, 15, 0.2)
    accessors('brink/post', envir, s, edge_points(envir.L))
    pixels('brink', s, [None, 'vort', 'quiver'])
    envir_pixels('brink', envir)
    exports('brink', envir)


def scenario_brinkman_varying():
    envir = brinkman(tspan=[0, 3], bndry='periodic')
    s = swarm(envir, 25, 6, cov=0.2)
    run('brinkt', s, 12, 0.2)
    accessors('brinkt', envir, s, edge_points(envir.L), times=(None, 0.7, 2.9))
    pixels('brinkt', s, [None, 'vort', 'quiver'], times=(None, 1.0))
    envir_pixels('brinkt', envir, time=1.0)
    exports('brinkt', envir)
    record_and_restore('brinkt', envir, s, 6, 0.2, ('vort', 'quiver'))


def scenario_brinkman_meshes():
    for ib in ('sliding', 'sticky'):
        envir = brinkman(bndry='periodic', mesh='static')
        s = swarm(envir, 40, 9, cov=0.2, ib_condition=ib)
        run('box_' + ib, s, 30, 0.1)
    pixels('box_sticky', s, [None, 'vort'])
    envir = brinkman(bndry='periodic', mesh='moving')
    s = swarm(envir, 30, 4, cov=0.5,
              init=np.column_stack([np.linspace(0.5, 3.0, 30),
                                    np.linspace(1.2, 3.8, 30)]))
    run('mvbox', s, 10, 0.02)
    pixels('mvbox', s, [None, 'vort'], frames=(0, 5, -1))
    envir_pixels('mvbox', envir)


def ib2d_dyadic(INUM, bndry='periodic'):
    src = copy_fixture('ib2d_fluid_dyadic')
    envir = planktos.Environment(x_bndry=bndry, y_bndry=bndry, rho=1000,
                                 mu=1e-3, char_L=0.1, U=0.5)
    envir.read_IB2d_fluid_data(str(src), dt=1/16, print_dump=4, INUM=INUM)
    tile_noop(envir)
    return envir


def scenario_ib2d():
    pts = edge_points((1.0, 0.5))
    for label, INUM in (('cubic', None), ('linear', True), ('window', 4)):
        tag = 'ib2d_' + label
        envir = ib2d_dyadic(INUM)
        s = swarm(envir, 30, 3, cov=1e-3)
        run(tag, s, 20, 0.05)
        accessors(tag, envir, s, pts, times=(None, 0.3, 0.95))
        pixels(tag, s, [None, 'vort', 'quiver'], frames=(0, 7, -1),
               times=(None, 0.5))
        envir_pixels(tag, envir, time=0.5)
        record_and_restore(tag, envir, s, 8, 0.05, ('vort', 'quiver'))
        if INUM is None:
            exports(tag, envir)

    envir = ib2d_dyadic(None, bndry='zero')
    s = swarm(envir, 30, 8, cov=1e-2)
    run('ib2d_leaving', s, 25, 0.05)

    envir = ib2d_dyadic(None)
    s = swarm(envir, 20, 12, cls=InertialSwarm, cov=1e-4)
    s.shared_props['R'] = 0.5
    s.shared_props['diam'] = 1e-3
    run('ib2d_inertial', s, 15, 0.05)
    envir = ib2d_dyadic(4)
    s = swarm(envir, 20, 13, cls=TracerSwarm, cov=1e-4)
    run('ib2d_tracer', s, 15, 0.05)


def vtk3d(INUM):
    envir = planktos.Environment(x_bndry='periodic', y_bndry='periodic',
                                 z_bndry='periodic')
    envir.read_IBAMR3d_vtk_data(str(copy_fixture('vtk3d_min')),
                                title='IBAMR_db_', INUM=INUM)
    tile_noop(envir)
    return envir


def scenario_vtk3d():
    pts = edge_points((4.0, 3.0, 2.0))
    for label, INUM in (('cubic', None), ('window', 4)):
        tag = 'vtk3d_' + label
        envir = vtk3d(INUM)
        s = swarm(envir, 20, 7, cov=0.05)
        run(tag, s, 12, 0.3)
        accessors(tag, envir, s, pts, times=(None, 1.1, 2.5))
        pixels(tag, s, [None], frames=(0, 4, -1))
        record_and_restore(tag, envir, s, 6, 0.3, None)


def scenario_stl():
    try:
        from stl import mesh as stlmesh
    except ImportError:
        return
    lo, hi = np.array([1.7, 1.2, 0.7]), np.array([2.3, 1.8, 1.3])
    c = [np.array([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1],
                   hi[2] if i & 4 else lo[2]]) for i in range(8)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    data = np.zeros(len(faces), dtype=stlmesh.Mesh.dtype)
    data['vectors'] = [[c[i] for i in f] for f in faces]
    path = fresh('stl') / 'cube.stl'
    stlmesh.Mesh(data).save(str(path))
    envir = planktos.Environment(x_bndry='periodic', y_bndry='periodic',
                                 z_bndry='periodic')
    envir.read_IBAMR3d_vtk_data(str(copy_fixture('vtk3d_min')),
                                title='IBAMR_db_')
    envir.read_stl_mesh_data(str(path))
    tile_noop(envir)
    s = swarm(envir, 40, 21, cov=0.3,
              init=np.column_stack([np.full(40, 1.2), np.linspace(1.0, 2.0, 40),
                                    np.linspace(0.6, 1.4, 40)]))
    run('stl', s, 15, 0.1)
    pixels('stl', s, [None], frames=(0, -1))


def scenario_ftle():
    envir = brinkman(bndry='periodic')
    envir.calculate_FTLE(grid_dim=(11, 9), T=0.4, dt=0.05)
    for name in ('FTLE_largest', 'FTLE_smallest', 'FTLE_loc', 'FTLE_loc_end'):
        put('ftle/' + name, getattr(envir, name))
    envir = ib2d_dyadic(None)
    envir.calculate_FTLE(T=0.2, dt=0.05)
    for name in ('FTLE_largest', 'FTLE_smallest', 'FTLE_loc'):
        put('ftle_default/' + name, getattr(envir, name))


if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
os.chdir(WORK)
for scenario in (scenario_brinkman_static, scenario_brinkman_varying,
                 scenario_brinkman_meshes, scenario_ib2d, scenario_vtk3d,
                 scenario_stl, scenario_ftle):
    scenario()
    plt.close('all')
np.savez(OUT, **out)
print('saved', len(out), 'items from', planktos.__file__)
