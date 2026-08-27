'''Shared machinery for the data-streaming acceptance tests.

These tests are written against the four claims in the task that opened this
directory, not against the implementation. Everything here is therefore
deliberately blunt: build a world, run it, and compare the numbers to a world
built a different way. Nothing imports a private name unless the claim is about
one.

The fixtures used are the committed ones (tests/fixtures/), so the whole
directory runs on a clean checkout with no downloaded data. Where a test needs
the larger example datasets it asks for them through ``needs_dir`` and skips.
'''

import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import planktos


REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / 'tests' / 'fixtures'
IB2D_FIXTURE = FIXTURES / 'ib2d_fluid_min'
VORT_FIXTURE = FIXTURES / 'ib2d_fluid_vort_min'
VTK3D_FIXTURE = FIXTURES / 'vtk3d_min'

# The cadence the ib2d fixtures were generated at: 8 dumps at t = 0, .1, ... .7
IB2D_DT = 0.1
IB2D_PRINT_DUMP = 10
IB2D_NDUMPS = 8


# --------------------------------------------------------------------------- #
#                               building worlds                               #
# --------------------------------------------------------------------------- #

def copy_ib2d(tmp_path, name='src', with_vorticity=False):
    '''The ib2d fluid fixture somewhere writable.

    With ``with_vorticity`` the solver's own Omega.####.vtk series comes too,
    which is the regime where a recording reads vorticity rather than writing
    it.
    '''

    src = VORT_FIXTURE if with_vorticity else IB2D_FIXTURE
    dest = Path(tmp_path) / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob('u.*.vtk')):
        shutil.copy(f, dest)
    if with_vorticity:
        for f in sorted(src.glob('Omega.*.vtk')):
            shutil.copy(f, dest)
    return dest


def ib2d_envir(src, INUM=None, bndry='periodic'):
    '''A 2D environment on the ib2d fixture.

    Periodic agent boundaries by default so that agents stay in the domain for
    the whole run: once every agent has left, ``move`` stops asking for fluid
    and a sliding window stops sliding, which quietly removes the thing several
    of these tests are measuring.
    '''

    envir = planktos.Environment(x_bndry=bndry, y_bndry=bndry)
    envir.read_IB2d_fluid_data(str(src), dt=IB2D_DT,
                               print_dump=IB2D_PRINT_DUMP, INUM=INUM)
    return envir


def vtk3d_envir(INUM=None, bndry='periodic'):
    '''A 3D environment on the committed rectilinear vtk fixture.'''

    envir = planktos.Environment(x_bndry=bndry, y_bndry=bndry, z_bndry=bndry)
    envir.read_IBAMR3d_vtk_data(str(VTK3D_FIXTURE), title='IBAMR_db_', INUM=INUM)
    return envir


def brinkman_envir(res=51):
    '''The analytic 2D flow the tutorials use. Time-invariant, always resident.'''

    envir = planktos.Environment(rho=1000, mu=1000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=res)
    return envir


def run(swrm, steps, dt=0.5):
    '''Advance one swarm, quietly.'''

    for _ in range(steps):
        swrm.move(dt, silent=True)
    return swrm


# --------------------------------------------------------------------------- #
#                              measuring fluid I/O                            #
# --------------------------------------------------------------------------- #

class LoadCounter:
    '''Counts calls to the fluid loader of one FluidData subclass.

    Patches the class rather than the instance so that it sees calls made
    through any path, including one that builds its own object. Use as a
    context manager; the patch is always undone.
    '''

    def __init__(self, envir):
        self.cls = type(envir.flow)
        self.calls = []

    def __enter__(self):
        original = self.cls.load_dumpfiles
        self._original = original
        calls = self.calls

        def counted(inner_self, d_start, d_finish):
            calls.append((d_start, d_finish))
            return original(inner_self, d_start, d_finish)

        self.cls.load_dumpfiles = counted
        return self

    def __exit__(self, *exc):
        self.cls.load_dumpfiles = self._original
        return False

    def __len__(self):
        return len(self.calls)


class VorticityReadCounter:
    '''Counts stored-vorticity dump reads, which LoadCounter does not see.

    An archive-backed backdrop reads Omega.####.vtk rather than the velocity
    dumps, so "zero loader calls" and "no disk I/O" are different statements.
    '''

    def __init__(self, envir):
        self.cls = type(envir.flow)
        self.reads = []

    def __enter__(self):
        original = self.cls.read_dump_vorticity
        self._original = original
        reads = self.reads

        def counted(inner_self, t_idx):
            reads.append(int(t_idx))
            return original(inner_self, t_idx)

        self.cls.read_dump_vorticity = counted
        return self

    def __exit__(self, *exc):
        self.cls.read_dump_vorticity = self._original
        return False

    def __len__(self):
        return len(self.reads)


# --------------------------------------------------------------------------- #
#                              drawing the frames                             #
# --------------------------------------------------------------------------- #

def walk_frames(swrm, fluid=None, frames=None, fps=10, playback_rate=1,
                clip=None, strides=(2, 2)):
    '''Draw every frame the way ``plot_all``'s ``animate`` does.

    ``plot_all()`` with no ``movie_filename`` renders **nothing** on the Agg
    backend: ``FuncAnimation`` only calls its function on a draw event, and
    ``plt.show()`` is a no-op on a non-interactive backend. So "plot_all did not
    raise" says almost nothing -- it exercises the figure setup and stops. This
    walks the frames explicitly through the same FrameSource ``animate`` uses.

    Returns the list of fluid fields drawn (empty when ``fluid`` is None).
    '''

    from planktos import _frames

    source = _frames.FrameSource(swrm, fluid=fluid, clip=clip)
    if frames is None:
        frames = swrm._select_frames(fps, playback_rate)
    DIM3 = len(swrm.envir.L) == 3
    if fluid == 'quiver':
        source.resolve_strides(strides)
    drawn = []
    for n in frames:
        t = source.time(n)
        source.positions(n)
        source.velocities(n)
        source.props(n)
        swrm._calc_basic_stats(DIM3=DIM3, t_indx=n)
        if (swrm.envir.ibmesh is not None and swrm.envir.ibmesh.ndim == 4):
            swrm.envir.interpolate_temporal_mesh(time=t)
        if fluid == 'vort' and not DIM3:
            drawn.append(source.vorticity(t))
        elif fluid == 'quiver' and not DIM3:
            drawn.append(source.quiver(t))
    return drawn


# --------------------------------------------------------------------------- #
#                            comparing two runs                               #
# --------------------------------------------------------------------------- #

def assert_same_state(a, b, label='', rtol=0, atol=0):
    '''Two masked arrays agree, mask included.

    Exact by default: everything these tests compare is either the same
    arithmetic performed twice or it is a defect.
    '''

    a = np.ma.asanyarray(a)
    b = np.ma.asanyarray(b)
    assert a.shape == b.shape, '{}: shapes {} vs {}'.format(label, a.shape, b.shape)
    np.testing.assert_array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b),
                                  err_msg='{}: masks differ'.format(label))
    keep = ~np.ma.getmaskarray(a)
    if rtol == 0 and atol == 0:
        np.testing.assert_array_equal(np.ma.getdata(a)[keep],
                                      np.ma.getdata(b)[keep],
                                      err_msg='{}: values differ'.format(label))
    else:
        np.testing.assert_allclose(np.ma.getdata(a)[keep],
                                   np.ma.getdata(b)[keep],
                                   rtol=rtol, atol=atol,
                                   err_msg='{}: values differ'.format(label))


def assert_same_run(swrm_a, swrm_b, rtol=0, atol=0):
    '''Two swarms performed the same simulation, step for step.'''

    assert len(swrm_a.pos_history) == len(swrm_b.pos_history), \
        'different number of recorded states'
    np.testing.assert_allclose(swrm_a.envir.time_history,
                               swrm_b.envir.time_history, rtol=1e-15, atol=0,
                               err_msg='time histories differ')
    assert swrm_a.envir.time == pytest.approx(swrm_b.envir.time, rel=1e-15)
    for j, (pa, pb) in enumerate(zip(swrm_a.pos_history, swrm_b.pos_history)):
        assert_same_state(pa, pb, 'pos_history[{}]'.format(j), rtol, atol)
    for j, (va, vb) in enumerate(zip(swrm_a.vel_history, swrm_b.vel_history)):
        assert_same_state(va, vb, 'vel_history[{}]'.format(j), rtol, atol)
    assert_same_state(swrm_a.positions, swrm_b.positions, 'positions', rtol, atol)
    assert_same_state(swrm_a.velocities, swrm_b.velocities, 'velocities',
                      rtol, atol)


def snapshot(swrm):
    '''Everything a plot must not disturb, deep-copied.'''

    envir = swrm.envir
    flow = envir.flow
    return dict(
        positions=np.ma.copy(swrm.positions),
        velocities=np.ma.copy(swrm.velocities),
        accelerations=np.ma.copy(swrm.accelerations),
        pos_history=[np.ma.copy(p) for p in swrm.pos_history],
        vel_history=[np.ma.copy(v) for v in swrm.vel_history],
        time=envir.time,
        time_history=list(envir.time_history),
        rng_state=swrm.rndState.bit_generator.state,
        flow_bnds=None if flow is None else getattr(flow, 'loaded_idx_bnds', None),
    )


def assert_unchanged(swrm, before, what='plotting'):
    '''The simulation is exactly where it was before ``what`` happened.'''

    after = snapshot(swrm)
    assert_same_state(after['positions'], before['positions'],
                      '{} moved positions'.format(what))
    assert_same_state(after['velocities'], before['velocities'],
                      '{} moved velocities'.format(what))
    assert_same_state(after['accelerations'], before['accelerations'],
                      '{} moved accelerations'.format(what))
    assert len(after['pos_history']) == len(before['pos_history']), \
        '{} changed the length of pos_history'.format(what)
    for j, (p, q) in enumerate(zip(after['pos_history'], before['pos_history'])):
        assert_same_state(p, q, '{} altered pos_history[{}]'.format(what, j))
    assert after['time'] == before['time'], '{} advanced the clock'.format(what)
    assert after['time_history'] == before['time_history'], \
        '{} altered time_history'.format(what)
    assert after['rng_state'] == before['rng_state'], \
        '{} consumed random numbers'.format(what)


# --------------------------------------------------------------------------- #
#                                  skipping                                   #
# --------------------------------------------------------------------------- #

def needs_dir(path, why):
    '''Skip unless an optional (gitignored) dataset is present.'''

    path = Path(path)
    if not path.is_dir() or not any(path.iterdir()):
        pytest.skip('{} not present: {}'.format(path, why))
    return path


def run_script(script, cwd, extra_env=None, timeout=1800):
    '''Run a Python file in a clean subprocess with a headless matplotlib.

    Returns the CompletedProcess. Used for the example scripts, which must be
    exercised as written rather than imported, and for the cold-start reads,
    where the point is that nothing is left over in this process.
    '''

    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    env.setdefault('PYTHONPATH', str(REPO))
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, str(script)], cwd=str(cwd),
                          env=env, capture_output=True, text=True,
                          timeout=timeout)
