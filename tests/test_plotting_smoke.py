'''Smoke tests for the plot_* methods: they must run without error on the
non-interactive Agg backend.

These are intentionally shallow -- they assert no exception (and a saved file
where the method writes one), not pixel content. But plotting code breaks often
and silently (e.g. the save_fluid/modern-pyvista regression), so a cheap
"runs without error" sweep across the public plotting surface is worthwhile.

The Agg backend is selected before planktos (and pyplot) are imported, so no
display is needed and plt.show() inside the plot methods is a no-op. Figures are
closed after every test.
'''

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

import planktos

sys.path.insert(0, str(Path(__file__).parent))
import _ib_harness as h

# Rendering (and especially the ffmpeg movie) is heavier than the analytic unit
# tests, so gate the whole module behind --runslow to keep the default run fast
# -- the same policy as the parallel-IB suite. CI / pre-release (`pytest
# --runslow`) exercises the full plotting surface.
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _envir_2d():
    x = y = np.linspace(0, 10, 11)
    Y = np.meshgrid(x, y, indexing='ij')[1]
    envir = planktos.Environment(Lx=10, Ly=10, flow=[0.1 * Y, np.zeros_like(Y)])
    envir.flow.flow_points = (x, y)                 # shear flow, for quiver/vorticity
    return envir


def _envir_3d():
    x = y = z = np.linspace(0, 10, 11)
    Y = np.meshgrid(x, y, z, indexing='ij')[1]
    envir = planktos.Environment(Lx=10, Ly=10, Lz=10,
                                 flow=[0.1 * Y, np.zeros_like(Y), np.zeros_like(Y)])
    envir.flow.flow_points = (x, y, z)
    return envir


def _spread_swarm(envir, n=40):
    '''A swarm with spread-out positions (so the KDE plot path is non-degenerate).'''
    swrm = planktos.Swarm(swarm_size=n, envir=envir, seed=1)
    swrm.shared_props['cov'] = np.eye(len(envir.L)) * 0.05
    for _ in range(3):
        swrm.move(0.1, silent=True)
    return swrm


# --------------------------------------------------------------------------- #
#                          Environment plot methods                           #
# --------------------------------------------------------------------------- #

def test_plot_envir_2d_runs():
    envir = _envir_2d()
    envir.ibmesh = h.wall_segments(10, 5.0)    # exercise the mesh-drawing path
    envir.max_meshpt_dist = h.max_meshpt_dist(envir.ibmesh)
    envir.plot_envir()


def test_plot_envir_3d_runs():
    _envir_3d().plot_envir()


def test_plot_flow_2d_static_runs():
    _envir_2d().plot_flow()


def test_plot_flow_3d_static_runs():
    _envir_3d().plot_flow()


def test_plot_2d_vorticity_runs():
    _envir_2d().plot_2D_vort()


def test_plot_2d_ftle_runs():
    envir = _envir_2d()
    envir.calculate_FTLE(grid_dim=(8, 8), T=0.5, dt=0.05)
    envir.plot_2D_FTLE()


# --------------------------------------------------------------------------- #
#                             Swarm plot methods                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kw', [
    {'dist': 'density'},
    {'dist': 'hist'},
    {'fluid': 'vort'},
    {'fluid': 'quiver'},
])
def test_swarm_plot_2d_variants_save_file(tmp_path, kw):
    swrm = _spread_swarm(_envir_2d())
    out = tmp_path / 'plot.png'
    swrm.plot(filename=str(out), blocking=False, **kw)
    assert out.is_file()


def test_swarm_plot_3d_saves_file(tmp_path):
    swrm = _spread_swarm(_envir_3d())
    out = tmp_path / 'plot3d.png'
    swrm.plot(filename=str(out), blocking=False)
    assert out.is_file()


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg not on PATH")
def test_swarm_plot_all_movie_saves_file(tmp_path):
    # The animation/movie path (uses ffmpeg). Kept small so the default run stays fast.
    swrm = _spread_swarm(_envir_2d(), n=10)
    out = tmp_path / 'movie.mp4'
    swrm.plot_all(movie_filename=str(out), fps=5)
    assert out.is_file()
