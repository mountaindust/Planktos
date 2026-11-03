'''Ground truth for the tiling tests: rendered pixels, and the materialized oracle.

Two roles:

1. **Pixel capture.** ``plot_all`` saves movies through
   ``matplotlib.animation.FFMpegWriter`` and ``Swarm.plot`` / ``plot_flow`` /
   ``plot_2D_vort`` finish with ``plt.show()``. ``capture_movie_frames`` and
   ``capture_shown`` swap those for collectors, so a test gets the exact RGBA
   bytes each frame rendered to, with no ffmpeg and no display. Nothing here
   imports planktos, so the baseline runner can use it against any revision.

2. **The oracle.** A tiled run must equal the same run over data materialized to
   the tiled size. The materialized data is built here, independently of the code
   under test: the fluid is concatenated tile by tile, the grid extended point by
   point, and the mesh translated copy by copy in the documented order (x
   slowest, the base copy first, so element ``t*N_base + e`` is element ``e`` of
   copy ``t``).

The common geometry is dyadic -- ``Lx=1, Ly=0.5`` (and ``Lz=0.25``) with
``33 x 17 (x 9)`` points -- so every node and every tile offset is exactly
representable and the tiled and materialized arithmetic can agree bitwise.
'''

import contextlib
import hashlib
import io
import itertools

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#                               pixel capture                                 #
# --------------------------------------------------------------------------- #

def _render(fig, dpi):
    '''Raw RGBA bytes of a figure as savefig renders it, with its pixel size.'''
    buf = io.BytesIO()
    fig.savefig(buf, format='rgba', dpi=dpi)
    return buf.getvalue(), (int(round(fig.get_figwidth()*dpi)),
                            int(round(fig.get_figheight()*dpi)))


class Frame:
    '''One rendered frame: its RGBA bytes and pixel size, compared by digest.'''

    def __init__(self, data, size):
        self.data = data
        self.size = size
        self.digest = hashlib.sha256(data).hexdigest()

    def array(self):
        w, h = self.size
        return np.frombuffer(self.data, np.uint8).reshape(h, w, 4)

    def __eq__(self, other):
        return self.size == other.size and self.data == other.data

    def __repr__(self):
        return 'Frame({}x{}, {})'.format(*self.size, self.digest[:12])


def _grabber_class(sink):
    '''A movie writer that appends each frame it is given to ``sink``.'''

    class FrameGrabber(animation.AbstractMovieWriter):
        def __init__(self, fps=5, metadata=None, codec=None, bitrate=None,
                     **kwargs):
            super().__init__(fps=fps, metadata=metadata, codec=codec,
                             bitrate=bitrate)

        def setup(self, fig, outfile, dpi=None):
            self.outfile = outfile
            self.fig = fig
            self.dpi = fig.dpi if dpi is None else dpi

        def grab_frame(self, **savefig_kwargs):
            sink.append(Frame(*_render(self.fig, self.dpi)))

        def finish(self):
            pass

    return FrameGrabber


@contextlib.contextmanager
def capture_movie_frames():
    '''Collect the frames ``plot_all(movie_filename=...)`` would encode.

    Yields a list that fills with one ``Frame`` per frame written. No file is
    created; any ``movie_filename`` will do. The global numpy random state is
    seeded on entry, since the porous-layer backdrop draws from it.
    '''

    np.random.seed(0)
    frames = []
    saved = animation.FFMpegWriter
    animation.FFMpegWriter = _grabber_class(frames)
    try:
        yield frames
    finally:
        animation.FFMpegWriter = saved
        plt.close('all')


@contextlib.contextmanager
def capture_shown(dpi=100):
    '''Collect what each ``plt.show()`` would display, one ``Frame`` per figure.

    Every open figure is rendered and closed at each ``show``, in figure-number
    order. The global numpy random state is seeded on entry, since the
    porous-layer backdrop draws from it.
    '''

    np.random.seed(0)
    frames = []
    saved = plt.show

    def show(*args, **kwargs):
        for num in plt.get_fignums():
            frames.append(Frame(*_render(plt.figure(num), dpi)))
        plt.close('all')

    plt.show = show
    try:
        yield frames
    finally:
        plt.show = saved
        plt.close('all')


# --------------------------------------------------------------------------- #
#                         the common dyadic geometry                          #
# --------------------------------------------------------------------------- #

DYADIC_L = (1.0, 0.5, 0.25)
DYADIC_N = (33, 17, 9)

# The tilings used throughout: every count differs from every other, on a domain
#   whose axes differ, so a swapped axis or a shared multiplier fails.
TILINGS_2D = ((3, 2), (5, 1))
TILINGS_3D = ((3, 2, 1), (1, 2, 3))


def dyadic_points(dim):
    '''The grid of the common geometry: N points spanning [0, L] per axis.'''
    return tuple(np.arange(n) * (L / (n - 1))
                 for L, n in zip(DYADIC_L[:dim], DYADIC_N[:dim]))


def make_periodic(field, dim):
    '''Copy the first line of each of the last ``dim`` axes onto its last line.

    A periodic axis's upper edge must repeat its lower one exactly, which
    evaluating a periodic function there does not achieve (sin(2 pi) is not
    sin(0) in floating point).
    '''

    field = np.array(field, dtype=float)
    for ax in range(field.ndim - dim, field.ndim):
        idx_last = [slice(None)] * field.ndim
        idx_first = [slice(None)] * field.ndim
        idx_last[ax] = -1
        idx_first[ax] = 0
        field[tuple(idx_last)] = field[tuple(idx_first)]
    return field


def dyadic_flow(dim, times=None):
    '''An analytic flow on the common geometry, periodic on every axis.

    Every component varies along every axis (and in time when ``times`` is
    given), so a transposed axis, a wrong tile offset or a time slip shows.
    Every component has a nonzero spatial mean, so a plot's printed fluid means
    are values rather than roundoff, which no two summation orders share.

    Returns
    -------
    flow : list of ndarray
        one per component, ``([t], x, y, [z])``
    flow_points : tuple of ndarray
    '''

    fp = dyadic_points(dim)
    grids = np.meshgrid(*fp, indexing='ij')
    phase = [2*np.pi*g/L for g, L in zip(grids, DYADIC_L)]
    tt = [0.0] if times is None else list(times)
    comps = [[] for _ in range(dim)]
    for t in tt:
        x = phase[0]
        y = phase[1]
        z = phase[2] if dim == 3 else 0.0
        comps[0].append(0.4 + 0.2*np.sin(x + 0.3*t)*np.cos(y) + 0.05*np.cos(z)
                        + 0.1*t)
        comps[1].append(0.08 + 0.15*np.cos(x + t)*np.sin(y) + 0.05*np.sin(z))
        if dim == 3:
            comps[2].append(0.03 + 0.1*np.sin(x)*np.cos(y + 0.5*t)*np.cos(z))
    flow = []
    for c in comps:
        arr = np.stack(c) if times is not None else c[0]
        flow.append(make_periodic(arr, dim))
    return flow, fp


# --------------------------------------------------------------------------- #
#                                 the oracle                                  #
# --------------------------------------------------------------------------- #

def tile_field(field, counts):
    '''Materialize ``field`` to the tiled size along its last ``len(counts)`` axes.

    The base is followed by ``field[1:]`` per extra tile along each axis, since a
    tile's first line is the previous tile's last.
    '''

    field = np.asarray(field)
    offset = field.ndim - len(counts)
    for n, k in enumerate(counts):
        ax = offset + n
        rest = np.take(field, np.arange(1, field.shape[ax]), axis=ax)
        field = np.concatenate([field] + [rest]*(k - 1), axis=ax)
    return field


def tile_points(flow_points, counts):
    '''The materialized grid: ``fp[1:] + n*fp[-1]`` appended for each extra tile.'''

    return tuple(np.concatenate([fp] + [fp[1:] + n*fp[-1] for n in range(1, k)])
                 for fp, k in zip(flow_points, counts))


def tile_offsets(L, counts):
    '''Each copy's translation, in the documented order: x slowest, base first.'''

    return [np.array([i*L[a] for a, i in enumerate(idx)])
            for idx in itertools.product(*(range(k) for k in counts))]


def tile_mesh_copies(mesh, L, counts):
    '''The materialized mesh: the base translated by each tile offset in turn.

    Works for a static 2D ``(N, 2, 2)`` or 3D ``(N, 3, 3)`` mesh and a moving
    ``(T, N, 2, 2)`` one, whose copies move in phase. Copies are translated from
    the base array, not rebuilt at each position, since the two differ in the
    last bit.
    '''

    mesh = np.asarray(mesh, dtype=np.float64)
    axis = 1 if mesh.ndim == 4 else 0
    return np.concatenate([mesh + off for off in tile_offsets(L, counts)],
                          axis=axis)


def fluid_envir(flow, flow_points, flow_times=None, INUM=None,
                periodic_dim=True, bndry='periodic'):
    '''An Environment over in-memory fluid, with the domain the grid spans.'''

    import planktos
    from planktos import fluid

    dim = len(flow_points)
    L = [float(fp[-1]) for fp in flow_points]
    kwargs = dict(Lx=L[0], Ly=L[1], x_bndry=bndry, y_bndry=bndry)
    if dim == 3:
        kwargs.update(Lz=L[2], z_bndry=bndry)
    envir = planktos.Environment(**kwargs)
    envir.flow = fluid.FluidData(flow, flow_points, flow_times, INUM=INUM,
                                 periodic_dim=periodic_dim)
    return envir


def oracle_envir(flow, flow_points, counts, flow_times=None, INUM=None,
                 periodic_dim=True, bndry='periodic'):
    '''The materialized counterpart of ``fluid_envir(...)`` tiled by ``counts``.'''

    return fluid_envir([tile_field(f, counts) for f in flow],
                       tile_points(flow_points, counts), flow_times, INUM,
                       periodic_dim, bndry)


def set_mesh(envir, mesh, ibmesh_times=None):
    '''Install an immersed mesh directly, as a loader would.

    A moving ``(T, N, 2, 2)`` mesh needs ``ibmesh_times``, one per frame.
    '''

    from planktos import _geom

    envir.ibmesh = np.asarray(mesh, dtype=np.float64)
    if envir.ibmesh.ndim == 4:
        envir.ibmesh_times = np.asarray(ibmesh_times, dtype=np.float64)
        envir.max_meshpt_dist = None
    else:
        envir.ibmesh_times = None
        envir.max_meshpt_dist = _geom.max_edge_length(envir.ibmesh)
