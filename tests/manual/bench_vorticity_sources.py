'''Where does vorticity cost the least: recompute, in-memory cache, or disk?

Supports the decision in docs/notes/stored_derived_fields.md sections 5 and 6.
Three ways to answer get_vorticity(t), against two fluid regimes:

    recompute   np.gradient on the temporally interpolated velocity (today)
    cache       per-dump vorticity computed once, held, blended linearly
    disk        per-dump vorticity READ from the solver's own Omega dumps,
                blended linearly, with a two-slot cache

    INUM=None   whole dataset resident, velocity splined cubically in time
    INUM=int    sliding window, velocity splined linearly

The interesting axis turns out not to be the strategy but whether the run needs
the VELOCITY as well. Rendering a backdrop does not; a live simulation does.

Run:  python tests/manual/bench_vorticity_sources.py [nframes]
Needs tests/data/Rubberband_with_Damped_Springs (76 dumps, 32x32, with Omega),
and uses tests/data/leaf_data (149 dumps, 128x192, no Omega) for scaling where
it is present.
'''

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planktos
from planktos import fluid, _dataio


def _wrap_scalar(a):
    '''Restore the periodic endpoint IB2d omits, for one 2D scalar dump.

    NB _wrap_flow cannot be reused here: it loops over range(len(flow_points))
    and so assumes one array per spatial dimension, i.e. a velocity field.
    This reproduces what it does to each component.
    '''
    out = np.empty((a.shape[0]+1, a.shape[1]+1), dtype=float)
    out[:-1, :-1] = a
    out[-1, :-1] = a[0]
    out[:-1, -1] = a[:, 0]
    out[-1, -1] = a[0, 0]
    return out

ROOT = Path(__file__).resolve().parents[1] / 'data'
RUBBER = ROOT / 'Rubberband_with_Damped_Springs' / 'viz_IB2d'
LEAF = ROOT / 'leaf_data'


def _t(fn, n=None, budget=1.0):
    '''Median seconds per call, over as many calls as fit in `budget`.'''
    fn()                                             # warm
    times = []
    t_end = time.perf_counter() + budget
    while time.perf_counter() < t_end and (n is None or len(times) < n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if n is not None and len(times) >= n:
            break
    return float(np.median(times)), len(times)


def _load(path, dt, print_dump, INUM=None, d_start=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fluid.IB2dData(str(path), dt=dt, print_dump=print_dump,
                              d_start=d_start, INUM=INUM)


# --------------------------------------------------------------------------- #
#                     the disk-backed prototype (section 4/6)                 #
# --------------------------------------------------------------------------- #

class OmegaReader:
    '''Reads IB2d Omega dumps on demand, two-slot cache, linear in time.

    Deliberately holds no velocity: this is the "backdrop without the fluid"
    path. Counts its own reads so the caching claim can be checked.
    '''

    def __init__(self, fd, path):
        self.path = Path(path)
        self.flow_times = fd.flow_times
        self.d_start = fd.d_start
        self.orig_points = fd._orig_flow_points
        self.cache = {}
        self.reads = 0

    def _dump(self, i):
        if i in self.cache:
            return self.cache[i]
        num = '{:04d}'.format(self.d_start + i)
        omega = _dataio.read_2DEulerian_Data_From_vtk(str(self.path), num, 'Omega')
        # same two transforms the velocity path applies: [y,x] -> [x,y], then
        # restore the periodic endpoint IB2d omits
        field = _wrap_scalar(omega.T)
        self.reads += 1
        if len(self.cache) >= 2:                     # keep only what a blend needs
            self.cache.pop(min(self.cache))
        self.cache[i] = field
        return field

    def __call__(self, t):
        ft = self.flow_times
        if t <= ft[0]:
            return self._dump(0)
        if t >= ft[-1]:
            return self._dump(len(ft)-1)
        i = int(np.searchsorted(ft, t) - 1)
        w = (t - ft[i]) / (ft[i+1] - ft[i])
        return (1-w)*self._dump(i) + w*self._dump(i+1)


class CachedVorticity:
    '''Per-dump vorticity computed once from resident velocity, then blended.'''

    def __init__(self, fd):
        self.flow_times = fd.flow_times
        self.fields = [fd.get_vorticity(t_idx=i) for i in range(len(fd.flow_times))]

    def __call__(self, t):
        ft = self.flow_times
        if t <= ft[0]:
            return self.fields[0]
        if t >= ft[-1]:
            return self.fields[-1]
        i = int(np.searchsorted(ft, t) - 1)
        w = (t - ft[i]) / (ft[i+1] - ft[i])
        return (1-w)*self.fields[i] + w*self.fields[i+1]


# --------------------------------------------------------------------------- #

def components(fd, path, label):
    print('\n--- per-call component costs: {} (grid {}) ---'.format(
        label, fd.fshape[1:]))
    rdr = OmegaReader(fd, path) if path is not None else None

    if rdr is not None:
        dt_read, _ = _t(lambda: (rdr.cache.clear(), rdr._dump(3))[1])
        print('  read one Omega dump from disk        {:9.3f} ms'.format(dt_read*1e3))

    mid = float(fd.flow_times[len(fd.flow_times)//2])
    dt_interp, _ = _t(lambda: fd(mid))
    print('  temporal interp of velocity          {:9.3f} ms'.format(dt_interp*1e3))

    flow = fd(mid)
    def _grad():
        np.gradient(flow[1], fd.flow_points[0], axis=0)
        np.gradient(flow[0], fd.flow_points[1], axis=1)
    dt_grad, _ = _t(_grad)
    print('  np.gradient curl only                {:9.3f} ms'.format(dt_grad*1e3))

    dt_full, _ = _t(lambda: fd.get_vorticity(time=mid))
    print('  get_vorticity (interp + curl)        {:9.3f} ms'.format(dt_full*1e3))

    a = np.asarray(flow[0]); b = np.asarray(flow[1])
    dt_blend, _ = _t(lambda: 0.5*a + 0.5*b)
    print('  linear blend of two fields           {:9.3f} ms'.format(dt_blend*1e3))
    return dt_read if rdr is not None else None


def scenarios(path, dt, print_dump, nframes):
    print('\n=== rendering {} vorticity frames across the series ==='.format(nframes))
    probe = _load(path, dt, print_dump, INUM=None)
    times = np.linspace(probe.flow_times[0], probe.flow_times[-1], nframes)
    del probe

    rows = []

    # -- velocity resident (INUM=None), recompute each frame: today's behavior
    fd = _load(path, dt, print_dump, INUM=None)
    t0 = time.perf_counter()
    for t in times:
        fd.get_vorticity(time=t)
    rows.append(('INUM=None', 'recompute (today)', time.perf_counter()-t0, ''))

    # -- velocity resident, vorticity computed once per dump then blended
    fd = _load(path, dt, print_dump, INUM=None)
    t0 = time.perf_counter()
    cache = CachedVorticity(fd)
    build = time.perf_counter()-t0
    t0 = time.perf_counter()
    for t in times:
        cache(t)
    rows.append(('INUM=None', 'in-memory cache', time.perf_counter()-t0,
                 '+{:.2f}s to build'.format(build)))

    # -- velocity resident, but read Omega from disk anyway
    fd = _load(path, dt, print_dump, INUM=None)
    rdr = OmegaReader(fd, path)
    t0 = time.perf_counter()
    for t in times:
        rdr(t)
    rows.append(('INUM=None', 'disk (Omega)', time.perf_counter()-t0,
                 '{} reads'.format(rdr.reads)))

    # -- streaming, recompute: every frame may slide the velocity window
    fd = _load(path, dt, print_dump, INUM=4)
    t0 = time.perf_counter()
    for t in times:
        fd.get_vorticity(time=t)
    rows.append(('INUM=4', 'recompute (today)', time.perf_counter()-t0,
                 'slides the velocity window'))

    # -- streaming, disk: velocity is never touched at all
    fd = _load(path, dt, print_dump, INUM=4)
    rdr = OmegaReader(fd, path)
    t0 = time.perf_counter()
    for t in times:
        rdr(t)
    rows.append(('INUM=4', 'disk (Omega)', time.perf_counter()-t0,
                 '{} reads, no velocity loaded'.format(rdr.reads)))

    print('\n  {:<10s} {:<20s} {:>10s}   {}'.format(
        'regime', 'vorticity from', 'seconds', 'note'))
    for regime, how, secs, note in rows:
        print('  {:<10s} {:<20s} {:>10.3f}   {}'.format(regime, how, secs, note))
    return rows


def _periodic_curl(fd, t):
    '''The same curl, but differencing across the periodic wrap at the edges.

    IB2d fields are periodic in both directions and Planktos restores the
    duplicated end line, so the last row/column repeats the first. np.gradient
    does not know that and falls back to a one-sided difference there.
    '''
    u, v = fd(t)
    x, y = fd.flow_points
    dx = x[1]-x[0]; dy = y[1]-y[0]

    def pgrad(f, h, axis):
        core = f[:-1, :-1]                           # drop the duplicated line
        g = (np.roll(core, -1, axis=axis)
             - np.roll(core, 1, axis=axis))/(2*h)
        return _wrap_scalar(g)

    return pgrad(v, dx, 0) - pgrad(u, dy, 1)


def agreement(path, dt, print_dump):
    '''Is the solver's Omega even the same field we compute? (section 5)'''
    fd = _load(path, dt, print_dump, INUM=None)
    rdr = OmegaReader(fd, path)
    print('\n--- stored Omega vs recomputed curl, at dump times ---')
    print('  {:>5s}  {:>10s}   {:>22s}   {:>22s}'.format(
        'dump', '|Omega|rms', 'np.gradient (today)', 'periodic differencing'))
    print('  {:>5s}  {:>10s}   {:>7s}{:>7s}{:>8s}   {:>7s}{:>7s}{:>8s}'.format(
        '', '', 'all', 'edge', 'interior', 'all', 'edge', 'interior'))
    for i in (1, 20, 60):
        t = float(fd.flow_times[i])
        stored = rdr(t)
        scale = np.sqrt(np.mean(stored**2))
        edge = np.zeros(stored.shape, bool)
        edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
        cells = []
        for f in (fd.get_vorticity(t_idx=i), _periodic_curl(fd, t)):
            d = f - stored
            cells += [100*np.sqrt(np.mean(d**2))/scale,
                      100*np.sqrt(np.mean(d[edge]**2))/scale,
                      100*np.sqrt(np.mean(d[~edge]**2))/scale]
        print('  {:>5d}  {:>10.4g}   '.format(i, scale) +
              '   '.join('{:6.2f}%{:6.2f}%{:7.2f}%'.format(*cells[k:k+3])
                         for k in (0, 3)))
    print('  => periodicity, not the finite difference, is what the edge error is.')


if __name__ == '__main__':
    nframes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    if not RUBBER.is_dir():
        raise SystemExit('missing {}'.format(RUBBER))

    fd = _load(RUBBER, 1e-3, 20, INUM=None)
    components(fd, RUBBER, 'rubberband')
    if LEAF.is_dir():
        leaf = _load(LEAF, 1e-3, 1, d_start=1)       # leaf_data starts at 0001
        components(leaf, None, 'leaf_data (no Omega on disk)')
        del leaf
    del fd

    agreement(RUBBER, 1e-3, 20)
    scenarios(RUBBER, 1e-3, 20, nframes)

    if LEAF.is_dir():
        # No Omega on disk, so only the two velocity-backed strategies can run.
        # What this shows is how the resident-fluid comparison scales with grid
        # size, and what a window slide costs when it is not trivial.
        print('\n=== the same, on leaf_data (129x193, 149 dumps, no Omega) ===')
        times = None
        for regime, INUM in (('INUM=None', None), ('INUM=4', 4)):
            fd = _load(LEAF, 1e-3, 1, INUM=INUM, d_start=1)
            if times is None:
                times = np.linspace(fd.flow_times[0], fd.flow_times[-1], nframes)
            t0 = time.perf_counter()
            for t in times:
                fd.get_vorticity(time=t)
            print('  {:<10s} {:<20s} {:>10.3f}'.format(
                regime, 'recompute (today)', time.perf_counter()-t0))
            del fd
        fd = _load(LEAF, 1e-3, 1, d_start=1)
        t0 = time.perf_counter()
        cache = CachedVorticity(fd)
        build = time.perf_counter()-t0
        t0 = time.perf_counter()
        for t in times:
            cache(t)
        print('  {:<10s} {:<20s} {:>10.3f}   +{:.2f}s to build'.format(
            'INUM=None', 'in-memory cache', time.perf_counter()-t0, build))
        # bound the disk strategy without Omega files: a u dump is a vector, so
        # reading one is an upper bound on reading a scalar Omega
        dt_u, _ = _t(lambda: _dataio.read_2DEulerian_Data_From_vtk(
            str(LEAF), '0005', 'u'))
        print('  (one u dump reads in {:.2f} ms; a scalar Omega would be less, so '
              'a 149-dump\n   disk sweep is at most {:.2f} s here)'.format(
                  dt_u*1e3, dt_u*149))
