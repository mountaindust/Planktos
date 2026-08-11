#! /usr/bin/env python3
'''Vet the dynamic-loading machinery against real 3D data.

This answers the "then, once it loads" list in TODO.md Phase 2: the windowed
loader is exercised in 3D by the fast suite, but only against synthetic sources
and tiny committed fixtures. This runs the same questions against the real
OpenFOAM oral-arm export -- 17 dumps of 68x68x180, ~350 MB raw, streamed from
1.5 GB of files -- which is the case the branch exists for.

WHAT EACH PART ANSWERS, AND WHY IT IS SEPARATE

  Part 0  Phase 1 (A): windowed-linear reproduces full-linear. Linear
          interpolation is local, so which window is resident cannot change a
          value; anything else is a bug in the slider. Checked over forward and
          backward sweeps, non-monotone access, exactly-on-node times, and
          out-of-bounds clamping, plus that the holdover round-off does not
          accumulate over many slides.
  Part 1  Phase 1 (B): the slider's bookkeeping -- bounded window, bounded load
          count, the two index spaces agreeing, fmin/fmax widening, and the
          jump-to-beginning fast path being one load rather than a walk.
  Part 2  Phase 1 (D): get_dudt under linear splining, which is piecewise
          constant and feeds the material derivative and the inertial models.
  Part 3  The branch's headline claim, and the one thing no test can make:
          that memory stays bounded to one window on a real dataset.
  Part 4  Phase 1 (C): interpolation error against withheld ground truth.
  Part 5  Phase 1 (C): convergence order, which is what makes a number from one
          dataset mean anything for another.
  Part 6  Phase 1 (C): what it costs an actual ensemble of agents.
  Part 7  The 3D material derivative end to end, through an inertial model.

A NOTE ON THIS DATASET'S TIMELINE
Four of the 21 declared dumps never arrived, leaving three interior holes. The
surviving series is 17 dumps, of which the FIRST TWELVE are unbroken at
dt = 0.125 s. Parts 4-6 use only those twelve, because a withholding study needs
a single well-defined dump interval. That caps the convergence study at two
subsample factors, against the 2D study's four -- so the fitted orders here are
weaker evidence than the 2D ones and are reported as such.

Run from the repository root:

    python tests/manual/vet_dynamic_loading_3d.py [part ...]

with no arguments to run everything, or e.g. "0 1 2 3" for the machinery parts
only. Requires the real dataset at tests/unsteady_3D_testdata/, which is
gitignored; this lives in tests/manual/ for that reason and is excluded from
pytest collection by collect_ignore in the root conftest.
'''

import gc
import os
import sys
import time as timer
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
import planktos
from planktos import fluid

try:
    import psutil
    PROC = psutil.Process(os.getpid())
except ImportError:
    PROC = None


# ----------------------------- configuration ------------------------------- #

DATA = 'tests/unsteady_3D_testdata/VTK'
INUM = 4                    # window size for the dynamic checks
N_UNIFORM = 12              # leading dumps with unbroken dt (see module docstring)
SUBSAMPLE_FACTORS = (2, 3)  # all that 12 dumps supports with >= 4 cubic knots
N_TRACERS = 512
TRAJ_STEPS = 60

PARTS = [int(a) for a in sys.argv[1:]] or list(range(8))


def rss():
    return PROC.memory_info().rss/1e6 if PROC is not None else float('nan')


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def banner(txt):
    print('\n' + '='*74)
    print(txt)
    print('='*74)


def check(label, ok, detail=''):
    '''Report a pass/fail line and remember any failure for the final summary.'''
    FAILURES.append(label) if not ok else None
    print('  [{}] {}{}'.format('PASS' if ok else 'FAIL', label,
                               '  ' + detail if detail else ''))


FAILURES = []


class CountingOpenFOAM(fluid.OpenFOAMData):
    '''OpenFOAMData that records every window load the slider asks for.'''

    def load_dumpfiles(self, d_start, d_finish):
        self.load_calls = getattr(self, 'load_calls', [])
        self.load_calls.append((d_start, d_finish))
        return super().load_dumpfiles(d_start, d_finish)


def load(INUM_, counting=False):
    cls = CountingOpenFOAM if counting else fluid.OpenFOAMData
    import warnings
    with warnings.catch_warnings():
        # The absent dumps, the uneven spacing they leave, and the outlet/wall
        # boundary-condition corner are all known and reported elsewhere.
        warnings.simplefilter('ignore', UserWarning)
        return cls(DATA, INUM=INUM_)


def diff(a, b):
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
#                                  context                                     #
# --------------------------------------------------------------------------- #

banner('DATASET')
t0 = timer.perf_counter()
full_lin = load(True)
t_all = np.asarray(full_lin.flow_times)
raw = [np.asarray(c) for c in full_lin.get_raw_loaded_data()]
fpoints = full_lin.flow_points
speed = np.sqrt(raw[0]**2 + raw[1]**2 + raw[2]**2)
U_RMS, U_MAX = _rms(speed), float(speed.max())
dt_dump = float(t_all[1] - t_all[0])

print('{} dumps, grid {}, domain {}'.format(
    len(t_all), raw[0].shape[1:], np.round(full_lin.L, 5)))
print('loaded in {:.1f}s; raw data {:.0f} MB; RSS {:.0f} MB'.format(
    timer.perf_counter() - t0, 3*raw[0].nbytes/1e6, rss()))
print('times {:.3g} .. {:.3g}; leading uniform run {} dumps at dt = {:.4g}'.format(
    t_all[0], t_all[-1], N_UNIFORM, dt_dump))
print('speed rms {:.4g}, max {:.4g} m/s'.format(U_RMS, U_MAX))
gaps = np.nonzero(np.diff(t_all) > dt_dump*1.5)[0]
print('interior holes after t = {}'.format(
    ', '.join('{:g}'.format(t_all[i]) for i in gaps)))


# --------------------------------------------------------------------------- #
#      Part 0 -- (A) windowed-linear == full-linear, on the real dataset       #
# --------------------------------------------------------------------------- #

if 0 in PARTS:
    banner('PART 0 -- (A) windowed (INUM={}) vs full linear, real 3D data'.format(INUM))
    # Linear interpolation depends only on the two bracketing samples, so the
    # resident window cannot change a value. Any disagreement beyond the
    # holdover round-off is a slider bug.
    tol = 1e-12*max(U_MAX, 1.0)

    for label, queries in (
            ('forward sweep', np.linspace(t_all[0], t_all[-1], 61)),
            ('backward sweep', np.linspace(t_all[-1], t_all[0], 61)),
            ('non-monotone random access',
             np.random.default_rng(0).uniform(t_all[0], t_all[-1], 40)),
            ('exactly on node times', t_all.copy()),
            ('out of bounds (clamped)',
             np.array([t_all[0]-5, t_all[0]-0.01, t_all[-1]+0.01, t_all[-1]+5]))):
        dyn = load(INUM)
        worst = 0.0
        for q in queries:
            worst = max(worst, diff(dyn(q), full_lin(q)))
        check(label, worst <= tol, 'max |diff| = {:.3e} ({:.1e} of U_max)'.format(
            worst, worst/U_MAX))
        del dyn; gc.collect()

    # Holdover round-off: each forward slide carries two window-boundary values
    # by EVALUATING the outgoing spline rather than re-reading raw data, which
    # costs an ulp. If that compounded, a long run would drift.
    dyn = load(INUM)
    worst_by_pass = []
    for sweep in range(6):
        qs = np.linspace(t_all[0], t_all[-1], 25)
        if sweep % 2:
            qs = qs[::-1]
        worst_by_pass.append(max(diff(dyn(q), full_lin(q)) for q in qs))
    check('holdover round-off does not accumulate',
          worst_by_pass[-1] <= max(worst_by_pass[0], tol),
          'per-sweep max: ' + ', '.join('{:.1e}'.format(w) for w in worst_by_pass))
    del dyn; gc.collect()


# --------------------------------------------------------------------------- #
#            Part 1 -- (B) the slider's bookkeeping on real files              #
# --------------------------------------------------------------------------- #

if 1 in PARTS:
    banner('PART 1 -- (B) window-sliding behavior')
    n = len(t_all)

    dyn = load(INUM, counting=True)
    dyn.load_calls = []
    widths, idx_ok = [], True
    for q in np.linspace(t_all[0], t_all[-1], 41):
        dyn(q)
        widths.append(len(dyn._flow[0].x))
        lo, hi = dyn.loaded_idx_bnds
        dlo, dhi = dyn.loaded_dump_bnds
        # dump numbers are a dense index over surviving dumps, so the two index
        # spaces must be identical here -- that is the whole point of the choice
        idx_ok &= (lo, hi) == (dlo, dhi) and 0 <= lo <= hi <= n-1
        idx_ok &= np.allclose(dyn._flow[0].x, t_all[lo:hi+1])
    check('window stays bounded to INUM+1 samples', max(widths) <= INUM+1,
          'max resident = {} samples ({} intervals)'.format(max(widths), max(widths)-1))
    check('index spaces agree at every slide', idx_ok)
    dumps_read = sum(f-s+1 for s, f in dyn.load_calls)
    check('load count bounded over a monotone sweep', dumps_read <= 3*n,
          '{} loads, {} dumps read for {} dumps of data'.format(
              len(dyn.load_calls), dumps_read, n))

    # a query already inside the window must not touch storage
    before = len(dyn.load_calls)
    mid = 0.5*(dyn._flow[0].x[0] + dyn._flow[0].x[-1])
    dyn(mid)
    check('no load when the query stays inside the window',
          len(dyn.load_calls) == before)

    # jump to the beginning: one load, not a walk back through every window
    dyn.load_calls = []
    dyn(t_all[0])
    check('jump-to-beginning fast path is a single load',
          len(dyn.load_calls) == 1, '{} load(s)'.format(len(dyn.load_calls)))
    check('opening window restored', dyn.loaded_idx_bnds == (0, INUM))

    # fmin/fmax are running extrema over everything seen, and must stay tuples
    check('fmin/fmax are tuples',
          isinstance(dyn.fmin, tuple) and isinstance(dyn.fmax, tuple))
    dyn(t_all[-1])
    check('fmin/fmax bracket the true extrema',
          all(dyn.fmin[k] >= raw[k].min() - 1e-12 and
              dyn.fmax[k] <= raw[k].max() + 1e-12 for k in range(3)),
          'x-component seen [{:.3e}, {:.3e}] vs true [{:.3e}, {:.3e}]'.format(
              dyn.fmin[0], dyn.fmax[0], raw[0].min(), raw[0].max()))

    check('extrapolate flags closed in the interior',
          dyn._flow[0].extrapolate == (False, True),
          'at the end of the dataset: {}'.format(dyn._flow[0].extrapolate))
    check('get_raw_loaded_data returns only the resident window',
          dyn.get_raw_loaded_data()[0].shape[0] <= INUM+1,
          'shape {}'.format(dyn.get_raw_loaded_data()[0].shape))
    try:
        dyn[0]
        refused = False
    except TypeError:
        refused = True
    check('indexing a streaming FluidData is refused', refused)
    del dyn; gc.collect()


# --------------------------------------------------------------------------- #
#              Part 2 -- (D) get_dudt under linear splining                    #
# --------------------------------------------------------------------------- #

if 2 in PARTS:
    banner('PART 2 -- (D) get_dudt on real 3D data')
    # Under linear splining du/dt is the finite difference over the bracketing
    # interval: piecewise constant, discontinuous at each timestamp, and
    # right-closed -- a time landing exactly on a knot takes the interval to its
    # LEFT. This is what reaches the material derivative and the inertial models.
    dyn = load(INUM)

    worst = 0.0
    for q in np.linspace(t_all[0]+1e-6, t_all[-1]-1e-6, 41):
        worst = max(worst, diff(dyn.get_dudt(time=q), full_lin.get_dudt(time=q)))
    check('windowed du/dt matches full linear across slides', worst <= 1e-10,
          'max |diff| = {:.3e}'.format(worst))

    # piecewise constant within an interval, and equal to the finite difference
    i = 5
    lo, hi = t_all[i], t_all[i+1]
    fd = [(raw[k][i+1] - raw[k][i])/(hi - lo) for k in range(3)]
    inside = [dyn.get_dudt(time=lo + f*(hi-lo)) for f in (0.1, 0.5, 0.9)]
    check('du/dt constant within an interval',
          max(diff(inside[0], a) for a in inside[1:]) <= 1e-12)
    check('du/dt equals the interval finite difference',
          diff(inside[1], fd) <= 1e-10, 'max |diff| = {:.3e}'.format(diff(inside[1], fd)))

    jump = diff(dyn.get_dudt(time=t_all[i+1]),
                dyn.get_dudt(time=t_all[i+1] + 1e-9))
    check('du/dt jumps at a breakpoint (it is C^-1 in time)', jump > 0,
          'jump = {:.3e}'.format(jump))

    dudt_scale = _rms([(raw[k][1:] - raw[k][:-1])/dt_dump for k in range(3)])
    print('  reference |du/dt| rms scale: {:.4g} m/s^2'.format(dudt_scale))
    del dyn; gc.collect()


# --------------------------------------------------------------------------- #
#        Part 3 -- memory stays bounded to one window (the headline)           #
# --------------------------------------------------------------------------- #

if 3 in PARTS:
    banner('PART 3 -- memory boundedness across a long streaming run')
    if PROC is None:
        print('  psutil not installed; skipping.')
    else:
        # The claim the branch exists to make. Measured against the same run done
        # with everything resident, on the same data, in the same process.
        per_dump = 3*raw[0][0].nbytes/1e6
        print('  one dump = {:.1f} MB; the whole series = {:.0f} MB'.format(
            per_dump, 3*raw[0].nbytes/1e6))

        base = rss()
        dyn = load(INUM)
        after_ctor = rss()
        peak = after_ctor
        for q in np.linspace(t_all[0], t_all[-1], 61):
            dyn(q)
            peak = max(peak, rss())
        for q in np.linspace(t_all[-1], t_all[0], 61):
            dyn(q)
            peak = max(peak, rss())
        end = rss()
        print('  streaming (INUM={}): ctor {:+.0f} MB, peak {:+.0f} MB, '
              'end {:+.0f} MB (vs pre-load baseline)'.format(
                  INUM, after_ctor-base, peak-base, end-base))
        # Allow a couple of windows' slack: the slider holds the outgoing window
        # while the new one is being built, and numpy does not return freed
        # arena pages to the OS promptly.
        budget = 3*(INUM+1)*per_dump
        check('resident memory bounded by a small multiple of the window',
              peak - base < budget,
              'peak growth {:.0f} MB, budget {:.0f} MB = 3 windows'.format(
                  peak-base, budget))
        check('memory does not grow monotonically over 122 queries',
              end - after_ctor < (INUM+1)*per_dump,
              'end - ctor = {:+.0f} MB'.format(end-after_ctor))
        del dyn; gc.collect()

        print('  for comparison, the whole series resident cost {:.0f} MB '
              '(measured at load time above)'.format(3*raw[0].nbytes/1e6))
        print('  => streaming holds ~{:.0f}x less field data '
              '({} dumps vs {})'.format(len(t_all)/(INUM+1), INUM+1, len(t_all)))


# --------------------------------------------------------------------------- #
#        Part 4 -- (C) interpolation error against withheld ground truth       #
# --------------------------------------------------------------------------- #
# Both schemes are built on a subsampled set of dumps and evaluated at the dumps
# left out, whose raw values neither scheme saw. Comparing linear(t) to cubic(t)
# directly would measure their disagreement, not either one's error.

def build(idx, INUM_):
    return fluid.FluidData([raw[k][idx].copy() for k in range(3)],
                           fpoints, t_all[idx].copy(), INUM=INUM_)


def withheld(coarse, pool):
    '''The dumps left out, EXCLUDING any past the last build point.

    FluidData clamps outside its time range, so a withheld dump beyond the last
    coarse one is not interpolated by either scheme -- both return the last build
    point's raw values, identically. Scoring that as interpolation error inflates
    both schemes by the same amount, which flatters linear, contaminates the max,
    and breaks the convergence fit outright (a clamped point's error does not
    scale with the dump interval at all). With 12 dumps it was 1 sample in 6.
    '''
    held = np.setdiff1d(pool, coarse)
    return held[held < coarse[-1]]


if 4 in PARTS or 5 in PARTS:
    uni = np.arange(N_UNIFORM)          # the unbroken leading run

if 4 in PARTS:
    banner('PART 4 -- (C) velocity error vs withheld dumps')
    coarse = uni[::2]
    held = withheld(coarse, uni)
    lin, cub = build(coarse, True), build(coarse, None)
    print('  built on {} dumps (spacing {:.4g} s); tested at {} withheld'.format(
        len(coarse), 2*dt_dump, len(held)))

    err = {'linear': [], 'cubic': []}
    for n in held:
        truth = [raw[k][n] for k in range(3)]
        for name, obj in (('linear', lin), ('cubic', cub)):
            got = obj(t_all[n])
            err[name].append(np.sqrt(sum((got[k]-truth[k])**2 for k in range(3))))

    print('\n  {:<8} {:>12} {:>12} {:>12} {:>12}'.format(
        'scheme', 'rms err', 'max err', 'rms/U_rms', 'max/U_max'))
    res_u = {}
    for name in ('linear', 'cubic'):
        e = np.asarray(err[name])
        res_u[name] = (_rms(e), float(e.max()))
        print('  {:<8} {:>12.4e} {:>12.4e} {:>11.3f}% {:>11.3f}%'.format(
            name, _rms(e), e.max(), 100*_rms(e)/U_RMS, 100*e.max()/U_MAX))
    print('\n  linear/cubic rms error ratio : {:.1f}x'.format(
        res_u['linear'][0]/res_u['cubic'][0]))

    # du/dt error, against a 4th-order central difference on the FULL-resolution
    # series -- twice as fine as either interpolant was built on. Stated because
    # the choice of reference is a real decision: using the fine cubic's own
    # derivative would flatter the cubic scheme.
    def dudt_ref(n):
        return [(-c[n+2] + 8*c[n+1] - 8*c[n-1] + c[n-2])/(12*dt_dump) for c in raw]

    interior = held[(held >= 2) & (held <= N_UNIFORM-3)]   # room for the 4th-order stencil
    derr = {'linear': [], 'cubic': []}
    for n in interior:
        ref = dudt_ref(n)
        for name, obj in (('linear', lin), ('cubic', cub)):
            got = obj.get_dudt(time=t_all[n])
            derr[name].append(np.sqrt(sum((got[k]-ref[k])**2 for k in range(3))))
    scale = _rms([dudt_ref(n) for n in interior])
    print('\n  du/dt error (reference |du/dt| rms = {:.4g}):'.format(scale))
    print('  {:<8} {:>12} {:>12} {:>14}'.format('scheme', 'rms err', 'max err', 'rms/scale'))
    res_d = {}
    for name in ('linear', 'cubic'):
        e = np.asarray(derr[name])
        res_d[name] = (_rms(e), float(e.max()))
        print('  {:<8} {:>12.4e} {:>12.4e} {:>13.2f}%'.format(
            name, _rms(e), e.max(), 100*_rms(e)/scale))
    print('\n  linear/cubic rms du/dt error ratio : {:.1f}x'.format(
        res_d['linear'][0]/res_d['cubic'][0]))
    del lin, cub; gc.collect()


# --------------------------------------------------------------------------- #
#              Part 5 -- (C) convergence order in the dump interval            #
# --------------------------------------------------------------------------- #

if 5 in PARTS:
    banner('PART 5 -- (C) convergence order (only {} factors available)'.format(
        len(SUBSAMPLE_FACTORS)))
    print('  NOTE: {} uniform dumps supports only factors {}, so each order below'
          .format(N_UNIFORM, SUBSAMPLE_FACTORS))
    print('  is a fit through {} points with no residual to check. The 2D study'
          .format(len(SUBSAMPLE_FACTORS)))
    print('  (tests/manual/quantify_temporal_interp.py) used four. Treat these as')
    print('  corroboration of the 2D orders, not as an independent measurement.')

    pools = {}
    print('\n  {:>3} {:>12} {:>14} {:>14}'.format('s', 'dt', 'linear rms', 'cubic rms'))
    for s in SUBSAMPLE_FACTORS:
        c_idx = uni[::s]
        h_idx = withheld(c_idx, uni)
        l_s, c_s = build(c_idx, True), build(c_idx, None)
        el, ec = [], []
        for n in h_idx:
            tr = [raw[k][n] for k in range(3)]
            gl, gc_ = l_s(t_all[n]), c_s(t_all[n])
            el.append(np.sqrt(sum((gl[k]-tr[k])**2 for k in range(3))).ravel())
            ec.append(np.sqrt(sum((gc_[k]-tr[k])**2 for k in range(3))).ravel())
        pools[s] = (np.concatenate(el), np.concatenate(ec))
        print('  {:>3} {:>12.4g} {:>14.4e} {:>14.4e}'.format(
            s, s*dt_dump, _rms(pools[s][0]), _rms(pools[s][1])))
        del l_s, c_s; gc.collect()

    log_dt = np.log(np.array([s*dt_dump for s in SUBSAMPLE_FACTORS]))

    def fit(k, q):
        vals = [np.percentile(pools[s][k], q) if q < 100 else pools[s][k].max()
                for s in SUBSAMPLE_FACTORS]
        return np.polyfit(log_dt, np.log(np.array(vals)), 1)[0]

    print('\n  fitted order by percentile    theory: linear 2, cubic 4')
    print('  {:<12} {:>10} {:>10}'.format('percentile', 'linear', 'cubic'))
    for label, q in (('median', 50), ('90th', 90), ('99th', 99), ('max', 100)):
        print('  {:<12} {:>10.2f} {:>10.2f}'.format(label, fit(0, q), fit(1, q)))
    conc = pools[2][1].max()/np.median(pools[2][1])
    print('\n  error concentration (cubic, dt={:.4g}): max/median = {:.0f}x'.format(
        2*dt_dump, conc))

    # Whether the fit means anything at all is decided by the data, not by us.
    # Two independent signs that it does not, both checked rather than assumed:
    coarse_cubic_worse = _rms(pools[3][1]) > _rms(pools[3][0])
    lever = max(SUBSAMPLE_FACTORS)/min(SUBSAMPLE_FACTORS)
    print('\n  DIAGNOSTIC')
    print('  lever arm in dt across the fit : {:.2f}x'.format(lever))
    print('  cubic worse than linear at the coarsest dt : {}'.format(
        'YES' if coarse_cubic_worse else 'no'))
    if coarse_cubic_worse or lever < 2:
        print('''
  => THE ORDERS ABOVE ARE NOT A MEASUREMENT. Two things break the fit. The dt
     lever arm is only {:.2f}x, far too short a baseline for a log-log slope; and
     at the coarsest factor the cubic spline has just {} knots, where not-a-knot
     degenerates to a single cubic polynomial over the whole interval and does
     no better than linear. Both are consequences of having {} uniform dumps.

     The underlying reason is physical, and is the useful finding: the pulse
     period is 1.25 s, so the raw dt = {:.4g} s gives {:.0f} samples per cycle and
     the subsampled dt = {:.4g} s gives {:.0f}. Neither is deep enough into the
     asymptotic regime for a convergence order to appear. Quote the 2D orders
     (TODO Phase 1 (C)) and treat this dataset as evidence about ERROR SIZE at a
     coarse cadence, not about convergence rate.'''.format(
            lever, len(uni[::max(SUBSAMPLE_FACTORS)]), N_UNIFORM,
            dt_dump, 1.25/dt_dump,
            max(SUBSAMPLE_FACTORS)*dt_dump, 1.25/(max(SUBSAMPLE_FACTORS)*dt_dump)))


# --------------------------------------------------------------------------- #
#         Part 6 -- (C) what it costs an ensemble of real agents               #
# --------------------------------------------------------------------------- #

if 6 in PARTS:
    banner('PART 6 -- tracer trajectories, cubic vs dynamic-linear')
    # Free the context arrays first: a cubic spline over all 17 dumps costs
    # 4 coefficients x 16 intervals x 832,320 points x 3 components ~ 1.3 GB, on
    # top of which holding the raw series as well would be gratuitous.
    raw = None; full_lin = None; speed = None
    gc.collect()

    # Pure advection by explicit Euler, integrated by hand rather than through
    # Swarm.move: no diffusion, no RNG, no boundary conditions, no immersed mesh,
    # so the only difference between the two runs is the temporal interpolation.
    envir_c = planktos.Environment()
    envir_d = planktos.Environment()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        envir_c.read_openfoam_vtk_data(DATA, INUM=None)
        envir_d.read_openfoam_vtk_data(DATA, INUM=INUM)

    L = envir_c.L
    g = int(round(N_TRACERS**(1/3)))
    px, py, pz = np.meshgrid(np.linspace(0.2*L[0], 0.8*L[0], g),
                             np.linspace(0.2*L[1], 0.8*L[1], g),
                             np.linspace(0.05*L[2], 0.5*L[2], g))
    p0 = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
    t_end = float(t_all[N_UNIFORM-1])
    step = t_end/TRAJ_STEPS

    pc, pd = p0.copy(), p0.copy()
    travelled = np.zeros(len(p0))
    for k in range(TRAJ_STEPS):
        t = k*step
        vc = envir_c.interpolate_flow(pc, time=t)
        vd = envir_d.interpolate_flow(pd, time=t)
        vc = np.column_stack(vc) if isinstance(vc, list) else vc
        vd = np.column_stack(vd) if isinstance(vd, list) else vd
        pc = pc + step*vc
        pd = pd + step*vd
        travelled += np.linalg.norm(step*vc, axis=1)
    sep = np.linalg.norm(pc - pd, axis=1)
    diag = float(np.linalg.norm(L))
    print('  {} tracers, {} steps to t = {:.4g} s'.format(len(p0), TRAJ_STEPS, t_end))
    print('  rms separation      : {:.3e} m'.format(_rms(sep)))
    print('  as % of path travelled : {:.3f}%'.format(
        100*_rms(sep)/max(_rms(travelled), 1e-30)))
    print('  as % of domain diagonal: {:.3f}%'.format(100*_rms(sep)/diag))

    print('\n  ensemble statistics (what an ABM actually reports):')
    print('  {:<28} {:>14} {:>14} {:>10}'.format('quantity', 'cubic', 'dyn-linear', 'rel diff'))
    dispc = np.linalg.norm(pc - p0, axis=1)
    dispd = np.linalg.norm(pd - p0, axis=1)
    stats = [('mean x', pc[:, 0].mean(), pd[:, 0].mean()),
             ('mean y', pc[:, 1].mean(), pd[:, 1].mean()),
             ('mean z', pc[:, 2].mean(), pd[:, 2].mean()),
             ('std z', pc[:, 2].std(), pd[:, 2].std()),
             ('mean net displacement', dispc.mean(), dispd.mean()),
             ('std net displacement', dispc.std(), dispd.std()),
             ('10th pct displacement', np.percentile(dispc, 10), np.percentile(dispd, 10)),
             ('median displacement', np.percentile(dispc, 50), np.percentile(dispd, 50)),
             ('90th pct displacement', np.percentile(dispc, 90), np.percentile(dispd, 90))]
    worst_rel = 0.0
    for name, a, b in stats:
        rel = abs(a-b)/max(abs(a), 1e-30)
        worst_rel = max(worst_rel, rel)
        print('  {:<28} {:>14.6e} {:>14.6e} {:>9.3f}%'.format(name, a, b, 100*rel))
    print('\n  worst ensemble-statistic difference: {:.3f}%'.format(100*worst_rel))
    del envir_c, envir_d; gc.collect()


# --------------------------------------------------------------------------- #
#     Part 7 -- the 3D material derivative, end to end through a model         #
# --------------------------------------------------------------------------- #

if 7 in PARTS:
    banner('PART 7 -- 3D material derivative and inertial particles')
    # DuDt = du/dt + (u.grad)u is what the inertial models consume, and under
    # linear splining its du/dt term is a step function. Check it is finite,
    # correctly shaped, and that an inertial swarm actually runs on it.
    import warnings
    envir = planktos.Environment()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        envir.read_openfoam_vtk_data(DATA, INUM=INUM)
    envir.bndry = [['noflux', 'noflux']]*3

    swrm = planktos.Swarm(swarm_size=200, envir=envir)
    DuDt = swrm.get_DuDt()
    check('DuDt is finite everywhere', bool(np.all(np.isfinite(DuDt))))
    check('DuDt has the agent shape', DuDt.shape == (200, 3), str(DuDt.shape))
    print('  |DuDt| rms {:.4g}, max {:.4g} m/s^2'.format(
        _rms(np.linalg.norm(DuDt, axis=1)), np.linalg.norm(DuDt, axis=1).max()))

    from planktos import motion

    class Inertial(planktos.Swarm):
        def apply_agent_model(self, dt):
            return motion.Euler_brownian_motion(
                self, dt, ode=motion.inertial_particles(self))

    # Physical scales for the Reynolds/Stokes numbers the model needs. Water at
    # the README's conditions, and the disk radius as the length scale.
    envir.char_L = 0.025            # oral-arm disk radius, m
    envir.U = U_MAX                 # peak speed in the data, m/s
    envir.nu = 1e-6                 # kinematic viscosity of water, m^2/s
    envir.rho = 1000.               # fluid density, kg/m^3

    isw = Inertial(swarm_size=200, envir=envir)
    isw.shared_props['R'] = 0.9         # 0 < R < 2/3 aerosol, 2/3 neutral, >2/3 bubble
    isw.shared_props['diam'] = 1e-4     # 100 micron
    isw.shared_props['cov'] = np.eye(3)*1e-12
    t0 = timer.perf_counter()
    for _ in range(20):
        isw.move(0.02)
    ok = bool(np.all(np.isfinite(isw.positions.compressed())))
    check('inertial model runs 20 steps on streamed 3D fluid', ok,
          '{} of 200 agents still in domain, {:.1f}s'.format(
              int(np.ma.count(isw.positions[:, 0])), timer.perf_counter()-t0))
    print('  window after the run: {}'.format(envir.flow.loaded_idx_bnds))


# --------------------------------------------------------------------------- #

banner('SUMMARY')
if FAILURES:
    print('{} CHECK(S) FAILED:'.format(len(FAILURES)))
    for f in FAILURES:
        print('  - ' + f)
    sys.exit(1)
print('All checks passed.')

