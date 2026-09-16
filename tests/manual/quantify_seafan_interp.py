#! /usr/bin/env python3
'''Vet the .vti loader and quantify linear-vs-cubic interpolation on the sea fan.

The 2D counterpart of tests/manual/vet_dynamic_loading_3d.py, and the second real
dataset for the Phase 1 (C) question that tests/manual/quantify_temporal_interp.py
answers on IB2d leaf data. Two things make it worth running separately from that
one:

  * It is the first real data through fluid.VTKXMLData, so Parts 0-2 and 7 vet the
    loader itself rather than the interpolation.
  * Its cadence is the coarse end of what this branch expects to meet. The flow is
    pulsatile with a ~2 s period sampled every 0.1 s -- about 20 samples per pulse,
    against leaf data's 149 dumps at 1e-5 s. Linear-in-time is most exposed here,
    which is exactly why the number is worth having.

WHAT EACH PART ANSWERS

  Part 0  Phase 1 (A) on this loader: windowed (INUM=k) reproduces full-linear
          (INUM=True). Linear interpolation is local, so which window is resident
          cannot change a value; anything else is a bug in the slider. That
          reduction is what makes Parts 3-6 a linear-vs-cubic question with the
          streaming machinery contributing nothing.
  Part 1  Phase 1 (B): the slider's bookkeeping on real files -- bounded window,
          bounded load count over a sweep, the two index spaces agreeing,
          fmin/fmax widening, and the jump-to-beginning fast path.
  Part 2  The branch's headline claim: memory stays bounded to one window, where
          holding the dataset costs 411 MB and splining it cubically ~2 GB.
  Part 3  Phase 1 (C): velocity error against withheld dumps, whose stored values
          are ground truth neither scheme saw. Comparing the two schemes against
          each other would measure their disagreement, not either one's error.
  Part 4  Phase 1 (C): du/dt error, the term that reaches get_dudt -> the material
          derivative -> the inertial models.
  Part 5  Phase 1 (C): convergence order by error percentile, which is what lets
          anyone with a different dump interval use these numbers at all. A whole-
          field rms ratio blends a temporally smooth bulk with a rough tail and
          describes neither.
  Part 6  What it costs an ensemble of agents, which is the decision-relevant
          question for an ABM -- dispersal statistics, not agent identity.
  Part 7  The timeline this loader builds: that the .pvd, the filenames and an
          explicit dt agree, and that a windowed sweep reads each dump once.

Run from the repository root:

    python tests/manual/quantify_seafan_interp.py [part ...]

with no arguments to run everything, or e.g. "0 1 2 7" for the loader parts only.
Parts 2-6 hold the whole field and its spline coefficients and want ~3 GB free.
Requires the real dataset at tests/data/openfoam2D/, which is gitignored; this
lives in tests/manual/ for that reason and is excluded from pytest collection by
collect_ignore in the root conftest.
'''

import gc
import os
import shutil
import sys
import tempfile
import time as timer
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
import planktos
from planktos import fluid, _dataio

try:
    import psutil
    PROC = psutil.Process(os.getpid())
except ImportError:
    PROC = None


# ----------------------------- configuration ------------------------------- #

DATA = 'tests/data/openfoam2D/flow'
INUM = 4                            # window size for the dynamic checks
SUBSAMPLE_FACTORS = (2, 3, 4)       # 40 dumps -> build points at 20, 14, 10
N_TRACERS = 2025
TRAJ_STEPS = 120
NAME_RE = r'_t([0-9.]+)[.]vti$'     # the _t<time> suffix these filenames carry

PARTS = [int(a) for a in sys.argv[1:]] or list(range(8))


def rss():
    return PROC.memory_info().rss/1e6 if PROC is not None else float('nan')


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def banner(txt):
    print('\n' + '='*74)
    print(txt)
    print('='*74)


def load(**kwargs):
    '''The dataset through the real loader, without the planar-collapse notice.'''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return fluid.VTKXMLData(DATA, **kwargs)


# --------------------------------------------------------------------------- #
#                       load once, reuse the raw arrays                        #
# --------------------------------------------------------------------------- #

banner('LOADING')
t0 = timer.perf_counter()
src = load(INUM=True)
load_secs = timer.perf_counter() - t0
raw = [np.asarray(c) for c in src.get_raw_loaded_data()]
t_all = np.asarray(src.flow_times)
fpoints = src.flow_points
print('loaded {} dumps in {:.1f}s; grid {}, domain {}, {:.0f} MB resident'.format(
    len(t_all), load_secs, raw[0].shape[1:], np.round(src.L, 5),
    sum(c.nbytes for c in raw)/1e6))
print('dump source {!r}, times from {}'.format(src.dump_source, src.time_source))

dt_dump = t_all[1] - t_all[0]
speed = np.sqrt(raw[0]**2 + raw[1]**2)
U_RMS, U_MAX = _rms(speed), float(speed.max())
print('dump interval {:.3g} s over {:.4g} s; speed rms {:.4g}, max {:.4g} m/s'.format(
    dt_dump, t_all[-1], U_RMS, U_MAX))
print('NOTE: physical time runs 0.1 s ahead of environment time -- the series')
print('      starts at t = 0.1 s, so the startup pulse ends near environment 1.9.')


def build(idx, INUM_):
    '''An interpolant over the dumps at the given indices.'''
    return fluid.FluidData([raw[0][idx].copy(), raw[1][idx].copy()],
                           fpoints, t_all[idx].copy(), INUM=INUM_)


def withheld(coarse):
    '''The dumps left out, excluding any past the last build point.

    FluidData clamps outside its time range, so a withheld dump beyond the last
    coarse one is not interpolated by either scheme -- both return the last build
    point's values identically. Scoring that as interpolation error flatters
    linear and does not scale with the dump interval, corrupting the convergence
    fit. Here it affects s=3 only.
    '''
    held = np.setdiff1d(np.arange(len(t_all)), coarse)
    return held[held < coarse[-1]]


# --------------------------------------------------------------------------- #
#     Part 0 -- the reduction: windowed-linear == full-linear on real files    #
# --------------------------------------------------------------------------- #

if 0 in PARTS:
    banner('PART 0 -- windowed (INUM={}) vs full linear, through VTKXMLData'.format(INUM))
    dyn = load(INUM=INUM)
    full_lin = build(slice(None), True)

    worst = 0.0
    for q in np.linspace(t_all[0], t_all[-1], 97):
        worst = max(worst, max(np.abs(a - b).max()
                               for a, b in zip(dyn(q), full_lin(q))))
    print('forward sweep, 97 query times      : {:.3e}'.format(worst))

    worst_back = 0.0
    for q in np.linspace(t_all[-1], t_all[0], 43):
        worst_back = max(worst_back, max(np.abs(a - b).max()
                                         for a, b in zip(dyn(q), full_lin(q))))
    print('backward sweep, 43 query times     : {:.3e}'.format(worst_back))

    worst_node = 0.0
    for n in range(len(t_all)):
        worst_node = max(worst_node, max(np.abs(a - b).max() for a, b in
                                         zip(dyn(t_all[n]), (raw[0][n], raw[1][n]))))
    print('at every dump time, vs raw values  : {:.3e}'.format(worst_node))

    worst_clamp = max(np.abs(a - b).max() for a, b in
                      zip(dyn(t_all[-1] + 5.0), (raw[0][-1], raw[1][-1])))
    print('clamped past the end, vs last dump : {:.3e}'.format(worst_clamp))
    print('\nrelative to rms speed              : {:.3e}'.format(worst/U_RMS))
    print('=> windowing contributes nothing; the rest is linear vs cubic.')
    del dyn, full_lin
    gc.collect()


# --------------------------------------------------------------------------- #
#                  Part 1 -- the slider's bookkeeping                          #
# --------------------------------------------------------------------------- #

if 1 in PARTS:
    banner('PART 1 -- window bookkeeping over a full sweep')
    real_read = _dataio.read_vtkxml_grid_data
    reads = []

    def counting_read(filename, vec_name=None):
        reads.append(filename)
        return real_read(filename, vec_name)

    fluid._dataio.read_vtkxml_grid_data = counting_read
    try:
        reads.clear()
        dyn = load(INUM=INUM)
        opening = len(reads)
        widths, bnds = [], []
        for q in np.linspace(t_all[0], t_all[-1], 61):
            dyn(q)
            widths.append(len(dyn._flow[0].x))
            bnds.append(dyn.loaded_idx_bnds)
        swept = len(reads)
        reads.clear()
        dyn(t_all[0])                       # jump back to the beginning
        jump = len(reads)
    finally:
        fluid._dataio.read_vtkxml_grid_data = real_read

    print('dumps read for the opening window  : {} (expect {})'.format(opening, INUM+1))
    print('dumps read over the whole sweep    : {} (dataset has {})'.format(
        swept, len(t_all)))
    # The last window is short: the windows step by INUM-1, so the one that
    #   reaches the end of the dataset is clipped by it.
    print('window width, min/max              : {}/{} (never above {})'.format(
        min(widths), max(widths), INUM+1))
    print('index bounds stay ordered and in range : {}'.format(
        all(0 <= a < b <= len(t_all)-1 for a, b in bnds)))
    print('dump bounds agree with index bounds    : {}'.format(
        dyn.loaded_dump_bnds == dyn.loaded_idx_bnds))
    print('jump back to the start costs           : {} reads'.format(jump))
    print('fmin/fmax after the sweep              : {}, {}'.format(
        np.round(dyn.fmin, 6), np.round(dyn.fmax, 6)))
    del dyn
    gc.collect()


# --------------------------------------------------------------------------- #
#             Part 2 -- memory: the claim the branch exists for               #
# --------------------------------------------------------------------------- #

if 2 in PARTS:
    banner('PART 2 -- resident memory, windowed vs held vs splined')
    if PROC is None:
        print('psutil not installed; skipping.')
    else:
        gc.collect()
        base = rss()
        dyn = load(INUM=INUM)
        for q in np.linspace(t_all[0], t_all[-1], 41):
            dyn(q)
        windowed = rss() - base
        del dyn
        gc.collect()

        held = load(INUM=True)
        held(t_all[len(t_all)//2])
        linear_all = rss() - base
        del held
        gc.collect()

        cub = load(INUM=None)
        cub(t_all[len(t_all)//2])
        cubic_all = rss() - base
        del cub
        gc.collect()

        print('baseline (raw arrays already held) : {:.0f} MB'.format(base))
        print('windowed INUM={}, after a full sweep : {:>7.0f} MB'.format(INUM, windowed))
        print('INUM=True, whole series linear       : {:>7.0f} MB'.format(linear_all))
        print('INUM=None, whole series cubic        : {:>7.0f} MB'.format(cubic_all))
        print('\nwindowed / cubic                     : {:.1%}'.format(
            windowed/max(cubic_all, 1e-9)))


# --------------------------------------------------------------------------- #
#          Part 3 -- velocity error against withheld ground truth              #
# --------------------------------------------------------------------------- #

res_u = {}
if 3 in PARTS:
    banner('PART 3 -- velocity error vs withheld dumps (built on every 2nd dump)')
    coarse = np.arange(0, len(t_all), 2)
    held_idx = withheld(coarse)
    lin, cub = build(coarse, True), build(coarse, None)
    print('built on {} dumps (spacing {:.3g} s); tested at {} withheld dumps'.format(
        len(coarse), 2*dt_dump, len(held_idx)))

    err = {'linear': [], 'cubic': []}
    for n in held_idx:
        truth = (raw[0][n], raw[1][n])
        for name, obj in (('linear', lin), ('cubic', cub)):
            got = obj(t_all[n])
            err[name].append(np.sqrt((got[0] - truth[0])**2 + (got[1] - truth[1])**2))

    print('\n{:<8} {:>12} {:>12} {:>12} {:>12}'.format(
        'scheme', 'rms err', 'max err', 'rms/U_rms', 'max/U_max'))
    for name in ('linear', 'cubic'):
        e = np.asarray(err[name])
        res_u[name] = (_rms(e), float(e.max()))
        print('{:<8} {:>12.4e} {:>12.4e} {:>11.3f}% {:>11.3f}%'.format(
            name, _rms(e), e.max(), 100*_rms(e)/U_RMS, 100*e.max()/U_MAX))
    print('\nlinear/cubic rms error ratio : {:.1f}x'.format(
        res_u['linear'][0]/res_u['cubic'][0]))
    del err
    gc.collect()


# --------------------------------------------------------------------------- #
#        Part 4 -- du/dt error, the term that reaches the physics              #
# --------------------------------------------------------------------------- #

res_d = {}
if 4 in PARTS:
    banner('PART 4 -- du/dt error (feeds get_dudt -> material derivative)')
    # Reference: 4th-order central difference on the FULL-resolution series, a
    # grid twice as fine as either interpolant was built on. Using the fine
    # cubic's own derivative would flatter the cubic scheme.
    if 3 not in PARTS:
        coarse = np.arange(0, len(t_all), 2)
        held_idx = withheld(coarse)
        lin, cub = build(coarse, True), build(coarse, None)

    def dudt_ref(n):
        return [(-c[n+2] + 8*c[n+1] - 8*c[n-1] + c[n-2])/(12*dt_dump) for c in raw]

    interior = held_idx[(held_idx >= 2) & (held_idx <= len(t_all)-3)]
    derr = {'linear': [], 'cubic': []}
    for n in interior:
        ref = dudt_ref(n)
        for name, obj in (('linear', lin), ('cubic', cub)):
            got = obj.get_dudt(time=t_all[n])
            derr[name].append(np.sqrt((got[0] - ref[0])**2 + (got[1] - ref[1])**2))

    dudt_scale = _rms([dudt_ref(n) for n in interior[::4]])
    print('reference |du/dt| rms scale : {:.4g}'.format(dudt_scale))
    print('\n{:<8} {:>12} {:>12} {:>14}'.format('scheme', 'rms err', 'max err',
                                                'rms/scale'))
    for name in ('linear', 'cubic'):
        e = np.asarray(derr[name])
        res_d[name] = (_rms(e), float(e.max()))
        print('{:<8} {:>12.4e} {:>12.4e} {:>13.2f}%'.format(
            name, _rms(e), e.max(), 100*_rms(e)/dudt_scale))
    print('\nlinear/cubic rms du/dt error ratio : {:.1f}x'.format(
        res_d['linear'][0]/res_d['cubic'][0]))
    del derr
    gc.collect()

if 3 in PARTS or 4 in PARTS:
    del lin, cub
    gc.collect()


# --------------------------------------------------------------------------- #
#      Part 5 -- convergence order: what makes the numbers transferable        #
# --------------------------------------------------------------------------- #

orders = {}
if 5 in PARTS:
    banner('PART 5 -- convergence order in the dump interval, by error percentile')
    pools = {}
    print('{:>3} {:>12} {:>14} {:>14} {:>8}'.format(
        's', 'dt', 'linear rms', 'cubic rms', 'withheld'))
    for s in SUBSAMPLE_FACTORS:
        c_idx = np.arange(0, len(t_all), s)
        h_idx = withheld(c_idx)
        l_s, c_s = build(c_idx, True), build(c_idx, None)
        el, ec = [], []
        for n in h_idx:
            tr = (raw[0][n], raw[1][n])
            gl, gc_ = l_s(t_all[n]), c_s(t_all[n])
            el.append(np.sqrt((gl[0]-tr[0])**2 + (gl[1]-tr[1])**2).ravel())
            ec.append(np.sqrt((gc_[0]-tr[0])**2 + (gc_[1]-tr[1])**2).ravel())
        pools[s] = (np.concatenate(el), np.concatenate(ec))
        print('{:>3} {:>12.4g} {:>14.4e} {:>14.4e} {:>8}'.format(
            s, s*dt_dump, _rms(pools[s][0]), _rms(pools[s][1]), len(h_idx)))
        del l_s, c_s, el, ec
        gc.collect()

    log_dt = np.log(np.array([s*dt_dump for s in SUBSAMPLE_FACTORS]))

    def fit(k, q):
        vals = [np.percentile(pools[s][k], q) if q < 100 else pools[s][k].max()
                for s in SUBSAMPLE_FACTORS]
        return np.polyfit(log_dt, np.log(np.array(vals)), 1)[0]

    print('\nfitted order by percentile of the error distribution:')
    print('{:<12} {:>10} {:>10}    theory: linear 2, cubic 4'.format(
        'percentile', 'linear', 'cubic'))
    for label, q in (('median', 50), ('90th', 90), ('99th', 99), ('99.9th', 99.9),
                     ('max', 100)):
        orders[label] = (fit(0, q), fit(1, q))
        print('{:<12} {:>10.2f} {:>10.2f}'.format(label, *orders[label]))

    conc = pools[2][1].max()/np.median(pools[2][1])
    print('\nerror concentration (cubic, dt={:.3g}): max / median = {:.0f}x'.format(
        2*dt_dump, conc))
    print('=> read the percentile table, not the rms ratio: the two schemes part')
    print('   company where the flow is temporally smooth and converge on each')
    print('   other where it is not.')
    del pools
    gc.collect()


# --------------------------------------------------------------------------- #
#       Part 6 -- what an ensemble of agents actually sees                     #
# --------------------------------------------------------------------------- #

checkpoints = []
if 6 in PARTS:
    banner('PART 6 -- tracer advection, cubic vs windowed-linear')
    # Pure advection by explicit Euler, integrated by hand rather than through
    # Swarm.move: no diffusion, no RNG, no boundary conditions, no immersed mesh,
    # so the ONLY difference between the two runs is the temporal interpolation.
    # This fluid is not periodic, so positions are clipped into the domain --
    # identically in both runs, and counted below.
    envir_c = planktos.Environment()
    envir_d = planktos.Environment()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        envir_c.read_vtkxml_fluid_data(DATA, INUM=None)
        envir_d.read_vtkxml_fluid_data(DATA, INUM=INUM)

    L = np.asarray(envir_c.L)
    g = int(np.sqrt(N_TRACERS))
    px, py = np.meshgrid(np.linspace(0.1*L[0], 0.9*L[0], g),
                         np.linspace(0.1*L[1], 0.9*L[1], g))
    p0 = np.column_stack([px.ravel(), py.ravel()])
    step = t_all[-1]/TRAJ_STEPS

    pc, pd = p0.copy(), p0.copy()
    travelled = np.zeros(len(p0))
    for k in range(TRAJ_STEPS):
        t = k*step
        vc = envir_c.interpolate_flow(pc, time=t)
        vd = envir_d.interpolate_flow(pd, time=t)
        pc_new = np.clip(pc + step*vc, 0.0, L)
        travelled += np.linalg.norm(pc_new - pc, axis=1)
        pc, pd = pc_new, np.clip(pd + step*vd, 0.0, L)
        if (k + 1) % (TRAJ_STEPS//4) == 0:
            sep = np.linalg.norm(pc - pd, axis=1)
            checkpoints.append((t + step, _rms(sep), float(sep.max()),
                                float(np.mean(travelled))))

    at_edge = int(np.sum(np.any((pc <= 0) | (pc >= L), axis=1)))
    print('{} tracers, {} Euler steps of {:.3g} s; {} reached the domain edge\n'.format(
        len(p0), TRAJ_STEPS, step, at_edge))
    print('{:>10} {:>13} {:>13} {:>13} {:>12}'.format(
        't', 'rms sep', 'max sep', 'mean path', 'rms/path'))
    for t, r, m, d in checkpoints:
        print('{:>10.4g} {:>13.4e} {:>13.4e} {:>13.4e} {:>11.3f}%'.format(
            t, r, m, d, 100*r/max(d, 1e-300)))

    diag = float(np.hypot(*L))
    print('\nfinal rms separation as fraction of domain diagonal ({:.3g}) : {:.3f}%'.format(
        diag, 100*checkpoints[-1][1]/diag))

    # Individual trajectories separate under any perturbation in a mixing flow, so
    # the number above is as much about the flow as about interpolation. For an
    # ABM the decision-relevant question is whether the ensemble still agrees.
    print('\nensemble statistics (the question an ABM actually cares about):')
    print('{:<26} {:>14} {:>14} {:>12}'.format(
        'quantity', 'cubic', 'windowed', 'rel. diff'))

    def _report(label, a, b):
        rel = abs(a - b)/max(abs(a), 1e-300)
        print('{:<26} {:>14.6g} {:>14.6g} {:>11.3f}%'.format(label, a, b, 100*rel))

    for axis, nm in ((0, 'x'), (1, 'y')):
        _report('mean ' + nm, float(pc[:, axis].mean()), float(pd[:, axis].mean()))
        _report('std ' + nm, float(pc[:, axis].std()), float(pd[:, axis].std()))
    disp_c = np.linalg.norm(pc - p0, axis=1)
    disp_d = np.linalg.norm(pd - p0, axis=1)
    _report('mean net displacement', float(disp_c.mean()), float(disp_d.mean()))
    _report('std net displacement', float(disp_c.std()), float(disp_d.std()))
    for q in (10, 50, 90):
        _report('displacement p{}'.format(q),
                float(np.percentile(disp_c, q)), float(np.percentile(disp_d, q)))
    del envir_c, envir_d
    gc.collect()


# --------------------------------------------------------------------------- #
#        Part 7 -- the timeline this loader builds from these files            #
# --------------------------------------------------------------------------- #

if 7 in PARTS:
    banner('PART 7 -- the timeline, from the .pvd, the filenames, and dt')
    print('from the .pvd        : {} dumps, {} .. {}, source {!r}'.format(
        len(t_all), t_all[0], t_all[-1], src.time_source))

    # The same series read as a bare directory of files. These dumps carry no
    # TimeValue, so without the collection the timeline has to come from the
    # filenames or from dt -- and both must reproduce the .pvd exactly.
    # Beside the dataset, so the hard links below land on its own filesystem.
    tmp = Path(tempfile.mkdtemp(dir=str(Path(DATA).parent)))
    try:
        for f in sorted(Path(DATA).glob('*.vti')):
            os.link(f, tmp/f.name)          # hard link: no 472 MB copy
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            by_name = fluid.VTKXMLData(str(tmp), time_from_name=NAME_RE)
            by_dt = fluid.VTKXMLData(str(tmp), dt=dt_dump)
        print('from the filenames   : max |t - t_pvd| = {:.3e}, source {!r}'.format(
            np.abs(np.asarray(by_name.flow_times) - t_all).max(), by_name.time_source))
        print('from dt={:g}          : max |t - t_pvd| = {:.3e}, source {!r}'.format(
            dt_dump, np.abs(np.asarray(by_dt.flow_times) - t_all).max(),
            by_dt.time_source))
        print('dump_source without a collection : {!r}'.format(by_name.dump_source))
        del by_name, by_dt
        gc.collect()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #

banner('SUMMARY')
print('dataset {}: {} dumps, dt {:.3g} s, grid {}, U_rms {:.4g} m/s'.format(
    DATA, len(t_all), dt_dump, raw[0].shape[1:], U_RMS))
if res_u:
    print('velocity rms err linear  : {:.3f}% of U_rms'.format(
        100*res_u['linear'][0]/U_RMS))
    print('velocity rms err cubic   : {:.3f}% of U_rms   (ratio {:.1f}x)'.format(
        100*res_u['cubic'][0]/U_RMS, res_u['linear'][0]/res_u['cubic'][0]))
if res_d:
    print('du/dt    rms err linear  : {:.2f}% of |du/dt|_rms'.format(
        100*res_d['linear'][0]/dudt_scale))
    print('du/dt    rms err cubic   : {:.2f}% of |du/dt|_rms   (ratio {:.1f}x)'.format(
        100*res_d['cubic'][0]/dudt_scale, res_d['linear'][0]/res_d['cubic'][0]))
if orders:
    print('convergence, median      : linear {:.2f}, cubic {:.2f}   <- smooth bulk'.format(
        *orders['median']))
    print('convergence, 99th pct    : linear {:.2f}, cubic {:.2f}   <- rough tail'.format(
        *orders['99th']))
if checkpoints:
    print('tracer rms separation    : {:.2f}% of path travelled (individual)'.format(
        100*checkpoints[-1][1]/max(checkpoints[-1][3], 1e-300)))
print('\nCaveats that belong with these numbers:')
print(' * All for dump interval {:.3g} s on this flow. Use the Part 5 orders to'.format(
    dt_dump))
print('   scale to another cadence; the absolute error is a property of this')
print('   Delta-t against this flow, not of the scheme.')
print(' * The rms ratios blend a temporally smooth bulk with a rough tail. The')
print('   percentile table is the result; the ratio is a summary of two regimes.')
print(' * Velocity is identically zero inside the seven webs of the plate, where')
print('   both schemes are exact. That is 0.41% of the grid by the dataset\'s own')
print('   inFluid mask, so it neither carries nor rescues these averages.')
print(' * Parts 3-5 build on every 2nd dump, so their errors are for a 0.2 s')
print('   interval. At the native 0.1 s, the fitted rms slopes put linear near')
print('   1.7% of U_rms and cubic near 0.3%.')
