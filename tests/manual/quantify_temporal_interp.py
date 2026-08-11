#! /usr/bin/env python3
'''Quantify the cost of dynamic loading's linear-in-time interpolation.

This answers TODO.md Phase 1 (C): dynamic loading gives up cubic-in-time
interpolation for linear, and the size of that loss had only ever been eyeballed.

WHAT IS ACTUALLY BEING MEASURED
-------------------------------
Phase 1 (A) established that windowed-linear (INUM=k) reproduces full-linear
(INUM=True) to round-off: linear interpolation is local, depending only on the two
bracketing samples, so which window happens to be resident cannot change a value.
The dynamic-vs-cubic gap is therefore *entirely* a linear-vs-cubic question, with
the streaming machinery contributing nothing. Part 0 re-checks that reduction on
real data rather than inheriting it from the synthetic tests; Parts 1-3 then
measure the interpolation question directly.

Comparing linear(t) against cubic(t) would measure their *disagreement*, not
either one's error -- it cannot separate "cubic is more accurate" from "cubic is
different." So this uses withholding: both interpolants are built on a subsampled
set of dumps and evaluated at the dumps left out, whose raw values are exact
ground truth that neither scheme saw.

THE HEADLINE NUMBER IS NOT TRANSFERABLE ON ITS OWN
--------------------------------------------------
Interpolation error depends on the dump interval relative to the flow's own
timescales. A number from one dataset at one Delta-t says little about another.
That is what Part 3 is for: the convergence orders (expect ~2 for linear, ~4 for
cubic) are what let someone with a different dump interval extrapolate. Report
both, always.

AND A WHOLE-FIELD RMS RATIO HIDES THE ACTUAL RESULT
---------------------------------------------------
On real data the interpolation error is extraordinarily concentrated -- on
leaf_data the worst grid point is ~4 orders of magnitude above the median. So the
RMS-over-everything ratio is set by a small, temporally rough minority of the
field and says little about the bulk. Part 3 therefore fits the convergence order
per error *percentile*, which separates the two regimes that are being averaged
together:

  * where the flow is smooth in time, both schemes hit their theoretical orders
    and cubic is enormously better;
  * where the flow is temporally rough, neither converges at its nominal rate and
    the two nearly coincide -- the data, not the scheme, is the limit.

That structure, not a single ratio, is the answer to "what does dynamic loading
cost me." Note the rough regions are *not* the immersed-boundary neighbourhood
(checked: none of the worst 1% of points on leaf_data lie within 8 cells of the
mesh); peak error tracks local |d2u/dt2|, correlation ~0.7.

Run from the repository root:

    python tests/manual/quantify_temporal_interp.py

Requires real IB2d data, which is gitignored -- tests/data/leaf_data by default
(149 dumps; the dataset tests/manual/visualtest_2d.py uses). This lives in
tests/manual/ because of that data dependency; it is excluded from pytest
collection by collect_ignore in the root conftest.
'''

import sys
import time as timer
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
import planktos
from planktos import fluid


# ----------------------------- configuration ------------------------------- #

DATA = 'tests/data/leaf_data'
DT = 1.0e-5             # dt from input2d
PRINT_DUMP = 100        # print_dump from input2d
D_START = 1
INUM = 4                # window size for the dynamic checks

SUBSAMPLE_FACTORS = (2, 3, 4, 6)    # for the convergence study
N_TRACERS = 2000
TRAJ_STEPS = 120


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def banner(txt):
    print('\n' + '=' * 74)
    print(txt)
    print('=' * 74)


# --------------------------------------------------------------------------- #
#                       load once, reuse the raw arrays                        #
# --------------------------------------------------------------------------- #

banner('LOADING')
t0 = timer.perf_counter()
src = fluid.IB2dData(DATA, dt=DT, print_dump=PRINT_DUMP, d_start=D_START, INUM=True)
raw = [np.asarray(c) for c in src.get_raw_loaded_data()]
t_all = np.asarray(src.flow_times)
fpoints = src.flow_points
print('loaded {} dumps in {:.1f}s; grid {}, domain {}'.format(
    len(t_all), timer.perf_counter() - t0, raw[0].shape[1:], np.round(src.L, 5)))

dt_dump = t_all[1] - t_all[0]
speed = np.sqrt(raw[0] ** 2 + raw[1] ** 2)
U_RMS, U_MAX = _rms(speed), float(speed.max())
print('dump interval {:.3g} s over {:.4g} s; speed rms {:.4g}, max {:.4g}'.format(
    dt_dump, t_all[-1], U_RMS, U_MAX))


def build(idx, INUM_):
    '''An interpolant over the dumps at the given indices.'''
    return fluid.FluidData([raw[0][idx].copy(), raw[1][idx].copy()],
                           fpoints, t_all[idx].copy(), INUM=INUM_)


def withheld(coarse):
    '''The dumps left out, EXCLUDING any past the last build point.

    FluidData clamps outside its time range, so a withheld dump beyond the last
    coarse one is not interpolated by either scheme -- both return the last build
    point's raw values, identically. Scoring that as interpolation error inflates
    both by the same amount, which flatters linear, and its error does not scale
    with the dump interval at all, so it corrupts the convergence fit. Affects
    only the subsample factors that do not divide the series evenly: here s=3
    (1 sample of 99) and s=6 (4 of 124); s=2 and s=4 were always clean, so the
    Part 1 headline numbers never depended on this.''' + '''
    Found while building the 3D counterpart (tests/manual/vet_dynamic_loading_3d.py),
    where 12 dumps made it 1 sample in 6 and it moved the ratio from 1.9x to 8.4x.
    '''
    held = np.setdiff1d(np.arange(len(t_all)), coarse)
    return held[held < coarse[-1]]


# --------------------------------------------------------------------------- #
#      Part 0 -- the reduction: windowed-linear == full-linear on real data    #
# --------------------------------------------------------------------------- #

banner('PART 0 -- windowed (INUM={}) vs full linear (INUM=True), real data'.format(INUM))
dyn = fluid.IB2dData(DATA, dt=DT, print_dump=PRINT_DUMP, d_start=D_START, INUM=INUM)
full_lin = build(slice(None), True)
worst = 0.0
for q in np.linspace(t_all[0], t_all[-1], 97):
    worst = max(worst, max(np.abs(a - b).max() for a, b in zip(dyn(q), full_lin(q))))
print('max abs difference over 97 query times : {:.3e}'.format(worst))
print('relative to rms speed                  : {:.3e}'.format(worst / U_RMS))
print('=> windowing contributes nothing; the rest of this is linear vs cubic.')


# --------------------------------------------------------------------------- #
#          Part 1 -- velocity error against withheld ground truth              #
# --------------------------------------------------------------------------- #

banner('PART 1 -- velocity error vs withheld dumps (built on every 2nd dump)')
coarse = np.arange(0, len(t_all), 2)
held = withheld(coarse)
lin, cub = build(coarse, True), build(coarse, None)
print('built on {} dumps (spacing {:.3g} s); tested at {} withheld dumps'.format(
    len(coarse), 2 * dt_dump, len(held)))

err = {'linear': [], 'cubic': []}
for n in held:
    truth = (raw[0][n], raw[1][n])
    for name, obj in (('linear', lin), ('cubic', cub)):
        got = obj(t_all[n])
        err[name].append(np.sqrt((got[0] - truth[0]) ** 2 + (got[1] - truth[1]) ** 2))

print('\n{:<8} {:>12} {:>12} {:>12} {:>12}'.format(
    'scheme', 'rms err', 'max err', 'rms/U_rms', 'max/U_max'))
res_u = {}
for name in ('linear', 'cubic'):
    e = np.asarray(err[name])
    res_u[name] = (_rms(e), float(e.max()))
    print('{:<8} {:>12.4e} {:>12.4e} {:>11.3f}% {:>11.3f}%'.format(
        name, _rms(e), e.max(), 100 * _rms(e) / U_RMS, 100 * e.max() / U_MAX))
print('\nlinear/cubic rms error ratio : {:.1f}x'.format(
    res_u['linear'][0] / res_u['cubic'][0]))


# --------------------------------------------------------------------------- #
#       Part 2 -- du/dt error, the term that reaches the physics               #
# --------------------------------------------------------------------------- #

banner('PART 2 -- du/dt error (feeds get_dudt -> material derivative -> inertial)')
# Reference: 4th-order central difference on the FULL-resolution series, which is
# O(dt^4) on a grid twice as fine as either interpolant was built on. Stated
# explicitly because the choice of reference is a real methodological decision --
# using the fine cubic's own derivative would flatter the cubic scheme.
def dudt_ref(n):
    return [(-c[n + 2] + 8 * c[n + 1] - 8 * c[n - 1] + c[n - 2]) / (12 * dt_dump)
            for c in raw]

interior = held[(held >= 2) & (held <= len(t_all) - 3)]
derr = {'linear': [], 'cubic': []}
for n in interior:
    ref = dudt_ref(n)
    for name, obj in (('linear', lin), ('cubic', cub)):
        got = obj.get_dudt(time=t_all[n])
        derr[name].append(np.sqrt((got[0] - ref[0]) ** 2 + (got[1] - ref[1]) ** 2))

dudt_scale = _rms([dudt_ref(n) for n in interior[::10]])
print('reference |du/dt| rms scale : {:.4g}'.format(dudt_scale))
print('\n{:<8} {:>12} {:>12} {:>14}'.format('scheme', 'rms err', 'max err', 'rms/scale'))
res_d = {}
for name in ('linear', 'cubic'):
    e = np.asarray(derr[name])
    res_d[name] = (_rms(e), float(e.max()))
    print('{:<8} {:>12.4e} {:>12.4e} {:>13.2f}%'.format(
        name, _rms(e), e.max(), 100 * _rms(e) / dudt_scale))
print('\nlinear/cubic rms du/dt error ratio : {:.1f}x'.format(
    res_d['linear'][0] / res_d['cubic'][0]))


# --------------------------------------------------------------------------- #
#     Part 3 -- convergence order: what makes the number transferable          #
# --------------------------------------------------------------------------- #

banner('PART 3 -- convergence order in the dump interval, by error percentile')
pools = {}
print('{:>3} {:>12} {:>14} {:>14}'.format('s', 'dt', 'linear rms', 'cubic rms'))
for s in SUBSAMPLE_FACTORS:
    c_idx = np.arange(0, len(t_all), s)
    h_idx = withheld(c_idx)
    l_s, c_s = build(c_idx, True), build(c_idx, None)
    el, ec = [], []
    for n in h_idx:
        tr = (raw[0][n], raw[1][n])
        gl, gc = l_s(t_all[n]), c_s(t_all[n])
        el.append(np.sqrt((gl[0] - tr[0]) ** 2 + (gl[1] - tr[1]) ** 2).ravel())
        ec.append(np.sqrt((gc[0] - tr[0]) ** 2 + (gc[1] - tr[1]) ** 2).ravel())
    pools[s] = (np.concatenate(el), np.concatenate(ec))
    print('{:>3} {:>12.4g} {:>14.4e} {:>14.4e}'.format(
        s, s * dt_dump, _rms(pools[s][0]), _rms(pools[s][1])))

log_dt = np.log(np.array([s * dt_dump for s in SUBSAMPLE_FACTORS]))


def fit(k, q):
    vals = [np.percentile(pools[s][k], q) if q < 100 else pools[s][k].max()
            for s in SUBSAMPLE_FACTORS]
    return np.polyfit(log_dt, np.log(np.array(vals)), 1)[0]


print('\nfitted order by percentile of the error distribution:')
print('{:<12} {:>10} {:>10}    theory: linear 2, cubic 4'.format('percentile', 'linear', 'cubic'))
orders = {}
for label, q in (('median', 50), ('90th', 90), ('99th', 99), ('99.9th', 99.9), ('max', 100)):
    orders[label] = (fit(0, q), fit(1, q))
    print('{:<12} {:>10.2f} {:>10.2f}'.format(label, *orders[label]))

conc = pools[2][1].max() / np.median(pools[2][1])
print('\nerror concentration (cubic, dt={:.3g}): max / median = {:.0f}x'.format(
    2 * dt_dump, conc))
print('=> the rms ratio above is set by the rough tail, not the bulk. In the')
print('   smooth bulk cubic converges at ~{:.1f} and linear at ~{:.1f}, so cubic is'.format(
    orders['median'][1], orders['median'][0]))
print('   decisively better there; in the tail both stall and the gap closes.')


# --------------------------------------------------------------------------- #
#      Part 4 -- trajectory divergence: the number a user decides on           #
# --------------------------------------------------------------------------- #

banner('PART 4 -- tracer trajectory divergence, cubic vs dynamic-linear')
# Pure advection by explicit Euler, integrated by hand rather than through
# Swarm.move: no diffusion, no RNG, no boundary conditions, no immersed mesh, so
# the ONLY difference between the two runs is the temporal interpolation. The
# integrator's own truncation error is common to both and largely cancels in the
# difference. Positions are left unwrapped; interpolate_flow wraps internally for
# the periodic fluid, so trajectories stay continuous and distances are honest.
envir_c = planktos.Environment()
envir_c.read_IB2d_fluid_data(DATA, DT, PRINT_DUMP, d_start=D_START, INUM=None)
envir_d = planktos.Environment()
envir_d.read_IB2d_fluid_data(DATA, DT, PRINT_DUMP, d_start=D_START, INUM=INUM)

g = int(np.sqrt(N_TRACERS))
px, py = np.meshgrid(np.linspace(0.02, 0.18, g), np.linspace(0.03, 0.27, g))
p0 = np.column_stack([px.ravel(), py.ravel()])
step = t_all[-1] / TRAJ_STEPS

pc, pd = p0.copy(), p0.copy()
travelled = np.zeros(len(p0))
checkpoints = []
for k in range(TRAJ_STEPS):
    t = k * step
    vc = envir_c.interpolate_flow(pc, time=t)
    vd = envir_d.interpolate_flow(pd, time=t)
    pc_new = pc + step * vc
    travelled += np.linalg.norm(pc_new - pc, axis=1)
    pc, pd = pc_new, pd + step * vd
    if (k + 1) % (TRAJ_STEPS // 4) == 0:
        sep = np.linalg.norm(pc - pd, axis=1)
        checkpoints.append((t + step, _rms(sep), float(sep.max()),
                            float(np.mean(travelled))))

print('{} tracers, {} Euler steps of {:.3g} s\n'.format(len(p0), TRAJ_STEPS, step))
print('{:>10} {:>13} {:>13} {:>13} {:>12}'.format(
    't', 'rms sep', 'max sep', 'mean path', 'rms/path'))
for t, r, m, d in checkpoints:
    print('{:>10.4g} {:>13.4e} {:>13.4e} {:>13.4e} {:>11.3f}%'.format(
        t, r, m, d, 100 * r / d))

diag = float(np.hypot(*src.L))
print('\nfinal rms separation as fraction of domain diagonal ({:.3g}) : {:.3f}%'.format(
    diag, 100 * checkpoints[-1][1] / diag))

# Individual trajectories in a mixing flow separate exponentially under ANY
# perturbation, so the number above is as much a statement about the flow's
# Lyapunov growth as about interpolation, and it saturates once separations reach
# the scale of the flow features. For an agent-based model the decision-relevant
# question is whether the *ensemble* still agrees -- dispersal statistics, not
# agent identity. Compare distributions rather than pairs.
print('\nensemble statistics (the question an ABM actually cares about):')
print('{:<26} {:>14} {:>14} {:>12}'.format('quantity', 'cubic', 'dynamic', 'rel. diff'))


def _report(label, a, b):
    rel = abs(a - b) / max(abs(a), 1e-300)
    print('{:<26} {:>14.6g} {:>14.6g} {:>11.3f}%'.format(label, a, b, 100 * rel))


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

banner('SUMMARY')
print('windowing vs full linear       : {:.2e} ({:.1e} of U_rms) -- inert'.format(
    worst, worst / U_RMS))
print('velocity  rms err  linear      : {:.3f}% of U_rms'.format(100 * res_u['linear'][0] / U_RMS))
print('velocity  rms err  cubic       : {:.3f}% of U_rms   (ratio {:.1f}x)'.format(
    100 * res_u['cubic'][0] / U_RMS, res_u['linear'][0] / res_u['cubic'][0]))
print('du/dt     rms err  linear      : {:.2f}% of |du/dt|_rms'.format(100 * res_d['linear'][0] / dudt_scale))
print('du/dt     rms err  cubic       : {:.2f}% of |du/dt|_rms   (ratio {:.1f}x)'.format(
    100 * res_d['cubic'][0] / dudt_scale, res_d['linear'][0] / res_d['cubic'][0]))
print('convergence order, median      : linear {:.2f}, cubic {:.2f}   <- smooth bulk'.format(
    *orders['median']))
print('convergence order, 99th pct    : linear {:.2f}, cubic {:.2f}   <- rough tail'.format(
    *orders['99th']))
print('tracer rms separation          : {:.2f}% of path travelled (individual)'.format(
    100 * checkpoints[-1][1] / checkpoints[-1][3]))
print('\nCaveats that belong with these numbers:')
print(' * All for dump interval {:.3g} s on {}. Use the Part 3'.format(dt_dump, DATA))
print('   orders to scale to a different dump cadence -- the absolute error is a')
print('   property of this Delta-t relative to this flow, not of the scheme.')
print(' * The rms ratios blend two regimes. Read the percentile table, not the')
print('   ratio: cubic earns its order where the flow is temporally smooth and')
print('   earns nothing where it is not.')
print(' * Trajectory separation is dominated by the flow separating nearby')
print('   particles exponentially, which happens under any perturbation. The')
print('   ensemble comparison is the decision-relevant one.')
