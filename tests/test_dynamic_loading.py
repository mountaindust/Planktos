'''Pins the dynamic (windowed) fluid-loading machinery -- TODO.md Phase 1 (A),
(B) and (D), plus the per-dump mean cache that plotting reads instead of the
field (flow_field_interface.md §8.3.1).

`FluidData.update_spline` is the workhorse of this branch: it responds to a
request for a time outside the currently loaded window by loading the next or
previous set of dumps and re-splining. Until this module it had **no automated
coverage at all** -- `INUM` appeared in the suite only as None (cubic, all in
memory) or True (linear, all in memory), never as an int, which is the setting
that actually slides a window.

Phase 1 was written as needing real IB2d data, but the slider itself does not:
what it requires of a source is only a `load_dumpfiles(d_start, d_finish)` that
returns a list of per-component ndarrays with a leading time axis. The synthetic
subclass below supplies that from an in-memory array, so the real `update_spline`
is exercised exactly and deterministically with nothing on disk. Ingestion (the
vtk readers) and Phase 1 (C) -- the quantitative windowed-linear vs. full-cubic
error -- still need real data and are out of scope here.

The key property in (A) is that linear interpolation in time is *local*: it
depends only on the two bracketing samples, so which window happens to be
resident cannot change the answer. Windowed-linear must therefore agree with
full-linear to round-off at every query time. The test field is deliberately
non-linear in time, so that agreement is a real check on the slider rather than
an artifact of both schemes being exact.

Agreement is to a few ulp rather than bit-for-bit because the slider carries
window-boundary values by *evaluating* the outgoing spline rather than re-reading
raw data. That costs an ulp per slide; it does not compound, which is itself
pinned below (test_holdover_roundoff_does_not_accumulate) since a long 3D run
makes thousands of slides.

No RNG and no external data anywhere. The (A)/(B)/(D) and mean-cache sections
touch no files at all; the two closing sections drive the real loaders --
VTK3dData and IB2dData --
against small committed fixtures (tests/fixtures/vtk3d_min and
tests/fixtures/ib2d_fluid_min, regenerate with tests/fixtures/_gen_fixtures.py),
because what those sections are about is the timeline a loader builds from files.
'''

import os
from pathlib import Path

import numpy as np
import pytest

import planktos
from planktos import fluid

FIXTURES = Path(__file__).parent / 'fixtures'


# --------------------------------------------------------------------------- #
#                            synthetic dynamic source                          #
# --------------------------------------------------------------------------- #

class _InMemorySource(fluid.FluidData):
    '''A FluidData source whose "disk" is an in-memory array.

    Mirrors the structure IB2dData/VTK3dData use: record the dump bounds, load
    the opening window, then hand it to FluidData to spline. Dump numbers are
    chosen to equal time indices so the two index spaces coincide and assertions
    stay readable. Every load is recorded in `load_calls` so tests can assert the
    slider is not thrashing.
    '''

    def __init__(self, full_flow, flow_points, flow_times, INUM):
        self._full = [np.asarray(f) for f in full_flow]
        self.load_calls = []
        self.d_start = 0
        self.d_finish = len(flow_times) - 1
        self.loaded_dump_bnds = (0, INUM)
        self.loaded_idx_bnds = (0, INUM)
        window = self.load_dumpfiles(0, INUM)
        super().__init__(window, flow_points, flow_times, INUM)

    def load_dumpfiles(self, d_start, d_finish):
        self.load_calls.append((d_start, d_finish))
        return [f[d_start:d_finish + 1].copy() for f in self._full]


def _field_2d(nt=21, nx=6, ny=5, t_end=4.0):
    '''u(t,x,y) = sin(t)*x, v(t,x,y) = t**2*y -- deliberately NOT linear in t, so
    windowed-linear agreeing with full-linear is a statement about the slider and
    not about both schemes being trivially exact.'''
    t = np.linspace(0.0, t_end, nt)
    x = np.linspace(0.0, 10.0, nx)
    y = np.linspace(0.0, 8.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.stack([np.sin(tt) * X for tt in t])
    v = np.stack([tt ** 2 * Y for tt in t])
    return t, (x, y), [u, v]


def _field_3d(nt=13, n=4, t_end=3.0):
    '''3D counterpart. The slider is dimension-agnostic (it only ever touches the
    leading time axis); this pins that, cheaply, ahead of Phase 2.'''
    t = np.linspace(0.0, t_end, nt)
    g = np.linspace(0.0, 6.0, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    u = np.stack([np.sin(tt) * X for tt in t])
    v = np.stack([tt ** 2 * Y for tt in t])
    w = np.stack([tt * Z for tt in t])
    return t, (g, g, g), [u, v, w]


def _pair(field, INUM):
    '''Build a windowed source and a full-linear reference over the same data.'''
    t, fpoints, comps = field
    dyn = _InMemorySource([c.copy() for c in comps], fpoints, t.copy(), INUM)
    full = fluid.FluidData([c.copy() for c in comps], fpoints, t.copy(), INUM=True)
    return dyn, full, t


def _diff(got, ref):
    '''Max absolute difference between two lists of components.'''
    return max(np.abs(g - r).max() for g, r in zip(got, ref))


def _tol(ref):
    '''A few ulp of the reference field -- see the module docstring on why
    agreement is to round-off rather than bit-for-bit.'''
    return 8 * np.finfo(float).eps * max(1.0, max(np.abs(r).max() for r in ref))


_PKG_DIR = os.path.abspath(os.path.dirname(planktos.__file__))


def _from_planktos(caught):
    '''Keep only the warnings raised from inside the planktos package.

    Anchored to the package directory rather than matching the string
    "planktos" in the path, since the repository directory may differ from the
    package directory only by case.
    '''
    return [w for w in caught
            if os.path.abspath(str(w.filename)).startswith(_PKG_DIR)]


# --------------------------------------------------------------------------- #
#            (A) machinery correctness -- windowed == full, to round-off       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('INUM', [4, 5, 7])
def test_windowed_matches_full_linear_forward_sweep(INUM):
    dyn, full, t = _pair(_field_2d(), INUM)
    for q in np.linspace(t[0], t[-1], 97):
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


@pytest.mark.parametrize('INUM', [4, 5, 7])
def test_windowed_matches_full_linear_backward_sweep(INUM):
    dyn, full, t = _pair(_field_2d(), INUM)
    for q in np.linspace(t[-1], t[0], 97):
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_windowed_matches_full_linear_random_access():
    # Non-monotone access forces repeated slides in both directions, including
    # slides that skip several windows at once. Fixed sequence, no RNG.
    dyn, full, t = _pair(_field_2d(), 5)
    for q in [0.0, 3.9, 0.1, 2.0, 4.0, 1.05, 3.3, 0.55, 2.7, 0.0, 4.0]:
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_windowed_matches_full_linear_on_node_times():
    # Query exactly at the data timestamps, where the two bracketing samples
    # degenerate -- the case most likely to expose an off-by-one in the window.
    dyn, full, t = _pair(_field_2d(), 5)
    for q in t:
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_windowed_matches_full_linear_3d():
    dyn, full, t = _pair(_field_3d(), 5)
    assert dyn.ndim == 3
    for q in np.linspace(t[0], t[-1], 41):
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_windowed_constant_extrapolation_beyond_data_bounds():
    # FluidData.__call__ clamps to the endpoint values outside [t0, tN].
    dyn, full, t = _pair(_field_2d(), 5)
    lo, hi = full(t[0]), full(t[-1])
    for q in (-1.0, t[0] - 1e-9):
        assert _diff(dyn(q), lo) <= _tol(lo)
    for q in (t[-1] + 1e-9, 99.0):
        assert _diff(dyn(q), hi) <= _tol(hi)


def test_holdover_roundoff_does_not_accumulate():
    # The slider carries window-boundary values by evaluating the outgoing spline
    # (fluid.py, both update_spline branches) rather than re-reading raw data,
    # which costs an ulp per slide. If that error compounded, a long 3D run --
    # thousands of slides -- would drift away from the truth. It does not: the
    # error after hundreds of loads is the same order as after one. Linear
    # accumulation over this many slides would be ~100x the single-slide error
    # and would fail the bound below comfortably.
    t, fpoints, comps = _field_2d(nt=201, t_end=20.0)
    dyn = _InMemorySource([c.copy() for c in comps], fpoints, t.copy(), 4)
    full = fluid.FluidData([c.copy() for c in comps], fpoints, t.copy(), INUM=True)

    worst_per_sweep = []
    for sweep in range(4):
        qs = np.linspace(t[0], t[-1], 201)
        if sweep % 2:
            qs = qs[::-1]                      # alternate direction each pass
        worst_per_sweep.append(max(_diff(dyn(q), full(q)) for q in qs))

    assert len(dyn.load_calls) > 100           # we really did slide a lot
    assert max(worst_per_sweep) <= 32 * np.finfo(float).eps


# --------------------------------------------------------------------------- #
#                       (B) window-sliding behavior                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('INUM', [4, 5, 7])
def test_window_stays_bounded_across_a_full_sweep(INUM):
    # The whole point of dynamic loading: memory must not grow with the dataset.
    # `<=` rather than `==` because the final window is whatever remains of the
    # dataset and can be shorter -- see test_forward_slide_walks_to_the_end.
    dyn, _, t = _pair(_field_2d(nt=41), INUM)
    for q in np.linspace(t[0], t[-1], 120):
        dyn(q)
        for spline in dyn._flow:
            assert len(spline.x) <= INUM + 1
            assert spline.shape[0] == len(spline.x)


def test_forward_slide_walks_to_the_end_of_the_dataset():
    dyn, _, t = _pair(_field_2d(), 5)
    assert dyn.loaded_idx_bnds == (0, 5)
    assert dyn.loaded_dump_bnds == (0, 5)
    dyn(t[-1])
    assert dyn._flow[0].x[-1] == t[-1]                 # the end really is loaded
    assert len(dyn._flow[0].x) <= 6

    # Both bound tuples are inclusive, and this source numbers dumps to match
    # time indices, so they must agree -- including in the end-of-dataset branch,
    # which used to set idx_finish to len(flow_times), one past the last valid
    # index. That was latent rather than live (nothing reads idx_bnds[1] once the
    # end is reached, since the forward slide is gated off by extrapolate[1]) and
    # the window was unaffected, because the slice that builds it clips. Fixed,
    # and locked here so the two index spaces cannot silently drift apart again.
    assert dyn.loaded_dump_bnds[1] == len(t) - 1
    assert dyn.loaded_idx_bnds[1] == len(t) - 1
    assert dyn.loaded_idx_bnds == dyn.loaded_dump_bnds


@pytest.mark.parametrize('INUM', [4, 5, 7])
def test_index_spaces_agree_at_every_slide(INUM):
    # loaded_dump_bnds and loaded_idx_bnds are parallel inclusive descriptions of
    # the same window. This source numbers dumps to match time indices, so they
    # must be equal at all times, and both must describe the window the spline
    # actually holds. Sweep both directions so every branch of update_spline runs.
    dyn, _, t = _pair(_field_2d(nt=41), INUM)
    qs = list(np.linspace(t[0], t[-1], 100)) + list(np.linspace(t[-1], t[0], 100))
    for q in qs:
        dyn(q)
        lo, hi = dyn.loaded_idx_bnds
        assert dyn.loaded_idx_bnds == dyn.loaded_dump_bnds
        assert 0 <= lo <= hi <= len(t) - 1
        assert len(dyn._flow[0].x) == hi - lo + 1
        assert dyn._flow[0].x[0] == t[lo]
        assert dyn._flow[0].x[-1] == t[hi]


def test_backward_slide_returns_to_the_opening_window():
    dyn, _, t = _pair(_field_2d(), 5)
    dyn(t[-1])
    assert dyn.loaded_idx_bnds[0] > 0
    dyn(t[0])
    assert dyn.loaded_idx_bnds == (0, 5)


def test_jump_to_beginning_fast_path():
    # update_spline has a dedicated branch for `time <= flow_times[INUM]` that
    # reloads the opening window outright instead of sliding back one step at a
    # time. Reach it from the far end of the dataset in a single call.
    dyn, full, t = _pair(_field_2d(), 5)
    dyn(t[-1])
    n_loads_before = len(dyn.load_calls)
    dyn(t[1])
    assert dyn.loaded_idx_bnds == (0, 5)
    assert len(dyn.load_calls) - n_loads_before == 1     # one load, not a walk
    ref = full(t[1])
    assert _diff(dyn(t[1]), ref) <= _tol(ref)


def test_extrapolate_flags_track_position_in_the_dataset():
    # (left, right): open at the start, closed in the middle, open at the end.
    dyn, _, t = _pair(_field_2d(), 5)
    assert dyn._flow[0].extrapolate == (True, False)
    dyn(t[len(t) // 2])
    assert dyn._flow[0].extrapolate == (False, False)
    dyn(t[-1])
    assert dyn._flow[0].extrapolate == (False, True)


def test_slide_count_is_bounded_over_a_monotone_sweep():
    # A monotone walk must not reload the same window repeatedly. With 41 times
    # and a 5-interval window, ceil(35/5) = 7 forward slides suffice; allow the
    # opening load plus a small margin, but fail loudly on thrashing.
    dyn, _, t = _pair(_field_2d(nt=41), 5)
    for q in np.linspace(t[0], t[-1], 200):
        dyn(q)
    assert len(dyn.load_calls) <= 10


def test_no_load_when_query_stays_inside_the_window():
    dyn, _, t = _pair(_field_2d(), 5)
    n_loads = len(dyn.load_calls)
    for q in np.linspace(t[0], t[5], 25):
        dyn(q)
    assert len(dyn.load_calls) == n_loads


def test_fmin_fmax_stay_tuples_and_widen_across_slides():
    # Regression lock for the generator bug (flow_field_interface.md §3.3) on the
    # path where it actually bit: update_spline subscripts self.fmin on every
    # slide, which raised TypeError when it was a generator expression.
    dyn, _, t = _pair(_field_2d(), 5)
    assert isinstance(dyn.fmin, tuple) and isinstance(dyn.fmax, tuple)
    open_fmax = dyn.fmax
    dyn(t[-1])
    assert isinstance(dyn.fmin, tuple) and isinstance(dyn.fmax, tuple)
    # v = t**2*y grows monotonically in t, so sweeping to the end must widen fmax
    assert dyn.fmax[1] > open_fmax[1]
    # fmax is a running maximum over data seen, so it never shrinks
    assert all(new >= old for new, old in zip(dyn.fmax, open_fmax))
    # unpacking twice must work (the plotting failure mode)
    a, b = dyn.fmax
    c, d = dyn.fmax
    assert (a, b) == (c, d)


def test_get_raw_loaded_data_returns_window_ndarrays():
    # §3.5 regression, on a genuinely *sliding* window rather than INUM=True.
    dyn, _, t = _pair(_field_2d(), 5)
    _, _, comps = _field_2d()
    dyn(t[9])                                   # land mid-dataset
    lo, hi = dyn.loaded_idx_bnds
    raw = dyn.get_raw_loaded_data()
    assert len(raw) == 2
    for comp in raw:
        assert isinstance(comp, np.ndarray)
        assert comp.shape[0] == hi - lo + 1
        assert comp.shape[0] < len(t)           # the window, not the whole dataset
    expected = comps[0][lo:hi + 1]
    assert np.allclose(raw[0], expected, rtol=0, atol=_tol([expected]))


def test_indexing_a_dynamically_loaded_fluiddata_raises():
    # Container-style indexing is meaningless under dynamic loading, since the
    # time index would refer to a shifting window. FluidData refuses it.
    dyn, _, _ = _pair(_field_2d(), 5)
    with pytest.raises(TypeError):
        dyn[0]


def test_not_enough_data_for_the_requested_window_raises():
    t, fpoints, comps = _field_2d(nt=5)
    with pytest.raises(RuntimeError):
        _InMemorySource([c.copy() for c in comps], fpoints, t.copy(), INUM=5)


# --------------------------------------------------------------------------- #
#                  (D) get_dudt under linear splining                          #
# --------------------------------------------------------------------------- #

def test_dudt_is_the_finite_difference_on_each_interval():
    dyn, _, t = _pair(_field_2d(), 5)
    _, _, comps = _field_2d()
    for i in (0, 3, 9, 17):
        dt = t[i + 1] - t[i]
        expected = (comps[0][i + 1] - comps[0][i]) / dt
        got = dyn.get_dudt(time=0.5 * (t[i] + t[i + 1]))
        assert np.allclose(got[0], expected)


def test_dudt_interval_convention_is_right_closed():
    # A query landing exactly on a timestamp takes the slope of the interval to
    # its LEFT -- i.e. du/dt is constant on (t[i-1], t[i]], not [t[i], t[i+1]).
    # t0 is the documented exception, having no interval on its left. The choice
    # is arbitrary (the true derivative does not exist at a breakpoint) but it is
    # observable, so pin it.
    dyn, _, t = _pair(_field_2d(), 5)
    _, _, comps = _field_2d()
    i = 6
    left = (comps[0][i] - comps[0][i - 1]) / (t[i] - t[i - 1])
    right = (comps[0][i + 1] - comps[0][i]) / (t[i + 1] - t[i])
    assert not np.allclose(left, right)              # the intervals really differ

    assert np.allclose(dyn.get_dudt(time=t[i])[0], left)
    assert np.allclose(dyn.get_dudt(time=t[i] + 1e-9)[0], right)

    first = (comps[0][1] - comps[0][0]) / (t[1] - t[0])
    assert np.allclose(dyn.get_dudt(time=t[0])[0], first)


def test_dudt_is_piecewise_constant_within_an_interval():
    # Linear-in-time interpolation makes du/dt a step function -- constant on
    # each interval and discontinuous at the breakpoints. This is the documented,
    # deliberate cost of dynamic loading; pin it so a future change to the
    # temporal scheme has to be explicit about breaking it.
    dyn, _, t = _pair(_field_2d(), 5)
    i = 6
    inside = np.linspace(t[i], t[i + 1], 8)[1:]      # (t[i], t[i+1]] -- see above
    ref = dyn.get_dudt(time=inside[0])
    for q in inside[1:]:
        assert _diff(dyn.get_dudt(time=q), ref) == 0.0


def test_dudt_jumps_at_a_breakpoint():
    dyn, _, t = _pair(_field_2d(), 5)
    i = 6
    at = dyn.get_dudt(time=t[i])                     # slope of (t[i-1], t[i]]
    after = dyn.get_dudt(time=t[i] + 1e-9)           # slope of (t[i], t[i+1]]
    # sin(t) has non-constant slope, so consecutive intervals differ
    assert _diff(at, after) > 1e-3


def test_dudt_matches_full_linear_across_slides():
    dyn, full, t = _pair(_field_2d(), 5)
    for q in np.linspace(t[0] + 1e-3, t[-1] - 1e-3, 61):
        ref = full.get_dudt(time=q)
        assert _diff(dyn.get_dudt(time=q), ref) <= _tol(ref)


def test_dudt_is_zero_beyond_the_data_bounds():
    # Velocity is held constant outside [t0, tN], so its time derivative is zero.
    dyn, _, t = _pair(_field_2d(), 5)
    for q in (t[0] - 0.5, t[-1] + 0.5):
        for comp in dyn.get_dudt(time=q):
            assert np.all(comp == 0.0)


def test_dudt_3d_matches_full_linear():
    dyn, full, t = _pair(_field_3d(), 5)
    for q in np.linspace(t[0] + 1e-3, t[-1] - 1e-3, 31):
        ref = full.get_dudt(time=q)
        assert _diff(dyn.get_dudt(time=q), ref) <= _tol(ref)


# --------------------------------------------------------------------------- #
#        per-dump mean cache under a sliding window (plotting, §8.3.1)         #
# --------------------------------------------------------------------------- #
# Plot frames need the spatial mean of each velocity component. Computing it
# from the field would make plotting re-stream the whole dataset a second time,
# so FluidData caches the mean of each dump as that dump loads and evaluates it
# with the same interpolation weights the field uses. The mean is linear and the
# interpolation is a weighted sum of nodal fields, so this is exact, not an
# approximation (§8.5).
#
# The field here is not linear in time, so agreement with the field's own mean is
# a statement about that identity rather than about both being trivially exact.

def test_dump_means_match_the_field_across_slides():
    dyn, _, t = _pair(_field_2d(), 5)
    for q in np.linspace(t[0], t[-1], 37):
        ref = tuple(f.mean() for f in dyn(q))
        got = dyn.get_mean_velocity(time=q)
        assert np.allclose(got, ref, rtol=0, atol=1e-12)


def test_dump_means_match_full_linear():
    # The windowed cache must not depend on which window happened to be resident
    # -- the same property (A) pins for the field itself.
    dyn, full, t = _pair(_field_2d(), 5)
    for q in np.linspace(t[0], t[-1], 37):
        assert np.allclose(dyn.get_mean_velocity(time=q),
                           full.get_mean_velocity(time=q), rtol=0, atol=1e-12)


def test_dump_means_survive_the_window_moving_on():
    # This is the property the whole design rests on: after a run has swept
    # forward, replaying it (which is what plot_all does) must cost no loads at
    # all, even though the resident window is now at the far end of the dataset.
    dyn, full, t = _pair(_field_2d(), 5)
    for q in np.linspace(t[0], t[-1], 37):
        dyn(q)
    assert not np.isnan(dyn._dump_means).any(), 'a swept dump went unrecorded'

    n_loads = len(dyn.load_calls)
    for q in np.linspace(t[-1], t[0], 37):          # replay, backward
        assert np.allclose(dyn.get_mean_velocity(time=q),
                           full.get_mean_velocity(time=q), rtol=0, atol=1e-12)
    assert len(dyn.load_calls) == n_loads, 'replaying the means re-streamed data'


def test_dump_means_load_on_demand_when_never_seen():
    # Falling back to a load is the correct answer for a time whose dumps have
    # never been resident (a cache miss, not a cache lie).
    dyn, full, t = _pair(_field_2d(), 5)
    n_loads = len(dyn.load_calls)
    q = t[-2]                                       # far outside the opening window
    assert np.allclose(dyn.get_mean_velocity(time=q),
                       full.get_mean_velocity(time=q), rtol=0, atol=1e-12)
    assert len(dyn.load_calls) > n_loads


def test_dump_means_recorded_on_the_jump_to_start_path():
    # The backward slide has a separate fast path that reloads the opening window
    # outright; it must record means too, or a backward replay silently misses.
    dyn, full, t = _pair(_field_2d(), 5)
    dyn(t[-1])                                      # sweep to the end
    dyn(t[0])                                       # jump back to the beginning
    assert dyn.loaded_idx_bnds == (0, 5)
    for q in (t[0], t[1], t[3] + 0.01):
        assert np.allclose(dyn.get_mean_velocity(time=q),
                           full.get_mean_velocity(time=q), rtol=0, atol=1e-12)


def test_dump_means_3d():
    dyn, full, t = _pair(_field_3d(), 4)
    for q in np.linspace(t[0], t[-1], 23):
        got = dyn.get_mean_velocity(time=q)
        assert len(got) == 3
        assert np.allclose(got, full.get_mean_velocity(time=q),
                           rtol=0, atol=1e-12)


def test_dump_means_constant_extrapolation_beyond_data_bounds():
    dyn, _, t = _pair(_field_2d(), 5)
    assert np.allclose(dyn.get_mean_velocity(time=t[0] - 5),
                       dyn.get_mean_velocity(time=t[0]))
    assert np.allclose(dyn.get_mean_velocity(time=t[-1] + 5),
                       dyn.get_mean_velocity(time=t[-1]))


# --------------------------------------------------------------------------- #
#          VTK3dData end-to-end -- the windowed path against real files        #
# --------------------------------------------------------------------------- #
# Everything above drives update_spline through a synthetic in-memory source.
# These exercise the real 3D loader against the committed vtk fixture
# (tests/fixtures/vtk3d_min, 8 dumps, regenerate with _gen_fixtures.py), which is
# where the timeline actually comes from.
#
# The fixture field is u = t, v = x, w = t*z, with TIME = 0..7 in field data. u is
# uniform in space and linear in t, so u(t) reads back the simulation time
# directly -- which is what makes a frozen or truncated timeline obvious.

VTK3D = str(FIXTURES / 'vtk3d_min')


def _vtk3d(INUM):
    return fluid.VTK3dData(VTK3D, title='IBAMR_db_', d_start=0, INUM=INUM)


@pytest.mark.parametrize('INUM', [None, True, 4, 5])
def test_vtk3d_flow_times_span_the_whole_series(INUM):
    # Regression lock. flow_times used to be built from only the dumps loaded at
    # construction, so on the windowed path it held INUM+1 entries instead of 8.
    fd = _vtk3d(INUM)
    assert fd.d_start == 0 and fd.d_finish == 7
    assert len(fd.flow_times) == 8
    assert np.allclose(fd.flow_times, np.arange(8.0))


def test_vtk3d_windowed_actually_slides():
    # The consequence of the short timeline was not an error: INUM >= n_times-1
    # sent FluidData down the "all in memory" branch with extrapolate=(True,True),
    # which makes update_spline unreachable. Assert we are on the windowed path.
    fd = _vtk3d(4)
    assert isinstance(fd._flow[0], fluid.LinearSpline)
    assert fd._flow[0].extrapolate == (True, False)     # closed on the right
    assert fd.loaded_idx_bnds == (0, 4)
    assert len(fd._flow[0].x) == 5                      # window, not all 8
    fd(7.0)
    assert fd.loaded_idx_bnds[1] == 7                   # it really moved
    assert len(fd._flow[0].x) <= 5


def test_vtk3d_windowed_does_not_freeze_at_the_end_of_the_first_window():
    # The user-visible symptom: u = t everywhere, so a frozen fluid reports the
    # last time of the opening window forever. Before the fix u(6) and u(7) both
    # came back as 4.0, silently.
    fd = _vtk3d(4)
    for q in (0.0, 2.0, 4.0, 4.5, 6.0, 7.0):
        assert np.allclose(fd(q)[0], q), "u should equal t everywhere"


def test_vtk3d_windowed_matches_full_load():
    # Windowed vs. everything-in-memory, on the same files.
    dyn = _vtk3d(4)
    full = _vtk3d(True)
    for q in np.linspace(0.0, 7.0, 43):
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_vtk3d_windowed_mean_velocity_matches_full_load():
    # The plotting mean cache, on real files and a real slide. u = t everywhere,
    # so the x-component mean reads back the simulation time -- a frozen or
    # mis-indexed cache is immediately visible.
    dyn = _vtk3d(4)
    full = _vtk3d(True)
    for q in np.linspace(0.0, 7.0, 29):
        got = dyn.get_mean_velocity(time=q)
        assert np.allclose(got, full.get_mean_velocity(time=q), rtol=0, atol=1e-12)
        assert np.isclose(got[0], q)


def test_vtk3d_static_and_spatial_components_survive_the_slide():
    # v = x is steady and w = t*z varies in both; both must stay correct after
    # the window has moved, which catches a slide that loads the wrong dumps.
    fd = _vtk3d(4)
    x, y, z = [np.linspace(0, e, n) for e, n in zip((4.0, 3.0, 2.0), (5, 4, 3))]
    X, _, Z = np.meshgrid(x, y, z, indexing='ij')
    for q in (1.0, 6.5):
        u, v, w = fd(q)
        assert np.allclose(u, q)
        assert np.allclose(v, X)
        assert np.allclose(w, q * Z)


def test_vtk3d_grid_and_domain():
    fd = _vtk3d(4)
    assert fd.ndim == 3
    assert fd.fshape == (8, 5, 4, 3)
    assert np.allclose(fd.L, [4.0, 3.0, 2.0])
    for got, want in zip(fd.flow_points, (np.linspace(0, 4, 5),
                                          np.linspace(0, 3, 4),
                                          np.linspace(0, 2, 3))):
        assert np.allclose(got, want)


def test_inum_spanning_the_dataset_warns():
    # Asking for a window at least as wide as the dataset leaves nothing to
    # slide; the object silently holds everything in memory instead. Say so --
    # this is the guard that would have made the short-timeline bug loud.
    with pytest.warns(UserWarning, match='no dynamic loading'):
        fd = _vtk3d(7)
    assert fd._flow[0].extrapolate == (True, True)


def test_short_flow_times_is_rejected():
    # The direct lock on the defect: a loader that timestamps only the window it
    # loaded first must not be able to construct silently. INUM=True/None are
    # exempt, since neither slides a window.
    t, fpoints, comps = _field_2d(nt=21)

    class _ShortTimeline(_InMemorySource):
        def __init__(self, full_flow, flow_points, flow_times, INUM):
            # d_finish advertises 21 dumps, but only INUM+1 timestamps are passed
            self._full = [np.asarray(f) for f in full_flow]
            self.load_calls = []
            self.d_start = 0
            self.d_finish = len(flow_times) - 1
            self.loaded_dump_bnds = (0, INUM)
            self.loaded_idx_bnds = (0, INUM)
            window = self.load_dumpfiles(0, INUM)
            fluid.FluidData.__init__(self, window, flow_points,
                                     flow_times[0:INUM+1], INUM)

    with pytest.raises(RuntimeError, match='entire dump range'):
        _ShortTimeline([c.copy() for c in comps], fpoints, t.copy(), 5)


# --------------------------------------------------------------------------- #
#       OpenFOAMData end-to-end -- the finite-volume path, against files       #
# --------------------------------------------------------------------------- #
# The same questions as the VTK3dData block above, for the loader whose timeline
# is the one most able to go wrong: it is built from a series index, filtered by
# what is actually on disk, and indexed densely over the survivors. The fixture
# declares 8 dumps and 2 are absent, so 6 remain at t = 0,1,2,3,4,6.
#
# The field assembles to u = t, v = x, w = t*z over the interior, so u(t) reads
# back the simulation time and a frozen or mis-indexed timeline is obvious.

OPENFOAM = str(FIXTURES / 'openfoam_min')


def _openfoam(INUM):
    with pytest.warns(UserWarning):     # absent dumps, uneven spacing, BC corner
        return fluid.OpenFOAMData(OPENFOAM, INUM=INUM)


@pytest.mark.parametrize('INUM', [None, True, 4])
def test_openfoam_flow_times_span_the_surviving_series(INUM):
    # The VTK3dData trap, in the form it takes here. "The whole series" means the
    # dumps that EXIST, not the ones the index declares: building flow_times from
    # the manifest while indexing dumps by what is on disk would put the two
    # index spaces silently out of step.
    fd = _openfoam(INUM)
    assert fd.d_start == 0 and fd.d_finish == 5
    assert len(fd.flow_times) == 6
    assert np.allclose(fd.flow_times, [0., 1., 2., 3., 4., 6.])


def test_openfoam_windowed_actually_slides():
    fd = _openfoam(4)
    assert isinstance(fd._flow[0], fluid.LinearSpline)
    assert fd._flow[0].extrapolate == (True, False)
    assert fd.loaded_idx_bnds == (0, 4)
    assert len(fd._flow[0].x) == 5          # the window, not all 6
    fd(6.0)
    assert fd.loaded_idx_bnds[1] == 5       # it really moved


def test_openfoam_windowed_does_not_freeze_at_the_end_of_the_first_window():
    # u = t over the interior, so a frozen fluid reports the last time of the
    # opening window forever.
    fd = _openfoam(4)
    for q in (0.0, 2.0, 4.0, 4.5, 6.0):
        assert np.allclose(fd(q)[0][1:-1, 1:-1, 1:-1], q)


def test_openfoam_windowed_matches_full_load():
    dyn = _openfoam(4)
    full = _openfoam(True)
    for q in np.linspace(0.0, 6.0, 31):
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_openfoam_the_gap_is_interpolated_across_not_skipped():
    # t=5 is missing, so the interval 4->6 is twice as wide. The timeline must
    # still be in simulation time: a loader that indexed by dump position would
    # put the last dump at t=5 and shift everything after the hole.
    fd = _openfoam(4)
    assert np.allclose(fd(6.0)[0][1:-1, 1:-1, 1:-1], 6.0)
    # halfway across the wide interval, linear in time
    assert np.allclose(fd(5.0)[0][1:-1, 1:-1, 1:-1], 5.0)


def test_openfoam_boundary_data_follows_the_timestep():
    # The caps carry real data that varies in time, so each dump must read its
    # own patch files rather than the ones resolved when the mesh was built.
    fd = _openfoam(4)
    for q in (1.0, 6.0):
        u, _, w = fd(q)
        assert np.allclose(u[1:-1, 1:-1, 0], q)     # inlet cap
        assert np.allclose(u[1:-1, 1:-1, -1], q)    # outlet cap
        assert np.allclose(w[1:-1, 1:-1, -1], q*2.0)


# --------------------------------------------------------------------------- #
#        IB2dData end-to-end -- the 2D reference path, against real files       #
# --------------------------------------------------------------------------- #
# 2D IB2d is the dynamic-loading path that has actually been exercised by hand,
# and it was the only loader with no automated coverage whatsoever -- every
# read_IB2d_fluid_data call in the tree lives in tests/manual/ and needs external
# data. It is also the loader most exposed to the FluidData guards, since
# IB2dData publishes d_start/d_finish like any streaming source.
#
# Fixture: tests/fixtures/ib2d_fluid_min, 8 dumps of u.####.vtk on a 6x5 grid.
# IB2d omits the periodic endpoint in each direction and Planktos wraps it back
# on, so the loaded field is 7x6 over a 6x5 domain. Fields are
#     u = t                 -> u reads back the simulation time
#     v = sin(2*pi*x/Lx)    -> steady, periodic in x
# and dt=0.1, print_dump=10 puts the timestamps at 0, 1, ... 7.

IB2D = str(FIXTURES / 'ib2d_fluid_min')


def _ib2d(INUM):
    return fluid.IB2dData(IB2D, dt=0.1, print_dump=10, d_start=0, INUM=INUM)


@pytest.mark.parametrize('INUM', [None, True, 4])
def test_ib2d_flow_times_span_the_whole_series(INUM):
    fd = _ib2d(INUM)
    assert (fd.d_start, fd.d_finish) == (0, 7)
    assert len(fd.flow_times) == 8
    assert np.allclose(fd.flow_times, np.arange(8.0))


@pytest.mark.parametrize('INUM', [None, True, 4])
def test_ib2d_construction_is_warning_free(INUM):
    # The guards added to FluidData must not fire on a correct loader. IB2dData
    # derives flow_times analytically over the full dump range, so it satisfies
    # them -- but it is the path with the most to lose if they are wrong.
    #
    # Only warnings raised from inside planktos count. Promoting *every* warning
    # to an error also catches third-party deprecations that this suite neither
    # causes nor can fix: on numpy 2.5, vtk's own numpy_support does
    # `result.shape = shape`, which numpy now deprecates, so a blanket
    # simplefilter('error') fails here for reasons having nothing to do with the
    # guards. That the guards *do* fire when they should is pinned separately by
    # test_inum_spanning_the_dataset_warns and test_short_flow_times_is_rejected.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ib2d(INUM)
    ours = _from_planktos(caught)
    assert not ours, 'planktos warned during a correct load: {}'.format(
        [str(w.message) for w in ours])


def test_ib2d_windowed_actually_slides():
    fd = _ib2d(4)
    assert isinstance(fd._flow[0], fluid.LinearSpline)
    assert fd._flow[0].extrapolate == (True, False)
    assert fd.loaded_idx_bnds == (0, 4)
    assert len(fd._flow[0].x) == 5
    fd(7.0)
    assert fd.loaded_idx_bnds[1] == 7
    assert fd.loaded_idx_bnds == fd.loaded_dump_bnds


def test_ib2d_windowed_does_not_freeze():
    fd = _ib2d(4)
    for q in (0.0, 2.0, 4.0, 4.5, 6.0, 7.0):
        assert np.allclose(fd(q)[0], q), "u should equal t everywhere"


def test_ib2d_windowed_matches_full_load():
    dyn, full = _ib2d(4), _ib2d(True)
    for q in np.linspace(0.0, 7.0, 43):
        ref = full(q)
        assert _diff(dyn(q), ref) <= _tol(ref)


def test_ib2d_periodic_wrap_and_domain():
    # IB2d omits the duplicate endpoint; Planktos restores it. A 6x5 dump becomes
    # a 7x6 field spanning a 6x5 domain, with the appended edge copying the first.
    fd = _ib2d(4)
    assert fd.fshape == (8, 7, 6)
    assert np.allclose(fd.L, [6.0, 5.0])
    assert np.allclose(fd.flow_points[0], np.arange(7.0))
    assert np.allclose(fd.flow_points[1], np.arange(6.0))
    u, v = fd(3.0)
    assert np.allclose(u, 3.0)
    x = fd.flow_points[0][:, None]
    assert np.allclose(v, np.sin(2 * np.pi * x / 6.0))
    assert np.allclose(v[-1, :], v[0, :])      # wrapped edge duplicates the first
    assert np.allclose(u[:, -1], u[:, 0])
