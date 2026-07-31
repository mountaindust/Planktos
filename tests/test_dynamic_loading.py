'''Pins the dynamic (windowed) fluid-loading machinery -- TODO.md Phase 1 (A),
(B) and (D).

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

No RNG, no external data, no file I/O.
'''

import numpy as np
import pytest

from planktos import fluid


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
