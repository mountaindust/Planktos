# Fluid velocity field interface — analysis and refactor plan

Status: **plan / design note** (2026-07). Records the decisions reached while
investigating the `dyload` Phase-0 "`FlowArray` breaks numpy interop" item. Only
the `fmin`/`fmax` fix has been implemented so far; everything else here is the
agreed plan, to be executed in the sequence in §7.

**Revised 2026-07-31:** the tiling *stopgap* (2D materialization) was dropped —
tiling and domain extension are now deferred **wholesale**, raising
`NotImplementedError` until the real implementation lands after the plotting work.
See §5 and §9.

This note lives with the other `docs/notes/` design/derivation notes. It is the
source of truth for *why* the fluid-velocity-field abstraction is being reshaped
on `dyload`; keep it current as the work lands.

---

## 1. What the fluid velocity field is, and the roles it must play

Historically (`master`/`mvbnd`) `Environment.flow` was a plain
`list[np.ndarray]` — one array per spatial component, with a leading time axis
for time-varying flow. On `dyload` it became a `fluid.FluidData` object, and each
velocity component a `fluid.FlowArray` (an `ndarray` subclass). The motivation was
**dynamic loading**: time-dependent 3D fluid data is ~100 GB raw and larger once
splined, so it must be streamed a window of timesteps at a time rather than held
in memory all at once.

An exhaustive audit of every consumer (`_environment.py`, `_swarm.py`,
`motion.py`, `_dataio.py`, tests, examples) shows the flow object wears **three
distinct hats**:

- **A — list-of-arrays container.** `len(flow)` → number of components;
  `flow[dim]` → that component; iteration / `np.array(flow)` → stack. Used by
  `interpolate_flow`, `save_fluid`, `calculate_mag_gradient`, `_calc_basic_stats`,
  plotting.
- **B — each component behaves as a full ndarray.** Passed as `values` to
  `scipy.interpolate.interpn` (the per-move hot path); consumed by `np.gradient`
  (vorticity, `|u|` gradient), matplotlib `quiver`/`pcolormesh`/`set_UVC`;
  arithmetic/ufuncs (`**2`, `+`, `np.sqrt`), reductions (`.mean()`, `.max()`),
  N-D strided slicing `[::M,::N]`, `.T`, shape/ndim, element-wise `>`/`==` →
  `np.all`, and `np.allclose`/`np.isclose` in tests.
- **C — callable with dynamic reload.** `flow(time)` returns the temporally
  interpolated field. When `time` falls outside the currently loaded window,
  `FluidData.__call__` catches a `SplineRangeError` and calls `update_spline` to
  load the next/previous dump window and re-spline, then retries. This is the
  core `dyload` mechanism and lives entirely in `FluidData`/the spline classes —
  it does **not** depend on `FlowArray`.

Notes from the audit worth keeping:
- `_geom.py`, `_ibc.py`, `_dataio.py`, `__init__.py` do **not** touch the flow
  field. `_dataio` only *produces* the raw ndarrays that become flow data.
- `motion.py` never touches `Environment.flow` directly — it goes through the four
  `Swarm` accessors (`get_fluid_drift`, `get_dudt`, `get_DuDt`,
  `get_fluid_mag_gradient`), all of which funnel into `Environment.interpolate_flow`.
- So the interface's blast radius is essentially `_environment.py` + a few
  `_swarm.py` accessors + plotting. A redesign is tractable.

---

## 2. Why `FlowArray` exists

`FlowArray` earns its existence from exactly one feature: **`tile_flow`** —
representing a domain tiled `k` times in x/y without materializing `k` copies in
memory. Tiling is needed for agent-based work where agents must diffuse within a
*larger* area than a single periodic cell (e.g. a bed of seagrass): you tile the
flow rather than merely making it periodic.

Evidence that tiling is the sole purpose: for time-invariant flow the constructor
does `self._flow = [f.view(FlowArray) for f in flow]`. The only thing that view
buys over a plain ndarray is a place to hang `.tiling` and have the overridden
`.shape`/`__getitem__` report/serve a virtual `k×` grid off one stored tile:

```python
@property
def shape(self):
    if self.tiling is not None:
        return ((self.array.shape[0]-1)*self.tiling[0] + 1,
                (self.array.shape[1]-1)*self.tiling[1] + 1, *self.array.shape[2:])
    return self.array.shape

def __getitem__(self, pos):   # ~100 lines mapping virtual indices → stored tile via modulo
    ...
```

The intent: hand a tiled `FlowArray` to `interpn`/`np.gradient`/matplotlib and let
*them* index it as if it were the big array, with the modulo logic transparently
wrapping back into the one stored tile. The time-varying siblings
(`fCubicSpline`, `LinearSpline`) carry `.tiling` and reproduce the pattern.

### The numpy-subclassing tradeoff

NumPy offers `__array_ufunc__` (intercept ufuncs), `__array_function__` (intercept
numpy API functions), and subclass + `__array_finalize__`. None of the first two
can intercept a bare `values.shape` read or a `values[index_tuple]` index — and
those are exactly what scipy's interpolator, matplotlib, and `np.gradient` do. So
to virtualize tiling you are forced to override `.shape` and `__getitem__`
directly on an `ndarray` subclass. That is the path the NumPy docs warn against:
C-level numpy code reads the true shape from the array struct, not the Python
property, so the override only affects Python-level access and desynchronizes from
the buffer. That mismatch is the root of both defects below.

---

## 3. Two defects (with evidence)

### 3.1 numpy-interop bug — stale `self.array` on every derived array

`FlowArray.__array_finalize__` sets `self.array = obj` on view-casting and
`self.array = obj.array` on new-from-template (ufunc / reduction / slice results).
The derived array's own buffer holds the computed values, but `.shape`/`__getitem__`
keep reading the *input's* buffer via `self.array`. So any `FlowArray` produced by
numpy reads **stale** data through Python-level access. Empirically (fresh untiled
`FlowArray` `fa` over `arange(12).reshape(3,4)`):

| Operation | Expected | Actual |
|---|---|---|
| `(fa**2)[0]` | `[0,1,4,9]` | `[0,1,2,3]` (reads pre-square `self.array`) |
| `np.allclose(a, a)` | `True` | **`False`** |
| `repr(np.mean(fa, axis=0))` | prints | **raises `ValueError`** (buffer is `(4,)`, `.shape` claims `(3,4)`) |

numpy's high-level routines mix C-level buffer reads with Python-level
`__getitem__`/iteration/`.shape`; the two disagree for any derived `FlowArray`, so
results are wrong or raise. This is why the tests and `save_fluid` defensively wrap
components in `np.asarray(...)` before array-wide numpy calls (documented smoking
guns: `tests/test_io_loaders.py:184-185,210`, `tests/test_flow_generation.py:37`,
plus `_environment.py:2997` in `save_fluid`). Fresh, untiled, non-derived indexing
happens to work (`flow[0][20,-1]`), which is why scalar `np.isclose` checks pass
without the workaround.

### 3.2 Tiling virtualization is defeated by modern scipy

End-to-end: build steady `u(x,y)=x` on a 4×4 grid, `tile_domain(x=2)`, then
`interpolate_flow`:

```
After tile: flow[0].shape(prop)=(7,4)  stored buffer=(4,4)  len(flow_points[0])=7
=> TILED STEADY INTERP RAISES: ValueError  There are 7 points and 4 values in dimension 0
```

Root cause, pinned in the scipy source: `RegularGridInterpolator._check_values`
does `if is_array_api_obj(values): values = np.asarray(values)`. A `FlowArray`
*is* an array-API object, so scipy converts it to a base ndarray with the real
`(4,4)` buffer — the `.shape`→`(7,4)` and modulo-`__getitem__` overrides are never
consulted. (Tested on scipy 1.17.1; the array-API rewrite is what introduced the
coercion.) So the entire purpose of the ndarray subclass is currently
non-functional, and untested — **no test calls `interpolate_flow` after
`tile_domain`.** This is independent of §3.1: even a perfectly correct `FlowArray`
loses its virtual shape to `np.asarray`. The same mechanism hits the time-varying
tiled path (`flow(time)` returns tiled components into the same `interpn`).

### 3.3 `fmin`/`fmax` were generators (FIXED)

`FluidData.__init__` and `update_spline` built `self.fmin`/`self.fmax` as
**generator expressions** instead of tuples. Effects: `update_spline` did
`self.fmin[n]` → `TypeError: 'generator' object is not subscriptable` on every
window slide (the dynamic path); and plotting's `max_u, max_v = flow.fmax` worked
exactly once, then raised `ValueError: not enough values to unpack` on the second
plot (generator exhausted). Fixed by wrapping the three sites in `tuple(...)`
(`fluid.py` `__init__` + both `update_spline` slide branches). Values were always
correct; only the container type was wrong. Unrelated to `FlowArray`.

### 3.4 `max_spd` and `get_mean_fluid_speed` were corrupted by §3.1 (FIXED)

Found by the §7.2 pinning pass — the concrete user-visible damage §3.1 was doing.

`FlowArray` overrides `min()`/`max()` to return `self.array.min()`/`.max()`
(`fluid.py:259-263`) — i.e. the **original component's** buffer, not the derived
array's own values. `mean()` is *not* overridden, so it takes numpy's C-level path
and stays correct. That asymmetry is why the corruption went unnoticed:

- **`Swarm._calc_basic_stats`** computed `flow_spd = np.sqrt(flow[0]**2 + flow[1]**2)`
  and then `flow_spd.max()`, which returned **max |u|** instead of the max fluid
  speed. On the analytic field `u = x, v = 2y` over `[0,10]×[0,8]` it reported
  `10.0` where the true max speed is `sqrt(10² + 16²) ≈ 18.87`. This value is
  printed as "Fluid $v_{max}$" on **every plot and movie frame** (8 call sites in
  `_swarm.py`), so every figure produced on this branch carried a wrong number
  whenever the max speed was not attained by the x-component alone.
- The same routine's `avg_spd_x = flow[0].mean()` returned a 0-d `FlowArray` whose
  `.shape` misreported `(nx, ny)`, so `np.isclose(avg_spd_x, ...)` broadcast to a
  full array instead of yielding a scalar.
- **`Environment.get_mean_fluid_speed`** returned `np.mean(...)` over a derived
  `FlowArray`: the *value* was right, but the returned object misreported its own
  shape, breaking any caller that compared or reduced it.

Both fixed by converting components with `np.asarray` before the array math
(`_swarm.py` `_calc_basic_stats` 2D + 3D branches, `_environment.py`
`get_mean_fluid_speed`). Pinned by `test_calc_basic_stats_*` and
`test_get_mean_fluid_speed_static_known_answer`, which assert the max *speed* and
explicitly assert it differs from `max|u|`.

**Not changelog material:** on `master` the identical code is correct, because
`flow[dim]` is a plain ndarray there. This is a `dyload`-only regression
introduced by `FlowArray` and fixed within `dyload` — excluded per the changelog
rule in `CLAUDE.md`.

### 3.5 `get_raw_loaded_data` was broken on the dynamic path (FIXED)

Also found by the §7.2 pinning pass, and the more serious of the two finds because
it sits on this branch's headline feature.

`FluidData.get_raw_loaded_data` dispatched on the spline *type*:

```python
if isinstance(self._flow[0], fCubicSpline):
    return [flow.regenerate_data() for flow in self._flow]
else:
    return self._flow          # assumed "static, so already ndarrays"
```

The `else` branch was written for static flow, but it also catches `LinearSpline`
— i.e. **every dynamically-loaded run**, since `INUM` is exactly what selects
linear splining. So on the dynamic path the method handed back `LinearSpline`
*objects* instead of ndarrays, and any arithmetic on the result raised
`TypeError: unsupported operand type(s) for -: 'LinearSpline' and 'float'`. This is
the documented replacement for the old `Environment.regenerate_flow_data()` and is
user-visible (CLAUDE.md lists it on `FluidData`'s public surface).

Fixed two ways, together:
- `LinearSpline` gained `regenerate_data()`, mirroring `fCubicSpline` so both
  spline classes present the same surface — which is what §6 below already
  specified. It returns `self.flow` directly, with **no copy**: the raw data is
  stored outright, and under dynamic loading a window can be very large.
- `get_raw_loaded_data` now branches on `flow_times is None` (static vs
  time-varying) rather than on the spline class, so it cannot be wrong-footed by
  a third spline type later.

Pinned by `test_get_raw_loaded_data_round_trips` (both `INUM=None` and `INUM=True`)
and `test_get_raw_loaded_data_static`.

---

## 4. Decisions (this design pass)

1. **Delete `FlowArray`.** Velocity components — both the stored static arrays and
   the values returned by the spline classes' `__call__`/`__getitem__` — become
   **plain `np.ndarray`.** This eliminates §3.1 with **zero special cases** (a
   component *is* an ndarray, so `np.allclose`, `repr`, `np.gradient`, `interpn`,
   arithmetic, slicing, `.T` all just work), removes every `np.asarray(...)`
   workaround, and deletes the broken/untested tiling-through-`interpn` path
   (§3.2). It gives users, future methods, and plotting one intuitive contract
   with no gotchas.

2. **Defer tiling wholesale — no stopgap.** Tiling raises `NotImplementedError`
   in **both 2D and 3D** for the duration of the core refactor and the plotting
   work. It was only elevated in the first place because dynamic loading is
   memory-management work and tiling felt entangled; it is in fact separable from
   dynamic loading (§1 hat C is independent of `FlowArray`). See §5 for the
   rationale and §9 for the eventual implementation.

3. **`Environment.extend` stays removed for now,** and comes back — if it comes
   back — alongside the real tiling work in §9. Extending and tiling are the same
   kind of operation (they both change what the reported domain is relative to the
   stored grid), so they should be designed together, not one at a time. The
   parked test in `tests/test_flow_generation.py` stays skipped until then.

4. **The naming rule for tiling (adopt now, even though tiling is deferred):**
   > **Public geometry (domain `L`, plot extent, reported grid shape) reflects the
   > *tiled* domain. Stored data and all interpolation use the *base tile*.
   > Nothing materializes the full tiled grid-data array on a hot path.**
   If a consumer ever truly needs the big array, that is one explicit
   `materialize_tiled()`-style method that documents its memory cost. This is the
   single, predictable mental model the `FlowArray` subclass was groping toward.

5. **`fmin`/`fmax` → tuples.** Done (§3.3).

---

## 5. Tiling and domain extension — deferred wholesale

**What happens during the core refactor (§7) and the plotting work (§8):**
`FluidData.tile_flow` / `Environment.tile_domain` raise `NotImplementedError` in
**both 2D and 3D**. `Environment.extend` remains removed. Neither is patched,
stopgapped, or partially materialized in the interim.

**Why no 2D materializing stopgap** (this note originally proposed one; it was
dropped 2026-07-31):

- A 2D-only stopgap would not have bought much. Of the tiling consumers in the
  tree, `examples/ex_IBAMR_ibmesh.py` (`tile_domain(3,3)`),
  `examples/ex_sticky_seafan_3d.py` (`tile_domain(x=13)`), and the tiling
  discussion in `examples/basic_ex_3d.py` are all **3D** — they would have hit
  `NotImplementedError` under the stopgap anyway. Only
  `tests/test_flow_generation.py::test_tile_flow_replicates_and_resizes` and
  `examples/old_examples/old_ex_pltcyl.py` are 2D.
- `tests/IBAMR_test_data/` (3D IBAMR vtk dumps + mesh) **is now available** — it
  was not on the machine where this plan was first written. That removes the
  reason to build a 2D-only interim: the real, dimension-agnostic implementation
  in §9 can be verified end-to-end against actual 3D data, so a throwaway 2D
  materializer is wasted work that would have to be deleted again.
- Materializing also contradicts the §4.4 naming rule we are adopting now. Better
  to have the feature clearly and loudly unavailable than to have it quietly
  behave a different way in 2D than it eventually will in 3D.

**Consequences to expect and not be alarmed by:**
- `test_tile_flow_replicates_and_resizes` must be converted — either to a skip
  (mirroring the parked `extend` test) or to an assertion that
  `NotImplementedError` is raised. Prefer the latter: it pins the intended
  interim contract rather than silently dropping coverage.
- The 3D examples above will not run to completion until §9 lands. Flag this in
  their headers rather than editing the calls out.

---

## 6. Target interface (after `FlowArray` removal)

- `Environment.flow` is a `FluidData` (or subclass), or `None`.
- **Container (hat A):** `len(flow)`, `flow[dim]` → plain ndarray (static case),
  iteration, `np.array(flow)`. Unchanged contract, but returns plain ndarrays.
- **Callable (hat C):** `flow(time)` → list of plain ndarrays on the base-tile
  grid; triggers `update_spline` on out-of-window times. Unchanged mechanism.
- **Components (hat B):** plain `np.ndarray`. All numpy/scipy/matplotlib ops work
  natively; no subclass, no overrides, no interop caveats.
- **Metadata:** `flow_times` (None ⇒ static), `flow_points`, `periodic_dim`,
  `fshape`, `fmin`/`fmax` (tuples), `fluid_domain_LLC`, `L`; methods
  `tile_flow` (**raises `NotImplementedError`** until §9), `get_vorticity`,
  `get_dudt`, `calculate_DuDt`. `Environment.extend` remains absent.
- The spline classes (`fCubicSpline`, `LinearSpline`) keep their `__call__` /
  `__getitem__` / `min`/`max`/`absmax` / `regenerate_data` surface, but return
  **plain ndarrays** rather than `FlowArray` views. (`LinearSpline.regenerate_data`
  did not actually exist when this section was first written — it was added as part
  of the §3.5 fix, so the two classes now genuinely match.)

---

## 7. Implementation sequence & test strategy

Correctness-first: pin trusted behavior before removing anything.

1. **[done]** Fix `fmin`/`fmax` → tuples; add a regression test (unpack `fmax`
   twice; and, once IB2d dynamic data exists, survive a window slide).
2. **[done]** **Pin current trusted behavior** with tests before deleting
   `FlowArray` → `tests/test_flow_interface.py` (40 tests). Covers static/
   time-varying `interpolate_flow` (2D + 3D, on/off-node, extrapolation, explicit
   `flow=` argument), `interpolate_temporal_flow`, the container contract,
   `fmin`/`fmax` tuples (the §3.3 regression lock, which had never been written),
   `_calc_basic_stats`, `get_mean_fluid_speed`, `calculate_mag_gradient`,
   `get_raw_loaded_data`, `fshape`, the quiver-style strided-slice/transpose used
   by the plotting code, the **`LinearSpline`/`INUM` temporal path**, and 3D
   vorticity. 2D vorticity (`test_analysis.py`) and `save_fluid` round-trips
   (`test_io_loaders.py`) were already pinned and are not duplicated. No tiling
   values pinned — tiling is going away behind `NotImplementedError`, and the
   tiled interpolation path is broken today anyway per §3.2.

   The `LinearSpline` coverage matters disproportionately: `test_temporal_interp.py`
   unit-tests `fCubicSpline` thoroughly but never touched `LinearSpline`, even
   though it is what *every* dynamically-loaded run interpolates with and §7.3
   changes what both spline classes return. `Environment(flow=...)` hardcodes
   `INUM=None`, so those tests construct `FluidData` directly.

   **The pass found three live bugs**, all fixed: two from §3.1 leaking into
   user-visible output (§3.4, patched with the `np.asarray` idiom already used in
   `save_fluid` — remove in §7.3 below), and one independent of `FlowArray`
   entirely (§3.5, `get_raw_loaded_data` broken on the whole dynamic path).
3. **[done]** **Delete `FlowArray`;** static components and spline returns are
   plain ndarrays. All `np.asarray(...)` workarounds dropped from `save_fluid`,
   `get_mean_fluid_speed`, `_calc_basic_stats`, and the tests. Removing
   `FlowArray` also deleted the ~165-line class plus the tiling index-mapping
   branches it served inside `LinearSpline.__getitem__` and
   `fCubicSpline.__getitem__` (~73 lines each), and the `tiling`/`dshape`
   attributes throughout — roughly 380 lines net.
4. **[done]** **Gate tiling off:** `FluidData.tile_flow` and
   `Environment.tile_domain` raise `NotImplementedError` in 2D and 3D.
   `tile_domain` raises *before* mutating anything, so a failed call cannot leave
   a half-tiled environment (mesh and `L` updated, fluid not) — pinned by
   `test_tile_domain_leaves_environment_untouched`. The old
   `test_tile_flow_replicates_and_resizes` became three tests asserting the
   interim contract. The affected examples (`ex_IBAMR_ibmesh.py`,
   `ex_sticky_seafan_3d.py`, `basic_ex_3d.py`) and their docs pages carry a
   notice rather than having the calls edited out.
5. **[done]** Full suite green: **201 passed, 20 skipped** (`pytest`);
   **219 passed, 2 skipped** (`--runslow`); codespell clean.
6. **[done]** `changelog.txt` updated: the `FlowArray` line is replaced with the
   plain-ndarray guarantee, plus a line recording that tiling temporarily raises.

Then §8 (plotting streaming), and only after that §9 (real tiling + revisit
`extend`). Each of those is its own design pass with its own note-worthy
decisions.

---

## 8. Plotting streaming (outline — flesh out later)

Recorded now, to be designed after dynamic loading is solid. Plotting is already a
bottleneck in 2D runs; for 3D dynamic loading it is worse, because a naive design
pays the load+interpolate cost **twice** — once to advance agents, once to render
frames — re-streaming ~100 GB on the second pass.

Target architecture: **render each frame from the fluid window that is already
resident for the current simulation step, and cache a cheap (compressed) image per
frame; assemble the video at the end.** Sketch:
- an opt-in mode / shortcut method (e.g. `move_and_plot()`) that renders during the
  move loop from the already-loaded window, so the expensive data is touched once;
- persist frames as compressed images to disk (PNG per frame, or straight into an
  ffmpeg pipe) rather than holding figures in memory — memory stays O(one frame);
  this also helps the existing 2D bottleneck, independent of dynamic loading;
- `plot_all` reads from the frame cache if present, instead of re-simulating the
  flow for visualization;
- pairs naturally with the position-wrapping tiling of §9: a tiled quiver wraps
  coordinates the same way the interpolator does, so plotting never materializes a
  tiled array. Doing plotting *first* means §9 has a working renderer to validate
  tiled visualization against.
- open questions: keeping interactive/exploratory (random-access replay) plotting
  working alongside the streaming one-pass mode; the API surface; decoupling
  frame-rate from step-rate.

---

## 9. Tiling (and `extend`) — the real implementation, after §8

Done once, for 2D **and** 3D together, with `tests/IBAMR_test_data/` available to
verify the 3D path end-to-end.

**Position-wrapping, memory-free, dimension-agnostic.** A tiled domain is a
periodic extension, and `interpolate_flow` *already* implements periodic extension
by wrapping query positions (`positions[:,n] % flow_points[n][-1]`). So tiling
never needs a big array on any hot path:
- store `tiling = (tx, ty[, tz])` + the one base tile on `FluidData`;
- interpolation wraps agent positions into the base tile, then `interpn` against
  the base tile — identical in 2D and 3D;
- vorticity/`|u|`-gradient over a tiled domain = the base tile's field replicated,
  so compute on the tile and replicate the *result* only if a consumer needs the
  full field;
- reported `.shape` / domain extent = arithmetic on `base_shape × tiling`, no
  allocation (the §4.4 naming rule).

**Reconcile with `periodic_dim`:** a tiled dimension is effectively periodic for
interpolation. The implementation must define the interaction of `tiling` and
`periodic_dim` explicitly (a tiled dim implies wrapping regardless of the
`periodic_dim` flag for that dim).

**Revisit `extend` here.** `Environment.extend` (pad the fluid domain with copies
of the edge values) was removed on `dyload` in favor of extrapolation. Decide at
this point whether to bring it back for the specific fluid fields where padding
is the physically right answer — it is the same class of operation as tiling
(reported domain ≠ stored grid) and should share the mechanism rather than
re-materializing arrays. If it returns, un-skip
`test_extend_grows_domain_and_copies_edges` in `tests/test_flow_generation.py`.

### 9.1 Restoration checklist — everything §7.4 touched

Gating tiling off left notices, stubs, and replaced tests scattered across source,
tests, examples, docs, and prose. **This is the complete list**; work down it when
tiling returns, and delete this subsection once it is empty.

⚠️ **Read this first: the old bodies are preserved in place, commented out.**
`FluidData.tile_flow` and `Environment.tile_domain` both had their entire bodies
replaced by a `raise`, but the previous implementations sit directly beneath each
raise under a `PREVIOUS IMPLEMENTATION, KEPT FOR RESTORATION` banner. **Reuse them
rather than rewriting from memory** — parts of both are still correct:

- `tile_domain` — only its `self.flow.tile_flow(x,y)` call is superseded by
  position-wrapping. The ibmesh tiling (offsetting copies by `L[0]*ii`,
  `L[1]*jj`), the `self.L` scaling, and the `_reset_flow_deriv()` call are still
  correct verbatim.
- `tile_flow` — the `f.tiling` propagation is dead (`FlowArray` and the spline
  `tiling` attributes are gone), but the `fshape` arithmetic and the
  `flow_points` extension are the shape/geometry half of the §4.4 naming rule and
  carry over as-is. The reported coordinate arrays still have to grow with the
  tiling even though the velocity data will not.

**Source — remove the gates:**
- [ ] `planktos/fluid.py` — `FluidData.tile_flow`: replace the `raise` and its
      `.. note::` with the position-wrapping implementation, reusing the
      commented-out `fshape`/`flow_points` handling. Delete the commented block
      once its useful parts are back in force.
- [ ] `planktos/_environment.py` — `Environment.tile_domain`: same, restoring the
      commented-out ibmesh/`L`/`_reset_flow_deriv` logic. Note the docstring
      currently explains *why* it raises before mutating anything — that rationale
      stops applying once the call succeeds. Delete the commented block afterward.
- [ ] `planktos/_swarm.py` — the `Swarm` class docstring example lost its
      `>>> envir.tile_domain(3,3)` line (it would have raised). Restore if you want
      the example to show tiling again.

**Tests — replace the interim contract with a behavioral one:**
- [ ] `tests/test_flow_generation.py` — delete `test_tile_domain_raises_not_implemented`,
      `test_tile_domain_leaves_environment_untouched`, and
      `test_tile_flow_raises_on_fluiddata_directly`, plus the section comment above
      them. Restore a real check: the pre-§7.4 `test_tile_flow_replicates_and_resizes`
      is in git history (same commit as above) and is a reasonable starting point,
      **but it only covered 2D and only the stored values** — the new implementation
      needs interpolation-through-tiling and 3D coverage, which is exactly what was
      missing before (§3.2: no test ever called `interpolate_flow` after
      `tile_domain`, which is why the breakage went unnoticed).
- [ ] Add the `tiling` × `periodic_dim` interaction tests this section calls for.

**Examples — delete the notices:**
- [ ] `examples/ex_IBAMR_ibmesh.py` — "!!! THIS EXAMPLE DOES NOT CURRENTLY RUN TO
      COMPLETION !!!" block in the module docstring.
- [ ] `examples/ex_sticky_seafan_3d.py` — same block in the module docstring.
- [ ] `examples/basic_ex_3d.py` — the `# NOTE: tile_domain currently raises ...`
      comment.
- [ ] `examples/old_examples/old_ex_pltcyl.py` calls `tile_domain(3,3)` and was
      **deliberately left unflagged** — it is archived record-keeping code whose own
      header says to skip it. Nothing to undo; listed so its absence above does not
      read as an oversight.

**Docs — delete the warnings:**
- [ ] `docs/examples/IBAMR_ibmesh.rst` — the `.. warning::` after the `tile_domain`
      snippet.
- [ ] `docs/examples/basic_3d.rst` — the `.. warning::` after the tiling paragraph.
- [ ] When `docs/api/FluidData.rst` finally exists (an open TODO.md item), make sure
      `tile_flow`'s docstring no longer carries the unavailability note.

**Prose — retract the "temporarily unavailable" framing:**
- [ ] `CLAUDE.md` — the "**Domain tiling currently raises `NotImplementedError`**"
      paragraph in "Fluid data architecture", and the `test_flow_generation.py`
      bullet in the Tests section.
- [ ] `TODO.md` — Phase 1 item **(E)**, and the deferred `Environment.extend` item.
- [ ] This note — §5 (the deferral rationale), §7.4, and this section.

**Release coordination:** `changelog.txt` under 1.1.0 carries
`- Domain tiling temporarily raises NotImplementedError; it returns with 2D and 3D support.`
That line is accurate **only if 1.1.0 ships before tiling returns.** If tiling lands
first, delete it and describe the new implementation instead. Do not let both
statements ship together.

---

## Appendix — consumer catalogue pointers

Interface requirements were derived from a full audit. Highest-signal sites:
- **Spatial interp (hot path):** `_environment.py` `interpolate_flow` (~L2020-2053),
  `interpolate_temporal_flow` (returns `flow(time)`).
- **Swarm accessors → interp:** `_swarm.py` `get_fluid_drift`, `get_dudt`,
  `get_DuDt`, `get_fluid_mag_gradient` (~L1188-1343); `motion.py` uses only these.
- **Gradient/analysis:** `calculate_mag_gradient`, `get_mean_fluid_speed`,
  `get_vorticity`, `get_dudt`, `calculate_DuDt`; `_calc_basic_stats`
  (`_swarm.py` ~L1716-1807).
- **Save/round-trip:** `save_fluid` (`np.asarray` workaround at `_environment.py`
  ~L2997), `save_2D_vorticity`.
- **Plotting:** `plot_flow`, `plot_2D_vort`, `Swarm.plot`/`plot_all` (quiver
  strided slices `flow[k][::M,::N].T`, `fmax` unpack, `fshape[1:]` frame sizing).
- **`np.asarray` workaround sites (remove after §7.3):**
  `tests/test_io_loaders.py:184-185,210`, `tests/test_flow_generation.py:37`,
  `_environment.py` `save_fluid`, `_environment.py` `get_mean_fluid_speed`,
  `_swarm.py` `_calc_basic_stats` (both the 2D and 3D branches), and the
  defensive `np.asarray` wrappers throughout `tests/test_flow_interface.py`.
- **Tiling call sites:** this was the pre-work grep used to plan §7.4. It is now
  superseded by **§9.1, the restoration checklist**, which records what was actually
  done to each site rather than merely where they were. Use §9.1; line numbers here
  went stale the moment the work landed.

---

## Session handoff / cold start

If you are picking this up in a fresh session with no prior context, read this
whole note plus `CLAUDE.md` and `TODO.md` (root) first. Quick orientation:

**Where we are (as of the 2026-07-31 revision):**
- Branch `dyload`. This is the "dynamic loading of fluid data" feature branch.
- **Done:** the `fmin`/`fmax` generator fix (§3.3); this note; the §7.2 pinning
  suite `tests/test_flow_interface.py` (40 tests); the three bugs it surfaced
  (§3.4 `max_spd` / `get_mean_fluid_speed`, §3.5 `get_raw_loaded_data`); and
  **§7.3–§7.6 — `FlowArray` is deleted and tiling is gated off.** Suite is green:
  **201 passed, 20 skipped** with `pytest`; **219 passed, 2 skipped** with
  `--runslow`.
- The §7.2 suite passes with every `np.asarray` wrapper stripped out, which is
  what makes the deletion *provably* behavior-preserving rather than merely
  untested-and-green. Only one test in the whole suite had to change behavior:
  the tiling one, by design.
- **Next actionable step: §8 — the plotting streaming redesign.** Then §9 (real
  tiling + revisit `extend`).
- **Not started:** §8 and §9.
- **`tests/IBAMR_test_data/` is present** (`IBAMR_db_003/004/005.vtk`,
  `mesh_db.vtk`) — 3D IBAMR data for the `@vtk`-marked tests and for validating
  §9. Its absence on the original authoring machine is what led to the since-
  dropped 2D tiling stopgap.

**The immediate next actionable step** is §8, the plotting streaming redesign,
which is still an outline and needs its own design pass before any code. §9 (real
tiling, and whether `extend` returns) follows it.

**Re-confirming §7.3 landed cleanly**, if picking this up cold: the reproduction
snippets below for defects §3.1 and §3.2 should now fail at *import* /
`tile_domain` respectively, because `FlowArray` no longer exists and tiling
raises. That is the intended end state, not a regression.

**Still outstanding housekeeping:** update `TODO.md` Phase 0 — the `fmin`/`fmax`
item is done, and the `FlowArray` interop item is superseded by this note's plan
(the deeper finding is §3.2 — tiling-through-`interpn` is defeated by modern scipy,
so the fix is deletion + deferral, not patching the subclass). Phase 1 item (E)
("Tiling/periodic × dynamic") is likewise superseded: tiling is gated off until §9.

**Re-confirming the evidence (historical).** These reproduced the defects *before*
§7.3. Kept as the record of what was wrong; §3.1 and §3.2 no longer run at all
(`FlowArray` is gone; `tile_domain` raises), which is the point:

```python
# Defect 3.1 — FlowArray interop (stale self.array):
import numpy as np; from planktos.fluid import FlowArray
fa = np.arange(12., ).reshape(3,4).view(FlowArray)
print((fa**2)[0])            # -> [0 1 2 3] (stale; should be [0 1 4 9])
print(np.allclose(fa+0., fa+0.))   # -> False (should be True)
repr(np.mean(fa, axis=0))    # -> raises ValueError

# Defect 3.2 — tiling defeated by scipy asarray coercion:
import planktos
n=4; g=np.linspace(0,1,n); X,Y=np.meshgrid(g,g,indexing='ij')
e=planktos.Environment(Lx=1,Ly=1,flow=[X.copy(),np.zeros_like(X)],
                       x_bndry=['zero','zero'],y_bndry=['zero','zero'])
e.tile_domain(x=2,y=1)
e.interpolate_flow(np.array([[1.5,0.5]]))   # -> ValueError: 7 points and 4 values

# fmin/fmax fix (should print tuple, and unpack twice without error):
e2=planktos.Environment(Lx=1,Ly=1,flow=[X.copy(),(2*Y).copy()],
                        x_bndry=['zero','zero'],y_bndry=['zero','zero'])
print(type(e2.flow.fmax).__name__)          # -> tuple
a,b=e2.flow.fmax; c,d=e2.flow.fmax          # no error (was single-use generator)
```

(After §7.4 the Defect-3.2 snippet will raise `NotImplementedError` at
`tile_domain` instead — that is the intended new behavior, not a regression.)

Root-cause pointer for §3.2: `scipy.interpolate.RegularGridInterpolator`
`_check_values` calls `np.asarray(values)` on any array-API object, materializing
`FlowArray`'s real (untiled) buffer and discarding the virtual `.shape`/`__getitem__`.
Tested on scipy 1.17.1 / numpy 2.4.6.

The full consumer audit that this plan is built on was done by reading every
`.flow` interaction across `_environment.py`, `_swarm.py`, `motion.py`,
`_dataio.py`, tests, and examples; the highest-signal sites are in the appendix
above. If you need the exhaustive per-site catalogue again, re-run the audit
(grep `\.flow`, `flow_points`, `flow_times`, `np.asarray`, `interpolat` across
those paths) rather than trusting memory.
