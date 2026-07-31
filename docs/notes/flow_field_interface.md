# Fluid velocity field interface — analysis and refactor plan

Status: **plan / design note** (2026-07). Records the decisions reached while
investigating the `dyload` Phase-0 "`FlowArray` breaks numpy interop" item. §7 is
complete (`FlowArray` deleted, tiling gated off) and §8 step 1 (frame statistics)
has landed; §8 steps 2–5 and §9 remain the agreed plan. Individual sections carry
their own **[done]** markers.

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
`extend`). §8 has since had its design pass and is fully specified; §9 still needs
one.

---

## 8. Plotting streaming — implementation plan

Status: **step 1 built; steps 2–5 specified, not yet implemented** (design settled
2026-07-31). All design questions are decided; what follows is the specification and
build order, not a discussion. The deliberation that produced these choices — the
options weighed and rejected — is in this file's git history.

**Starting cold?** Read §8.1–§8.2 for the problem and scope, then §8.4 for what to
build first and §8.4.1 for the concrete entry points.

**The problem.** `Swarm.plot_all` replays the whole run after the fact, pulling fluid
data at every frame. Under dynamic loading that re-streams the entire dataset a second
time, having already streamed it once to advance the agents.

**The solution, in one line:** capture what each frame needs *while the data is already
resident*, into a small derived-quantity cache, and make the per-frame statistics cost
nothing.

> **Correction (supersedes the original outline).** That outline claimed plotting was
> also a *memory* bottleneck, because "the whole animation is built before anything is
> written", and proposed streaming frames to disk as an independent win. **This is
> false.** `Animation.save()` already wraps `writer.saving(...)` and calls
> `grab_frame()` per frame, so `plot_all` has always streamed into the ffmpeg pipe with
> O(one frame) encoding memory; `FuncAnimation` holds a single figure and redraws it,
> and `cache_frame_data` caches only the frame *indices*. The existing 2D bottleneck is
> **time** — recomputing vorticity and re-rendering every frame — which is what §8.3.1
> and the cache address. There is no separate "stream the video" work item.

---

### 8.1 Why — where the cost actually is

`plot_all`'s `animate(n)` is a random-access replay over `pos_history`; with
`frames=None` it renders one frame per `move()` call. Each frame pulls fluid data at
`envir.time_history[n]`:

| Per-frame call | Cost | Applies to |
|---|---|---|
| `_calc_basic_stats(t_indx=n)` → `interpolate_temporal_flow` | **full field**, then `.mean()`/`.max()` | 2D **and 3D**, unconditionally |
| `get_vorticity(t_indx=n)` (`fluid='vort'`) | full field + `np.gradient` | 2D only |
| `interpolate_temporal_flow(t_index=n)` (`fluid='quiver'`) | full field, then `[::M,::N]` | 2D only |
| `interpolate_temporal_mesh(...)` | mesh only, cheap | moving meshes |

Under dynamic loading each of those goes through `FluidData.__call__`, which reloads
from disk when the requested time leaves the resident window. Replaying frames 0..N
therefore slides the window back to the start and forward again — a full second pass.

Two facts drive the whole design:

1. **In 3D, the stats text is the entire per-frame fluid cost.** `fluid='vort'` and
   `'quiver'` are 2D-only, so a 3D frame draws nothing about the fluid — yet
   `_calc_basic_stats` still pulls the whole 3D field every frame to print
   `Fluid v_max` in the corner. The ~100 GB second pass exists to render a text label.
2. **2D and 3D are different problems.** The expensive *visualization* is 2D-only,
   where data usually fits in memory and replay is cheap. The expensive *data volume*
   is 3D, where the only fluid-dependent thing drawn is text.

### 8.2 Scope

- **§8 is a 2D feature.** The 3D deliverable is the statistics fix (§8.3.1) and
  nothing else.
- **3D plotting is a stand-in.** It is matplotlib today, awaiting a vtk-powered
  replacement, at which point it splits out from the 2D path entirely. Therefore:
  **do not invest in matplotlib 3D rendering**, and **do not contort the 2D design to
  stay symmetric with 3D** — shared abstractions would only have to be unpicked.
- **Not in scope:** agent-history retention (recorded in `TODO.md` as a possible
  future feature — a long-run memory question with consumers beyond plotting; nothing
  here depends on it).

---

### 8.3 Component specifications

#### 8.3.1 Frame statistics — **[done]**

**Remove** `avg_spd` and `max_spd` (whole-grid fluid reductions). **Add** the standard
deviation of agent speed. Result: `_calc_basic_stats` needs **no fluid field at any
frame**, in 2D or 3D.

Surviving fluid stats are the component means `avg_spd_x`, `avg_spd_y`, `avg_spd_z`,
served from a **per-dump mean sidecar**: cache `mean(uᵢ)` per component per dump as
each dump loads (a few floats, free), then evaluate exactly at any time via the
interpolation weights (§8.5, linearity). Agent statistics come from `velocities` /
`pos_history` and involve no fluid at all.

Rationale for the substitution, beyond cost: whole-grid reductions include regions
containing no agents. In an agent-based model a statistic over the agent population is
more informative, and the spread of agent speeds speaks directly to whether the
population is moving coherently. Whole-field values remain available on demand via
`FluidData.fmin`/`fmax` and `Environment.get_mean_fluid_speed()`.

Implementation notes:
- **Pair the statistic coherently.** The existing "Agent v̄" shows `‖⟨v⟩‖` — the norm
  of the mean velocity vector, which measures net directed transport and cancels for
  opposed motion. `std(|v|)` beside it is a mismatch. Either also show mean speed
  `⟨|v|⟩` and pair the two, or state that the lines measure different things.
- **Respect the mask** — only in-domain agents contribute. `t_indx == 0` defines
  velocity as zero, so the spread is zero there.
- **Ripple:** the returned tuple is unpacked at **eight** sites in `_swarm.py` (2D and
  3D variants). Private method, so no API promise, but all eight change together.

**As built** (the open call above was decided in favor of pairing):

- `_calc_basic_stats` returns `(perc_left, avg_spd_x, avg_spd_y[, avg_spd_z],
  avg_swrm_vel, avg_swrm_spd, std_swrm_spd)`. Both new agent statistics are computed
  from the same masked-row-filtered velocity data as `avg_swrm_vel`.
- The plot box shows **both** agent quantities, notated to say which is which:
  `Agent $|\overline{v}|$` (the norm of the mean velocity — net transport) and
  `Agent $\overline{|v|}$: m ± s` (mean speed and its spread). The `Fluid v_max` /
  `Fluid v̄` lines are gone; the per-axis `Fluid v̄ₓ` lines on the histogram axes stay,
  now served from the cache.
- The 3D stats box moved from `text2D(0.75, 0.9)` to `0.65` — the `±` line is wider
  than the lines it replaced and ran off the right edge of the axes.
- The sidecar is `FluidData._dump_means`, an `(n_times, n_components)` array of NaN
  filled in by `_record_dump_means` at every point where data lands in memory
  (`__init__`, and all three load sites in `update_spline`). `get_mean_velocity(time=,
  t_idx=)` is the public reader. For cubic splining it evaluates a `fCubicSpline` built
  over the means themselves — same class, same knots, therefore exactly the mean of the
  splined field, since the construction is linear in the data. For linear splining it
  interpolates the sidecar directly against `flow_times` rather than against the
  resident window, so **a mean stays available after the window has moved past it** —
  which is what makes replaying a finished run free. A time whose bracketing dumps were
  never loaded falls back to a load (a cache miss, not a cache lie).
- **Measured effect**, 25-dump IB2d dataset at `INUM=4`, 48 steps, then `plot_all` to a
  movie: fluid loader calls during plotting went **8 → 0** (25 dumps re-read → none).
  With `fluid='vort'` it stays at 8, because the vorticity backdrop genuinely needs the
  field — that is step 4's problem, not step 1's. In 3D, where nothing fluid is drawn,
  the 0-load case is the only case.

#### 8.3.2 Recorder API

**The recorder captures data only — it never renders** (§8.3.7). All video production
belongs to `plot_all`, reading the cache afterwards. So the recorder takes **no video
parameters at all**: no `fps`, no `playback_rate`, no colormap, no figure size. It
takes only what determines *what data is captured*.

`Environment.record(...)` is the implementation; `Swarm.record(...)` is sugar that
delegates with that swarm preselected. (A `Swarm`-level recorder could not express a
multi-swarm capture, since `Environment.move_swarms()` is the multi-swarm path. Joint
multi-swarm plotting is a known gap — issue #49 — and is not solved here, but the API
must not foreclose it.)

```python
with envir.record('run_cache/', fluid='vort') as rec:
    for _ in range(steps):
        rec.move(dt)            # advance + capture
```

Note `fluid='vort'` here means **which fluid quantity to cache**, not what to draw —
the same keyword on `plot_all` selects the backdrop. Same word, different side of the
capture/render line; worth distinct wording in the docstrings.
```python
with envir.record('run_cache/', fluid='vort') as rec:
    for _ in range(steps):
        swrm.move(dt)           # user's own loop body
        do_something_custom()
        rec.capture()           # explicit
```

- `rec.move(dt, **kwargs)` forwards to `move()` / `move_swarms()` and captures — the
  common case, with no way to forget it.
- `rec.capture()` stays available for loops doing work the recorder should not own.
  (Named `capture`, not `frame`: it records simulation state, and frames no longer
  exist at record time.)
- **No auto-hook on `move()`.** Users routinely subclass `Swarm` and override the move
  machinery; a plain `move()` call must not acquire invisible side effects.
- **`__exit__` finalizes on the exception path** — flush and write the cache metadata —
  then **re-raises**. A run that dies at hour eleven of twelve keeps a
  complete-to-that-point, fully renderable cache.
- Returns a handle carrying the cache path, which `plot_all` consumes (§8.3.6).

**Capture schedule.** Agent state is captured **every step by default**. A coarser
schedule may be specified later if memory demands it; the framing is deliberately
*not* "capture every N video frames" but **"as if `dt` were larger"** — the cache then
looks exactly like a run performed at the coarser timestep, and everything downstream
is unchanged with `Δt_capture` substituted for `dt`. Keeping this orthogonal to video
frame rate matters: capture resolution is a **data-fidelity** choice fixed at run
time, frame rate is a **presentation** choice changeable forever after. Conflating
them would be the original `dt`↔`fps` footgun in a new costume.

*Naming hazard:* `dump` is already this codebase's word for **fluid** data dumps
(`d_start`, `d_finish`, `load_dumpfiles`, `loaded_dump_bnds`), and IB2d's
`print_dump` is the same concept for its own output. An agent capture schedule is
conceptually identical but must not be called simply "dump" — always qualify, or use
`capture_interval` and reserve "dump" for fluid.

#### 8.3.3 Derived-quantity cache

Cache **derived quantities, not images**, so re-plotting stays possible without
re-running: colormap, clip, agent subset, figure size and dpi all stay adjustable.
Fixed at record time: the downsample factors and which fluid quantity was recorded.

**Container: a directory of `.npy` files — one per fluid dump, one per agent capture —
plus a metadata sidecar.** `.npz` is unusable here: `np.savez` writes the archive in
one call, so everything would have to be accumulated in memory first, defeating the
streaming property that motivates the whole design (~1 GB for full-resolution
vorticity over 500 dumps). HDF5/zarr would add a required dependency to a deliberately
lean `install_requires`.

**Cadence is hybrid, and neither base is the video frame rate** — frames do not exist
until render time:
- **Fluid-derived quantities (vorticity, downsampled quiver): once per fluid dump.**
  Permitted by linearity (§8.5) — exact reconstruction at *any* time. Usually smaller
  than per-frame would be (149 dumps vs 500 frames for the leaf dataset).
- **Agent-derived quantities: once per capture step** (every simulation step by
  default, §8.3.2).

Together these make the entire frame-rate choice post-hoc: any `Δt_frame ≥
Δt_capture` can be rendered from the same cache.

Per-dump full-resolution vorticity is an accepted disk cost: 2D-only, and only when
the user asks for vorticity. IB2d datasets commonly ship comparable vorticity fields
already.

**Schema — the metadata must carry:**
- format **version**;
- **source fingerprint** (dump range and `flow_times` extent, or a hash) so a cache
  from a different run or dataset is refused;
- **which quantity** was cached (`vort`, `quiver`, or both) and the **downsample
  factors** `M`, `N`;
- the **capture times** (agent time base) and the **dump times** (fluid time base) —
  there are no "frame times", since frames are chosen at render time;
- the **capture interval** actually used, since it is the floor on any later
  `Δt_frame` (§8.3.5);
- **axes**: `flow_points` and domain `L`, so the cache plots without touching fluid;
- **per-dump extrema** for colour normalization (§8.3.4);
- the **per-dump fluid component means** — the §8.3.1 sidecar, a few floats per dump,
  from which the surviving fluid statistics are exact at any time;
- the **agent positions** per capture (`N×D` plus mask). Easy to omit as "already in
  `pos_history`" — but that lives in memory and dies with the process, so without it
  the cache cannot render after a crash or be used in a later session;
- the **agent velocities** per capture. Do **not** plan to re-derive these from
  cached positions: `_calc_basic_stats` currently finite-differences consecutive
  history entries, which is only equivalent to the true velocity when capture is every
  step. Under any coarser schedule the derived value is a smoothed, different quantity
  — and the new agent-speed statistics (§8.3.1) depend on it. Storing velocities
  doubles this part of the cache and removes the trap entirely.

Note what is *not* stored: the `_calc_basic_stats` scalars. With positions,
velocities, and the per-dump fluid means all present, every displayed statistic is
derivable at render time — so caching them too would be redundant state that could
drift from the data it summarizes.

**Validation on load — missing ≠ mismatched:**
- **Mismatched** (wrong fingerprint, wrong grid) → hard refusal with a clear message.
  Silently plotting a foreign cache is the worst available outcome.
- **Missing** (a quantity not recorded) → **fall back to the fluid**. Free when
  `INUM=None`; the §8.3.6 warn-and-re-stream path otherwise. This is what keeps
  vorticity computable after the fact for someone who recorded without it.
- **Never derive vorticity from cached quiver arrays.** They are downsampled, so
  gradients taken on them are a coarser, different field — a plausible-looking wrong
  answer. Recording both `vort` and `quiver` is the cheap prevention.

#### 8.3.4 Video output and colour normalization

**`plot_all` is the sole video producer** (§8.3.7), and it already streams:
`Animation.save()` internally uses `writer.saving(...)` + `grab_frame()` per frame, so
encoding memory is already O(one frame). **No change is required to the video-writing
machinery at all** — the work in §8 is about where the *data* comes from, not how
pixels reach ffmpeg.

**No PNG-frames option.** Every argument for one is covered better elsewhere:
truncation and mid-run inspection by container choice; crash re-render by the cache;
single publication stills by `Swarm.plot(t, filename=...)`; resume is impossible
regardless, as Planktos has no simulation checkpointing.

**Document `.mkv` for long or unattended runs.** A hard kill (HPC walltime, OOM,
node failure) is `SIGKILL`: `__exit__` never runs, the pipe is never closed, and an
`.mp4` is then usually unplayable because ffmpeg writes the `moov` atom last. `.mkv`
survives truncation *and* is playable while still being written, which also covers
checking on a long run mid-flight. Remuxing afterwards is lossless and one call:
`ffmpeg -i out.mkv -c copy out.mp4`. Fragmented mp4
(`-movflags frag_keyframe+empty_moov`) is the alternative, passed via `writer_kwargs`.

**Colour normalization.** Replace the current per-frame `fld.autoscale()` — a drifting
colour scale is scientifically misleading — with a **global scale derived from the
cached per-dump extrema in a second pass over the cache** (small) rather than over the
fluid (huge). Note `FluidData.fmin`/`fmax` are *not* usable for this: they are
documented as covering "all the data seen so far", so under dynamic loading they grow
during the run and would reintroduce the drift. Max-over-dumps is an exact upper bound
under linear interpolation and very tight under cubic. If a live one-pass render mode
is ever offered it has no global scale available and must take an explicit
`clip`/`vmin`/`vmax`, or disclose the drift on the colorbar.

#### 8.3.5 Frame rate: `fps` and `playback_rate`

Users set two quantities they already understand; `dt` leaves the user-facing API
entirely:

| Parameter | Meaning | Default |
|---|---|---|
| `fps` | frames per second of output — *smoothness*, comparable to standard 24/25/30/60 | `10` today; see below |
| `playback_rate` | simulated seconds per second of video — *speed* vs real time | `1` |

```
Δt_frame = playback_rate / fps
```

| `playback_rate` | `fps` | `Δt_frame` | Reads as |
|---|---|---|---|
| 1 | 30 | 0.0333 s | real time, smooth |
| 0.5 | 30 | 0.0167 s | 2× slow motion |
| 10 | 24 | 0.417 s | 10× fast forward |

This replaces a long-standing footgun. With frames pinned to steps, `fps` was the only
lever: at `dt = 1e-3`, real-time playback demanded `fps = 1000`, while the default
`fps = 10` turned 10 s of simulation (10 000 steps, hence 10 000 frames) into a
**17-minute** movie. At `dt = 1e-4` the same settings give 2.8 hours.

- `per_dump=True` is an alternative specifier setting `Δt_frame` = dump spacing; report
  the resulting playback rate back to the user.
- **A raw step count (`every=k`) is rejected** — users vary `dt` between `move()`
  calls, so it silently means different things within one run.
- **`Δt_frame < Δt_capture` is the one failure mode.** Frames cannot be produced
  between captured states. Clamp to every captured state and **warn with the
  numbers**, including the achieved rate `Δt_capture × fps`. Silent clamping would
  reintroduce the footgun in a new form.
- **`fps ≤ playback_rate / Δt_capture`** follows from `Δt_frame ≥ Δt_capture`. With
  the default capture-every-step schedule, `Δt_capture = dt` and this reads
  `fps ≤ playback_rate / dt`. So **slow motion and smoothness trade off unless the
  capture interval is small**: at `dt = 0.025` captured every step, real time reaches
  40 fps but 10× slow motion caps at 4 fps. Document as: *smooth slow motion needs
  fine capture.*
- When rendering from a cache, `Δt_capture` is read from the metadata; when replaying
  live from `pos_history`, it is `dt`. Same rule, one substitution.
- **Frame times are not exactly uniform.** Frames are chosen at render time by picking,
  for each target time, the nearest available capture — so spacing jitters by up to one
  `Δt_capture` whenever `Δt_frame` is not an exact multiple of it. The video is encoded
  at constant `fps` regardless, so this shows as slightly uneven motion. Negligible
  when `Δt_frame >> Δt_capture`; **warn when `Δt_frame` is only a small multiple**
  (say < 3×), where the jitter is a large fraction of the interval.
- **Minor open call for step 2:** `plot_all`'s `fps` default is currently `10`. With
  `playback_rate=1` that yields `Δt_frame = 0.1` s, which is fine but choppy. Raising
  the default to 24 or 30 would look better and is a second (small) behavior change on
  top of `playback_rate`. Decide when implementing; either way it is changelog
  material only if changed.
- **Assumption to document:** "real time" presumes simulated time is in seconds.
  `Environment.units` covers *length* only; seconds is the convention throughout.
- **`fps` is re-encodable after the fact**, because dump-cadence caching (§8.3.3) can
  supply any `Δt_frame`. Only the downsample factors and recorded quantity are fixed.

#### 8.3.6 `plot_all`

- **Reads the cache when given one** (explicit path, or the handle returned by the
  recorder). Frames are then *selected* from the cache's **capture times** — the
  schema's capture-time list is the authority for what can be rendered, and
  `Δt_capture` from the metadata is the floor on `Δt_frame` (§8.3.5). Never assume
  cached entries correspond one-to-one with `pos_history` indices.
- **With no cache after a dynamically-loaded run: still works.** Re-streams as today,
  but emits a loud one-time warning with the estimated cost — detected by `INUM` being
  set and the requested frames spanning more than the resident window. Never break a
  working workflow silently; never let someone accidentally re-stream 100 GB unwarned.
- **`playback_rate=1` becomes the default here too.** Existing scripts will produce
  different videos. Accepted as a deliberate 1.1.0 change: the old behavior *is* the
  footgun.
- With `INUM=None` the whole dataset is in memory, replay costs nothing extra, and
  today's random-access behavior is otherwise preserved.

#### 8.3.7 Separation of concerns: capture vs render

**DECIDED: the recorder captures data only; `plot_all` does all rendering.**

`plot_all` is not made obsolete by the recorder — it drives the interactive on-screen
animation, replay is free when `INUM=None`, and the recorder requires deciding before
the run. The two do different jobs, and this decision draws the line between them
cleanly:

| | Recorder | `plot_all` |
|---|---|---|
| When | during the run | any time after |
| Job | write the cache while data is resident | turn a cache (or live history) into pixels |
| Knows about | fluid dumps, capture schedule | `fps`, `playback_rate`, colormap, clip, figure |

Rationale: the cache already holds everything needed to render, so rendering during
the run buys convenience only — and costs the thing the cache was chosen for. Every
video parameter stays adjustable forever, which an image cache or live rendering would
have re-fixed at run time.

Three consequences worth stating, because they simplify the build:

- **There is exactly one rendering path**, so no shared-renderer refactor is needed.
  `plot_all` keeps `FuncAnimation` and `animate()` essentially as they are; only the
  *source* of per-frame data changes.
- **The recorder takes no video parameters** (§8.3.2), which removes the config-
  duplication problem between it and `plot_all` entirely.
- **The video-writing machinery needs no work at all** (§8.3.4).

---

### 8.4 Build order

1. **[done]** **Frame statistics (§8.3.1)** — independent, low risk, and the **entire
   3D deliverable**. Needs none of the caching machinery, and touches no rendering.
   See "As built" in §8.3.1 for what shipped and the measured effect.
2. **`fps` / `playback_rate` (§8.3.5)** — user-facing, self-contained, removes the
   footgun. Lands entirely inside `plot_all` as a frame-selection computation (it
   already accepts a `frames` iterable), so it needs no caching and no recorder.
3. **Recorder + cache (§8.3.2, §8.3.3)** — the substantial piece, and pure data
   capture: no rendering, no video parameters, no matplotlib.
4. **`plot_all` reads the cache; colour normalization (§8.3.6, §8.3.4)** — the only
   step that touches rendering, and it changes where per-frame data comes from rather
   than how it is drawn.
5. **Examples and docs rewrite (§8.6)**.

Steps 1 and 2 are independently shippable and require no architectural commitment.
**Re-evaluate 3–4 after they land**: given that 3D plotting is awaiting a vtk rewrite
(§8.2), steps 1–2 may be the whole justified investment for now, with the cache
warranted only if 2D re-plotting turns out to be a real workflow pain.

*Two earlier versions of this list are worth not repeating.* One had "stream the
video" as an independent step, on the false premise that `plot_all` held frames in
memory — see the correction at the head of §8; `plot_all` has always streamed. The
other had "extract a shared frame renderer" as a prerequisite, which the §8.3.7
capture/render split removes: with exactly one rendering path there is nothing to
share.

#### 8.4.1 Entry points for a cold start

Line numbers drift; search for the names.

**Step 1 — frame statistics. [done]** Kept as the record of what the step involved.
- `Swarm._calc_basic_stats` (`planktos/_swarm.py`) — the 2D branch and the 3D branch
  each build the tuple; both change.
- **Eight unpack sites** in `_swarm.py` consume that tuple (in `plot`, `plot_all`, and
  the movie-writing paths, 2D and 3D variants). `grep -n "_calc_basic_stats" planktos/_swarm.py`
  finds all of them. The display strings alongside them (`Fluid $v_{max}$`,
  `Fluid $\overline{v}$`) are what change for users.
- ⚠️ **Four existing tests pin the behavior being removed** and must be rewritten as
  part of this step, not treated as breakage:
  `tests/test_flow_interface.py::test_calc_basic_stats_2d_known_answers`,
  `::test_calc_basic_stats_3d_known_answers`,
  `::test_calc_basic_stats_time_varying_uses_requested_time_index`,
  `::test_calc_basic_stats_returns_plain_scalars`. The first two assert
  `max_spd == max speed` explicitly (that assertion was itself a bug fix — see §3.4),
  so deleting the statistic means deliberately retiring a regression lock. Replace with
  equivalents for the agent-speed spread; keep the component-mean assertions.
- The per-dump mean sidecar belongs in `FluidData` (`planktos/fluid.py`), populated
  where dumps are loaded (`load_dumpfiles` / `update_spline`) so it costs nothing.

  **Tests as landed.** All four were rewritten. The retired `max_spd` regression lock
  is replaced by `test_calc_basic_stats_agent_speed_vs_mean_velocity`, which pins that
  `⟨|v|⟩` and `‖⟨v⟩‖` are genuinely the different quantities the plot now claims (four
  agents, two at +1 and two at −3 in x: `‖⟨v⟩‖ = 1`, `⟨|v|⟩ = 2`, `std = 1`). The
  strongest new test is `test_calc_basic_stats_pulls_no_fluid_field`, which monkeypatches
  `FluidData.__call__` and `Environment.interpolate_temporal_flow` to raise — reaching
  for the field is now a hard failure rather than a silent cost. `get_mean_velocity` is
  covered in `test_flow_interface.py` (static, cubic, linear, `t_idx`, extrapolation,
  the "requires a time" error) and, for the sliding window, in a new section of
  `test_dynamic_loading.py` — including that a **replay after a full sweep triggers
  zero loads**, and the jump-to-start fast path records means too.

**Step 2 — `fps` / `playback_rate`.**
- `Swarm.plot_all` signature (`fps`, and the `frames` argument it already accepts) —
  the change is a frame-*selection* computation feeding `frames`, plus the clamp/warn
  checks. `frames=None` currently expands to `range(len(self.pos_history)+1)`; that
  expansion is what `playback_rate` replaces.
- `FuncAnimation(..., interval=...)` controls **on-screen** playback and is separate
  from the saved-video `fps`. Do not conflate them.
- `Swarm.plot` (single frame) snaps a requested `t` to the nearest
  `Environment.time_history` entry without interpolation; that behavior is unchanged.

**Verification.** `pytest` (fast, ~1 s) plus `pytest --runslow` for the plotting
smokes, which exercise `plot_*` on the Agg backend and will catch signature breakage.
The movie test additionally needs ffmpeg on `PATH`.

### 8.5 The property everything rests on

Both spline classes evaluate as a **weighted sum of nodal fields**,
`u(t) = Σᵢ wᵢ(t)·uᵢ`, for `LinearSpline` and `fCubicSpline` alike. So any **linear**
functional of the field commutes with temporal interpolation:

```
F(u(t)) = Σᵢ wᵢ(t)·F(uᵢ)          for linear F
```

`mean`, `np.gradient` (hence vorticity), and subsampling (hence quiver arrays) are all
linear. This is what makes the per-dump mean sidecar exact (§8.3.1) and dump-cadence
caching exact (§8.3.3), using weights the interpolator already computes.

`max` and `mean(√(u²+v²))` are **not** linear and do not commute — which is why
`max_spd` and `avg_spd` were dropped rather than cached.

### 8.6 Obligations

- **Changelog (1.1.0)**, both user-visible relative to 1.0.x:
  - **[done]** fluid speed statistics replaced by agent-speed spread on plots;
  - `playback_rate` added and defaulting to 1, changing existing video output.
- **Examples rewrite.** The plotting portions change regardless. Current effective
  playback rates (`dt × fps`) show the footgun's fingerprint — a 27× spread with no
  evident intent:

  | Example | `dt` | `fps` | Effective rate |
  |---|---|---|---|
  | `ex_ib2d_ibmesh.py` | 0.025 | 3 | 0.075 — 13× slow motion |
  | `ex_ib2d_sticky.py` | 0.025 | 3 | 0.075 — 13× slow motion |
  | `ex_ib2d_mvbnd_sticky.py` | 0.025 | 6 | 0.15 — 6.7× slow motion |
  | `ex_ind_var.py` | 0.1 | 20 | 2.0 — 2× fast forward |

  Under today's scheme `Δt_frame = dt` identically, so the effective rate is just
  `dt × fps` — users could only choose `fps`, and the playback rate fell out wherever
  it fell. That is why the spread is incoherent: nobody chose these rates.

  When rewriting, choose the **playback rate** deliberately and keep it near current
  behavior where that makes sense — the fluid examples genuinely want slow motion for
  legible vortices. Then note the real constraint: at `dt = 0.025`,
  `playback_rate = 0.075` permits at most 3 fps, so a smoother version of those
  examples needs a **smaller `dt`**, not a different `fps`.
- **Docs:** the `fps`/`playback_rate` model and its `dt` ceiling; the seconds
  assumption; `.mkv` guidance for long runs; what the cache stores and when it is
  refused.

### 8.7 Deferred within §8

- **Async frame writing.** Matplotlib rendering is slow and currently serializes with
  the physics. Matplotlib is not thread-safe, but rendering in the main thread and
  handing only the encode/write to a writer thread would hide most of the I/O cost.
  **Measure before building** — it may be irrelevant next to the physics.
- **A live one-pass render mode** (rendering without a cache). Only meaningful if a
  workflow appears that cannot afford the cache; it inherits the colour-normalization
  problem (§8.3.4).

### 8.8 Interaction with §9

Position-wrapping tiling pairs naturally with this: a tiled quiver wraps coordinates
the same way the interpolator does, so plotting never materializes a tiled array.
Doing §8 first gives §9 a working renderer to validate tiled visualization against.

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
- **§8 (plotting streaming) has had its design pass** and is an implementation plan
  with a build order (§8.4), all design questions settled.
- **§8 step 1 (frame statistics) is done** — see "As built" in §8.3.1. The fluid speed
  reductions are gone, agent mean speed and spread replace them, and the surviving
  fluid component means come from `FluidData.get_mean_velocity` and its per-dump
  cache. Verified beyond the suite by running the examples: a windowed IB2d run
  followed by `plot_all` went from 8 loader calls (25 dumps re-read) to 0, and the 3D
  time-varying path was driven end to end against `tests/IBAMR_test_data`.
- **Next actionable step: §8 step 2** (`fps` / `playback_rate`). Like step 1 it is
  independently shippable and needs no architectural commitment; re-evaluate 3–4 after
  it lands.
- **Not started:** §8 steps 2–5, and §9 (which still needs its design pass).
- **Pre-existing breakage found while verifying step 1, unrelated to it** (all three
  reproduce identically against the pre-step-1 package, so they are not regressions —
  recorded here so the next session does not re-diagnose them):
  - **[fixed]** `_environment.py` `calculate_FTLE` read `self.props_history` where it
    meant `s.props_history`, so the user-`swrm` branch raised `AttributeError`. Fixing
    that exposed a second defect in the same block — the shallow `copy.copy` left
    `vel_history`/`props_history` aliased to the caller's Swarm — so both were fixed
    together. See TODO.md.
  - **[fixed]** `_environment.py` `read_IB2d_mesh_data` dereferenced
    `self.flow.fluid_domain_LLC` without checking `self.flow is not None`, so loading a
    mesh into a fluid-free environment raised. This one *is* a `dyload` regression —
    on `master` `fluid_domain_LLC` is an `Environment` attribute that always exists, so
    it only became reachable when the attribute moved onto `FluidData`. See TODO.md.
  - `_ibc._project_and_slide_static` raises
    `ValueError: operands could not be broadcast together with shapes (3,2) (3,)` on
    the ib2d channel mesh when agents are seeded uniformly across the domain (rather
    than at the example's point source). This one is in the load-bearing collision
    code and deserves a real look.
- **`tests/IBAMR_test_data/` is present** (`IBAMR_db_003/004/005.vtk`,
  `mesh_db.vtk`) — 3D IBAMR data for the `@vtk`-marked tests and for validating
  §9. Its absence on the original authoring machine is what led to the since-
  dropped 2D tiling stopgap.

**The immediate next actionable step** is §8.4 step 2 (`fps` / `playback_rate`), step 1
having landed. §9 (real tiling, and whether `extend` returns) follows the rest of §8,
and still needs its own design pass.

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
