# Fluid velocity field interface — analysis and refactor plan

Status: **plan / design note** (2026-07). Records the decisions reached while
investigating the `dyload` Phase-0 "`FlowArray` breaks numpy interop" item. Only
the `fmin`/`fmax` fix has been implemented so far; everything else here is the
agreed plan, to be executed in the sequence in §8.

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

2. **Defer tiling.** Near-term applications do not need it; it was only elevated
   because dynamic loading is memory-management work and tiling felt entangled. It
   is in fact separable from dynamic loading (§1 hat C is independent of
   `FlowArray`). See §5 for the stopgap and the eventual real implementation.

3. **The naming rule for tiling (adopt now, even though tiling is deferred):**
   > **Public geometry (domain `L`, plot extent, reported grid shape) reflects the
   > *tiled* domain. Stored data and all interpolation use the *base tile*.
   > Nothing materializes the full tiled grid-data array on a hot path.**
   If a consumer ever truly needs the big array, that is one explicit
   `materialize_tiled()`-style method that documents its memory cost. This is the
   single, predictable mental model the `FlowArray` subclass was groping toward.

4. **`fmin`/`fmax` → tuples.** Done (§3.3).

---

## 5. Tiling: stopgap now, real implementation later

**Stopgap (to keep existing 2D examples running after `FlowArray` is deleted):**
`tile_flow`/`tile_domain` **materializes** the tiled arrays in **2D**. 2D tiles are
small, so the memory cost is acceptable as an interim measure. **3D tiling raises
`NotImplementedError`** until the position-wrapping implementation lands.
- Known casualty to flag/revisit: `examples/ex_IBAMR_ibmesh.py` calls
  `tile_domain(3,3)` on 3D IBAMR data — this will hit the `NotImplementedError`
  until the real implementation exists. (Tiling is otherwise a 2D-only feature
  today per CLAUDE.md.)

**Eventual real implementation — position-wrapping, memory-free, 3D-ready:** a
tiled domain is a periodic extension, and `interpolate_flow` *already* implements
periodic extension by wrapping query positions
(`positions[:,n] % flow_points[n][-1]`). So tiling never needs a big array on any
hot path:
- store `tiling = (tx, ty[, tz])` + the one base tile on `FluidData`;
- interpolation wraps agent positions into the base tile, then `interpn` against
  the base tile — identical in 2D and 3D;
- vorticity/`|u|`-gradient over a tiled domain = the base tile's field replicated,
  so compute on the tile and replicate the *result* only if a consumer needs the
  full field;
- reported `.shape` / domain extent = arithmetic on `base_shape × tiling`, no
  allocation.

Reconcile with `periodic_dim`: a tiled dimension is effectively periodic for
interpolation. The implementation must define the interaction of `tiling` and
`periodic_dim` explicitly (a tiled dim implies wrapping regardless of the
`periodic_dim` flag for that dim).

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
  `fshape`, `fmin`/`fmax` (tuples), `fluid_domain_LLC`, `L`; methods `tile_flow`
  (2D-materialize stopgap / 3D `NotImplementedError`), `get_vorticity`,
  `get_dudt`, `calculate_DuDt`. Under tiling, public geometry follows the §4.3
  naming rule.
- The spline classes (`fCubicSpline`, `LinearSpline`) keep their `__call__` /
  `__getitem__` / `min`/`max`/`absmax` / `regenerate_data` surface, but return
  **plain ndarrays** rather than `FlowArray` views.

---

## 7. Implementation sequence & test strategy

Correctness-first: pin trusted behavior before removing anything.

1. **[done]** Fix `fmin`/`fmax` → tuples; add a regression test (unpack `fmax`
   twice; and, once IB2d dynamic data exists, survive a window slide).
2. **Pin current trusted behavior** with tests before deleting `FlowArray`:
   the static/time-varying `interpolate_flow` values, vorticity, `save_fluid`
   round-trips, `_calc_basic_stats`, and the 2D tile-then-interpolate result
   (which currently raises — capture the *intended* post-stopgap values).
3. **Delete `FlowArray`;** make static components and spline returns plain
   ndarrays. Drop the `np.asarray(...)` workarounds in tests and `save_fluid`.
4. **Tiling stopgap:** 2D `tile_flow` materializes; 3D raises `NotImplementedError`.
5. Run full suite (`pytest`, then `--runslow`). Confirm the workarounds are gone
   and nothing regressed.
6. Update `changelog.txt` (user-facing: interop fix, tiling behavior change) and
   this note.

Defer to a later pass (own design work): the position-wrapping tiling
implementation (§5) and the plotting streaming redesign (§8).

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
- pairs naturally with position-wrapping tiling: a tiled quiver wraps coordinates
  the same way the interpolator does, so plotting never materializes a tiled array.
- open questions: keeping interactive/exploratory (random-access replay) plotting
  working alongside the streaming one-pass mode; the API surface; decoupling
  frame-rate from step-rate.

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
  `_environment.py` `save_fluid`.

---

## Session handoff / cold start

If you are picking this up in a fresh session with no prior context, read this
whole note plus `CLAUDE.md` and `TODO.md` (root) first. Quick orientation:

**Where we are (as of the commit that adds this section):**
- Branch `dyload`. This is the "dynamic loading of fluid data" feature branch.
- **Done and committed:** the `fmin`/`fmax` generator fix (§3.3) and this note.
  Nothing else in the refactor has started.
- **Not started:** deleting `FlowArray` (§4.1), the tiling stopgap (§5), the
  position-wrapping tiling implementation, and the plotting streaming redesign.
- **Git state to reconcile before working:** `master` advanced to `1.0.1`
  (documentation-only, plus a minor code fix) and those changes were merged into
  `origin/dyload`. A `git pull` into local `dyload` is expected right after the
  commit that introduced this section. **First thing to do on resume:** pull, then
  confirm the `fmin`/`fmax` fix (three `tuple(...)` sites in `fluid.py`
  `__init__` + both `update_spline` slide branches) and this note both survived
  the merge, and re-run `pytest` (expect all green, ~160 passed + skips).

**The immediate next actionable step** is §7.2: *pin current trusted behavior with
tests before deleting `FlowArray`.* Do not delete `FlowArray` until that safety net
exists — correctness-first (see `CLAUDE.md`). After §7.2, proceed through §7.3-7.6.
Also update `TODO.md` Phase 0: the `fmin`/`fmax` item is now done, and the
`FlowArray` interop item is superseded by this note's plan (the deeper finding is
§3.2 — tiling-through-`interpn` is defeated by modern scipy, so the fix is deletion
+ deferral, not patching the subclass).

**Re-confirming the evidence (optional, ~10s each).** The failure modes and the
fix are reproducible from the repo root:

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
