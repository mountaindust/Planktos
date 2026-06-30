# TODO — `dyload` branch (dynamic loading of fluid data)

**Goal of this branch:** load/spline time-dependent fluid velocity data *on demand*
(a sliding window of timesteps) instead of holding the whole dataset in memory, so
that large 3D time-varying flows (~100 GB raw, larger once splined) can be used.

**Current state (2026-06-28):** the architecture is built — all fluid data is now a
`FluidData` object (`planktos/fluid.py`, ~2300 lines), dynamic windowed loading is
implemented and reported working for 2D IB2d data, and 3D (`VTK3dData`) is wired up
but unexercised. Temporal interpolation of dynamically-loaded data is **linear in
time** (`LinearSpline`); full-dataset loading defaults to **cubic in time**
(`fCubicSpline`). See the design-history section at the bottom for the cubic→linear
story.

**Just merged:** the large `mvbnd` **test-suite overhaul** (fast, modular tests; 4
latent defects found and fixed; no xfails) plus bug fixes. Post-merge the package
imports and the whole collision / geometry / swarm / periodic / parallel core passes:
**120 passed, 32 failed, 7 skipped.** All 32 failures are overhaul tests that exercise
the *`Environment`* fluid API which dyload moved onto `FluidData` — adapting them is
Phase 0 below. The package source itself carries no merge-introduced breakage.

Priority key: 🔴 do first · 🟡 next · 🟢 later · ⚪ deferred / low priority.

---

## Phase 0 — Adapt the overhauled suite to dyload's `FluidData` API + fix real bugs 🔴

The overhaul's tests were written against mvbnd's `Environment` fluid API. On dyload
that API moved onto `FluidData`. Most of the 32 failures are mechanical renames, but a
few are genuine dyload bugs or deferred ports — triage each as **(rename)** vs **(real
bug)** vs **(port)**. Goal: green suite = trustworthy baseline before Phase 1.

Common renames: `envir.flow_points`→`envir.flow.flow_points`,
`envir.flow_times`→`envir.flow.flow_times`, `get_2D_vorticity`→`get_vorticity`,
`envir.tile_flow`→`envir.tile_domain`. `Environment.extend` was **removed** on dyload.

- [x] **`test_flow_generation.py`** — DONE (10 passed, 1 skipped). Renames +
  `tile_flow`→`tile_domain`; `extend` test skipped (`Environment.extend` removed on
  dyload). Surfaced the `FlowArray` numpy-interop bug (below).
- [x] **`test_temporal_interp.py`** — DONE (7 passed). `create_temporal_interpolations`
  is gone on dyload (absorbed into `FluidData`); rewrote the two tests against
  `FluidData` / `fCubicSpline` directly, keeping the off-node cubic-reproduction check.
- [x] **`test_analysis.py`** — DONE (17 passed). Vorticity renamed to `get_vorticity`/
  `flow.flow_points`; the 3 FTLE value tests now pass after the periodic-default fix
  (below). Still TODO (not a failure): **add the 3D vorticity known-answer test** the
  overhaul deferred to the dyload merge (`FluidData.get_vorticity` supports 3D).
- [x] **`test_io_loaders.py`** — DONE (10 passed, 1 skipped; COMSOL `@vtu` skip).
  Renames (`flow.flow_times`/`flow.flow_points`) fixed the 2 IBAMR loads. **Source fix:**
  `save_fluid`/`save_2D_vorticity` were latently broken on dyload — they passed `self.L`
  (domain lengths) to writers that expect coordinate arrays, and had no static-flow
  guard. Corrected to pass `self.flow.flow_points` + a static guard (this also fixes the
  earlier merge resolution, which had restored dyload's broken versions). Two static
  asserts use the `np.asarray` FlowArray workaround.
- [x] **`test_material_derivative.py` + `test_agent_models.py` massive-particle** — DONE.
  Was **not** a 3D broadcast bug (that label came from the old `test_massive_physics`);
  the focused tests pinpointed two real, dimension-agnostic bugs, both fixed:
  - **(A)** `Swarm.get_dudt` called `self.envir.dudt(...)`, but dyload renamed that to
    `Environment.get_dudt` (a leftover-rename from the FluidData move that came in via
    the mvbnd merge) → `AttributeError`. Fixed `_swarm.py` to call `get_dudt`.
  - **(B)** `FluidData.get_dudt`'s out-of-range branch (`fluid.py`) was wrong two ways:
    it used `<=`/`>=` (spuriously zeroing the derivative *at* the data endpoints t0/tN)
    and built the zeros with `self.fshape` (which includes the time axis for time-varying
    flow) → a time-series-shaped array that broadcast-failed in `calculate_DuDt` at a
    boundary time. Fixed to strict `<`/`>` and `self.fshape[1:]`.
  - Added `test_dudt_time_boundaries_and_extrapolation` pinning endpoint + extrapolation
    behavior; updated the file's helpers to `envir.flow.flow_points` (dyload API).

### Other real bugs that matter (fix in Phase 0)

- [ ] **`FlowArray` breaks numpy interop** (found while adapting `test_flow_generation`).
  `__array_finalize__` propagates `self.array` to every derived array, and the
  overridden `shape`/`__getitem__` read from `self.array` rather than the array's own
  buffer — so a `FlowArray` produced by a ufunc/comparison reads stale data. Result:
  array-wide `np.allclose`/`np.isclose` give wrong answers and even `repr()` raises on
  a `FlowArray` (`fluid.py:103-265`). User-facing (people run numpy ops on flow). Fix
  carefully (delicate view/tiling machinery) with a dedicated test covering allclose/
  isclose/arithmetic/printing on both tiled and untiled `FlowArray`s. Workaround in
  tests for now: `np.asarray(envir.flow[i])` before array-wide numpy calls.
- [x] **FTLE wrong values — DONE.** Root cause was **not** the FTLE math (byte-identical
  to mvbnd) but a **periodic-by-default** bug: `FluidData` defaulted `periodic_dim=True`,
  and the bare `flow=` constructor + analytic setters never overrode it, so every such
  flow was treated as periodic. `interpolate_flow` then wraps the upper grid edge to the
  lower (`pos % flow_points[-1]`, so `y=L → y=0`); FTLE seeds tracer particles exactly on
  the domain edge, so the top-edge seeds (max velocity) read `u_x(y=0)=0`, never advected/
  exited, and corrupted the boundary-row flow-map gradient → spurious large FTLE that
  `nanmax` picked up. **Fix (Approach 1):** default `FluidData.periodic_dim=False`; thread
  a `periodic_dim` kwarg through `Environment(flow=...)` and the analytic setters; loaders
  keep their explicit values (IB2d `True`, VTK3d `(T,T,F)`, COMSOL `(F,F,F)`, NetCDF
  `False`). Periodicity stays independent of `bndry`. Regression tests:
  `test_flow_{non_periodic_by_default,periodic_dim_true_wraps}_at_upper_edge`; the FTLE
  closed-forms now pass. NB: this was a general latent bug (any flow sampled exactly at
  the upper/right edge), not FTLE-specific — FTLE just exposed it.
- [ ] **`FluidData.fmin`/`fmax` are generators, not values.** Built as generator
  *expressions* (`fluid.py:1069-1070`) then re-bound in `update_spline` as
  `(min(self.fmin[n], ...) for ...)` (`fluid.py:1206-1207, 1266-1267`) — subscripts a
  generator and will `TypeError` if consumed after a window slide. On the dynamic path.

### Cleanup (low urgency)

- [ ] 🟢 **Orphaned discarded code:** `fCubicSpline._left_based_cspline` /
  `_extend_prev_spline` (`fluid.py:581-763`) — the abandoned cubic-window approach,
  now unreachable (the only `fCubicSpline(...)` caller uses default `bc_type`). Remove
  or annotate as "abandoned — see history."

---

## Phase 1 — Test dynamic loading in 2D 🟡

Use 2D IB2d data (cheap, deterministic, reported working). Separate two questions:

- [ ] **(A) Machinery correctness — exact.** Dynamic windowed-linear (`INUM=k`) must
  return **identical** values (machine precision) to full linear (`INUM=True`) at every
  query time — linear interp is local, so window-sliding can't change the value. A
  strong, exact, cheap regression test of `update_spline`, independent of (C).
- [ ] **(B) Window-sliding behavior.** Forward slide, backward slide, the
  "jump to beginning" fast path, dataset-end extrapolation flips
  (`update_spline`, `fluid.py:1153-1268`). Assert the loaded window stays bounded.
- [ ] **(C) Comparability — the key scientific question.** Quantify dynamic-linear
  (`INUM=k`) vs. full-cubic (`INUM=None`) error and **record a number**. Only ever
  checked visually so far (`tests/manual/visualtest_2d.py`).
- [ ] **(D) `get_dudt` under linear splining** is a piecewise-constant, discontinuous
  finite difference (`LinearSpline.derivative`, `fluid.py:479-494`). Pin current behavior.
- [ ] **(E) Tiling/periodic × dynamic.** `FlowArray` view + `tiling` propagation through
  `update_spline` (there are `assert ... "Tiling did not propagate correctly"` guards).

---

## Phase 2 — Test dynamic loading in 3D 🟡 (blocks 3D moving boundaries)

The actual end goal (the ~100 GB case). **Needs real 3D dynamic fluid data** — the user
has a sample from their collaborator to stage on this machine when we reach this step.

**Assume a rectilinear fluid grid.** Data is expected as a sequence of **vtk files
exported from VisIt/ParaView**, where the source field (IBFE SAMRAI / OpenFOAM FEM) was
*already* interpolated onto a rectilinear grid externally. Planktos just reads that
rectilinear vtk — source-specific ingestion is **out of scope** (see CLAUDE.md "3D
fluid data sources").

- [ ] Stage the real 3D dynamic dataset on-machine (gitignored; `proj_dev/` convention);
  record the path/loader call.
- [ ] End-to-end `VTK3dData` dynamic load of rectilinear vtk via
  `read_IBAMR3d_vtk_data(..., INUM=...)`; un-skip / fix the IBAMR load tests on real data.
- [ ] Re-run Phase 1 (A)–(E) equivalents in 3D.
- [ ] 3D material derivative end-to-end (after the Phase 0 `calculate_DuDt` fix) for
  massive / inertial particle models.
- [ ] **Memory profiling:** confirm RAM stays bounded to one window across a long 3D run.

---

## Phase 3 — 3D moving immersed boundaries 🟢 (future)

Blocked on Phase 2. Moving boundaries are currently 2D only. 3D immersed boundaries are
**STL triangular (FEM) surface meshes** (3D vertex-point input deprecated; 2D vertex
points still used). Inherited blockers from the overhaul's notes:

- The 3D *moving*-mesh code path currently raises (not implemented; blocked on dyload).
  Static 3D collision coverage is already in place (`test_collisions_static_3d.py`,
  `test_collisions_stl_3d.py`).
- **Moving-mesh FTLE:** `calculate_FTLE` never advances `envir.time`, so a moving mesh
  is frozen at t0; it raises `NotImplementedError`. A real fix threads integration time
  into `interpolate_temporal_mesh` (forward + reversed) — delicate collision-path work.

---

## Inherited follow-ups from the mvbnd overhaul (non-blocking) 🟢

- [x] **`motion.RK45` contract** — DONE on mvbnd (commit `890113b`), merged in: public
  contract pinned with tests (`test_agent_models.py`) + docstring clarified. Passes on dyload.
- [x] **Plotting smoke tests** — DONE on mvbnd (commit `a013dbd`), merged in:
  `test_plotting_smoke.py` (Agg-backend "runs without error" smokes). Adapted the 2
  `flow_points` setters to `flow.flow_points`; passes/skips on dyload (headless).
- [ ] **Backward FTLE for non-tracer models** is intentionally unsupported (reverse-time
  dissipative blow-up). Needs a stabilized/adjoint approach if ever wanted.
- [ ] **Diffusion-statistics test** (`test_agent_models.py::test_brownian_diffusion_statistics`)
  uses 20k agents, fixed seed, ~10% tolerance. If flaky, tighten seed/count, not tolerance.

---

## Deferred / low priority ⚪

> **Source-specific fluid ingestion is out of scope for this branch.** We assume the
> fluid arrives as rectilinear vtk (pre-interpolated in VisIt/ParaView). Reading IBFE
> SAMRAI / OpenFOAM / COMSOL directly — including porting the old VisIt
> `read_IBAMR3d_py27.py` SAMRAI→vtk script — is lower priority than 3D moving boundaries
> and scratched here. Background is in CLAUDE.md ("3D fluid data sources").

- [ ] **COMSOL VTU loader** (`ComsolVTUData`) — existing, full-load only. Verify only if
  needed; collaborator no longer uses COMSOL and the export format has likely changed.
  Also: the skipped `test_io_loaders.py::test_vtu_load` needs a committed COMSOL fixture
  (`tests/data/comsol/vtu_test_data.txt`) or stays gated.
- [ ] **NetCDF** (`load_NetCDF` / `read_NetCDF_flow`) — existing, full-load only. Never
  actually used (reviewer-requested for a prior publication). Lowest priority.
- [ ] **Rectilinear (non-uniform) grid support in `calculate_FTLE`**. Relevant since we
  assume a rectilinear fluid grid, but a diagnostic and non-blocking.
- [ ] Changelog housekeeping (`changelog.txt`, 1.1.0): drop "TODO: test dynamic loading"
  once Phases 1–2 land; resolve the `tiling`-setter TODO (make tiling a setter of
  `FluidData.tiling`, with `Environment.L` updating off it).
- [ ] `Environment.extend` was removed (extrapolation is the intended replacement). Re-add
  only if there's demand.

---

## How to run the tests

- Fast loop: `pytest` (≈1s; skips `slow` / `vtk`-absent / `vtu`-absent).
- Full: `pytest --runslow` (adds the parallelization checks, ~30s).
- Regenerate IB2d fixtures after editing the generator:
  `python tests/fixtures/_gen_fixtures.py`.

---

## Design history & rationale (how we got to linear-in-time)

The branch set out to keep **cubic-in-time** interpolation while streaming windows. The
path tried and discarded:

1. **No resplining** — resplining each window shifts polynomials and makes derivatives
   discontinuous at breakpoints (`ec7b3b8`), so windows must be stitched.
2. **`valid_times` → deletion** — first a `valid_time_bnds` attribute tracked the
   trustworthy part of a freshly-splined window (`9710f98`); replaced hours later by
   simply **deleting** the boundary-contaminated coefficients (`7b385d7`, `trim_end`).
3. **Left-based cubic spline — failed.** `_left_based_cspline` / `_extend_prev_spline`
   forced both boundary conditions onto the *left/known* end so the window could grow
   rightward. Abandoned as **numerically unstable** (`bbd093b`).
4. **Pivot to `LinearSpline`** (`f70cc99`, `9183c0f`): piecewise-linear in time —
   unconditionally stable, trivially window-extensible (carry two raw boundary values,
   no derivatives to match), needs less held data. Dynamic load then worked for IB2d
   (`a61c7fc`).

**The tradeoff (dynamic linear vs. full cubic):** smoothness C²→C⁰ (velocity kinks at
each timestamp); between-sample accuracy ~O(Δt⁴)→O(Δt²); ∂u/∂t becomes a
piecewise-constant step function (feeds `get_dudt` → material derivative → inertial
models). Full **cubic** stays the default for in-memory datasets (`INUM=None`); **linear**
is the price of *dynamic* loading. Quantifying that gap is Phase 1(C).

`INUM` regimes: `None` = cubic, all in memory (default/trusted) · `True` = linear, all in
memory · odd `int` = dynamic windowed linear (`INUM` intervals held at a time).

*(Supersedes the older `planktos/TODO for dynamic loading.txt`, folded in here, and the
mvbnd overhaul's `TODO.md`, whose non-blocking follow-ups are merged into the sections
above.)*
