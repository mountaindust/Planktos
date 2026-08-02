# TODO — `dyload` branch (dynamic loading of fluid data)

**Goal of this branch:** load/spline time-dependent fluid velocity data *on demand*
(a sliding window of timesteps) instead of holding the whole dataset in memory, so
that large 3D time-varying flows (~100 GB raw, larger once splined) can be used.

**Current state (2026-07-31):** the architecture is built and the API has settled —
all fluid data is a `FluidData` object (`planktos/fluid.py`), dynamic windowed
loading is implemented and reported working for 2D IB2d data, and 3D (`VTK3dData`)
is wired up but unexercised. Temporal interpolation of dynamically-loaded data is
**linear in time** (`LinearSpline`); full-dataset loading defaults to **cubic in
time** (`fCubicSpline`). See the design-history section at the bottom for the
cubic→linear story.

**Phase 0 is essentially complete and the suite is green:** **495 passed, 22
skipped** with `pytest`, **515 passed, 2 skipped** with `pytest --runslow`. No
failures, no xfails. Every test-adaptation item under Phase 0 is done. The
`fmin`/`fmax` generator bug is fixed. The `FlowArray` numpy-interop item turned out
to be the tip of a larger design problem and has been **superseded by a dedicated
plan** — see below.

**The flow-field interface refactor** (`docs/notes/flow_field_interface.md`) is
**through step §7.6 — the core work is done.** Investigating the `FlowArray` interop
bug showed that `FlowArray`'s sole reason to exist — virtualizing tiled flow for
`interpn` — is **defeated by modern scipy** (`RegularGridInterpolator` calls
`np.asarray` on any array-API object, discarding the subclass's virtual
`.shape`/`__getitem__`), so the tiled interpolation path was broken *and* untested.
What landed:

- **§7.2** — `tests/test_flow_interface.py` (40 tests) pins the flow-interface
  contract: `interpolate_flow` values, the container/spline surface, `fmin`/`fmax`,
  `_calc_basic_stats`, `get_raw_loaded_data`, the `LinearSpline`/`INUM` temporal
  path, and 3D vorticity. Writing it surfaced three live bugs, all fixed:
  - **(note §3.4)** `max_spd` on every plot frame reported max |u| rather than the
    max fluid speed; `get_mean_fluid_speed` returned a value misreporting its shape.
  - **(note §3.5)** `get_raw_loaded_data` returned `LinearSpline` objects instead of
    ndarrays on the **entire dynamic-loading path** — it dispatched on "is it an
    fCubicSpline" and the else-branch assumed static flow. Fixed by giving
    `LinearSpline` the `regenerate_data` method `fCubicSpline` already had and
    branching on `flow_times is None`.
- **§7.3** — **`FlowArray` is deleted.** Velocity components are plain ndarrays
  everywhere; every `np.asarray` workaround is gone. The §7.2 suite stays green
  *with the wrappers stripped out*, which is what makes this provably
  behavior-preserving.
- **§7.4** — **tiling raises `NotImplementedError`** in 2D and 3D
  (`FluidData.tile_flow`, `Environment.tile_domain`), the latter before mutating
  anything so no half-tiled environment is possible. Affected examples and docs
  carry a notice.

**Next: §8 (plotting streaming), then §9 (real position-wrapping tiling for 2D and 3D,
and whether `Environment.extend` returns).**

- **§8 step 1 (frame statistics) is done.** The whole-grid fluid speed reductions
  (`avg_spd`, `max_spd`) are gone from every plot; agent mean speed and its spread
  replace them, and the surviving fluid component means come from a per-dump mean
  cache on `FluidData` (`get_mean_velocity`) rather than the field. A plot or movie now
  costs **zero** fluid loads unless a fluid backdrop is actually drawn — measured 8 → 0
  on a windowed 25-dump IB2d run. This was the entire 3D deliverable of §8. See the
  note's §8.3.1 "As built".
- **§8 step 2 (`fps`/`playback_rate`) is done.** `plot_all` no longer renders one frame
  per timestep: frames are placed `playback_rate/fps` apart in *simulated* time and
  show the nearest captured state, so speed (`playback_rate`, default 1 = real time)
  and smoothness (`fps`, default unchanged at 10) are chosen separately and `dt` leaves
  the plotting API. Asking for frames between recorded states clamps with a warning
  that carries the numbers; an explicit `frames` list still overrides everything.
  On-screen playback now uses the same rate. It is one new parameter and one private
  method (`Swarm._select_frames`) — the note's §8.3.5 `per_dump` bullet was
  deliberately not built, and an initial module-level/method split was collapsed.
  `tests/test_frame_selection.py` (19 fast, closed-form tests) pins it, and the example
  call sites name their playback rate so their movies are unchanged. See the note's
  §8.3.5 "As built".
- **§8 steps 3–5 remain**, with the build order in note §8.4 — but steps 3–4 (recorder
  + derived-quantity cache, then `plot_all` reading it) are explicitly **due a
  re-evaluation before being built**. Step 1 removed the per-frame fluid cost except
  where a 2D backdrop is actually drawn, and step 2 cut how many frames draw it, so the
  remaining case is 2D re-plotting of a dynamically-loaded run with `fluid='vort'`.
  All design questions are settled if they do go ahead, including the capture/render
  split (§8.3.7: the recorder writes a data cache and never draws; `plot_all` does all
  rendering from it). Step 5 is the wider examples/docs prose pass; the call sites and
  the `fps`/`playback_rate` model in `docs/quickstart.rst` are already done.
- **§9 still needs its design pass.** When tiling returns, **§9.1 of the note is the
  restoration checklist**: every notice, stub, and replaced test that gating it off
  left behind. Both old implementations are preserved commented-out beneath their
  `raise`, so the still-valid parts (`tile_domain`'s ibmesh/`L` handling,
  `tile_flow`'s `flow_points` extension) can be reused rather than rewritten.

**Then Phase 1** — actually exercising dynamic loading in 2D. Item (C) there,
quantifying dynamic-linear vs. full-cubic error, is the key scientific question and
is still unanswered; don't quote a magnitude for that gap until it is.

**Also merged since:** `master`'s 1.0.1 documentation release (README restructured as
a landing page, docs synced to the code, Open Graph link previews, repo-wide spelling
pass). Documentation only — no library behavior on this branch changed.

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
  (below). The deferred **3D vorticity known-answer test** has since landed in
  `tests/test_flow_interface.py` (solid-body rotation, general linear field, shape).
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

- [x] **`FlowArray` breaks numpy interop — RESOLVED BY DELETION.** (Found while
  adapting `test_flow_generation`.) `__array_finalize__` propagated `self.array` to
  every derived array, and the overridden `shape`/`__getitem__` read from
  `self.array` rather than the array's own buffer — so a `FlowArray` produced by a
  ufunc/comparison read stale data. The fix was not to patch the subclass: its only
  purpose (virtual tiling through `interpn`) no longer worked at all under modern
  scipy, so `FlowArray` was **deleted** and tiling deferred. Velocity components are
  now plain ndarrays and every `np.asarray` workaround is gone. See
  `docs/notes/flow_field_interface.md` §7.3.
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
- [x] **`FluidData.fmin`/`fmax` were generators, not values — DONE.** Built as generator
  *expressions* then re-bound in `update_spline` as `(min(self.fmin[n], ...) for ...)` —
  subscripted a generator (`TypeError` on every window slide, the dynamic path), and
  plotting's `max_u, max_v = flow.fmax` worked exactly once before the generator was
  exhausted. Fixed by wrapping all three sites in `tuple(...)`: `fluid.py` `__init__`
  (~L1074-1075) and both `update_spline` slide branches (~L1211-1212, ~L1271-1272).
  Values were always correct; only the container type was wrong. Details in
  `docs/notes/flow_field_interface.md` §3.3.

### Found while verifying §8 step 1 against the examples (unrelated to it) 🔴

All three reproduce identically against the pre-step-1 package, so they are
long-standing, not regressions. None is covered by the test suite — they only showed up
because step 1 was verified by running the examples end to end.

- [ ] 🔴 **`_ibc._project_and_slide_static` raises on the ib2d channel mesh.**
  `ValueError: operands could not be broadcast together with shapes (3,2) (3,)` at
  `adj_vec_u = adj_vec/np.linalg.norm(adj_vec, axis=-1)`. Reproduce: load
  `examples/ib2d_data` fluid + `channel.vertex` mesh, seed a swarm *uniformly* over the
  domain (not the examples' point source) so some agents start inside/near the cylinder,
  `cov *= 0.0001`, then `move(0.02)` — it dies at step 21 with `seed=1`,
  `swarm_size=50`. This is the load-bearing collision code, so it is the most serious of
  the three. Likely a missing `keepdims=True` (or an `axis` that should be `-1`) on a
  multi-element slide, but that is a guess — diagnose before patching, and add the case
  to `test_collisions_static.py`.
- [x] **`Environment.calculate_FTLE` breaks whenever a `swrm=` is supplied — DONE.**
  Two defects in the same branch, both present on `master` as well (so this is a real
  1.1.0 fix, not a dev regression):
  1. It read `self.props_history` where it meant `s.props_history` (the correct
     spelling is two lines above, in the same block), so the user-swarm branch raised
     `AttributeError: 'Environment' object has no attribute 'props_history'` on the
     very first step, always.
  2. `copy.copy(swrm)` is shallow and only `pos_history` was re-initialized on the
     copy, so `vel_history` and `props_history` stayed aliased to the caller's Swarm.
     Fixing (1) alone therefore turned a crash into silent corruption: the caller's
     `vel_history` collected grid-sized entries (1 → 12 in the regression test),
     contradicting the docstring's "The Swarm object itself will not be altered."
  Regression tests: `test_analysis.py::test_FTLE_with_user_swarm_shear_closed_form`
  (closed-form value via Euler advection through `apply_agent_model`, both
  `store_prop_history` settings) and `::test_FTLE_with_user_swarm_leaves_it_unaltered`.
  Both fail without the fix. `examples/ex_produce_ftle_2d.py` now runs to completion.
- [x] **`Environment.read_IB2d_mesh_data` assumed a fluid was already loaded — DONE.**
  The static-file branch dereferenced `self.flow.fluid_domain_LLC` with no
  `self.flow is not None` guard under `method='proximity'` and `method='hull'`, so
  reading a mesh into a fluid-free environment raised `AttributeError`. That is a
  supported workflow: passing `res` explicitly is how you say you have no fluid grid to
  infer the connection radius from, and `examples/ex_vicsek_model_2d.py` does exactly
  that. The correct form was already in the same file twice (`read_stl_mesh_data`,
  `read_3D_vertex_data_to_convex_hull`); the two IB2d sites now match it.

  **`dyload`-only regression** — on `master` `fluid_domain_LLC` is an `Environment`
  attribute that always exists, so `self.fluid_domain_LLC` cannot fail there. It became
  reachable when the attribute moved onto `FluidData`. No changelog entry, per the
  rule in CLAUDE.md, and not cherry-picked to `master`.

  Regression tests in `test_io_loaders.py`: `proximity` and `hull` without fluid, that
  both still shift when a fluid *is* loaded, and that `res=None` still raises its
  "Must import flow data first!" assertion. Neither method had any coverage before,
  which is why this went unnoticed. `examples/ex_vicsek_model_2d.py` runs again.

- [x] **`method='adjacent'` never applied the LLC shift — DONE, and it was two
  branches, not one.** The fluid loaders translate their data so its lower-left corner
  sits at the origin and record the original corner in `fluid_domain_LLC`; mesh
  coordinates arrive in that original frame and have to follow, or mesh and fluid end
  up offset with nothing raising. `proximity` and `hull` did this; **`adjacent` and the
  moving-mesh (directory of `lagsPts.####.vtk`) branch did not.**

  Fixed at the root rather than per site: one nested `_shift_to_fluid_frame` helper in
  `read_IB2d_mesh_data` that all four branches now call, so the omission cannot recur.
  It carries the `self.flow is not None` guard, which also keeps the fluid-free
  workflow working.

  Latent until now because IB2d grids start at the origin, making the shift a no-op —
  `examples/ib2d_data` has `fluid_domain_LLC == (0.0, 0.0)`. It bites a vertex or
  lagsPts file paired with fluid whose grid does not start at the origin.

  Present on `master` in both branches too, so it ships as a real fix on both.
  Regression tests in `test_io_loaders.py` parametrize the three static methods over
  fluid-free and shifted-fluid environments, plus a moving-mesh shift test; exactly the
  two previously-unshifted branches fail without the fix.

- [x] **`shift_ibmesh_to_match_LLC` corrupted moving meshes — DONE.** It indexed
  `self.ibmesh[:,:,ii]`, which is the coordinate axis only for a static mesh. A moving
  mesh is `(T,N,2,2)` where axis 2 is the **endpoint** axis, so it subtracted `LLC[0]`
  from *both* coordinates of every segment's first endpoint and `LLC[1]` from both of
  the second — **shearing each segment rather than translating it**, and silently,
  since the shapes line up in 2D. Fixed by indexing the last axis with an ellipsis
  (`self.ibmesh[...,ii]`), which is the coordinate axis for all three layouts: static 2D
  `(N,2,2)`, static 3D `(N,3,3)`, moving 2D `(T,N,2,2)`. Also added a
  `self.ibmesh is not None` assertion — it used to fail with a bare `TypeError` from
  subscripting `None` while its two neighbouring guards had clear messages.

  **`dyload`-only**: the method does not exist on `master` at all. No changelog entry
  (never shipped), nothing to cherry-pick.

  It is public API with **no callers anywhere** — not in the package, tests, examples,
  or docs prose — so it was pure untested surface that Sphinx nonetheless documents.
  Now covered in `test_io_loaders.py`: static, moving (asserting the segments stay
  rigid, which is what the shear destroyed), static 3D, the missing-mesh guard, and an
  equivalence test pinning the actual contract — loading fluid-then-mesh (loader
  shifts) must give the same mesh as mesh-then-fluid plus this call.

  Docstring now warns that calling it on a mesh loaded *after* a fluid double-shifts;
  nothing can detect that from stored state, so it is the caller's responsibility.

### Cleanup (low urgency)

- [ ] 🟢 **Orphaned discarded code:** `fCubicSpline._left_based_cspline` /
  `_extend_prev_spline` (`fluid.py:581-763`) — the abandoned cubic-window approach,
  now unreachable (the only `fCubicSpline(...)` caller uses default `bc_type`). Remove
  or annotate as "abandoned — see history."

---

## `_ibc.py` — what is left after the 2026-08 collision pass 🟡

**This work was done on `master` and merged here (`27c810b`); the code and its
tests live on both branches.** Six defects were found and fixed, each with
regression tests: the boundary back-off scaling with domain size and going NaN
at negative coordinates; joints where three or more mesh elements meet, in both
the static and moving sliders; the 3D ranking measuring the wrong angle; a
duplicated mesh vertex producing NaN positions (static) or a step that never
finished (moving); the continue/stop decision; and stack exhaustion surfacing as
a bare `RecursionError`. See `changelog.txt` under 1.0.2.

**`_ibc.py` is not "done".** Measured coverage after all of it is **91% of
statements, 45 lines never executed** (`python -m coverage run --source=planktos
-m pytest -q --runslow`, then `coverage report --include="*_ibc.py"
--show-missing`). Ranked by risk:

- [ ] 🔴 **The rotation branch of the moving slider has never run**
  (`rotated_past_bool`, the largest uncovered block). It handles a mesh element
  *rotating* out from under an agent, and it solves a root-finding problem
  (`optimize.root_scalar`, brentq) to find when the perpendicular velocities
  matched. This is the same shape as the bug that started the pass — an entire
  branch no test reaches — and it contains the root find, which is what hung on
  a degenerate mesh. Rotating boundaries are real for the 2D moving-mesh work,
  so this is the highest risk-per-line left in the file. Note that a
  rotating-junction probe produced almost no recursion, which in hindsight was
  the signal the path was not being reached; constructing a case that genuinely
  enters it is the first task.
- [ ] 🟡 **The moving free-flight recursion is untested** (the `newendpt =
  newstartpt + (1-t_edge)*vec` path near the end of `_project_and_slide_moving`).
  Its static counterpart dominates recursion depth at ordinary step sizes —
  44 of 45 recursions in measurements — so the moving one is likely exercised
  constantly in real runs while never being asserted on.
- [ ] 🟢 Remaining uncovered lines are early-return special cases
  (`1-t_I < 10e-7`, "practically finished with this step"), some 3D sticky
  paths, and the parallel-worker dispatch.

**Two findings recorded rather than fixed** (issue-tracker material by the
`xfail` rule in CLAUDE.md — real, but not worth stopping the cycle for):

- [ ] 🟡 **An exception during the immersed-boundary stage leaves the `Swarm`
  mid-update.** `Swarm.move` appends to `pos_history` and recomputes velocities
  *before* `apply_boundary_conditions`, and advances `envir.time` only *after*.
  So any raise in between leaves history one entry longer than `time_history`,
  positions a mix of IB-corrected and raw (possibly *inside* a boundary), and
  time not advanced. Not hypothetical: it is the state the original junction
  `ValueError` produced. Loud, but a caller who caught and continued would have
  penetrating agents and desynchronised history.
- [ ] 🟢 **The free-flight termination argument has a floating-point gap.** That
  branch terminates because the remaining movement `(1-t_edge)*vec` strictly
  shrinks, which needs `t_edge > t_I` to hold *in floating point*. A step small
  enough to make them equal numerically would stall. Could not be constructed;
  the `RecursionError` re-raise now bounds the consequence either way.

**Testing approach that worked**, if picking this up cold: the existing suite
only ever built *chains* (walls, polylines, corners, grooves, dihedrals), where
a vertex joins at most two segments — so whole branches were unreachable by
construction. `tests/test_collisions_junctions.py` adds `star_2D`,
`closed_polygon` and `book_3D` builders for the missing geometry class, and its
candidate counts were **measured by instrumenting the branch**, not assumed. Its
invariant assertions (finite, motion not amplified, stays outside a closed
obstacle, and rigid-motion equivariance) are what caught the wrong-but-finite
answers that no arithmetic check could see.

---

## Phase 1 — Test dynamic loading in 2D 🟡

Use 2D IB2d data (cheap, deterministic, reported working). Separate two questions:

NB: the in-memory linear path (`INUM=True`) has unit coverage in
`tests/test_flow_interface.py` (`LinearSpline` call/index/extrema/derivative/
`regenerate_data`, and linear-vs-cubic agreement on data linear in time).

**(A), (B) and (D) are DONE** — `tests/test_dynamic_loading.py`. The earlier note here
said window sliding "still needs real data"; that turned out to be wrong.
`update_spline` asks of a source only a `load_dumpfiles(d_start, d_finish)` returning
per-component ndarrays with a leading time axis, so a ~20-line synthetic `FluidData`
subclass backed by an in-memory array exercises the real slider exactly,
deterministically, in the fast suite.

**Ingestion is covered too, in both dimensions.** `IB2dData` — the 2D reference path,
and previously the *only* loader with no automated coverage at all (every
`read_IB2d_fluid_data` call in the tree is under `tests/manual/` and needs external
data) — now loads end to end from `tests/fixtures/ib2d_fluid_min` (8 dumps, vector
form) with `INUM=None/True/4`, pinning the full timeline, a real window slide, the
periodic wrap, and warning-free construction under the new `FluidData` guards. The
scalar `uX`/`uY` branch is pinned separately in `test_io_loaders.py` against
`ib2d_fluid_scalar_min`. **Only (C) still needs real data.**

- [x] **(A) Machinery correctness — DONE.** Windowed-linear (`INUM=4,5,7`) agrees with
  full linear (`INUM=True`) on forward sweeps, backward sweeps, non-monotone random
  access, exactly-on-node times, out-of-bounds clamping, and in 3D. Agreement is to a
  few ulp, not bit-for-bit: the slider carries window-boundary values by *evaluating*
  the outgoing spline rather than re-reading raw data, which costs an ulp per slide.
  **That error does not accumulate** — measured flat at 1–2 ulp across 400 loads and
  6 full sweeps, pinned by `test_holdover_roundoff_does_not_accumulate`.
- [x] **(B) Window-sliding behavior — DONE.** Forward slide, backward slide, the
  "jump to beginning" fast path (asserted to be one load, not a walk), extrapolation
  flag flips at both dataset ends, bounded window across a full sweep, bounded load
  count (no thrashing), no load when the query stays inside the window, `fmin`/`fmax`
  tuples widening across slides (the §3.3 lock on the path where it actually bit),
  and `get_raw_loaded_data` on a genuinely sliding window (the §3.5 lock).
  - **Bookkeeping bug found and fixed.** At the dataset end `idx_finish` was set to
    `len(flow_times)` — one past the last valid index — while `loaded_dump_bnds[1]`
    stays inclusive, so the two index spaces disagreed there and only there. Latent,
    not live: the sole reader of `loaded_idx_bnds[1]` is the forward slide, which is
    gated off by `extrapolate[1]` once the end is reached, and the window itself was
    unaffected because the slice that builds it clips. Pinned first, then fixed to
    `len(self.flow_times) - 1`; the whole suite stayed green, which is the evidence
    the change is inert. `test_index_spaces_agree_at_every_slide` now locks
    `loaded_idx_bnds == loaded_dump_bnds` across forward and backward sweeps so the
    two cannot drift apart again.
  - NB: the closing window is *structurally* ≤ `INUM` samples rather than `INUM`+1,
    because every forward slide pins `idx_start` to the outgoing window's
    next-to-last index (the two-sample holdover) and the dataset then runs out. That
    is unrelated to the bug above and is harmless — less memory, same values — which
    is why the bounded-window test asserts `<=` rather than `==`.
- [x] **(C) Comparability — DONE.** Measured on real IB2d data (`tests/data/leaf_data`,
  149 dumps, 129×193, Δt=1e-3 s) by `tests/manual/quantify_temporal_interp.py`, which
  is reproducible and documents its own methodology. **Errors are measured by
  withholding** — both schemes are built on every 2nd dump and evaluated at the dumps
  left out, whose raw values are exact truth neither scheme saw. Comparing linear to
  cubic directly would have measured their *disagreement*, not either one's error.

  | quantity (Δt=1e-3 s, leaf_data) | linear | cubic | ratio |
  |---|---|---|---|
  | velocity, rms err / U_rms | 1.27% | 0.54% | 2.4× |
  | ∂u/∂t, rms err / \|∂u/∂t\|_rms | 5.62% | 2.40% | 2.3× |
  | convergence order, **median** point | 1.75 | **4.85** | — |
  | convergence order, 99th-pct point | 1.62 | 2.40 | — |

  - **Windowing is inert on real data too**, confirming the (A) result end to end:
    `INUM=4` vs `INUM=True` differ by 1.1e-16 (4e-17 of U_rms). So the whole question
    is linear-vs-cubic; the streaming machinery costs nothing.
  - **Do not quote the 2.4× ratio on its own — it blends two regimes.** Interpolation
    error is wildly concentrated (cubic max/median = 2.4e4×), so an rms-over-everything
    ratio is set by a rough minority. Where the flow is temporally smooth, both schemes
    hit their theoretical orders (cubic **4.85**, linear 1.75) and cubic is decisively
    better; where it is temporally rough, both stall (2.40 / 1.62) and the gap nearly
    closes — there the *data* is the limit, not the scheme. The rough regions are not
    the immersed-boundary neighbourhood (none of the worst 1% of points lie within 8
    cells of the mesh); peak error tracks local |∂²u/∂t²|, correlation ≈0.7.
  - **The absolute numbers are not transferable; the orders are.** Error depends on the
    dump interval relative to the flow's own timescales. The harness was calibrated
    against analytic fields and recovers ~2/~4 when well resolved and degrades
    predictably when not, so the fitted orders are trustworthy for extrapolating to a
    different dump cadence.
  - **The practical answer for an ABM: dynamic loading does not change dispersal
    statistics.** Individual tracer trajectories decorrelate (rms separation reaches
    9.9% of path travelled, ~6% of the domain diagonal) — but that is the flow
    separating nearby particles exponentially, which happens under *any* perturbation,
    and it saturates at the flow-feature scale. Ensemble statistics over 1936 tracers
    agree to **within 0.6%, most under 0.3%** (mean/std of x and y, mean/std of net
    displacement, and its 10th/50th/90th percentiles).
  - ⚠️ One dataset, one Δt, tracer particles, no immersed boundary in the trajectory
    run (deliberately, to isolate the interpolation). Re-run on 3D data in Phase 2
    before generalizing; `INUM` docs should quote the orders and the ensemble result,
    not the bare ratio.
- [x] **(D) `get_dudt` under linear splining — DONE.** Pinned as a piecewise-constant,
  discontinuous finite difference: the value on each interval, the jump at a
  breakpoint, zero beyond the data bounds, and agreement with full-linear across
  slides in 2D and 3D. Also pinned the **interval convention**, which the docstring
  had backwards: `du/dt` is constant on **right**-closed intervals `(t[i-1], t[i]]`,
  so a time landing exactly on a timestamp takes the slope of the interval to its
  *left* (`t0` excepted, having none). Docstring corrected in `fluid.py`.
- [~] **(E) Tiling/periodic × dynamic — SUPERSEDED / on hold.** Was: `FlowArray` view +
  `tiling` propagation through `update_spline`. Tiling now raises
  `NotImplementedError` for the duration of the plotting work, and the `tiling`
  propagation (and its `assert ... "Tiling did not propagate correctly"` guards) is
  gone from `update_spline`, so there is nothing to test here yet. Revisit as part of
  the real tiling implementation (`docs/notes/flow_field_interface.md` §9), which
  covers 2D and 3D together and must define how `tiling` interacts with
  `periodic_dim`. **Periodic × dynamic on its own is still worth testing** and stays
  in scope for Phase 1.

---

## Phase 2 — Test dynamic loading in 3D 🟡 (blocks 3D moving boundaries)

The actual end goal (the ~100 GB case). **Needs real 3D dynamic fluid data** — the user
has a sample from their collaborator to stage on this machine when we reach this step.

**Assume a rectilinear fluid grid.** Data is expected as a sequence of **vtk files
exported from VisIt/ParaView**, where the source field (IBFE SAMRAI / OpenFOAM FEM) was
*already* interpolated onto a rectilinear grid externally. Planktos just reads that
rectilinear vtk — source-specific ingestion is **out of scope** (see CLAUDE.md "3D
fluid data sources").

- [x] **Stage the real 3D dynamic dataset on-machine — DONE.** It is at
  `tests/unsteady_3D_testdata/` (gitignored as a whole directory; `.vtp`/`.vtm`/
  `.vtm.series` were added to `.gitignore` alongside it). Contents: `VTK/` with 21
  timestep `.vtm`s + a `.vtm.series` (case `case08_alpha2_1e8`),
  `oral_arm_disk.stl`, and `README_oral_arm_setup.md` with the full setup spec.
  - **Physics:** OpenFOAM, Cassiopea oral-arm porous disk. Water (ρ=1000,
    ν=1e-6), Re=500, laminar transient. Pulsing annular inlet,
    u_z(t) = 0.01·½(1−cos(2π·0.8·t)) m/s. Export covers the last two pulse cycles
    (t ≥ 7.5 s, period 1.25 s). Domain x,y ∈ [−0.05, 0.05], z ∈ [0.003, 0.271] m,
    **all lengths in meters**. Fields: `U`, `p` (kinematic), `vorticity`, `Q`.
  - ⚠️ **Not directly loadable yet.** This is OpenFOAM **unstructured** XML
    (`.vtu`/`.vtm`), but this branch assumes a **rectilinear** grid (see CLAUDE.md
    "3D fluid data sources"). It must be resampled to a rectilinear grid in
    ParaView/VisIt and re-exported before `VTK3dData`/`read_IBAMR3d_vtk_data` can
    read it. Record the resample recipe here once done — that step is the actual
    first task of Phase 2, not the staging.
  - `oral_arm_disk.stl` is a ready-made 3D immersed boundary for Phase 3.
- [x] **`VTK3dData` dynamic loading was silently broken — found and fixed ahead of the
  real data.** `flow_times` was built from only the dumps read at construction, so on
  the windowed path it held `INUM+1` timestamps instead of the whole series. That did
  not raise: it made `INUM >= len(flow_times)-1`, which sends `FluidData` down its
  "everything is in memory" branch with `extrapolate=(True, True)`, making
  `update_spline` unreachable. The run then completed with the fluid **frozen** at the
  end of the opening window — no error, no warning. (`IB2dData` was never affected; it
  computes `flow_times` analytically over the full range, which is why 2D worked.)
  - **Fix.** `TIME` lives in each legacy-vtk header (`FIELD FieldData` sits right after
    `DATASET`, ahead of the coordinate arrays), so the whole series can be timestamped
    with one ~4 kB header read per dump — measured 62× faster than a full parse on the
    IBAMR test files, and *constant* in file size, so the margin only grows with real
    dumps. New `_dataio.read_vtk_time_only`, with a per-file fallback to a full read
    since the format also permits `FIELD` outside the header. `_read_vtkfiles` no
    longer parses times at all (it is the per-window loader; that was both wasted work
    and the origin of the bug).
  - **Guards added to `FluidData`,** so the next loader making this mistake fails
    loudly: a dynamically-loading subclass that exposes `d_start`/`d_finish` must pass
    `flow_times` covering the full dump range, and an int `INUM` that spans the dataset
    now warns that no dynamic loading will occur (a pre-existing footgun of its own).
  - **Coverage.** `tests/fixtures/vtk3d_min/` — 8 tiny rectilinear vtk dumps with
    `TIME`, generated by `_gen_fixtures.py`, field `u=t, v=x, w=t*z` so a frozen or
    truncated timeline is unmistakable. Exercised end to end in
    `test_dynamic_loading.py`; `read_vtk_time_only` unit-tested in `test_io_loaders.py`.
- [ ] Re-run the above against **real** 3D data once the OpenFOAM resample lands;
  un-skip / fix the IBAMR load tests on real data.
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

## Documentation 🟡

- [x] **`FluidData` on readthedocs — DONE.** `docs/api/FluidData.rst` autoclasses
  `FluidData` and the three per-source subclasses and is listed in
  `docs/api/index.rst`. Alongside the autodoc it carries narrative sections on how
  to get velocity data out (call it with a time vs. index it like a list, and why
  indexing is refused while streaming) and on the `INUM` tradeoff (below).
  `Environment.rst` gained a pointer, since `INUM` is met first through the
  `Environment` reader methods. Writing it turned up a `FluidData` class docstring
  that rendered badly and claimed the time-varying case is always an `fCubicSpline`
  (true only for `INUM=None`); rewritten, along with the malformed numpydoc in its
  `Attributes` block. Sphinx builds clean, no warnings.
- [ ] **Sweep the prose docs for the master-era fluid API.** `docs/quickstart.rst`
  and `README.md` still frame fluid handling as `Environment`-level. The 1.0.1
  merge fixed the two that were outright wrong (`get_2D_vorticity` → `get_vorticity`,
  and the claim that flow can be "extended"), but the overall framing still assumes
  master's API. Worth a pass once the fluid API stops moving — **still deliberately
  held**, since §9 decides what `tile_flow` becomes and whether `Environment.extend`
  returns, and both are exactly what this prose would describe. The new
  `docs/api/FluidData.rst` covers the reference side in the meantime.
- [x] **`INUM` and the linear-vs-cubic tradeoff — DONE.** Documented for users in
  `docs/api/FluidData.rst` (anchor `inum-tradeoff`): the `INUM` table, why linear
  in time is permanent rather than a placeholder, and what it costs. Per the Phase
  1 (C) findings it quotes the **convergence orders and the ensemble result, not
  the bare rms ratio** — the ratio blends a smooth regime where cubic is decisively
  better with a rough one where neither converges, so on its own it would mislead
  in either direction. Ends with the practical reading (ensemble statistics are
  unlikely to change; prefer `INUM=None` for inertial particles when the data
  fits) and a note that absolute errors do not transfer between datasets while the
  orders do, that the figures are from one 2D dataset, and that the measurement is
  reproducible via `tests/manual/quantify_temporal_interp.py`.
  - Remaining: re-check the numbers on 3D data in Phase 2 and update the note.

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
- [ ] ⚠️ **Watch: vtk emits a numpy 2.5 deprecation that would become fatal.**
  `vtkmodules/util/numpy_support.py` assigns `result.shape = shape`, which numpy 2.5
  deprecates in favour of `np.reshape`. Harmless today (a `DeprecationWarning`), but
  if numpy promotes it to an error, **every vtk read breaks** — `_dataio`'s
  `read_vtk_Structured_Points` and `read_vtk_Rectilinear_Grid_Vector` both go through
  `numpy_support`, so IB2d, IBAMR/VTK3d and the committed fixtures all fail at once.
  Nothing to fix here; the fix belongs upstream in vtk. Surfaced by CI (Python 3.12
  job, which installs the newest numpy) failing a test that promoted *all* warnings to
  errors — that test now filters to planktos-origin warnings only, so the deprecation
  no longer masquerades as our bug. Re-check when bumping the vtk or numpy minimum.
- [ ] Changelog housekeeping (`changelog.txt`, 1.1.0): drop "TODO: test dynamic loading"
  once Phases 1–2 land; resolve the `tiling`-setter TODO (make tiling a setter of
  `FluidData.tiling`, with `Environment.L` updating off it) — folded into the real
  tiling implementation, `docs/notes/flow_field_interface.md` §9.
- [ ] `Environment.extend` was removed (extrapolation is the intended replacement).
  Whether it returns is decided in `docs/notes/flow_field_interface.md` §9, alongside
  the real tiling work — the two are the same class of operation (reported domain ≠
  stored grid) and should share a mechanism. The parked test
  `test_flow_generation.py::test_extend_grows_domain_and_copies_edges` un-skips if so.
- [ ] **Optional agent-history retention (maybe-feature).** `Swarm.pos_history` grows
  every step, so long runs accumulate memory whether or not anyone plots them. A flag
  along the lines of `store_pos_history='all' | 'frames' | None` would let a user cap
  it (`store_prop_history` is the naming precedent, and `'frames'` — keep history only
  where plot frames were captured — keeps history and the plotting cache mutually
  consistent).
  - **Not free, and the loss is unrecoverable:** decimating history breaks `plot_all`
    at full step resolution, `save_data`, `save_pos_to_csv`, `save_pos_to_vtk`, and any
    post-hoc agent analysis (per-step displacement/dispersal statistics). Must be
    opt-in, off by default, and loudly documented.
  - Considered for the §8 plotting redesign and **deliberately left out** — it is a
    long-run memory question with consumers well beyond plotting, and nothing in §8
    depends on it (`docs/notes/flow_field_interface.md` §8.2, "Not in scope").

---

## How to run the tests

- Fast loop: `pytest` (≈1s; skips `slow` / `vtk`-absent / `vtu`-absent).
- Full: `pytest --runslow` (adds the parallelization checks and the plotting
  smokes; ≈13s).
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
