# TODO — `dyload` branch (dynamic loading of fluid data)

**Goal of this branch:** load/spline time-dependent fluid velocity data *on demand*
(a sliding window of timesteps) instead of holding the whole dataset in memory, so
that large 3D time-varying flows (~100 GB raw, larger once splined) can be used.

**Current state (2026-08-11).** The architecture is built and the API has settled —
all fluid data is a `FluidData` object (`planktos/fluid.py`). Dynamic windowed
loading works and is tested in **both** 2D and 3D against committed fixtures.
Temporal interpolation of dynamically-loaded data is **linear in time**
(`LinearSpline`); full-dataset loading defaults to **cubic in time**
(`fCubicSpline`). See the design-history section at the bottom for the cubic→linear
story.

**Suite is green: 731 passed / 22 skipped (`pytest`), 751 passed / 2 skipped
(`pytest --runslow`).** No failures, no xfails.

**What is done:**

- **Phase 0** (adapt the suite to the `FluidData` API, fix what it surfaced) — complete.
- **Phase 1** (2D dynamic loading) — complete, including **(C)**, the
  linear-vs-cubic error measurement, which is answered with numbers. Only
  periodic × dynamic testing is left over, folded into Phase 2.
- **Flow-field interface refactor** — **complete.** `FlowArray` deleted, tiling gated
  off, `test_flow_interface.py` pinning the consumer contract. Its design note
  (`docs/notes/flow_field_interface.md`) was deleted 2026-08-18 once everything still
  load-bearing had moved into `docs/notes/run_persistence.md`; Appendix A there is the
  surviving summary, and the git history has the full analysis.
- **Run persistence / plotting** (`docs/notes/run_persistence.md`) — **§3.1, §4.1, §5
  and steps §6.1 A0–A1 complete** (plot statistics served from a per-dump mean cache;
  `fps`/`playback_rate` frame selection; the three prerequisite bug fixes; the movement
  start point decoupled from `pos_history`, which is what makes a capture schedule safe;
  and provenance recorded at load time by every fluid and mesh entry point,
  `planktos/_provenance.py`). The rest is the plan below, starting at **A2**.
- **`_ibc.py` collision passes** — done on `master` and merged here; coverage 91% → 99%.

**Phase 2 is complete (2026-08-11).** The loader reads the real OpenFOAM dataset, the
windowed machinery is vetted against it, and the memory claim is measured rather than
asserted. That was the branch's stated remaining reason to exist.

**Where the work goes next, in priority order:**

Item 2 is the active work; item 1 has one sub-item left and can run beside it.

1. 🔴 **The robustness pass on the OpenFOAM loader** — making the Phase 2 loader usable
   on the next dataset. The list is under Phase 2, "Robustness follow-ups". **One item
   is left** — surfacing a stored `vorticity` — and it was folded into
   `run_persistence.md` §3.3, so it is really component B of item 2 rather than a
   competing thread.
2. 🔴 **The run archive and the plotting work it feeds** — `docs/notes/run_persistence.md`,
   components A–C. **This is the work in hand.** §5.3 and §6.1 steps **A0** and **A1**
   are done (2026-08-21); **next is A2**, the archive writer. Detail below.
3. 🟡 **Note §9** (real position-wrapping tiling, 2D and 3D; whether `Environment.extend`
   returns) — the design pass is now written up in `run_persistence.md` §9; §9.3 is the
   restoration checklist. This also **unblocks the prose-docs sweep**, which is
   deliberately held because §9 decides exactly what that prose would describe.
4. ✅ **The 1.0.3 decision — settled (2026-08-19).** The patch release was cut from
   `master` and tagged `v1.0.3`; the cherry-pick queue at the bottom of this file is the
   record of what moved and what deliberately did not. **The next release is meant to be
   `1.1.0`, with no `1.0.4`** — see that queue for the one thing that would change it.

**The run archive, in detail.** The go/no-go was **decided 2026-08-11 in favor of
building it**: plotting is a measured bottleneck on our own runs, and it is cheapest to
do while the memory architecture is in hand. The recorder interface was revised at the
same time (capture is automatic and hooks the environment time advance; the format is
crash-valid by construction rather than by finalization).

**Reframed 2026-08-18.** What was specified as a *plot cache* is in fact a general
**run archive**: chunked, append-only, crash-valid, mmap-readable capture of agent state
written on the fly. That solves three problems the plan was only incidentally touching —
plotting a dynamically-loaded run without re-streaming it, capping `pos_history` memory
on long runs without losing data, and (with the provenance record designed in) reloading
or restarting a run at all, none of which Planktos can do today. The note was rewritten
around that framing and split into four components: **A** the archive, **B** fluid-side
streaming, **C** rendering, **D** tiling. A and B are independently shippable.
`run_persistence.md` §0 is the orientation; §2.1–§2.3 and §4.2 carry the interface
reasoning; §6.1 has the build order and §6.3 the entry points. **Start with §5**, two
prerequisite bug fixes found during the reframe (see the cherry-pick queue).

**Also merged since:** `master`'s 1.0.1 documentation release and its 1.0.2 bug
fixes. No dyload-specific behavior changed by either. **`v1.0.3` is released and
tagged** (2026-08-19), and **the next release is meant to be `1.1.0`, not a `1.0.4`**.
Master-applicable fixes made here are still logged in the cherry-pick queue at the bottom
of this file, against the chance that enough accumulates to be worth cutting one.

Priority key: 🔴 do first · 🟡 next · 🟢 later · ⚪ deferred / low priority.

---

## Phase 0 — Adapt the suite to `FluidData` + fix what it surfaced ✅ COMPLETE

The overhauled suite was written against mvbnd's `Environment` fluid API; on dyload
that API moved onto `FluidData`. All 32 failures are resolved, every test module is
adapted, and the suite is green. The API renames that caused most of them —
`envir.flow_points`→`envir.flow.flow_points`, `envir.flow_times`→
`envir.flow.flow_times`, `get_2D_vorticity`→`get_vorticity`, `envir.tile_flow`→
`envir.tile_domain`, `Environment.extend` removed — are recorded in CLAUDE.md and are
the standing hazard when merging `master` text.

**Nine real defects were found and fixed along the way**, each with a regression test.
They are described user-facing in `changelog.txt` (1.0.2 and 1.1.0) and in full detail
in the git history; one line each here, since the interesting part is now the tests
that pin them:

| Defect | Where | Now pinned by |
|---|---|---|
| `FlowArray` returned stale data from any derived array | `fluid.py` | resolved by **deleting** `FlowArray` (`run_persistence.md` Appendix A) |
| `FluidData` defaulted `periodic_dim=True`, wrapping the upper grid edge to the lower — corrupted every FTLE field | `fluid.py` | `test_flow_generation.py` edge tests + `test_analysis.py` closed-forms |
| `fmin`/`fmax` were generator *expressions*, not tuples — `TypeError` on every window slide | `fluid.py` | `test_flow_interface.py`, `test_dynamic_loading.py` |
| `Swarm.get_dudt` called the pre-rename `envir.dudt` | `_swarm.py` | `test_material_derivative.py` |
| `FluidData.get_dudt` zeroed the derivative *at* the data endpoints and built zeros with the wrong shape | `fluid.py` | `test_material_derivative.py` |
| `save_fluid`/`save_2D_vorticity` passed domain lengths where coordinate arrays were expected | `_environment.py` | `test_io_loaders.py` |
| `calculate_FTLE(swrm=...)` raised on step 1, and its shallow copy aliased the caller's history | `_environment.py` | `test_analysis.py` (2 tests) |
| `read_IB2d_mesh_data` required a loaded fluid; `method='adjacent'` and the moving-mesh branch never applied the LLC shift | `_environment.py` | `test_io_loaders.py` (parametrized over all methods) |
| `shift_ibmesh_to_match_LLC` **sheared** moving-mesh segments instead of translating them | `_environment.py` | `test_io_loaders.py` (rigidity assertion) |

Two notes worth keeping:

- The **periodic-by-default** bug was general, not FTLE-specific — any flow sampled
  exactly at an upper/right grid edge read the opposite edge. FTLE only exposed it
  because it seeds tracers exactly on the domain boundary.
- The last three were found by **running the examples end to end**, not by the suite.
  None had any test coverage; `ex_produce_ftle_2d.py` and `ex_vicsek_model_2d.py` were
  both broken. That remains the cheapest way to find this class of defect.

### Cleanup (low urgency)

- [x] ✅ **Orphaned discarded code deleted (2026-08-11).**
  `fCubicSpline._left_based_cspline` and `_extend_prev_spline` — the abandoned
  cubic-window approach — are gone, along with the two `__init__` branches that reached
  them, the now-unused `dydx0`/`dydx1`/`direction` parameters, and the `solve_banded`
  import that existed only for them. 183 lines.
  - **Verified orphaned before deleting**, not assumed: every `fCubicSpline(...)` call
    site in the package and the suite passes only `(flow_times, flow)` and at most
    `extrapolate=`, never `bc_type='left'` and never `dydx0`; nothing subclasses the
    class; and it is not in the API docs, so the signature change is internal. (The
    hits under `build/` are a stale, untracked build artifact.)
  - The knowledge is not lost with the code: what was tried and why it failed is in the
    design-history section at the bottom of this file, with commit `bbd093b`.
- [x] ✅ **`fCubicSpline.trim_end` deleted too (2026-08-11).** The survivor of
  design-history item 2 (deleting boundary-contaminated coefficients, `7b385d7`), from
  the same abandoned effort. Verified orphaned three ways before removing: no textual
  reference anywhere outside its own definition; no `getattr`/`hasattr` path to it; and
  **`LinearSpline` has no counterpart**, so it cannot have been part of a duck-typed
  interface — `FluidData` swaps the two spline classes freely, so any polymorphic
  `.trim_end()` call would already have crashed on the linear path. Recover with
  `git log --oneline -S "trim_end" -- planktos/fluid.py` if ever needed.

---

## `_ibc.py` — the 2026-08 collision passes ✅ COMPLETE (3 findings left open)

**Done on `master` and merged here** (`27c810b`, `e448d0a`); the code and tests live
on both branches. Two passes fixed **seven** defects — six in the first pass, plus
**BUG-TCRIT** in the second (the moving slider's two critical times were computed
wrongly, one of them in a way that was not rotation invariant, so the resolved
position depended on how the problem sat in the coordinate frame). All are described
in `changelog.txt` under 1.0.2, each with regression tests.

Coverage went **91% → 99% of statements, 45 → 7 lines never executed**
(`python -m coverage run --source=planktos -m pytest -q --runslow`, then
`coverage report --include="*_ibc.py" --show-missing`). **The 7 remaining lines are
not worth chasing:** three are placeholders behind the 3D-moving
`NotImplementedError`, two are numerical corners documented in place as unreachable,
two are solver non-convergence raises.

**Testing method that worked**, if picking this up cold: the old suite only ever built
*chains* (walls, polylines, corners, grooves, dihedrals), where a vertex joins at most
two segments — so whole branches were unreachable by construction.
`tests/test_collisions_junctions.py` adds `star_2D`, `closed_polygon` and `book_3D` for
the missing geometry class, and its candidate counts were **measured by instrumenting
the branch**, not assumed. The invariant assertions (finite, motion not amplified,
stays outside a closed obstacle, rigid-motion equivariance — now in
`tests/test_collisions_invariants.py`) are what caught the wrong-but-finite answers no
arithmetic check could see. Where a branch differs from a covered one only by an
arbitrary choice (which vertex of an element is stored first), pin an **invariance**
rather than a value.

### The three findings — (1) and (2) are now FIXED, (3) remains open

By the `xfail` rule in CLAUDE.md these were issue-tracker material, not `xfail`s: each
is real, but none was worth stopping the development cycle for. Recorded here in full
because the summaries alone are not actionable.

---

#### 1. ✅ FIXED — an exception during the immersed-boundary stage left the `Swarm` mid-update

**What the code does.** `Swarm.move` ([_swarm.py:941](planktos/_swarm.py#L941)) runs in
this order:

1. copy `old_positions` / `old_velocities` ([L982](planktos/_swarm.py#L982));
2. `self.positions[:,:] = self.apply_agent_model(dt)` ([L992](planktos/_swarm.py#L992));
3. **append to `pos_history` / `vel_history`** ([L995](planktos/_swarm.py#L995));
4. **recompute `velocities` and `accelerations`** by finite difference ([L1001](planktos/_swarm.py#L1001));
5. `apply_boundary_conditions(...)` — *this is where domain BCs and all immersed-boundary
   collision handling run* — then `after_move` ([L1006](planktos/_swarm.py#L1006));
6. append to `time_history` and advance `envir.time` ([L1011](planktos/_swarm.py#L1011)).

**The problem.** Steps 3–4 commit state that step 5 is supposed to correct, and step 6
has not happened yet. So a raise anywhere in step 5 leaves the object in a state that
is internally inconsistent in four separate ways at once:

- `pos_history` is one entry **longer** than `time_history`, so every subsequent index
  into the two is misaligned — including `plot_all`'s frame selection, which assumes
  they correspond;
- `positions` holds a **mix**: agents processed before the raise are IB-corrected,
  agents after it are raw output from the agent model, which may be **inside an
  immersed boundary or outside the domain**. This violates the branch's hard
  no-penetration invariant;
- `velocities` / `accelerations` are a matching mix — corrected agents get theirs
  recomputed from the corrected displacement by `_apply_ib_result`, the rest keep the
  raw finite difference. Each agent is self-consistent with its own position; what is
  wrong is that the un-processed ones describe motion through a boundary;
- `ib_collision_idx` is **stale** for the un-processed agents — it is never reset per
  step, so they silently keep the *previous* step's values. In the reproduction the
  whole array came out byte-identical to the step before. It does not look wrong;
- `envir.time` was never advanced, and `after_move` never ran; other swarms in the
  environment never got their freeze-append (step 6's loop).

**Why it matters.** The raise itself is loud, so a caller who lets it propagate is
fine. The damage is to a caller who **catches and continues** — a parameter sweep, a
batch harness, or a `try/except` around the move loop. Verified: the next `move()` used
to succeed silently, marching the un-processed agents further past the wall, after which
they never intersect the mesh again and are **permanently** on the wrong side.

**Not hypothetical:** this is the state the original junction `ValueError` produced
before that bug was fixed.

**The key structural fact, verified by building the failure** (6 agents driven into a
wall, the collision routine patched to raise on the 3rd): **the recorded history is
entirely clean.** The failed step appends its *pre-move* state, so `pos_history[-1]` is
the last good state and it sits at `envir.time`. Nothing about the failed step is
recorded anywhere. Only the four live attributes are wrecked. This is better news than
it first looked and it shapes the whole fix.

Two more measured details:

- **Failure granularity depends on `self.pool`.** Serial `map()` is lazy, so workers
  and `_apply_ib_result` interleave and the step is applied up to the failing agent.
  `pool.map()` is **eager**, so a worker exception means **zero** results applied and
  every agent holds raw output. Both verified.
- **In every consistent state `len(pos_history) == len(time_history)`** — verified across
  `move()`, `move_swarms()` and multiple swarms. The mismatch appears only transiently
  inside `move_swarms` (resolved before it returns), after a failed step, and after a
  bare `move(update_time=False)` with no matching time bump.

**What was built — `envir.time = None` as the error marker.** Two earlier and larger
designs were built and reverted as more machinery than the problem warrants: a full
rollback, then a recovery method plus a time-lookup helper plus plotting that stopped
before the failure. What landed adds no attributes, no methods, and no per-step cost:

- **The except clause appends `envir.time` to `time_history`** before anything else.
  The failed step had already appended its pre-move state to `pos_history`, so this
  closes the histories off *consistently*: both end with the state as it was when the
  step began, at the time it began. The record stays a valid record.
- **Then `envir.time = None`**, which marks everything current as untrustworthy. It
  lives on the `Environment`, so it blocks every swarm in it, not just the one that
  failed — which is right, since the environment's clock is what stopped meaning
  anything.
- **Then re-raise a `RuntimeError` chained `from` the original**, whose message is the
  actual deliverable: which attributes are unreliable, that the histories were closed
  off consistently, and the three lines that back the step out.
- **`move()` checks `self.envir.time is None` first** and refuses. This is the part
  that prevents harm — continuing was verified to march the un-processed agents further
  past the wall, after which they never intersect the mesh again and are permanently
  through it.
- Recovery is symmetric with what the failure did, and needs no saved state because the
  history entries *are* the pre-move state:

  ```python
  envir.time = envir.time_history.pop()
  swrm.positions = swrm.pos_history.pop()
  swrm.velocities = swrm.vel_history.pop()
  ```

  `accelerations` need no restoring — the next `move()` recomputes them from positions
  and velocities. `ib_collision_idx` is left stale, which is tolerable: it is a
  diagnostic output and feeds nothing.
- Two tests in `test_swarm_lifecycle.py` pin the closed-off histories, the marker, the
  block, and the documented recovery.

- **Plotting works in the error state and leaves the failed step out.**
  `_select_frames` builds its time axis with `envir.time` on the end, which was
  arithmetic on `None`; it now branches on that, warns, and uses `time_history` alone.
  Frames are then exactly the recorded states, so the incomplete positions are never
  drawn. Pinned by a third test.

**Rejected:** reordering the history appends to after the boundary stage. It looks
free — the appended values are pre-move copies — but `apply_boundary_conditions` reads
[`pos_history[-1]`](planktos/_swarm.py#L1405) as each agent's movement start point, so
the append must precede it.

**Also worth fixing, independently:** the movie-save handler's misleading ffmpeg advice
(print the actual exception first — it already re-raises).

---

#### 2. ✅ FIXED — the free-flight termination argument had a floating-point gap

**Where.** `_project_and_slide_static`, the "continue on the original trajectory"
branch ([_ibc.py:949-965](planktos/_ibc.py#L949-L965)).

**What the branch does.** An agent sliding along a mesh element runs off the end, and
there is no adjacent element to transfer onto. It is then released back into free
flight from the separation time `t_edge`, and the routine **recurses** on itself with
`newstartpt = x_edge + EPS*normal` and `newendpt = newstartpt + (1-t_edge)*vec` — i.e.
whatever fraction of the original step remains.

**The termination argument, as written in the comment:** the recursion terminates
because the remaining movement `(1-t_edge)*vec` **strictly shrinks** at every level.
Note the comment already flags that this is weaker than the adjacent-element transfer
branch above it — nothing bounds *how much* it shrinks, so it bounds depth only loosely.

**The gap.** The next level receives `(1-t_edge)*vec` against this level's `vec`, so
strict shrinkage is exactly **`t_edge > 0`** — an earlier version of this entry said
`t_edge > t_I`, which is not the requirement (`t_I` does not enter the comparison).
`t_edge` is computed as a ratio of norms ([_ibc.py:700](planktos/_ibc.py#L700)) and
nothing in that arithmetic guarantees a positive result in floating point; a
near-tangent hit could round it to zero. The remaining movement would then be
unchanged, the next level handed an identical sub-problem, and the recursion would sit
at a fixed point until the stack ran out.

**The fix.** One guard at each of the **three** free-flight release sites, placed
between `newstartpt` and `newendpt`:

```python
if not t_edge > 0: return newstartpt
```

Spelled `not t_edge > 0` rather than `t_edge <= 0` so a NaN stops too. Returning
`newstartpt` is safe: it is the separation point already displaced `EPS` along the
outward normal, so the agent ends outside the boundary — what `'sticky'` does anyway.
The sites are [_ibc.py:960](planktos/_ibc.py#L960) (static free-flight), and
[L1410](planktos/_ibc.py#L1410) / [L1418](planktos/_ibc.py#L1418) in the moving slider
(rotate-away release on `t_rot`, and its own free-flight on `t_edge`).

**No test**, deliberately: the case is unconstructible, which is why it stayed open.
The evidence the guard is inert is that the suite is unchanged with it in place.

**What this does *not* fix**, and cannot: `t_edge` positive but tiny still shrinks by a
factor of ~1 and recurses very deep. That is the "nothing bounds how much" property the
comment already flagged, and `_slide_too_deep` ([_ibc.py:189](planktos/_ibc.py#L189))
remains the backstop. ⚠️ Its message would be *misleading* in that scenario — it says
the step is long compared to the mesh, which is the opposite of what happened.

---

#### 3. 🟢 The mid-step excursion check misses about half the cases it exists for — **issue #73**

**Where.** `_project_and_slide_moving`, the "did we go past the end of the element?"
block ([_ibc.py:1122-1211](planktos/_ibc.py#L1122-L1211)).

**Why the check exists.** On a **moving** mesh the element translates, rotates *and
stretches* underneath the sliding agent. So "did the agent run off the end?" cannot be
answered from the end-of-step state alone: the agent can slide past an endpoint and
come **back onto** the element before the step ends. Missing that means it should have
been released at the endpoint and was not.

**What the check actually does.** It samples the along-element position at up to three
times and asks, at each, whether the distance from the agent to either endpoint exceeds
the element's own length ([L1129](planktos/_ibc.py#L1129), [L1181](planktos/_ibc.py#L1181)):

- **t = 1**, the end of the step;
- **`t_crit_elem`** ([L1153](planktos/_ibc.py#L1153)) — where the element is
  **shortest**. `|Qvec(u)|²` is an upward quadratic in the normalized parameter, so
  this is the sole root of its derivative;
- **`t_crit_x`** ([L1167](planktos/_ibc.py#L1167)) — where the direction of travel
  along the element **reverses**, i.e. where the projection of `vec` onto the element
  vanishes.

If any sample says "past", a least-squares solve ([L1204](planktos/_ibc.py#L1204))
finds the actual exit time.

**Why it is incomplete.** The excursion is the interval where the normalized
along-element coordinate `s(t)` leaves `[0,1]`, so its endpoints are the **roots of
`s(t)=0` and `s(t)=1`**. Those roots are determined by the slide ODE. Neither critical
time is such a root — both were chosen because they have closed forms, and they are
merely *plausible* places for an excursion to be caught. An excursion that opens and
closes strictly between the three samples is invisible to the test.

**What it costs.** The agent is carried along the element through the overshoot instead
of being released at the endpoint — a **position accuracy** error. It is **not** a
penetration: throughout the excursion the agent is projected onto the element's line,
so it stays on the correct side. No exception is raised.

**Measured:** 15 of 26 constructed excursion cases caught, **no false positives** (it
never claims an excursion that did not happen). Unchanged by the BUG-TCRIT fix — that
corrected the *algebra* of the two critical times; this is about the *heuristic built
on them*, which is a separate question.

**Fix shape.** Root-find `s(t)=0` and `s(t)=1` directly instead of sampling. `s(t)`
comes from the slide ODE, so these are numerical root-finds on an ODE solution rather
than closed forms — the same class as the least-squares already run at
[L1204](planktos/_ibc.py#L1204), but more of them and from a worse initial guess.
Substantial, delicate work in the most load-bearing code in the project.

**Why deferred.** Accuracy only, no penetration, no false positives — and moving
boundaries are **2D-only**. Revisit with 3D moving boundaries, where the moving slider
gets rewritten for triangles anyway and this decision has to be made again from
scratch. That work is **parked and not scheduled** — see the PARKED section below.

---

## Phase 1 — Test dynamic loading in 2D ✅ COMPLETE

All of (A)–(D) are done and pinned in `tests/test_dynamic_loading.py`; (E) is
superseded. **Only the periodic × dynamic combination is still untested**, and it
carries forward into Phase 2.

**The key structural insight, if extending this:** `update_spline` asks of a data
source only a `load_dumpfiles(d_start, d_finish)` returning per-component ndarrays
with a leading time axis. So a ~20-line synthetic `FluidData` subclass backed by an
in-memory array exercises the **real** slider exactly and deterministically, with no
files, in the fast suite. An earlier version of this plan wrongly assumed window
sliding could only be tested against real data.

**Ingestion is covered in both dimensions** against committed fixtures: `IB2dData`
from `tests/fixtures/ib2d_fluid_min` (8 dumps, vector form, `INUM=None/True/4`) and
the scalar `uX`/`uY` branch from `ib2d_fluid_scalar_min`; `VTK3dData` from
`tests/fixtures/vtk3d_min`. The in-memory linear path (`INUM=True`) has unit coverage
in `tests/test_flow_interface.py`.

- [x] **(A) Machinery correctness — DONE.** Windowed-linear (`INUM=4,5,7`) agrees with
  full linear (`INUM=True`) on forward sweeps, backward sweeps, non-monotone random
  access, exactly-on-node times, out-of-bounds clamping, and in 3D. Agreement is to a
  few ulp, not bit-for-bit, because the slider carries window-boundary values by
  *evaluating* the outgoing spline rather than re-reading raw data. **That error does
  not accumulate** — flat at 1–2 ulp across 400 loads and 6 full sweeps
  (`test_holdover_roundoff_does_not_accumulate`).
- [x] **(B) Window-sliding behavior — DONE.** Forward/backward slides, the
  jump-to-beginning fast path (asserted to be one load, not a walk), extrapolation
  flags at both dataset ends, bounded window and bounded load count across a full
  sweep, no load when the query stays inside the window, `fmin`/`fmax` widening across
  slides, and `get_raw_loaded_data` on a genuinely sliding window.
  - **Index-space bookkeeping bug found and fixed:** at the dataset end `idx_finish`
    was `len(flow_times)`, one past the last valid index, while `loaded_dump_bnds[1]`
    is inclusive. Latent rather than live (its only reader is gated off by
    `extrapolate[1]` at that point). `test_index_spaces_agree_at_every_slide` now
    locks the two together.
  - NB: the closing window is *structurally* ≤ `INUM` samples rather than `INUM`+1 —
    every forward slide pins `idx_start` to the outgoing window's next-to-last index
    (the two-sample holdover) and the dataset then runs out. Harmless (less memory,
    same values), which is why the bounded-window test asserts `<=`.
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
  | convergence order, **median** point | 1.68 | **4.75** | — |
  | convergence order, 99th-pct point | 1.44 | 2.20 | — |

  ⚠️ **The two convergence-order rows were corrected 2026-08-11** (from 1.75/4.85 and
  1.62/2.40). The withheld set included dumps lying *past the last build point*, where
  `FluidData` clamps and neither scheme interpolates — so both returned the same value
  and the "error" recorded there did not scale with Δt at all. It affected only the
  subsample factors that do not divide the series evenly (s=3, 1 sample of 99; s=6,
  4 of 124), which is why the shift is small and the qualitative picture is unchanged.
  **The velocity and ∂u/∂t rows never depended on it** — those are built at s=2, which
  divides 149 evenly and was always clean. Found while building the 3D counterpart,
  where 12 dumps made it 1 sample in 6 and it moved the ratio from 1.9× to 8.4×.

  - **Windowing is inert on real data too**, confirming the (A) result end to end:
    `INUM=4` vs `INUM=True` differ by 1.1e-16 (4e-17 of U_rms). So the whole question
    is linear-vs-cubic; the streaming machinery costs nothing.
  - **Do not quote the 2.4× ratio on its own — it blends two regimes.** Interpolation
    error is wildly concentrated (cubic max/median = 2.4e4×), so an rms-over-everything
    ratio is set by a rough minority. Where the flow is temporally smooth, both schemes
    hit their theoretical orders (cubic **4.75**, linear 1.68) and cubic is decisively
    better; where it is temporally rough, both stall (2.20 / 1.44) and the gap nearly
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
    run (deliberately, to isolate the interpolation). `INUM` docs should quote the
    orders and the ensemble result, not the bare ratio.
  - **Re-run on real 3D data 2026-08-11** — see Phase 2, "the machinery is vetted in
    3D". The ensemble result replicates (0.35% there vs 0.6% here); the convergence
    orders could **not** be measured on the 3D dataset, so these remain the only
    measured orders and the ones to quote.
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
  the real tiling implementation (`docs/notes/run_persistence.md` §9), which
  covers 2D and 3D together and must define how `tiling` interacts with
  `periodic_dim`. **Periodic × dynamic on its own is still worth testing** and stays
  in scope for Phase 1.

---

## Phase 2 — Dynamic loading in 3D against real data ✅ COMPLETE (2026-08-11)

The actual end goal (the ~100 GB case) and **the branch's remaining reason to exist.**

### The dataset is staged and analyzed ✅

`tests/unsteady_3D_testdata/` (gitignored as a whole directory). Contents: `VTK/` with
21 timestep `.vtm`s + a `.vtm.series` (case `case08_alpha2_1e8`), `oral_arm_disk.stl`,
and `README_oral_arm_setup.md` with the full setup spec.

**Physics:** OpenFOAM, Cassiopea oral-arm porous disk. Water (ρ=1000, ν=1e-6), Re=500,
laminar transient. Pulsing annular inlet, u_z(t) = 0.01·½(1−cos(2π·0.8·t)) m/s. Export
covers the last two pulse cycles (t ≥ 7.5 s, period 1.25 s). Domain x,y ∈ [−0.05, 0.05],
z ∈ [0.003, 0.271] m, **all lengths in meters**. Fields: `U`, `p` (kinematic),
`vorticity`, `Q`. `oral_arm_disk.stl` is a ready-made 3D immersed boundary, for whenever
moving boundaries in 3D are taken off the parking lot.

> ### ⛔ CORRECTION — no resample is needed
>
> An earlier version of this plan said the data was unstructured and had to be
> resampled to a rectilinear grid in ParaView/VisIt first. **That is wrong**, and it
> mis-framed Phase 2 as a data-prep task when it is a **loader-writing** task.
>
> The data *is* stored as `vtkUnstructuredGrid`, but that is only OpenFOAM's
> `foamToVTK` writer being generic. Verified by reading the files: 803,531 points /
> 775,368 cells, **every** cell a `VTK_HEXAHEDRON`, forming an exact 67×67×179 point /
> 66×66×178 cell **uniform Cartesian lattice** with constant spacing to float32
> roundoff and no refinement anywhere. `snappyHexMesh` was never run; the disk is a
> porosity source term, not mesh geometry. The point coordinates are **bit-identical
> across every timestep**, so geometry can be computed once and only fields streamed.

**Full structural analysis: [docs/notes/openfoam_oral_arm_dataset.md](docs/notes/openfoam_oral_arm_dataset.md).**
It was originally written into the gitignored data directory and was therefore
untracked; moved into `docs/notes/` 2026-08-10. §7 of that note is the loader work
list — it is not restated here.

### The real work: a loader for this format ✅ COMPLETE (2026-08-11)

Nothing in `planktos` read it before 2026-08-10. `VTK3dData` reads *legacy* rectilinear
vtk; `ComsolVTUData` reads a single `.vtu` holding all times. This data is a
per-timestep `.vtu` referenced through `.vtm` manifests — the **loop shape of
`VTK3dData`** with the **file format of `ComsolVTUData`**, and neither one as-is.

**Step 1 of 2 — the `_dataio` file readers — ✅ DONE (2026-08-10).** Three generic VTK
XML functions: `read_vtm_series` (JSON index → paths + times), `read_vtm_manifest`
(XML manifest → `{name: path}` + `TimeValue`), `read_vtkxml_cell_data` (`.vtu`/`.vtp` →
cell centers + selected cell arrays + time). Neither manifest reader touches VTK.
Pinned by 16 tests in `tests/test_io_loaders.py` against a new committed fixture,
`tests/fixtures/openfoam_min/` (305 kB in the working tree, **~29 kB packed** —
base64 of structured floats compresses ~10×; regenerate with
`tests/fixtures/_gen_fixtures.py`).
The fixture is a structural miniature of the real export and four of its properties
are load-bearing: **cell order is scrambled** by a fixed permutation, **two declared
dumps are absent** (interior hole + end truncation) with their manifests still present,
fields are analytic (`U=(t,x,t·z)`, `vorticity=(z,y,x)`, `p=t+y`, `U≡0` on walls), and
the `.vtu`/`.vtp` are written **uncompressed, inline base64, `UInt64` headers** to match
foamToVTK's byte layout — pyvista's `save()` defaults to zlib/`UInt32`, which would
quietly invalidate the fixture for the §7 direct-parse optimization.
Verified against the real 1.5 GB dataset during development; per the decision that the
data lives on one machine, **no permanent test depends on it.** Full loader details in
note §7.

**Step 2 of 2 — `OpenFOAMData(FluidData)` + `Environment.read_openfoam_vtk_data` — ✅
DONE (2026-08-11).** The real dataset loads, streams, and drives agents: 17 dumps,
`fshape (17, 68, 68, 180)`, `L = [0.1, 0.1, 0.268]` — the true domain, not the
half-cell-short 0.0985 x 0.0985 x 0.2665 raw cell centers report. Construction with
`INUM=4` takes 5.8 s; a full sweep with three window slides 12.2 s; 200 agents over
20 steps ran with none lost and a finite material derivative. 19 tests
(`test_io_loaders.py` for ingestion, `test_dynamic_loading.py` for the timeline and
the slide), all closed-form against the fixture.

Three implementation notes worth keeping:

- **Dump numbers are a dense 0-based index over the dumps that exist**, not the numbers
  in the directory names. This is forced, not chosen: `update_spline` does index
  arithmetic on `d_start`/`d_finish` (`d_start = loaded_dump_bnds[1]+1`, and so on), so
  dump numbers must step by one in lockstep with `flow_times`. OpenFOAM's (787, 800, …
  1034, with holes) are neither. Indexing the survivors makes the `FluidData` guard hold
  by construction and needs no change to `update_spline`.
- **Faces are identified geometrically, not by patch name.** Each is decided by which
  interior axis range its cells fall outside of. So one file holding four planes
  (`walls.vtp`) splits correctly, case-specific names (`inlet`/`outlet`/`walls`) do not
  leak into the loader, and a future flow tank whose top is not a wall works unchanged.
  Every face carries its own patch data rather than an assumed no-slip zero.
- **The mesh is assumed static across the series** (verified for this dataset: the point
  arrays are bit-identical). Only a differing cell *count* is caught; a same-count
  reordering would pass silently. See the robustness list below.

Four things make it more than a format swap (all detailed in the note):

- [x] **Fields are `CELL_DATA`, not point data** — OpenFOAM is finite-volume, and
  `point_data` is empty. Handled by the new `_dataio.read_vtkxml_cell_data`, which
  reads `GetCellData()` and returns `vtkCellCenters` output rather than `GetPoints()`
  (hexahedron **corners**, the wrong lattice). Written as a **new** function rather
  than an edit to `read_vtu_Unstructured_Grid_Points_FEM` as originally planned —
  that one is live for `ComsolVTUData`, which wants point data on FEM corner points.
- [x] **Cell ordering is not lexicographic.** A bare `.reshape((66,66,178))` silently
  scrambles the field. Handled by `OpenFOAMData._build_lattice`, computed **once** since
  the mesh is static.
  - Built by **clustering** each coordinate into levels, not by the note's
    `np.rint((x-min)/dx)`: rint needs `dx`, i.e. a *uniform* lattice, whereas Planktos
    assumes only *rectilinear* — and the assembled grid stops being uniform the moment
    the boundary patches are spliced on. Robust across `rel_tol` from 1e-3 to 1e-8 on
    the real data (roundoff sits ~1e-8 below, true spacing ~5.6e-3 above).
  - ⚠️ **`np.unique` is unusable here too, not just `lexsort`** — the float64 cell
    centers show **79 distinct y-levels where 66 exist**. But the *values* are exact:
    a level holds at most 8 distinct float64s spanning 7.6e-19, which is **5e-16 of a
    cell width**, and mean/first/min all reproduce an independently-written boundary
    patch's lattice bit-for-bit. What `np.unique` gets wrong is the grouping, not the
    coordinates. Note §2.
  - The completeness check — the linear index must be a permutation of `arange` — is
    what verifies Planktos' rectilinear-grid assumption for a given dataset. It is
    total: it fails on a missing cell, a duplicate, refinement, or any non-tensor mesh.
- [x] **Cell centers are inset half a cell**, so raw centers report the domain as
  0.0985 × 0.0985 × 0.2665 instead of the true 0.1 × 0.1 × 0.268. The `.vtp` boundary
  patches close this **exactly**: `inlet`/`outlet` carry real data on the identical x/y
  lattice, `walls` is exactly zero (no-slip) and can simply be padded. Assembly is
  66×66×178 → 66×66×180 → 68×68×180. ⚠️ The 12 edges and 8 corners appear in no file;
  the claim that they all lie on no-slip walls was **half right** — see the corner item
  below, where the outlet turns out to impose no no-slip.
  - Both claims now **verified through the reader** rather than asserted: once snapped,
    the inlet's x/y grid vectors match the interior's bit-for-bit, and `walls.vtp` `U`
    is exactly zero everywhere.
- [x] **Times come from the `.vtm.series` JSON** (plain JSON, no VTK needed), or the
  per-file `TimeValue`. Both readers exist (`read_vtm_series`, `read_vtm_manifest`) and
  the two sources are confirmed to agree on all 21 real dumps. ⚠️ **The `VTK3dData`
  timeline trap still applies to the consumer** — see the fixed bug below. `flow_times`
  must span the **whole series** from the start, not just the opening window.

**I/O budget, which is the whole point of the branch:** 61% of every 85 MB file is
geometry that never changes. Going through `vtkXMLUnstructuredGridReader` re-reads
51 MB per timestep to extract 12 MB of velocity. Options in the note §7 (disable unused
arrays / parse the `U` `DataArray` directly / one-time `.npy` preprocess).
**Get the VTK path correct first; treat this as an optimization.**

- [x] ✅ **The loader tolerates a dump series with holes — warns, does not fail.**
  4 of 21 timesteps in this dataset are missing (t = 9.0, 9.375, 9.75, 10.0): the
  `.vtm` manifests exist but the directories they name do not, a truncated transfer.
  **Decision was to work with the data as-is** and treat gap tolerance as a loader
  requirement in its own right — truncated or interrupted exports are normal.
  As built: existence resolved **eagerly at construction** (never at the window slide
  that needs it, which under streaming would land arbitrarily deep into a run); one
  warning naming the missing times and the count; `flow_times` and the dump index built
  **densely over the survivors**; and a **separate** warning about the resulting uneven
  spacing, naming where it widens, since interpolation error scales with the dump
  interval (Phase 1 (C)). Confirmed on the real data: 17 dumps kept, the three widened
  intervals correctly identified.
  - ⚠️ The `flow_times`-must-span-the-whole-series guard reads "whole series" as **the
    dumps that exist**, not the set the index declares. Pinned by
    `test_openfoam_flow_times_span_the_surviving_series`.

### Robustness follow-ups — 🟡 ONE LEFT

**Done (2026-08-11): the time-source fallback chain — items 1, 2, 3 and 7.**
**Done (2026-08-12): mesh verification (5), `require_boundary=False` (4), and the
boundary-condition corner policy (8).**
Still open: **surfacing a stored `vorticity` (6)**, which is the last item in the
Phase 2 block that is not explicitly deferred.

**Decision (2026-08-11), superseded the same day.** The original call was to write
fallbacks only against a delivery that actually exhibits the problem, since fallbacks
built for input nobody has sent tend to be wrong when it arrives. That still governs
items 4, 6 and 8. The *timeline* fallbacks (1–3) were built up front instead: they are
pure reads of information the export already carries, and the three `_dataio` readers
were written as independent functions precisely so this could be a chain.

- [x] ✅ **Missing `.vtm.series`** → glob the `.vtm` files; take each time from its
  `TimeValue`. Built as `OpenFOAMData._candidates_from_manifests`. The manifests
  survive for absent dumps, so the gap warning still fires on this path.
- [x] ✅ **Missing `.vtm` manifests too** → glob the dump directories; read `TimeValue`
  from `internal.vtu`. ⚠️ Sort **numerically**: directories are named with unpadded
  numbers (`case08_alpha2_1e8_787`, `..._1008`), so a lexical sort puts 1008 before 787.
  Built as `_candidates_from_dirs` + `_natural_key`. Two notes on how it landed:
  - Time comes from a **bounded header scan**, new `_dataio.read_vtkxml_time_only`
    (1.0 ms vs 0.75 s per 84 MB file). Without it, timestamping the series would
    re-read ~1.4 GB of static mesh to recover 17 floats. Patches are found at
    `boundary/*.vtp`, since no manifest names them any more.
  - ⚠️ **This tier cannot see a missing dump** — nothing declares the ones that never
    arrived once the manifests are gone, so the widened interval and its warning are
    the only trace. Pinned as such.
- [x] ✅ **No time information anywhere** → warn, assume unit steps (the
  `VTK3dData._read_all_times` precedent). Deliberately narrower than that precedent,
  which takes unit steps when *any* dump is untimed: a **partly** timed series raises
  instead, because overwriting a mostly-real timeline with indices would move every
  dump that did carry a time. This is also the only place the numeric sort is
  load-bearing — with no times to sort by, the filename order *is* the timeline.
- [x] ✅ **`require_boundary=False`** → extrapolates the interior out to any face with
  no boundary patch, instead of raising. Two steps, both done.
  - [x] ✅ **(2026-08-12) `fluid.center_cell_regrid` is back as a module-level
    function, extended to rectilinear grids and tested.** Takes `(flow, flow_points)`
    and returns them with one grid plane added at each end of each spatial axis, at the
    domain boundary. Handles 2D/3D, a leading time axis, and periodic axes; returns
    points in the input coordinate system, so shifting to quadrant 1 stays the caller's
    business as it is for the loaders' own grids.
    - **Rewritten, not restored.** The legacy body assembled the shell of boundary
      points, interpolated it in one call, and unpacked the flat result back into
      faces/edges/corners — ~150 lines of index bookkeeping, and the 3D corner list
      and its unpacker disagreed, so corner `(x+,y-,z-)` was silently filled from
      `(x+,y-,z+)`. Extending one axis at a time is ~20 lines, gets edges and corners
      for free as the tensor-product extension, and has no corner table to get wrong.
      **The commented-out legacy block has been deleted**: its useful parts are back
      in force, and leaving it would invite restoring the bug.
    - ⚠️ **Where the boundary is, is a guess on a stretched grid.** n cell centers
      give n equations in n+1 faces, so the sequence is short one piece of
      information; the two outermost faces are taken half the distance to the
      neighboring center. Exact on a uniform grid, biased by the local stretch ratio
      otherwise (widths w, rw give w(1+r)/4 against a true w/2). Warns, naming the
      axis. A `bounds=` argument takes the true edge per end where the caller knows
      it, which suppresses both the inference and the warning for that end.
  - [x] ✅ **(2026-08-12) Folded into the OpenFOAM loader.** `require_boundary=False`
    now fills each uncovered face from `center_cell_regrid` and warns; `True` still
    raises, with the message pointing at the option.
    - **Per face, not per dataset.** A patch carries the boundary condition the solver
      applied and is strictly better than extrapolating, so every face that has one
      keeps using it. Dropping `walls` from the fixture leaves inlet/outlet on their
      patches while the four lateral faces are extrapolated — and since the walls are
      no-slip zero while the linear extension is not, that difference is what the test
      asserts.
    - `_build_grid` resolves all six edges (patch plane where there is one, the
      `_infer_domain_edges` closure otherwise) and `_read_dump` passes those back as
      `bounds=`, so the grid the loader publishes and the grid the field was extended
      onto cannot disagree.
    - Edges and corners are still decided by the existing stage 2/3 averaging, whatever
      mix of patched and extrapolated faces meets there — one rule, unchanged, so the
      mixed case needed no new policy. The regrid's own edge/corner values are
      discarded; only each face's interior run is taken, which depends solely on the
      sweep along that face's own axis.
    - ⚠️ **`_verify_dump_mesh` (item 5) reads the interior lattice as `_grid[d][1:-1]`.**
      Still true: `_build_grid` always puts an edge coordinate at each end, inferred
      or from a patch, so the slice still means "the cell centers".
    - ⚠️ **The `bounds=` pass-through is not distinguishable on the reference
      dataset** — its mesh is uniform and its patches sit exactly half a cell out, so
      the inferred and true edges coincide. `center_cell_regrid`'s own tests pin the
      mechanism; the loader-level call is correctness insurance for a stretched mesh.
- [x] ✅ **Mesh verification (second dump).** The mesh is read once and the permutation
  built from it; a same-count reordering (a series stitched from two runs, a corrupt
  file) would otherwise pass silently with every value in the wrong place. Built as an
  **automatic check of the second dump**, interior and boundary patches alike, plus
  **unconditional cell-count checks** on every dump — the interior one that existed,
  and a new one per patch, since a patch is indexed by a selection built at
  construction and a shorter one would take the wrong cells or raise a bare
  `IndexError`. Tolerance is `_cluster_axis`'s own `rel_tol=1e-5` of the axis span.
  - **Automatic, and second-dump rather than every-dump.** An opt-in gets forgotten by
    the people who need it, and per-dump checking measured **+8.2%** on the real
    series (18.2 s → 19.7 s for a full load). Checking one dump is free — dump 1 is
    always in the opening load, so it adds only that dump's cell centers — and the
    benefit falls off after the first comparison, since cell ordering is a property of
    the writer and a rectilinear mesh is not adaptively refined.
  - ⚠️ **Does not catch a series stitched mid-run** (dumps 0–11 from one case, 12–20
    from another). Widening to every dump is a one-line change at the caller —
    `_read_dump` already takes the flag per call.
  - Pinned on the **windowed path as well as full-load**, since a check that quietly
    stopped running under streaming would be missing in the configuration this branch
    exists for. A separate test asserts no other dump ever loads cell coordinates,
    across a slide to the end of the series and back.
- [ ] **Surface the stored `vorticity`** instead of regenerating it. Exports usually
  ship it; `_dataio.read_vtkxml_cell_data(arrays=...)` already reads it on request. The
  work is on the `FluidData`/`get_vorticity` side, not on the reader.
  - **Decided (2026-08-12): load on demand, do not carry it in the window.** A stored
    field is the same shape as the velocity, so holding one would roughly double the
    resident fluid (the measured +106 MB streaming window at `INUM=4` becomes ~212 MB,
    the 340 MB full load ~680 MB) — against the constraint this whole branch exists
    for. Reading it costs little: 0.87 s → 0.94 s per dump, only +8%, because 61% of
    every file is geometry re-read regardless. And `get_vorticity` is an analysis and
    plotting call, not something the move loop touches, so paying a read at call time
    is fine. Shape: a per-subclass hook returning None by default, which
    `OpenFOAMData` implements by reading the bracketing dumps and interpolating with
    the same weights the velocity uses.
  - ⚠️ **The measurement that motivates it — recomputation is exact except in one cell
    layer.** Comparing `get_vorticity` against the solver's stored field, dump 0 of the
    reference dataset, by depth from the nearest domain face:

    | cells in from nearest face | count | rms difference | % of stored rms |
    |---|---|---|---|
    | 0 | 54,472 | 0.0969 | **84%** |
    | 1 | 52,040 | 2.3e-08 | 0.00% |
    | 2 | 49,656 | 3.4e-08 | 0.00% |
    | ≥5 | 526,848 | 1.7e-08 | 0.00% |

    So in the bulk Planktos and OpenFOAM compute literally the same central-difference
    curl — 1e-7 relative agreement validates `get_vorticity` rather than indicting it.
    The whole problem is one cell deep, over 7% of the domain, where it is unusable
    (worst cell 11× the field's own rms). Suspected cause, **unverified**: at the
    outermost interior cell `np.gradient` differences against the spliced boundary
    plane across a half-cell, where that plane holds the boundary *condition* value,
    while the solver uses its own finite-volume wall treatment.
  - For contrast, on a plain (unspliced) 2D grid the edge is merely less accurate, not
    broken: 2.81% of rms against 0.11% in the interior, on an analytic Taylor-Green
    field. **Trimming the boundary layer out of 2D vorticity plots is therefore not
    warranted** — and would not reach the 84% case anyway, which is 3D, where no fluid
    backdrop is drawn at all.
  - 📋 **Folded into `docs/notes/run_persistence.md` §3.3** *(2026-08-13)*, which
    is now the single plan — the standalone `stored_derived_fields.md` was merged into
    it and deleted. Item 6 is subsumed there rather than standing alone, because what
    settled it was the plot cache's needs. In short: nothing is written under
    `INUM=None` (recompute from resident velocity, cubic); under `INUM=int` the source's
    field is read if it has one, and otherwise Planktos writes one in the source's own
    format. Reproduce the measurements with
    `tests/manual/bench_vorticity_sources.py`.
- [x] ✅ **Whichever fallback is taken, say so.** Silently accepting a degraded timeline
  is the shape of the `VTK3dData` frozen-fluid bug. Every step of the chain past the
  first warns, and the step taken is recorded on the object as `dump_source` /
  `time_source` so it can be inspected after the fact. Two guards came with it, since
  a rebuilt timeline is the thing most likely to come out wrong: dumps found by
  globbing are **reordered by their recovered times** when the filenames disagree
  (warning if so), and a timeline that is not strictly increasing **raises** — both
  splines divide by the interval between successive times, so a repeat is unusable
  rather than merely degraded.
- [x] ✅ **Boundary-condition corners — zero wins.** The 12 edges and 8 corners appear
  in no patch file and are filled from the faces meeting there. Where those disagree,
  **a velocity of exactly zero is taken as no-slip and wins**; everything else is still
  the average. A wall is no-slip along its whole length, including where an inflow runs
  into it, so averaging smeared a nonzero velocity onto a surface the fluid cannot move
  along.
  - **Exact zero, whole vector** — no-slip makes every component vanish and the
    exporter writes 0.0, whereas one component vanishing is ordinary (w = 0 on a
    z-normal inlet plane). A tolerance would misread slow near-wall flow as no-slip.
    Corners apply the same rule to their three edges.
  - **264 edge cells on the real dataset**, with the wall strips now reading exactly
    0.0 where they carried about half the outlet's 7e-4. The count is exact, not
    approximate: 2 of the 4 edges running in x and 2 of the 4 in y put a wall against
    the *outlet* (2×66 + 2×66). The inlet ring is itself zero, and the 4 edges running
    in z are wall-against-wall, so neither is affected. Supersedes the "~272" estimate.
  - **No `bc_corner` argument** — an option nobody sets is dead weight, and this is a
    strict improvement for any export marking no-slip as zero. Restoring pure averaging
    is a one-line change at the two sites.

### Already fixed ahead of the real data ✅

- [x] **`VTK3dData` dynamic loading was silently broken.** `flow_times` was built from
  only the dumps read at construction, so on the windowed path it held `INUM+1`
  timestamps instead of the whole series. That did not raise: it made
  `INUM >= len(flow_times)-1`, which sends `FluidData` down its "everything is in
  memory" branch with `extrapolate=(True, True)`, making `update_spline` unreachable.
  The run completed with the fluid **frozen** at the end of the opening window — no
  error, no warning. (`IB2dData` computes `flow_times` analytically over the full
  range, which is why 2D was never affected.)
  - **Fix:** `TIME` lives in each legacy-vtk header, so the whole series is timestamped
    with one ~4 kB header read per dump — 62× faster than a full parse and *constant*
    in file size, so the margin grows with real dumps. New `_dataio.read_vtk_time_only`
    with a per-file fallback to a full read.
  - **Guards added to `FluidData`** so the next loader making this mistake fails loudly:
    a dynamically-loading subclass exposing `d_start`/`d_finish` must pass `flow_times`
    covering the full dump range, and an int `INUM` spanning the dataset now warns that
    no dynamic loading will occur.
  - **Coverage:** `tests/fixtures/vtk3d_min/` — 8 tiny rectilinear vtk dumps with `TIME`
    (`u=t, v=x, w=t*z`, so a frozen or truncated timeline is unmistakable).

### The machinery is vetted in 3D ✅ (2026-08-11)

Run by **`tests/manual/vet_dynamic_loading_3d.py`**, the 3D counterpart to
`quantify_temporal_interp.py`, against the real 17-dump OpenFOAM export. It is
self-documenting and reproducible; run it with no arguments, or with part numbers
(`0 1 2 3`) to pick sections. It needs the gitignored dataset, which is why it lives in
`tests/manual/`.

- [x] **(A) Windowed-linear == full-linear on real 3D data.** Forward sweep, backward
  sweep, non-monotone random access, exactly-on-node times, out-of-bounds clamping:
  **max |diff| = 0.0 exactly** on the sweeps and the clamp, 1.2e-33 on random access,
  5.1e-21 on-node. Holdover round-off stays at **0.0 across six alternating sweeps** —
  it does not merely fail to accumulate, it does not arise. (2D saw 1–2 ulp.)
- [x] **(B) Slider bookkeeping.** Window bounded to 5 samples; both index spaces agree at
  every slide and match `flow_times`; a monotone sweep costs **4 loads / 12 dumps read**
  for 17 dumps of data; no load when the query stays inside the window; jump-to-start is
  **one** load; `fmin`/`fmax` stay tuples and bracket the true extrema exactly; indexing
  a streaming object is refused.
- [x] **(D) `get_dudt`.** Matches full linear to **0.0** across slides; constant within an
  interval; equal to the interval finite difference to 0.0; jumps at a breakpoint.
- [x] 🔴 **Memory profiling — the branch's headline claim, now measured.** Streaming at
  `INUM=4` costs **+106 MB** against **340 MB** for the whole series resident — one
  window (5 dumps x 20 MB), as designed. Flat across 122 queries spanning two full
  sweeps (**end − ctor = −0 MB**), so nothing leaks per slide.
- [x] **3D material derivative end to end.** `DuDt` finite and correctly shaped
  (|DuDt| rms 9.9e-4 m/s²); an `inertial_particles` swarm runs 20 steps on streamed
  fluid with 200/200 agents retained.
- [x] **(C) Interpolation error re-checked on 3D data.** At the only interval a
  withholding study can reach here (Δt = 0.25 s, subsampled from 0.125), linear is
  **9.46%** of U_rms against cubic's **1.13%** — a **8.4× ratio**, much wider than 2D's
  2.4×. ∂u/∂t: 10.42% vs 3.36%, ratio 3.1×.
  - ⚠️ **The convergence orders could NOT be measured on this dataset, and the fitted
    values it produces are meaningless.** Twelve uniform dumps allows only subsample
    factors 2 and 3 — a Δt lever arm of 1.5×, far too short for a log-log slope — and at
    factor 3 the cubic has just 4 knots, where not-a-knot degenerates to a single cubic
    polynomial and comes out *worse than linear*. The script now detects both conditions
    and says so rather than printing an order.
  - **The underlying reason is physical and is the real finding: this export's cadence is
    marginal.** The pulse period is 1.25 s, so Δt = 0.125 s is 10 samples per cycle and
    the subsampled 0.25 s is 5. Neither is deep into the asymptotic regime. **Quote the
    2D orders**; treat the 3D dataset as evidence about error *size* at a coarse cadence.
  - **The practical answer replicates.** 512 tracers over 60 steps: rms separation 0.209%
    of path travelled (0.001% of the domain diagonal), and ensemble statistics — mean/std
    of each coordinate, mean/std of net displacement, its 10th/50th/90th percentiles —
    agree to **within 0.348%**, against 0.6% in 2D.

### Found by the vetting ✅

- [x] **`update_spline` mislabelled the final window** (`fluid.py`). The end-of-dataset
  test was `> d_finish` where it needed `>= d_finish`, so a window landing *exactly* on
  the last dump fell through to the middle branch and was flagged
  `extrapolate=(False, True)`→`(False, False)` — claiming there was more data to the
  right when there was none. Forward slides advance the window end by `INUM-1`, so for
  `INUM=4` it needs `len(flow_times) ≡ 2 (mod 3)`; **17 dumps lands exactly**, which is
  why the real dataset exposed it and the fixtures never did.
  - **Latent, not live:** `__call__` clamps to `flow_times[-1]`, so it never asks for a
    time past a window that already ends there, and the degenerate branch stays
    unreachable. The damage was a dishonest flag for anything else reading it.
  - ⚠️ **The new loader had a sharper edge than the old ones here.** Had that branch been
    reached, `load_dumpfiles` would get `d_start > d_finish`; `IB2dData`/`VTK3dData` slice
    arrays and get a correctly-shaped `(0, …)` empty for free, but `OpenFOAMData` builds a
    list and produced `np.array([])` of shape `(0,)`, which fails to concatenate against
    the window with a message naming neither the range nor the cause. Now returns properly
    shaped empties.
  - Pinned by `test_final_window_is_flagged_as_the_end_whatever_the_dataset_length`,
    parametrized over nt=17..21 — verified to fail at 17 and 20 without the fix.
- [x] **A methodology bug in the withholding studies**, in the harness rather than the
  library: the withheld set included dumps past the last build point, where `FluidData`
  clamps and neither scheme interpolates. Corrected in both scripts; it moved the 3D
  ratio from 1.9× to 8.4× and the 2D convergence orders slightly (see Phase 1 (C)).

### A second pass over what else exercises the slider ✅ (2026-08-11)

Everything above calls the `FluidData` object directly. A pass over the paths a *user*
goes through found six that had never been run against a streaming source, all now
pinned in `test_dynamic_loading.py` under "the slider driven through the public API".
Each asserts the streamed answer equals the everything-in-memory answer, so a slide
landing on the wrong dumps cannot pass. **No new defects** — the value is that these
combinations are now locked rather than assumed.

- [x] **Periodic × dynamic** — the surviving half of Phase 1 (E), untested in either
  dimension until now. Verified in 2D and 3D, sampled at the upper grid edge where
  wrapping decides the answer, after the window has moved off the opening one. They are
  independent, as expected: `periodic_dim` is a property of the spatial grid and the
  slider only ever touches the time axis. Expected, but no longer only expected.
- [x] **A slide inside `Swarm.move()`** — the usage that actually matters, and the one
  nothing drove before: 38 steps carry the window from (0,4) to (15,19). With diffusion
  off, the streamed trajectory matches the resident-fluid trajectory to 1e-12.
  - ⚠️ Needed a **gentler test field**: `_field_2d`'s `v = t²y` ejects every agent within
    a few steps, and a test whose agents have all left the domain proves nothing about
    the slider. `_field_gentle` exists for that.
- [x] **`Environment.move_swarms`** — several swarms sharing one environment reload the
  window once per slide, not once per swarm.
- [x] **`calculate_FTLE`, forward and backward** — walks the window across the dataset,
  and `backward=True` walks it the other way, the direction least exercised elsewhere.
  Matches a resident fluid to 1e-10.
- [x] **`get_vorticity(time=)`** — the `fluid='vort'` plot backdrop is the one plotting
  path that still pulls the field (the statistics come from the mean cache), so it can
  trigger a load.
- [x] **`save_fluid(flow_times=True)` from a window parked at the end** — jumps back to
  the beginning, then sweeps forward writing every dump. The strongest of these: what is
  checked is the *bytes on disk* against a resident fluid, so a slide landing on the
  wrong dumps shows up as wrong data rather than a wrong index.
- [x] **The OpenFOAM loader's own mean cache and `get_dudt` across a slide**, mirroring
  the `VTK3dData` coverage.
- [x] **Parallel immersed boundaries × dynamic loading** — verified by hand that serial,
  threads and processes give bit-identical results with the window sliding. **No test
  added:** the fluid never crosses the process boundary (the pool receives collision
  geometry, and the fluid is evaluated in `apply_agent_model` before it is used), so the
  combination is structurally orthogonal and a slow test would buy little.

### Closed out by the Phase 2 work ✅

Both of these were listed as remaining; neither is.

- [x] ~~Update `docs/api/FluidData.rst` with the 3D figures.~~ **Done** — the
  "the 3D case has not yet been characterized" caveat is replaced by the 3D result, and
  the note now records that the orders quoted are the 2D ones because the 3D dataset
  cannot support the fit. One conclusion changed: the old text said a coarser cadence
  means "a smaller gap between them", but the 3D dataset is the more coarsely sampled
  and shows the **wider** gap (8.4x vs 2.4x), so that claim is withdrawn.
- [x] ~~Un-skip / fix the IBAMR load tests on real data.~~ **Stale item — they were
  never skipped.** `tests/IBAMR_test_data/` (3 dumps + mesh) is **tracked in git**, and
  the `vtk` marker's gate resolves to True because vtk is a mandatory dependency, so
  `test_IBAMR_load_single_time`, `test_IBAMR_load_time_series` and
  `test_unstructured_grid_points_reader` run everywhere, including CI, and pass. Note
  the 3-dump series is too short for windowed loading (`INUM>=4` needs 5 points), so
  that data exercises ingestion only.

## PARKED — 3D moving immersed boundaries ⚪ (not scheduled; do not start)

> **This is a parking lot, not a plan.** 3D moving boundaries are **out of scope for
> this release line** and no work on them should be picked up from this file. The list
> survives only so the pieces are not rediscovered from scratch much later. Phase 2
> unblocked this, which is a fact about dependencies, not a scheduling decision.

Moving boundaries are currently 2D only. 3D immersed boundaries are **STL triangular
(FEM) surface meshes** (3D vertex-point input deprecated; 2D vertex points still used).
Inherited blockers from the overhaul's notes:

- The 3D *moving*-mesh code path currently raises (not implemented).
  Static 3D collision coverage is already in place (`test_collisions_static_3d.py`,
  `test_collisions_stl_3d.py`).
- **Moving-mesh FTLE:** `calculate_FTLE` never advances `envir.time`, so a moving mesh
  is frozen at t0; it raises `NotImplementedError`. A real fix threads integration time
  into `interpolate_temporal_mesh` (forward + reversed) — delicate collision-path work.
- `_ibc` **finding #3 / issue #73** (the mid-step excursion check) is deferred to here,
  since the moving slider gets rewritten for triangles anyway. See that section above.

---

## Documentation 🟡

- [x] **`FluidData` on readthedocs — DONE.** `docs/api/FluidData.rst` autoclasses
  `FluidData` and the three per-source subclasses, plus narrative on how to get
  velocity data out (call it with a time vs. index it like a list, and why indexing is
  refused while streaming) and on the `INUM` tradeoff. `Environment.rst` points to it,
  since `INUM` is met first through the `Environment` reader methods. Sphinx builds
  clean.
- [ ] **Sweep the prose docs for the master-era fluid API.** `docs/quickstart.rst`
  and `README.md` still frame fluid handling as `Environment`-level. The 1.0.1
  merge fixed the two that were outright wrong (`get_2D_vorticity` → `get_vorticity`,
  and the claim that flow can be "extended"), but the overall framing still assumes
  master's API. Worth a pass once the fluid API stops moving — **still deliberately
  held**, since §9 decides what `tile_flow` becomes and whether `Environment.extend`
  returns, and both are exactly what this prose would describe. The new
  `docs/api/FluidData.rst` covers the reference side in the meantime.
- [x] **`INUM` and the linear-vs-cubic tradeoff — DONE.** In `docs/api/FluidData.rst`
  (anchor `inum-tradeoff`): the `INUM` table, why linear in time is permanent rather
  than a placeholder, what it costs, and the practical reading (ensemble statistics
  are unlikely to change; prefer `INUM=None` for inertial particles when the data
  fits). Per Phase 1 (C) it quotes the **convergence orders and the ensemble result,
  not the bare rms ratio**, and states that absolute errors do not transfer between
  datasets while the orders do.
  - Remaining: re-check the numbers on 3D data in Phase 2 and update the page.

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

> **Most source-specific fluid ingestion is out of scope for this branch.** Planktos
> assumes a **rectilinear fluid grid**. Reading IBFE SAMRAI or COMSOL directly —
> including porting the old VisIt `read_IBAMR3d_py27.py` SAMRAI→vtk script — is lower
> priority than the robustness pass above and is scratched here.
>
> **The exception is the Phase 2 OpenFOAM reader, which is in scope and is 🔴.** That
> dataset is already on a uniform Cartesian grid despite its `vtkUnstructuredGrid`
> container, so it needs a loader rather than a resample. See Phase 2 and
> `docs/notes/openfoam_oral_arm_dataset.md`. Background in CLAUDE.md ("3D fluid data
> sources").

- [ ] **COMSOL VTU loader** (`ComsolVTUData`) — existing, full-load only. Verify only if
  needed; collaborator no longer uses COMSOL and the export format has likely changed.
  Also: the skipped `test_io_loaders.py::test_vtu_load` needs a committed COMSOL fixture
  (`tests/data/comsol/vtu_test_data.txt`) or stays gated.
- [ ] **NetCDF** (`load_NetCDF` / `read_NetCDF_flow`) — existing, full-load only. Never
  actually used (reviewer-requested for a prior publication). Lowest priority.
- [ ] **Rectilinear (non-uniform) grid support in `calculate_FTLE`** — a *feature*, not
  a bug fix, and **not** needed by the OpenFOAM dataset.
  - ⚠️ **Correcting a claim made in this file on 2026-08-11**, that Phase 2 had made this
    urgent because the boundary splice produces a non-uniform grid. It does not follow.
    `Swarm.grid_init` seeds tracers on `np.linspace(0, L, n)` — a **uniform** grid built
    with no reference to `flow_points` — and `calculate_FTLE`'s `dx = L[0]/(grid_dim[0]-1)`
    is exactly that grid's spacing. The fluid grid never enters the gradient. Measured
    against an analytic strain field whose FTLE is exactly `a` on any grid: a uniform
    fluid grid and one with **79x spacing variation** give **bit-identical** FTLE fields.
    For the oral-arm grid the seed spacing would be 0.001493 against the fluid's 0.001515
    interior — 1.5%, and irrelevant.
  - What it would actually buy: resolution matching on a genuinely refined mesh, where a
    uniform seed grid over-resolves the coarse regions and under-resolves the fine ones.
    `grid_dim` defaulting to `len(flow_points[d])` implies an intent to match the fluid
    grid that is not honoured for non-uniform grids.
  - Shape of the change: `grid_init` takes explicit coordinate arrays; `calculate_FTLE`
    takes `grid_points`; and the central-difference denominator `x_mult*dx` becomes
    `xgrid[n1] - xgrid[n0]` from the stencil indices it already holds, which is correct
    for both cases and deletes the `x_mult`/`y_mult`/`z_mult` bookkeeping. ⚠️ A centred
    difference on a non-uniform grid is only **first**-order unless the weighted
    three-point formula is used — decide that deliberately.
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
  tiling implementation, `docs/notes/run_persistence.md` §9.
- [ ] `Environment.extend` was removed (extrapolation is the intended replacement).
  Whether it returns is decided in `docs/notes/run_persistence.md` §9, alongside
  the real tiling work — the two are the same class of operation (reported domain ≠
  stored grid) and should share a mechanism. The parked test
  `test_flow_generation.py::test_extend_grows_domain_and_copies_edges` un-skips if so.
- [ ] **Optional agent-history retention (maybe-feature).** `Swarm.pos_history` grows
  every step, so long runs accumulate memory whether or not anyone plots them. A flag
  along the lines of `store_pos_history='all' | 'frames' | None` would let a user cap
  it (`store_prop_history` is the naming precedent).
  - **Without an archive the loss is unrecoverable:** decimating history breaks
    `plot_all` at full step resolution, `save_data`, `save_pos_to_csv`,
    `save_pos_to_vtk`, and any post-hoc agent analysis (per-step
    displacement/dispersal statistics). Opt-in, off by default, loudly documented.
  - ⚠️ **Revisit this once the run archive lands** (`docs/notes/run_persistence.md`
    component A). The archive writes agent state to disk as the run proceeds, so disk
    then holds what memory drops and every one of those consumers can read it back.
    The feature changes character completely — from "lossy, opt-in, loudly documented"
    to "cap memory, lose nothing, provided you are recording." `run_persistence.md`
    §2.10 records the reasoning. It was left out of the original plotting redesign as
    out of scope; the reframe brought it in.

---

## How to run the tests

- Fast loop: `pytest` (≈4s; skips `slow` / `vtk`-absent / `vtu`-absent).
- Full: `pytest --runslow` (adds the parallelization checks and the plotting
  smokes; ≈15s).
- List any `xfail`s with `pytest -rX` — a non-empty list means work is in flight.
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
   `trim_end` was itself deleted 2026-08-11, having outlived the approach unreferenced.
3. **Left-based cubic spline — failed.** `_left_based_cspline` / `_extend_prev_spline`
   forced both boundary conditions onto the *left/known* end so the window could grow
   rightward. Abandoned as **numerically unstable** (`bbd093b`).
   - **The code was deleted 2026-08-11** once it was confirmed unreachable, so this entry
     is the only remaining description of it. Recovery anchors, checked rather than
     assumed: it was **written** in `1a53915` ("Creates a left-based cubic spline"), the
     only commit that ever touched the symbol, and the bodies are intact at `bbd093b`
     via `git show bbd093b:planktos/fluid.py` — note `bbd093b` records the *decision* and
     does not itself modify those functions. If both hashes go stale, use
     `git log --oneline -S "_left_based_cspline" -- planktos/fluid.py`, which finds the
     commit that wrote it and the one that removed it.
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
memory · `int` >= 4 = dynamic windowed linear (`INUM` intervals held at a time).

⚠️ **`INUM` does not have to be odd**, despite what several docstrings said until
2026-08-11. The only constraint the code has ever enforced is `INUM > 3`, and even
values were measured to agree with full-linear to round-off and to slide correctly.
The "odd integer >= 5" language was a leftover from the cubic-window approach of item 3,
where a window symmetric about a center point plausibly needed one; `LinearSpline`
carries no such requirement. Corrected in `_environment.py` and `fluid.py`.

*(Supersedes the older `planktos/TODO for dynamic loading.txt`, folded in here, and the
mvbnd overhaul's `TODO.md`, whose non-blocking follow-ups are merged into the sections
above.)*

---

## Candidates for a patch release (cherry-pick queue) ✅ 1.0.3 RELEASED AND TAGGED

✅ **Ported to `master` and shipped: `v1.0.3` is tagged (2026-08-19, at `a1c2128`).**
Five of the six entries below are in it — the `move()`/`_ibc` work, the `move()` override
guard, the FTLE normalization and the vorticity backdrop flash by commit `bf7112c`
(2026-08-13); the agent-velocity statistics and `reset()` histories by `1c5334c`
(2026-08-19). `master`'s `__version__` is `1.0.3` and its changelog carries a 1.0.3
section; the same section was added here and those lines removed from 1.1.0, since they
are no longer new in it.

⛔ **Not ported, and not waiting on a later patch: the periodic edge-ring fixes**
(`get_vorticity`, `calculate_DuDt`, `calculate_mag_gradient`). They stay on `dyload` by
decision rather than oversight and reach `master` with 1.1.0 — see the ⛔ note in that
entry.

Three things the port turned up, worth knowing before the next one:

- The FTLE fix broke four shear assertions exactly as its entry predicted; they were
  reworked onto full-integration stencils.
- `test_frame_selection_in_the_error_state_plots_history_only` came along with the
  failed-step tests and had to be dropped — it needs `Swarm._select_frames`, which is
  dyload's fps/playback_rate work.
- The `_vorticity_norm` tests had nowhere to live on `master`, so they became
  `tests/test_plot_helpers.py` there. **If `master` is merged here, fold that file into
  `test_frame_selection.py` rather than keeping both.**
- Every ported fix was mutation-checked on `master`. One is not covered *on either
  branch*: the `_ibc` free-flight guard survives being mutated away.

  ⛔ **Investigated 2026-08-13 and deliberately left untested.** With the static guard
  removed, **20,000 randomized trials found no input that reaches it** — half of them
  near-tangent (the case its own comment names), all aimed exactly at a mesh vertex,
  over 1–3 element chains. Degenerate zero-length elements and duplicated vertices do
  not reach it either. That is consistent with the arithmetic: in the 2D static branch
  `t_edge` is `norm(...)/norm(...)`, so it is non-negative by construction and can only
  be 0 on an exact vector cancellation.

  Reaching it at all would mean calling the private `_project_and_slide_static` with
  hand-forged internal state — testing the forgery, and freezing internal structure the
  collision code is explicitly allowed to change. Against that: if the guard were ever
  removed the failure is a **loud** `RecursionError`, and
  `test_exhausting_the_stack_reports_the_cause` already pins that a stack exhaustion
  reports its cause legibly.

  ⚠️ **The moving-mesh guards are the ones to revisit if this ever does fire.** They
  take `t_rot`/`t_edge` from a numerical solver (`sol.x[0]`) rather than from a
  closed-form norm, so 0 or NaN is far more plausible there than in the static case
  tested above. A real case would be worth a test; a manufactured one would not.

**Why this list exists.** A released tag closes to new fixes, so master-applicable work
done here lands under **1.1.0** in `changelog.txt` and `master` does not get it until
1.1.0 ships. This list is what would be cherry-picked if waiting that long ever stops
being acceptable.

✅ **`1.0.3` shipped and is closed.** `master` carries `__version__ = '1.0.3'`, a 1.0.3
changelog section, and the tag `v1.0.3` (2026-08-19, at `a1c2128`). When a line does ship
in a patch section it moves there on both branches and is **not** duplicated under 1.1.0
— which is what happened to the eight lines now under 1.0.3.

**Decided (2026-08-19): the next release is `1.1.0`, and there is no `1.0.4`.** So keep
filing master-applicable fixes under **1.1.0** and logging them below, but treat the
1.0.4 queue as a holding pen rather than a release in preparation: **do not propose
porting anything to `master` unless asked.** The trigger is the list growing long enough
that a patch release becomes worth cutting, and that call is the user's.

Add to this list whenever a fix made on `dyload` is not dyload-specific. The test for
that is simple: does the code it touches look the same on `master`? Check with
`git diff master -- <file>` before assuming.

### Queued for a possible 1.0.4 — one entry

Everything below the next heading is history — it shipped, or was deliberately not
ported — kept for the porting notes rather than because anything is pending. Per the
release plan there is no `1.0.4` in preparation; this is a holding pen.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| **`Swarm.move` raises when the Environment holds more than one Swarm** (2026-08-21), replacing the warn-and-freeze block that appended to `pos_history` alone and left `vel_history` / `props_history` behind. Ships with the docstring rewrite of `update_time`, three tests in `test_swarm_lifecycle.py`, and the `test_agent_models.py` fix for a test that was accidentally stacking four Swarms into one Environment | `planktos/_swarm.py` (`Swarm.move`), `tests/test_swarm_lifecycle.py`, `tests/test_agent_models.py`, `docs/quickstart.rst` | **Yes, by hunk.** The freeze-append block is byte-identical on `master` (`git show master:planktos/_swarm.py`, the `len(s.pos_history) < len(self.pos_history)` guard). ⚠️ It is a **behavior break** — a warning becomes a raise — so it is semver-visible and belongs in a minor release, not a patch |

### Shipped in 1.0.3 — the 2026-08-10 `move()`/`_ibc` work

Both findings are written up in full in the "`_ibc.py` — the 2026-08 collision passes"
section above; that is the reference for *why*, this is the reference for *what to move*.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| **Free-flight termination guards** (finding #2) — three `if not t_edge > 0: return newstartpt` / `if not t_rot > 0:` guards | `planktos/_ibc.py`, in `_project_and_slide_static` (the static free-flight release) and `_project_and_slide_moving` (the rotate-away and free-flight releases) | **Yes.** `git diff master -- planktos/_ibc.py` is *exactly* these 12 added lines, so the file is otherwise identical |
| **Failed-step error handling** (finding #1) — the `try`/`except BaseException` around `apply_boundary_conditions` + `after_move` that appends `time_history` and sets `envir.time = None`, plus the `if self.envir.time is None` check at the top of `move()` | `planktos/_swarm.py`, `Swarm.move` only | **Yes, by hunk.** `_swarm.py` as a whole diverges heavily (~280+/141- vs master, mostly the plotting work), but the two touched regions are byte-identical on `master` — its `move()` has the same `apply_boundary_conditions`/`after_move`/`time_history.append` block and no `try`/`except` of its own |
| **Interrupt handling** (2026-08-11 follow-up) — the same `except` catches `BaseException`, so a Ctrl-C landing in the boundary-condition loop marks the state like an error does, then re-raises as itself rather than being wrapped in `RuntimeError` (an outer `except Exception` must not swallow an interrupt) | `planktos/_swarm.py`, `Swarm.move` only — same hunk as the row above | **Yes**, and it must move *with* the row above: `master` has no `try`/`except` here at all, so the two are one change |
| **Tests** — `test_failed_step_closes_histories_and_marks_the_environment`, `test_interrupted_step_is_marked_and_the_interrupt_propagates`, `test_error_state_blocks_moves_until_it_is_backed_out`, and the `_wall_swarm` / `_fail_on_third_agent` helpers above them | `tests/test_swarm_lifecycle.py` | **Yes.** Both `tests/test_swarm_lifecycle.py` and `tests/_ib_harness.py` exist on `master`, including the `horizontal_wall` and `max_meshpt_dist` builders these use |

**Corresponding `changelog.txt` lines**, now under 1.0.3 on both branches:

```
- A time step that fails or is interrupted partway through now marks the Environment (time=None) and reports the state; move() refuses until it is restored.
- Bug fix: a sliding agent released back into free flight can no longer recurse without limit on a degenerate step.
```

**Note the commit split:** the failed-step work is all in `816e6d7`, but the interrupt
row above is a later follow-up, so cherry-picking `816e6d7` alone leaves `except
Exception` in place and Ctrl-C still corrupting silently. Take both.

### Shipped in 1.0.3 — the `move()` override guard (2026-08-11)

`Swarm.__init_subclass__` warns when a subclass puts a `move` in its own namespace
without appearing to delegate to the base (`super` or `move` in the override's
`co_names`). Replacing `move()` silently drops history recording, boundary
conditions, the velocity/acceleration finite difference and the time advance;
`apply_agent_model` and `after_move` are the extension points. Warns rather than
raises, so an existing subclass that extends-and-delegates keeps importing.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| `__init_subclass__` inserted between the class docstring and `__init__` | `planktos/_swarm.py` | **Yes.** The anchor is byte-identical on `master` and `warnings` is already imported there |
| **Tests** — the five in the "guard on overriding move()" section at the end of the file | `tests/test_agent_models.py` | **Yes.** The module and its `_still_envir` helper exist on `master` |

**Corresponding `changelog.txt` line**, now under 1.0.3 on both branches:

```
- A Swarm subclass that replaces move() instead of apply_agent_model now warns at class definition.
```

### Shipped in 1.0.3 — the FTLE normalization fix (2026-08-11)

`calculate_FTLE` divided by `T` even where a stencil point had left the domain early
and the flow map had therefore only been integrated to `t_calc`. Stretching from one
interval, normalized by another's length: it under-reports, always in the same
direction. Measured at **0.34-0.72 of truth** on an analytic field, in a band 3-4 cells
deep around the domain edge — and in a through-flow domain, where tracers leave
continuously, that band is not a rim.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| `elapsed = t_calc - t0` and its guard, plus `log(sqrt(w))/T` → `/elapsed` in both places | `planktos/_environment.py`, `calculate_FTLE` only | **Yes, by hunk.** `_environment.py` has diverged enormously (568+/977- vs master), but this block is byte-identical there — master carries the same defect at its lines 3315 / 3362 / 3366 |
| **Docstring** — the "points whose neighbors leave the domain early" paragraph, and the `last_time` return note pointing at it | same method | **Yes** |
| **Tests** — `test_FTLE_normalizes_by_the_time_actually_integrated` and the `_full_stencil_values` helper | `tests/test_analysis.py` | **Yes** |
| **The four shear assertions**, reworked off `nanmax` onto full-integration stencils | `tests/test_analysis.py` | **Yes, and required.** Master has the same `nanmax(envir.FTLE_largest)` assertions at its lines 115 / 150 / 202. They break under the fix: the shear field's FTLE depends on integration time, so `nanmax` picks up a truncated point with a legitimately *higher* rate |

**Corresponding `changelog.txt` line**, now under 1.0.3 on both branches:

```
- Bug fix: FTLE is now normalized by the time actually integrated, correcting values where neighboring tracers left the domain early.
```

### Not ported, by decision — the periodic vorticity edge ring (2026-08-13)

`get_vorticity` never consulted `periodic_dim`. A periodic axis carries a duplicated end
line — the contract `FluidData` documents — so the field genuinely continues past either
end, but `np.gradient` cannot know that and fell back to a **first-order one-sided**
difference there (its default `edge_order=1`). The outermost ring of every vorticity
plot was therefore wrong, and wrong at a lower order than the interior.

Measured against IB2d's own `Omega` on `tests/data/Rubberband_with_Damped_Springs`:
the edge ring was **5.0–8.4% off** across three dumps while the interior matched
**exactly**. Differencing across the wrap brings the edge to 0.00% as well — i.e. the
recomputed curl then reproduces the solver's stored vorticity everywhere. Reproduce
with `tests/manual/bench_vorticity_sources.py`.

New `fluid._spatial_gradient` adds one ghost point from the far side at each end, calls
`np.gradient`, and trims — which keeps its non-uniform-spacing handling and leaves the
interior bit-identical.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| `_spatial_gradient`, and `get_vorticity` calling it per axis with `periodic_dim[axis]` | `planktos/fluid.py` | ⚠️ **Not as a hunk — needs a port.** Master has no `fluid.py`; the method is `Environment.get_2D_vorticity` (its line 3386), 2D only, and carries the identical two `np.gradient` calls |
| — | — | ⚠️ **And master has no `periodic_dim` state to consult.** It appears there only as an argument passed to `_wrap_flow` at IB2d load time (its line 1071) and to `center_cell_regrid`; there is no `self.periodic_dim`. So the port needs one of: (i) store `periodic_dim` on `Environment` at load time, or (ii) detect it in `get_2D_vorticity` by testing whether the last grid line duplicates the first, which is exactly the documented contract and is self-contained. **(ii) is the smaller patch-release change**; (i) is the right long-term shape and is what `dyload` already has |
| **Tests** — the "2D vorticity on a periodic grid" section of `tests/test_analysis.py` | `tests/test_analysis.py` | **Mostly.** The file exists on master. The 3D case must be dropped (master's is 2D-only), and `_periodic_envir` must set `flow_points` the way master expects |

Rows for the two physics call sites, fixed the same day:

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| `calculate_DuDt` unrolled from one all-axes `np.gradient` into a per-axis `_spatial_gradient` loop, `edge_order=2` preserved | `planktos/fluid.py` | ⚠️ **Port, not a hunk.** Master has no `fluid.py`; the method is `Environment.calculate_DuDt`. Same `periodic_dim` problem as the vorticity row above |
| `calculate_mag_gradient` likewise, plus the `speed` expression lifted out of the two branches so both share one gradient call | `planktos/_environment.py` | **Closer.** The method exists on master with the same body; it still needs a `periodic_dim` source |
| `_spatial_gradient` gained an `edge_order` argument, so the non-periodic path is bit-for-bit what it was | `planktos/fluid.py` | port with the helper |
| **Tests** — the "periodic dimensions: the edge is not a boundary" section | `tests/test_material_derivative.py` | **Yes**, the file exists on master |

⛔ **NOT ported to 1.0.3 — by decision, recorded below.** Every other queued entry
shipped in 1.0.3; these three did not, because **`master` has no `periodic_dim` state to
consult.** It exists there only as an argument passed to `_wrap_flow` at IB2d load time
and to `center_cell_regrid`; there is no `self.periodic_dim` anywhere.

⚠️ **An earlier row above suggested detecting periodicity by testing whether the last
grid line duplicates the first. That suggestion was wrong and must not be taken.**
Equality of the end lines does not imply periodicity: `f = sin(pi x / L)` is zero at
both ends and is not periodic, and wrapping it gives a derivative of **0.0 at x=0 where
the truth is pi/L — 100% wrong**, where the current one-sided difference is accurate to
0.1%. Detection would silently corrupt any field that happens to vanish, or be equal,
at both ends.

So the real options are:

1. **Add `periodic_dim` state to `Environment` on `master`**, as `dyload` has. Correct,
   but it is a new user-facing parameter in a patch release — arguably a feature.
2. **Leave these dyload-only** until 1.1.0 carries `periodic_dim` across anyway.
3. Port with an explicit opt-in argument on the three methods, which nobody would set.

**Decided (2026-08-13): (2) — leave them on `dyload`.** They arrive on `master` with
1.1.0, which carries `periodic_dim` across anyway. The magnitude supports it: `get_vorticity`'s edge ring was
first-order and genuinely wrong (5-8% against IB2d's own `Omega`); that is the one worth
wanting on `master`. But `calculate_DuDt` and `calculate_mag_gradient` already used
`edge_order=2`, converge at order 2.00 either way, and improve only from 1.72-2.19x the
interior error to 1.08-1.25x — a uniformity fix, not a wrong-answer fix, and thin
justification for adding state to a patch release.


**The same defect in `calculate_DuDt` and `calculate_mag_gradient` is now fixed too**
*(2026-08-13)*, and queued below. It was held back one commit because those feed the
physics rather than a plot; the call was that **correctness outranks reproducing
previous output** for a research code, now written into `CLAUDE.md`.

⚠️ **The magnitude is much smaller than the vorticity case, and the difference is worth
understanding before assuming otherwise.** Both of these already passed `edge_order=2`,
so their edge was a *second-order* one-sided difference — converging at the same rate as
the interior, just with a larger constant. `get_vorticity` took numpy's default of 1 and
was therefore first-order at the edge, which is why it disagreed with IB2d's own `Omega`
by 5–8% there. Measured on an analytic periodic field, all four affected components:

| | edge error vs interior |
|---|---|
| one-sided (before) | 1.72–2.19× |
| across the wrap (after) | 1.08–1.25× |

and a convergence check confirms **order 2.00 either way** — so this is a uniformity and
accuracy fix, not a wrong-answer fix. Results in the outermost ring of a periodic domain
shift by roughly a factor of two in their error, which reaches agents through
`Swarm.get_DuDt` (the inertial-particle models) and `get_fluid_mag_gradient` (behavior
code).

### Shipped in 1.0.3 — the vorticity backdrop flash (2026-08-12)

`plot_all` called `ScalarMappable.autoscale()` on every animation frame, which sets the
colour limits to that frame's own min/max. Two consequences, both visible: RdBu is a
diverging map with white at its midpoint, so asymmetric limits put zero somewhere other
than white and tinted the whole quiescent background — and since the limits changed
every frame, the tint flickered through the video. It also silently discarded any
`clip` the caller passed, so the documented way to stabilize a movie did not work.

Fixed by `_vorticity_norm`: limits symmetric about zero, grown across a movie but never
shrunk, and left alone entirely when `clip` is given.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| `_vorticity_norm` helper, and the four call sites using it (`Swarm.plot`, the `plot_all` movie setup, and both movie update branches) | `planktos/_swarm.py` | **Yes, by hunk.** `_swarm.py` diverges heavily overall (347+/141- vs master), but master carries the same defect — `git show master:planktos/_swarm.py \| grep autoscale` finds the same two calls, at its lines 2709 and 2997 |
| **Tests** — the `_vorticity_norm` section of `tests/test_frame_selection.py` | `tests/test_frame_selection.py` | ⚠️ **Not the module.** `test_frame_selection.py` was added for the frame-selection work (`run_persistence.md` §4.1) and does not exist on master. Port the seven tests into a plotting test module there, or add the file carrying only this section |

### Shipped in 1.0.3 — agent-velocity statistics and `reset()` histories (2026-08-19)

✅ **Ported to `master` in commit `1c5334c` (2026-08-19)**, into its 1.0.3 section, and
released in `v1.0.3`. Kept here as the record of what moved and how.

Two latent defects found while reframing the plotting plan into
`docs/notes/run_persistence.md`; §5 there carries the full analysis. Both predate this
branch and both are on `master`.

**1. `_calc_basic_stats` finite-differenced `pos_history`** rather than reading
`vel_history`. `move()` sets `velocities` from *pre*-boundary-condition positions and
`apply_boundary_conditions` then mutates positions, so the two part company for any agent
that hit an immersed boundary or the domain edge. Across a **periodic wrap** the position
difference is nearly the whole domain, so the reported agent speed spikes to a fictitious
value — 9 against a true 1 in the regression test. On `dyload` this also fed the mean
agent speed and its spread, which the run-persistence §3.1 work added on top of the same
quantity.

A second, deliberate effect: `t_indx=0` no longer reports the zero vector. `Swarm.__init__`
sets the initial velocities to the local fluid drift, so the first frame's statistics show
that drift. Flow-free runs are unchanged, their initial velocities being zeros.

**2. `Environment.reset()` cleared `pos_history` only,** leaving `vel_history` and
`props_history` behind and permanently misaligned with it — which reaches the plotted
heading markers, and `_calc_basic_stats` once fix 1 lands. The FTLE stencil copy in the
same file already gets this right, which is what makes `reset()` look like an oversight
rather than intent.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| **`reset()` clearing all three histories** | `planktos/_environment.py`, `Environment.reset` only | **Yes.** Byte-identical on `master` (its line 3670) |
| **Tests** — `test_reset_clears_every_history_not_just_positions`, `test_reset_leaves_props_history_off_when_it_was_never_on` | `tests/test_swarm_lifecycle.py` | **Yes.** The module and its `_envir` helper exist on `master` |
| **The `_calc_basic_stats` velocity read**, plus the docstring paragraphs explaining it | `planktos/_swarm.py`, `_calc_basic_stats` only | ⚠️ **Port, not a hunk.** `master`'s version returns only `avg_swrm_vel` — the mean speed and its spread are dyload additions — and its branch structure differs, computing `avg_swrm_vel` separately inside each branch. The *defect* is the same two lines; the code around them is not |
| **Tests** — `test_calc_basic_stats_agent_velocity_at_initial_time_is_the_recorded_drift`, `..._agent_speed_at_initial_time_is_zero_without_flow`, `..._velocity_survives_a_periodic_wrap` | `tests/test_flow_interface.py` | ⚠️ **Not the module.** `test_flow_interface.py` does not exist on `master` — it was written for the `FlowArray` removal. Port the three into a statistics/plotting module there, dropping the mean-speed and spread assertions, which pin dyload-only return values |

**Corresponding `changelog.txt` lines**, filed under **1.0.3** on both branches and
shipped there:

```
- Bug fix: plotted agent velocity statistics use recorded velocities instead of differenced positions, which were wrong after any collision or periodic wrap.
- Bug fix: Environment.reset clears velocity and props history alongside position history, which it left behind and misaligned.
- Plot statistics at the initial time now show the agents' starting fluid drift instead of zero.
```

### ⛔ Explicitly NOT cherry-pickable — dyload-only

- **The `_select_frames` error-state branch** and its test
  (`test_frame_selection_in_the_error_state_plots_history_only`), plus the changelog
  line `- Plotting after a failed time step shows the recorded history and leaves the
  incomplete step out.` **`Swarm._select_frames` does not exist on `master`** — it
  arrived with the `fps`/`playback_rate` work (`run_persistence.md` §4.1). On `master`, `plot_all`
  still expands `frames=None` to one frame per recorded state and never builds a time
  axis, so there is nothing there to guard and no equivalent failure. This line stays
  under 1.1.0 only.

### How to find these later

**All of it landed in one commit: `816e6d7`** ("Make a failed time step legible instead
of silently corrupting the run"). That commit also carries the dyload-only
`_select_frames` change and these notes, so it cannot be cherry-picked wholesale —
take the hunks listed in the table above.

Backup anchors if the hash goes stale (a rebase, say):

- `git log --oneline -S "envir.time = None" -- planktos/_swarm.py`
- `git log --oneline -S "not t_edge > 0" -- planktos/_ibc.py`
- the three `changelog.txt` lines quoted above.
