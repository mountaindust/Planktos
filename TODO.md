# TODO — `dyload` branch (dynamic loading of fluid data)

**Goal of this branch:** load/spline time-dependent fluid velocity data *on demand*
(a sliding window of timesteps) instead of holding the whole dataset in memory, so
that large 3D time-varying flows (~100 GB raw, larger once splined) can be used.

**Current state (2026-08-10).** The architecture is built and the API has settled —
all fluid data is a `FluidData` object (`planktos/fluid.py`). Dynamic windowed
loading works and is tested in **both** 2D and 3D against committed fixtures.
Temporal interpolation of dynamically-loaded data is **linear in time**
(`LinearSpline`); full-dataset loading defaults to **cubic in time**
(`fCubicSpline`). See the design-history section at the bottom for the cubic→linear
story.

**Suite is green: 551 passed / 22 skipped (`pytest`), 571 passed / 2 skipped
(`pytest --runslow`).** No failures, no xfails.

**What is done:**

- **Phase 0** (adapt the suite to the `FluidData` API, fix what it surfaced) — complete.
- **Phase 1** (2D dynamic loading) — complete, including **(C)**, the
  linear-vs-cubic error measurement, which is answered with numbers. Only
  periodic × dynamic testing is left over, folded into Phase 2.
- **Flow-field interface refactor** (`docs/notes/flow_field_interface.md`) — **§7
  complete** (`FlowArray` deleted, tiling gated off, `test_flow_interface.py` pinning
  the consumer contract) and **§8 steps 1–2 complete** (plot statistics served from a
  per-dump mean cache; `fps`/`playback_rate` frame selection).
- **`_ibc.py` collision passes** — done on `master` and merged here; coverage 91% → 99%.

**Where the work goes next, in priority order:**

1. **Phase 2 — 3D dynamic loading against the real OpenFOAM dataset.** This is the
   branch's remaining reason to exist and it blocks 3D moving boundaries. The
   dataset is staged; the task is a **loader**, not data prep. See Phase 2.
2. **Note §8 steps 3–4** (recorder + derived-quantity cache) — explicitly **due a
   re-evaluation before being built**, since steps 1–2 removed most of the cost they
   were designed for. Step 5 (prose pass) follows whatever is decided.
3. **Note §9** (real position-wrapping tiling, 2D and 3D; whether `Environment.extend`
   returns) — still needs its design pass. §9.1 is the restoration checklist.

**Also merged since:** `master`'s 1.0.1 documentation release and its 1.0.2 bug
fixes. No dyload-specific behavior changed by either. **`v1.0.2` is released and
tagged**, so master-applicable fixes made here now queue for a possible **1.0.3** —
see the cherry-pick queue at the bottom of this file.

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
| `FlowArray` returned stale data from any derived array | `fluid.py` | resolved by **deleting** `FlowArray` (note §7.3) |
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

- [ ] 🟢 **Orphaned discarded code:** `fCubicSpline._left_based_cspline` /
  `_extend_prev_spline` (`fluid.py:581-763`) — the abandoned cubic-window approach,
  now unreachable (the only `fCubicSpline(...)` caller uses default `bc_type`). Remove
  or annotate as "abandoned — see history."

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
boundaries are **2D-only** while this branch's whole purpose is unblocking 3D. Revisit
with Phase 3 (3D moving boundaries), where the moving slider gets rewritten for
triangles anyway and this decision has to be made again from scratch.

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

## Phase 2 — Dynamic loading in 3D against real data 🔴 (blocks 3D moving boundaries)

The actual end goal (the ~100 GB case) and **the branch's remaining reason to exist.**

### The dataset is staged and analyzed ✅

`tests/unsteady_3D_testdata/` (gitignored as a whole directory). Contents: `VTK/` with
21 timestep `.vtm`s + a `.vtm.series` (case `case08_alpha2_1e8`), `oral_arm_disk.stl`,
and `README_oral_arm_setup.md` with the full setup spec.

**Physics:** OpenFOAM, Cassiopea oral-arm porous disk. Water (ρ=1000, ν=1e-6), Re=500,
laminar transient. Pulsing annular inlet, u_z(t) = 0.01·½(1−cos(2π·0.8·t)) m/s. Export
covers the last two pulse cycles (t ≥ 7.5 s, period 1.25 s). Domain x,y ∈ [−0.05, 0.05],
z ∈ [0.003, 0.271] m, **all lengths in meters**. Fields: `U`, `p` (kinematic),
`vorticity`, `Q`. `oral_arm_disk.stl` is a ready-made 3D immersed boundary for Phase 3.

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

### The real work: a loader for this format 🔴

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

**Step 2 — `OpenFOAMData(FluidData)` in `fluid.py`** — the permutation, the boundary
splice, the gap policy, and `flow_times`. Not started; everything unchecked below is
its work.

Four things make it more than a format swap (all detailed in the note):

- [x] **Fields are `CELL_DATA`, not point data** — OpenFOAM is finite-volume, and
  `point_data` is empty. Handled by the new `_dataio.read_vtkxml_cell_data`, which
  reads `GetCellData()` and returns `vtkCellCenters` output rather than `GetPoints()`
  (hexahedron **corners**, the wrong lattice). Written as a **new** function rather
  than an edit to `read_vtu_Unstructured_Grid_Points_FEM` as originally planned —
  that one is live for `ComsolVTUData`, which wants point data on FEM corner points.
- [ ] **Cell ordering is not lexicographic.** A bare `.reshape((66,66,178))` silently
  scrambles the field. A permutation index is required — built by snapping coordinates
  to an integer lattice, **not** by `lexsort` on raw floats, which fails on float32
  roundoff. Computed **once**, since the mesh is static.
  - ⚠️ **`np.unique` is unusable here too, not just `lexsort`** — measured through the
    new reader: the float64 cell centers show **79 distinct y-levels where 66 exist**.
    Snapping recovers 66/66/178 exactly (max deviation 8.5e-7 of a cell in x/y, 8.0e-6
    in z). Note §2.
- [ ] **Cell centers are inset half a cell**, so raw centers report the domain as
  0.0985 × 0.0985 × 0.2665 instead of the true 0.1 × 0.1 × 0.268. The `.vtp` boundary
  patches close this **exactly**: `inlet`/`outlet` carry real data on the identical x/y
  lattice, `walls` is exactly zero (no-slip) and can simply be padded. Assembly is
  66×66×178 → 66×66×180 → 68×68×180. The 12 edges and 8 corners appear in no file but
  lie on no-slip walls, so filling zeros there is exact.
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

- [ ] 🔴 **The loader must tolerate missing *metadata* files, not just missing dumps.**
  Fluid data arrives from collaborators through a pipeline nobody remembers choosing
  last time, so which index/manifest layer is present varies between deliveries. The
  three `_dataio` readers were deliberately built as independent functions so this
  degradation can be a chain of fallbacks in `FluidData` rather than a rewrite:

  | Missing | Fallback |
  |---|---|
  | `.vtm.series` | glob the `.vtm` files; take each time from its `TimeValue` |
  | `.vtm` manifests too | glob the dump directories; read `TimeValue` from `internal.vtu` |
  | all time info | warn, assume unit steps (the `VTK3dData._read_all_times` precedent) |
  | boundary `.vtp` patches | interior only, and say so — the domain is then a half cell short in every direction, which is the failure the `center_cell_regrid` docstring describes |

  Sort order is the trap in the glob paths: dump directories are named with unpadded
  numbers (`case08_alpha2_1e8_787`, `..._1008`), so a lexical sort puts 1008 before
  787. Parse the number and sort numerically. Warn about **which** fallback was taken —
  silently accepting a degraded timeline is the `VTK3dData` frozen-fluid bug's shape.
- [ ] 🔴 **The loader must tolerate a dump series with holes — warn, do not fail.**
  4 of 21 timesteps in this dataset are missing (t = 9.0, 9.375, 9.75, 10.0): the
  `.vtm` manifests exist but the directories they name do not, a truncated transfer.
  17 are present, confirmed 2026-08-10. **Decision: work with the data as-is** rather
  than waiting on a re-send, and treat gap tolerance as a loader requirement in its
  own right — truncated or interrupted exports are normal in practice.

  Requirements:
  - **Resolve which dumps actually exist eagerly, at construction**, when the
    timeline is built. Do *not* discover a missing file at the window slide that
    needs it. Under dynamic loading that raise could land hours into a long run,
    which is the worst possible time for it and exactly what streaming makes likely.
  - **Warn once**, naming the missing times and how many were dropped. Silence here
    would be worse than the failure: the run completes and nothing indicates the
    timeline has holes.
  - **Build `flow_times` and the dump index over the present dumps only**, densely.
    Then `d_start`/`d_finish` index the surviving series, `load_dumpfiles` is never
    handed a filename that is not there, and everything downstream is unchanged. The
    only visible effect is a wider interval between two adjacent entries.
  - **Warn separately about the resulting non-uniform spacing.** Interpolation error
    scales with the dump interval (Phase 1 (C)), so a 0.125 s series with a 0.5 s hole
    is measurably worse across that hole and the user should know where.
  - ⚠️ Interacts with the `flow_times`-must-span-the-whole-series guard below: the
    "whole series" is the set of dumps that exist, not the set the manifest declares.

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

### Then, once it loads

- [ ] Re-run the Phase 1 (A)–(D) equivalents on real 3D data.
- [ ] **Periodic × dynamic** — carried over from Phase 1; still untested in either
  dimension. Note this dataset has walls, not periodic sides, so it may not be the
  vehicle for it.
- [ ] Re-check the Phase 1 (C) interpolation-error numbers on 3D data and update
  `docs/api/FluidData.rst`, which currently carries 2D figures with that caveat stated.
- [ ] 3D material derivative end-to-end for massive / inertial particle models.
- [ ] **Memory profiling:** confirm RAM stays bounded to one window across a long 3D run.
  This is the branch's headline claim and is still unmeasured on real data.
- [ ] Un-skip / fix the IBAMR load tests on real data.

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
> priority than 3D moving boundaries and is scratched here.
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

---

## Candidates for a 1.0.3 patch release (cherry-pick queue) 🟡

**Why this list exists.** `v1.0.2` is **released and tagged**, so master-applicable
fixes made here can no longer be filed under it. They currently land under **1.1.0** in
`changelog.txt`, which means `master` does not get them until 1.1.0 ships. If enough
accumulates first, cut a **1.0.3** from `master` and cherry-pick the entries below.

Add to this list whenever a fix made on `dyload` is not dyload-specific. The test for
that is simple: does the code it touches look the same on `master`? Check with
`git diff master -- <file>` before assuming.

### Queued — the 2026-08-10 `move()`/`_ibc` work

Both findings are written up in full in the "`_ibc.py` — the 2026-08 collision passes"
section above; that is the reference for *why*, this is the reference for *what to move*.

| What | Where | Applies cleanly to `master`? |
|---|---|---|
| **Free-flight termination guards** (finding #2) — three `if not t_edge > 0: return newstartpt` / `if not t_rot > 0:` guards | `planktos/_ibc.py`, in `_project_and_slide_static` (the static free-flight release) and `_project_and_slide_moving` (the rotate-away and free-flight releases) | **Yes.** `git diff master -- planktos/_ibc.py` is *exactly* these 12 added lines, so the file is otherwise identical |
| **Failed-step error handling** (finding #1) — the `try`/`except` around `apply_boundary_conditions` + `after_move` that appends `time_history` and sets `envir.time = None`, plus the `if self.envir.time is None` check at the top of `move()` | `planktos/_swarm.py`, `Swarm.move` only | **Yes, by hunk.** `_swarm.py` as a whole diverges heavily (~280+/141- vs master, mostly the §8 plotting work), but the two touched regions are byte-identical on `master` — its `move()` has the same `apply_boundary_conditions`/`after_move`/`time_history.append` block |
| **Tests** — `test_failed_step_closes_histories_and_marks_the_environment`, `test_error_state_blocks_moves_until_it_is_backed_out`, and the `_wall_swarm` / `_fail_on_third_agent` helpers above them | `tests/test_swarm_lifecycle.py` | **Yes.** Both `tests/test_swarm_lifecycle.py` and `tests/_ib_harness.py` exist on `master`, including the `horizontal_wall` and `max_meshpt_dist` builders these use |

**Corresponding `changelog.txt` lines**, currently under 1.1.0 — move these two to a
1.0.3 section on `master` (they stay under 1.1.0 here as well, since 1.1.0 ships them
too if no 1.0.3 happens first):

```
- A time step that fails partway through now marks the Environment (time=None) and reports the state; move() refuses until it is restored.
- Bug fix: a sliding agent released back into free flight can no longer recurse without limit on a degenerate step.
```

### ⛔ Explicitly NOT cherry-pickable — dyload-only

- **The `_select_frames` error-state branch** and its test
  (`test_frame_selection_in_the_error_state_plots_history_only`), plus the changelog
  line `- Plotting after a failed time step shows the recorded history and leaves the
  incomplete step out.` **`Swarm._select_frames` does not exist on `master`** — it
  arrived with the `fps`/`playback_rate` work (note §8 step 2). On `master`, `plot_all`
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
