# CLAUDE.md

Guidance for working in the Planktos repository. Keep this file current; it is
loaded into every session.

## Git policy (strict — this has repeatedly been mishandled; read carefully)

**Commits and pushes require explicit, per-action authorization in the user's
most recent message. Authorization NEVER carries forward.**

- **Never `git commit` or `git push` automatically or on your own initiative.**
- **Each commit needs its own fresh green light.** A commit request authorizes
  exactly ONE commit, right then. The next commit requires a new, explicit
  request. Past authorizations do not propagate forward — not across turns, and
  not within a single multi-step task.
- **"Do X, then commit, then do Y" authorizes committing X only.** It does NOT
  authorize committing Y or any later/again work. Map each commit authorization
  to the one specific step it was attached to, and nothing else.
- **Do not move toward a commit without authorization.** Running `git add`/staging
  as a prelude to an unauthorized commit counts as moving toward it — don't do it.
  (Read-only git — `status`, `diff`, `log` — is always fine.)
- **A request to commit is NOT a request to push.** Push only when the user
  explicitly and separately asks to push, each time.
- **When in doubt, do not commit.** Show the diff, summarize, and ask. Leaving
  changes uncommitted in the working tree is always the safe default.

## Versioning & changelog (the user wants active reminders here)

The user has explicitly asked for help **remembering to maintain the version
number and the changelog** — these are easy to forget. Be proactive about it:

- The version lives in `planktos/__init__.py` (`__version__`); `setup.cfg` reads
  it via `attr: planktos.__version__`, and `docs/conf.py` imports it — so the one
  string in `__init__.py` is the single source of truth. The current development
  version on this branch is `1.1.0`.
- **`v1.0.0`, `v1.0.1` (documentation-only), `v1.0.2` and `v1.0.3` are all
  released and tagged.** `master`'s `__version__` is `1.0.3` — i.e. it currently
  sits *on* a released version, so the next change landing there needs a bump
  first. The existing `1.0.2` and `1.0.3` entries in `changelog.txt` are shipped
  history: leave them alone, on both branches.
- **A fix made here that is not dyload-specific has nowhere to go but `1.1.0`,**
  since `1.0.3` is closed. File it under `1.1.0` in `changelog.txt` *and* add it
  to the **cherry-pick queue at the bottom of `TODO.md`**. Check portability with
  `git diff master -- <file>` before assuming it applies — `_swarm.py` in
  particular has diverged a long way.
- **The plan is that the next release is `1.1.0`, with no `1.0.4`.** That queue is
  a holding pen in case the plan changes, not a release in preparation: log fixes
  there, but **do not propose cherry-picking them to `master`.** The user will ask
  if the list ever grows long enough to be worth a patch release.
- `changelog.txt` is hand-maintained, terse, and grouped by version. When a
  change is user-facing, prompt to add an entry under the appropriate version.
- When work looks release-worthy (or a user-facing change lands) but the version
  or changelog hasn't been touched, **say so** and confirm the right action.
- Do NOT bump the version or rewrite the changelog silently — surface the need
  and let the user decide (a version bump is a semver judgment call).

**Changelog style (strict):**

- **Length:** each entry ≤ 180 characters, and ≤ 100 in most cases.
- **Say what changed, not the story.** No details about *what happened* and no
  *how the bug was found*. State the user-facing effect only.
- **Skip regression fixes entirely.** If it worked on `master`, we broke it in
  dev, and we fixed it back, it does NOT go in the changelog.
- **Lump minor edge-case fixes.** Small bug fixes for edge cases we only found by
  running a test get collapsed into a single line: `- other minor bug fixes`.
  Do not give them individual entries.

## Docstring and comment style (strict)

**A docstring is a user-facing overview.** It says what the thing does and how to
use it, briefly. Design reasoning, revision history and implementation detail are
not user-facing and do not belong there — unless a user genuinely needs them to
understand what is in front of them (an ODE solver naming the methods it supports;
`FluidData.get_stored_vorticity` saying it is linear-splining only).

- **Be brief.** Summary line, then parameters and returns.
- **Never cite `docs/notes/` or `TODO.md`.** Those are working plans and are
  deletable once their work is done and vetted, so a docstring pointing at one
  rots. *Exception:* notes written to hold mathematics (e.g.
  `project_and_slide_moving.md`) may be cited for a derivation.
- **No bug-fix narrative, no "this used to…"**, and no catalogue of approaches
  tried and discarded.
- **No implementation considerations** — cost measurements, why a branch is
  ordered as it is, what a guard protects against.
- **No philosophy.** Value judgments about the design belong nowhere in the
  source; state the behavior and stop. `Environment.record`'s `path` entry is the
  standing example: *"overwriting a previous run is never the right default, and
  refusing outright would strand a job that was ready to start"* is two sentences
  arguing for a decision the caller cannot change. The caller needs one fact —
  a non-empty directory redirects to a timestamped sibling, and `.path` says
  where the data went.

**The code under the docstring is where design reasoning goes**, as ordinary `#`
comments, and it is meant to be read. Still no history there either.

**Comments say what the code does and how it works — not what it does not do, or
what would not have worked.** Point a code-level reader at the answer, not at the
non-answers.

> ✅ `# NaN marks a dump the run never loaded, so reduce with nanmax.`
> ❌ `# A zero would be indistinguishable from a still fluid and would drag the
>    scale down.`

If the worry behind a long explanation is regression, that is what the tests are
for. Test files are a partial exception to all of this: a regression test may state
the defect it pins, since that is the test's purpose.

⚠️ **Much of the existing tree does not follow this yet** — many methods were
written by past sessions and run long. A sweep is queued as item 3 in `TODO.md`;
`Swarm._calc_basic_stats` is the worked example and is deliberately left unfixed
until then. Do not add more.

## What Planktos is

Planktos is an **agent-based modeling framework** for simulating the movement and
dispersal of small organisms (plankton, tiny insects, etc.) in 2D or 3D fluid
environments. The defining assumption is that agents are small enough that their
effect on the surrounding fluid is **negligible** — fluid drives agents, agents
do not drive fluid. It is an active research project (NSF DMS-2410988, 2024–2027).

Primary uses: studying collective/emergent behavior, dispersal, and interaction
with immersed structures (e.g. flow around a cylinder, a jellyfish, a seafan).

Cite: Strickland, Battista, Hamlet, Miller (2022), *Bulletin of Mathematical
Biology* 84(72). Docs: https://planktos.readthedocs.io

## Branch context (read this first)

- **`dyload`** — **current development branch.** The main feature work: **dynamic
  loading of fluid data** (streaming/loading fluid time steps on demand rather
  than all at once). This matters because time-dependent 3D fluid data is often
  ~100 GB raw and significantly larger once splined, so it cannot be held in
  memory all at once. **You are usually here.**
- **`master`** — stable/published; carries the released `1.0.x` line. Small,
  self-contained changes (docs, typos, packaging) are sometimes made here
  directly and then merged into `dyload`.
- **`mvbnd`** — **gone.** This was the 2D **moving immersed boundaries** work; it
  was merged into `master`, released as tag `v1.0.0`, and the branch has since
  been deleted locally and on origin. Older notes and commit messages still
  mention it; that history is preserved in `master`.

**3D moving boundaries are parked and not scheduled.** They were blocked on `dyload`
(3D dynamic fluid loading), which is no longer the case — that work completed
2026-08-11 — but nothing about them is being picked up. `TODO.md` keeps the list under
a PARKED heading purely so the pieces are not rediscovered from scratch later; do not
start work from it. Moving boundaries are currently **2D only**.

**3D fluid data sources (terse, for future reference).** The 3D flows come from
either **IBFE** (fluid solved over the *entire* domain, including inside immersed
objects, as SAMRAI files on an adaptive rectangular mesh) or **OpenFOAM**. Planktos
**assumes a rectilinear fluid grid** — that assumption stands, and is the thing to
check about any new dataset.

What it does *not* mean is that the data must arrive as rectilinear vtk. **The Phase 2
OpenFOAM dataset is stored as `vtkUnstructuredGrid` yet sits on a perfectly uniform
Cartesian grid** — `foamToVTK` writes `.vtu` regardless of the underlying mesh, so the
container says nothing about the geometry. It needs **no resampling**, only a reordering
permutation (cell ordering is not lexicographic) and a half-cell boundary splice. Fluid
there is defined *everywhere*, including inside the immersed object, which carries a
porosity source term rather than being cut out of the mesh. Verify a dataset's actual
lattice before assuming it needs a VisIt/ParaView pass; see
`docs/notes/openfoam_oral_arm_dataset.md`.

Reading that dataset **is** in scope for `dyload` — it is Phase 2, and it blocks 3D
moving boundaries. What stays out of scope is *other* source-specific ingestion:
porting the old VisIt SAMRAI→vtk script, reading COMSOL directly, and OpenFOAM cases
whose mesh is genuinely unstructured (`snappyHexMesh` refinement, tets, polyhedra).

**`TODO.md` is the working plan for this branch** — phased, prioritized, and kept
current. Read it before starting work here: it records what is done, what is
known-broken, and the design history behind the fluid architecture. Keep it
updated as items land.

**How `dyload` diverges from `master`.** Almost all of it is the fluid API: on
this branch `Environment.flow` is a `FluidData` object, and a number of
fluid-related `Environment` methods were renamed, moved onto `FluidData`, or
removed outright (see "Fluid data architecture"). The practical hazard is that
documentation and tests written against `master` **merge cleanly and are still
wrong** — this has already happened once. When merging from `master`, grep the
incoming text for fluid API names and check each against the source rather than
trusting a conflict-free merge.

When making cross-cutting changes (like this CLAUDE.md), expect them to be
merged from `master` into `dyload` later.

## Package layout

The installable package is `planktos/`. Public API is intentionally tiny;
internal modules carry a **leading underscore** and are not part of the public
surface (a deliberate convention — see `changelog.txt`).

(Line counts are deliberately omitted — they went stale faster than they were
useful. `_environment.py` and `_swarm.py` are the big ones, with `fluid.py` close
behind since the dynamic-loading work.)

| File | Public? | Purpose |
|------|---------|---------|
| `planktos/__init__.py` | yes | Exports `Environment`, `Swarm`. `motion` is reachable as `planktos.motion`. |
| `planktos/_environment.py` | `Environment` class | The domain: boundary conditions, immersed boundary mesh, swarms, time. Loads fluid/mesh data, generates analytical flows, plots, computes vorticity/FTLE. Fluid data itself lives in `fluid.py` (see below). |
| `planktos/_swarm.py` | `Swarm` class | A group of agents: positions/velocities/props, the move loop, boundary-condition application, plotting, data saving. |
| `planktos/motion.py` | yes (`planktos.motion`) | Equation-of-motion generators & solvers: `Euler_brownian_motion` (default SDE), `inertial_particles`, `highRe_massive_drift`, `tracer_particles`, `RK45`. |
| `planktos/fluid.py` | `FluidData` is user-visible via `Environment.flow`; the rest internal | All fluid velocity data and its temporal interpolation: `FluidData` (+ per-source `IB2dData`, `VTK3dData`, `ComsolVTUData`), `LinearSpline`, `fCubicSpline`, `SplineRangeError`. See "Fluid data architecture" below. |
| `planktos/_geom.py` | internal | Pure geometry workhorses: segment/line/triangle intersections, closest distances, multilinear-polynomial intersection (for moving meshes). Formerly static methods of `Swarm`. |
| `planktos/_ibc.py` | internal | Immersed-boundary collision handling: `apply_internal_static_BC`, `apply_internal_moving_BC`, and the project-and-slide routines for static and moving meshes. |
| `planktos/_dataio.py` | internal | Low-level read/write of vtk, vtu, .vertex, stl, NetCDF. Use `Environment` loader methods instead of calling these directly. |
| `planktos/archive.py` | `RunArchive` and `load_run` are public; the writer, the recorder and the format machinery are internal | The run-archive on-disk format: chunked, append-only, crash-valid capture of agent state written as a run proceeds, plus the per-dump fluid quantities a later plot needs. Streams agents *out* the way `fluid.py` streams the field *in*. See `docs/notes/run_persistence.md` §2 and §3. |
| `planktos/_frames.py` | internal | Where a plot frame's data comes from. `FrameSource` settles "what states exist, and where does the fluid backdrop come from" once, so `Swarm.plot`/`plot_all` index it rather than branching on it. Agent state is always live history; the backdrop comes from the Environment's run archive when it has one. |
| `planktos/_provenance.py` | internal | Records what produced an `Environment`'s fluid and ibmesh: every loader and analytic flow generator is decorated so it logs its own call into `_fluid_provenance` / `_ibmesh_provenance`. `jsonable()` is the JSON-safety guarantee the run archive's metadata relies on. |

## Core mental model

1. **`Environment`** is the world: domain size `L`, boundary conditions `bndry`,
   a fluid velocity field `flow`, an immersed boundary mesh `ibmesh`, and a list
   of `swarms`. It owns the simulation `time` and `time_history`.
2. **`Swarm`** is a vectorized group of agents (NOT individual objects — agents
   are rows in numpy arrays for speed). It belongs to one `Environment`.
3. `positions` (and `velocities`, `accelerations`) are **masked arrays** of shape
   `Nx2`/`Nx3`. **A masked row = that agent has left the domain** and is no
   longer updated. Respect/preserve the mask.
4. **`Environment.flow` is a `fluid.FluidData` object**, not a list of ndarrays.
   Index it (`envir.flow[0]`) to get a spatial component. Spatial interpolation
   is **linear**; interpolation in time is **cubic when the whole dataset is in
   memory and linear when dynamically loading**. See the next section — this is
   the biggest single difference from `master`.

## Fluid data architecture (the heart of this branch)

`Environment.flow` is a **`fluid.FluidData` instance**. This is the central change
on `dyload` and the thing most likely to trip up code, docs, or tests written
against `master`.

- `FluidData` owns the velocity field, the spatial grid (`flow_points`), the time
  stamps (`flow_times`), periodicity (`periodic_dim`), and the temporal
  interpolation. Per-source subclasses handle ingestion: `IB2dData`, `VTK3dData`,
  `ComsolVTUData`.
- Fluid-level operations live on the object, not on `Environment`: `tile_flow`,
  `get_vorticity`, `get_dudt`, `calculate_DuDt`, `get_mean_velocity`,
  `update_spline`, `load_dumpfiles`.
- **`get_mean_velocity(time=|t_idx=)` reads a cache, not the field.** The spatial
  mean of each component is recorded per dump as that dump loads, and evaluated
  with the same interpolation weights the field uses — exact, because the mean is
  linear and both splines are weighted sums of nodal fields. This is what lets
  plotting print fluid statistics without re-streaming the dataset (see
  `docs/notes/run_persistence.md` §3.1). A mean stays valid after the
  sliding window has moved past its dump; only a never-loaded dump costs a load.
- `FluidData.get_raw_loaded_data()` is the nearest thing to the old
  `Environment.regenerate_flow_data()`. Note *loaded*: under dynamic loading only
  the current window exists.
- **`FluidData.is_windowed` is the regime discriminator**, not `INUM` itself. It is
  True only when a sliding window is actually in use — False for time-invariant
  flow, for `INUM=None`, for `INUM=True`, and for an int `INUM` that spans the
  dataset (which holds everything and never slides). Anything deciding "is the
  whole field resident?" must ask this rather than testing `INUM is None`.
- **`add_dump_observer(fn)` fires `fn(idx_start, flow)` wherever fluid data lands in
  memory**, dispatched from `_dumps_arrived` — the one method called at all four load
  sites, which caches the per-dump means and then fans out. An observer must be
  **idempotent** (the jump-to-start slide re-reports dumps already seen) and must not
  expect to fire for time-invariant flow, which has no dumps to arrive; use
  `iter_resident_dumps()` to take whatever is already in memory, which covers that
  case and never materializes more than one dump at a time.
- **Per-dump vorticity is sourced, not cached**, by regime: `probe_stored_vorticity`,
  `read_dump_vorticity`, `write_dump_vorticity` (a per-source pair) and the generic
  `get_stored_vorticity(time)`, which blends the two bracketing dumps with
  `LinearSpline`'s own weights through a two-slot cache. It **raises** under cubic
  splining, where the weights are global — that regime recomputes from the resident
  field. `docs/notes/run_persistence.md` §3.3 has the reasoning.
- ⚠️ **A single-dump load is ordinary**, and two of the readers behind
  `load_dumpfiles` drop the leading time axis for one. `FluidData._load_dumps` is what
  the slider calls and restores it, so a subclass implements `load_dumpfiles` without
  having to think about it. A single-dump slide happens whenever the dump count is
  `k*(INUM-1)+3` — with `INUM=4`, any of 6, 9, 12, 15, … time points.
- `periodic_dim` is a property of the fluid data and is **independent of the
  agent boundary conditions** in `Environment.bndry`. It defaults to `False`.

**`INUM` controls dynamic loading, and with it the interpolation in time:**

| `INUM` | Held in memory | Interpolation in time |
|--------|----------------|-----------------------|
| `None` (default) | the whole dataset | **cubic** (`fCubicSpline`) |
| `True` | the whole dataset | linear (`LinearSpline`) |
| `int` (< number of intervals) | a sliding window of `INUM`+1 time points | linear (`LinearSpline`) |

**Linear-in-time is a deliberate, permanent tradeoff of dynamic loading — not a
placeholder.** Cubic was tried and abandoned: stitching a cubic spline across a
window that gains data on either side is numerically unstable, and resplining each
window makes derivatives discontinuous at the breakpoints. Linear is
unconditionally stable, trivially window-extensible (carry two raw boundary
values, no derivatives to match), and needs less data held. The design history —
including the specific approaches that failed — is at the bottom of `TODO.md`.

The cost, worth stating plainly because it reaches the physics: smoothness drops
C²→C⁰ (velocity kinks at each timestamp), between-sample accuracy goes
O(Δt⁴)→O(Δt²), and ∂u/∂t becomes a piecewise-constant step function — which feeds
`get_dudt` → the material derivative → the inertial-particle models. Full cubic
stays the default for datasets that fit in memory.

**That gap has been measured** — Phase 1(C), done 2026-08-11; detail in `TODO.md`,
reproducible via `tests/manual/quantify_temporal_interp.py` and
`vet_dynamic_loading_3d.py`. Linear vs cubic rms error: **1.27% vs 0.54% of U_rms** in
2D (Δt=1e-3 s), **9.46% vs 1.13%** in 3D at the coarser cadence reachable there. Quote
the **2D convergence orders** (median point: linear 1.68, cubic 4.75) — the 3D export
is too coarsely sampled to fit one, and the script refuses to print a meaningless
slope. Ensemble agent statistics agree to within 0.35% (3D) / 0.6% (2D).

**Velocity components are plain `np.ndarray`.** Index a `FluidData` (`envir.flow[0]`)
for a static component, or call it (`envir.flow(t)`) for a temporally interpolated
one; either way you get an ordinary array on which every numpy/scipy/matplotlib
operation works normally. There are no interop caveats and no `np.asarray()`
defensive wrapping — if you find such a wrapper, it is a leftover.

This replaced `FlowArray`, an `ndarray` subclass that virtualized tiled flow by
overriding `.shape`/`__getitem__`. It was **deleted** in the fluid-interface
refactor: modern scipy defeats the trick (`RegularGridInterpolator` calls
`np.asarray` on any array-API object, discarding the virtual shape), so the tiled
interpolation path never actually worked, while the subclass corrupted ordinary
numpy operations on flow data. `docs/notes/run_persistence.md` Appendix A is the
surviving record; the full analysis is in the git history of the deleted
`docs/notes/flow_field_interface.md`.

**Domain tiling currently raises `NotImplementedError`** (`FluidData.tile_flow`,
`Environment.tile_domain`) — it went away with `FlowArray` and returns as a
position-wrapping implementation covering 2D *and* 3D, after the plotting work.
`Environment.extend` remains removed, and is decided at the same time. Do not
reintroduce a materializing tiling stopgap — nor a virtualizing one; see
`docs/notes/run_persistence.md` §9.2 for why both failed.

**When tiling comes back, work from the restoration checklist at
`docs/notes/run_persistence.md` §9.3.** Gating it off left notices and
replaced tests across source, tests, examples, docs, and prose; §9.3 lists every
one. Both old bodies are preserved **commented out beneath their `raise`**, under a
`PREVIOUS IMPLEMENTATION, KEPT FOR RESTORATION` banner — reuse them rather than
rewriting. Only the fluid halves are superseded; `tile_domain`'s ibmesh/`L` logic
and `tile_flow`'s `flow_points` extension still stand.

## The canonical workflow

```python
import planktos
envir = planktos.Environment()              # define the world
envir.set_brinkman_flow(...)                # or read_IB2d_fluid_data / load_NetCDF / etc.
envir.read_IB2d_mesh_data(...)              # optional immersed boundaries
swrm = planktos.Swarm(swarm_size=100, envir=envir)   # add agents
for _ in range(steps):
    swrm.move(dt)                           # advance one step
swrm.plot_all(movie_filename='out.mp4')     # visualize
```

See `examples/` for runnable scripts (start with `basic_ex_2d.py`,
`basic_ex_3d.py`). `ex_ib2d_mvbnd_sticky.py` is the **2D moving-boundary**
showcase (needs external data — see the file header for the download link).

### Plotting: 2D is the real thing, 3D is a stand-in

All plotting is matplotlib today. **The 3D plotting is explicitly a placeholder**
awaiting a vtk-powered library, and when that lands 3D plotting will be **split out
entirely** from the 2D path. Two working consequences:

- **Do not invest in matplotlib 3D rendering.** Effort spent enriching 3D frames is
  written off at the rewrite. Keep 3D changes minimal and cheap.
- **Do not contort 2D designs to stay symmetric with 3D.** Shared abstractions
  spanning both would only have to be unpicked at the split. 2D and 3D diverging is
  the intended direction, not a wart.

Already visible in the code: `Swarm.plot_all`'s `fluid='vort'|'quiver'` backdrops
are 2D-only, so a 3D frame draws nothing about the fluid — and `fluid=` is forced to
`None` in 3D at record time *and* ignored there at render time. See
`docs/notes/run_persistence.md` §0.2.

### Plotting goes through `_frames.FrameSource`

**`plot` and `plot_all` build a `_frames.FrameSource` and index it.** Agent state is
always the Swarm's own history. The *fluid backdrop* comes from the run archive the
Environment recorded — `Environment._archive_path`, set by `record()` and kept after
recording stops, so the ordinary workflow needs no argument:

```python
with envir.record('run/', fluid='vort'):
    ...
swrm.plot_all(movie_filename='out.mkv', fluid='vort')   # reads no fluid data
```

Two rules it enforces:

- **No render may read the fluid dataset without saying so.** With a recording, a
  backdrop that was not recorded is a **refusal** — the only fallback is re-reading the
  whole dataset to draw a picture of it. Without one, a windowed replay warns with an
  estimate.
- **What is available is decided by what is *resident*, not by what was recorded.** With
  the whole field in memory the curl is derived from it and the arrows subsampled from
  it: free, exact, never missing.

⚠️ **`animate(n)` has no final-frame branch.** The last state *is* the present, so
`source.positions(n)` returns `self.positions` at `n == len(pos_history)`. The old
`n >= len(pos_history)` duplicate is deleted; do not reintroduce one.

**Plotting a run in a later session is not a plotting problem.** It needs the
Environment and Swarm restored to where the run left off, which Planktos does not do
yet. That is **component R, the full-state reboot** — `run_persistence.md` §2.11 for
the specification and §6.1 Step R for the build order — and it is the next substantial
item on this branch, ahead of tiling. Do not solve it inside `plot`/`plot_all`.

## Customizing agent behavior — the one rule that matters

**To change how agents move, subclass `Swarm` and override `apply_agent_model(self, dt)`.**
It must *return* (not assign) the new `NxD` positions array. Do **not** override
`move()` — `move()` is the harness that records history, applies boundary
conditions, recomputes velocity/acceleration by finite difference, and advances
time. Optionally override `after_move(self, dt)` to act on final positions/props
(e.g. marking stuck agents).

Inside `apply_agent_model`, typically call a `planktos.motion` generator, e.g.
`planktos.motion.Euler_brownian_motion(self, dt)`. Default behavior is a random
walk: drift = local fluid velocity + `shared_props['mu']`, diffusion =
`shared_props['cov']`.

Helper accessors for use inside behavior code: `get_fluid_drift()`, `get_dudt()`,
`get_fluid_mag_gradient()`, `get_prop(name)`, `add_prop(...)`. Per-agent variation
lives in the pandas DataFrame `Swarm.props`; shared values in `Swarm.shared_props`.

## Immersed boundaries & collisions

- Agents treat the `ibmesh` as solid. Collision behavior is set per-`Swarm` via
  `ib_condition` (and per-move via `move(..., ib_collisions=...)`):
  - `'sliding'` (default): no flux normal to the boundary; remaining movement is
    projected onto the boundary (recursive vector projection).
  - `'sticky'`: agent stops at the point of intersection for that step.
  - `None`: ignore immersed boundaries entirely.
- After each move, `Swarm.ib_collision_idx` is a length-N int array: `-1` if no
  collision that step, else the index of the first mesh element struck. (This
  replaced the old boolean `ib_collision` — see `changelog.txt`.)
- **Mesh assumption:** segments must not cross except at shared vertices. Verify
  imported meshes with `Environment.plot_envir()`. `add_vertices_to_static_2D_ibmesh`
  exists to repair crossings but is discouraged.
- **3D immersed boundaries are STL triangular (FEM) surface meshes** — this is now
  the norm; **3D vertex-point input is deprecated** (2D vertex points are still used).
  2D meshes (static or **moving**) load from IB2d data via `read_IB2d_mesh_data`
  (directory of `lagsPts.####.vtk` → moving; single `.vtk`/`.vertex` → static).

## Correctness invariants & development priorities

This code prioritizes **scientific accuracy and robustness above all** — "nothing
breaks" is a hard requirement, not an aspiration.

**Correctness outranks reproducing previous output.** This is a research code, so a fix
that makes results *more correct* ships even when it changes numbers users have already
published — including in a patch release, and including where the change is small. Do
not propose keeping a known-wrong result for compatibility, and do not pin a defect with
a regression test "to make the change visible": that leaves a lock encoding the bug.
Test the correct answer instead, and record the size of the shift in the changelog and
the cherry-pick queue so users can tell whether it reaches them. (Decided 2026-08-13,
over the periodic-gradient fixes.)

Treat the following as load-bearing:

- **The workhorses are the agent–boundary intersection routines** (`_geom.py`)
  and the **collision/interaction handlers** (`_ibc.py`). These are the riskiest,
  most subtle code in the project. Change them with extreme care.
- **Hard invariant: no agent may ever end up on the wrong side of a boundary
  (penetration).** This must hold for *arbitrary* geometry and movement,
  including the hard cases: where two or more mesh elements join (concave/convex
  joints), and under moving boundaries. Roundoff error is the enemy — penetration
  caused by floating-point error at joints or near-tangent hits is a real bug, not
  noise. Preserve the careful epsilon/tolerance handling already in place.
- **Sliding collisions are the most delicate path.** They handle many distinct
  geometric situations and are potentially **recursive** (project onto a boundary,
  which may push the agent into another boundary, repeat until the move vector is
  exhausted). Reason through all cases before touching this.
- When in doubt about a change to intersection/collision code, prefer to add a
  test that pins the current (trusted) behavior before refactoring.

## Where the math lives

Algorithm/derivation notes are in `docs/notes/` (Markdown with LaTeX):
- `project_and_slide_moving.md` — the moving-boundary project-and-slide math
  (the core of the 2D moving-boundary work). Implemented in
  `_ibc._project_and_slide_moving`.
- `Equations_of_motion.md`, `Intersection_w_multilinear_polynomial.md`,
  `Lines_closest_points.md` — supporting derivations.

## Documentation

- Source of truth for behavior is the **docstrings** in the source (NumPy style),
  which Sphinx autodoc renders. `docs/` builds the readthedocs site
  (`docs/index.rst`, `docs/quickstart.rst`, `docs/api/`, `docs/examples/`).
- `README.md` is a **landing page, not a reference.** Its hand-maintained API
  listing was removed in `1.0.1` (it had drifted) and replaced with links to the
  generated docs. Do not reintroduce a duplicated API listing there — if
  something is undocumented, fix the docstring in the source.
- Run `codespell README.md docs/ planktos/ examples/` after documentation work;
  the tree was made clean in `1.0.1`. Note it has ambiguous cases it will not
  auto-fix (`-w`), so read its output rather than trusting a zero exit alone.

## CI and pre-commit

- **GitHub Actions** (`.github/workflows/tests.yml`) runs the test suite and
  codespell on every push and pull request. It is the authority: it cannot be
  bypassed and it runs on Linux, which has already caught a failure that did not
  reproduce on the user's Windows machine (numpy 2.5 removing `np.cross` for
  2-vectors). If CI fails and you cannot reproduce locally, **check the
  dependency versions first** — the runner installs the newest of everything.
- **`.pre-commit-config.yaml`** mirrors the codespell check locally. It is inert
  until `pre-commit install` is run **once per clone**, which is easy to forget
  on a new machine — if the user is setting up a fresh clone, remind them.
  `git commit --no-verify` bypasses it. Documented under "Development" in the
  README.
- Keep the pre-commit codespell skip list in sync with the workflow's, so the
  two cannot disagree about what is checked.

## Tests

The suite is organized into focused, deterministic, fast modules (overhauled
2026-06). Run `pytest` from the repository root. **Two flags widen the run, and
they are independent:**

| Invocation | Covers | Time |
|---|---|---|
| `pytest` | the focused modules | ~20 s (1013 passed / 141 skipped) |
| `pytest --runslow` | plus the parallelization tests, the plotting smokes and the movie renders | ~40 s (1046 / 108) |
| `pytest --runstreaming` | plus the fast half of `tests/test_data_streaming/` | ~30 s (1099 / 50 / 5 xfailed) |
| `pytest --runslow --runstreaming` | everything | ~4 min (1147 / 2 / 5) |

**Before a commit that touches the archive, the fluid streaming or the plotting
paths, run both flags.** `--runstreaming` is off by default because that suite is
a goal line for work in progress rather than a regression net (see the carve-out
below), and it roughly doubles the default run on its own.

- **Run** the whole thing with `pytest`; a specific area with e.g.
  `pytest tests/test_collisions_static.py`.
- **Modules** (all self-contained / analytic-answer unless noted):
  - `test_geom.py` — `_geom` intersection & closest-distance functions.
  - `test_collisions_static.py` / `test_collisions_moving.py` /
    `test_collisions_static_3d.py` — call `_ibc.apply_internal_static_BC` /
    `apply_internal_moving_BC` directly across a geometry × movement matrix (2D
    segments, 3D triangle meshes; convex/concave joints, grazing, deep recursive
    multi-element slides); assert no-penetration and exact post-collision
    positions. `test_collisions_moving.py` also pins a deterministic multi-step
    `Swarm.move()` trajectory (golden drift detector), the moving slider's
    rotate-away release branch (needs an element pivoting about an *interior*
    point — see `_ib_harness.pivoting_segment`), and frame-independence of the
    slide.
  - `test_collisions_junctions.py` — the cases the chain-shaped builders above
    structurally cannot reach: vertices of degree > 2 and non-manifold edges,
    where a slide running off an element has several candidates to continue
    onto. Asserts invariants (finite, motion not amplified, stays outside a
    closed obstacle, rigid-motion equivariance) rather than exact positions.
  - `test_collisions_invariants.py` — the checks that are *not* tied to any
    geometry: the answer must be finite, must not amplify the motion, must not
    depend on where the problem sits or which way the axes point, and must
    behave the same in any units. Also holds the stack-exhaustion cases, since
    recursion depth is set by step length against mesh spacing rather than by
    shape. Deliberately uses plain geometries, so a failure is attributable to
    the property and not to an exotic mesh.
  - `test_ibc_helpers.py` — the small helpers and guard rails inside `_ibc`:
    `_boundary_eps`, `_point_in_triangle`, `make_ib_worker` unpacking, and the
    2D-only guard on moving meshes.
  - `test_collisions_stl_3d.py` — end-to-end 3D: load a generated STL via
    `Environment.read_stl_mesh_data` and drive agents into it with `Swarm.move()`
    (needs the optional numpy-stl; module skips otherwise).
  - `test_flow_generation.py` — brinkman/channel/canopy, `tile_domain` (now pinned
    as raising `NotImplementedError`, including that a failed call leaves the
    environment unmutated), `flow_points` axis order. One deliberate skip: `Environment.extend`
    was removed on this branch (extrapolation is the intended replacement), and
    the test is parked rather than deleted because `extend` may come back for the
    specific fluid fields where it makes sense. Un-skip it if that happens.
  - `test_temporal_interp.py` — `fluid.fCubicSpline` and `FluidData`'s temporal
    interpolation (`create_temporal_interpolations` was absorbed into `FluidData`).
  - `test_flow_interface.py` — pins the `Environment.flow` consumer contract (it
    was written as the safety net for the `FlowArray` removal, which it survived
    unchanged): `interpolate_flow`/`interpolate_temporal_flow`
    values, the container + spline-indexing surface, `fmin`/`fmax` tuples,
    `_calc_basic_stats` (including that it pulls **no** fluid field),
    `get_mean_velocity`, `get_mean_fluid_speed`, `calculate_mag_gradient`,
    `get_raw_loaded_data`, `fshape`, the plotting strided-slice path, the
    `LinearSpline`/`INUM` temporal path (the in-memory half of dynamic loading —
    `test_temporal_interp.py` covers only `fCubicSpline`), and 3D vorticity. All
    closed-form. See `docs/notes/run_persistence.md` Appendix A — **this is the
    safety net for that refactor; keep it green as the work lands.**
  - `test_run_archive.py` — the **on-disk archive format**, driven directly with
    synthetic arrays and no simulation: chunk boundaries, the fingerprint, atomic
    file replacement, and crash validity demonstrated by actually `SIGKILL`ing a
    subprocess mid-recording. Reads the bytes back with raw `np.load`/`json.load`
    rather than through a reader of our own, since a round-trip through our own
    code can be self-consistently wrong.
  - `test_recording.py` — `Environment.record` and the capture hooks against **real
    runs**: the round-trip versus `pos_history`/`vel_history`/`time_history`, a
    swarm added mid-run, `capture_interval` (including that a coarse schedule gives
    **bit-identical** trajectories, over two mesh geometries — see the note in
    `run_persistence.md` §6.1 A3b for why one is not enough), the five refusals, and
    the headline: recording a windowed run costs *identically* many loader calls as
    the same run without it.
  - `test_run_reader.py` — `planktos.load_run` / `RunArchive`: that reading one
    capture opens exactly two files, that chunks are memmapped and the open-file
    cache stays bounded, that a missing middle chunk refuses rather than
    short-reads, and that reading never writes.
  - `test_provenance.py` — that every fluid and mesh entry point is wrapped by
    `_provenance.records_provenance`, asserted **structurally**: a loader nobody
    decorated produces no error, just an environment that cannot say what it is.
  - `test_fluid_recording.py` — the **fluid half of a run archive** (component B of
    `run_persistence.md`): which of §3.3's three vorticity regimes gets chosen, that a
    blended per-dump field equals the live curl for the sourced *and* the written case
    and that the two agree, the two-slot read cache reading each dump once on a
    monotone sweep in either direction, the per-dump statistics sidecar (including
    that it is written in 3D, which is the whole 3D deliverable), quiver, and the B
    counterpart of the headline: recording the fluid costs **no extra loader calls**.
  - `test_data_streaming/` — an **adversarial suite** written from
    `run_persistence.md`, covering the streaming story end to end (in-RAM,
    windowed replay, recorded replay, restart). **Opt-in: `--runstreaming`**, and
    its slow members additionally need `--runslow`. Its strict `xfail`s are the
    pre-release list — see "What an `xfail` means here". Run it after any change
    to the archive, the fluid streaming or the plotting paths.
  - `test_archive_rendering.py` — **drawing a run whose fluid was recorded**
    (component C of `run_persistence.md`). Its headline is the third of the series:
    replaying a recorded windowed run costs **zero** loader calls, with the same
    replay unrecorded costing a full second pass beside it so the zero means
    something. Also the refusals, the global colour limit (including that `NaN` for a
    dump the run never reached does not poison it), the quiver grid fixed at record
    time, and `record(plot_all=)`. Mostly drives `_frames.FrameSource` rather than a
    figure — it is what decides a frame, and it is exactly what `animate` calls; the
    end-to-end movie renders are `@slow` and need ffmpeg.
  - `test_dynamic_loading.py` — the **windowed** (`INUM=int`) path, i.e. this
    branch's headline feature: `FluidData.update_spline`. Covers TODO Phase 1
    (A) windowed-linear == full-linear to round-off, (B) slide behavior (forward,
    backward, jump-to-start, extrapolation flags, bounded window and load count),
    and (D) `get_dudt` under linear splining, plus the per-dump mean cache behind
    the plot statistics (`get_mean_velocity` exact across slides; a replay after a
    full sweep costs zero loads). Those sections drive the
    real slider from a synthetic `FluidData` subclass whose `load_dumpfiles`
    slices an in-memory array, so they touch no files. Two further sections run
    the actual loaders — `VTK3dData` and `IB2dData` — against committed fixtures,
    because what they pin is the **timeline a loader builds from files**: that it
    spans the whole dump series rather than the opening window, which is what
    decides whether a window ever slides. Only Phase 1 (C), the quantitative
    linear-vs-cubic number, still needs real data.
  - `test_agent_models.py` — `apply_agent_model`/`after_move` overrides, the
    `motion` generators, and the public `motion.RK45` solver contract.
  - `test_material_derivative.py` — `Swarm.get_DuDt` / `get_dudt` (closed-form).
  - `test_swarm_lifecycle.py` — `move()` bookkeeping, mask contract, and domain
    BCs (zero/noflux/periodic) in 2D, 3D, and mixed-per-dimension combinations.
  - `test_periodic_ib.py` — periodic domain boundary × immersed boundary (an
    agent wraps across the domain and immediately meets a wall on the far side).
  - `test_swarm_save.py` — round-trips for `save_pos_to_csv` / `save_data` /
    `save_pos_to_vtk`.
  - `test_analysis.py` — `get_vorticity`, forward & backward FTLE (closed-form).
  - `test_io_loaders.py` — IB2d moving/static mesh import (committed fixtures),
    the IB2d fluid `uX`/`uY` scalar branch, `_dataio.read_vtk_time_only` (the
    header-only `TIME` scan), IBAMR vtk (`@vtk`), COMSOL vtu (`@vtu`).
  - `test_frame_selection.py` — the parts of `plot_all` that decide a frame
    without drawing one. `Swarm._select_frames` (frame spacing in simulated time,
    snapping to the nearest recorded state, index alignment with `pos_history`, a
    run whose `dt` changed partway, the clamp and uneven-spacing warnings), and
    `_vorticity_norm`, the colour limits of the RdBu backdrop. Drives real tiny
    runs but renders nothing; rendering is `test_plotting_smoke.py`.
  - `test_parallel_ib.py` — serial == threads == processes (`@slow`).
  - `test_plotting_smoke.py` — `plot_*` methods run without error on the Agg
    backend (`@slow`; the movie test also needs ffmpeg).
- **Helpers / fixtures**: `tests/_ib_harness.py` (mesh builders + invariant
  assertions; also drives the parallel scenarios and the golden moving-boundary
  trajectory); `tests/fixtures/` holds tiny committed fixtures (~100 kB total),
  all regenerable via `tests/fixtures/_gen_fixtures.py` — edit that script rather
  than hand-editing a vtk:
  - `lagspts_min/`, `mesh_min/` — IB2d moving and static immersed boundaries.
  - `ib2d_fluid_min/` (vector `u.####.vtk`), `ib2d_fluid_scalar_min/`
    (`uX`/`uY.####.vtk`) — 2D fluid. Fields are `u = t` and `v = sin(2πx/Lx)`, so
    `u` reads back the simulation time and a frozen or truncated timeline is
    immediately visible. IB2d omits the periodic endpoint, so a 6×5 dump loads as
    a 7×6 field over a 6×5 domain.
  - `vtk3d_min/` — 8 rectilinear 3D dumps carrying `TIME` field data, for
    `VTK3dData`. Field is `u = t`, `v = x`, `w = t·z`.
- **Markers** (registered in `pytest.ini`): `slow` (only with `--runslow`),
  `streaming` (only with `--runstreaming`; applied to the whole of
  `tests/test_data_streaming/` by a module-level `pytestmark`),
  `vtk` (skipped if vtk data absent), `vtu` (skipped if COMSOL data absent).
  The two `--run*` flags are independent, so a test carrying both markers needs
  both flags.
- **Non-automated** visual/exploratory scripts live in `tests/manual/` —
  excluded from collection via `collect_ignore` in `conftest.py` (at the
  repository root, not in `tests/`).

### What an `xfail` means here (a priority signal, not a parking space)

**An `xfail` in this suite marks a bug serious enough to stop the development
cycle for.** It is not a way to defer something inconvenient. The convention:

- **`xfail` ⇒ drop other work and fix it.** If a defect does not warrant that,
  it does not get an `xfail` — it goes in the **issue tracker** with
  reproduction notes instead.
- **Always `strict=True`**, always with a `reason` naming the defect. Strict
  means the suite *fails* the moment the test starts passing, so a marker cannot
  outlive its bug.
- **No issue-tracker entry is needed for an `xfail`ed bug.** The test already
  catalogues it, with an executable reproduction — that is strictly better than
  prose. Duplicating it in the tracker just creates two things to keep in sync.
- **Delete the marker in the same commit as the fix.**
- Outside the carve-out below, the steady state is zero `xfail`s. One appearing
  in ordinary work means drop everything and fix it.

Check with `pytest -rX` (lists xfails) or `pytest -rxX` (xfails and xpasses).

**The carve-out: `tests/test_data_streaming/`.** This is an adversarial suite
written against `docs/notes/run_persistence.md`, covering the streaming story end
to end — in-RAM, windowed replay, recorded replay, restart. Its strict `xfail`s
are **the pre-release list for `dyload`**: known gaps to be worked before the
branch ships, not stop-the-cycle defects. A non-empty list there is expected, so
watch the **count** rather than the presence — it should only ever go down, and
each one that goes down takes its marker with it.

**It is opt-in, behind `--runstreaming`** (its slow members want `--runslow`
too). That is why: it is a statement of where the work is going, so it is read
deliberately rather than swept past in every run, and it costs more than the rest
of the suite put together. The flip side is that **it never runs unless someone
asks**, CI included — so:

⚠️ **Run `pytest --runslow --runstreaming` after any change to the archive, the
fluid streaming or the plotting paths, and before a release.** It has already
caught three defects the focused modules missed, all in component C.

Its own `tests/test_data_streaming/README.md` is the standing record: the four
claims, the verdict on each, and every finding with what fixed it.

### Resolved defects & FTLE notes

The overhaul and its follow-ups uncovered and fixed a series of latent bugs, each
with a regression test; see `changelog.txt` for the list. Two FTLE specifics
worth knowing (`calculate_FTLE`):
- `FTLE_smallest` is the smallest-eigenvalue (contraction) exponent, **not**
  backward-time FTLE (the old "negate it" guidance was wrong). For attracting LCS,
  call `calculate_FTLE(..., backward=True)` — it integrates the reversed flow and
  stores the backward field in `FTLE_largest`. Backward is **tracer-only** (reverse-
  time inertial/custom dynamics are dissipative/ill-posed). Forward works for
  tracer, `ode_gen` (inertial/custom), and user-`swrm` models.
- FTLE respects **static** immersed boundaries but **not moving** ones (it doesn't
  advance `envir.time`, so a moving mesh would be frozen) — a moving mesh now raises
  `NotImplementedError`.

### Testing goals (ongoing)

- Favor small, exact analytic setups with known answers over large simulations;
  keep the default run fast and deterministic.
- The key property is the **no-penetration invariant** (agents end on the correct
  side of every boundary) plus correctness of the resulting position. Extend the
  geometry × movement matrix (convex/concave joints, grazing, multi-element,
  moving vs static, sliding vs sticky) in `test_collisions_*`.
- Pin trusted moving-boundary behavior with regression locks before refactors.

## Conventions & gotchas

- **Underscored modules are internal.** Add new public surface only to
  `__init__.py` exports; keep helpers in underscored modules.
- Classes `Environment` and `Swarm` are capitalized (a deliberate `1.0.0` rename).
- **Masked arrays everywhere** for agent state — use `.copy()` before mutating
  `self.positions`/`velocities`/`accelerations`; direct assignment is by reference
  and the auto-update in `move()` will overwrite velocity/acceleration anyway.
- Multiple swarms in one environment: advance them with `Environment.move_swarms()`.
  A bare `Swarm.move()` **raises** when the environment holds more than one swarm —
  one swarm advancing the clock while the others stand still is no longer supported
  (it used to warn and freeze the others, which desynchronized their histories).
  `update_time=False` exists for `move_swarms` to call and is not a user-facing knob.
- **FFmpeg** must be on `$PATH` to save animation videos.
- **Data files are gitignored** (`*.vtk`, `*.vtu`, `*.vertex`, `*.stl`, `*.mp4`,
  `*.npz`, `data/`, etc.). Large example/test datasets are downloaded separately.
- `proj_dev/` is the gitignored scratch/dev folder convention for work-in-progress
  with data.
- `past_projects/` holds prior research code (e.g. `brine_shrimp/`) kept for
  reference; not part of the package.
- Build: setuptools via `setup.cfg` (deps: numpy, scipy>=1.10.1, matplotlib>=3,
  pandas, vtk>=9.2, pyvista>=0.44; optional extras: STL, netCDF, test). Editable
  install with `pip install -e .`.
- `changelog.txt` is hand-maintained — update it for user-facing changes.
- `TODO.md` is the working plan for this branch (phases, known bugs, design
  history). Keep it current as items land; see "Branch context" above.
