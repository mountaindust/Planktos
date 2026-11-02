# CLAUDE.md

Guidance for working in the Planktos repository. It is loaded into every session.

**Keep it durable.** Anything likely to change often — counts, versions, what is next,
what a test module covers — does not get stated here as a fact. Point to the place the
current answer lives; where there is no such place, date the claim, so that its age is
obvious. Claims below marked *(as of …)* are the dated kind: check them before relying
on them.

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
  string in `__init__.py` is the single source of truth.
- **`git tag` lists what has been released, and a tagged version is closed.** Its
  `changelog.txt` entry is shipped history: leave it alone, on both branches. If
  `master`'s `__version__` equals its newest tag, the next change landing there needs
  a bump first.
- *(As of 2026-10-30: `dyload` is developing `1.1.0`; the newest release is `1.0.3`.)*
- **A fix made here that is not dyload-specific goes under the development version,**
  since the released ones are closed: in `changelog.txt` *and* in the
  **cherry-pick queue at the bottom of `TODO.md`**. Check portability with
  `git diff master -- <file>` before assuming it applies — `_swarm.py` in
  particular has diverged a long way.
- *(Plan as of 2026-10-30.)* **The next release is `1.1.0`, with no `1.0.4`.** That queue is
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
- **Brevity must not cost specificity.** The short version says *more* concretely
  what the thing does, not less: name it — the dict a record actually is, the
  exception raised, the method that raises it — and give every pronoun a
  referent. Vague shorthand is not brief, only short: *"names them"*, *"a reader
  must not act on one silently"*, *"jsonable() is the guarantee that it can"*.
  Do not explain away an alternative nobody would choose. Where a claim is about
  what some consumer does, read that consumer and say what it actually does.
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

**The branch's docstrings were swept to this standard** (finished 2026-10-22) — every
module this branch added or changed. What the sweep did *not* reach: docstrings
`master` also has and this branch never touched, and the texts held back until the
`master` merge (the merge item's follow-up list in `TODO.md`). Do not add more of what
the sweep removed.

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

- **`dyload`** — *(as of 2026-10-30)* **the development branch.** The main feature work: **dynamic
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

*(As of 2026-10-30.)* **3D moving boundaries are parked and not scheduled.** `TODO.md` keeps the list under
a PARKED heading purely so the pieces are not rediscovered from scratch later; do not
start work from it. Moving boundaries are currently **2D only**.

**3D fluid data sources (terse, for future reference).** The 3D flows come from
either **IBFE** (fluid solved over the *entire* domain, including inside immersed
objects, as SAMRAI files on an adaptive rectangular mesh) or **OpenFOAM**. Planktos
**assumes a rectilinear fluid grid** — that assumption stands, and is the thing to
check about any new dataset.

What it does *not* mean is that the data must arrive as rectilinear vtk. **The oral-arm
OpenFOAM dataset is stored as `vtkUnstructuredGrid` yet sits on a perfectly uniform
Cartesian grid** — `foamToVTK` writes `.vtu` regardless of the underlying mesh, so the
container says nothing about the geometry. It needs **no resampling**, only a reordering
permutation (cell ordering is not lexicographic) and a half-cell boundary splice. Fluid
there is defined *everywhere*, including inside the immersed object, which carries a
porosity source term rather than being cut out of the mesh. Verify a dataset's actual
lattice before assuming it needs a VisIt/ParaView pass; see
`docs/notes/openfoam_oral_arm_dataset.md`.

`OpenFOAMData` reads that dataset. What stays out of scope is *other* source-specific
ingestion: porting the old VisIt SAMRAI→vtk script, reading COMSOL directly, and OpenFOAM cases
whose mesh is genuinely unstructured (`snappyHexMesh` refinement, tets, polyhedra).

**`TODO.md` is the working plan for this branch**, and the authority on what is done,
what is known-broken and what comes next. Read it before starting work here, keep it
updated as items land, and look there rather than here for status. It also holds the
design history behind the fluid architecture.

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

| File | Public? | Purpose |
|------|---------|---------|
| `planktos/__init__.py` | yes | The public API: its imports are the list. `motion` is reachable as `planktos.motion`. |
| `planktos/_environment.py` | `Environment` class | The domain: boundary conditions, immersed boundary mesh, swarms, time. Loads fluid/mesh data, generates analytical flows, plots, computes vorticity/FTLE. Fluid data itself lives in `fluid.py` (see below). |
| `planktos/_swarm.py` | `Swarm` class | A group of agents: positions/velocities/props, the move loop, boundary-condition application, plotting, data saving. |
| `planktos/motion.py` | yes (`planktos.motion`) | Equation-of-motion generators & solvers: `Euler_brownian_motion` (default SDE), `inertial_particles`, `highRe_massive_drift`, `tracer_particles`, `RK45`. |
| `planktos/fluid.py` | `FluidData` is user-visible via `Environment.flow`; the rest internal | All fluid velocity data and its temporal interpolation: `FluidData` (+ per-source `IB2dData`, `VTK3dData`, `ComsolVTUData`, `OpenFOAMData`, `VTKXMLData`), `LinearSpline`, `fCubicSpline`, `SplineRangeError`. See "Fluid data architecture" below. |
| `planktos/_geom.py` | internal | Pure geometry workhorses: segment/line/triangle intersections, closest distances, multilinear-polynomial intersection (for moving meshes). Formerly static methods of `Swarm`. |
| `planktos/_ibc.py` | internal | Immersed-boundary collision handling: `apply_internal_static_BC`, `apply_internal_moving_BC`, and the project-and-slide routines for static and moving meshes. |
| `planktos/_dataio.py` | internal | Low-level read/write of vtk, vtu, .vertex, stl, NetCDF. Use `Environment` loader methods instead of calling these directly. |
| `planktos/archive.py` | `RunArchive` and `load_run` are public; `RunRecorder` is re-exported here | **Reading** a run archive — chunked, append-only, crash-valid capture of agent state written as a run proceeds, plus the per-dump fluid quantities a later plot needs. Streams agents *out* the way `fluid.py` streams the field *in*. Its module docstring is the on-disk format spec. `RunArchive.restore` rebuilds the Environment and Swarms. See `docs/notes/run_persistence.md` §2 and §3. |
| `planktos/_recording.py` | internal (`RunRecorder` reached through `Environment.record`) | **Writing** a run archive: `RunRecorder`, the agent writer `_ArchiveWriter`, the fluid planning and `_FluidWriter`, and the choice of directory (`_resolve_archive_path`, `_appendable`). |
| `planktos/_archive_format.py` | internal | What reading and writing share: format constants, `_atomic_write`, file naming, and the fingerprint functions that decide whether an archive matches an Environment. |
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
  `ComsolVTUData`, `OpenFOAMData`, `VTKXMLData` (`.vti`/`.vtr` series, with or without a `.pvd` index).
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
  field. `docs/notes/run_persistence.md` §3.3 has the reasoning. A series Planktos
  writes is **stamped** in each file's vtk header with a digest of its field, so a
  later recording rewrites a stale one and a replay refuses a rewritten one; an
  unstamped file is the solver's and is never touched.
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

**That gap has been measured** (2026-08-11); detail in `TODO.md`,
reproducible via `tests/manual/quantify_temporal_interp.py` and
`vet_dynamic_loading_3d.py`. Linear vs cubic rms error: **1.27% vs 0.54% of U_rms** in
2D (Δt=1e-3 s), **9.46% vs 1.13%** in 3D at the coarser cadence reachable there. Quote
the **2D convergence orders** (median point: linear 1.68, cubic 4.75) — the 3D export
is too coarsely sampled to fit one, and the script refuses to print a meaningless
slope. Ensemble agent statistics agree to within 0.35% (3D) / 0.6% (2D).

A third dataset, the 2D sea fan, sits at the coarse end — ~20 samples per pulse —
and is the one to quote for a sparse cadence: **5.49% vs 1.50% of U_rms** at
Δt=0.2 s, median orders 1.91/4.86, ensemble means within 0.09%. It also pins the
memory claim on real data: a windowed sweep holds **41 MB where cubic holds 1604 MB**.
Reproduce with `tests/manual/quantify_seafan_interp.py`; detail in
`docs/notes/run_persistence.md` Appendix C.

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

*(As of 2026-11-02; `TODO.md` has its status.)* **Domain tiling raises
`NotImplementedError`** (`FluidData.tile_flow`, `Environment.tile_domain`) — it went
away with `FlowArray` and returns as a position-wrapping implementation covering 2D
*and* 3D. Do not reintroduce a materializing tiling stopgap — nor a virtualizing one;
see `docs/notes/run_persistence.md` §9.2 for why both failed. `Environment.extend`
remains removed; whether it returns is a separate pass.

**The plan for bringing tiling back is `docs/notes/run_persistence.md` §9** (settled
2026-11-02): the design, a consumer audit, the ground-truth tests and the build order,
with the restoration checklist at §9.6. Two things there overrule older guidance.
**`flow_points` stays the stored tile's grid** — only `L` and the plotted extent are
tiled — so `tile_flow`'s old `flow_points` extension does not come back. And **the
commented-out `tile_domain` mesh loop is not to be reused verbatim**: it duplicates the
base mesh and mis-indexes moving meshes, bugs the released `1.0.3` shares.

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

**Plotting a run in a later session is not a plotting problem.** It goes through
`planktos.load_run(path).restore()`, which rebuilds the Environment and Swarms where the
run left off (`run_persistence.md` §2.11), and then plots them as usual. Do not solve it
inside `plot`/`plot_all`.

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
- **An STL meeting a 2D fluid is flattened, not sliced.** A 2D simulation's geometry
  arrives as a straight extrusion (the sea-fan `plate.stl`); `_geom.flatten_extruded_mesh`
  drops the axis the fluid dropped (`FluidData._flat`, else z) and refuses anything
  else. **`Environment.flow` is a property** whose setter fits an already-loaded mesh
  to the incoming fluid or refuses the fluid with `ValueError`, leaving the environment
  unchanged — so a new fluid entry point gets the check for free, and must assign
  `self.flow` before changing anything else (see `read_NetCDF_flow`). Slicing a genuinely
  3D STL at the fluid's plane is deferred until a dataset needs it.

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

| Invocation | Covers | Time *(2026-10-30, Windows)* |
|---|---|---|
| `pytest` | the regular suite | ~1 min |
| `pytest --runslow` | plus the parallelization tests, the plotting smokes, the movie renders and the cross-version comparison | ~1½ min |
| `pytest --runexamples` | plus the example scripts, run as written | ~4 min |
| `pytest --runslow --runexamples` | everything; this is what CI runs | ~4½ min |

For the current count, run it. `pytest --durations=20` shows where the time goes.

**Before a commit that touches the archive, the fluid streaming or the plotting
paths, and before a release, run both flags.** Examples needing downloaded data skip
themselves where it is absent, as they do in CI, so only a machine holding that data
runs them.

- **Run** the whole thing with `pytest`; a specific area with e.g.
  `pytest tests/test_collisions_static.py`.
- **Modules.** Each module's docstring says what it pins, and is the place to look; this
  is only a map. A new module does not need adding here unless it starts a new area.

  | Area | Modules |
  |---|---|
  | Intersections and collisions | `test_geom`, `test_collisions_*`, `test_ibc_helpers`, `test_periodic_ib`, `test_parallel_ib` |
  | Fluid data and analysis | `test_flow_generation`, `test_flow_interface`, `test_temporal_interp`, `test_dynamic_loading`, `test_material_derivative`, `test_analysis` |
  | File loaders and low-level I/O | `test_io_loaders` |
  | Run archive | `test_run_archive` (the bytes), `test_recording` (real runs, restore, append), `test_run_reader`, `test_fluid_recording`, `test_archive_rendering`, `test_provenance` |
  | Swarm behavior | `test_agent_models`, `test_swarm_lifecycle`, `test_swarm_save` |
  | Plotting | `test_frame_selection`, `test_plotting_smoke` |
  | End-to-end acceptance | `test_data_streaming/` (see its `README.md`) |

- **Helpers and fixtures.** `tests/_ib_harness.py` holds the mesh builders and invariant
  assertions. `tests/fixtures/README.md` describes every committed fixture and its
  analytic field; regenerate them with `tests/fixtures/_gen_fixtures.py` rather than
  hand-editing a vtk.
- **Markers** are registered in `pytest.ini`. `slow` needs `--runslow`, and `examples`
  needs `--runexamples`.
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
- The steady state is zero `xfail`s, everywhere. One appearing means drop everything
  and fix it.

Check with `pytest -rX` (lists xfails) or `pytest -rxX` (xfails and xpasses).

**`tests/test_data_streaming/` is an adversarial acceptance suite** covering the streaming
story end to end — in-RAM, windowed replay, recorded replay, restart — and it runs with
everything else. Its own `README.md` is the standing record: the four claims, the verdict
on each, and every finding with what fixed it.

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
- Build: setuptools; `setup.cfg` lists the dependencies and optional extras. Editable
  install with `pip install -e .`.
- `changelog.txt` is hand-maintained — update it for user-facing changes.
- `TODO.md` is the working plan for this branch (phases, known bugs, design
  history). Keep it current as items land; see "Branch context" above.
