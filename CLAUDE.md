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
  version on this branch is `1.1.0`; `1.0.1` is the documentation-only release
  that shipped from `master`.
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

**3D moving boundaries are planned but not started.** They are blocked on `dyload`
(3D dynamic fluid loading) working first, because of the data-size problem above.
Moving boundaries are currently **2D only**.

**3D fluid data sources (terse, for future reference).** The 3D flows come from
either **IBFE** (fluid solved over the *entire* domain, including inside immersed
objects, as SAMRAI files on an adaptive rectangular mesh) or **OpenFOAM** (fluid only
*outside* objects, on a 3D FEM point mesh). For now, in both cases the field is
interpolated to a **rectilinear grid** externally (VisIt/ParaView → vtk) and that vtk
is what Planktos loads — i.e. **assume a rectilinear fluid grid for the time being.**
Source-specific ingestion (porting the old VisIt SAMRAI→vtk script, reading
OpenFOAM/COMSOL directly, etc.) is intentionally **out of scope for `dyload`**: it is
lower priority than getting 3D moving boundaries working.

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
  `get_vorticity`, `get_dudt`, `calculate_DuDt`, `update_spline`, `load_dumpfiles`.
- `FluidData.get_raw_loaded_data()` is the nearest thing to the old
  `Environment.regenerate_flow_data()`. Note *loaded*: under dynamic loading only
  the current window exists.
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
stays the default for datasets that fit in memory. **Quantifying that gap is still
an open task** (TODO.md Phase 1C); do not assert a magnitude for it until then.

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
numpy operations on flow data. `docs/notes/flow_field_interface.md` is the full
record.

**Domain tiling currently raises `NotImplementedError`** (`FluidData.tile_flow`,
`Environment.tile_domain`) — it went away with `FlowArray` and returns as a
position-wrapping implementation covering 2D *and* 3D, after the plotting work.
`Environment.extend` remains removed, and is decided at the same time. Do not
reintroduce a materializing tiling stopgap; see §5/§9 of the note for why.

**When tiling comes back, work from the restoration checklist at
`docs/notes/flow_field_interface.md` §9.1.** Gating it off left notices and
replaced tests across source, tests, examples, docs, and prose; §9.1 lists every
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
are 2D-only, so a 3D frame draws nothing about the fluid. See
`docs/notes/flow_field_interface.md` §8.2.

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
breaks" is a hard requirement, not an aspiration. Treat the following as load-bearing:

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
2026-06). Run `pytest` from the repository root. The default run is ~1s; add
`--runslow` for the slower checks — the full-simulation parallelization tests and
the plotting smokes — which brings it to roughly 13s.

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
    `Swarm.move()` trajectory (golden drift detector).
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
    `_calc_basic_stats`, `get_mean_fluid_speed`, `calculate_mag_gradient`,
    `get_raw_loaded_data`, `fshape`, the plotting strided-slice path, the
    `LinearSpline`/`INUM` temporal path (the in-memory half of dynamic loading —
    `test_temporal_interp.py` covers only `fCubicSpline`), and 3D vorticity. All
    closed-form. See `docs/notes/flow_field_interface.md` §7.2 — **this is the
    safety net for that refactor; keep it green as the work lands.**
  - `test_dynamic_loading.py` — the **windowed** (`INUM=int`) path, i.e. this
    branch's headline feature: `FluidData.update_spline`. Covers TODO Phase 1
    (A) windowed-linear == full-linear to round-off, (B) slide behavior (forward,
    backward, jump-to-start, extrapolation flags, bounded window and load count),
    and (D) `get_dudt` under linear splining. The (A)/(B)/(D) sections drive the
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
  `vtk` (skipped if vtk data absent), `vtu` (skipped if COMSOL data absent).
- **Non-automated** visual/exploratory scripts live in `tests/manual/` —
  excluded from collection via `collect_ignore` in `conftest.py` (at the
  repository root, not in `tests/`).

### Resolved defects & FTLE notes

The overhaul and its follow-ups uncovered and fixed a series of latent bugs, each
with a regression test (the suite has no xfails); see `changelog.txt` for the
list. Two FTLE specifics worth knowing (`calculate_FTLE`):
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
- Multiple swarms in one environment: advance them with `Environment.move_swarms()`
  (or call each `Swarm.move(update_time=False)` then bump time), not a bare
  per-swarm `move()` loop, which warns about un-advanced swarms.
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
