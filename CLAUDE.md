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
  it via `attr: planktos.__version__`. The current development version is `1.0.0`.
- `changelog.txt` is hand-maintained, terse, and grouped by version. When a
  change is user-facing, prompt to add an entry under the appropriate version.
- When work looks release-worthy (or a user-facing change lands) but the version
  or changelog hasn't been touched, **say so** and confirm the right action.
- Do NOT bump the version or rewrite the changelog silently — surface the need
  and let the user decide (a version bump is a semver judgment call).

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

- **`mvbnd`** — current working branch. Goal: 2D **moving immersed boundaries**
  (largely done; this is the `1.0.0` work). Not yet merged to master pending a
  manuscript submission. **You are usually here** for changes *unrelated* to
  dynamic fluid loading, which then get merged into `dyload` as needed.
- **`dyload`** — the main feature branch under active development: **dynamic
  loading of fluid data** (streaming/loading fluid time steps on demand rather
  than all at once). Not your default working branch. This matters because
  time-dependent 3D fluid data is often ~100 GB raw and significantly larger once
  splined, so it cannot be held in memory all at once.
- **`master`** — stable/published.

**3D moving boundaries are planned but not started.** They are blocked on `dyload`
(3D dynamic fluid loading) working first, because of the data-size problem above.
Moving boundaries are currently **2D only**.

When making cross-cutting changes (like this CLAUDE.md), expect them to be
merged from `mvbnd` into `dyload` later.

## Package layout

The installable package is `planktos/`. Public API is intentionally tiny;
internal modules carry a **leading underscore** and are not part of the public
surface (a deliberate convention — see `changelog.txt`).

| File | Public? | Purpose |
|------|---------|---------|
| `planktos/__init__.py` | yes | Exports `Environment`, `Swarm`. `motion` is reachable as `planktos.motion`. |
| `planktos/_environment.py` (~4250 ln) | `Environment` class | The fluid domain: holds flow field, immersed boundary mesh, swarms, time. Loads fluid/mesh data, generates analytical flows, plots, computes vorticity/FTLE. |
| `planktos/_swarm.py` (~3160 ln) | `Swarm` class | A group of agents: positions/velocities/props, the move loop, boundary-condition application, plotting, data saving. |
| `planktos/motion.py` (~550 ln) | yes (`planktos.motion`) | Equation-of-motion generators & solvers: `Euler_brownian_motion` (default SDE), `inertial_particles`, `highRe_massive_drift`, `tracer_particles`, `RK45`. |
| `planktos/fluid.py` (~380 ln) | mostly internal | Fluid data loading helpers + `fCubicSpline` (subclass of `scipy.interpolate.CubicSpline`) and `create_temporal_interpolations`. Newer module (2025); relevant to `dyload`. |
| `planktos/_geom.py` (~840 ln) | internal | Pure geometry workhorses: segment/line/triangle intersections, closest distances, multilinear-polynomial intersection (for moving meshes). Formerly static methods of `Swarm`. |
| `planktos/_ibc.py` (~1170 ln) | internal | Immersed-boundary collision handling: `apply_internal_static_BC`, `apply_internal_moving_BC`, and the project-and-slide routines for static and moving meshes. |
| `planktos/_dataio.py` (~680 ln) | internal | Low-level read/write of vtk, vtu, .vertex, stl, NetCDF. Use `Environment` loader methods instead of calling these directly. |

## Core mental model

1. **`Environment`** is the world: domain size `L`, boundary conditions `bndry`,
   a fluid velocity field `flow`, an immersed boundary mesh `ibmesh`, and a list
   of `swarms`. It owns the simulation `time` and `time_history`.
2. **`Swarm`** is a vectorized group of agents (NOT individual objects — agents
   are rows in numpy arrays for speed). It belongs to one `Environment`.
3. `positions` (and `velocities`, `accelerations`) are **masked arrays** of shape
   `Nx2`/`Nx3`. **A masked row = that agent has left the domain** and is no
   longer updated. Respect/preserve the mask.
4. The fluid `flow` is a list of ndarrays (one per spatial dim). For
   time-dependent flow the first axis is time. On first temporal interpolation
   these arrays are replaced in place by `fCubicSpline` objects; the raw data can
   be recovered with `Environment.regenerate_flow_data()`. Interpolation is
   **cubic spline in time, linear in space**.

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
`basic_ex_3d.py`). `ex_ib2d_mvbnd_sticky.py` is the **moving-boundary** showcase
for this branch (needs external data — see the file header for the download link).

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
- Static 3D meshes load from STL; 2D meshes (static or **moving**) load from
  IB2d data via `read_IB2d_mesh_data` (directory of `lagsPts.####.vtk` → moving;
  single `.vtk`/`.vertex` → static).

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
  (the core of this branch). Implemented in `_ibc._project_and_slide_moving`.
- `Equations_of_motion.md`, `Intersection_w_multilinear_polynomial.md`,
  `Lines_closest_points.md` — supporting derivations.

## Documentation

- Source of truth for behavior is the **docstrings** in the source (NumPy style),
  which Sphinx autodoc renders. `docs/` builds the readthedocs site
  (`docs/index.rst`, `docs/quickstart.rst`, `docs/api/`, `docs/examples/`).
- The **API listing in `README.md` is a hand-maintained mirror that can drift**
  out of sync with the code (its method names were last reconciled with the source
  on 2026-06-24). When the README and the source disagree, **trust the source
  docstrings**, and fix the README.

## Tests

The suite was overhauled (2026-06) into focused, deterministic, fast modules.
Run `pytest` from the repository root. The default run is ~1s; add `--runslow`
for the full-simulation parallelization checks (~30s).

- **Run** the whole thing with `pytest`; a specific area with e.g.
  `pytest tests/test_collisions_static.py`.
- **Modules** (all self-contained / analytic-answer unless noted):
  - `test_geom.py` — `_geom` intersection & closest-distance functions.
  - `test_collisions_static.py` / `test_collisions_moving.py` — call
    `_ibc.apply_internal_static_BC` / `apply_internal_moving_BC` directly on
    single trajectories across a geometry × movement matrix; assert no-penetration
    and exact post-collision positions.
  - `test_flow_generation.py` — brinkman/channel/canopy, `tile_flow`, `extend`,
    `flow_points` axis order.
  - `test_temporal_interp.py` — `fluid.fCubicSpline` / `create_temporal_interpolations`.
  - `test_agent_models.py` — `apply_agent_model`/`after_move` overrides and
    `motion` generators.
  - `test_swarm_lifecycle.py` — `move()` bookkeeping, mask contract, domain BCs.
  - `test_analysis.py` — `get_2D_vorticity`, FTLE (closed-form answers).
  - `test_io_loaders.py` — IB2d moving/static mesh import (committed fixtures),
    IBAMR vtk (`@vtk`), COMSOL vtu (`@vtu`).
  - `test_parallel_ib.py` — serial == threads == processes (`@slow`).
- **Helpers / fixtures**: `tests/_ib_harness.py` (mesh builders + invariant
  assertions, also drives the parallel scenarios); `tests/fixtures/` holds tiny
  committed IB2d fixtures, regenerable via `tests/fixtures/_gen_fixtures.py`.
- **Markers** (registered in `pytest.ini`): `slow` (only with `--runslow`),
  `vtk` (skipped if vtk data absent), `vtu` (skipped if COMSOL data absent).
- **Non-automated** visual/exploratory scripts live in `tests/manual/`
  (`visualtest_*.py`, `mvib2d.py`, `rubberband.py`, the `.ipynb`, the perf
  benchmark) — excluded from collection via `collect_ignore` in `conftest.py`.

### Resolved defects & FTLE notes

The overhaul uncovered four latent bugs; **all four are now fixed** with regression
tests (the suite has no remaining xfails): sticky moving-boundary NaN on
axis-aligned elements in `_ibc`; the zero-length-segment `ValueError` in
`_geom.closest_dist_btwn_lines_and_pt`; `save_fluid`/`save_2D_vorticity` on modern
pyvista; and backward-time FTLE. See `TODO.md` for details and the remaining
non-blocking follow-ups.

FTLE specifics worth knowing (`calculate_FTLE`):
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
</content>
</invoke>
