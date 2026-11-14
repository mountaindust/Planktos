# TODO — `dyload` branch

**Goal of this branch:** load time-dependent fluid velocity data on demand (a sliding
window of dumps) instead of holding the whole dataset in memory, so that large 3D
time-varying flows can be used — and, around that, persist a run as it proceeds so it
can be plotted, analysed and resumed later without re-reading the fluid.

**State *(2026-11-13)*.** Built, tested and documented: `FluidData` and dynamic
loading in 2D and 3D, the OpenFOAM and `.vti`/`.vtr` loaders, the run archive
(`Environment.record`, `planktos.load_run`, `RunArchive.restore`, append), plotting
from a recording, tiling, `Swarm.retire`/`release` and simultaneous `move_swarms`. What
is left before 1.1.0 is the release itself.

This file holds what is still to do. Finished work is in `changelog.txt` and in the
git history, including this file's own: the full build record of the branch is
`git log -p -- TODO.md docs/notes/run_persistence.md` (the latter deleted at the
1.1.0 cleanup).

Priority key: 🔴 do first · 🟡 next · 🟢 later · ⚪ deferred / low priority.

---

## ▶️ Before 1.1.0

1. 🔴 **Document cleanup** *(started 2026-11-13)*. `TODO.md` cut to what is open;
   `docs/notes/run_persistence.md` deleted, with its deferred items moved below and
   the references to it in `CLAUDE.md`, source comments and test docs repointed.
2. 🔴 **Release check:** `__version__` in `planktos/__init__.py` → `1.1.0`; a read
   through the 1.1.0 section of `changelog.txt`; `pytest --runslow --runexamples`; a
   fresh docs build; `codespell README.md docs/ planktos/ examples/`. Then tag, and
   merge to `master`. Version bump and changelog are the user's call — surface, don't
   do silently.
   - **The version-bump commit carries `Closes #58, closes #59`** (decided
     2026-11-13): dynamic loading and memory-free tiling are what those issues ask
     for. GitHub closes them when that commit reaches `master`, i.e. at the merge.

---

## Open 🟡

- **`examples/10_sticky_seafan_3d.py` has not been run since tiling** *(2026-11-03)*:
  it needs `Velocity2to1_8mmps.vtu`, which is not on the development machine. Its
  cylinder must lie strictly inside the stored tile along x; the STL loader's shift
  by the fluid's lower-left corner should put it there.
- **The 2D sea-fan dataset** (`tests/data/openfoam2D/`). Two questions for the
  collaborator, neither blocking a read, both changing what a study on it means:
  1. The export is a cropped core window — ±20 mm of an OpenFOAM domain of
     [−50, 50] × [−40, 40] mm — so the environment's boundary conditions act where
     the real flow continues. A full-domain re-export would remove the question.
  2. `inFluid` is unused: Planktos ignores it, so the plate outline must come from
     `plate.stl` (which now loads into a 2D environment).

  Any example built on it must scale the diffusion and say so in its header: the
  domain is 4 cm across in metres, where the default `shared_props['cov']` empties
  the domain within about 14 steps at `dt = 0.05`.

---

## Deferred / low priority ⚪

**Run archive and plotting**

- **History-free running** (`store_pos_history=None`): keep no `pos_history` and
  rely on the archive. `capture_interval` already caps history memory while losing
  nothing; what is left needs its own pass, because live `plot_all` and
  `_calc_basic_stats` read the histories directly and would have nothing to read
  without an archive.
- **`save_*` re-expressed as exports from the archive.** The three save methods need
  the whole history resident and have no loader; `save_pos_to_csv` writes one dense
  `N+1 × (1+D)·T` text matrix in a single call (over a gigabyte at 1000 agents ×
  10 000 steps in 3D).
- **Async frame writing.** Rendering serializes with the physics. Matplotlib is not
  thread-safe, but rendering in the main thread and handing the encode/write to a
  writer thread would hide most of the I/O. Measure first — it may be irrelevant
  next to the physics.
- **A live one-pass render mode** (rendering without an archive), only if a workflow
  appears that cannot afford the archive. It would have to solve global colour and
  arrow scales without the whole run in hand.
- **Restore with several fluid loads.** A mesh loaded between two *different* fluid
  loads replays against the last fluid only, since only the last fluid record is
  kept.

**Fluid**

- **`Environment.extend`** was removed on this branch (`3eb9d1d`) and still exists on
  `master`. Whether it returns is its own pass (decided 2026-11-02). If it does, it
  fits as a clamping out-of-grid policy beside extrapolation and wrapping in
  `interpolate_flow`; the brine-shrimp projects in `past_projects/` used
  `extend(y_plus=384)`, and
  `test_flow_generation.py::test_extend_grows_domain_and_copies_edges` un-skips.
- **Rectilinear (non-uniform) seed grids in `calculate_FTLE`** — a feature, for
  resolution matching on a refined mesh; no current dataset needs it. FTLE seeds on a
  uniform grid built from `L`, independent of `flow_points`, so a non-uniform fluid
  grid already gives correct results. Shape of the change: `grid_init` takes
  coordinate arrays, `calculate_FTLE` takes `grid_points`, and the central-difference
  denominator becomes `xgrid[n1] - xgrid[n0]`. A centred difference on a non-uniform
  grid is first order unless the weighted three-point formula is used — decide that
  deliberately.
- **COMSOL** (`ComsolVTUData`): full-load only, unverified against a current export;
  the collaborator no longer uses COMSOL. `test_io_loaders.py::test_vtu_load` stays
  skipped without a committed fixture.
- **NetCDF** (`load_NetCDF` / `read_NetCDF_flow`): full-load only, never used in
  practice. Lowest priority.
- **Slicing a genuinely 3D STL at a 2D fluid's plane.** Only straight extrusions are
  handled (flattened); wait for a dataset that needs more.
- **Backward FTLE for non-tracer models** is unsupported (reverse-time dynamics are
  dissipative and ill-posed); it would need a stabilized or adjoint approach.
- ⚠️ **Watch: vtk on numpy 2.5.** `vtkmodules/util/numpy_support.py` assigns
  `result.shape = shape`, which numpy 2.5 deprecates. If numpy makes it an error,
  every legacy vtk read breaks at once. The fix is upstream; re-check when raising
  the vtk or numpy minimum.

**When 3D plotting moves to vtk** (the matplotlib 3D path is a placeholder)

- `OpenFOAMData` ships a `vorticity` cell array that is deliberately not read, since
  nothing draws a 3D backdrop. Read it then, through `probe_stored_vorticity` /
  `read_dump_vorticity`. Recomputed vorticity matches the solver's to 1e-7 in the
  interior but differs by 84% of rms in the outermost cell layer of the oral-arm
  dataset (2026-08-12; suspected cause, unverified: differencing against the spliced
  boundary plane), so a 3D backdrop should prefer the stored field.

---

## PARKED — 3D moving immersed boundaries ⚪ (not scheduled; do not start)

A parking lot, not a plan: kept so the pieces are not rediscovered from scratch.
Moving boundaries are 2D only; 3D immersed boundaries are STL surface meshes.

- The 3D moving-mesh path raises `NotImplementedError`. Static 3D collisions are
  covered (`test_collisions_static_3d.py`, `test_collisions_stl_3d.py`).
- **Moving-mesh FTLE:** `calculate_FTLE` never advances `envir.time`, so it raises
  for a moving mesh. A fix threads integration time into
  `interpolate_temporal_mesh`, forward and reversed.
- **Issue #73** — the moving slider's mid-step excursion check samples three times
  rather than root-finding, and misses some excursions (accuracy only, never a
  penetration). Revisit when the moving slider is rewritten for triangles.
- The `_ibc` free-flight guard against a zero-length step is untested by decision:
  20 000 randomized static trials never reached it, and removing it fails loudly as
  a `RecursionError`. The moving-mesh guards take their times from a numerical
  solver, so 0 or NaN is more plausible there — a real case reaching one is worth a
  test.
