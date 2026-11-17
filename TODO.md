# TODO — `vtk` branch

**Goal of this branch:** move 3D plotting from matplotlib to vtk, unlocking 3D
visualization the matplotlib placeholder cannot do. **Timeboxed:** 3D moving immersed
boundaries are the research priority, so if vtk plotting proves slow it is set aside
and work moves to them, possibly on a new branch.

**State *(2026-11-14)*.** 1.1.0 is released from `master` (tag `v1.1.0`). Work on
this branch has not started.

This file holds what is still to do. Finished work is in `changelog.txt` and in the
git history, including this file's own: 1.1.0's build record is
`git log -p v1.1.0 -- TODO.md docs/notes/run_persistence.md`.

Priority key: 🔴 do first · 🟡 next · 🟢 later · ⚪ deferred / low priority.

---

## ▶️ vtk 3D plotting 🔴

1. 🔴 **Agents as points, first.** This is what stalled the move before, so settle it
   before building anything around it:
   - **Size:** the sphere glyphs tried were always too large. Glyphs are sized in
     world units, so a fixed size is wrong at every domain scale but one.
     *Untested leads:* render agents as points sized in screen pixels
     (`Plotter.add_points(..., render_points_as_spheres=True, point_size=px)`),
     which stay the same size on screen at any zoom; or keep world-unit glyphs with
     the scale derived from the domain extent.
   - **Heading mark,** as 2D plots draw. *Untested lead:* `PolyData.glyph` with an
     arrow or cone geometry (`pv.Arrow()`, `pv.Cone()`), oriented by the agents'
     velocity array, with `scale=False` and a fixed `factor` so every mark is the
     same size.
2. **Build on PyVista.** It is already a required dependency (`setup.cfg`), chosen
   in part with 3D plotting in mind, and its `Plotter` covers points, glyphs,
   meshes and movie writing. Drop to raw vtk only where PyVista falls short.
3. **Scope.** 3D only: the matplotlib 2D path stays as it is (see "Plotting" in
   `CLAUDE.md`). Keep the changes self-contained, so that setting the work aside
   leaves nothing half-replaced.
4. Once a 3D fluid backdrop exists: `OpenFOAMData` ships a `vorticity` cell array
   that is deliberately not read, since nothing draws a 3D backdrop yet. Read it
   then, through `probe_stored_vorticity` / `read_dump_vorticity`. Recomputed
   vorticity matches the solver's to 1e-7 in the interior but differs by 84% of rms
   in the outermost cell layer of the oral-arm dataset (2026-08-12; suspected cause,
   unverified: differencing against the spliced boundary plane), so a 3D backdrop
   should prefer the stored field.

---

## Open 🟡

- **2D info block runs off the right edge.** The "% remain / Info" text in 2D
  `Swarm.plot` and `plot_all` is left-aligned at figure x = 0.77, the left edge of
  the vertical histogram, so its last line clips (seen in a 900×1200 `plot_all`
  movie: "Agent |v|: 4 ± 1.9 m…"). Shift it left into the empty space beside the
  horizontal histogram; the histogram stays put. It's a 2D-path fix, which
  `master` can take directly.

- **Movie file size.** Suspected needlessly large for what they show: flat-colour
  plots of points and lines. `plot_all` passes no codec or quality settings to
  `animation.FFMpegWriter` (`_swarm.py`, `movie_filename` branch; `dpi=150`), so
  the encoding is whatever matplotlib and ffmpeg default to. Look into whether
  tuned settings (e.g. libx264 CRF/preset/tune) shrink the files with no visible
  loss, and apply the answer to the vtk movie writer too.

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

## Next — 3D moving immersed boundaries 🟡

After the vtk plotting work, or in its place if that is set aside. Not on the `vtk`
branch. Moving boundaries are 2D only; 3D immersed boundaries are STL surface meshes.
The known pieces:

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

- **`Environment.extend`** was removed in 1.1.0 (`3eb9d1d`). Whether it returns is its
  own pass (decided 2026-11-02). If it does, it fits as a clamping out-of-grid policy
  beside extrapolation and wrapping in `interpolate_flow`; the brine-shrimp projects
  in `past_projects/` used `extend(y_plus=384)`, and
  `test_flow_generation.py::test_extend_grows_domain_and_copies_edges` un-skips.
- **Rectilinear (non-uniform) seed grids in `calculate_FTLE`** — a feature, for
  resolution matching on a refined mesh; no current dataset needs it. FTLE seeds on a
  uniform grid built from `L`, independent of `flow_points`, so a non-uniform fluid
  grid already gives correct results. Shape of the change: `grid_init` takes
  coordinate arrays, `calculate_FTLE` takes `grid_points`, and the central-difference
  denominator becomes `xgrid[n1] - xgrid[n0]`. A centred difference on a non-uniform
  grid is first order unless the weighted three-point formula is used — decide that
  deliberately.
- **Faster OpenFOAM dump reads.** 61% of each oral-arm `internal.vtu` is geometry
  that never changes, and `vtkXMLUnstructuredGridReader` re-reads it every dump
  (unused field arrays are already skipped). Two options: parse the `U` `DataArray`
  straight out of the XML (inline base64, uncompressed, `UInt64` headers, so about
  7× less I/O per dump), or preprocess each dump to `.npy` once. The
  `openfoam_min` fixture keeps foamToVTK's byte layout for this.
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
