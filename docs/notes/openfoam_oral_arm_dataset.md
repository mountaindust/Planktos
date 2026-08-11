# OpenFOAM oral-arm dataset — structural analysis

Analysis of the Phase 2 test dataset: the `VTK/` export shipped with
`README_oral_arm_setup.md` (case `case08_alpha2_1e8`), performed 2026-07-31 to
determine whether the data is usable by Planktos and what an ingestion path would
look like. Everything below was verified by reading the files, not inferred from
the README.

**The data itself lives at `tests/unsteady_3D_testdata/`, which is gitignored in
its entirety.** This note was originally written into that directory and was
therefore untracked; it was moved here on 2026-08-10 so the analysis survives
independently of the data. §7 is the work list for the Phase 2 loader — see
`TODO.md` Phase 2, which points here rather than restating it.

**Headline:** despite being stored as `vtkUnstructuredGrid`, the fluid data sits on a
**perfectly uniform Cartesian grid** that never changes in time. No regridding is
needed. Fluid velocity is defined *everywhere*, including inside the disk.

---

## 1. What the file formats are

All VTK XML formats. None are new dataset types — the `.vtm`/`.vtm.series` pair is
just a container/index scheme.

| Ext | VTK type | Role here |
|---|---|---|
| `.vtu` | `UnstructuredGrid` | the actual volume data (points, cells, fields) |
| `.vtp` | `PolyData` | boundary surface patches (inlet / outlet / walls) |
| `.vtm` | `MultiBlockDataSet` | a **manifest**, contains no data — names child files |
| `.vtm.series` | ParaView convention | a **JSON index** mapping `.vtm` files → times |

`.vtm.series` is plain JSON and needs no VTK to read:

```json
{"file-series-version":"1.0",
 "files":[{"name":"case08_alpha2_1e8_787.vtm","time":7.5}, ...]}
```

Hierarchy:

```
case08_alpha2_1e8.vtm.series          JSON: filename -> time
  └── case08_alpha2_1e8_<idx>.vtm     XML manifest
        ├── <idx>/internal.vtu        <-- the volume data (85 MB)
        └── <idx>/boundary/{inlet,outlet,walls}.vtp
```

This is exactly what OpenFOAM's `foamToVTK` writer emits. Every OpenFOAM export is
`.vtu` regardless of the underlying mesh — the "unstructured" label is the writer
being generic, not a statement about the mesh.

Total on disk: **1.5 GB**, ~85 MB per timestep.

---

## 2. Is it actually on a rectilinear grid?

**Yes — and stronger than rectilinear: uniform Cartesian.** Verified, not assumed:

- 803,531 points / 775,368 cells; **every** cell is VTK type 12 (`VTK_HEXAHEDRON`).
  No tets, prisms, or polyhedra.
- 67 × 67 × 179 = **803,531** points exactly; 66 × 66 × 178 = **775,368** cells
  exactly. Zero duplicate points.
- Spacing constant to float32 roundoff: dx = dy = 0.00151515 m, dz = 0.00150561 m
  (relative variation ~1e-6). **No mesh refinement anywhere, not even at the disk.**
- Every cell mapped to an integer (i,j,k) against a uniform lattice: max deviation
  8e-6 of a cell width; all 775,368 triples distinct and complete. Full tensor
  product, no holes.

Conclusion: `snappyHexMesh` was never run. This is a plain `blockMesh` box, and the
disk is represented **purely by the porosity source term**, not by the mesh.

### The one real wrinkle: fields are CELL data

`U`, `p`, `vorticity`, `Q` are all `CELL_DATA` (OpenFOAM is finite-volume).
`point_data` is empty. So the grid Planktos would interpolate on is the
**66 × 66 × 178 cell-center lattice**, which is inset half a cell from the walls:

```
x, y ∈ [-0.049242, 0.049242]   (walls at ±0.05)
z    ∈ [ 0.003753, 0.270247]   (inlet 0.003, outlet 0.271)
```

### Cell ordering is NOT lexicographic

A bare `.reshape((66,66,178))` silently scrambles the field. Actual strides:

```
cc[1]     - cc[0] = (+dx,   0,    0)
cc[66]    - cc[0] = (2dx, 4dy,    0)      # not a clean row stride
cc[66*66] - cc[0] = (5dx, 3dy, 20dz)
```

A permutation index is required. Build it by snapping to an integer lattice, not by
sorting raw floats — the coords are float32 and an exact `lexsort` + 1e-9 comparison
fails on x from roundoff alone:

```python
ix = np.rint((cc[:,0] - cc[:,0].min()) / dx).astype(int)
```

(For a merely-rectilinear, non-uniform future dataset, `np.searchsorted` against
rounded `np.unique` grid vectors generalizes.)

**The same trap survives into the derived cell centers, which is where a loader
actually meets it.** `vtkCellCenters` averages the float32 corners and returns
float64, so it looks like it should be clean — it is not. Measured through
`_dataio.read_vtkxml_cell_data` on dump 787: `np.unique` on the center coordinates
reports **79 distinct y-levels where only 66 exist** (and 77 / 75 on the inlet and
outlet patches), because cells that share a y-level average to values differing in
the last bits. `np.unique` is therefore unusable for recovering the grid vectors,
not merely for sorting. Snapping recovers exactly 66 / 66 / 178 levels, with max
deviation from an integer of **8.5e-7** of a cell width in x and y and **8.0e-6**
in z.

---

## 3. Are the sample points fixed in time?

**Fixed — bit-for-bit identical.** The point-coordinate array and the connectivity
array were md5'd across all 17 available timesteps: every one matches the first
exactly.

The mesh is written redundantly into each file (most of the 85 MB) but never
changes. For dynamic loading this means geometry + permutation can be computed
**once** and only the field arrays streamed.

---

## 4. Is fluid specified inside the oral-arm disk?

**Yes, everywhere.** There is no hole in the mesh, no blanking, no mask, no NaN.
The disk region is ordinary fluid cells carrying a momentum sink.

At t = 7.5 s:

| Region | cells | mean \|U\| | max \|U\| |
|---|---|---|---|
| inside disk (r < 0.025, 0.015 < z < 0.021) | 3,392 | 1.13e-4 | 2.00e-4 |
| — solid stem (r < 0.0025) | 48 | 1.27e-8 | 1.63e-8 |
| — porous annulus | 3,344 | 1.15e-4 | 2.00e-4 |
| below disk (z < 0.015) | 34,848 | 1.37e-3 | 5.65e-3 |
| above disk (z > 0.021) | 723,096 | 1.41e-4 | 2.83e-3 |

Domain-wide: |U| ∈ [9.3e-9, 1.07e-2], **zero NaNs, zero exactly-zero cells**.

Interpretation: the Brinkman term is working — the annulus carries ~2% of peak
speed, the stem is effectively stagnant but not exactly zero. Agents wandering into
the disk will see a real, very slow velocity field rather than garbage, and
**nothing in the fluid data will stop them**. To make the disk solid to agents,
load `oral_arm_disk.stl` as an `ibmesh` (16,793 points / 33,582 triangles; bounds
r ≈ 0.025, z ∈ [0.015, 0.021], matching the README).

---

## 5. Boundary patches — they splice in exactly

The half-cell inset is precisely the failure mode the commented-out
`center_cell_regrid` docstring warns about (`planktos/fluid.py`, ~line 2070).
`ComsolVTUData.__init__` sets `self.L` from `flow_points[i][-1]`, so raw cell
centers would report the domain as 0.09848 × 0.09848 × 0.26649 instead of the true
0.1 × 0.1 × 0.268.

The `.vtp` patches close that gap **exactly** — no extrapolation needed:

| Patch | Cells | Geometry | `U` |
|---|---|---|---|
| `inlet.vtp` | 4,356 = 66×66 | plane z = 0.003; x/y centers **identical** to interior lattice | forcing (0 at t = 7.5, a cycle start) |
| `outlet.vtp` | 4,356 = 66×66 | plane z = 0.271; same x/y lattice | nonzero, up to 5.4e-4 |
| `walls.vtp` | 46,992 = 4×66×178 | four planes x = ±0.05, y = ±0.05 | **exactly 0** (no-slip) |

Assembly: interior 66×66×178 → glue inlet/outlet caps → 66×66×**180** → pad the
four lateral walls → **68×68×180**, spanning the full domain.

The 12 edges and 8 corners appear in no file, but every one lies on a no-slip wall,
so filling zeros there is **exact**, not an approximation.

Practical notes:
- `walls.vtp` is **optional** — padding zeros is equivalent, and skips parsing
  4 MB × 21 files. `walls.vtp` is also a single PolyData holding all four planes, so
  using it means splitting by coordinate yourself.
- `inlet` / `outlet` carry real data and should be read.

**"Identical x/y lattice" is now verified through the reader, not just asserted.**
Re-measured 2026-08-10 via `_dataio.read_vtkxml_cell_data`: once each patch's
centers are snapped as in §2, the recovered inlet x and y grid vectors match the
interior's **bit-for-bit** (max \|diff\| exactly 0.0). Compare raw `np.unique`
output between patch and interior and it appears to disagree — that is the
roundoff artifact above, not a real lattice mismatch. Also confirmed through the
reader: `walls.vtp` `U` is exactly zero everywhere, so the zero-padding shortcut is
exact rather than approximate.

---

## 6. Time series

- Declared: 21 snapshots, t = 7.5 … 10.0 s, Δt = 0.125 s = T/10 (T = 1.25 s).
  Two full pulse cycles.
- Mean vertical velocity below the disk traces a clean pulse
  (4.6e-6 → 1.24e-3 → 4.0e-6 over t = 7.5–8.75, then repeats), consistent with the
  README's `U_peak·½(1 − cos 2πft)` at f = 0.8 Hz, U_peak = 0.01 m/s.

### ⚠ Four timesteps are missing

The `.vtm` manifests exist but the directories they point to do not, so ParaView
will error on these times. Looks like a truncated transfer.

| Missing `.vtm` | t |
|---|---|
| `case08_alpha2_1e8_943.vtm` | 9.0 |
| `case08_alpha2_1e8_982.vtm` | 9.375 |
| `case08_alpha2_1e8_1021.vtm` | 9.75 |
| `case08_alpha2_1e8_1047.vtm` | 10.0 |

Present: 787, 800, 813, 826, 839, 852, 865, 878, 891, 904, 917, 930, 956, 969, 995,
1008, 1034 (17 of 21). Still 17 as of 2026-08-10.

**Decision (2026-08-10): work with the data as-is, and make gap tolerance a loader
requirement.** Rather than block on a re-send, the loader is to handle a dump series
with holes — a truncated or interrupted export is a normal thing to be handed, and a
reader that only works on a complete series is the more fragile artifact. See §7.

The resulting timeline, worth having explicitly since it is what the loader must
produce (index → t at Δt = 0.125 s from 787 → 7.5):

```
7.500 7.625 7.750 7.875 8.000 8.125 8.250 8.375 8.500 8.625 8.750 8.875   <- 12 dumps, unbroken
      [9.000 missing]   9.125 9.250   [9.375 missing]   9.500 9.625
      [9.750 missing]   9.875         [10.000 missing — end truncation]
```

So **three interior holes** (9.0, 9.375, 9.75), each widening one interval from
0.125 s to 0.25 s, plus one **end truncation** (10.0). The first two-thirds of the
series is unbroken at full cadence. End truncation is benign — the series just stops
early. Interior holes are the case that changes interpolation, and there are three.

---

## 7. Implications for a Planktos reader

### Structure
The loop shape is `VTK3dData._read_vtkfiles` (iterate files, append per-component,
stack), **not** `ComsolVTUData._read_vtufile` (single file, all times inside).
COMSOL packs three scalar arrays per timestep with `t=` in the name; OpenFOAM writes
one `Nx3` vector array named `U` per file. Times come from the `.vtm.series` JSON
(or the per-file `TimeValue` field array).

### A dump series with holes must not break the loader ✅ DONE (2026-08-11)

Built as specified below, in `OpenFOAMData._read_series`. Required behavior, not an
optional nicety — this dataset has three interior holes and
one end truncation (§6), and a truncated or interrupted export is a normal thing to be
handed:

1. **Resolve which dumps actually exist eagerly, at construction,** while the timeline
   is being built. Never discover a missing file at the window slide that needs it:
   under dynamic loading that raise lands arbitrarily deep into a long run, which is
   the worst possible moment and precisely what streaming makes likely.
2. **Warn once**, naming the missing times and the count. A silent hole is worse than
   a failure — the run completes and nothing says the timeline is not what the
   manifest declared.
3. **Build `flow_times` and the dump index densely over the surviving dumps.** Then
   `d_start`/`d_finish` index the series that exists, `load_dumpfiles` is never handed
   an absent filename, and nothing downstream changes except that one interval is
   wider.
4. **Warn separately about the resulting non-uniform spacing**, since interpolation
   error scales with the dump interval and the user should know which stretch of the
   timeline is degraded.

⚠️ This interacts with the `FluidData` guard that a dynamically-loading subclass must
supply `flow_times` spanning the whole dump range: "whole range" means **the dumps
that exist**, not the ones the `.vtm.series` manifest lists. Building `flow_times`
from the manifest and the file index from the directory listing would put the two
index spaces silently out of step — the same class of bug as the `VTK3dData` frozen
timeline (`TODO.md` Phase 2).

### Low-level read ✅ DONE (2026-08-10)

Three new functions in `_dataio.py`, all generic to the VTK XML container scheme
rather than to OpenFOAM, covered by `tests/fixtures/openfoam_min/` and
`tests/test_io_loaders.py`:

| Function | Reads | Uses VTK? |
|---|---|---|
| `read_vtm_series` | `.vtm.series` → member paths + times | no (`json`) |
| `read_vtm_manifest` | `.vtm` → `{name: path}` + `TimeValue` | no (`xml.etree`) |
| `read_vtkxml_cell_data` | `.vtu`/`.vtp` → cell centers, cell arrays, time | yes |

Notes on the shape they landed in:

- **A new function, not a change to `read_vtu_Unstructured_Grid_Points_FEM`.** An
  earlier version of this section framed it as two edits to that function
  (`GetPointData()`→`GetCellData()`, `GetPoints()`→cell centers). Both changes are
  right in substance and `read_vtkxml_cell_data` makes them — but that function is
  live for `ComsolVTUData`, which genuinely wants point data on FEM corner points.
  Editing it in place would have broken COMSOL for no gain.
- `.vtm` is parsed with the standard library, **not** `vtkXMLMultiBlockDataReader`,
  which would read every child file (85 MB) merely to report their names. Hand
  parsing a 700-byte XML is what makes "resolve which dumps exist, eagerly, at
  construction" cheap enough to be unconditional.
- `read_vtkxml_cell_data(arrays=...)` deselects unwanted cell arrays before
  `Update()`. Default is `('U',)`; `vorticity` is one argument away, since
  collaborators' exports often ship it and reading it beats regenerating it.
- Missing files raise `FileNotFoundError`. vtk's XML readers otherwise report the
  problem on stderr and return an *empty dataset*, which surfaces much later as a
  confusing shape mismatch.
- Measured on the real data: `internal.vtu` reads in **0.87 s** with `U` alone,
  0.94 s with `vorticity` as well. `cell_centers=False` saves ~0.06 s — the option
  is there to express "the mesh is static, I already have the lattice", not as a
  speed win. The 51 MB of geometry dominates and this reader cannot skip it (§7,
  I/O budget).

### No regridding, but a permutation ✅ DONE (2026-08-11)
No `LinearNDInterpolator` step is needed. The permutation of §2 is required instead,
and since the mesh is static it is computed **once** and reused for every timestep.
Implemented as `OpenFOAMData._build_lattice`, which **clusters** each coordinate into
levels rather than using the `np.rint` snap suggested in §2 — rint needs `dx`, i.e. a
uniform lattice, and the grid stops being uniform as soon as the boundary patches are
spliced on. Its completeness check (the linear index must be a permutation of `arange`)
is what verifies the rectilinear assumption for a given dataset.

### Boundary splice ✅ DONE (2026-08-11), but not by patch name

Faces are identified **geometrically** — by which interior axis range a patch's cells
fall outside of — rather than by the names `inlet`/`outlet`/`walls`. That splits
`walls.vtp`'s four planes correctly, keeps case-specific names out of the loader, and
means every face carries its own data instead of an assumed no-slip zero, so a future
case whose top is not a wall works unchanged.

⚠️ **The no-slip assumption for the edges was wrong, and only half-checked.** §5 says
the 12 edges and 8 corners "lie on no-slip walls, so filling zeros there is exact."
Measured through the loader: the **inlet** ring is exactly zero at every phase of the
pulse, but the **outlet** is an outflow BC that does not impose no-slip, and its
outermost ring runs to ~7e-4 against the walls' 0 — about 10% of local peak speed, over
272 of 832,320 cells. Edges are now filled with the average of the two adjoining faces
and warn when they disagree. Physically the wall should win (it is no-slip along its
whole length), but establishing that needs the BC types, which the VTK export does not
carry.

### I/O budget — relevant to the dyload goal

Per-timestep payload in `internal.vtu` (all inline base64, `format='binary'`,
**no compressor**, `header_type='UInt64'`):

| Array | MB (base64) |
|---|---|
| `Points` + `connectivity` + `offsets` + `types` | **51.1** |
| `U` | 12.4 |
| `vorticity` | 12.4 |
| `p`, `Q` | 8.3 |
| **total** | **84.2** |

**61% of every file is geometry that never changes.** Going through
`vtkXMLUnstructuredGridReader` re-reads all 51 MB per timestep to extract 12 MB of
velocity — the wrong trade for a streaming loader.

Options, roughly in order of effort:
1. `reader.GetCellDataArraySelection().DisableArray('vorticity'/'p'/'Q')` — recovers
   ~20 MB, but cannot skip connectivity.
2. Parse the XML directly for just the `U` `DataArray`: uncompressed inline base64
   with a `UInt64` byte-count header, so regex + `base64.b64decode` +
   `np.frombuffer` is straightforward and ~7× less I/O per step.
3. One-time preprocess to `.npy`/`.npz` per timestep. Cheapest at read time, given
   the mesh is static.

Recommendation: get the VTK path correct first; treat 2/3 as an optimization.

---

## Summary answers

| Question | Answer |
|---|---|
| Is the unstructured data actually on a grid? | **Yes** — uniform Cartesian 66×66×178 cells, all hexahedra. `UnstructuredGrid` is just OpenFOAM's generic container. |
| Do the sample points change in time? | **No** — bit-identical across all timesteps. |
| Is fluid specified inside the oral-arm disk? | **Yes** — everywhere, finite and nonzero (~2% of peak in the annulus, ~1e-8 in the stem). No mask, no NaN. |
| Regridding needed? | **No.** A one-time reordering permutation is, plus a boundary splice to recover the full domain extent. |
