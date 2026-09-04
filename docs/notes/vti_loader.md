# A `.vti` / `.pvd` fluid loader — analysis and plan

Analysis of the 2D sea-fan dataset at `tests/data/openfoam2D/`, performed 2026-09-02
to determine whether Planktos can read it and what a loader would look like.
Everything below was verified by reading the files and running the existing loaders
against them, not inferred from the dataset's own README.

**The data is gitignored in its entirety** (`.gitignore:10`, `data/`), so this note is
what survives it — the same arrangement as `openfoam_oral_arm_dataset.md`. §7 is the
work list.

**Headline:** the data is clean, genuinely planar, and on a uniform Cartesian point
grid — everything Planktos wants. It is simply in a container we have no reader for.
`.vti` is VTK XML ImageData; every vtk entry point in `_dataio.py` is a *legacy*
reader. Bridging that gap is the whole job, and once bridged the rest of the branch's
machinery handles the data correctly, verified end to end.

---

## 1. What the dataset is

Pulsatile flow past a perforated plate (an idealized sea fan), OpenFOAM `pimpleFoam`,
resampled onto a Cartesian grid with `vtkProbeFilter` before it was shipped. From
Laura Miller, August 2026.

```
tests/data/openfoam2D/
├── flow/
│   ├── flow_0001_t0.1.vti … flow_0040_t4.vti    40 dumps, ~12 MB each, 472 MB total
│   └── flow.pvd                                 ParaView collection index
├── geometry/{plate.stl, plate_outline_2d.csv, plate_geometry.png}
├── metadata/{params.json, provenance.md, gap_timeseries.npz}
├── figures/, scripts/, README.md, flow_summary.npz
```

Every `.vti` is `<VTKFile type="ImageData">`, zlib-compressed binary:

| | |
|---|---|
| `WholeExtent` | `0 800 0 800 0 0` → **801 × 801 × 1 points**, zero cells thick in z |
| Origin / spacing | `(-0.02, -0.02, 0.0005)` / `5e-5` m, uniform in every direction |
| Extent covered | x, y ∈ [−20, 20] mm |
| Point arrays | `U` (3-comp Float32, declared `Vectors="U"`), `p`, `vtkValidPointMask`, `vtkGhostType`, `inFluid` |
| Cell arrays | `vtkGhostType` only |
| **FieldData** | **empty — no `TIME`, no `TimeValue`, in any dump** |

Verified across all 40 dumps: identical dimensions, origin, spacing and array
inventory; times uniform at 0.1 s; `flow.pvd`'s declared timesteps identical to the
`_t<value>` in the filenames.

**It is genuinely 2D.** `w` spans ±2.6e-17 — roundoff, not a slab of a 3D flow.

**Velocity is exactly zero inside the seven webs.** `inFluid` is 1 on 99.59% of
points; `vtkValidPointMask` marks exactly the same set, because the webs are omitted
blocks in the source mesh and so probe as invalid.

### `flow.pvd`

A ParaView "Collection" file — plain XML, no VTK needed to read it, the XML analogue
of the `.vtm.series` JSON index `OpenFOAMData` consumes:

```xml
<VTKFile type="Collection" version="0.1">
  <Collection>
    <DataSet timestep="0.1" group="" part="0" file="flow_0001_t0.1.vti"/>
    ...
```

Since the dumps carry no `TimeValue` of their own, **this is the only clean timeline
source in the dataset.** The filenames encode the same times, but lossily
(`flow_0010_t1.vti` for t = 1.0), so the `.pvd` is authoritative and the filenames are
a fallback.

### Two facts the dataset's README does not foreground

- **The export is a cropped core window, not the whole domain.** `params.json` gives
  the OpenFOAM domain as [−50, 50] × [−40, 40] mm; the `.vti` covers ±20 mm.
- **There is no t = 0 dump.** The series starts at 0.1 s, so Planktos environment time
  0 corresponds to physical t = 0.1 s. The README's guidance to discard the startup
  pulse (t > 2 s) therefore means environment t ≥ 1.9.

---

## 2. What Planktos reads today

| Format | Reader | Reads this? |
|---|---|---|
| legacy `.vtk` `RECTILINEAR_GRID` | `read_vtk_Rectilinear_Grid_Vector` (`vtkRectilinearGridReader`) | no — legacy only |
| legacy `.vtk` `STRUCTURED_POINTS` | `read_vtk_Structured_Points` (`vtkStructuredPointsReader`) | no — legacy only |
| XML `.vtu` / `.vtp` | `read_vtkxml_cell_data`, `read_vtu_Unstructured_Grid_Points_FEM` | different dataset type |
| `.vtm` / `.vtm.series` | `read_vtm_manifest`, `read_vtm_series` | different container |
| **XML `.vti`** | — | **nothing** |
| **`.pvd`** | — | **nothing** |

Verified failures:

- `envir.read_IBAMR3d_vtk_data(<a .vti>)` — vtk logs
  `Unrecognized file type: <?xml version="1.0"?>`, then `AttributeError` on the
  `None` returned by `GetXCoordinates()`. `VTK3dData`'s `<title>###.vtk` filename
  convention would not parse `flow_0001_t0.1.vti` in any case.
- `envir.read_openfoam_vtk_data('.../flow')` — `FileNotFoundError`, no `.vtm.series`,
  `.vtm` or `internal.vtu`.

**`OpenFOAMData` is the wrong tool, not a near miss.** It exists to recover a lattice
from unordered finite-volume cell data and splice half-cell boundary patches onto it.
The collaborator already ran `vtkProbeFilter`: this is point data on a known uniform
grid with the domain boundaries included. None of that machinery applies. The right
ancestor is `VTK3dData` — iterate files, append per component, stack.

---

## 3. Everything downstream already works

Proved by conversion: the first 8 dumps were written out as legacy rectilinear-grid
vector `.vtk` (via `_dataio.write_vtk_rectilinear_grid_vectors`) in scratch outside
the repo, then loaded with `read_IBAMR3d_vtk_data`.

- **The flat-axis collapse does the right thing.** `_collapse_flat_axes` drops z and
  the w-component: `flow.ndim == 2`, `L == [0.04, 0.04]`, spatial shape `(801, 801)`,
  `fluid_domain_LLC == (-0.02, -0.02)`.
- **Values are exact.** `envir.flow(0.2)[0]` against the source `flow_0003_t0.3.vti`:
  max difference 1.7e-18. Grid points match to round-off.
- **Streaming works.** `INUM=4` gave `is_windowed == True`; the window slid
  `(0, 4) → (3, 7)` on demand, and a 50-agent `Swarm.move()` loop ran across the slide.

So the loader is the only missing piece. It is not a question of whether the branch
can carry this data.

### Regime to recommend: load it all

| | |
|---|---|
| On disk | 472 MB compressed |
| Resident, 2 components float64 | 801 × 801 × 40 × 2 × 8 B = **411 MB** |
| With cubic spline coefficients | ~2 GB |
| Read time, all 40 dumps | ~5 s (0.126 s/dump measured) |

`INUM=None` is free on any machine that would run this, and buys cubic in time rather
than linear — which matters here, because 0.1 s dumps over a T = 2 s period is only 20
samples per pulse. Streaming should still be *exercised* against this dataset, since
it is the branch's feature and this is the first real 2D data to test it on, but it is
not what a user should choose.

---

## 4. Design of the loader

### Structure

A new `FluidData` subclass, not an extension of `VTK3dData`. `VTK3dData` is bound to
one filename convention and one legacy reader; teaching it a second of each would put
two dump-discovery schemes and two low-level readers in one class. `OpenFOAMData` is
the precedent for a source-specific class with its own discovery chain.

Proposed names, following the `vtkxml` prefix `_dataio` already uses:

| Layer | Name |
|---|---|
| low-level read | `_dataio.read_vtkxml_image_data(filename)` |
| timeline index | `_dataio.read_pvd_series(filename)` |
| `FluidData` subclass | `fluid.VTKXMLData` |
| `Environment` method | `read_vtkxml_fluid_data` |

### Dump discovery and the timeline

Same shape as `OpenFOAMData._find_dumps`: try sources in turn, record which one
answered in `dump_source` / `time_source`, warn on every step past the first.

| # | Source | Times from |
|---|---|---|
| 1 | the `.pvd` collection | the `timestep` it declares |
| 2 | glob of `*.vti`, ordered by `_natural_key` | `TimeValue` field data, if present |
| 3 | same glob | the `_t<value>` suffix in the filename |
| 4 | — | unit steps |

Source 3 is specific to this dataset's naming and should be opt-in (a `time_from_name`
regex parameter, default off) rather than a silent guess — a filename is not a
timestamp, and inferring one that happens to parse would be exactly the "run completes
on a timeline other than the one the user believes they loaded" failure `OpenFOAMData`
is written to avoid. For *this* dataset the `.pvd` answers, so 3 never fires.

Reuse from `OpenFOAMData` verbatim: `_natural_key`, the dense 0-based dump index over
the dumps that actually exist, the declared-but-absent skip with a warning, and the
non-uniform-spacing warning. Those are general, and lifting them into module-level
helpers is preferable to a third copy.

`flow_times` must span the **entire** dump series before any fluid is loaded, not the
opening window — the frozen-timeline bug in `TODO.md` Phase 2. Here that is cheap:
the `.pvd` gives every time in one small read.

### The low-level read

`vtkXMLImageDataReader`, verified against these files:

- It honors the `Vectors="U"` attribute, so `GetPointData().GetVectors()` returns `U`
  without the reader having to know OpenFOAM's naming. Keep a `vec_name` parameter for
  a file that declares no active vectors.
- `GetPointDataArraySelection().DisableArray(...)` works and drops the unwanted
  arrays. **It saves little here** — 0.103 vs 0.126 s/dump, because zlib
  decompression dominates and `p`/the masks compress well. Worth the one line, not
  worth designing around.
- Origin, spacing and dimensions come off the output directly; the coordinate arrays
  are `origin[d] + spacing[d] * arange(n[d])`. `ImageData` is uniform by construction,
  so the rectilinearity check `OpenFOAMData._build_lattice` performs is unnecessary.
- Point ordering is x-fastest, so the reshape is `(nz, ny, nx)` then transpose to
  `[x, y, z]` — identical to `read_vtk_Rectilinear_Grid_Vector`.

`.vtr` (XML RectilinearGrid) is the same function with `vtkXMLRectilinearGridReader`
and explicit coordinate arrays, and is worth taking in the same pass since a `.pvd`
may name either. `.vts` (StructuredGrid) is curvilinear and stays out of scope —
Planktos interpolates on a tensor-product grid.

The `.pvd` parser needs no VTK: `xml.etree.ElementTree`, iterate `DataSet` elements,
resolve `file` relative to the `.pvd`'s own directory. It should carry the dataset
extension through so the subclass can refuse a `.pvd` naming file types it cannot
read, rather than failing at the first load.

### Flat axes and the mesh

`_collapse_flat_axes` at construction, remembering the dropped axes in `self._flat`
and reapplying them with `_drop_flat_axes` in `load_dumpfiles` — exactly as
`VTK3dData` does. This is what turns the 801 × 801 × 1 grid into 2D data, and it is
already verified working (§3).

The grid is read once and reused, as in `OpenFOAMData`: the mesh does not move. Check
the second dump's dimensions/origin/spacing against it and raise on a change, which is
the cheap analogue of `_verify_dump_mesh` and needs no coordinate comparison.

### What the subclass must supply beyond `load_dumpfiles`

- `path`, so the inherited `source_dir` works.
- `d_start = 0`, `d_finish = len(dumps) - 1`, dense over the dumps that exist, so the
  inherited `dump_number` is right and `update_spline`'s index arithmetic holds.
- **Per-dump vorticity: inherit, do not override.** `probe_stored_vorticity` returning
  `'absent'` is correct — these files carry no vorticity — and the base
  `read_dump_vorticity` / `write_dump_vorticity` pair writes rectilinear-grid scalar
  vtk, which expresses this grid fine. Writing `.vti` sidecars instead would be
  matching the source's format for its own sake; revisit only if the collaborator
  starts shipping vorticity.
- The `@_provenance.records_provenance('_fluid_provenance')` decorator on the
  `Environment` method, plus the new name added to the loader list in
  `tests/test_provenance.py` — that list is explicit and will not notice on its own.

---

## 5. Two defects this uncovered

Both are `dyload`-only (`read_vtk_time_only` and `_collapse_flat_axes` are absent from
`master`), so neither needs a cherry-pick queue entry.

### `read_vtk_time_only` cannot read binary legacy vtk

Its docstring claims "The header of a legacy VTK file is ASCII even when the data that
follows is BINARY." That is false for `FIELD` data *values*: in a binary legacy vtk
the `TIME 1 1 double` declaration is ASCII but the number on the next line is raw
bytes, so `float(lines[n+1].split()[0])` fails and the function returns `None` for
every dump. `VTK3dData._read_all_times` then falls back to a **full parse of every
file** — the exact cost the function exists to avoid.

Nothing catches this: every committed fixture is written `binary=False`
(`tests/fixtures/_gen_fixtures.py`), so the binary path is unexercised. And
`write_vtk_rectilinear_grid_vectors` defaults to `binary=True`, so Planktos's own
writer produces files its own fast-scanner cannot read.

The consequence scales badly in exactly the direction this branch exists to serve: on
a 100 GB IBAMR series, building the timeline would read the entire dataset.

Fix: parse the value as bytes rather than text — locate the `TIME 1 1 <type>` line in
the byte stream and unpack the following 4 or 8 bytes big-endian (legacy vtk binary is
big-endian) when the file declares `BINARY`. Test with an ASCII and a binary fixture
of the same field; the ASCII-only fixture set is the reason this survived.

### The flat-axis warning has no tolerance

`_collapse_flat_axes` decides whether a dropped component is real with
`np.any(flow[d] != 0)` — exact. This dataset's `w` is ±2.6e-17, so genuinely 2D data
is reported as "a slab of a 3D flow rather than 2D data" with a warning telling the
user their z-velocity is being discarded. Any real solver output will trip this.

Fix: compare against a scale drawn from the retained components, e.g. flag only when
`max|w|` exceeds a small multiple of the in-plane velocity scale, and say the ratio in
the warning so the user can judge it.

---

## 6. Beyond the loader

These are not loader work, but the loader is useless without answers, and the first
three want a question put to the collaborator.

1. **The cropped domain.** Planktos will take `L = [0.04, 0.04]` and apply the
   environment's boundary conditions at ±20 mm, where the real flow continues. With
   the default `'zero'` BC agents are simply removed there. Whether that is acceptable
   depends on the study; a re-export covering the full [−50, 50] × [−40, 40] mm domain
   would remove the question, and the collaborator offers more output.
2. **The geometry does not fit the environment.** `plate.stl` is a 3D extrusion (84
   triangles, z ∈ [0, 1] mm); `read_stl_mesh_data` yields an N×3×3 ibmesh, which is
   the wrong dimensionality for the 2D environment this flow produces. The usable file
   is `plate_outline_2d.csv` — 7 webs × 4 corners — for which there is no loader. The
   options are to set `envir.ibmesh` directly as a (28, 2, 2) array, or to emit a
   `.vertex` file and use `read_IB2d_mesh_data` with `brk_idx_list`/`add_idx_list` to
   close each web and break between them. A small CSV-outline path may be worth having
   if more datasets arrive shaped this way.
3. **`inFluid` is unused.** Planktos ignores it, so without an ibmesh agents drift
   into the webs and find zero fluid velocity rather than a wall. The mask is exactly
   the plate interior and could in principle seed a boundary, but deriving segments
   from a raster is a worse route than the CSV corners already provided.
4. **SI metres bite.** In the §3 test run all 50 agents left the 4 cm domain within 14
   steps at dt = 0.05, because the default `shared_props['cov']` is enormous at this
   scale. Any example script built on this dataset must scale it; worth a line in the
   example's header, since every dataset in these units will hit it.

---

## 7. Work list

1. `_dataio.read_pvd_series` — parse the collection, return `(time, path)` in declared
   order with the dataset extension carried through. No VTK.
2. `_dataio.read_vtkxml_image_data` — `vtkXMLImageDataReader`, active vectors with a
   `vec_name` override, array selection, returning components indexed `[x,y,z]` plus
   coordinate arrays and `TimeValue` if present. `.vtr` alongside it.
3. `fluid.VTKXMLData` — discovery chain (§4), grid read once and checked on the second
   dump, `_collapse_flat_axes` at construction and `_drop_flat_axes` in
   `load_dumpfiles`, dense `d_start`/`d_finish`, inherited vorticity.
4. Lift `_natural_key`, the absent-dump skip and the non-uniform-spacing warning out of
   `OpenFOAMData` into module-level helpers rather than copying them a third time.
5. `Environment.read_vtkxml_fluid_data`, decorated with `records_provenance`; add the
   name to the loader list in `tests/test_provenance.py`.
6. Fixture: a `vtixml_min/` series in `_gen_fixtures.py` — a handful of `.vti` dumps
   one point thick in z, indexed by a `.pvd`, with a declared-but-absent dump and one
   dump carrying `TimeValue` and one not, so the fallback chain is exercised. Fields
   analytic, as elsewhere (`u = t` reads the timeline back).
7. Tests: values and grid against the fixture; the flat-z collapse; the full timeline
   spanning the series rather than the opening window (the frozen-timeline lock, as in
   `test_dynamic_loading.py`); each step of the fallback chain and its warning; a
   `.pvd` naming an unreadable type refusing at construction.
8. Fix `read_vtk_time_only` for binary legacy vtk, with a binary fixture (§5).
9. Give the flat-axis warning a tolerance (§5).
10. Changelog under `1.1.0`: one line for the new format support. The two fixes in 8
    and 9 are `dyload`-only regressions in unreleased code and do not go in.
11. Once it loads: run `pytest --runslow --runstreaming`, then exercise this dataset
    both in-RAM and windowed, and compare — the 2D analogue of the Phase 1(C)
    linear-vs-cubic measurement, now on real 2D data rather than a synthetic field.

---

## Summary answers

| Question | Answer |
|---|---|
| Do we read `.vti`? | **No.** Every vtk reader in `_dataio.py` is legacy; `.vti` is VTK XML ImageData. |
| Do we read or use `.pvd`? | **No.** It is a ParaView collection index — and here the only clean source of times, since the dumps carry no `TimeValue`. |
| Is the data on a grid Planktos can use? | **Yes** — uniform Cartesian, 801 × 801 points, identical in every dump, domain boundaries included. |
| Is it really 2D? | **Yes** — one point thick in z, `w` at round-off. |
| Can it be loaded once bridged? | **Yes, verified** — bit-exact values and grid via a legacy-format conversion, in-RAM and windowed alike. |
| Load all at once, or stream? | **All at once.** 411 MB resident, ~2 GB splined, ~5 s to read — and cubic in time is worth having at 20 samples per pulse. |
