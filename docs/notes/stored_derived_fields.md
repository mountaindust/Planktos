# Stored derived fields — reading a solver's own vorticity

**Status: plan, nothing built (2026-08-12).** Implements `TODO.md` Phase 2 robustness
item 6, whose decision was: read a stored derived field **on demand**, never carry one
in the sliding window. Consumer side is `flow_field_interface.md` §8.3.3.

---

## 1. The requirement

`get_vorticity` recomputes vorticity from the velocity field every time it is asked.
Several exports ship the solver's own vorticity, which is better (§6) and free of the
recomputation's boundary error. Three things follow:

1. **Vorticity must be retrievable without the velocity field.** A plot needs the
   backdrop, not the flow: rendering must not slide the velocity window, and ideally
   must not load velocity at all. This is the requirement that shapes the design.
2. **A request is by *time*, not by dump.** So a time has to be resolved to bracketing
   dumps, those read, and the result interpolated.
3. **It must be per-source.** OpenFOAM always ships vorticity; IB2d ships it only when
   the run asked for `Omega`; nothing else ships it.

---

## 2. What already exists

Most of the machinery is in place and the plan is mostly wiring.

| Piece | Where | State |
|---|---|---|
| IB2d scalar dump reader | `_dataio.read_2DEulerian_Data_From_vtk(path, numSim, 'Omega')` | **Works today.** `Omega` is already named in the reference comment block in `_read_IB2d_dumpfiles`; the scalar branch is exercised by `uX`/`uY` |
| OpenFOAM cell-array reader | `_dataio.read_vtkxml_cell_data(f, arrays=('vorticity',))` | **Works today**, interior and patches |
| Timeline | `FluidData.flow_times` | present for every source |
| Index → dump | `self.d_start + i` | uniform: IB2d's `d_start` is the first dump number, OpenFOAM's is 0 over a dense index into `self._dumps` |
| Per-dump sidecar precedent | `_record_dump_means` / `_interp_dump_means` | the pattern to copy, including its cubic/linear split (§6) |
| Interpolation weights | `LinearSpline.__call__` | `searchsorted` → two nodes → linear weight |

**Nothing new is needed in `_dataio`.** That is worth stating up front: the work is a
new sidecar object plus per-subclass hooks, not new file parsing.

---

## 3. Reading IB2d `Omega`

`Omega.####.vtk` sits beside `u`/`uX`/`uY` in the same viz directory, ASCII Structured
Points, one scalar per dump. Three transformations must match what the velocity path
does, or the field will not line up with `flow_points`:

1. **Transpose.** `read_2DEulerian_Data_From_vtk` returns data indexed `[y,x]`;
   `_read_IB2d_dumpfiles` does `.T` to get `[x,y]`. Same here.
2. **Restore the periodic endpoint.** IB2d omits the duplicate grid line at the top of
   each periodic dimension, so a 6×5 dump becomes a 7×6 field. ⚠️ **`_wrap_flow` cannot
   be reused for this** — it loops over `range(len(flow_points))` and so assumes one
   array per spatial dimension, i.e. a velocity field. Passing `[omega]` raises
   `IndexError`. Either generalize it or write the four-line scalar version
   (`_wrap_scalar` in the benchmark script).
3. **Do not re-shift the domain.** `flow_points` is already shifted to quadrant 1; the
   sidecar borrows it rather than recomputing.

⚠️ `IB2dData` does **not** store `dt` or `print_dump`. It does not need to: dump number
is `d_start + i`, and the times are `flow_times`. Do not add them as attributes to make
the sidecar work — the mapping already exists.

**Availability is a probe, not an assumption.** `tests/data/leaf_data` has `u` dumps
only, so the reference 2D dataset has no `Omega`. Decide availability by globbing
`Omega.*.vtk` once at construction, and check the count covers `d_start..d_finish` —
a partial `Omega` series is the same class of trap as a partial timeline (`TODO.md`
Phase 2), and should be treated the same way: refuse, or warn and fall back, but never
silently serve a field for the wrong dump.

---

## 4. Architecture

A **helper object owned by the `FluidData`** — one that sits alongside the velocity and
holds none of it. Concretely, two additions:

**A new method on `FluidData`,** which every source inherits:

```python
def get_stored_field(self, name, time):
    '''Return the named field as the source stored it, or None if it has none.

    name : 'vorticity'; time : a simulation time, not a dump number.
    '''
    return None                    # base class: no source ships anything
```

`OpenFOAMData` and `IB2dData` override it. Because the base returns `None`, every other
loader keeps working untouched, and `get_vorticity` can call it unconditionally.

**A small class holding the per-dump reads,** one instance per field name, stored on the
`FluidData` as `self._derived['vorticity']`:

```python
class _DerivedFieldReader:
    flow_times      # borrowed from the FluidData, never recomputed
    d_start         # so that dump number = d_start + index
    read_dump(i)    # supplied by the subclass: dump index -> field on the grid
    _cache          # {index: field}, at most two entries (see section 7)

    def __call__(self, time):
        # find the two dumps bracketing `time`, read whichever is not cached,
        # return the weighted blend
```

The reason to put `__call__` in a class rather than in each loader is that the bracket-
read-blend logic and the two-slot cache are identical for every source; only
`read_dump` differs. A subclass supplies that one function and nothing else.

Construction is **lazy** — built on the first `get_vorticity` call that wants it, so a
run that never plots vorticity never globs for `Omega` and never opens a file.

Construction is **lazy** — built on the first `get_vorticity` call that wants it, so a
run that never plots vorticity never globs for `Omega` and never opens a file.

---

## 5. Measured (2026-08-13) — and it changes the motivation

`tests/manual/bench_vorticity_sources.py`, against
`tests/data/Rubberband_with_Damped_Springs` (76 dumps, 33×33 after the wrap, **with**
`Omega`) and `tests/data/leaf_data` (149 dumps, 129×193, no `Omega`).

### 5.1 The 2D boundary error is a periodicity bug, not a discretization limit

`get_vorticity` does not respect `periodic_dim`. IB2d fields are periodic in both
directions and Planktos restores the duplicated end line, but `np.gradient` does not
know that and falls back to a one-sided difference at the array edge. Against IB2d's
own `Omega`:

| | all cells | edge ring | interior |
|---|---|---|---|
| `np.gradient` (today) | 1.7 – 2.9% | 5.0 – 8.4% | **0.00%** |
| differencing across the wrap | **0.00%** | **0.00%** | **0.00%** |

So for a periodic source the recomputed curl is *exactly* the solver's vorticity once
the edges wrap — there is no accuracy argument for reading `Omega` at all. This is the
2D counterpart of the OpenFOAM measurement in `TODO.md` item 6, and it localizes that
one too: **the error is boundary handling, not finite differencing.** OpenFOAM's case
is not periodic, so the fix there is different (its edge sits against a spliced
boundary-condition plane), but the diagnosis is the same shape.

**This is a bug worth fixing on its own**, independent of everything else here, and it
is a 1.0.3 candidate: every IB2d vorticity plot ever drawn has a wrong edge ring.

### 5.2 What is fastest

300 frames rendered across the whole series:

| grid | regime | vorticity from | seconds |
|---|---|---|---|
| 33×33, 76 dumps | `INUM=None` | in-memory cache | **0.001** (+0.00 build) |
| | `INUM=None` | recompute (today) | 0.015 |
| | `INUM=4` | disk (`Omega`) | 0.051 |
| | `INUM=None` | disk (`Omega`) | 0.057 |
| | `INUM=4` | recompute (today) | 0.152 |
| 129×193, 149 dumps | `INUM=None` | in-memory cache | **0.004** (+0.07 build) |
| | `INUM=None` | recompute (today) | 0.129 |
| | `INUM=4` | disk (`Omega`) | ≤ 4.64 (bounded by a `u` sweep) |
| | `INUM=4` | recompute (today) | 4.652 |

Per-call costs behind it (33×33 / 129×193): reading one `Omega` dump 0.58 ms / —;
velocity temporal interpolation 0.014 / 0.214 ms; the curl itself 0.022 / 0.101 ms;
`get_vorticity` end to end 0.037 / 0.335 ms; blending two fields 0.002 / 0.008 ms.

Three conclusions:

1. **If the velocity is resident, never go to disk.** Reading is 4× slower at 33×33 and
   an order of magnitude at leaf size. Disk is for when the fluid does *not* fit.
2. **An in-memory per-dump cache is the fastest thing available** — 15× and 30× faster
   than recomputing, because it replaces a spline evaluation plus two `np.gradient`
   passes with one blend of two arrays. Note it is also **cheaper than it sounds in 2D**:
   vorticity is a scalar where velocity is two components, so caching every dump costs
   **half** the velocity's memory, not double. (In 3D it is a 3-vector and the doubling
   argument from item 6 does apply.)
3. **Streaming is where disk wins, and only because recompute drags the velocity window
   behind it.** The 4.65 s at leaf size is almost entirely `load_dumpfiles`. Which
   means the honest framing is not "disk vs recompute" but **is the vorticity request
   driving the window, or riding along?**
   - *Post-hoc rendering* (the plot-cache case): nothing else needs velocity, so
     recompute pays for loads that exist only to be differentiated and thrown away.
     Disk wins.
   - *Live plotting during a simulation*: the window is being slid by the agents
     anyway, so the marginal cost of vorticity is just the curl — 0.1 ms. Recompute
     wins, and disk would add a read.

### 5.3 What this does to the plan

The IB2d `Omega` reader is **not** needed for accuracy (5.1) and **not** needed for
speed in the resident case (5.2). It earns its keep in exactly one place: a dataset too
large to hold, being rendered without a simulation running. That is a real case — it is
the case §8.3 exists for — but it is narrower than "IB2d support", and it is worth
deciding whether to build it now or after the cheaper wins.

Cheaper wins that this measurement surfaced, in order:

1. **Fix the periodic edge in `get_vorticity`** (5.1). Small, self-contained, fixes
   every existing IB2d vorticity plot, cherry-pickable.
2. **Have the plot cache store computed vorticity** rather than reading it back from
   the source, for 2D. Half the memory of the velocity, 30× faster to render, and after
   (1) it is exactly the solver's field anyway.
3. The disk reader, for the streaming-render case and for OpenFOAM/3D, where no
   periodicity fix can help because the boundary is a spliced BC plane.

---

## 6. The crux: which interpolation, and when it can be exact

§8.5 of `flow_field_interface.md` establishes that both splines evaluate as a weighted
sum of nodal fields, so for **linear** `F` — and vorticity is linear —
`F(u(t)) = Σᵢ wᵢ(t)·F(uᵢ)`. Serving stored vorticity with *the field's own weights* is
therefore exactly the curl of the velocity being used.

That is achievable in one regime and not the other, and the difference is not a detail:

| fluid regime | weights | nodes needed | exact? |
|---|---|---|---|
| `INUM=True` or `int` (linear) | `LinearSpline` | **2**, local | **yes** |
| `INUM=None` (cubic) | `fCubicSpline` | **all of them** | no — see below |

**A not-a-knot cubic spline is not local.** Its coefficients come from a global
tridiagonal solve, so the weights on any interval depend on every node. Matching them
would mean holding vorticity for the entire series — precisely the memory doubling item
6 rejected. The mean sidecar gets away with it only because a mean is three floats per
dump, so it can afford to keep them all (`_interp_dump_means` has exactly this
cubic/linear split, and its cubic branch is reachable only because everything was
resident).

So a two-dump sidecar is **exact whenever the fluid is splined linearly, and an
approximation when it is splined cubically.** Options, in the order I would consider
them:

- **(a) Linear always, and say so.** The discrepancy is bounded by the measured
  linear-vs-cubic gap (2D: 1.27% vs 0.54% of U_rms, Phase 1(C)), it applies to a
  *backdrop*, and it is zero at every dump time. Simple, one code path.
- **(b) Stored only when the fluid is linear; recompute when cubic.** Always exactly
  consistent with the velocity in use, but the vorticity a user sees then depends on
  `INUM`, silently — including whether it has the boundary error. I dislike this.
- **(c) Cache all dumps' vorticity when the fluid is cubic.** Exact everywhere, and
  reintroduces the doubling. Note the case is less absurd than it sounds: `INUM=None`
  already means the whole dataset is resident, so memory was not the binding constraint
  for that user. Still a doubling of it.

**Recommendation: (a).** State the approximation in the docstring, and note that it
vanishes at dump times, which is where a movie frame usually lands.

---

## 7. Read caching

A movie renders many frames per dump interval, so the naive path re-reads two files per
frame. **A two-slot cache keyed on dump index reduces that to one read per dump** for
any monotone sweep, forward or backward: consecutive frames share a bracketing pair, and
advancing evicts only the trailing one.

Two slots, not more: the whole point is to hold no more field data than the
interpolation needs. Keying on the *global* dump index (not a window-relative one) is
what lets it stay correct across a velocity-window slide it knows nothing about.

---

## 8. Where it plugs in

`FluidData.get_vorticity(time=, t_idx=)` currently computes unconditionally. New shape:

```
stored = self.get_stored_field('vorticity', time)
if stored is not None:
    return stored
... existing np.gradient path ...
```

Consequences to handle:

- **Return shape must match.** The computed 2D path returns a bare array; the 3D path
  returns a 3-tuple. A stored 3D field arrives as `(nx,ny,nz,3)` and must be split to
  match, or every consumer breaks.
- **`Environment.get_vorticity` needs no change** — it already resolves `t_indx` to a
  time before delegating.
- **`plot_all` needs no change**, which is the point of doing it here.
- **The OpenFOAM boundary splice applies.** Stored vorticity is cell data like `U`, so
  it needs the same reorder-and-splice `_read_dump` performs. Factor that assembly out
  of `_read_dump` rather than duplicating it — it is where the no-slip corner rule and
  the `require_boundary=False` regrid live, and a second copy would drift.

---

## 9. Fixtures and tests

- **IB2d**: add `Omega.####.vtk` to `ib2d_fluid_scalar_min/` via `_gen_fixtures.py`,
  with an analytic field distinct from `uX`/`uY` (so a wrong-array read cannot pass) and
  **deliberately not** the curl of the fixture's velocity — that is what proves the
  stored field is being served rather than recomputed.
- A second variant with `Omega` **absent** pins the capability probe, and one with a
  *partial* `Omega` series pins the refusal.
- **OpenFOAM**: `openfoam_min` already carries `vorticity = (z,y,x)`, steady and
  distinct from `U`. Nothing to add.
- Exactness test: at a dump time, the served field equals the file's field to
  round-off; between dump times, it equals the linear blend of the two.
- **Zero velocity loads.** Assert that serving vorticity across a whole movie's worth of
  times does not slide the velocity window — count `load_dumpfiles` calls, as
  `test_openfoam_the_mesh_check_costs_one_read_not_one_per_dump` counts reads.

---

## 10. Build order

1. `_DerivedFieldReader` + `FluidData.get_stored_field` returning None. No behavior change.
2. `IB2dData` hook: probe for `Omega`, closure reading one dump (transpose, `_wrap_flow`).
   Fixtures. Still not wired into `get_vorticity`.
3. Wire `get_vorticity` to prefer stored. 2D only at this step.
4. `OpenFOAMData` hook — requires factoring the splice out of `_read_dump` first.
   3D tuple-splitting.
5. `flow_field_interface.md` §8.3.3: the recorder queries `get_stored_field` and skips
   caching what it can get.

Steps 1–3 stand alone and deliver the IB2d case, which is the 2D plotting case and the
only one the plot cache currently touches.

---

## 11. Open decisions

1. **§5: (a), (b) or (c).** Recommendation (a).
2. **Partial `Omega` series** — refuse, or warn and recompute? The timeline work chose
   "warn loudly and carry on" for missing dumps; the same logic probably applies.
3. **Should `get_vorticity` gain an override** (`stored=True/False/None`) so a user can
   force recomputation for comparison? Cheap, and this is exactly the kind of switch
   that makes a discrepancy investigable — but it is also an option nobody sets.
