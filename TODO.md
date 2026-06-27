# TODO — loose ends from the test-suite overhaul

Context: the `tests/` suite was overhauled on branch **`mvbnd`** (2026-06) into
fast, deterministic modules. Default `pytest` run ≈1s; `pytest --runslow` adds the
parallelization checks (~30s). See the **Tests** section of `CLAUDE.md` for the
module map and for the four latent defects that the overhaul found and fixed
(all now have regression tests; the suite has no xfails).

The items below are **non-blocking follow-ups** — relative priority only.

---

## Broader follow-ups

- **Full moving-immersed-boundary support in FTLE.** `calculate_FTLE` never advances
  `self.envir.time`, so a moving mesh would be frozen at its t0 position; it now
  raises `NotImplementedError`. A real fix threads the integration time into
  `interpolate_temporal_mesh` (forward, and backward for the reversed case) — a
  delicate change in the collision path. Static meshes already work in both directions.
- **Backward FTLE for non-tracer models** is intentionally unsupported (dissipative
  reverse-time integration blows up). If ever wanted, it needs a stabilized/adjoint
  approach, not naive negation.

- **COMSOL vtu test is skipped** (no committed data): `test_io_loaders.py::test_vtu_load`
  needs `tests/data/comsol/vtu_test_data.txt`. Either commit a tiny COMSOL fixture
  or leave gated.
- **3D collision coverage via real STL meshes.** `test_collisions_static_3d.py`
  now covers the 3D project-and-slide on hand-built triangle meshes (known answers +
  no-penetration). Still untested: loading an actual STL and running agents against
  it end-to-end (the seafan/convex-hull path), and 3D *moving* meshes (not
  implemented — the moving entry raises in 3D).
- **`motion.RK45` direct calling convention** is fragile (shape mismatch when used
  outside the swarm path). The agent-model tests use the documented
  `Euler_brownian_motion(self, dt, ode=...)` pattern instead. If `RK45` is meant to
  be public, give it a clear contract + a unit test; otherwise underscore it.
- **Diffusion statistics test** (`test_agent_models.py::test_brownian_diffusion_statistics`)
  uses 20k agents with a fixed seed and ~10% tolerance. If it ever proves flaky,
  tighten the seed/agent count rather than the tolerance.

### Remaining test-coverage gaps (from the pre-3D triage)

- **3D vorticity has no implementation on `mvbnd`.** There is no
  `Environment.get_vorticity` here — `get_2D_vorticity` hard-asserts 2D, and the
  only 3D vorticity (per the `get_2D_vorticity` comment) lives on `dyload`'s
  `FluidData.get_vorticity`. When `dyload` merges, add a known-answer test (solid-
  body rotation about an axis gives a constant vorticity vector). Not a test gap
  on this branch — there is nothing to test.
- **3D / mixed domain boundary conditions.** `test_swarm_lifecycle.py` tests
  zero/noflux/periodic in 2D only; 3D and mixed combinations (e.g. periodic-x,
  noflux-y) are exercised only indirectly via the IBAMR loader test.
- **Plotting smoke tests.** None of the `plot_*` methods are tested. A few
  Agg-backend "runs without error" smokes would cheaply catch the most common
  breakage (shallow, but plotting bugs are common).

## How to run

- Fast loop: `pytest` (≈1s, skips `slow`/`vtk`-absent/`vtu`-absent).
- Full: `pytest --runslow`.
- Regenerate IB2d fixtures after editing the generator:
  `python tests/fixtures/_gen_fixtures.py`.
