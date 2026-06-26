# TODO — follow-ups from the test-suite overhaul

Context: the `tests/` suite was overhauled on branch **`mvbnd`** (2026-06) into
fast, deterministic modules. Default `pytest` run ≈1s; `pytest --runslow` adds the
parallelization checks (~30s). See the **Tests** section of `CLAUDE.md` for the
module map. Writing tests uncovered four latent defects; **all four are now
fixed** (below) with regression tests, so the suite has **no remaining xfails**.
A few non-blocking follow-ups (some surfaced while fixing the four) are listed at
the end.

---

## Fixed (2026-06)

- **BUG-STICKY-AXIS** — sticky moving-boundary collisions returned NaN for
  perfectly axis-aligned (vertical/horizontal) moving elements. `_ibc.py`
  computed the contact parameter as `max(ratio_x, ratio_y)`; for a degenerate
  axis one ratio was `0/0 → NaN` and `max` propagated it. Fixed by taking the
  ratio on the axis with the largest extent (`np.argmax(np.abs(Q1-Q0))`).
  Regression test: `tests/test_collisions_moving.py::`
  `test_sticky_moving_axis_aligned_wall_stops_on_wall` (vertical + horizontal).
- **BUG-ZEROLEN-SEG** — `_geom.closest_dist_btwn_lines_and_pt` raised
  `ValueError` on any mix of zero-length and normal segments
  (`seg_lengths_2[~z_check] = seg_lengths_2` → `seg_lengths_2 = seg_lengths_2[~z_check]`).
  Regression tests: `tests/test_geom.py::test_closest_dist_lines_and_pt_mixed_zero_length`
  and `::test_closest_dist_lines_and_pt_all_zero_length`.
- **BUG-SAVEFLUID** — `Environment.save_fluid` / `save_2D_vorticity` were broken on
  modern pyvista, three layers deep: the writers set the now-forbidden
  `.origin`/`.dimensions` on a `RectilinearGrid`; the save methods passed `self.L`
  (domain lengths) where coordinate arrays were needed; and static flows
  (`flow_times is None`) died in `interpolate_temporal_flow`. Fixed by removing the
  invalid grid attributes (a RectilinearGrid's geometry is its coordinate arrays),
  passing `self.flow_points`, and short-circuiting static flow to write `self.flow`
  directly. Saved coordinates are origin-centered (LLC convention untouched).
  Regression tests: `tests/test_io_loaders.py::test_save_fluid_static_2D_roundtrips`,
  `::test_save_fluid_time_varying_2D_roundtrips`, `::test_save_fluid_static_3D_roundtrips`,
  `::test_save_2D_vorticity_static_roundtrips`.
- **BUG-FTLE-BACKWARD** — `Environment.calculate_FTLE` only integrated forward
  (`T<0` raised `IndexError`), and the documented "negate `FTLE_smallest` for
  backward time" was mathematically wrong (it is identically `−FTLE_largest` for
  incompressible flow). Added a `backward=True` option that computes the true
  backward-time field as the forward integration of the reversed flow (wrap the ODE
  to sample velocity at the mirrored time `2*t0−t` and negate), reusing the existing
  trajectory/eigenvalue machinery; corrected the `FTLE_smallest` docstrings and the
  `plot_2D_FTLE` text. Scoped to tracer particles (reverse-time inertial/custom
  dynamics are dissipative and ill-posed); moving meshes and `T≤t0` now raise
  clearly. Regression tests: `tests/test_analysis.py::test_backward_FTLE_*` and
  `::test_FTLE_forward_vs_backward_differ_time_dependent_shear` (direction-discriminating,
  closed-form) plus the guard tests.

---

## Broader follow-ups (not blocking, lower priority)

- **Full moving-immersed-boundary support in FTLE.** `calculate_FTLE` never advances
  `self.envir.time`, so a moving mesh would be frozen at its t0 position; it now
  raises `NotImplementedError`. A real fix threads the integration time into
  `interpolate_temporal_mesh` (forward, and backward for the reversed case) — a
  delicate change in the collision path. Static meshes already work in both directions.
- **`motion.highRe_massive_drift` is 3D-only.** It hardcodes three spatial dims
  (`np.stack((diff, diff, diff))` ~`motion.py:502`) and raises in 2D. Generalize to
  the environment's dimension.
- **Backward FTLE for non-tracer models** is intentionally unsupported (dissipative
  reverse-time integration blows up). If ever wanted, it needs a stabilized/adjoint
  approach, not naive negation.

- **conftest marker registration is duplicated.** `pytest.ini` now registers
  `slow`/`vtk`/`vtu`; `conftest.py::pytest_configure` still re-registers `vtk`/`vtu`.
  Harmless but redundant — drop the conftest copies for a single source of truth.
- **COMSOL vtu test is skipped** (no committed data): `test_io_loaders.py::test_vtu_load`
  needs `tests/data/comsol/vtu_test_data.txt`. Either commit a tiny COMSOL fixture
  or leave gated.
- **3D immersed-boundary collisions** (`apply_internal_static_BC` 3D path,
  `seg_intersect_3D_triangles` is unit-tested in `test_geom.py`, but the 3D
  collision/slide response in `_ibc` has no direct no-penetration tests). Add a 3D
  analogue of `test_collisions_static.py` (STL/convex-hull mesh, known answers).
- **`motion.RK45` direct calling convention** is fragile (shape mismatch when used
  outside the swarm path). The agent-model tests use the documented
  `Euler_brownian_motion(self, dt, ode=...)` pattern instead. If `RK45` is meant to
  be public, give it a clear contract + a unit test; otherwise underscore it.
- **Golden trajectory locks for moving boundaries.** Current moving-collision
  coverage is invariant + single-step known answers. Consider one small pinned
  multi-step trajectory (extend `_ib_harness`) as a drift detector, per the hybrid
  plan.
- **Diffusion statistics test** (`test_agent_models.py::test_brownian_diffusion_statistics`)
  uses 20k agents with a fixed seed and ~10% tolerance. If it ever proves flaky,
  tighten the seed/agent count rather than the tolerance.

## How to run

- Fast loop: `pytest` (≈1s, skips `slow`/`vtk`-absent/`vtu`-absent).
- Full: `pytest --runslow`.
- Regenerate IB2d fixtures after editing the generator:
  `python tests/fixtures/_gen_fixtures.py`.
