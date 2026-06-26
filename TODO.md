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
- **HIGHRE-2D** — `motion.highRe_massive_drift` hardcoded three spatial components
  (`np.stack((diff, diff, diff)).T`) and raised in 2D. Fixed with a
  dimension-agnostic broadcast (`diff[:, None]`); identical output in 3D.
  Regression test: `tests/test_agent_models.py::test_massive_particle_models_run_deterministically`
  now parametrized over 2D and 3D.
- **BUG-3D-SLIDE-STICK** — found while adding 3D collision tests. In the 3D
  project-and-slide, agents stuck at a shared triangle edge instead of sliding onto
  the adjacent triangle (broke coplanar tiled surfaces and concave folds; no
  penetration, but wrong sliding). Cause: when recursing onto the neighbour the
  edge-detector re-reported the just-crossed edge (seg_intersect_2D returns the
  start-on-edge touch) and `prev_idx` then stopped the agent — a 3D-only asymmetry
  (2D uses a distance test that ignores the endpoint it sits on). Fixed by deciding
  off `slide_pt` (did it overshoot the triangle? via `_point_in_triangle`) and
  taking the *last* edge crossed, mirroring 2D. Audited independently for
  no-penetration across single-triangle/ridge/corner/groove/coplanar/fold cases;
  2D path untouched. New tests: `tests/test_collisions_static_3d.py`.

---

## Broader follow-ups (not blocking, lower priority)

- **Full moving-immersed-boundary support in FTLE.** `calculate_FTLE` never advances
  `self.envir.time`, so a moving mesh would be frozen at its t0 position; it now
  raises `NotImplementedError`. A real fix threads the integration time into
  `interpolate_temporal_mesh` (forward, and backward for the reversed case) — a
  delicate change in the collision path. Static meshes already work in both directions.
- **Backward FTLE for non-tracer models** is intentionally unsupported (dissipative
  reverse-time integration blows up). If ever wanted, it needs a stabilized/adjoint
  approach, not naive negation.

- **conftest marker registration is duplicated.** `pytest.ini` now registers
  `slow`/`vtk`/`vtu`; `conftest.py::pytest_configure` still re-registers `vtk`/`vtu`.
  Harmless but redundant — drop the conftest copies for a single source of truth.
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
- **Golden trajectory locks for moving boundaries.** Current moving-collision
  coverage is invariant + single-step known answers. Consider one small pinned
  multi-step trajectory (extend `_ib_harness`) as a drift detector, per the hybrid
  plan.
- **Diffusion statistics test** (`test_agent_models.py::test_brownian_diffusion_statistics`)
  uses 20k agents with a fixed seed and ~10% tolerance. If it ever proves flaky,
  tighten the seed/agent count rather than the tolerance.

### Remaining test-coverage gaps (from the pre-3D triage)

- **Deeper recursive 2D sliding.** `test_collisions_static.py` covers a single
  convex L-corner and a concave V (one recursion each). Add a multi-element case —
  an agent driven into a narrowing wedge/corridor where the move vector is
  exhausted only after sliding across 3+ elements — to exercise the recursive
  project-and-slide more thoroughly (CLAUDE.md flags this as the most delicate path).
- **Material derivative** `Swarm.get_DuDt` / `get_dudt` have no known-answer test,
  yet inertial particles and FTLE depend on them. For a steady linear flow (e.g.
  `u=(a*y,0)`) `Du/Dt` has a closed form — add an exact check.
- **3D vorticity** `Environment.get_vorticity` (the 3D vector form) is untested;
  `test_analysis.py` only covers `get_2D_vorticity`. Solid-body rotation about an
  axis gives a known constant vorticity vector.
- **Periodic boundary × immersed boundary.** `_domain_BC_loop` re-checks IB
  collisions after wrapping an agent across a periodic boundary — a subtle
  interaction with no test (agent wraps and immediately meets a wall on the far side).
- **Swarm data-save round-trips.** `Swarm.save_data` / `save_pos_to_csv` /
  `save_pos_to_vtk` are untested (we round-tripped `Environment.save_fluid`). Easy
  write→read checks.
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
