# TODO — follow-ups from the test-suite overhaul

Context: the `tests/` suite was overhauled on branch **`mvbnd`** (2026-06) into
fast, deterministic modules. Default `pytest` run ≈1s; `pytest --runslow` adds the
parallelization checks (~30s). See the **Tests** section of `CLAUDE.md` for the
module map. Writing tests uncovered four latent defects; **three are now fixed**
(below), and the **one remaining** is pinned as a strict `xfail` so it flips to a
failure (forcing marker removal) the moment it is fixed.

Run the remaining bug's tracker, e.g.:
`pytest tests/test_analysis.py::test_backward_FTLE_produces_a_field -rx`

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

---

## Bug 1 — backward-time FTLE is missing and the documented workaround is wrong

- **Where:** `planktos/_environment.py`, `calculate_FTLE` (def at ~2877).
- **State today (verified):** forward FTLE works and is correct — steady-flow tests
  pin uniform→0 and simple-shear→`ln(φ)/T`; time-dependent forward also runs.
- **Two problems with "backward" FTLE:**
  1. **No working backward integration.** The solver is `while current_time < T:`
     starting at `current_time = t0` (line ~3040), i.e. forward only. Passing
     `T < 0` (the natural request) runs zero steps → `pos_history` is empty →
     `IndexError` at `self.FTLE_loc = s.pos_history[0]` (line ~3310). There is no
     `direction`/backward parameter.
  2. **The documented shortcut is mathematically incorrect.** Comments at
     `_environment.py:199` and `:371` say to negate `FTLE_smallest` for a
     "backward-time picture". But `FTLE_smallest = ln(√λ_min(C))/T` from the
     **forward** Cauchy-Green tensor `C = φᵀφ` (lines ~3297-3305). For an
     incompressible flow `det φ = 1 ⟹ λ_min = 1/λ_max ⟹ FTLE_smallest ≡
     −FTLE_largest` (verified exactly for the shear case), so negating it just
     returns the forward field — zero independent backward information. In general
     the smallest forward exponent is a contraction rate, **not** the backward-time
     FTLE (whose ridges are attracting LCS and live in different places).
- **Correct fix:** integrate the flow map backward in time (reverse the velocity
  field / integrate `t0 → t0−T`) and compute the FTLE from that backward flow
  map's Cauchy-Green tensor. Then remove/replace the "negate FTLE_smallest" guidance.
- **Tracker (xfail):** `tests/test_analysis.py::test_backward_FTLE_produces_a_field`
- **Also seen:** when no integration steps run (`t0 >= T`), the method crashes with
  the same empty-`pos_history` `IndexError` rather than erroring cleanly — worth a
  guard regardless of the backward-FTLE work.

---

## Broader follow-ups (not blocking, lower priority)

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
- See the xfail reasons: add `-rx`. A fixed bug will turn its xfail into an
  unexpected pass (XPASS) → suite failure (strict), signalling "remove the marker
  and keep the now-passing assertions."
- Regenerate IB2d fixtures after editing the generator:
  `python tests/fixtures/_gen_fixtures.py`.
