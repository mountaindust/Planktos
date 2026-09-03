# `test_data_streaming/` — acceptance tests for the four data-streaming claims

Adversarial acceptance tests written against four claims about run persistence,
not against the implementation that serves them. Written 2026-08-27 while the
component-C diff was still uncommitted; **no source was changed to make any of
them pass.**

The claims, in the user's words:

1. An existing example that creates or loads fluid flow and runs a simulation,
   all in RAM, must not break or change.
2. Fluid data can be dynamically loaded and `plot`/`plot_all` used as always,
   with the admitted cost that the dataset is streamed twice.
3. A run can be streamed to a structured directory on disk, and then plotted
   and filmed without re-streaming the fluid — faster still when the whole
   field is resident, since vorticity then comes from memory.
4. A run can be streamed out, both objects deleted, and the state recovered
   from disk into a new `Environment` and `Swarm` and resumed as if nothing
   had happened.

| Module | Claim |
|---|---|
| `test_stream_a_inram.py` | 1 — in-RAM runs behave as they always did |
| `test_stream_b_windowed_replay.py` | 2 — unrecorded windowed replay |
| `test_stream_c_recorded_replay.py` | 3 — recorded replay |
| `test_stream_d_restart.py` | 4 — checkpoint and restart |
| `_streaming.py` | the shared harness (world builders, I/O counters, `walk_frames`) |

Run with `pytest tests/test_data_streaming/`. Add `--runslow` for the example
scripts, the cross-version comparison and the ffmpeg movie renders.

---

## Verdicts

**Claim 1 — holds.** All eleven runnable example scripts complete.
Recording and plotting leave a run bit-identical, in 2D and 3D, with and
without an immersed boundary, and a run continues
bit-identically after being plotted. Separately, three in-RAM scenarios were run
under this working tree and under a `master` worktree and agree **bit for bit**
(`test_an_in_ram_run_gives_the_numbers_the_released_line_gives`).

**Claim 2 — holds.** A windowed replay's backdrop equals the fully resident one
exactly, costs at most one pass over the dumps, costs nothing at all with no
backdrop, leaves the run able to continue bit-identically, and says it is
re-reading — with the right count, since F2 was fixed.

**Claim 3 — holds.** Replaying a recorded windowed run costs zero velocity-dump
loads; recording plus replay costs exactly what the bare run cost; with the
field resident nothing is read at all. The values match a fully resident run of
the same simulation. The directory reads back in a fresh process, survives being
moved, and can be deleted after being plotted. The two defects found here (F3,
F4) were both fixed.

**Claim 4 — half built, and being worked.** It is component **R** in
`run_persistence.md` §2.11, scheduled ahead of tiling and built in four steps.
R1 and R2 have landed: the `Environment` rebuilds from the provenance record,
which now carries `char_L`, `U`, `nu` and `ibmesh_color` as well; agent
positions, velocities and the clock come back exactly; and a **checkpoint**
beside the archive carries everything that makes the *swarm* itself —
`rndState`, `props`, `shared_props`, `ib_condition`, `color`, and the Swarm
subclass name — read back through `RunArchive.checkpoint()`. What is left is
R3, the entry point that turns all of that into a live `Environment` and its
`Swarm`s. **Two xfails** in `test_stream_d_restart.py` mark it.

⚠️ The Environment side is *nearly* complete, not complete. An attribute-by-
attribute audit of a rebuild found five things `provenance['environment']`
(`{L, units, bndry, rho, mu}`) does not carry: **`char_L` and `U`** — which
`motion.inertial_particles` asserts are set, so an inertial-particle run cannot
be restarted at all — `nu` in the `Environment(nu=...)`-only construction, and
the cosmetic `ibmesh_color` and `plot_structs`. Not yet pinned by a test.

---

## Findings

Each was pinned by a `strict=True` xfail naming it, so the marker failed the
suite the moment the defect was fixed. **F2, F3 and F4 have since been fixed and
their markers cleared**; the tests stay as regression locks. **F1 has since been
fixed too** (2026-08-31). F5 stands.

### F1 (fixed) — `plot_all(fluid='vort')` raised on any time-invariant flow

`_swarm.py`, `plot_all`: the vorticity placeholder is sized from
`flow.fshape[1:]`. `fshape` carries a leading time axis **only for time-varying
data**, so for a steady flow this drops the *x* axis and hands `pcolormesh` a
1-D array — `ValueError: not enough values to unpack (expected 2, got 1)`.
Every analytic flow is time-invariant (`set_brinkman_flow` with scalar `U`,
`set_channel_flow`, `set_canopy_flow`), as is any `Environment(flow=[...])`
given no `flow_times`.

**Pre-existing, not a regression.** `master` v1.0.3 fails identically at
`_swarm.py:2549` with `flow[0].shape[1:]`; verified by running it in a worktree.
`Swarm.plot(fluid='vort')` is fine — only `plot_all` builds a placeholder. No
example hits it, which is why it survived so long.

**Fixed** by `FluidData.spatial_shape`, which names the concept once —
`fshape` when `flow_times is None`, `fshape[1:]` otherwise — and is read at
both plotting sites that built a placeholder. Logged in `TODO.md`'s
cherry-pick queue: the `master` port is a hand edit, since there is no
`FluidData` there to hang the property on.

Pinned: `test_stream_a_inram.py::test_plot_all_draws_vorticity_over_a_time_invariant_flow`,
which now also walks the frames — `plot_all` renders none on Agg, so the
original test would have exercised figure setup and stopped.

### F2 (fixed) — the re-streaming warning promised more dumps than exist

`_frames.warn_if_restreaming` counted the dumps a replay spans as
`searchsorted(hi,'right') - searchsorted(lo,'left') + 1`, one too many: over the
whole series, "will re-read about **9** of this dataset's **8** dumps".
Cosmetic, but the number is the entire content of the warning.

**Fixed** by `_frames._dump_span`, which the coverage check below shares — so
there is now one definition of which dumps a time range reads.

Pinned: `test_stream_b_windowed_replay.py::test_the_warning_counts_dumps_that_exist`

### F3 (fixed) — the quiver arrow scale was taken from `flow.fmax`, not the archive

`_frames.FrameSource._global_scales` set
`quiver_scale = norm(flow.fmax)`. `run_persistence.md` §3.5 says of `fmax`, in
as many words, that it is **not** usable for this — it covers all data seen so
far and grows during a run — and its "as built" text records the scale as
`norm(nanmax(dump_stats['vmax'], axis=0))`. `Swarm.plot`'s own comment says the
value comes "from the archive's per-dump extrema where there is one". The code
did neither.

The two agree for a run that swept the whole dataset, which is why it had not
shown up. They diverged as soon as the field sees anything the recorded stretch
did not: measured **13.88 → 42.45** for the same archive after one unrelated
`envir.flow(7.0)`. That is exactly the render-to-render drift §3.5 exists to
remove. **Fixed** in `_global_scales`, which now reduces `dump_stats['vmax']` where
there is a recording and keeps `fmax` only as the no-archive fallback. The
repo-side test that pinned the old behavior became
`test_archive_rendering.py::test_the_arrow_scale_comes_from_the_recorded_extrema`,
with a companion asserting `fmax` genuinely moved.

Pinned: `test_stream_c_recorded_replay.py::test_two_renders_of_the_same_run_share_an_arrow_scale`
(with `test_the_arrow_scale_is_the_one_the_archive_recorded` naming the wanted value)

### F4 (fixed) — no check that the frames to be drawn were covered by the recording

§2.8 refuses up front when the archive lacks a *quantity*. There was no
counterpart for *coverage*. Where a recording stopped before the run did and the
vorticity lives in the archive, the first frame past the last recorded dump
raises a bare `FileNotFoundError` from inside `_dataio` — partway through the
render, which for a long movie is after the expensive part. `RunArchive.quiver`
had a good message for the same situation; the vorticity path had none.

**Fixed** by `FrameSource._check_stored_coverage`, which runs while the source is
being built — so nothing is drawn before the refusal — and names the missing
dump range and three remedies. Verified not to fire on a complete recording in
either vorticity regime, on frames running past the end of `flow_times`, or on a
resident field.

Pinned: `test_stream_c_recorded_replay.py::test_frames_outside_the_recorded_dumps_are_refused_before_any_are_drawn`

### F5 (being worked) — claim 4 was unbuilt (five items, three closed)

`run_persistence.md` §2.11's table, made executable. The archive carried no
`rndState` (so a restart could not be reproducible), no `props` or
`shared_props` (so a restarted swarm was a default swarm on recorded
coordinates), no `ib_condition`/`color`, and there was no reader-side entry
point that turns an archive back into an `Environment` and a `Swarm`.

**R2 closed the first three** (2026-09-02) by writing a checkpoint beside the
archive — `agents/checkpointNN.npz` plus `_props.csv` and `_meta.json` — and reading it back through
`RunArchive.checkpoint()`. Those three tests were **retargeted rather than
merely un-xfailed**: they asserted that a string appeared in `meta.json`, which
is written once and never rewritten and so cannot hold state that changes. They
now assert the round trip instead, and are marked as scaffolding to delete once
Step R is done.

What remains is R3, the entry point itself, and the end-to-end test — which
attempts the most careful hand reconstruction today's public API allows and
still diverges from step one.

Pinned: the two remaining xfails in `test_stream_d_restart.py`

---

## One hazard worth knowing before writing more tests here

**`plot_all()` with no `movie_filename` renders no frames on Agg.**
`FuncAnimation` only calls its function on a draw event and `plt.show()` is a
no-op on a non-interactive backend, so "`plot_all` did not raise" exercises the
figure setup and nothing else — F4 above is invisible to it. `walk_frames` in
`_streaming.py` walks the frames through the same `FrameSource` that `animate`
uses; the slow tests render real movies through ffmpeg as the end-to-end check.
