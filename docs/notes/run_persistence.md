# Run persistence, plot streaming, and tiling — design and implementation plan

Status: **working plan.** Components A–D below; each carries its own **[done]**
markers. Started 2026-07 as "plotting streaming" (§8 of the now-deleted
`flow_field_interface.md`), reframed 2026-08-18 around data persistence, which is
what the plan had quietly become.

**This note supersedes and replaces `docs/notes/flow_field_interface.md`,** which was
deleted once everything still load-bearing in it had been folded in here. That note
recorded the `FlowArray` analysis and removal — work that is complete and needs no
further reference; Appendix A summarizes what it concluded and where the code now
stands. Its git history holds the deliberation behind every decision restated here,
including the options weighed and rejected:
`git log --follow -- docs/notes/flow_field_interface.md`.

This note lives with the other `docs/notes/` design notes. It is the source of truth
for *why* run state is persisted the way it is on `dyload`; keep it current as the
work lands.

---

## 0. Orientation — read this first

### 0.1 The problem, in one paragraph

Planktos holds an entire run in memory and can only write it out at the end.
`Swarm.pos_history` and `vel_history` grow one masked `N×D` array per step forever;
the three save methods (`save_data`, `save_pos_to_csv`, `save_pos_to_vtk`) all
require the whole history resident, and `save_pos_to_csv` additionally materializes
a dense text copy of it in a single `np.savetxt` call. **Nothing reads any of them
back** — there is no loader for any Planktos output anywhere in the package, so a run
cannot be reloaded, only re-run. Meanwhile `plot_all` replays a finished run by
pulling fluid data at every frame, which under dynamic loading re-streams the entire
dataset a second time. These look like two problems. They are one: **run state has
nowhere to go except memory.**

The symmetry worth holding onto: **dynamic loading streams the fluid *in*; this
streams the agents *out*.** Same architecture, opposite direction, same reason — the
thing is too big to hold at once.

### 0.2 The components

| | What | Why | Status |
|---|---|---|---|
| **A** | **Run archive** — append-only, chunked, crash-valid on-the-fly capture of agent state, with a public reader and a capture schedule that also governs history retention | persistence: crash survival, later sessions, larger-than-RAM analysis, bounded history memory, run speed, and eventually restart | **[done]** — §6.1 A0–A5, 2026-08-21 to 2026-08-25 |
| **B** | **Fluid-side streaming** — per-dump means, vorticity by regime, whole-run extrema | dyload: never re-stream the dataset to draw a picture of it | **[done]** — §6.1 B1–B3, 2026-08-25 |
| **C** | **Rendering** — frame selection by time, archive-backed `plot_all`, global colour/arrow scales | consumes A and B | **[done]** — §6.1 C1–C2, 2026-08-27 |
| **R** | **Full-state reboot** — a checkpoint beside the archive, a reader that turns a directory back into an `Environment` and its `Swarm`s, and appending to the archive a run resumed from | the third problem this architecture solves, and the one A was built for: a run that outlives the process that made it | **in progress** — §6.1 R0–R3 done 2026-08-31 to 2026-09-03; R4 and R5 ahead |
| **D** | **Tiling and `extend`** — the real position-wrapping implementation | cleanup: tiling has raised `NotImplementedError` since the `FlowArray` removal | specified (§9), not built |

⚠️ **Two lettering schemes overlap, and the letters do not agree.** The components here
are A, B, C, R, D; the build steps in §6.1 are Step 0, A, B, C, R, D — and **Step D is
the prose pass (§7), not component D (tiling, §9)**. That collision predates this note's
current shape. When a sentence says "D", check which list it is counting.

**A and B are independently shippable and should be shipped independently.** B is a
dyload optimization that writes nothing at all under `INUM=None` — i.e. it does
nothing for the majority of users. A is a persistence feature everyone benefits from.
The original plan interleaved them into one build step; that entanglement was an
artifact of both being called "the cache", not a dependency.

**Two standing scope rules, from `CLAUDE.md`, that constrain B and C throughout.**
All plotting is matplotlib today, and the 3D plotting is explicitly a placeholder
awaiting a vtk-powered library, at which point 3D plotting **splits out entirely**
from the 2D path. Therefore: **do not invest in matplotlib 3D rendering** — effort
spent enriching 3D frames is written off at the rewrite — and **do not contort 2D
designs to stay symmetric with 3D**, since shared abstractions spanning both would
only have to be unpicked at the split. 2D and 3D diverging is the intended direction.
Concretely: everything in B and C beyond the frame statistics is **2D-only**, and the
statistics fix (§3.1, done) was the entire 3D deliverable. **A is dimension-agnostic**
— agent state is agent state — and is the one component of this plan 3D users get in
full.

### 0.3 What to do next

In order:

1. ~~**§5 — the prerequisite bug fixes.**~~ **[done]** §5.1 and §5.2 on 2026-08-19,
   §5.3 on 2026-08-21. Each settled state the archive was going to store, so they came
   before anything wrote it to disk.
2. ~~**§2 — build the run archive.**~~ **[done]** A0 landed 2026-08-21 (the movement
   start point now comes from `Swarm._prev_positions` rather than `pos_history[-1]`, so
   a capture schedule can no longer reach collision detection), A1–A5 by 2026-08-25.
   Agent state streams to disk as a run proceeds (`Environment.record`), survives a hard
   kill, and reads back through `planktos.load_run`.
3. ~~**§3 — fluid-side streaming.**~~ **[done 2026-08-25]** The archive now carries
   the fluid half: per-dump statistics always, vorticity by regime, quiver on
   request.
4. ~~**§4 — rendering.**~~ **[done 2026-08-27]** `Swarm.plot`/`plot_all` take
   `archive=` and read every frame through `planktos/_frames.py`; replaying a windowed
   run costs zero loader calls; the colour and arrow scales are global;
   `Environment.record(plot_all=)` renders from the archive at the end of a `with`
   block. **This was the last step that consumes A and B.**
5. **§2.11 — the full-state reboot** is **in progress** *(scheduled 2026-08-27; R0–R3
   done, R4 and R5 ahead — §6.1)*. It was filed as
   a follow-on until the acceptance suite made the gap concrete; §6.1's Step R says why
   it goes ahead of tiling rather than after it, and the one-line version is that the
   on-disk format has to grow and every archive written before it does cannot be
   rebooted.
6. **§9 — tiling** is the cleanup after that. It has its own restoration checklist
   (§9.3) because gating it off left notices scattered across source, tests, examples,
   docs and prose. The §7 prose pass rides on it.

⚠️ **Branch-level priority this note cannot see:** check `TODO.md` before assuming
"next section in this note" means "next work to do."

---

## 1. How this plan got here, and why it changed shape

Kept because the reframe is not obvious from the specification that follows, and
because the original framing is still visible in commit messages, `TODO.md`, and
several source comments.

### 1.1 The original problem: plotting re-streams the fluid

`Swarm.plot_all`'s `animate(n)` is a random-access replay over `pos_history`. Each
frame pulled fluid data at `envir.time_history[n]`:

| Per-frame call | Cost | Applies to |
|---|---|---|
| `_calc_basic_stats(t_indx=n)` → `interpolate_temporal_flow` | **full field**, then `.mean()`/`.max()` | 2D **and 3D**, unconditionally |
| `get_vorticity(t_indx=n)` (`fluid='vort'`) | full field + `np.gradient` | 2D only |
| `interpolate_temporal_flow(t_index=n)` (`fluid='quiver'`) | full field, then `[::M,::N]` | 2D only |
| `interpolate_temporal_mesh(...)` | mesh only, cheap | moving meshes |

Under dynamic loading each of those goes through `FluidData.__call__`, which reloads
from disk when the requested time leaves the resident window. Replaying frames `0..N`
therefore slides the window back to the start and forward again — a full second pass
over a dataset that may be ~100 GB.

Two facts drove the original design and still hold:

1. **In 3D the statistics text was the entire per-frame fluid cost.** `fluid='vort'`
   and `'quiver'` are 2D-only, so a 3D frame draws nothing about the fluid — yet
   `_calc_basic_stats` pulled the whole 3D field every frame to print `Fluid v_max`
   in a corner. A ~100 GB second pass to render a text label.
2. **2D and 3D are different problems.** The expensive *visualization* is 2D-only,
   where data usually fits in memory and replay is cheap. The expensive *data volume*
   is 3D, where the only fluid-dependent thing drawn is text.

> **A correction worth not re-discovering.** The original outline claimed plotting was
> also a *memory* bottleneck, because "the whole animation is built before anything is
> written", and proposed streaming frames to disk as an independent win. **This is
> false.** `Animation.save()` already wraps `writer.saving(...)` and calls
> `grab_frame()` per frame, so `plot_all` has always streamed into the ffmpeg pipe
> with O(one frame) encoding memory; `FuncAnimation` holds a single figure and redraws
> it, and `cache_frame_data` caches only the frame *indices*. The 2D bottleneck is
> **time** — recomputing vorticity and re-rendering every frame. There is no "stream
> the video" work item, and the video-writing machinery needs no work at all.

### 1.2 What the agent arrays were actually doing in a plot cache

The original spec cached agent positions, velocities and times alongside the fluid
quantities. Three separate reasons got bundled, and only one is a plotting reason:

- **Times — intrinsic.** Once frames are chosen by *simulated time* rather than by
  step index (§4.1, built), the renderer selects frames against a list of capture
  times. A render backed by anything other than live history must carry that list.
- **Velocities — second-order.** Removing the whole-grid fluid reductions and
  replacing them with agent-speed statistics (§3.1) made the statistics box depend on
  agent velocities. Caching them exists only because that substitution happened.
- **Positions — not a plotting or a dyload reason at all.** The stated justification
  was that `pos_history` "lives in memory and dies with the process, so without it the
  cache cannot render after a crash or be used in a later session." That is a
  **persistence** argument. It was in the very first draft of the spec (2026-07-31)
  and was never re-derived. Under dyload alone there is nothing to solve: `plot_all`
  is a `Swarm` method, the swarm is in hand, `pos_history` is right there, and reading
  it re-streams zero bytes of fluid.

A later revision (2026-08-11) made the agent arrays *structural* rather than optional,
by ruling that a cache-backed render is cache-only. But that rule's own stated
motivation — "mixing sources is what would let a 'free' plot quietly re-stream the
dataset" — is about the **fluid**. Reading agents live re-streams nothing. The rule
was broader than its justification, and that over-breadth is what promoted a
persistence nice-to-have into a mandatory plotting component. §4.2 narrows it back to
the fluid.

Meanwhile the scope section disowned the original reason outright — "the cache
replaces the fluid data, not the `Environment`", explicitly *not* renderable in a
fresh session. So the plan carried agent data for a purpose it had declared out of
scope. That incoherence is what this reframe resolves: the agent half was always a
persistence feature and needed a persistence design rather than a plotting one.

### 1.3 Persistence in Planktos today

Verified 2026-08-18:

- **Everything writes; nothing reads.** `Swarm.save_data`, `save_pos_to_csv`,
  `save_pos_to_vtk`. There is no loader for any of them in the package.
- **`save_pos_to_csv` is the worst-shaped output in the codebase for a long run.** One
  `np.savetxt` of a dense `N+1 × (1+D)·T` matrix: the entire history must be resident
  *and* a full text copy is materialized. 1000 agents × 10 000 steps in 3D at `%.18e`
  is over a gigabyte of ascii in a single call.
- **`save_data` saves only the *current* velocity and acceleration**, not their
  histories. `props_history` is explicitly not saved. `rndState`, `envir.time`,
  boundary conditions, `ibmesh` and fluid provenance are saved nowhere.
- **History is unbounded and single-copy.** `TODO.md`'s optional-history-retention
  maybe-feature correctly notes that decimating history is "unrecoverable" and breaks
  `plot_all`, `save_data`, `save_pos_to_csv`, `save_pos_to_vtk` and all post-hoc
  analysis — **but that is true only because the in-memory copy is the only copy.**
  §2.10.

### 1.4 The reframe

Strip the word "plot" from the original cache specification and read what is left:
chunked `.npy` written incrementally; metadata written when recording *starts*; every
chunk self-describing; the timeline reconstructed by scanning disk; no finalizer
load-bearing for correctness, so a `SIGKILL` costs at most one buffer; the reader
opening with `mmap_mode='r'` so a store larger than RAM stays readable; a `store=`
option selecting which arrays are kept; reserved schema slots for `accelerations` and
`ib_collision_idx`; capture hooked to the environment's time advance. The
specification even half-admits it — "this is continuous simulation data, useful for
analysis and not only for display."

That is not a plot cache. It is a **run archive with a plotting consumer.** Almost all
of the expensive thinking already done — crash validity, chunking, self-description,
mmap, the capture/render split, the linearity property (§3.2) — transfers unchanged.
What changes is the label, the public surface (§2.7), and four additions (§2.6, §2.7,
§2.10, §2.11) that turn it into something more than one consumer can use.

---

## 2. Component A — the run archive

### 2.1 Recorder API and lifecycle

**`Environment.record(...)` is the only entry point.** *(Decided 2026-08-11; an
earlier design added `Swarm.record(...)` as sugar with that swarm preselected.)*

A `Swarm`-level entry point implies a per-swarm recorder, which cannot exist. The
recorder is environment-scoped by construction: it hooks the environment's time
advance, its metadata is environment state (`L`, `flow_points`, the fingerprint), and
the fluid half of the output belongs to the environment and not to any swarm. Two
`swrm.record()` calls in one environment would either be refused as a second
concurrent recorder or duplicate every fluid file on disk — so the method would fail
precisely when used for the thing its name suggests.

Restricting *which* swarms are captured is a real want — agent data runs a few hundred
MB per large swarm (§2.4) — but that is an argument for a `swarms=` argument on
`Environment.record`, defaulting to all of them, not for a second entry point. Joint
multi-swarm plotting remains a known gap (issue #49); the archive stores every
recorded swarm, so it does not foreclose a fix.

```python
with envir.record('run_archive/', fluid='vort'):
    for _ in range(steps):
        swrm.move(dt)           # the user's ordinary loop, unchanged
```

**The signature**, gathering parameters the rest of this section and §3.4 justify
individually. Collected here because they were previously scattered across six
sections, which is how a specification drifts from its implementation:

```python
Environment.record(path, *, fluid='vort', swarms=None,
                   store=('positions', 'velocities'), capture_interval=1,
                   chunk_size=100, quiver_shape=(60, 60), plot_all=None)
```

⚠️ **This is the *final* signature, and it accumulates across several build steps.**
The "Arrives at" column is load-bearing: a parameter is added with the thing it
controls, never ahead of it. A parameter that accepts nothing but its default is worse
than no parameter, and one that promises something not yet built — `fluid='vort'` before
component B writes any vorticity — mints archives that A4's reader will correctly refuse
to plot (§2.8). Nothing ships between the steps, so deferring costs no compatibility.

| Parameter | Default | Meaning | Specified in | Arrives at |
|---|---|---|---|---|
| `path` | — | archive directory; created if missing (below) | §2.1 | **A3a** |
| `swarms` | all of them | which swarms to capture | §2.1 | **A3a** |
| `store` | positions + velocities | which per-agent arrays to keep | §2.4 | **A3a** |
| `chunk_size` | `100` | captures buffered before a chunk is written | §2.3 | **A3a** |
| `capture_interval` | `1` | capture — and retain history — every *k*-th step | §2.2 | **A3b** |
| `fluid` | `'vort'` (2D) | which fluid quantity the render will need: `'vort'`, `'quiver'`, a tuple of both, or `None` | §3.3, §3.4 | **B3** |
| `quiver_shape` | `(60, 60)` | target arrow grid, fixed at record time | §3.4 | **B3** |
| `plot_all` | `None` | dict of `Swarm.plot_all` kwargs to render on `__exit__` | §2.1 | **C1** |

`fluid` is **forced to `None`** in two cases, neither an error and both silent, because
in neither is there anything the user could have meant: in **3D**, where no fluid
backdrop is drawn at all (§0.2), and when **`envir.flow is None`** — an analytic or
flow-free run has no vorticity to record, and defaulting to `'vort'` there would fail
on several of the examples and much of the test suite.

**The directory.** Created if missing, parents included. **If it exists and is
non-empty, the archive goes to a sibling directory with a timestamp appended** —
`run_archive/` becomes `run_archive_20260818143052/` (`_%Y%m%d%H%M%S`, no separators).
Overwriting a previous run's data is never the right default, and refusing outright
would strand a long job that was ready to start. **The redirect is never silent:** the
recorder warns naming the path it actually chose, and the handle exposes it as
`.path`, which is also what `plot_all=` renders from. Without that, a user's later
`load_run('run_archive/')` would quietly read the *previous* run.

⚠️ **Recording must start before the fluid window has moved.** Under `INUM=int` the
per-dump fluid quantities (§3.3) are written as each dump lands, by an observer riding
`_record_dump_means`. Dumps the sliding window has already passed are gone and are
never re-reported, so a recording started mid-run would have holes in its fluid series
— which §2.8 refuses at *render* time, i.e. after the run instead of before it. So
`record()` **raises** when the fluid is dynamically loaded and the resident window no
longer starts at the first dump of the series. The message names the two remedies:
start recording before the loop, or load with `INUM=None`.

**Backfilling was considered and rejected** *(2026-08-18)*: re-reading the passed dumps
is a second streaming pass over exactly the data this design exists to avoid
re-streaming, built to serve a workflow — start recording partway through a long
dynamically-loaded run — with no evident use. Refusing early is the whole fix.

⚠️ One nuance that does **not** rescue the mid-run case, and must not be mistaken for a
backfill: `FluidData._dump_means` is populated for every dump ever loaded and is never
evicted, so means for already-passed dumps are still in memory and the opening sweep
can harvest them. Per-dump **extrema** (§3.5) and any written vorticity or quiver
cannot be — those are new work the observer does as a dump lands.

**`envir.record(path, ...)` starts recording immediately** and returns a handle — the
`open()` model, where the call does the work and `with` only adds the guaranteed
close. `__enter__` returns the handle; `__exit__` closes. This matters: if the work
lived in `__enter__`, a bare `envir.record(path)` would silently record nothing, which
is a very expensive thing to discover after a twelve-hour run.

| Call | Does |
|---|---|
| `envir.record(path, ...)` | resolve and create the directory, check the fluid window, write the metadata and the fingerprint, take capture 0, register the hooks. Recording is live from here. **At B** it also sweeps the fluid state already in memory — component means, plus extrema and any per-dump quantity for currently resident dumps |
| `envir.flush_recording()` | write buffered captures to disk. **Keeps recording** |
| `envir.stop_recording()` | flush, then unregister the hooks. Idempotent |
| `with envir.record(...)` | as above, plus `stop_recording()` on exit and the optional auto-plot |

**A second `record()` while one is active raises**, naming `stop_recording()`. There is
one recorder per environment by construction — the time-advance hook finds it through a
single reference on the `Environment` — so a second call could only replace the first,
which would abandon a partly-written archive without saying so, or run beside it, which
the hook cannot express. Refusing is the only honest option, and the directory-redirect
rule above means the remedy (stop, then record again) never overwrites anything either.
Note this is the same argument §2.1 opens with, applied to the environment-scoped
recorder that survived it, so it is not a new constraint — only one that was never
written down as a behavior.

*(Naming: the original spec called the middle one `flush_cache()`. Renamed with the
reframe so the triple reads coherently; the stored data is an archive, not a cache.)*

Both spellings are supported because a `with` block cannot span notebook cells, and
interactive exploration — run 200 steps, plot, run 800 more — is a normal Planktos
workflow. `stop_recording()` and `flush_recording()` live on the `Environment` rather
than only on the handle because the `Environment` must hold a reference to the active
recorder anyway (that is how the time-advance hook finds it), so no variable has to
survive across cells.

**`flush_recording()` is separate from `stop_recording()` on purpose.** A mid-run
notebook plot needs a flush that does not end the recording — otherwise the next
`record()` would refuse the now-non-empty directory. Readers never flush on the
writer's behalf (§4.2).

**`plot_all=` renders automatically at the end of a `with` block.** Given a dict of
`Swarm.plot_all` keyword arguments, `__exit__` flushes and then renders:

```python
with envir.record('run_archive/', plot_all=dict(movie_filename='out.mkv', fluid='vort')):
    for _ in range(steps):
        swrm.move(dt)
```

- **It renders when the run raises, and not when it is interrupted.** A crash is
  unexpected and the movie is diagnostic; a `KeyboardInterrupt` is the user asking for
  things to stop *now*, and kicking off a ten-minute render at that moment — requiring
  a second Ctrl-C to escape — is the opposite of what was asked for. Both still flush.
- **A failure inside the auto-plot must not mask the run's exception.** Catch it, warn
  with it, and let the original propagate.
- **Exactly one swarm.** `plot_all` is a `Swarm` method and joint multi-swarm plotting
  is issue #49, so a recorder covering more than one swarm rejects `plot_all=` — at
  `record()` time, before the run, not at the end of it. Inventing a filename-suffixing
  convention here would be committing to an answer for #49 as a side effect.

The recorder takes **no video parameters of its own** — no `fps`, no `playback_rate`,
no colormap, no figure size (§2.9). `plot_all=` is not an exception: it carries a dict
the recorder never inspects and hands straight through.

Note `fluid='vort'` on `record()` means **which fluid quantity the render will need**,
not what to draw — the same keyword on `plot_all` selects the backdrop. Same word,
different side of the capture/render line; worth distinct wording in the docstrings.
It is deliberately *not* "which quantity to write": asking for `'vort'` frequently
writes nothing at all, because the field is already available or cheap to recompute
(§3.3). What the keyword guarantees is that the render will have it.

### 2.2 Capture trigger and schedule

**Capture is automatic and hooks the environment's time advance.** *(Reversed
2026-08-11. The original design had `rec.move()` and `rec.capture()` and refused to
hook `move()`, on the stated grounds that "users routinely subclass `Swarm` and
override the move machinery." That premise contradicts the codebase: `move()`'s own
docstring says "DO NOT override this method when subclassing", and `apply_agent_model`
/ `after_move` — the actual extension points — are called from inside `move()`, below
any hook. A subclass that replaces `move()` without delegating has already lost
history recording, boundary conditions, the velocity/acceleration finite difference
and the time advance, so a missed capture is strictly subsumed by a far larger
failure. That misuse is now warned about at class-definition time by
`Swarm.__init_subclass__`, independently of this plan.)*

- **The trigger is the environment time step, not `Swarm.move`.** `move_swarms` calls
  `s.move(dt, update_time=False)` for each swarm and *then* advances time, so a hook on
  `Swarm.move` would fire once per swarm, at a time that has not advanced, with later
  swarms not yet moved. Capture fires from exactly two places, both meaning "the
  environment just advanced one step": the end of `Swarm.move` when `update_time=True`,
  and the end of `Environment.move_swarms`. One `envir._notify_step_complete()` called
  from both, a no-op when nothing is recording.
- **Those two are the only paths that advance simulation time.** `calculate_FTLE`
  inlines its own move loop rather than calling `move()`, so it cannot fire a capture —
  which is what is wanted, since it would otherwise write FTLE probe trajectories into
  the archive.
- **There is no `capture()` method.** The manual multi-swarm pattern (move each swarm
  with `update_time=False`, then bump time by hand) is not a real workflow — and since
  §5.3 it is not even reachable, because a bare `Swarm.move` raises once the environment
  holds more than one swarm. `Environment.move_swarms` is the way to advance them, and
  it fires the hook itself.
- **`Environment.reset()` during recording raises.** It sets `time = 0.0` and clears
  the histories, which would give the archive a rewound clock and two captures at t=0.
  (See also §5.2 — `reset()` has a latent bug of its own.)

**Capture schedule — `capture_interval`, and it governs history too.** *(Brought into
scope 2026-08-18; the original spec deferred it.)* Agent state is captured every step by
default (`capture_interval=1`). A coarser schedule is a **data-fidelity** choice fixed
at run time, and the framing is deliberately *not* "capture every N video frames" but
**"as if `dt` were larger"** — the archive then looks exactly like a run performed at
the coarser timestep, and everything downstream is unchanged with `Δt_capture`
substituted for `dt`. Keeping it orthogonal to video frame rate matters: frame rate is a
**presentation** choice changeable forever after (§4.1). Conflating them would be the
original `dt`↔`fps` footgun in a new costume.

It earns its place on two counts, not one. Space is the obvious one. **Speed is the
larger one:** writing every step to disk is a per-step I/O cost paid against the
physics, and on a long run that is the dominant reason to coarsen.

**A *captured time* is t₀, t_k, t_2k, … and everything records exactly those states.**
That includes `Environment.time_history` and each swarm's `pos_history` / `vel_history`,
which under `capture_interval=k` are appended only at captured steps. This is the
decision that keeps the whole design aligned:

> capture *j* ↔ `pos_history[j]` ↔ `vel_history[j]` ↔ `time_history[j]` ↔ archive
> capture *j*, always, with no index translation anywhere and no second concept of
> "a recorded state".

`time_history` becomes what its name has always suggested and only accidentally been:
**the simulation times at which state is available.** Today that is every `dt`; with a
coarser interval it is every *k*·`dt`. `plot_all`'s existing `frames=` argument indexes
that same list and needs no change in meaning.

Three consequences, in descending order of how much work they are:

⚠️ ~~**Blocker — `apply_boundary_conditions` takes its start point from
`pos_history[-1]`.**~~ **[fixed 2026-08-21, step A0.]** It landed first and alone,
before anything else in this plan, because `capture_interval` silently corrupts
collision handling without it. **What follows is the analysis that justified it and the
reason the fix is what it is** — kept because the coupling is the sort of thing that
gets reintroduced by someone tidying up, and because §6.1 A3b's test is written directly
against these three failure modes.

**What the coupling is.** `Swarm.move` runs:

```python
old_positions = self.positions.copy()              # where agents are now
self.positions[:,:] = self.apply_agent_model(dt)   # propose new positions
self.pos_history.append(old_positions)             # record
...
self.apply_boundary_conditions(dt, ib_collisions=ib_collisions)
```

and `apply_boundary_conditions` then does

```python
prev_pos = self.pos_history[-1]
args = [(int(n), prev_pos[n,:].copy(), self.positions[n,:].copy()) for n in active]
```

That `(start, end)` pair is the **movement segment** handed to `_ibc` — the line tested
against every mesh element to decide whether the agent crossed a boundary and, if so,
where to project it. It is correct today for an incidental reason: after the append two
lines earlier, `pos_history[-1]` **is** `old_positions`, the same object. The history
list is serving as a free alias for a local variable, so the physics reads its start
point out of a structure whose only other job is recording.

**What decimation does to it.** At `capture_interval=3`, history is appended at steps
0, 3, 6, …:

| step | append? | `pos_history[-1]` holds | segment actually tested |
|---|---|---|---|
| 0 | yes | P₀ | P₀ → P₁ ✅ |
| 1 | no | **P₀** | **P₀ → P₂** ❌ spans two steps |
| 2 | no | **P₀** | **P₀ → P₃** ❌ spans three |
| 3 | yes | P₃ | P₃ → P₄ ✅ |

On *k*−1 steps out of every *k*, the collision code is told the agent started somewhere
it has not been for up to *k*−1 steps.

**Three ways that goes wrong, none of which raises:**

- **The chord cuts corners the agent went around.** An agent that legitimately traveled
  around the end of a wall over three steps has a straight chord P₀→P₃ passing *through*
  that wall. The collision code sees a crossing that never happened and relocates the
  agent.
- **Already-applied collisions are re-litigated from the wrong origin.** If step 0
  pushed the agent onto a wall, P₁ lies *on* it; at step 1 the segment starts back at P₀,
  off the wall on the free side. Project-and-slide is recursive — it projects the
  movement vector onto the boundary and continues with the remainder — so a wrong origin
  makes the remaining-movement vector wrong in magnitude *and* direction. The output is
  not a slightly-off position; it is an unrelated one.
- **An agent can end up on the wrong side.** That is the no-penetration invariant
  `CLAUDE.md` calls hard and non-negotiable.

It degrades in the worst direction, too: the larger *k*, the longer the stale chord and
the worse the corruption — so the setting reached for on the longest and most expensive
runs is the one that breaks them most.

**It is the only site of its kind.** A sweep of every read of `pos_history` /
`vel_history` in the package (2026-08-19) finds roughly sixty, of which exactly **one**
is in the movement path: the line above. The rest are plotting (~40 across
`plot`/`plot_all`), saving, statistics, the multi-swarm warning in `move()`, and FTLE's
extraction of its own flow map. This is a single line, not a pattern.

**`vel_history` needs no equivalent.** `move()` takes `old_velocities` as a local and
uses *that local* for the acceleration difference
(`accelerations = (velocities - old_velocities)/dt`); it never reads back through
`vel_history`. The motion generators read the live `swarm.velocities`, not history. So
positions were the only quantity whose history entry got used as a control value, and
velocities already have the shape A0 is giving positions. What decimation means for the
recorded velocity *values* is a separate and benign question — §2.4.

**The fix, and where.** Publish the local:

```python
old_positions = self.positions.copy()
self._prev_positions = old_positions        # added
...
prev_pos = self._prev_positions             # was self.pos_history[-1]
```

**Three sites take that local and then call `apply_boundary_conditions`** — `Swarm.move`,
and the two inlined loops inside `Environment.calculate_FTLE` — and all three must set
it. Plus an `__init__` default (the positions at construction), and a docstring
correction: `apply_boundary_conditions` documented the `pos_history` dependency in
prose, so that contract changed with it. All of that is in place; no `pos_history[-1]`
remains anywhere in `planktos/`.

⚠️ **The decimation gate must never reach FTLE.** `calculate_FTLE` appends history every
step as its *integration state* — the flow map is built by indexing `s.pos_history` — so
decimating it would corrupt the result outright. It is safe automatically, because the
gate lives in `Swarm.move` and FTLE inlines its own loop (which is also why FTLE cannot
fire a capture, above). Stated so that nobody later unifies the two loops and quietly
breaks FTLE.

**Why it is risky, and how it is verified.** The edit is four lines, but it lands in the
code `CLAUDE.md` singles out as the most subtle in the project. Verification is unusually
strong, though: at `capture_interval=1`, `self._prev_positions` and `pos_history[-1]` are
**the same object**, so the refactor is provably a no-op — and the existing collision
suite pins *exact* post-collision positions plus a golden multi-step moving-boundary
trajectory. If those stay bit-identical, A0 is correct by construction. §6.1 A0 lists the
four checks in full, including the one that proves the decoupling outright by making
`pos_history` unusable before the boundary stage and asserting the trajectory does not
move. The forward-looking test — a `capture_interval=k` run reproducing an every-step
run's trajectory bit-for-bit — belongs to **A3b**, since the interval does not exist
until then.

**This knot needed untying anyway.** `TODO.md` records an earlier refactor — reordering
the history appends to after the boundary stage — that was **rejected specifically
because** `apply_boundary_conditions` reads `pos_history[-1]`. So A0 is not a tax
`capture_interval` imposes; it is a coupling that has already blocked one desirable
change, and `capture_interval` is what finally forces it.

- **The environment owns the counter.** `len(time_history)` is no longer the step count,
  so `Environment` keeps its own step counter and both trigger sites ask it the same
  question. History appends and archive captures are gated on that one answer, so they
  cannot diverge. Precisely: `Swarm.move` appends history at the *start* of step *n*
  when *n* ≡ 0 (mod *k*), and the capture hook fires at the *end* of step *n* when
  *n*+1 ≡ 0 (mod *k*) — the same set of states, seen from the two ends of a step.
  Capture 0 at `record()` covers t₀.
  - ⚠️ **The counter has exactly two advance sites, and a hand-rolled time bump is
    neither of them.** `Swarm.move(update_time=True)` and `Environment.move_swarms` are
    where it moves. A user who instead writes `swrm.move(dt, update_time=False)` and
    then `envir.time += dt` by hand leaves the counter frozen at whatever *n* it held —
    so under `capture_interval=k` history appends on *every* such step if that frozen
    *n* ≡ 0 (mod *k*) and on *none* of them otherwise, while `time_history` grows by
    hand each step either way. `len(time_history) == len(pos_history)` — the invariant
    this whole schedule rests on — breaks, and no capture ever fires because the hook
    is on the paths that were bypassed. **Today this is harmless**, because nothing is
    gated: history appends unconditionally and the hand-rolled `time_history.append`
    keeps step with it. The counter is what makes it reachable.
  - **§5.3 closes the multi-swarm half of this and leaves the single-swarm half.** A
    bare per-swarm `move()` loop over several swarms now raises, so the pattern the
    codebase used to document is gone. What survives is one swarm moved with
    `update_time=False` and the clock advanced by hand, which has no legitimate use —
    `update_time=False` exists for `move_swarms` to call. **So `Swarm.move` warns when
    it is passed `update_time=False` while a recorder is active**, naming
    `move_swarms`. A warning rather than a raise: it is legal today, it is nobody's
    documented workflow, and only a recording makes it wrong. Lands with **A3a**, beside
    the counter that creates the hazard.
- **The failed-step handler appends `envir.time` to `time_history` unconditionally**
  (`move()`'s `except BaseException` block, which keeps the histories consistent for
  debugging). Under decimation it must append only when the failed step was a capture
  step, or it pushes `time_history` one ahead of `pos_history` — the exact inconsistency
  that block exists to prevent.

⚠️ **`capture_interval` counts steps, not time.** §4.1 rejects a step count (`every=k`)
as a *frame* specifier because users vary `dt` between `move()` calls, so it silently
means different things within one run. The objection is weaker here and is accepted
deliberately: a capture schedule only has to produce a defensible *subset* of states,
`_select_frames` derives `Δt_capture` from the recorded times rather than from the
interval, and "as if `dt` were larger" stays true under varying `dt`. What follows is
that with varying `dt` the capture spacing is uneven and `Δt_capture` is a mean —
already stated and already warned about in §4.1.

*Naming hazard:* `dump` is already this codebase's word for **fluid** data dumps
(`d_start`, `d_finish`, `load_dumpfiles`, `loaded_dump_bnds`), and IB2d's `print_dump`
is the same concept for its own output. An agent capture schedule is conceptually
identical but must not be called simply "dump" — always qualify, or use
`capture_interval` and reserve "dump" for fluid.

### 2.3 Container and schema

**A directory of `.npy` files — chunked for agent captures, one per dump for fluid
quantities — plus a metadata sidecar.** `.npz` is unusable for the bulk data:
`np.savez` writes the archive in one call, so everything would have to be accumulated
in memory first, defeating the streaming property that motivates the whole design
(~1 GB for full-resolution vorticity over 500 dumps). HDF5/zarr would add a required
dependency to a deliberately lean `install_requires`.

```
run_archive/
  meta.json                 written ONCE at record() and never rewritten: format
                            version, the grid summary, dtype,
                            chunk_size, quantity recorded, quiver shape, where
                            vorticity lives (source dir / here / nowhere), and the
                            provenance record (§2.6)
  grid.npz                  flow_points, L, flow_times, periodic_dim -- the fingerprint
                            itself, written once at record()
  fluid/
    quiver_00042.npy        indexed by GLOBAL flow_times index, written as the dump
                            lands; only when quiver was requested
    Omega.0042.vtk          vorticity, ONLY in the fall-back case where the source
                            directory could not be written -- see §3.3; normally it
                            goes beside the source's own dumps instead, and under
                            INUM=None it is not written at all
    dump_stats.npz          per-dump component means and whole-run extrema;
                            rewritten
                            whole, every STATS_INTERVAL dumps
  agents/
    swarm00.json            name, N, D, first_capture -- written when that swarm
                            joins the recording, which is record() for most and
                            mid-run for one added later
    swarm00_pos_0000.npy    (rows, N, D) float64   -- swarm index, then chunk index
    swarm00_vel_0000.npy    (rows, N, D) float64
    swarm00_mask_0000.npy   (rows, N) bool
    times_0000.npy          (rows,) float64, shared across swarms
```

**Indices in filenames are zero-padded to four digits, and the reader sorts them
numerically — the padding is for humans, the parse is for correctness.** Four digits at
the default `chunk_size=100` covers 10 000 chunks, i.e. a million captures, which is
past any run this is built for; five-digit `quiver_00042.npy` is keyed on the dump index
and already had the room. But padding is not the rule and must not be relied on as one:
`%04d` simply grows a fifth digit at chunk 10 000, at which point lexical order puts
`_10000` before `_9999` and a reader that globbed-and-sorted would silently assemble the
run out of order. **This exact failure has already been paid for once on this branch**
— the OpenFOAM dump directories are named with unpadded numbers, and a lexical sort put
`..._1008` before `..._787` (`TODO.md`, Phase 2, `_natural_key`). So the reader parses
the integer out of each name and sorts on that, and the same rule covers the fluid
files.

Two checks come with the scan, since §2.5 makes disk the authority on the timeline:

- **The recovered chunk indices must be a contiguous run** from 0 (or from the chunk
  holding a late swarm's `first_capture`). Chunks are written in order, so a hard kill
  costs the *last* buffer and never a middle one — a gap therefore means a lost or
  corrupt file, not an interrupted run, and gets §2.8's refusal naming the missing
  index rather than a silent short read.
- **Every chunk but the permitted short ones has exactly `chunk_size` rows** (the last
  chunk is short at the end, a late swarm's first is short at the front). This is what
  turns a chunk index into a global capture index without trusting a recorded count.

**Files are keyed by swarm *index*, names live in the metadata.** The default `Swarm`
name is `'organism'` for every swarm, so two swarms in one environment collide by name
by default — a filename built from the name would silently overwrite. The index is the
position in the recorder's `swarms` list, fixed when recording starts (plus any added
later, below). `agents/swarmNN.json` carries that swarm's `name`, `N`, `D` and
`first_capture`; the roster is assembled by scanning for those files, like everything
else on disk.

**Chunk *j* covers global capture indices [j·chunk_size, (j+1)·chunk_size) for every
swarm.** Aligning chunk boundaries across swarms on a global index — rather than
counting each swarm's own rows — is what makes a swarm added mid-run work without a
second indexing scheme: its first chunk is simply short at the front. So a chunk file
is `chunk_size` rows except the last (short at the end) and a late swarm's first (short
at the start); `first_capture` in that swarm's sidecar resolves the offset. Everything
else is `rows == chunk_size`.

**Agent captures are chunked, not one file per capture.** Capture-every-step on a
10 000-step run would make 10 000 files, which is punishing on network and HPC
filesystems. Buffering `chunk_size` captures (default 100) and flushing a chunk bounds
recording memory at a few MB, keeps the file count at tens, preserves the streaming
property that ruled out `.npz`, and loses at most one chunk to a hard kill. The reader
opens chunks with `mmap_mode='r'`, so **an archive larger than RAM stays readable** —
which matters because this is continuous simulation data, useful for analysis and not
only for display.

#### What goes where — the rule *(settled 2026-08-21, at the top of A2)*

> **`meta.json` holds only what is known when recording starts and never changes
> afterwards. Anything that accumulates during the run lives in the files that
> accumulate with it, and the reader learns it by scanning.**

This resolves a contradiction the plan carried through several drafts. §2.5 requires
that metadata be written when recording **starts** and that the reader reconstruct the
timeline **by scanning what is on disk**, never by trusting a recorded count — yet the
metadata list here used to include the capture times, which by definition do not exist
at the start and grow with every flush. Putting an accumulating series in the one file
whose defining property is "written once, at the beginning" is a contradiction, and it
degrades badly: `meta.json` would be rewritten on every flush, making it the file most
likely to catch a hard kill, and a killed run would leave chunks on disk that the
metadata does not know about. The chunks would then be right and the metadata wrong —
so the metadata cannot be the authority, and there is no reason for it to hold a second
copy at all.

With the rule applied, `meta.json` is written **once**, with a single `os.replace`, and
never touched again. That is the strongest form of §2.5's crash validity available to it.

| Lives in | What | Why |
|---|---|---|
| `meta.json` | format **version**; the **grid summary** (below — a description, not a checksum); **dtype** and **chunk_size**; **which fluid quantity** the render will need (`vort`, `quiver`, or both), where vorticity lives — source directory, this archive, or nowhere because it is recomputed (§3.3) — and the **quiver grid** (`quiver_shape` and the `M`/`N` it resolved to); the **provenance record** (§2.6) | all fixed when recording starts |
| `grid.npz` | `flow_points`, `L`, `flow_times`, `periodic_dim` | the fingerprint itself, and the axes that let the archive plot without touching fluid. Fixed at `record()` — see the verification below |
| `agents/swarmNN.json` | that swarm's `name`, `N`, `D`, `first_capture` | written when the swarm *joins*, which is `record()` for most and mid-run for one added later |
| `agents/times_NNNN.npy` | the **capture times** — the sole authority for the agent time base | accumulates; nothing summarizes it anywhere |
| `agents/swarmNN_{pos,vel,mask}_NNNN.npy` | positions (`N×D`), velocities, and the row mask, per capture | accumulates |
| `fluid/dump_stats.npz` | per-dump **extrema** (§3.5) and **component means** (§3.1) | accumulates as dumps land |

**The per-swarm roster moved out of `meta.json` into per-swarm sidecars,** and that
falls straight out of the rule. This section used to require a late-added swarm's
metadata entry to be written *immediately* — the one thing that broke "written once". A
sidecar per swarm makes early and late swarms identical in the format, discovered by the
same scan as everything else, with `first_capture` as the only thing distinguishing
them. The format stops having a special case for the mid-run swarm; only the offset
remains.

Two further things are **not** stored, for the same reason:

- ~~the **capture interval** actually used~~. `_select_frames` derives
  `dt_state = span/(len(times)-1)` from the times themselves. Capture times go in; the
  interval comes out.
- ~~a **capture count or time span** summary~~, even as a human convenience. It would
  accumulate, so it would either need rewriting or be written by a finalizer — and
  §2.5's whole point is that no finalizer is load-bearing. `load_run(path).times` gives
  it in one line.

**Agent velocities are stored, not derived.** Do **not** plan to re-derive them from
stored positions — §5.1 explains why the derivation is wrong even today, and §2.4 why it
becomes a different physical quantity under any capture schedule coarser than every
step. Storing them doubles this part of the archive and removes the trap entirely.

#### The fingerprint *(settled 2026-08-21, at the top of A2)*

**It is structural, it is stored as values rather than only as a hash, and it lives in
`grid.npz`.** Contents: **dimension, `L`, `flow_points` (the per-axis coordinate
arrays), `flow_times`, and `periodic_dim`.**

**Hashing the fluid data itself is ruled out, and not narrowly.** It would mean
streaming the whole dataset to compute it — exactly the ~100 GB cost this design exists
to avoid — and under `INUM=int` only the opening window is resident when `record()`
runs, so a hash over whatever happens to be in memory would depend on `INUM` and on
where the window sat. Two recordings of the same dataset would disagree. The fingerprint
has to be something small and complete, which the grid and the timeline are.

**`periodic_dim` is in it** because it changes the vorticity computed in the outermost
ring (the 2026-08 wrap fix), so a stored vorticity field recorded under a different
setting is a different field.

**Two questions, two mechanisms — keep them apart:**

| Question | Answered by | On mismatch |
|---|---|---|
| Is this the same coordinate system and timeline? | the fingerprint | **hard refusal** (§2.8) — the stored arrays are not interpretable otherwise |
| Did the same thing produce it? | the **provenance record** (§2.6) | **warn**, naming both sides |

That split is what §2.6 promised when it said a provenance record "beats a bare
fingerprint mismatch", and it gets the two realistic cases right: replotting a run whose
script moved directories should not be refused, while a *different simulation* that
happens to share a mesh and a cadence should at least say so out loud. ⚠️ Be plain about
the residue: two runs on the same grid at the same timestamps fingerprint
**identically**, and nothing cheap catches a dataset regenerated in place at the same
path. The fingerprint bounds the damage; it does not eliminate it.

**The comparison is over the arrays themselves, and there is no hash of them
anywhere.** §2.8 requires "a hard refusal with a clear message, naming the provenance of
both sides", and a hash can only ever say `3a7f… != 9b21…`. Reading the arrays lets the
message say *what* differs — "this archive has 149 dump times spanning 0–14.9; this
environment has 200 spanning 0–20". Cost is nil: `flow_points` is a few hundred floats
even for a large 3D grid, `flow_times` one per dump, and this section already requires
storing both so the archive can plot without touching fluid. The fingerprint is
therefore **not a new stored artifact** — it is a comparison over `grid.npz`, which had
to exist anyway. What `meta.json` carries is a `grid` **summary** (dimension, `L`,
`periodic_dim`, grid shape, dump count and time span), which describes the archive for
someone reading that file and is never the match test.

> ⚠️ **A checksum was built here and then cut** *(2026-08-21, at review)*, because the
> reason given for it was wrong. It was to "check `grid.npz` against itself" and catch a
> truncated or edited file — but **`.npz` is a zip, and numpy verifies a CRC32 per
> member on read**: a corrupted one raises `BadZipFile: Bad CRC-32`, a truncated one
> raises too. So thirty lines, including byte-order normalization for cross-machine
> stability, were duplicating an integrity check the container already performs. The
> other justifications did not survive either — a "fast reject before loading
> `grid.npz`" is worthless when `grid.npz` is a few kB, and it can never be the match
> test for the reason above. Pinned by
> `test_a_corrupted_grid_file_is_caught_by_the_container`. **Do not add one back**
> without a job that the zip CRC and the summary do not already do.

**Comparison is exact** — shape, dtype and `np.array_equal` on values — because a
rebuilt environment re-runs the same loader over the same files and gets bit-identical
arrays. ⚠️ **Verified rather than assumed (2026-08-21):** `flow_points`, `flow_times`
and `L` are built in each loader's `__init__` and are **never reassigned by
`load_dumpfiles` or `update_spline`** — driven across a full windowed sweep of a series
and back, they remain not merely equal but the *same objects*. This is what makes
"`grid.npz` written once at `record()`" true under dynamic loading, and it is the same
property the `VTK3dData` fix established for `flow_times` (a dynamically-loading
subclass must publish a timeline covering the whole dump range, not the opening window).
If exactness ever proves too strict in practice, loosening to `allclose` is a one-line
change; starting loose and discovering a run plotted against the wrong grid is not
recoverable.

**When `envir.flow is None`** — an analytic or flow-free run, where §2.1 forces
`fluid=None` — the fingerprint is dimension and `L` alone, and `grid.npz` holds only
those. Nothing about it becomes optional; it just gets smaller.

⚠️ **A consequence for A3a: loading a new fluid while recording must raise.** Every
loader reassigns `flow_points`, `flow_times` and `L`, so it would invalidate a
fingerprint already written to disk and leave the archive describing a grid the run
stopped using. It joins `reset()` on the refusal list (§2.2).

**A swarm added mid-recording is captured from that point on.** `Environment.add_swarm`
can be called at any time, so the swarm set is not fixed at `record()`. The new swarm
gets the next index, its `agents/swarmNN.json` sidecar is written **immediately** —
metadata is always written when a thing starts, never at the end (§2.5) — and its
`first_capture` is the global capture index the other swarms are already at, so indices
correspond across all swarms with no per-swarm time base. Reading is by time regardless
(§2.7), so a consumer never has to think about the offset. Note that the sidecar is what
makes this an ordinary case rather than a special one: `meta.json` is not touched, so
"written once at `record()`" survives a swarm joining an hour into the run.

⚠️ **There is no notification hook in `Swarm` at all: the recorder discovers swarms at
capture time.** *(Settled 2026-08-25, at A3a, after two wrong answers.)* This section
originally said to hook `Environment.add_swarm`; that is a convenience wrapper, and the
overwhelmingly common spelling `planktos.Swarm(swarm_size=N, envir=envir)` never touches
it, so the ordinary case would have been recorded not at all, silently. The correction
was to hook the **two sites where a swarm appends itself** — `Swarm.__init__` and
`Swarm._change_envir` — and building that showed it is also wrong, twice over:

- `__init__` appends to `envir.swarms` **partway through its own construction**, before
  `shared_props` exists, so the recorder read the name off a half-built object and
  raised `AttributeError`.
- **`calculate_FTLE` builds a swarm on the environment and pops it again.** Its grid of
  probe agents would have got a sidecar written and then been expected in every
  subsequent capture. This note already warns that FTLE must not fire *captures*, and
  is safe there because it inlines its own move loop — nobody noticed it also
  *constructs a Swarm*.

Both fall away if the recorder syncs its roster against `envir.swarms` at the start of
each capture instead. That is the semantically right moment rather than a convenient
one: **a swarm's existence only matters at a capture**, so a swarm that comes and goes
between two of them is correctly never seen, and `first_capture` is by construction the
index it actually starts at. It also leaves `_swarm.py` with no knowledge of recording.
The cost is a set comparison per step over a list that is almost always length one.

Note what is *not* stored: **the `_calc_basic_stats` scalars.** With positions,
velocities, and the per-dump fluid means all present, every displayed statistic is
derivable at render time — so caching them too would be redundant state that could
drift from the data it summarizes.

**There is no completeness status in the metadata, and no `allow_incomplete` on the
reader.** *(Decided 2026-08-11, reversing an earlier `complete`/`interrupted`/`failed`
field that gated rendering.)* The argument for it was that a truncated archive renders
a video that looks like the whole run. It does not: `plot_all` draws the simulated time
in every frame, so a movie that ends at t=7.3 says so. And a run that stopped at 200
steps **is** a run of 200 steps — the intended step count lives in the user's script,
is never communicated to the `Environment`, and is not a property the data owes an
account of. The corrupt case the flag seemed to protect against — a step half-applied
across agents, some possibly inside an immersed boundary — never reaches the archive at
all, because captures fire only after a step completes.

The decisive point is symmetry: **`plot_all` has never had a concept of a finished
run.** It renders `pos_history` plus the present state and has always worked mid-run.
A completeness gate would have made archive-backed rendering stricter than live
rendering for a run in exactly the same state.

The `KeyboardInterrupt`-versus-exception distinction survives, but only in memory and
only at `__exit__`, where it decides whether the auto-plot runs (§2.1). Nothing is
persisted.

### 2.4 What a capture is

Per environment time step, per swarm: `positions` data (`N×D`), the position row mask
(`N` bools — agents leave whole rows), `velocities` data (`N×D`), and one timestamp
shared across swarms.

**Live attributes are read, not `pos_history`.** The archive therefore does not depend
on history *existing*, only on the two agreeing about *when* a state is recorded (§2.2).
That is what would let a future "no history at all, archive only" mode work (§2.10),
and it costs nothing now.

Capture 0 is taken when recording starts, so capture *j* is exactly `full_pos_history[j]`
at `(time_history + [envir.time])[j]` — the same index convention `_select_frames` and
`animate(n)` already use, and therefore no index translation at render time. ⚠️ That
identity assumes the recording covers the run from t=0 and the swarm existed at the
start; when either is false the archive carries the offset explicitly and consumers
resolve by time instead (§2.3, §2.7, §4.2).

Budget ≈ `(2·D·8 + 1)` bytes per agent per capture with velocities — 49 B in 3D, so a
1000-agent, 10 000-step 3D run is ~490 MB of agent data, separate from any fluid arrays.
A `store=` option selects which arrays are kept; `accelerations` is a reserved schema
slot.

⚠️ **`store=` defaults to `('positions',)`, and velocities are opt-in** *(decided
2026-09-02; §2.11.5 has the measurements)*. This paragraph previously said the opposite
— that velocities were "not practically optional" because `_calc_basic_stats` needs
them. Component R's derived quantities remove that dependency: the statistics box is
served by a per-capture sidecar of `avg_swrm_vel`, `avg_swrm_spd` and `std_swrm_spd`
(40–48 bytes per capture, **independent of N**), the 2D heading markers by a stored
`angle` column, and `perc_left` by the mask, which is written regardless. In 3D nothing
else is needed at all, because 3D draws no heading markers.

Measured, that default takes **48% off the archive and 59% off the recording overhead**
— the second being the better argument, since a smoke run pays the write cost on every
step. What it gives up is per-agent velocity at a past time, which is **unrecoverable**:
§5.1 established that differencing stored positions is wrong for any agent that collided
or wrapped, and the sidecar is a swarm aggregate rather than per-agent data. That is the
trade opt-in recording exists to let the user make.

**Because it is a choice with a delayed cost, `record()` says so at the start** — a
printed notice, not a warning, naming what is being dropped and how to keep it:

    Recording to run_archive/. Storing positions; velocity history will not be
    kept -- plots and statistics are served from the recorded summaries. To keep
    per-agent velocities for later analysis, pass
    store=('positions','velocities').

§2.8 still makes the reader's refusal name `store=` as the cause, so the two ends agree.
Dropping positions is different — there is no consumer at all without them, in or out of
plotting — so `store` must include `'positions'` and **raises** otherwise. *(The `dtype` field §2.3 lists in `meta.json` records what was written; it is
`float64` throughout and no parameter offers anything else. It is in the schema so a
later single-precision option cannot silently change what an old archive means.)*

**What a coarser schedule means for the recorded velocities.** `self.velocities` is
recomputed every step from consecutive positions, so `vel_history[j]` is the
**instantaneous, one-`dt` velocity at capture time *j*** — a *sample* of a per-step
quantity, not an average over the *k* steps since the previous capture. That is the
right thing to store: it is the velocity the agent actually had at that moment, it is
what the statistics box should show, and it costs nothing extra. (It is also why A0 has
no velocity counterpart — §2.2.)

⚠️ It is the one place where "**as if `dt` were larger**" (§2.2) is not literally true.
A run genuinely performed at *k*·`dt` would carry velocities `(P_j − P_{j−1})/(k·dt)` —
displacement over the long step, a smoothed quantity. The archive's are not those, and
should not be. This is the precise reason velocities are **stored rather than differenced
from stored positions** (§2.3, §5.1): differencing recovers the smoothed coarse-step
quantity instead, which is a different physical thing that happens to look plausible.

**`props_history` is not stored,** following the precedent `save_data` already sets
("props_history is not saved"). It costs nothing on the default path: heading markers
fall back to `arctan2` on velocities, which are stored. Reserve a schema slot.

⚠️ **Capture 0's velocities are not zero, and `_calc_basic_stats` today says they
are.** `Swarm.__init__` initializes `velocities` to the *local fluid drift* when a flow
exists, so `full_vel_history[0]` is generally non-zero — while `_calc_basic_stats(
t_indx=0)` deliberately reports the zero vector, on the reasoning that velocity is
undefined before the first step. The archive stores the truth. **§4.3 decides this**,
and decides it in favor of the truth, in both the live and the archive-backed path — so
the two cannot silently disagree.

**Cadence is hybrid, and neither base is the video frame rate** — frames do not exist
until render time:

- **Fluid-derived quantities (vorticity, downsampled quiver): once per fluid dump.**
  Permitted by linearity (§3.2) — exact reconstruction at *any* time, using the
  interpolator's own weights. Usually smaller than per-frame would be (149 dumps vs 500
  frames for the leaf dataset).
- **Agent-derived quantities: once per capture step** (every simulation step by
  default, §2.2).

Together these make the entire frame-rate choice post-hoc: any `Δt_frame ≥ Δt_capture`
can be rendered from the same archive.

### 2.5 Crash validity

**The format must be valid with nothing having run at the end.** A hard kill — HPC
walltime, OOM, node failure — is `SIGKILL`, which defeats `__exit__`, `close()`,
`atexit` and `__del__` alike; §4.4 concedes the same for the video, which is why it
recommends `.mkv`. So:

- metadata is written when recording **starts**, not when it ends -- and, since §2.3's
  what-goes-where rule, it is written *only once*: everything that accumulates lives
  in the files that accumulate with it;
- every chunk is self-describing;
- the reader reconstructs the timeline by **scanning what is on disk**, not by
  trusting a recorded count;
- **every file appears atomically.** Write to a temporary name in the same directory
  and `os.replace` it into place, which is atomic on POSIX and on Windows for an
  existing-or-not destination on one volume. Without this the guarantee is wrong in a
  way that is worse than a missing chunk: a kill *during* `np.save` leaves a **truncated
  `.npy`** that raises on read, so one unlucky moment costs the whole archive rather
  than one buffer. The same applies to `meta.json` and to `dump_stats.npz`, which is
  rewritten periodically and would otherwise be the likeliest file to catch a kill.
  - ⚠️ **A per-dump fluid field needs a temporary *directory*, not a temporary name**
    *(found at B3, 2026-08-26)*. §3.3's write-back goes through a per-source vtk
    writer, which builds its own filename from the quantity and the dump number and so
    cannot be handed one — `_atomic_write` does not reach it. Staging into
    `.planktos_partial/` beside the destination and renaming out of it gets the same
    guarantee on the same filesystem. Without it a killed run leaves a truncated
    `Omega.####.vtk` **in the source's own dump directory**, where it outlives the
    archive entirely and breaks a dataset other runs share. That is a worse failure
    than the `.npy` case this bullet was written for, since the damage escapes the
    archive; it is the one place the crash-validity argument reaches outside
    `run_archive/`.

Once that holds, **no finalizer is load-bearing for correctness**: the most any of them
can save is one unflushed chunk. This is the property the interface rests on, not the
context manager. It is also what makes the archive worth having for HPC work at all —
the runs most likely to be killed are exactly the ones most expensive to repeat.

### 2.6 Provenance — recording the world without serializing it

*(New with the reframe, 2026-08-18.)*

An archive holds agent state. It does **not** hold the `Environment`: `plot_all` needs
`L`, `bndry`, `ibmesh`, `units` and `_plot_setup`, and serializing all of that — moving
mesh included — is a much larger feature and is not this one. The original scope note
was right about that and the conclusion stands.

But the gap between "cannot deserialize the world" and "cannot reconstruct it" is
almost entirely bookkeeping, and closing it is nearly free **if it is designed in now
and awkward to retrofit later.** So `meta.json` carries a **provenance record**: the
loader calls that produced the fluid and the mesh, by name and arguments, plus the
fingerprint already required for validation.

```json
"provenance": {
  "planktos_version": "1.1.0",
  "environment": {"L": [...], "units": "m", "bndry": [...],
                  "rho": ..., "mu": ..., "nu": ...,
                  "char_L": ..., "U": ..., "ibmesh_color": "k"},
  "fluid":  {"loader": "read_IB2d_fluid_data",
             "kwargs": {"path": "...", "dt": ..., "print_dump": ..., "d_start": ...,
                        "d_finish": ..., "INUM": 4}},
  "ibmesh": {"loader": "read_IB2d_mesh_data", "kwargs": {"path": "..."}}
}
```

Three things this buys, in increasing order of ambition:

1. **A stronger validation message.** "This archive was recorded against
   `read_IB2d_fluid_data(path='leaf_data', ...)`; this environment's fluid is
   `read_openfoam_vtk_data(...)`" beats a bare fingerprint mismatch.
2. **A self-describing dataset.** Six months later the archive says what produced it
   without anyone having to find the script.
3. **Reconstruction, and therefore restart** (§2.11). Reload becomes "re-run the
   recorded loader calls", which is cheap to implement and honest about its cost — the
   fluid is re-read from its own files, which is where it lives anyway.

**Provenance is captured at load time, by the loaders — which is the real work in
it.** The information exists only at the moment `read_IB2d_fluid_data(...)` is called;
by the time `record()` runs it is gone. So **every loader records its own call** into
environment state (`Environment._fluid_provenance` / `._ibmesh_provenance`, say) and the
recorder merely serializes what it finds. Nothing in the codebase did this before A1 —
there was no such attribute anywhere — so it was a small edit to each of
`read_IB2d_fluid_data`, `read_vtk_data`, `read_openfoam_vtk_data`, `read_comsol_vtu_data`,
`read_npy_data`/`load_NetCDF` and the analytic generators (`set_brinkman_flow`,
`set_channel_flow`, `set_canopy_flow`), plus `read_IB2d_mesh_data`,
`read_stl_mesh_data` and `read_vertex_data`. It is easy to overlook when planning
step A because §2.6 reads like a serialization task; it is a *loader* task. §6.1 A1
carries it.

⚠️ **`Environment.__init__` is a fluid entry point too, and it is the one the test
suite uses.** `Environment(flow=[u, v], flow_times=t)` takes a list of ndarrays and
never calls a loader at all — it is a documented constructor argument, it is how most
of `tests/` builds fluid, and Appendix A notes it hardcodes `INUM=None`. A1 that edits
only the eleven loaders therefore leaves the most-exercised construction path with **no
provenance attribute at all**, and the writer meets an `AttributeError` on the first
archive anyone records in a test. Two things follow, both one-liners, both easy to miss
precisely because they are not loaders:

- **Initialize `_fluid_provenance` and `_ibmesh_provenance` to `None` in `__init__`**,
  so the attribute always exists and the writer never has to `getattr`-with-default
  around a hole in its own schema.
- **Record `flow=` as honestly unreconstructible.** Arrays handed over in process have
  no call to replay, so the record is `{"loader": null, "note": "arrays supplied to
  Environment()"}` rather than a fabricated loader name. That is exactly the case the
  paragraph below is about: mark it `null`, and never let a reader silently act on it.

**The environment scalars are deliberately duplicated.** `L` and `units` also appear at
`meta.json`'s top level and in `grid.npz`, which the rule against redundant derivable
state (§2.3) would normally forbid. The exception is worth it: the provenance block is
what someone opens to see what a run *was*, months later, and a block that omits the
domain size and the fluid density to avoid three duplicated numbers is worse at exactly
the job it exists for. The cost is nil and the values are written once, at the start,
from one source — they cannot drift within a run.

**What provenance is not:** a guarantee. Paths go stale, datasets move, and a user who
built the environment by hand (an analytic field, a programmatically-modified `bndry`)
leaves a record that is accurate but not sufficient. Record what can be recorded, mark
the rest `null`, and never let a reader *silently* act on a provenance record it could
not verify.

### 2.7 The reader — public, and not only for plotting

*(New with the reframe.)* The original spec's reader existed solely to feed `plot_all`
and was internal. For persistence it has to be a documented object in its own right:

```python
import planktos
run = planktos.load_run('run_archive/')      # -> RunArchive

run.times                 # (n_captures,) float64 -- the global capture time base
run.swarms                # [('organism', 0), ('organism', 1)] -- name, index
run.positions(0)          # -> CaptureSeries, shape (n_captures, N, D)
run.velocities(0)
run.capture_at(3.4)       # -> int: index of the nearest capture time
run.meta                  # the schema dict, provenance included
run.grid                  # the fingerprint arrays
run.check_against(envir)  # refuse a foreign archive (section 2.8)
```

⚠️ **`positions()` returns a `CaptureSeries`, not a masked array** *(as built, A4)*.
The line above used to say "masked, mmap-backed", which cannot be both things at once:
`np.load(mmap_mode='r')` gives a memmap per *chunk*, and concatenating chunks into one
array materializes every one of them -- exactly what "never load every chunk to answer a
question about one time" forbids. A `CaptureSeries` is a plain read-only sequence:
`series[j]` is one capture and reads only the chunk it lives in, `series[a:b]` reads
only the chunks that span touches, and `series.asarray()` materializes the lot when
that is what you want and says so in its name.

**It is a sequence, not an ndarray, and never claims otherwise.** That is the
`FlowArray` lesson applied (Appendix A): something that pretends to be an array it is
not gets `np.asarray`'d into the wrong buffer, silently. This hands back real masked
arrays and holds no opinions about being one. It is the same shape `FluidData` already
established here -- a container you index to get plain arrays.

**Swarms are addressed by index, with names as a convenience.** `run.positions(0)` is
always unambiguous; `run.positions('organism')` also works and **raises** when the name
is not unique, which the default name makes common (§2.3). `run.swarms` lists both so
the caller can see the collision rather than guess.

**`capture_at(t)` returns an index, and there is no interpolation of agent state.**
An earlier draft left this as "nearest capture, or the interpolation weights for it";
it is the former. Agent state is *snapped*, never blended — that is already what
`Swarm.plot(t)` and `_select_frames` do (§4.1, §4.2), and interpolating positions across
a domain wrap or an immersed-boundary slide would invent trajectories that never
happened. Temporal interpolation weights belong to the **fluid** side, where the field
is smooth and §3.2 licenses it.

**Reading is by time, not by index into someone else's list.** A swarm added mid-run
starts at `first_capture > 0`, and a recording started after t=0 (permitted when
`INUM=None` — §2.1) has its capture 0 partway into the run. Resolving a request by time
against `run.times` is correct in every one of those cases; assuming archive index *j*
equals history index *j* is correct only in the common one. Per-swarm arrays are padded
at the front with fully-masked rows up to `first_capture`, so every swarm's array is
`n_captures` long and aligned to `run.times` — masked meaning "not present", which is
already what a masked row means everywhere in Planktos.

Requirements that follow from being public rather than a plotting detail:

- **mmap-backed and lazily concatenated.** Never load every chunk to answer a question
  about one time. An archive larger than RAM must stay usable — that is half the point
  of chunking.
- **Masked-array semantics preserved.** A masked row means the agent has left the
  domain, and every downstream consumer in Planktos depends on that. The mask goes in
  as its own array and comes back out attached.
- **Read-only.** A reader that mutates the thing it reads is the wrong shape (§4.2).
- **Documented in `docs/api/`,** exported from `planktos/__init__.py`.

**Module placement: `planktos/archive.py`, not `_archive.py`.** The convention is that
underscored modules are internal; `fluid.py` is un-underscored precisely because
`FluidData` is user-visible through `Environment.flow`. `RunArchive` is user-visible
through `load_run`, so it follows `fluid.py` exactly. The recorder (`RunRecorder`,
returned by `Environment.record`) lives in the same module, as `fluid.py` holds both
`FluidData` and its loaders.

### 2.8 Validation on load — missing ≠ mismatched

- **Mismatched** (wrong fingerprint, wrong grid) → hard refusal with a clear message,
  naming the provenance of both sides (§2.6). Silently plotting a foreign archive is
  the worst available outcome.
- **Missing** (a quantity not recorded) → **hard refusal too**, naming what is absent.
  For fluid quantities there is no fallback path by design (§4.2), and no way for a
  supposedly-free plot to quietly re-stream 100 GB. The remedy is to re-record, or to
  plot live without an archive.
- **Never derive vorticity from stored quiver arrays.** They are downsampled, so
  gradients taken on them are a coarser, different field — a plausible-looking wrong
  answer. Recording both `vort` and `quiver` is the cheap prevention.
- **Partial fluid series are a refusal, not a silent fill.** Same trap as a partial
  timeline (`TODO.md` Phase 2) and the same treatment: refuse, or warn and fall back to
  writing — but never serve one dump's field for another's.

### 2.9 Capture versus render — the separation

**DECIDED: the recorder captures data only; `plot_all` does all rendering.**

| | Recorder | `plot_all` |
|---|---|---|
| When | during the run | any time after |
| Job | write the archive while data is resident | turn an archive (or live history) into pixels |
| Knows about | fluid dumps, capture schedule | `fps`, `playback_rate`, colormap, clip, figure |

`plot_all` is not made obsolete by the recorder — it drives the interactive on-screen
animation, replay is free when `INUM=None`, and the recorder requires deciding before
the run.

Rationale: the archive already holds everything needed to render, so rendering during
the run buys convenience only — and costs the thing the archive was chosen for. Every
video parameter stays adjustable forever, which an image cache or live rendering would
have re-fixed at run time. Cache **derived quantities, not images**: colormap, clip,
agent subset, figure size and dpi all stay adjustable. Fixed at record time: the quiver
grid, and which fluid quantity was recorded.

Three consequences that simplify the build:

- **There is exactly one rendering path**, so no shared-renderer refactor is needed.
  `plot_all` keeps `FuncAnimation` and `animate()` essentially as they are; only the
  *source* of per-frame data changes.
- **The recorder takes no video parameters** (§2.1), which removes the
  config-duplication problem between it and `plot_all` entirely.
- **The video-writing machinery needs no work at all** (§1.1's correction).

### 2.10 What the archive unlocks downstream

Not in the first build, but the reason the first build is worth more than a plot cache.
Each of these becomes cheap once A exists, and none is possible without it.

**Bounded history memory — the `TODO.md` maybe-feature is mostly *subsumed*, not just
enabled.** That item proposes `store_pos_history='all' | 'frames' | None` and correctly
calls the loss "unrecoverable", because decimating history breaks `plot_all` at full
resolution, `save_data`, `save_pos_to_csv`, `save_pos_to_vtk` and every post-hoc
analysis. **With an archive recording, none of that is true**: disk holds what memory
drops, and every one of those consumers can read it back.

`capture_interval` (§2.2) then *is* the `'frames'` case, and better than the item
imagined it: history and archive are not merely "mutually consistent", they are the same
set of states, so there is no second retention concept to reason about. What remains
distinct is the `None` case — no history at all, archive only — which **A0 makes
possible** by decoupling collision handling from `pos_history`, but which is not built
here: it would leave live `plot_all` and `_calc_basic_stats` with nothing to read
without an archive, and that interaction wants its own pass. Update the `TODO.md` item
when A lands, narrowing it to that residue.

**`save_*` become exports rather than the primary path.** `save_data` and
`save_pos_to_csv` are public API and are not going anywhere, but once an archive
exists they are naturally re-expressed as *exports from* one (or from memory) — which
is also the fix for `save_pos_to_csv`'s all-at-once dense text write. **Not first-pass
work**, and a behavior change there needs its own changelog line; noted so it is not
re-derived.

**Post-hoc analysis.** Per-step displacement, dispersal statistics, residence times,
trajectory clustering — all of it currently requires either holding the run in memory
or re-running it. An mmap-backed `RunArchive` makes a finished run an ordinary data
object.

### 2.11 Component R — full-state reboot

**Scheduled, and next** *(promoted from "the follow-on" 2026-08-27)*. Until then this
section was a sketch filed under "not part of the first build": the metadata was
designed for it (§2.6) but no step built it, and §0.2's four components did not
include it. It is now component **R**, and §6.1 has a Step R ahead of Step D. The
reason for the promotion is in §6.1; the short version is that the on-disk format has
to grow, and every archive written before it does is one that cannot be rebooted.

**The goal, stated as a user would:** run a simulation streaming to disk, delete the
`Environment` and the `Swarm`, and rebuild both from the directory at the state the
run left off — same positions, same properties, same random stream — and carry on as
if nothing had happened.

The distinction that keeps it simple:

- an **archive** is append-only history — every capture, no state that history does not
  contain;
- a **checkpoint** is one latest state plus everything history cannot give you.

Same format, different file, written on request (and optionally every *k* captures).

⚠️ **The pre-flight analysis has run** *(R0, 2026-08-31)*. It verified the state list
below, measured what a restore costs, and settled the two questions the build could not
start without. §2.11.5 carries the findings and the decisions; read it before R2.

#### 2.11.1 The organizing rule

Swarm state divides in two, and the division is not "big versus small" but **"does a
history of this mean anything?"**

> **Every variable that could have a history gets its current state stored
> unconditionally. `positions` and `velocities` additionally get their history stored
> unconditionally, because plotting needs it. Every other history is opt-in.**

Two consequences worth stating, because they are what makes the rule cheap:

- **A checkpoint is O(N) per swarm**, a handful of arrays and two small objects. It can
  be written every *k* captures without thinking about it.
- **The opt-in histories are the only thing that scales with run length**, so the one
  decision a user makes at `record()` time is which of them to pay for.

⚠️ **The time base is not on either list.** `times` is shared across every swarm in the
environment and every consumer of the archive needs it, so it is neither per-swarm
state nor an optional history — it is the spine the format is already built around
(§2.3). It is called out here only because it is the one thing that looks like it
belongs in the table below and does not.

#### 2.11.2 What a Swarm is made of

Every attribute a `Swarm` carries, audited against a live one rather than from memory.
"State" is what a checkpoint must hold; "History" is what a series would mean.

| Variable | Shape | State | History | Why |
|---|---|---|---|---|
| `positions` | N×D masked | **required** | **always** | the run itself, and what every frame draws |
| `velocities` | N×D masked | **required** | **always** | the plot statistics and the heading markers read `vel_history[n]`; re-deriving them from positions is wrong for any agent that collided or wrapped (§5.1) |
| `accelerations` | N×D masked | **required** | opt-in | `move` recomputes it by finite difference on the first resumed step, so only an agent model that *reads* it needs the state restored — but there is no `accel_history` in memory, so the archive is the only place a series can exist |
| `props` | DataFrame, N rows | **required** | opt-in | per-agent variation. Already opt-in in memory via `store_prop_history`, and the archive flag should mean the same thing |
| `ib_collision_idx` | int N | **required** | opt-in, **sparse** | `after_move` overrides read it. It is −1 for almost every agent on almost every step, so a dense N-per-capture series is the wrong shape — store collision *events* (see below) |
| `shared_props` | dict | **required** | opt-in | ⚠️ **an addition to the list.** It is mutable and user code changes it mid-run — a ramping `mu`, a schedule on `cov` — so a series of it is meaningful. It also carries `name` and `color`, which therefore need no separate slot |
| `rndState` | `Generator` | **required** | opt-in | ⚠️ **an addition.** The bit generator state advances on every draw. As state it is what makes a restart reproducible at all; as a *history* it buys something extra — a per-capture series lets a run be resumed from **any** capture, not only the last |
| `ib_condition` | str | **required** | — | a plain attribute a user could change mid-run, but in practice fixed. If that ever stops being true it moves to opt-in |
| the `Swarm` subclass | class | **required, as a name** | — | `apply_agent_model` *is* the behavior. Record it the way §2.6 records a fluid loader: a name and nothing more. It cannot be reconstructed without the class being importable, and the reader must say so plainly rather than silently rebuilding a plain `Swarm` |
| `_prev_positions` | N×D masked | — | — | derived: `move` sets it from the previous positions at the top of every step, and `__init__` seeds it from `positions`. A history of it is `pos_history` shifted by one |
| `pool` | worker pool | — | — | a runtime resource the caller supplies; not state |
| `store_prop_history` | bool | derived | — | it is `props_history is not None` |
| `envir` | backreference | — | — | the rebuilt Environment supplies it |

⚠️ **"Nothing else exists" was not quite true**, and the R0 audit against a live
`Swarm` (§2.11.5) found two corrections. `pos_history`, `vel_history` and
`props_history` are attributes too — the table treats them as the *History* column
rather than as rows, which is coherent, but a reader checking `vars(swarm)` against it
will find three more names than the table has. And `store_prop_history` is a row here
yet is **not an attribute at all**: it is a constructor argument, and the derived value
the row describes is `props_history is not None`. Everything else in the table matches
the object exactly.

**The sparse format for `ib_collision_idx`.** One list per agent of
`(capture, element)` pairs, appended when the index is not −1 — so a run where nothing
collides costs nothing, and the dense array is reconstructed by scanning forward. Not
a new file format: it is small enough to sit in json beside the swarm sidecar until
measurement says otherwise.

#### 2.11.3 What the Environment is missing

The Environment half is **nearly** complete already, which was not obvious and is worth
recording. `provenance['environment']` is exactly `{L, units, bndry, rho, mu}`, and
between that, the fluid and ibmesh provenance replay, and the archive's `times`, a
rebuilt Environment matches the original attribute for attribute — audited, and
`RunArchive.check_against` passes on the result.

Five things it did not carry. **Four are closed as of R1** *(2026-09-02)*; the fifth
cannot be:

| Missing | Consequence | Fix |
|---|---|---|
| ✅ **`char_L`, `U`** | `motion.inertial_particles` asserts both are set, so an **inertial-particle run cannot be restarted at all** — it raises before it moves. `Environment.calc_re` is dead for the same reason | two floats into `provenance['environment']` |
| ✅ **`nu`**, in the `Environment(nu=…)`-only construction | `rho` and `mu` are both `None` there, and only those two were recorded, so `nu` was lost silently. Every other construction recovers it as `mu/rho` | record `nu` beside them |
| ✅ `ibmesh_color` | cosmetic; the rebuilt mesh draws in the default colour | one string, recorded **as resolved** (`'k'` in 2D, `'dimgrey'` in 3D) so the reader never repeats the default |
| `plot_structs`, `plot_structs_args` | the extra structures a plot draws (e.g. `ex_poisson_search.py`'s target circle) are gone | **unfixable in principle** — they are functions. The reader should say so rather than appear to have restored them |

`g` is a constant, the FTLE fields and `mag_grad`/`mag_grad_time` are recomputable
outputs, and `swarms` is rebuilt — none of those are gaps.

#### 2.11.4 What a reboot then reads as

Rebuild the `Environment` from provenance, rebuild each `Swarm` from its checkpoint,
restore the RNG, and continue. The two halves fail differently and should say so
differently: a missing fluid file is an error, an unimportable `Swarm` subclass is an
error, and a lost `plot_structs` is a warning.

**Do not serialize `flow` or `ibmesh`** — §2.6's provenance record re-runs the loader.
That is the whole reason provenance was designed in at A2 rather than bolted on here.

**A reboot materializes `pos_history` and `vel_history` from the archive, and
`props_history` only on request** *(decided 2026-08-31, R0)*. The physics does not need
any of them — a resume from an empty history is bit-identical (§2.11.5) — but the
*plot* does, and it degrades silently without them rather than failing: `perc_left`
takes its original agent count from `pos_history[0]`, so a restored swarm reports 100%
remaining when a quarter of it has already gone, and `plot_all` prints "No position
history" and draws a single frame.

Positions and velocities are the pair the plot actually reads, and neither is
recoverable from the other: the statistics box and the 2D heading markers read
velocities, which cannot be differenced back out of positions for any agent that
collided or wrapped (§5.1) and which mean a different physical quantity under a coarse
schedule (§2.4). `props_history` stays opt-in, matching what `store_prop_history`
already means in memory.

What bounds the cost is the recording, not the reboot: a materialized history is as
coarse as `capture_interval` made it, so the knob already exists. Measured, as live
masked arrays: 4.9 MB at N=100/2D/1000 captures, **529 MB** at the §2.4 budget case
(N=1000, 3D, 10 000), 12.9 GB at N=5000/3D/50 000. RAM runs ~13% above the on-disk
figures because numpy masks a full `N×D` bool array where the archive writes one byte
per row.

#### 2.11.5 R0 — what the pre-flight analysis settled

*(2026-08-31. Run before R1 so that R2 and R3 could start from measurements rather than
from the plan's assumptions. Baseline: 1147 passed / 2 skipped / 5 xfailed.)*

**The state list in §2.11.2 is verified sufficient.** Restoring exactly its "State"
column — and nothing else — into a fresh `Swarm` and running on gives a **bit-identical**
continuation against an uninterrupted reference: max position error 0.0, mask identical,
clock identical, through an immersed-boundary mesh over a windowed `INUM=4` fluid.
Dropping one item at a time separates what is necessary from what is merely stored:

| Dropped | Result |
|---|---|
| `rndState` | diverges (2.34) |
| `shared_props` | diverges (5.03) |
| `props`, `accelerations`, `ib_collision_idx` | identical — they reach the physics only through a user model that reads them, which is exactly why §2.11.1 stores them unconditionally |
| `_prev_positions` | identical — `move` resets it from `positions` at the top of every step, so **R3 need not restore it** |

**The checkpoint cannot use `DataFrame.to_json`.** That was §6.3's suggested precedent,
inherited from `save_data`, and it silently truncates: pandas caps `double_precision` at
15 digits and a float64 needs **17** to round-trip, so props come back wrong by 4.7e-11
at the default and 2.9e-16 at the cap. The constraint is pandas' json *writer* specifically
— stdlib `json` on `df[col].tolist()` is exact, and so is `to_csv(float_format='%.17g')`
**provided the reader passes `float_precision='round_trip'`**, without which pandas' fast
CSV parser loses a ulp. `_provenance.jsonable` is not an escape either: it renders an
ndarray as a *description* of its shape and dtype, not its values, so `shared_props`
(`mu`, `cov`) cannot round-trip through it.

⚠️ Whether that precision matters at all is worth stating plainly, because it is easy to
over-weight: for the science it does not — 4.7e-11 is far under any modeling error and is
swamped by the Brownian noise — and it reaches the physics only through a user model that
reads a float prop. It matters because the acceptance test asserts a resumed run lands
*bit-identically* where an uninterrupted one did, and because exactness here costs one
keyword. Take it and stop thinking about it.

**The container, then.** Three categories, which is §2.3's what-goes-where rule restated
by lifetime rather than by content, and the archive already has a working example of each:

| Category | Rewritten? | Already in the archive |
|---|---|---|
| (1) fixed at `record()` | never | `meta.json`, `grid.npz`, `agents/swarmNN.json` |
| (2) accumulating | appended in chunks | `agents/times_NNNN.npy`, `swarmNN_{pos,vel,mask}_NNNN.npy` |
| (3) current state | whole, every hunk | `fluid/dump_stats.npz` — the precedent R2 copies, including its `_atomic_write` |

The checkpoint is a category-(3) file per swarm, mirroring `save_data`'s existing split
with the precision defect fixed:

```
agents/checkpoint00_props.csv   props, one row per agent, pandas' default
                                float format -- which already round-trips
agents/checkpoint00_meta.json   ib_condition, the Swarm subclass name,
                                rndState, the capture index and time this
                                aligns to, and a manifest of which props
                                column went where
agents/checkpoint00.npz         positions, velocities, accelerations, the row
                                mask, ib_collision_idx, shared_props, and any
                                props column whose stacked shape is > 1-D
```

⚠️ **Named for the role, not with the `swarmNN` prefix** *(as built, R2)*. The roster
scan globs `agents/swarm*.json`, so a checkpoint called `swarm00_state.json` is read as
a swarm sidecar and the archive fails to open at all. Renaming removes the coupling
instead of teaching the scan to skip things. Each name then says what is in it: the csv
holds nothing but `props`, and the json is the parameterization of that moment in the
run.

⚠️ **A checkpoint's spilled props columns go in its own npz, not in separate `.npy`
files.** The spill rule below is written for `props_history`, where there is one chunk
series per column; a checkpoint is a single state with an npz already open. Same reason,
one fewer file. **`shared_props` goes there too**, following `Swarm.save_data`'s existing
precedent of an npz — it is a mixture of scalars and arrays, and npz takes both without
pickle, which sidesteps `_save_json`'s `allow_nan=False` for a non-finite scalar.

**Positions and velocities are in the checkpoint even though the archive's last capture
holds them.** §2.3 forbids redundant derivable state, and this is the exception: a hard
kill costs the last unflushed chunk, so a checkpoint that merely *referenced* a capture
index could point at a capture that is not on disk. Holding them makes the checkpoint
self-sufficient and independent of the chunk buffer, which is the whole of §2.5's
argument applied one level down.

**A restore materializes history** — §2.11.4 carries that decision and its measurements.

#### The derived quantities, and what props are stored in

*(Decided 2026-09-02, after measuring the containers.)* Two things replace the velocity
history that §2.4 no longer stores by default, and neither is `props_history`:

- **A per-capture statistics sidecar** — `avg_swrm_vel` (D), `avg_swrm_spd`,
  `std_swrm_spd`. 40 bytes (2D) / 48 (3D) per capture, **independent of N**;
  391 kB over 10 000 captures. `perc_left` is not in it: it counts unmasked rows at
  capture 0 and capture *n*, and the mask is stored regardless.
- **A stored `angle` column**, float32, for the 2D heading markers. `plot_all` already
  prefers `props['angle']` over `arctan2` on velocities, and already knows that column
  is only valid per-frame when a props history exists — so this uses a hook that is
  there rather than adding one. 3.9 kB per capture at N=1000; **38 MB** over 10 000
  captures, against 248 MB for the cheapest full props history. It is the recorder's own
  column, so it must be named distinctly (`angle_calc`) and **`restore()` must not inject
  it into `swrm.props`** — a swarm coming back with a property it never had is a
  behavior change a user model reading `'angle'` would silently pick up.

**Props containers, by lifetime.** float32 as an in-memory dtype is ruled out — a value
integrated over 100 000 steps at `dt=1e-3` drifts by 4.3e-2 — but as a *storage* dtype it
is free, and the format already carries a `dtype` field for exactly this (§2.4).

| | container | why |
|---|---|---|
| checkpoint props (O(N), once) | **csv**, pandas' default float format | human-readable and exact. `%.17g` is unnecessary: the default writer already emits shortest-round-trip repr, verified over 52 004 values including 1e±300. The lossy half was `read_csv`, so **readers must pass `float_precision='round_trip'`** |
| `props_history` (O(N·T), opt-in) | **csv per chunk**, written atomically like every other file | one file per chunk rather than one per column, which the file-count argument in §2.3 demands. csv also beats naive binary on strings, since numpy's fixed-width unicode is 4 bytes per character |
| any column whose stacked shape is > 1-D | **spills to its own `.npy`** | a props column may hold ndarrays — `ex_ind_var.py` gives every agent a 2×2 covariance, and `get_prop` is built on `np.stack(col.array)`. Such a column renders to csv as a **broken multi-line row**, so csv cannot be the only container. `np.stack` turns any column into a uniform `(N, …)` array, which is exactly a `.npy`. A typical run spills nothing |

⚠️ **Not `np.savez` and not `to_pickle`**, both of which round-trip perfectly and are
disqualified on the same ground: an object column requires `allow_pickle=True` on read,
which is arbitrary code execution on a file the user may have been handed rather than
produced. **Not a structured array** either — it is the one layout whose columns are
genuinely strided, so reading one field touches every page; a plain 2-D array with
columns as rows is contiguous per column and needs no separate files. **Not HDF5**:
respectable at `format='table'` (85 kB fixed overhead, and its 1.06 MB was an artifact of
the default `format='fixed'`), but `to_hdf` requires PyTables, which is not among
Planktos's dependencies.

**A props schema change mid-run is allowed** *(decided 2026-09-02)*. Chunked csv absorbs
it with no bookkeeping — a later chunk carrying a new column concatenates cleanly and
earlier rows fill with NaN — and a *spilled* column appearing mid-run reuses the
`first_capture` and short-first-chunk machinery a mid-run swarm already has. The cost is
that NaN then means both "the column did not exist yet" and "the value was NaN"; that
ambiguity is accepted rather than carrying a presence marker, since nothing in the tree
changes the schema mid-run today and a user wanting to signal absence has other markers
available.

**How R is finished against its tests** *(decided 2026-08-31)*. The five `xfail`s in
`test_stream_d_restart.py` are the acceptance criteria, but three of them assert a
*location* — `meta.json`, or the swarm sidecar — that this section has now settled
differently, so they are retargeted at the checkpoint files rather than merely
un-`xfail`ed. Alongside them go behavioral tests that assert a restore round-trips the
RNG stream, the props values and `ib_condition`, rather than that a string appears in a
named file.

⚠️ **The retargeted checklist tests are temporary and are deleted when Step R is
confirmed done.** The behavioral tests cover the same ground — full coverage of the
checklist is what makes them pass — and are merely harder to read as a list. A
one-item-per-line checklist is worth having *while building* and is dead weight
afterwards, so it does not survive into maintenance. Record that here rather than in the
test file, which is the thing being deleted.

---

## 3. Component B — fluid-side streaming — **[done]**

The dyload half. **[done 2026-08-25]** — §6.1 B1, B2 and B3; see the "As built"
notes in §3.3, §3.4, §3.5 and §3.6.

Note what it does *not* do: **under `INUM=None` no fluid *field* is written at all**,
because the whole field is resident and recomputation is cheaper than I/O (§3.3).

⚠️ **One thing is always written, and the specification did not originally say so
plainly** *(settled at B3)*: the per-dump statistics sidecar,
`fluid/dump_stats.npz` — component means (§3.1), per-component extrema and the
per-dump vorticity scale (§3.5). §2.3's table listed it unconditionally and §2.1's
lifecycle table has `record()` sweeping "component means, plus extrema" at B, but
§3's opening sentence read as though the whole component were conditional on `INUM`.
It is not, and the resolution is not a compromise: **the sidecar is a handful of
floats per dump, while the thing "writes nothing" is protecting against is ~1 GB of
field data per 500 dumps.** They are six orders of magnitude apart and there is no
regime in which the sidecar is not wanted — the statistics box shows the component
means on every plot, in 2D and in 3D, so without it an archive cannot be rendered
at all. So: `dump_stats.npz` is written for any fluid whatever, and `INUM` decides
only whether *fields* land on disk. `fluid=` is likewise about fields, and a
`fluid=None` archive still carries the sidecar.

### 3.1 Frame statistics — **[done]**

**Removed** `avg_spd` and `max_spd` (whole-grid fluid reductions). **Added** the
standard deviation of agent speed. Result: `_calc_basic_stats` needs **no fluid field
at any frame**, in 2D or 3D — which was the entire 3D deliverable.

Surviving fluid statistics are the component means `avg_spd_x`, `avg_spd_y`,
`avg_spd_z`, served from a **per-dump mean sidecar**: cache `mean(uᵢ)` per component per
dump as each dump loads (a few floats, free), then evaluate exactly at any time via the
interpolation weights (§3.2, linearity). Agent statistics come from `velocities` /
`pos_history` and involve no fluid at all.

Rationale for the substitution, beyond cost: whole-grid reductions include regions
containing no agents. In an agent-based model a statistic over the agent population is
more informative, and the spread of agent speeds speaks directly to whether the
population is moving coherently. Whole-field values remain available on demand via
`FluidData.fmin`/`fmax` and `Environment.get_mean_fluid_speed()`.

**As built:**

- `_calc_basic_stats` returns `(perc_left, avg_spd_x, avg_spd_y[, avg_spd_z],
  avg_swrm_vel, avg_swrm_spd, std_swrm_spd)`. Both new agent statistics are computed
  from the same masked-row-filtered velocity data as `avg_swrm_vel`.
- The plot box shows **both** agent quantities, notated to say which is which:
  `Agent $|\overline{v}|$` (the norm of the mean velocity — net transport) and
  `Agent $\overline{|v|}$: m ± s` (mean speed and its spread). These measure different
  things and are paired deliberately: `‖⟨v⟩‖` cancels for opposed motion, `⟨|v|⟩` does
  not. The `Fluid v_max` / `Fluid v̄` lines are gone; the per-axis `Fluid v̄ₓ` lines on
  the histogram axes stay, now served from the sidecar.
- The 3D statistics box moved from `text2D(0.75, 0.9)` to `0.65` — the `±` line is
  wider than the lines it replaced and ran off the right edge of the axes.
- Only in-domain agents contribute (the mask is respected).
- The sidecar is `FluidData._dump_means`, an `(n_times, n_components)` array of NaN
  filled in by `_record_dump_means` at every point where data lands in memory
  (`__init__`, and all three load sites in `update_spline`). `get_mean_velocity(time=,
  t_idx=)` is the public reader. For cubic splining it evaluates an `fCubicSpline`
  built over the means themselves — same class, same knots, therefore exactly the mean
  of the splined field, since the construction is linear in the data. For linear
  splining it interpolates the sidecar directly against `flow_times` rather than
  against the resident window, so **a mean stays available after the window has moved
  past it** — which is what makes replaying a finished run free. A time whose bracketing
  dumps were never loaded falls back to a load (a cache miss, not a cache lie).
- **Measured effect**, 25-dump IB2d dataset at `INUM=4`, 48 steps, then `plot_all` to a
  movie: fluid loader calls during plotting went **8 → 0** (25 dumps re-read → none).
  With `fluid='vort'` it stays at 8, because the vorticity backdrop genuinely needs the
  field — that is §3.3's problem, not this one's. In 3D, where nothing fluid is drawn,
  the 0-load case is the only case.
- **Tests:** the four that pinned the removed behavior were rewritten, not treated as
  breakage. The retired `max_spd` regression lock is replaced by
  `test_calc_basic_stats_agent_speed_vs_mean_velocity`, pinning that `⟨|v|⟩` and `‖⟨v⟩‖`
  are genuinely different quantities (four agents, two at +1 and two at −3 in x:
  `‖⟨v⟩‖ = 1`, `⟨|v|⟩ = 2`, `std = 1`). The strongest new test is
  `test_calc_basic_stats_pulls_no_fluid_field`, which monkeypatches `FluidData.__call__`
  and `Environment.interpolate_temporal_flow` to raise — reaching for the field is now
  a hard failure rather than a silent cost. `get_mean_velocity` is covered in
  `test_flow_interface.py` (static, cubic, linear, `t_idx`, extrapolation, the "requires
  a time" error) and, for the sliding window, in `test_dynamic_loading.py` — including
  that a **replay after a full sweep triggers zero loads**, and that the jump-to-start
  fast path records means too.

⚠️ §5.1 revisits the *agent* half of this method: the velocity it reduces is derived
the wrong way, and both new statistics inherit the error.

### 3.2 The property everything rests on

Both spline classes evaluate as a **weighted sum of nodal fields**,
`u(t) = Σᵢ wᵢ(t)·uᵢ`, for `LinearSpline` and `fCubicSpline` alike. So any **linear**
functional of the field commutes with temporal interpolation:

```
F(u(t)) = Σᵢ wᵢ(t)·F(uᵢ)          for linear F
```

`mean`, the curl (hence vorticity), and subsampling (hence quiver arrays) are all
linear. **The weights are computed in one place** — `fluid._linear_blend`, which
`LinearSpline.__call__`, the mean sidecar and the per-dump vorticity read all go
through — so "the same weights the velocity uses" is structural rather than a property
three copies happen to share. This is what makes the per-dump mean sidecar exact (§3.1) and dump-cadence
caching exact (§2.4), using weights the interpolator already computes. The periodic
wrap added to the curl in 2026-08 does not disturb this: differencing across the wrap
is still a fixed linear combination of nodal values, just a different one.

`max` and `mean(√(u²+v²))` are **not** linear and do not commute — which is why
`max_spd` and `avg_spd` were dropped rather than cached.

⚠️ **Linearity makes it exact; it does not make it *local*, and the difference decides
the design.** `LinearSpline`'s weights are two and adjacent, so a per-dump file supports
them directly. `fCubicSpline` is not-a-knot, whose coefficients come from a **global**
tridiagonal solve — every `wᵢ(t)` depends on every node — so applying its weights from
per-dump files would mean holding the entire series, which is the memory cost the whole
design exists to avoid. Two consequences, both already taken:

- The per-dump **mean** sidecar can afford it: three floats per dump, so it keeps them
  all and splines them with the real weights (`_interp_dump_means` has exactly this
  cubic/linear split, and its cubic branch is reachable only because everything was
  resident anyway).
- A per-dump **field** cannot. So §3.3 does not try: under `INUM=None` it computes
  vorticity from the interpolated velocity instead of reconstructing it from dumps.
  That is why the rule is written by regime rather than as one mechanism.

### 3.3 Vorticity is not cached — it is sourced, by regime — **[done]**

*(Decided 2026-08-13; built 2026-08-25.)* Unlike quiver, vorticity is a quantity solvers already write
and Planktos can write back in the same format. **Which of three things happens is
decided by `INUM` and by whether the source has vorticity already:**

| regime | during the run | at render | interpolation in time |
|---|---|---|---|
| `INUM=None` | **nothing written** | compute from the resident velocity | cubic |
| `INUM=int`, source **has** vorticity | **nothing written** | read the source's per-dump field | linear |
| `INUM=int`, source has **none** | write one file per dump as it lands | read back what was written | linear |

The reasoning, measured on `tests/data/Rubberband_with_Damped_Springs` (76 dumps,
33×33, with `Omega`) and `tests/data/leaf_data` (149 dumps, 129×193, without) —
reproduce with `tests/manual/bench_vorticity_sources.py`:

- **`INUM=None` needs nothing on disk.** The whole field is resident, so recomputing
  costs 0.34 ms per frame at 129×193 and rendering 300 frames takes 0.129 s. Sourcing
  the same frames from disk is *slower* — 0.066 s against 0.023 s on the smaller dataset
  where both could be measured — so writing ~1 GB would buy negative performance.
- **The compute is never the cost; the write is.** Deriving a dump's vorticity as it
  lands is +0.4% on a streaming sweep (4.738 s → 4.755 s over 149 dumps) — free. Writing
  it is ~1 ms per dump and ~1 GB per 500 dumps at 512×512. So the only thing worth
  avoiding is the write, and it is avoidable exactly when the source already has the
  field.
- **Under `INUM=int` the velocity is not resident**, so recomputing at render drags
  `load_dumpfiles` behind it: 4.76 s against 0.165 s resident, for the same 300 frames,
  essentially all of it velocity I/O. Worse, velocity is a *vector* and vorticity a
  scalar in 2D, so recomputing reads roughly twice the bytes to produce a field it then
  discards. Reading per-dump vorticity is 3–4× faster and is why this regime sources
  from disk at all.

**Both `INUM` regimes come out exactly consistent with the velocity in use, which is
the point of splitting them.** Under `INUM=int`, blending per-dump vorticity with
`LinearSpline`'s two weights *is* the curl of the interpolated velocity, by §3.2. Under
`INUM=None` the cubic weights are not local (§3.2's caveat), so that regime does not try
to reconstruct from dumps — it differentiates the interpolated velocity directly, which
is the same field by construction. So vorticity inherits the velocity's cubic-vs-linear
tradeoff exactly, rather than stacking a second, different approximation on top of it.
That is the tradeoff `INUM` already documents, and no new one.

⚠️ One caveat on the middle row: it serves the *solver's* vorticity, not Planktos' curl
of the solver's velocity. For IB2d those coincide — 0.00% difference at every dump
tested, once the periodic edge fix landed. That is an empirical property of IB2d
computing the same central difference, not a guarantee for every source.

**Where the written files go.** Into the **source's own fluid directory**, in the
source's own naming — `Omega.0042.vtk` beside `u.0042.vtk` — so that a later run,
ParaView, or IB2d's own tooling reads them with no knowledge of Planktos, and so that
the source becomes indistinguishable from one whose solver had printed vorticity all
along. Two guards:

- **Never clobber.** If `Omega` for a dump already exists, that is the middle row of the
  table, not this one.
- **Fall back to `run_archive/fluid/` if the source directory cannot be written** —
  read-only mounts and shared datasets are normal. `meta.json` records which of the two
  happened, so the reader knows where to look.

**Format: binary VTK, not ascii, and not `.npy`.** Measured at 512×512 over 500 dumps:

| format | write | read | disk |
|---|---|---|---|
| `.npy` | 0.69 s | 0.28 s | 1.049 GB |
| **VTK binary** | **5.85 s** | **1.12 s** | **1.049 GB** |
| VTK ascii | 35.10 s | 47.04 s | 1.941 GB |

Binary VTK costs ~5 s of writing across an entire run and ~1 s at render for *identical*
disk — negligible against the simulation, and interoperability is worth far more than
that. Ascii is not negligible: 6× the write, 42× the read, 1.85× the disk. Write binary
even though IB2d writes ascii; `vtkStructuredPointsReader` takes both, verified by
round-trip. (Small grids are dominated by fixed pyvista overhead — a 33×33 binary write
is *slower* than a 512×512 one — so this only matters at scale, where it is cheap.)

Availability is a per-source *capability*, to be asked rather than assumed:

| Source | Ships vorticity? |
|---|---|
| OpenFOAM (`OpenFOAMData`) | **Always**, as a `vorticity` cell array on `internal.vtu` *and* on every boundary patch — verified on the reference export |
| IB2d (`IB2dData`) | **Optionally** — `Omega.####.vtk`, present only if the run's `input2d` asked for it. `tests/data/leaf_data` has `u` dumps only, so the reference 2D dataset does **not** have it |
| everything else | no |

**Reading a source's per-dump field — the mechanics.** Nothing new is needed in
`_dataio` for the read itself: `read_2DEulerian_Data_From_vtk(path, numSim, 'Omega')`
already reads IB2d's scalar dumps (the branch `uX`/`uY` use, and `Omega` is already
named in the reference comment block inside `_read_IB2d_dumpfiles`), and
`read_vtkxml_cell_data(f, arrays=('vorticity',))` already reads OpenFOAM's. Four things
must line up, none of them obvious:

- **A time resolves to dumps as `d_start + i`**, uniformly: IB2d's `d_start` is the
  first dump number, OpenFOAM's is 0 over a dense index into `_dumps`. ⚠️ `IB2dData`
  stores neither `dt` nor `print_dump`, and does not need to — do not add them as
  attributes to make a reader work.
- **Transpose.** `read_2DEulerian_Data_From_vtk` returns `[y,x]`; the velocity path does
  `.T` to reach `[x,y]`. A derived field must do the same.
- **Restore the periodic endpoint** IB2d omits, so a 6×5 dump becomes a 7×6 field.
  ⚠️ **`_wrap_flow` cannot be reused**: it loops over `range(len(flow_points))` and so
  assumes one array per spatial dimension. Passing a single scalar raises `IndexError`.
  Generalize it, or write the four-line scalar version.
- **Do not re-shift the domain** — `flow_points` is already in quadrant 1.

**Two-slot read cache.** A movie renders many frames per dump interval, so the naive
path re-reads two files per frame. Keeping the two most recent dumps reduces that to one
read per dump for any monotone sweep, forward or backward: consecutive frames share a
bracketing pair, and advancing evicts only the trailing one. Two slots and no more —
holding more field data than the interpolation needs is the thing being avoided. Key on
the **global** dump index so it stays correct across a velocity-window slide it knows
nothing about.

**Probe availability once, and check it covers the range.** Glob for the field at
construction; a *partial* series gets §2.8's treatment.

**As built.** The per-source half is three methods on `FluidData`, and the generic
half is one:

- `probe_stored_vorticity()` → `('complete'|'partial'|'absent', directory)`, called
  once when recording starts. `IB2dData` globs `Omega.*.vtk` and compares against
  `range(d_start, d_finish+1)` — **against the range, not for any file at all**,
  which is what makes the partial case visible.
- `read_dump_vorticity(t_idx)` / `write_dump_vorticity(t_idx, vort, path)`, a pair
  overridden together. The base pair is rectilinear-grid scalar vtk, which can
  express any grid `FluidData` supports; `IB2dData` overrides both to structured
  points with the wrap stripped and the field transposed back to `[y,x]`, on the
  solver's own unshifted coordinates — so a written series is byte-comparable to
  one IB2d printed.
- `get_stored_vorticity(time)` is generic: it blends the two bracketing dumps with
  the same weights `LinearSpline` uses, through a two-slot cache keyed on the
  global dump index. It **raises** under cubic splining rather than blending
  linearly, since a linear blend of dumps would not be the curl of a cubically
  interpolated velocity — that regime has the field resident and calls
  `get_vorticity` instead.

Two decisions the specification left open, taken here:

- **A partial stored series is warned about and written past, into the archive's
  own `fluid/`** — not into the source directory beside the solver's own files.
  §2.8 offers "refuse, or warn and fall back to writing"; writing is the more
  useful of the two, but writing *beside* a partial series would leave a mixed
  series, some dumps the solver's and some ours, which is the "never serve one
  dump's field for another's" trap wearing a different hat. A separate directory
  keeps what a render reads homogeneous, and leaves the solver's files untouched.
- **Never-clobber is now unreachable through `record()`, and is kept anyway.** The
  probe covers every case that could reach it: a complete series is read, a partial
  one is written past, and an absent one cannot collide. The guard survives because
  it makes `_write_vorticity` correct standing alone rather than by a policy
  decided elsewhere, and it costs one `exists()` per dump.
  `test_the_writer_refuses_to_overwrite_a_dump_it_finds` calls it directly and says
  so.

**Measured as built, on the real datasets** — `tests/data/leaf_data` (149 dumps,
129×193, no `Omega`) and `tests/data/Rubberband_with_Damped_Springs` (76 dumps,
33×33, with one). Both refine what §3.3 estimated:

| per dump, 129×193 | cost |
|---|---|
| derive the curl | 0.15 ms |
| write it (binary vtk) | **5.1 ms** |
| read it back | 0.61 ms |
| disk | 197 kB |

- **The write is ~5× the ~1 ms this section estimated**, and the reason is the
  caveat already written here: pyvista's per-save overhead dominates at small
  grids, and 129×193 is small. The shape of the conclusion is unchanged — the
  compute is free (0.15 ms), the write is what costs — but the constant is
  bigger than the specification assumed, so quote 5 ms and not 1 at this size.
- **On a 60-step, 50-agent run over 25 dumps** — deliberately dump-dense, a dump
  arriving every ~2.4 steps — recording cost **+8.2% for the agent half alone**
  (component A), **+22.7% with `fluid='vort'`** and **+35.0% with quiver as well**.
  Those percentages are an artifact of an unusually cheap simulation, not a
  general figure: the absolute costs above are what scale, and a run doing real
  physics per step amortizes them away. Quote the milliseconds.
- **Sourced vorticity agrees with Planktos' own curl to 3e-11 relative** through
  the blend, over a full window sweep of the 76-dump IB2d series at 15 times off
  the dump cadence. That is this section's "0.00% difference" claim reproduced
  end to end through the blending path, on real solver output rather than a
  fixture. The written case is exact to 9e-16, since binary vtk round-trips
  losslessly and the only arithmetic left is the blend.

⚠️ **The regime is decided by what is *resident*, not by which spline class is in
use.** `FluidData.is_windowed` is the discriminator: false for time-invariant flow,
for `INUM=None`, for `INUM=True` — and for an int `INUM` that spans the dataset,
which holds everything and never slides. Keying on `INUM is None` would have put
`INUM=True` in the wrong row and written ~1 GB for a field that was in memory the
whole time.

Two points that are easy to get wrong:

- **3D writes no vorticity at all** — `fluid=` is forced to `None` in 3D, where no
  fluid backdrop is drawn. All of the above is 2D-only today. It becomes live for 3D if
  a backdrop arrives with the vtk rewrite, which is when OpenFOAM's always-present field
  starts to matter and when the rectilinear reader (§3.6) stops being optional.
- **Reading the source is not merely cheaper, it can also be more accurate — but no
  longer for IB2d.** Recomputing by finite difference used to disagree with the solver
  in the outermost cell ring; for a *periodic* source that was the missing wrap, now
  fixed, and the two agree exactly. For OpenFOAM the ring sits against a spliced
  boundary-condition plane instead, no wrap can fix it, and the stored field remains the
  better one (`TODO.md` item 6 has the measured depth profile).

### 3.4 Quiver — **[done]**

**As built.** `quiver_shape` resolves to integer strides at `record()`
(`_quiver_strides`, floored at 1 so asking for more arrows than grid points cannot
produce a zero stride), and `meta['fluid']` records the target, the strides *and*
the grid they resolved to. What is stored per dump is exactly the strided slice
`plot_all` draws, `flow[c][::M, ::N]`, stacked over components — so nothing is
resampled at render time. Written under **both** `INUM` regimes, since no solver
ships a quiver and the "already available" reasoning that keeps vorticity off disk
does not apply to it.

**Quiver is opt-in** — `fluid='quiver'` or `fluid=('vort','quiver')` — because
vorticity is what gets plotted in almost every case, and quiver is a second
full-cadence array on disk for a backdrop most runs never use. `fluid=` on `record()`
defaults to `'vort'` in 2D and is forced to `None` in 3D and on a flow-free
environment — see the signature table in §2.1.

**Quiver stays `.npy` in the archive.** It is a downsampled subsample of velocity chosen
at record time, not a quantity any solver writes or any other tool would want, so none
of §3.3's format reasoning applies to it. Written per dump whenever it is requested.

**The quiver grid is the one genuine conflict between record time and render time.**
`plot_all` currently derives its downsample factors `M`, `N` from the **figure size and
axis extent**, aiming at roughly 4.15 arrows per inch — quantities that do not exist
while the simulation is running. The resolution: `record()` takes a target arrow grid
(`quiver_shape`, default ~60×60), and an archive-backed `plot_all` uses the stored grid
regardless of figure size, warning when the figure would have wanted a noticeably
denser one. The rejected alternative was caching full-resolution velocity and
downsampling at render time, which costs 2–3× a per-dump scalar and gives back the disk
saving that motivates downsampling at all. So `quiver_shape` joins the recorded quantity
as the second thing fixed when recording starts.

### 3.5 The global colour and arrow scales — **[done]**

**As built (the storage half; the render half is C2, also done — see the end of this
section).** `fluid/dump_stats.npz` carries `means`, which is `FluidData`'s own per-dump
mean cache serialized rather than a second copy of it; `vmax`, `(n_components,)`; and,
in 2D when vorticity was requested, the scalar `vort_absmax`. Read back with
`RunArchive.dump_stats()`.

⚠️ **The extrema were per dump until 2026-09-03, and are now single running values.**
`vmin`, `vmax` and `vort_absmax` were all `(n_dumps, …)` arrays, NaN-marked for dumps a
sliding window never loaded. But **nothing ever read one per dump**: the only consumers
reduce over the whole run — `nanmax(vort_absmax)` for the colour limit and
`norm(nanmax(vmax, axis=0))` for the arrow scale — so the arrays were built, indexed,
NaN-marked and serialized per dump only to be collapsed to one number the moment anyone
looked. They are now maintained as running maxima (`np.fmax`, which takes the number
over the NaN), which:

- deletes the NaN semantics from the extrema, and with them `_frames._nanreduce` and
  the test pinning that each row is its own dump's reduction rather than a neighbour's
  — a class of bug that no longer has anywhere to occur;
- makes an append trivial, since combining two runs' extrema is one `max` rather than a
  merge of two NaN-marked arrays. §6.1 Step R5 is what that is for;
- costs the per-dump extrema as a diagnostic. Nothing asked for it, and vorticity peaks
  per dump are of little interest with self-propelled agents in any case.

⚠️ **They start at NaN, not zero.** `_vorticity_norm` draws a zero limit as a uniformly
white field rather than collapsing the colormap, so a zero start would make "no dump has
arrived" indistinguishable from "the vorticity really is zero". One NaN replaces *n*, and
the "the archive has nothing to say about the scale, fall back" behaviour is unchanged.

⚠️ **`vmin` was deleted outright.** It had no consumer anywhere in the package — written,
documented and asserted on by tests, never read. It is also close to meaningless as a
fluid statistic: with a no-slip condition anywhere in the domain it is pinned at zero by
the geometry, so it measures the discretization rather than the flow.

**`means` stays per dump**, and its NaN keeps meaning what it meant. It is not a
reduction: `_interp_dump_means` blends the two bracketing dumps with the interpolator's
own weights to serve the statistics box at an arbitrary time, and a dump that never
loaded genuinely cannot be interpolated through.

⚠️ **Rewritten whole, but not on every dump** *(settled 2026-08-26)*. `.npz` cannot be
appended to, so the file is rewritten entire — which keeps it a single atomic replace
and therefore always readable. Doing that per dump arrival costs **O(n²) bytes** over a
series: the file is `n` rows, and a forward sweep at `INUM=4` arrives `n/3` times, so a
900-dump sweep wrote 8.6 MB to persist 29 kB and a 10 000-dump one would write ~2 GB.
Throttled to every `_FluidWriter.STATS_INTERVAL` dumps (100), a 900-dump sweep writes
0.3 MB — 30× less — and `flush()`/`stop()` still write unconditionally, so a completed
or explicitly flushed run is always current. **The tradeoff is exposure, and it is the
one the agent chunks already carry:** a hard kill costs at most the last interval's
rows, whose dumps a re-run would have to reload anyway.

⚠️ **A dump the run never loaded is `NaN`, not zero, and a consumer must reduce with
`np.nanmax`.** Under a sliding window a run that stops partway genuinely never sees
the later dumps, and NaN is the honest record; a zero would be indistinguishable
from a still fluid and would drag the global scale it is supposed to fix. C2 is the
consumer that has to get this right.

**`vort_absmax` is computed even under `INUM=None`**, where §3.3 writes no field.
The per-dump curl is 0.34 ms at 129×193, so this is ~50 ms across a 149-dump series,
and it makes the global scale available in every regime rather than only the
streaming one — otherwise two renders of a resident run would still disagree with
each other, which is the whole defect §3.5 exists to fix.

**Colour normalization — half done (2026-08-13).** The per-frame `fld.autoscale()` this
was written against **is gone**; `Swarm._vorticity_norm` replaced it. That call rescaled
to each frame's own min/max, which put zero off the white centre of RdBu and tinted the
background differently every frame — the reported "flashing" — and silently discarded
any `clip` the caller passed. Limits are now symmetric about zero, grow across a movie
but never shrink, and are left alone entirely when `clip` is given.

**What remains is the *global* scale.** Monotone growth removes the flicker but the
scale still changes during a movie, so two renders of the same run still differ. Derive
it from the **stored per-dump extrema in a second pass over the archive** (small) rather
than over the fluid (huge), and set it once before the first frame. `_vorticity_norm` is
where it plugs in — pass the global maximum as `clip` and it is already fixed and never
rescaled.

⚠️ `FluidData.fmin`/`fmax` are **not** usable for this: they are documented as covering
"all the data seen so far", so under dynamic loading they grow during the run and would
reintroduce the drift.

**The same `fmax` drift reaches the quiver arrow scale, not just the colour scale.**
`plot_all` sets `scale=max_mag*5` once at figure setup from `self.envir.flow.fmax`, so
under dynamic loading the arrow length representing a given speed depends on how far the
run had progressed when plotting began — two movies of the same simulation are not
comparable, and neither is comparable to itself across a re-plot. The stored per-dump
extrema fix both, in the same second pass. Max-over-dumps is an exact upper bound under
linear interpolation and very tight under cubic. A live one-pass render mode, if ever
offered, has no global scale available and must take an explicit `clip`/`vmin`/`vmax`,
or disclose the drift on the colorbar.

**As built — the render half (C2, 2026-08-27).** Both scales are reduced once in
`_frames.FrameSource._global_scales`, before the first frame:

- the colour limit is `dump_stats['vort_absmax']`, handed to `_vorticity_norm` as its
  `clip` — which that function already treats as fixed and never rescales, so §3.5's
  "pass the global maximum as `clip` and it is already fixed" was exactly right and the
  rendering side needed no other change;
- the arrow scale is `norm(dump_stats['vmax'])`, which is what `fmax` reaches after a
  full sweep and then **stops** at. `fmax` does not: it grows with every later fluid
  access, so a scale taken from it moves between two renders of one recorded run.

*(Both were `nanmax` reductions over per-dump arrays until 2026-09-03; the values are
identical, the reduction now happens as the dumps arrive rather than at render time.)*

Three things the specification did not say and one it did:

- **`np.nanmax`, and a whole-slice guard.** NaN marks a dump the run never loaded, which
  §3.5 warned about; `_nanreduce` also returns `None` for an all-NaN array rather than
  handing matplotlib a NaN limit, and the caller then falls back. Pinned by
  `test_a_dump_the_run_never_reached_is_nan_and_does_not_poison_the_scale`.
- **An explicit `clip=` still wins outright.** The global limit fills in only where the
  caller supplied none, which is what `_vorticity_norm` already promised.
- **No archive, no global scale.** A live render keeps the growing-but-never-shrinking
  norm and `flow.fmax`, unchanged. That is §8's deferred one-pass mode inheriting the
  problem, made concrete:
  `test_two_renders_of_different_stretches_of_a_run_share_a_colour_scale` asserts both
  halves — equal with an archive, unequal without one.
- **The quiver *grid* is the other half of §3.4's conflict**, and it is settled the way
  §3.4 said: `resolve_strides` takes what the figure wanted, returns the stored strides
  when the arrows come off disk, and warns only when the figure wanted a *noticeably*
  denser grid (1.5×). Rounding a target arrow count against a grid lands a stride off by
  one routinely, and a warning that fires on every plot is one nobody reads. With the
  field resident nothing is read from disk, so the figure chooses and nothing warns.

### 3.6 Scalar rectilinear VTK I/O — **[done]**, and it was the missing half

⚠️ **§3.3's write-back required scalar rectilinear-grid I/O that did not exist.**
`STRUCTURED_POINTS` carries only an origin and a spacing, so it can express IB2d's
uniform grid but not a rectilinear one — and the OpenFOAM grid is deliberately
non-uniform at its two outermost intervals. `_dataio` has
`write_vtk_2D_rectilinear_grid_scalars` (used by `Environment.save_2D_vorticity`) but
**no matching scalar reader**; only `read_vtk_Rectilinear_Grid_Vector` exists.

Both halves are needed:

- a `RECTILINEAR_GRID` scalar **reader**, to pair with the existing writer;
- a `STRUCTURED_POINTS` scalar **writer**, for uniform grids, so a written field is
  indistinguishable from the solver's own.

Independent of everything else, and testable on its own with a round-trip.

**As built (2026-08-25), and it was indeed the cheapest thing to land first.**
`_dataio.read_vtk_Rectilinear_Grid_Scalars` and
`_dataio.write_vtk_structured_points_scalars`, with the round-trips in
`test_io_loaders.py`. Four things worth carrying forward:

- **The reader squeezes by default.** VTK datasets are always 3D, so a 2D field is
  written with a singleton z; `squeeze=True` drops any axis whose coordinate array
  has length 1, from the data *and* the grid points together so the two cannot
  disagree. `squeeze=False` returns the raw 3D form.
- **The structured-points writer refuses uneven spacing** rather than writing the
  mean, which would move every interior grid point. That is the whole reason both
  formats exist: `STRUCTURED_POINTS` carries an origin and a spacing and nothing
  else.
- **A `sep` argument** on both scalar writers picks `Omega_0042.vtk` (the
  convention of the other writers here) or `Omega.0042.vtk` (IB2d's, and what a
  field written beside a solver's own dumps needs).
- The hand-rolled `_read_scalar_vtk` helper in `test_io_loaders.py` — which existed
  precisely because `_dataio` had no scalar reader — now delegates to the real one.

**Also landed here, because the write-back needs it: the three legacy vtk readers
raise `FileNotFoundError` on a missing file.** vtk reports one only on stderr and
then hands back an empty dataset, which surfaced much later as
`AttributeError: 'NoneType' object has no attribute 'GetDataType'` out of
`numpy_support` — naming neither the file nor the cause. A per-dump series is
exactly where a missing file is a normal outcome (a run under a sliding window
writes only the dumps it loaded), so this had to be legible.

---

## 4. Component C — rendering

### 4.1 Frame rate: `fps` and `playback_rate` — **[done]**

Users set two quantities they already understand; `dt` leaves the user-facing API
entirely:

| Parameter | Meaning | Default |
|---|---|---|
| `fps` | frames per second of output — *smoothness*, comparable to standard 24/25/30/60 | `10` |
| `playback_rate` | simulated seconds per second of video — *speed* vs real time | `1` |

```
Δt_frame = playback_rate / fps
```

| `playback_rate` | `fps` | `Δt_frame` | Reads as |
|---|---|---|---|
| 1 | 30 | 0.0333 s | real time, smooth |
| 0.5 | 30 | 0.0167 s | 2× slow motion |
| 10 | 24 | 0.417 s | 10× fast forward |

This replaced a long-standing footgun. With frames pinned to steps, `fps` was the only
lever: at `dt = 1e-3`, real-time playback demanded `fps = 1000`, while the default
`fps = 10` turned 10 s of simulation (10 000 steps, hence 10 000 frames) into a
**17-minute** movie. At `dt = 1e-4` the same settings give 2.8 hours.

**Constraints that survive into archive-backed rendering:**

- **`Δt_frame < Δt_capture` is the one failure mode.** Frames cannot be produced between
  captured states. Clamp to every captured state and **warn with the numbers**,
  including the achieved rate `Δt_capture × fps`.
- **`fps ≤ playback_rate / Δt_capture`** follows. With capture-every-step,
  `Δt_capture = dt`, so **slow motion and smoothness trade off unless the capture
  interval is small**: at `dt = 0.025` captured every step, real time reaches 40 fps but
  10× slow motion caps at 4 fps. Document as: *smooth slow motion needs fine capture.*
- **Frame times are not exactly uniform.** Frames are chosen by picking, for each target
  time, the nearest available capture — so spacing jitters by up to one `Δt_capture`
  whenever `Δt_frame` is not an exact multiple of it. Warn when `Δt_frame` is only a
  small multiple (< 3×).
- **`Δt_capture` is always derived from the recorded times**, never from a nominal
  `dt` — the `span/(n-1)` `_select_frames` already computes, over the archive's capture
  times or over `time_history`, which since §2.2 hold the same states. It equals `dt`
  only at the default `capture_interval=1`; at interval *k* it is *k*·`dt`, and under a
  varying `dt` it is a mean either way. *(The interval itself is deliberately not stored
  — §2.3.)*
- **Assumption to document:** "real time" presumes simulated time is in seconds.
  `Environment.units` covers *length* only; seconds is the convention throughout.
- **`fps` is re-encodable after the fact**, because dump-cadence caching supplies any
  `Δt_frame`. Only the quiver grid and recorded quantity are fixed.
- **`per_dump=True` was deliberately dropped.** It is a second way to say what
  `playback_rate` already says, and the user can say it exactly:
  `playback_rate = np.diff(envir.flow.flow_times).mean() * fps`. **A raw step count
  (`every=k`) is rejected** — users vary `dt` between `move()` calls, so it silently
  means different things within one run.

**As built:**

- `Swarm.plot_all` gained exactly one parameter, `playback_rate=1`, immediately after
  `fps`. `frames`, if given, still overrides the selection entirely — it is an explicit
  list of history indices and always was.
- The selection is `Swarm._select_frames(fps, playback_rate)`, one private method: it
  assembles the recorded times (`time_history[:len(pos_history)]` plus the present time,
  index-aligned with what `animate(n)` expects), places the frames, and issues both
  warnings. **A first pass split this into a module-level pure function plus a method
  that fed it**; that was the wrong trade — `_swarm.py` has no module-level code at all,
  so it bought a new structural precedent and ~25 extra lines to save constructing a
  Swarm in tests that are testing a Swarm method.
- **`fps` stays at 10.** At the examples' `dt = 0.025` and `playback_rate=1`, `fps=10`
  gives `Δt_frame` exactly 4×`dt` — even spacing, silent. `fps=30` would give 1.33×,
  precisely the case the jitter warning exists to flag, so the friendlier-looking
  default would have shipped a warning on the runs people actually have.
- **Both warnings are computed from the selection itself, not from a nominal `dt`**,
  because `dt` may vary between `move()` calls. The clamp fires when `Δt_frame <
  Δt_capture` *or* when two frames would land on the same recorded state; the jitter
  warning fires when achieved spacing departs from `Δt_frame` by more than a sixth.
- **Roundoff tolerance on the clamp is load-bearing.** `playback_rate/fps` is a division
  and `time_history` accumulates `dt` by repeated addition (0.3 arrives as
  0.30000000000000004), so an exact choice like `playback_rate=0.075, fps=3` at
  `dt=0.025` compares as *just* under the recording interval. Without the `1e-9` relative
  slack it would warn and clamp on every such call — i.e. on the examples, which are
  written that way. Pinned by
  `test_frame_interval_equal_to_the_recording_interval_is_not_clamped`.
- **The first and last recorded states are always frames**, so the movie spans the run
  even when the span is not a whole multiple of `Δt_frame`.
- **On-screen playback honors the same numbers.** `FuncAnimation`'s `interval` is
  `1000/fps` ms instead of the old `dt*100` heuristic, so preview and saved movie agree.
- **Tests:** `tests/test_frame_selection.py`, 19 closed-form tests, no rendering, in the
  fast run (~0.2 s). Each drives a real, tiny run. `test_plotting_smoke.py`'s movie test
  is parametrized over `fps`, `playback_rate`, and explicit `frames`.
- **Examples updated at the call sites**: `ex_ib2d_ibmesh.py` and `ex_ib2d_sticky.py`
  `playback_rate=0.075`, `ex_ib2d_mvbnd_sticky.py` `0.15`, `ex_ind_var.py` `2`,
  `ex_sticky_seafan_3d.py` `2` — each the old `dt × fps` product, so their movies are
  unchanged. The two Vicsek examples needed no edit. `ex_ib2d_ibmesh.py`'s prose about
  "one frame per time step" was rewritten, along with the same passage in
  `docs/examples/ib2d_ibmesh.rst`; `docs/quickstart.rst` gained the model and its `dt`
  ceiling.

### 4.2 `plot_all` reads an archive — **[done]**

**The rule is about the fluid, and only the fluid: no render may trigger a fluid load
without saying so.** *(Narrowed 2026-08-18 from the original "an archive-backed render
is archive-only".)* Agent state is small, in memory, and — after §4.3 — numerically
identical wherever it is read from, so where it comes from is an availability question,
not a correctness one. The fluid is neither.

**`Swarm.plot_all` and `Swarm.plot` gain one parameter, `archive=None`**, accepting a
path (`str` or `Path`), a `RunArchive`, or the handle returned by `Environment.record`
— from which the archive's own `.path` is taken, so that a redirected directory (§2.1)
cannot be missed. `plot_all`'s existing `frames=` argument is unchanged in meaning: it
indexes `time_history`, which is exactly the list of captured times (§2.2), so it
selects the same states whether or not an archive is in play.

Three modes, and they are distinguished by what is available, not by a flag:

| Mode | Agent data from | Fluid data from |
|---|---|---|
| `plot_all(archive='run/')`, or any later session | the archive | the archive / the source (§3.3) |
| live, no archive | `pos_history` / `vel_history` | re-streams, with a loud warning |
| live, recorder active | live history (the same states by construction, §2.2) | the archive / the source |

- **Frames are selected from the archive's capture times** when one is given — the
  schema's capture-time list is the authority for what can be rendered, and the capture
  spacing derived from it is the floor on `Δt_frame` (§4.1).
- **Resolve archive entries against the live histories by *time*, not by index.** They
  do coincide in the ordinary case, because history now holds exactly the captured
  states (§2.2) — but not when a swarm was added mid-run (`first_capture > 0`, §2.3) or
  when recording started after t=0, which §2.1 permits under `INUM=None`. Matching on
  `run.times` is correct in all of those; assuming index equality is correct only in the
  common one, and fails silently rather than loudly when it is wrong.
- **A quantity the archive lacks is a refusal, not a silent fluid read** (§2.8).
- **The live final-frame branch still has to be rewritten, but for one reason instead of
  three.** `animate(n)` for `n >= len(pos_history)` currently reads live state:
  `envir.time`, `_calc_basic_stats(t_indx=None)`, and — the part that bites —
  `envir.get_vorticity()` / `interpolate_temporal_flow()` with **no time argument**,
  which evaluate at the current time and can trigger a load. Left alone, that one frame
  leaks the whole "zero loads while plotting" property. The fix is to pass the time
  explicitly and source the fluid quantity the way every other frame does. The *agent*
  half of that branch may stay as it is when live state exists; in a later session there
  is no live state and the archive path supplies it.
- **A stale archive warns; it does not silently flush.** Compare `envir.time` against
  the most recent capture time and warn on a mismatch, naming `envir.flush_recording()`
  — a mismatch means captures are still buffered. Flushing is the recorder's business,
  and a reader that mutates the thing it is reading is the wrong shape.
  ⚠️ **Guard that comparison against `envir.time is None`.** A step that failed or was
  interrupted sets it to `None` (`move()`'s `except BaseException` block), and that is
  precisely the state the `plot_all=` auto-render fires in (§2.1, "it renders when the
  run raises"), so the comparison would raise on a `NoneType` at the worst possible
  moment — while reporting a crash. `_select_frames` already special-cases the same
  state and warns; follow it, and skip the staleness check rather than inventing a
  second message.
- **`Swarm.plot(t=...)` prefers the archive for a historical time** when one is
  available, which keeps a single-frame look-back from paying for a fluid load. For
  `t=None` — the current time — live state is right there and may be newer, so live wins.
  Its snapping behavior is unchanged either way: a requested `t` snaps to the nearest
  recorded state without interpolation (`Environment.time_history` live, the capture
  times from an archive).
- **With no archive after a dynamically-loaded run: still works.** Re-streams as today,
  but emits a loud one-time warning with the estimated cost — detected by `INUM` being
  set and the requested frames spanning more than the resident window. Never break a
  working workflow silently; never let someone accidentally re-stream 100 GB unwarned.
- With `INUM=None` the whole dataset is in memory, replay costs nothing extra, and
  today's random-access behavior is otherwise preserved.
- **`playback_rate=1` is the default here too.** Existing scripts produce different
  videos. Accepted as a deliberate 1.1.0 change: the old behavior *is* the footgun.

**A later session rebuilds its own `Environment`** — cheaply, since the fluid no longer
has to be re-streamed to plot — and hands the archive to `plot_all`. `plot_all` is a
`Swarm` method and needs an `Environment` for `L`, `bndry`, `ibmesh`, `units` and
`_plot_setup`; the stored `L` and `flow_points` are there to *validate* that
reconstruction, not to replace it. §2.6's provenance is what makes the rebuild
mechanical rather than a matter of finding the original script.

**As built (2026-08-27), and the interface is smaller than this section
specifies.** `plot`/`plot_all` gained **no parameter at all**. `Environment`
remembers the archive it recorded to (`_archive_path`, set by `record()` and kept
after recording stops), and a plot reads the fluid from it. So the ordinary
workflow — record in a `with` block, plot after it — needs nothing added:

```python
with envir.record('run/', fluid='vort'):
    for _ in range(steps):
        swrm.move(dt)
swrm.plot_all(movie_filename='out.mkv', fluid='vort')   # reads no fluid data
```

⚠️ **`archive=` was built as this section specifies, then removed** *(2026-08-27,
at the user's call)*. It existed to serve one case — plotting in a **later
session**, against a freshly built Environment — and that is not a plotting
problem. It is the problem of restoring an Environment and its Swarms to where a
run left off, which §2.11 designs and nothing yet builds. Solving it inside
`plot_all` would have answered it in the wrong place and in a way only plotting
could use. So the later-session row of §4.2's table waits for restore, and agent
state is always read from live history.

What the narrower interface removed, beyond the parameter: reading agent arrays
out of an archive, resolving which recorded swarm a `Swarm` is, aligning
`props_history` against capture times, the staleness warning, and
`FluidData.restore_dump_means` — which existed so a freshly built fluid could
serve the statistics box without reloading. In the same session no frame time can
have an unrecorded mean, since the run loaded every dump it visited. All of it
comes back with restore.

Five things worth carrying forward:

- **`animate`'s final-frame branch is gone, not rewritten.** §4.2 above asks for
  its fluid reads to be fixed; what happened is that the branch disappeared,
  because a frame source makes the last live state *be* the present. That deleted
  ~270 lines of near-duplicate code and, with them, **three latent bugs the
  duplicate had drifted into**, all of them also on `master`. `TODO.md`'s
  cherry-pick queue has them.
- **The figure and animation machinery is untouched.** `FuncAnimation`, blitting,
  `_plot_setup`, the axes repositioning, the returned artist lists, the writer:
  zero changed lines. Only the source of per-frame data moved.
- **What is available is decided by what is *resident*,** the same discriminator
  §3.3 uses at record time. A resident field gives the curl and the arrows
  directly; only a windowed field reads from disk, and only there can something
  be missing and a render be refused.
- **Both global scales of §3.5 are built, and the argument for skipping the arrow
  one was wrong.** It was removed on the reasoning that an archive holds extrema
  for exactly the dumps the run loaded, which is what `fmax` already covers —
  true only at the instant recording stops. `fmax` goes on growing with **any**
  later fluid access: `envir.flow(t)`, an unrecorded backdrop, or simply running
  on after `stop_recording()`. The recorded extrema do not, which is what §3.5
  meant and what makes two renders of one run agree. Restored, and pinned by
  `test_the_arrow_scale_holds_still_after_the_recording_stops`.
- **`check_against` now ignores `INUM`** when comparing provenance, and is what
  passes over an archive describing a fluid that has since been replaced.

- **A recording that stops before the run does is refused, not read past.** The
  per-dump files exist for the stretch the recording covered; frames beyond it
  have nothing to read, so `FrameSource` checks at construction that every dump
  its states need has a file. Existence is checked rather than inferred from
  `dump_stats`, since a source that shipped a complete `Omega` series has every
  file whatever the run reached. The refusal offers the three things that
  actually clear it: record the whole run, reload the fluid with `INUM=None`, or
  draw no backdrop.

**Tests:** `tests/test_archive_rendering.py`. Most drive `FrameSource` rather than
a figure; the end-to-end movie renders are slow and gated on ffmpeg. The headline
is `test_replaying_a_recorded_run_costs_no_fluid_loads`, with
`test_replaying_an_unrecorded_run_costs_a_second_streaming_pass` beside it so the
zero means something.

⚠️ **Three defects in this work were caught by the adversarial suite in
`tests/test_data_streaming/`, not by the tests above** — the arrow scale, an
off-by-one in the re-read warning's dump count, and the missing coverage check.
All three are fixed and covered here now. The suite is written from this note,
so it is worth running against any change to component C.

### 4.3 One definition of agent velocity

*(New with the reframe. Depends on §5.1 and must be decided with it.)*

Once agent velocity can be read from two places — live history and an archive — the two
must agree, and today they would not. §5.1 fixes the derivation so that both read the
recorded velocity. Two consequences to accept deliberately:

1. **The zero-at-index-0 display convention goes away.** `Swarm.__init__` sets
   `velocities` to the local fluid drift when a flow exists, so the first frame's agent
   statistics become that drift instead of zeros. This is the truth, it is meaningful
   (it is the velocity the agents actually have), and `CLAUDE.md`'s rule that
   correctness outranks reproducing previous output applies. For flow-free runs the
   initial velocities are zeros anyway and nothing changes.
2. **The change is visible and silent-looking** — a first frame whose statistics
   changed with no other visible cause. It gets a changelog line (§7).

Do this *before* the archive writes anything, so no archive is ever recorded against the
losing convention.

### 4.4 Video output and containers

`plot_all` is the sole video producer and already streams: `Animation.save()` internally
uses `writer.saving(...)` + `grab_frame()` per frame, so encoding memory is O(one
frame). **No change is required to the video-writing machinery at all** — this component
is about where the *data* comes from, not how pixels reach ffmpeg.

**No PNG-frames option.** Every argument for one is covered better elsewhere: truncation
and mid-run inspection by container choice; crash re-render by the archive; single
publication stills by `Swarm.plot(t, filename=...)`; resume by §2.11 if it is ever
built.

**Document `.mkv` for long or unattended runs.** A hard kill (HPC walltime, OOM, node
failure) is `SIGKILL`: `__exit__` never runs, the pipe is never closed, and an `.mp4` is
then usually unplayable because ffmpeg writes the `moov` atom last. `.mkv` survives
truncation *and* is playable while still being written, which also covers checking on a
long run mid-flight. Remuxing afterwards is lossless and one call:
`ffmpeg -i out.mkv -c copy out.mp4`. Fragmented mp4
(`-movflags frag_keyframe+empty_moov`) is the alternative, passed via `writer_kwargs`.

---

## 5. Prerequisite bug fixes — **[done]**

Three defects found while reframing this plan and while planning step A. All three
predate the plan, all three are present on `master`, and all three touch state the
archive is about to persist — so they were settled first, not folded into the build.
§5.1 and §5.2 **landed 2026-08-19**; §5.3 **landed 2026-08-21**. Each came with tests, a
changelog line, and an entry in `TODO.md`'s cherry-pick queue; what follows is kept as
the record of what was wrong and why the fix is what it is.

### 5.1 `_calc_basic_stats` finite-differences positions instead of using recorded velocities

[`planktos/_swarm.py`, `_calc_basic_stats`] computes

```python
vel_data = (self.pos_history[t_indx] - self.pos_history[t_indx-1]) / (
            self.envir.time_history[t_indx] - self.envir.time_history[t_indx-1])
```

while `vel_history` sits unused except for heading arrows (`arctan2` in the plotting
code). **These are not the same quantity.** In `Swarm.move`:

1. `apply_agent_model` updates `positions`;
2. the pre-move state is appended to `pos_history` / `vel_history`;
3. `velocities` is set to `(positions - old_positions)/dt` — **from pre-boundary-
   condition positions**;
4. `apply_boundary_conditions` then mutates `positions` in place.

So `vel_history[j]` is the velocity that actually carried the agents over the interval
ending at time `j`, while `pos_history[j] - pos_history[j-1]` is the *post*-boundary
displacement. They differ for any agent that collided with an immersed boundary or a
domain boundary — and on a **periodic domain a wrap makes the finite difference a
spurious near-domain-width velocity**, which then contaminates the mean and, since
§3.1, the mean speed and its spread as well.

**Fix:** read `full_vel_history[t_indx]` (index-aligned with `pos_history` — see §2.4),
falling back to `self.velocities` for `t_indx is None` as it already does. This also
makes the live and archive-backed paths agree by construction (§4.3), which is why it
comes first.

**Delete the `elif t_indx == 0` branch in the same edit.** It substitutes a zero vector
on the reasoning that velocity is undefined before the first step; `full_vel_history[0]`
is the initial local fluid drift and is the truth. §4.3 decides this and carries the
consequences — it is listed separately there because it is the user-visible half, but
there is one line of code and it changes here.

- **Applies to `master`:** the defect does, at its `_swarm.py:1935` — but as a **port,
  not a hunk**: `master`'s `_calc_basic_stats` returns only `avg_swrm_vel` and computes
  it separately in each branch. Filed under **1.0.3**, which is prepared but not yet
  tagged, so it is still open (`TODO.md`, cherry-pick queue).
- **As landed.** `vel_data` reads `self.full_vel_history[t_indx]`, the `t_indx == 0`
  branch is gone, and the docstring says both why the history is read and what index 0
  now reports. Three tests in `tests/test_flow_interface.py`:
  `..._agent_velocity_at_initial_time_is_the_recorded_drift` (closed-form drift on the
  exactly-linear field, and *not* zero), `..._agent_speed_at_initial_time_is_zero_without_flow`
  (the same numbers the retired convention gave, now for a true reason), and
  `..._velocity_survives_a_periodic_wrap` — an agent stepping 9.5 → 0.5 across a periodic
  edge, whose reported speed is 1 and would have been 9 under the old derivation.

### 5.2 `Environment.reset()` leaves `vel_history` behind

[`planktos/_environment.py`, `Environment.reset`] clears `time_history` and each
swarm's `pos_history`, but **not** `vel_history` or `props_history`. After a reset the
lists are misaligned, so `full_vel_history` is wrong for the remainder of the session —
which reaches heading markers today and `_calc_basic_stats` after §5.1.

The FTLE stencil copy in the same file gets this right (`_environment.py:2721-2723`
clears both, with a comment explaining why), which is what makes `reset()` look like an
oversight rather than intent.

**Fix:** clear `vel_history` and `props_history` alongside `pos_history`.

- **Applies to `master`:** yes — byte-identical there. Filed under **1.0.3**
  (`TODO.md`, cherry-pick queue).
- **As landed.** All three cleared, with `props_history` left as `None` when the swarm
  was never storing it — clearing must not quietly switch the feature on. Two tests in
  `tests/test_swarm_lifecycle.py`: `test_reset_clears_every_history_not_just_positions`
  (including that the histories stay in step as the run continues past the reset) and
  `test_reset_leaves_props_history_off_when_it_was_never_on`.
- ⚠️ **§2.2's rule that `reset()` must *raise* while recording is not part of this**, and
  landed with §6.1 **A3a** (2026-08-25), alongside four other refusals — see the
  `_refuse_while_recording` guard.


### 5.3 A bare `Swarm.move()` froze the other swarms into an inconsistent history

*(Found 2026-08-21 while scoping A3, which rewrites this exact block — A3b, since the
split.)*

[`planktos/_swarm.py`, `Swarm.move`] ended its `update_time` block by freezing every
other swarm in the environment:

```python
for s in self.envir.swarms:
    if s is not self and len(s.pos_history) < len(self.pos_history):
        s.pos_history.append(s.positions.copy())      # and nothing else
```

**`vel_history` and `props_history` were not appended.** So a frozen swarm's histories
came apart and stayed apart for the rest of the session — the same failure mode as
§5.2, from a different site. Measured, two swarms, three moves of the first:

```
s1 pos/vel: 3 3
s2 pos/vel: 3 0
s2 full_pos 4  full_vel 1
```

Both consumers that pair the two by index then raise `IndexError`: `_calc_basic_stats(
t_indx=2)` (which reads `full_vel_history` after §5.1) and the `plot_all` heading arrows
(`np.arctan2(vel_history[n][:,1], ...)`). A warning was issued, but it named the wrong
problem — it said the other swarms had not been moved, not that their records had been
corrupted.

**Fix: raise instead of freezing.** *(Decided 2026-08-21.)* Advancing the environment
clock on behalf of one swarm while the others stand still is no longer supported at all.
`Swarm.move` refuses when `update_time` is true and the environment holds more than one
swarm, and points at `Environment.move_swarms`. The freeze-append and its warning are
**deleted**, not repaired.

Repairing it was the obvious alternative and is the worse one. A frozen swarm has no
velocity for the interval — it did not move, but neither did it hold still as a modelled
fact — so any value appended to `vel_history` would be an invention, and appending zeros
would flow straight into the statistics box and the heading arrows as a real measurement.
There is no half-moved state worth recording. The plan already disowns the workflow on
independent grounds (§2.2: "the manual multi-swarm pattern … is not a real workflow"),
so the archive loses nothing it wanted.

- **Applies to `master`:** yes — the block is byte-identical there. ⚠️ But it is a
  **behavior break**, a warning becoming a raise, so it is semver-visible and belongs in
  **1.1.0** rather than a patch. Logged in the cherry-pick queue with that caveat.
- **As landed.** The guard sits directly after `move()`'s existing `envir.time is None`
  check (that one keeps precedence: it carries recovery instructions for a broken state,
  where this one is a usage error). `update_time`'s docstring now says what it is
  actually for — `move_swarms` calls it, users do not — and a `Raises` section was added.
  Three tests in `tests/test_swarm_lifecycle.py`:
  `test_bare_move_refuses_when_the_environment_holds_more_than_one_swarm` (including that
  nothing was moved, recorded, or advanced on the way to the raise),
  `test_move_swarms_keeps_every_history_in_step` (which exercises the two consumers that
  used to raise), and `test_a_single_swarm_still_moves_itself`.
- ⚠️ **It surfaced a latent test bug**, which is the kind of thing this change is for:
  `test_agent_models.py::test_brownian_is_seed_reproducible_and_seed_sensitive` built one
  `Environment` outside a helper that was called four times, so it was quietly stacking
  four swarms into it. The runs were meant to be independent; the environment is now
  constructed per run.
- **Three consequences for the rest of this plan**, all simplifications:
  - §2.2's rule that capture fires from "the end of `Swarm.move` when `update_time=True`,
    and the end of `Environment.move_swarms`" is now unambiguous: with more than one
    swarm only the second path exists, so a capture can never fire against a
    partly-moved environment.
  - The multi-swarm warning is gone, so it cannot become intermittent under
    `capture_interval` (it lived inside the freeze-append, which A3b gates).
  - `full_vel_history` and `full_pos_history` are now the same length for every swarm in
    every reachable state, which is what §2.4's capture-index identity assumes.

---

## 6. Build order

### 6.1 Steps

**Step 0 — the two bug fixes (§5). [done 2026-08-19]** Independent, small, and §5.1
settled a number the archive persists. See "As landed" in §5.1 and §5.2.

**Step A — the run archive (§2).** Pure data capture: no rendering, no video
parameters, no matplotlib. Seven sub-steps, each independently testable. ✅ **All
seven are done** (2026-08-21 to 2026-08-25):

  A0. ✅ **[done 2026-08-21] Decouple collision handling from `pos_history`.** `apply_boundary_conditions`
      takes each agent's movement start point from `pos_history[-1]`; publish `move()`'s
      existing `old_positions` local instead (as `self._prev_positions`), at all
      **three** sites that take it — `Swarm.move` and the two inlined loops in
      `Environment.calculate_FTLE` — plus an `__init__` default and the docstring that
      documents the old dependency. **This touches the riskiest code in the project**
      (`CLAUDE.md`: the no-penetration invariant), so it lands first and alone. Nothing
      else in step A is safe until it is done, because `capture_interval` silently
      corrupts collisions without it. **§2.2 carries the failure analysis, the three
      failure modes, the sweep showing this is the only such site, and the verification
      argument** — read it before touching the code.

      **How A0 is validated — all of it available at A0.** The obvious test, that a
      `capture_interval=k` run reproduces an every-step run's trajectory exactly, cannot
      be written here: `capture_interval` does not exist until **A3b**, and that is
      where it now lives. What A0 can prove, it can prove more directly:

      - **The existing collision suite, bit-identical.** `test_collisions_*` pin *exact*
        post-collision positions plus a golden multi-step moving-boundary trajectory,
        and at `capture_interval=1` `self._prev_positions` and `pos_history[-1]` are
        **the same object** — so the refactor is a no-op by construction and the suite
        is the check on that.
      - **Assert the object identity, don't just argue it.** `self._prev_positions is
        self.pos_history[-1]` holds at every step today. Pinning it leaves behind a
        guard that fails the moment A3b's gating touches one append site and not the
        other — which is the way this decoupling would silently come undone.
      - **Prove the decoupling behaviorally, without `capture_interval`.** Make
        `pos_history` unusable immediately before the boundary stage — replace it with
        `[]`, or with rows of `nan` — and assert the trajectory through a mesh is
        unchanged. That is the actual claim A0 makes ("the physics no longer reads the
        recording"), it is stronger than any interval test, and nothing in it waits on
        A3b.
      - **FTLE bit-identical.** `test_analysis.py`'s closed-form forward and backward
        fields cover the two inlined `calculate_FTLE` loops, which are the two of the
        three edit sites most easily missed — they are in a different file from the one
        the change is *about*.

      **As landed.** `Swarm._prev_positions` is set in `__init__` (to the construction
      positions, since `apply_boundary_conditions` is reachable on step 1 while
      `pos_history` is still empty) and at all three loops that move agents and then
      apply boundary conditions. `apply_boundary_conditions` reads it; its docstring no
      longer documents the history dependency. No `pos_history[-1]` remains anywhere in
      `planktos/`.

      Verified three ways, all bit-identical:

      - **A 13-array numeric fingerprint** taken before and after — the static and
        moving `_ib_harness` scenarios (positions, velocities, and `ib_collision_idx`
        per step), the golden moving-boundary trajectory under both `ib_condition`s,
        and five FTLE fields covering both inlined loops (forward, backward, smallest,
        and the `swrm=` path). Bit-identical, as the same-object argument requires.
      - **The suite**, 694 passed / 2 skipped with `--runslow`.
      - ⚠️ **The new decoupling test was checked against the old coupling**, which is
        the step that makes it worth having. With `prev_pos = self.pos_history[-1]`
        restored, `test_collisions_do_not_read_the_position_history` fails exactly as
        §2.2 predicts: the poisoned history makes the collision check miss entirely and
        all four agents pass **through** the wall to the far domain edge at x=10. That
        is the no-penetration invariant breaking, reproduced on demand. A test that
        passes both before and after would have proved nothing.

      Tests live in `tests/test_swarm_lifecycle.py`, since what they pin is `move()`'s
      contract rather than any geometry: `test_collisions_do_not_read_the_position_history`
      (parametrized over sliding/sticky, collecting the trajectory from the live
      `positions` attribute rather than from the recording it is poisoning — reading the
      answer out of the history would be the very coupling under test),
      `test_prev_positions_is_the_history_entry_while_capture_is_every_step` (the object
      identity, left behind as the guard for A3b),
      `test_prev_positions_is_set_before_the_first_step`, and
      `test_ftle_sets_the_start_point_in_its_own_move_loops`.
  A1. ✅ **[done 2026-08-21] Provenance at load time** (§2.6). Each fluid and mesh
      loader, and each analytic flow generator, records its own call into `Environment`
      state. Independent of everything else, testable on its own, and easy to
      under-scope: it is a *loader* edit across ~11 methods, not a serialization detail
      of the writer. ⚠️ **Plus `Environment.__init__`**, which is not a loader but is a
      fluid entry point (`Environment(flow=[u, v], flow_times=t)`) and is the one most
      of `tests/` uses — it initializes both attributes to `None` and records the
      direct-array case as unreconstructible. Miss it and the writer raises
      `AttributeError` on the first archive recorded in a test.

      **As landed — the mechanism, which is a decorator rather than a line per loader.**
      `planktos/_provenance.py` (new, internal) holds `records_provenance(slot)`,
      `note_modifier(slot)` and `jsonable(value)`; `_environment.py` imports it and
      carries one decorator line per entry point. A decorator beat the obvious
      alternative — an explicit `self._record_provenance(path=path, dt=dt, ...)` call
      inside each loader — on three counts, and the third is the one that would have
      bitten:

      - **No drift.** The record is built from `inspect.signature`, so a parameter added
        to a loader later is recorded without anyone remembering to. A hand-written
        argument list silently goes stale, and a *silently incomplete* provenance record
        is precisely what §2.6 says must not exist.
      - **It cannot record a failure as a success**, because the wrapper records only
        after the call returns.
      - **The outer call wins when loaders nest.** Recording at the top of each method
        would let an inner helper overwrite the user's actual call.

      **A failed load clears the slot rather than leaving the previous record.** A loader
      that raises partway can leave the fluid in any state, so the honest record is
      "unknown" — and specifically not the record of whatever was loaded before it, which
      would now describe data that has been partly overwritten.

      **In-place modifiers append to `modified_by`.** `shift_ibmesh_to_match_LLC` and
      `add_vertices_to_static_2D_ibmesh` both alter a loaded mesh, and a record that kept
      claiming the mesh is exactly what the loader produced would let a reconstruction
      silently differ from the mesh the run actually used — the one failure mode §2.6
      exists to prevent. Both are deterministic given the loaded data, so replaying the
      loader and then the listed modifiers reproduces the mesh; what a reader must not do
      is replay the loader alone and assume it matches.

      **NetCDF needed a two-call record.** `load_NetCDF` opens the dataset and
      `read_NetCDF_flow` reads a field out of it; neither reconstructs the fluid alone.
      `records_provenance(..., preceded_by=...)` folds the first into the second, so
      replaying the record means replaying both in order.

      ⚠️ **Four method names in this note were wrong**, which is why A1 starts by
      listing them from the source rather than from here: `read_vtk_data` is
      **`read_IBAMR3d_vtk_data`**, `set_channel_flow` is
      **`set_two_layer_channel_flow`**, `read_vertex_data` is
      **`read_3D_vertex_data_to_convex_hull`**, and there is **no `read_npy_data`** at
      all. Eleven entry points plus `__init__`, and `tests/test_provenance.py` asserts
      structurally that every one of them is wrapped — a loader nobody decorated
      produces no error, just an environment that cannot say what it is.

      **`jsonable` records what can be recorded and marks the rest.** An ndarray records
      its shape and dtype but never its contents (those are the data this design exists
      to avoid duplicating); a callable records its name; a non-finite float becomes a
      marker, because bare `NaN`/`Infinity` are what Python's `json` emits by default and
      are not valid JSON. ⚠️ **The type checks are ordered numpy-first, and the tests
      caught this:** `np.float64` *is* a subclass of `float`, so a plain
      `isinstance(value, float)` branch ahead of the numpy ones passed numpy scalars
      straight through while claiming to have converted them. `np.bool_` and `np.integer`
      are the opposite case, subclassing neither `bool` nor `int`.

      **Sphinx was verified, not assumed.** `functools.wraps` plus `inspect.signature`
      following `__wrapped__` means autodoc renders a decorated loader exactly as before;
      the built `api/Environment.html` shows full argument lists on decorated and
      undecorated methods alike. Losing that would have silently emptied the API
      reference for every loader.

      Nothing here is user-visible — the attributes are private and nothing reads them
      yet — so A1 gets no changelog line; the entry owed at step A (§7) covers the
      feature they serve.
  A2. ✅ **[done 2026-08-21] `planktos/archive.py`: the writer.** Schema, fingerprint,
      atomic file replacement (§2.5), chunked agent writer keyed on the global capture
      index.

      **As landed.** `_ArchiveWriter` is handed data and writes it: it knows nothing
      about `Environment`, `Swarm`, time steps or hooks, and `Environment.record` (A3)
      merely drives it. That split was a deliberate constraint rather than a
      convenience — it is what keeps the format testable without running a simulation,
      and it means a later change to the capture schedule cannot reach into the format.
      Beside it: `build_fingerprint` / `fingerprint_summary` / `compare_fingerprints`,
      `_resolve_archive_path`, and `_atomic_write`.

      **The tests read the bytes back with raw `np.load` and `json.load`, not through a
      reader of our own.** A round-trip through our own code can be self-consistently
      wrong; reading the bytes pins the format. A4 gets its own tests.

      ⚠️ **Crash validity was verified with an actual kill, not a simulation of one.** A
      subprocess recorded continuously and was `SIGKILL`ed mid-run with no `close()`, no
      `flush()`, no `atexit`: **400 captures across 40 chunks came back intact**, the
      `grid.npz` on disk still matched the environment that wrote it, the recovered
      chunk indices were contiguous, and no `.partial` file was left behind. This is the property the
      whole design rests on, and it is the one that would otherwise be asserted rather
      than demonstrated.

      Four decisions the specification left open, taken here:

      - **`fsync` before every `os.replace`, chunks included.** `os.replace` alone
        survives process death, which is the common case — but node failure and power
        loss take the page cache with them, and those are exactly the runs an archive
        exists for. The cost is one sync per `chunk_size` captures, negligible against
        the physics of that many steps. A knob to disable it would be dead weight.
      - **`flush()` rewrites the open partial chunk in place**, atomically, and leaves
        the buffers alone so recording continues into the same chunk. That makes it
        idempotent and makes a mid-run plot free, which §2.1 requires.
      - **A swarm that missed a chunk entirely gets no file for it**, rather than a
        zero-row one. A zero-row file would contradict its own `first_capture`; absence
        is the honest record and the sidecar resolves the offset.
      - **A partially masked row is refused, not flattened.** A masked row means the
        agent left the domain — agents leave whole rows — so the mask is stored per row.
        Reducing a half-masked row would silently discard the evidence that an invariant
        broke upstream.

      `add_capture` also validates two things the writer is the only place that can:
      that capture indices are **contiguous** (a gap would become a gap on disk, which a
      reader could only read as a lost file), and that **exactly** the swarms whose
      `first_capture` has been reached are present.

      **`compare_fingerprints` describes both sides on any difference**, shape mismatches
      included — `array_equal` is `False` rather than an error on mismatched shapes, so
      one branch covers both, and "6 values spanning 0 to 1.4" tells a reader more than
      "shape (6,) vs (9,)". That is what §2.8's requirement for an actionable refusal
      comes down to in practice.

      **Public surface is deliberately narrow**: `RunArchive` and `load_run` (A4), plus
      the three fingerprint functions, which are what a refusal message is assembled from
      and are worth having to hand when diagnosing one. Everything else is underscored.
      The module is un-underscored because `RunArchive` is user-visible, exactly as
      `fluid.py` is un-underscored for `FluidData`.

      Nothing user-visible yet — the writer is private and unexported — so A2 gets no
      changelog line, like A0 and A1. 51 tests in `tests/test_run_archive.py`.
  **A3 is split in two** *(2026-08-24)*, because the recorder's API and the capture
  schedule have near-opposite risk profiles and mixing them makes a regression
  unattributable.

  | | Surface | Risk to existing behavior | Verified by |
  |---|---|---|---|
  | **A3a** | large — the whole recorder API | \~none; the hook is a no-op when nothing is recording | the existing suite unchanged, plus the zero-extra-loads test |
  | **A3b** | four lines in the move loop | high — changes what `pos_history` and `time_history` contain | bit-identical trajectories at `capture_interval=k` |

  **The split is clean because the capture index and the step index are different
  numbers, and only one of them needs the environment counter.** In A3a every step is a
  capture, so the recorder counting its own captures — 0, 1, 2, … — is both sufficient
  and correct. A3b has it count them *exactly the same way* and adds a predicate
  deciding whether a step produces one at all. **A3a therefore builds nothing A3b has to
  undo**, which is the test of whether a split is real or merely chronological.

  A3a. ✅ **[done 2026-08-25] `Environment.record` / `flush_recording` / `stop_recording`** — the recorder,
      capturing every step. The `open()` model: `record()` does the work immediately and
      returns a handle, `with` only adds the guaranteed close, so a bare
      `envir.record(path)` cannot silently record nothing (§2.1). The
      `_notify_step_complete` hook in `Swarm.move` (when `update_time=True`) and
      `Environment.move_swarms`, a no-op when nothing is recording. Driving
      `_ArchiveWriter`, and building the fingerprint from the environment. The swarm
      registration notification (§2.3). And **all five refusals**, which belong together
      here because every one of them is about the recorder's lifecycle rather than the
      schedule: `reset()` while recording, a second concurrent `record()`, loading a new
      fluid while recording, `record()` on a dynamically-loaded fluid whose window has
      already slid, and the `update_time=False` warning.

      **The headline test lands here** — recording a run against a windowed `FluidData`
      costs *identically* many loader calls as the same run without it (§6.2). It is the
      property the whole design exists for and it needs no capture schedule to express;
      `test_dynamic_loading.py`'s `_InMemorySource.load_calls` already counts them.

      **As landed.** `RunRecorder` in `planktos/archive.py`, driven by
      `Environment.record`; 31 tests in `tests/test_recording.py`. The headline holds:
      the loader-call sequences with and without recording are identical, with a guard
      asserting the window actually slid. The capture-index identity is exact — the
      archive reproduces `full_pos_history`, `full_vel_history` and
      `time_history + [time]` — so nothing needs translating at render time.

      Two things building it changed:

      - **Swarms are discovered at capture time, not notified from `Swarm`.** The
        §2.3 block above carries the full story; short version, both earlier answers
        were wrong and the third is better than either.
      - ⚠️ **A guard can be present in the source and still be dead.** The eight fluid
        setters were first given their refusal by pattern-matching for the end of the
        docstring — which matched the *opening* `'''` of any docstring that starts on
        its own line, putting the guard **inside the docstring**, where it reads exactly
        like working code and does nothing. Reinserted by parsing with `ast`, and
        `test_every_fluid_setter_guards_against_loading_while_recording` now verifies by
        parse rather than by grep that each guard is an executable statement. The
        failure is invisible to a text search, which is why the test has to be
        structural.

      `_refuse_while_recording()` takes no arguments and issues **one message for every
      site**: the traceback already names the call that raised, and a per-site variant
      would be one more thing to keep in step with the guard list.

  A3b. ✅ **[done 2026-08-25] The capture schedule.** Adds `capture_interval=k` to the signature, the
      `Environment` step counter, the capture-step predicate, and the history-append
      gating in `Swarm.move` and `Environment.move_swarms`. Also `move()`'s
      `except BaseException` block, whose unconditional `time_history.append` must
      become conditional or it pushes `time_history` one ahead of `pos_history` — the
      exact inconsistency that block exists to prevent. **Read §2.2 before touching
      any of it.**

      Adding the parameter only here matters: a parameter that accepts nothing but its
      default is worse than no parameter, and since nothing ships between the two
      landings there is no compatibility question in deferring it.

      **A0's forward-looking test lands here**, because this is where the thing it tests
      first exists: a run at `capture_interval=k` produces **bit-identical agent
      trajectories** to the same run captured every step. Drive it through a mesh so the
      collision path is exercised. What is recorded must not change what happens.

      **As landed.** One predicate, `Environment._records_this_step()`, gates the history
      appends and the archive capture alike, so the two cannot drift: `Swarm.move` asks
      it at the *start* of a step about the state that step begins from, and
      `_notify_step_complete` asks it again after incrementing, about the state the step
      produced — the two ends of a step, the same set of states. When nothing is
      recording the interval is 1 and the modulus is satisfied by every step, so there is
      no "are we recording" branch in the move loop at all.

      **Captures are counted from the step recording began at**, not from step zero, so a
      recording started mid-run is evenly spaced rather than short at the front. The
      interval returns to 1 when recording stops, and `reset()` returns the counter and
      the phase with the clock.

      ⚠️ **The test was checked against the defect it exists for.** With A0 reverted —
      `prev_pos = self.pos_history[-1]` restored — it fails on **5 of its 6 cases**,
      exactly as §2.2 predicts. Both geometries earn their place, and for different
      reasons:

      - the **full-span wall** diverges only in `ib_collision_idx`, not in position:
        collisions re-litigated from a stale origin resolve to a different mesh element
        while landing in the same place. Asserting on positions alone would have missed
        it.
      - the **short wall** — agents travelling around its end over several steps, which
        is the case a stale start point turns into a chord straight *through* the wall —
        diverges in **position**, leaving an agent on the wrong side. That is the
        no-penetration invariant breaking, reproduced on demand.

      *The obvious single geometry would have been the full-span wall, and it would have
      caught this only via a field most people would not think to assert on.*
  A4. ✅ **[done 2026-08-25] The reader** (`RunArchive`, `planktos.load_run`), mmap-backed, resolving by
      time and snapping — **not** interpolating — agent state (§2.7). Including it here
      makes A testable end to end without touching a line of rendering code, which is
      the natural place to cut.
  A5. ✅ **[done 2026-08-25] Docs and export** — `docs/api/RunArchive.rst` (recording,
      reading, validation, and the on-disk layout), `load_run` and `RunArchive`
      exported from `planktos/__init__.py`, and the API index updated. Sphinx builds
      clean. ⚠️ **The one warning worth knowing about:** documenting
      `Environment.record` on the archive page as well as on `Environment`'s is a
      duplicate object description, so the archive page cross-references it instead.

      **The changelog lines land here**, all of them together, per §7 — until the
      reader existed the archive was write-only, and a changelog entry announcing it
      would have described something a user could not yet use. Six lines: `record`,
      crash validity, `load_run`, `capture_interval`, the non-empty-directory redirect,
      and loader provenance.

      **As landed (A4).** `RunArchive` scans the roster from the per-swarm sidecars
      rather than from `meta.json` — which is written once at the start and so cannot
      know about a swarm that joined an hour into the run — and validates on open:
      format version, chunk contiguity, and each chunk's row count against what the
      capture count implies. A gap is a **refusal**, because chunks are written in
      order, so a hard kill costs the last buffer and never a middle one; a gap
      therefore means a lost or corrupt file, and reading around it would hand back a
      run with a hole nobody was told about.

      `check_against(envir)` is the §2.8 validation, and keeps the two questions apart
      as §2.3 settled: a **grid** mismatch refuses, naming the field that differs and
      the provenance of both sides; a **provenance** difference on a matching grid only
      warns, so replotting a run whose script moved directories is not refused.

      Open chunk files are kept in a small FIFO cache (`CACHE_SIZE = 8`), because a
      memmap holds a file descriptor and caching every chunk of a long run would
      exhaust them. FIFO rather than LRU because the access pattern that matters is a
      monotone sweep — a render walking frames in order.

**Step B — fluid-side streaming (§3). ✅ All three are done (2026-08-25).**
Independent of A except that it writes into the same directory.

⚠️ **The order was B2 → B1 → B3, not B1 → B2 → B3.** B2 is independent of
everything and testable on its own with a round-trip, and B3 needs it, so it was
the cheapest thing to land first. The list below keeps its original numbering.

  B1. ✅ **[done 2026-08-25] `fluid.py` groundwork.** Extract the gradient math from `FluidData.get_vorticity`
      into a module-level `_vorticity_from_field(flow, flow_points, periodic_dim)` —
      note the third argument, which the periodic edge fix made load-bearing — and add a
      dump-arrival observer dispatched from `_record_dump_means`. That method is already
      called at every one of the four places fluid lands in memory, with raw ndarrays
      and global time indices — the hook exists, it just needs to fan out. Riding it
      inherits the correctness argument for free, including the forward slide's
      deliberate `idx_start+2` skip of the two holdover dumps. **Two hazards:** the
      jump-to-start branch re-reports dumps already recorded, so the observer must be
      idempotent; and static flow never calls `_record_dump_means` at all, so it needs a
      one-shot capture when recording starts. The recorder must compute vorticity from
      the raw arrays rather than through `get_vorticity(time=)`, which calls `self(time)`
      and can trigger a load — exactly what is being avoided.

      Also here: a **per-source probe** for whether the fluid already carries vorticity,
      and a **per-dump reader** for it, since two of the three regimes in §3.3 need one
      and neither needs the observer to fire at all.

      **As landed.** `_vorticity_from_field(flow, flow_points, periodic_dim)` at module
      level, called by `get_vorticity` as well as by the observer, so the two cannot
      compute different fields. `add_dump_observer` / `remove_dump_observer` fan out
      from the end of `_record_dump_means`; both hazards the plan named are handled
      where it said they would be — the writer keeps a set of dump indices it has
      written, so the jump-to-start re-report is a no-op, and static flow is swept once
      at construction because the observer never fires for it. Registering the same
      observer twice is a no-op rather than a double fire. Also `FluidData.is_windowed`,
      `dump_number`, `source_dir`, `probe_stored_vorticity`, `read_dump_vorticity`,
      `write_dump_vorticity`, `get_stored_vorticity`, and `_wrap_scalar` /
      `_unwrap_scalar` — the four-line scalar wrap §3.3 said to write rather than trying
      to generalize `_wrap_flow`.

      ⚠️ **A real bug on the windowed path was found here and fixed, and it was not in
      the new code.** `load_dumpfiles` is contracted to return arrays with a leading
      time axis; two of the readers behind it drop it for a **single dump**
      (`_read_IB2d_dumpfiles` branches on `d_start != d_finish`, `_read_vtkfiles` calls
      `squeeze()`), which is right for the constructor's one-shot read and wrong on the
      streaming path. A single-dump load is not exotic: a forward slide takes `INUM-1`
      dumps until it reaches the end of the series and then takes the remainder, which
      is one dump whenever the dump count is `k*(INUM-1)+3` — with `INUM=4`, any of 6,
      9, 12, 15, … time points. It raised (a broadcast error out of
      `_record_dump_means`, ahead of a concatenate that would also have failed), so no
      result was ever silently wrong; but the run died. Fixed at the contract boundary
      in `FluidData._load_dumps`, the one method every streaming load comes
      through, and pinned in `test_dynamic_loading.py` for both loaders. **The observer would have hit it too**, which is how it surfaced — the
      committed fixtures are 8 dumps, and 8 is not of that form.
  B2. ✅ **[done 2026-08-25] Scalar rectilinear VTK I/O in `_dataio`** (§3.6). Independent of everything else
      and testable on its own with a round-trip. **Landed first**, see the note above.
  B3. ✅ **[done 2026-08-25] The per-dump fluid writer** and the extrema/means sidecars.

      **As landed.** `archive._FluidWriter` derives and writes; `archive.plan_fluid`
      decides. The split is not cosmetic: the decision has to be made **before** the
      archive directory is resolved, because it goes into `meta.json`, which is written
      once and never rewritten. So `plan_fluid` runs on the environment alone, its
      `['meta']` block goes into the metadata, and `_FluidWriter` is constructed against
      the resolved path afterwards and carries the plan out.

      `_normalize_fluid` is where `fluid=` is forced to `None` in 3D and on a flow-free
      environment — silently, and verified silent, since in neither case is there
      anything else the caller could have meant. An unknown quantity raises, and leaves
      nothing recording.

      ⚠️ **The vtk write-back is the one place `_atomic_write` could not reach**, and
      it went unnoticed until the module docstring was written and the claim checked.
      The per-source writers name their own files, so `_write_vorticity` stages into a
      `.planktos_partial/` directory beside the destination and renames out of it. §2.5
      carries why this matters more than the `.npy` case: the truncated file would land
      in the *source's* dump directory, outliving the archive and damaging a dataset
      other runs share. Pinned by three tests, two of which fail against the direct
      write.

      **Cleanup pass, 2026-08-26.** A four-angle review (reuse, simplification,
      efficiency, altitude) over the whole of B found one thing that mattered and a
      pile of small duplication. The one that mattered: **`record()` transiently
      doubled peak memory under `INUM=None`**, the default and the regime whose whole
      premise is "the dataset fits, but only just". The opening sweep called
      `get_raw_loaded_data()`, whose cubic branch is
      `np.stack([self(t) for t in self.x])` — a full second copy of the series, with
      every component's copy alive at once. Measured at 71.8 MB extra peak against a
      47.8 MB dataset. Replaced by `FluidData.iter_resident_dumps()`, which yields one
      dump at a time (0.80 MB peak, a 90× reduction) and covers the time-invariant,
      all-resident and windowed cases identically — which also deleted the separate
      static-flow branch in `_FluidWriter.__init__` and the `try/except BaseException`
      that guarded observer registration, since the sweep now happens before the hook
      goes on.

      Also from that pass: the plan dict became `_FluidPlan`, a namedtuple, so
      `vorticity_dir` means one thing and `_FluidWriter` is the single place that tells
      the fluid where its vorticity is; the three regimes moved into a flat
      `_plan_vorticity` with one `return` per row of §3.3's table; `_record_dump_means`
      grew a `_dumps_arrived` wrapper so the name matches what it does; the time-axis
      normalization moved into `FluidData._load_dumps` so that contract is
      structural rather than remembered per subclass; the archive stopped rebuilding the vorticity
      filename that `FluidData.vorticity_filename` owns; and the writer's duplicate
      means array went away in favour of the fluid's own.

      Also here: **a read surface for the fluid half**, `RunArchive.dump_stats()` and
      `RunArchive.quiver(t_idx)`. Strictly this is the reader's territory (A4) and the
      consumer is C1, but A5's lesson was that a write-only feature is not finished —
      without these, nothing but a test with raw `np.load` could read what B writes.
      Kept deliberately thin: §2.8's *render-time* refusals stay with the rendering.

      ⚠️ **`fluid=` and `quiver_shape=` join `Environment.record`'s signature here, not
      at A3** *(moved 2026-08-24)*. §2.1 shows them because it documents the *final*
      signature, which accumulates across A3, B and C — but accepting `fluid='vort'` as
      a default before this step is built would mint archives claiming a vorticity
      nothing wrote, and A4's reader would then correctly refuse to plot them (§2.8).
      Add the parameter with the thing it controls; the same argument defers
      `capture_interval` to A3b.

**Step C — rendering (§4). ✅ Both are done (2026-08-27).** The only step that touches
rendering, and it changes where per-frame data comes from rather than how it is drawn.

  C1. ✅ **[done 2026-08-27]** `plot_all` / `plot` read an archive (§4.2); rewrite the
      live final-frame branch's fluid reads. **As landed, the branch was deleted rather
      than rewritten** — see §4.2's "As built", which also carries the per-dump-mean gap
      that turned out to stand between this step and its own headline.

      ⚠️ **`plot_all=` joins `Environment.record`'s signature here, not at A3**
      *(moved 2026-08-24)*. Its whole value is that `__exit__` renders **from the
      archive** — §2.1 has it render from the handle's `.path`, which is what stops a
      redirected directory being missed. Landing it before this step would give a
      version that renders from live history and re-streams the fluid, which is
      precisely what the feature exists to prevent. The auto-render rules (§2.1) come
      with it: it fires on an exception but not on a `KeyboardInterrupt`, both still
      flush, a failure inside it must not mask the run's own exception, and a recorder
      covering more than one swarm rejects it at `record()` time.
  C2. ✅ **[done 2026-08-27]** Global colour and arrow normalization (§3.5) —
      `Swarm._vorticity_norm` already takes a `clip` it never rescales, so a global
      maximum passed there is the whole change on the rendering side. It was; see
      §3.5's "As built — the render half".

**Step R — the full-state reboot (§2.11).** The specification is §2.11; this is the
order to build it in and, first, why it goes here.

**Why ahead of tiling.** The two are independent — tiling touches `FluidData` and the
domain, reboot touches the archive format — so this is a scheduling call, not a
dependency. Three things decide it:

- **The format has to grow, and archives are being written now.** A checkpoint file, a
  Swarm-class name, `char_L`/`U`/`nu` in the environment provenance: every one is a new
  field, and every archive written before they exist is one that cannot be rebooted.
  That is a one-way door for real runs, and it is the only item on the queue that has
  one.
- **The knowledge is in hand.** §2.6's provenance was designed for exactly this, A–C
  are freshly built, and §2.11's audit is done. Tiling has been sitting behind a
  `NotImplementedError` for weeks and will keep.
- **Nothing is blocked by deferring D.** The §7 prose pass rides on tiling, so both
  slip together, and neither is on any user's path today.

The cost, stated plainly: tiling and the prose pass move back by however long this
takes, and `Environment.tile_domain` keeps raising in the meantime.

*Sub-steps, in dependency order. **R0 is done** — §2.11.5 has what it settled, and R2
and R3 below now assume it.*

- **R0 — pre-flight. ✅ [done 2026-08-31].** Baseline the suite, verify §2.11.2's state
  list against a live `Swarm`, and settle the container and history questions before
  either is baked into a format. It found that §6.3's suggested `DataFrame.to_json`
  precedent silently truncates, that `_provenance.jsonable` cannot serialize
  `shared_props`, and that `_prev_positions` needs no restoring. §2.11.5.
- **R1 — the environment gaps (§2.11.3). ✅ [done 2026-09-02].** `char_L`, `U` and `nu`
  into `provenance['environment']`, plus `ibmesh_color`. Additive, so no format-version
  bump and old archives still read; done first because it is the one part that improves
  archives written from the moment it lands. Three tests in `test_recording.py` cover
  the scalars, the `Environment(nu=…)`-only case where `nu` was the one being lost, and
  the resolved colour in both dimensions.
- **R2 — the checkpoint file. ✅ [done 2026-09-02].** One latest state per swarm, per
  §2.11.2's "State" column, in the three files §2.11.5 fixes, each replaced atomically
  the way `_FluidWriter.flush` rewrites `dump_stats.npz`. Measured at **80 B per agent**
  — 80 kB at N=1000 — so the cadence needed no cleverness: it rides the chunk boundary,
  plus `record()` and `stop()`. That bound is the point rather than a convenience — a
  hard kill costs the captures buffered since the last chunk, and the checkpoint is never
  older than that same boundary. It holds positions and velocities itself rather than
  naming a capture index, so it does not depend on the chunk the kill took.

  `RunArchive.checkpoint(swarm)` reads one back, on A5's principle that a write-only
  feature is not finished; R3 is what turns it into an `Environment` and `Swarm`s.
  Ten tests in `test_recording.py`, and three of the acceptance suite's five `xfail`s
  come off here — retargeted at the checkpoint and rewritten as round-trips, per the
  decision recorded above.
- **R3 — the reader. ✅ [done 2026-09-03].** `RunArchive.restore(history=True)` returns
  `(envir, swarms)`. It replays the recorded loader calls — including `preceded_by`
  chains and `modified_by` modifiers, both of which take no arguments and so replay by
  name — rather than deserializing anything, and distinguishes its three failure modes
  as §2.11.3 requires. **This is where the acceptance suite's last two `xfail`s came
  off, so claim 4 now holds.**

  Four things it settled that the specification had left open:

  - **The fingerprint check is skipped when the fluid could not be replayed.** A run
    built with `Environment(flow=[...])` has no loader call, so the rebuilt environment
    genuinely has no fluid — and `check_against` would then refuse with a mismatch that
    is exactly what the warning has already said. Warn, hand back the environment, and
    check the fingerprint only where a replay actually happened.
  - **Boundary conditions replay as pairs**, not as one end of each. `bndry[axis][0]`
    alone — which the acceptance test's hand reconstruction used — loses a domain that
    is periodic on one side only.
  - **`has_plot_structs` joins the environment provenance.** They are functions and
    cannot be recorded, but *whether there were any* can, which is what makes the
    warning truthful rather than boilerplate on every restore. Additive, like R1.
  - **A restored run that keeps recording writes a second archive.** `record()` on the
    directory it came from meets §2.1's non-empty-directory rule and redirects to a
    timestamped sibling. Correct by that rule and documented rather than special-cased;
    appending to an existing archive would need capture-index continuation and is a
    feature, not a footnote.

  `history=False` leaves the three histories empty, and a test pins that the physics is
  identical either way — which is R0's finding made permanent.

  ⚠️ **`restore()` sets `Environment._archive_path`** *(found and fixed 2026-09-03,
  after the rest of R3 had landed)*. That attribute is how §4.2's `FrameSource` finds
  the archive, and `record()` was the only thing setting it — so a restored run plotted
  its fluid by **re-reading the dataset**, the one cost components B and C exist to
  remove, and silently, since with no archive linked there is nothing to warn about.
  Frames past what the recording covers are still refused where they are asked for, so
  linking it is correct in both directions.
- **R4 — the derived quantities and the opt-in histories.** Two halves, and the first
  is not optional:

  - **The derived quantities, always on**: the per-capture statistics sidecar and the
    stored `angle` column (§2.11.5). These are what let `store=` drop velocities, so
    they land *with* that default change, never after it — an archive minted in between
    would claim a default it cannot serve.
  - **`store=` becomes `('positions',)`** (§2.4), with the printed notice at
    `record()`. Rewrite the existing warning, which says the opposite and becomes false
    here. **No changelog line is owed for the reversal itself** — `Environment.record`
    is new in the unreleased 1.1.0, so there is no shipped default to have changed; the
    feature's own entry simply describes what it does.
  - **The opt-in series**: `store=` extends to `props`, `shared_props`, `rndState` and
    the sparse `ib_collision_idx` events, and `props_history` becomes the restore's
    opt-in series (§2.11.4). The sparse format is the one piece here that wants
    measurement before it is fixed.

  Last because R1–R3 deliver the reboot claim without any of it.

- **R5 — appending to the archive a run was restored from.** *(Specified 2026-09-03,
  after R3 shipped and the question "why does resuming write a second directory?" turned
  out to have a good answer.)* Today `record()` on the directory a run came from meets
  §2.1's non-empty rule and redirects to a timestamped sibling, so a resumed run sits
  beside its own history rather than continuing it. Restoring already carries everything
  needed to continue instead.

  **The trigger is a checkable fact, not a remembered one: append when the archive's
  last capture is exactly where the Environment now is** — `envir.time ==
  archive.times[-1]` — and `store`, `chunk_size` and `capture_interval` all match
  `meta.json`. That is better than "this Environment came from a restore" three ways: it
  is verifiable from state; it fails safe, since restoring and then running before
  recording would otherwise punch a hole in the series, and §2.8 already makes a partial
  series a refusal rather than a silent fill; and it picks up the notebook workflow §2.1
  already cares about, where `stop_recording()`, a look at the data, and a second
  `record()` become one continuous archive. **Nothing already in the archive is ever
  rewritten except the tail chunk**, which is the one piece that has to grow.

  *What it needs:*

  - **Refill the last chunk rather than starting a new one.** `_validate_chunks`
    requires chunk *i* to hold exactly
    `min(n, (i+1)·chunk_size) − max(first_capture, i·chunk_size)` rows, so a short chunk
    in the middle of a series is refused. Read its rows back into the buffer and carry
    on. Bounded by one chunk.
  - **Do not take capture 0**, which `RunRecorder.__init__` otherwise always does — it
    would duplicate the archive's last capture at the same timestamp.
  - **Do not rewrite `meta.json`, and validate the roster instead of adding to it.**
    `_ArchiveWriter.__init__` writes the metadata and `add_swarm` raises on a duplicate
    index; both need an append path. A swarm joining *during* the appended stretch is an
    ordinary mid-run swarm and needs nothing new.
  - **Bypass `_resolve_archive_path`'s redirect**, which is the whole point.
  - **Seed `means` from the existing sidecar**, which is the one fluid array that is
    still per dump. The extrema are single running values as of 2026-09-03 (§3.5), so
    combining them is one `max` — that simplification was made for this step and removes
    what was otherwise the only silent-data-loss hazard here. Seed `_written` too, so
    vorticity already on disk is not written again.
  - **Refuse a mismatch** in `store`, `chunk_size` or `capture_interval` rather than
    redirecting: silently starting a second archive when the user asked to append is the
    confusing outcome, and a changed `capture_interval` makes the timeline unevenly
    spaced halfway through.

  *The headline test, and the reason to trust the feature:* **a run recorded, stopped,
  restored and appended must produce an archive byte-identical to the same run recorded
  in one go.** If that holds, every consumer is automatically correct and nothing else
  needs arguing.

  *Where it goes:* after R4, which changes the `store=` default and adds a sidecar the
  append path has to know about. The §3.5 extrema simplification is a prerequisite and
  landed first, on 2026-09-03.

*What Step R is finished against:* `tests/test_data_streaming/test_stream_d_restart.py`.
Its five strict `xfail`s were the acceptance criteria, and the headline —
`test_a_run_resumes_from_disk_as_if_nothing_had_happened` — is the whole of R stated as
one assertion. **All five are cleared as of R3 (2026-09-03)**, each with the sub-step
that earned it. §2.11.5 records how: three asserted a file location this plan had since
decided differently, so they were retargeted rather than merely un-`xfail`ed and
rewritten as round-trips; a fourth was tightened, since it checked for an attribute name
that handing back state satisfies without rebuilding anything. The retargeted checklist
is **scaffolding and gets deleted once Step R is confirmed done**, R4 and R5 included.

⚠️ **Those markers being gone does not mean Step R is finished.** R4 and R5 are still
ahead of it; what the cleared list means is that the *claim* holds, not that the step is
closed.

**Step D — examples and docs prose pass (§7).**

*Two earlier versions of this list are worth not repeating.* One had "stream the video"
as an independent step, on the false premise that `plot_all` held frames in memory (see
§1.1's correction). The other had "extract a shared frame renderer" as a prerequisite,
which §2.9's capture/render split removes: with exactly one rendering path there is
nothing to share.

### 6.2 Tests

New `tests/test_run_archive.py`, except the loader-count assertion, which belongs beside
the machinery it counts in `test_dynamic_loading.py`.

**The headline test** ✅ **[A3a]** is that recording a run against a windowed
`FluidData` costs *identically* many loader calls as the same run without it. That
single assertion is the property the whole design exists for.
`test_recording.py::test_recording_costs_no_extra_fluid_loads`, with a guard that the
window actually slid — otherwise it would pass against a dataset that never streamed.

**The second one makes §3.2 executable** ✅ **[B3]**: under `INUM=int`, per-dump vorticity blended
to a time between dumps equals `envir.get_vorticity(time=t)` computed live, to
round-off — for both the sourced and the written case, which must agree with each other
as well. Under `INUM=None` there is nothing to compare, since the render calls
`get_vorticity` itself; what to assert there is that **no vorticity file was written
anywhere**, which is the regime's whole content and would otherwise fail silently by
costing disk nobody asked for.

**The third belongs to A3b and is the one that protects the physics** ✅ **[A3b]**: a
run at `capture_interval=k` produces **bit-identical agent trajectories** to the same
run captured every step. What is recorded must not change what happens. *(It reads like
an A0 test and was filed there through several drafts, but `capture_interval` does not
exist until A3b — §6.1 A0 lists the four checks that do land with the refactor itself,
one of which proves the decoupling more directly than this one does.)*

⚠️ **Drive it through two meshes, not one.** §6.1 A3b records what validating this
found: against the reverted A0 coupling, a wall spanning the domain diverges only in
`ib_collision_idx` while positions match — so the obvious single geometry catches the
defect only via a field most people would not think to assert on. A *short* wall, which
agents round the end of over several steps, diverges in position instead. Assert on both
positions and collision indices, over both geometries.

Round out with the following, grouped by the step that makes each expressible. ✅ marks
what has landed. The writer tests (`tests/test_run_archive.py`) drive the format directly
with synthetic arrays and no simulation; the recorder tests (`tests/test_recording.py`)
drive real runs.

**Landed with A2 — the format:**

- ✅ chunk boundaries at exactly one, one-plus-one, and a partial chunk;
- ✅ **a hard-kill simulation** — write chunks, then read without any finalizer having
  run (§2.5). This is the crash-validity property and nothing else tests it. *As built
  it goes further than a simulation: a subprocess is actually killed mid-recording;*
- ✅ **a truncated chunk file must not be produced** — assert the writer's temp-then-
  replace discipline, e.g. that no partially-written `.npy` is ever visible under the
  archive path (§2.5);
- ✅ a non-empty target directory redirecting to a timestamped sibling, with the
  handle's `.path` and the warning both naming it (§2.1);
- ✅ chunk files recovered in **numeric** and not lexical order;
- ✅ two swarms with the same default name recorded without collision (index access and
  the name-ambiguity raise are §2.7, so they belong to **A4**);
- ✅ a masked agent round-tripping as masked, and a partially masked row refused;
- ✅ a corrupted `grid.npz` caught by the zip CRC, which is why the archive carries no
  checksum of its own (§2.3).

**Landed with A3a — the recorder:**

- ✅ a round-trip against `pos_history` / `vel_history` / `time_history`;
- ✅ the fingerprint matching the environment that wrote it, and differing from a
  differently-gridded one in a way that names the field;
- ✅ a raise mid-run leaving a complete-to-that-point readable archive;
- ✅ a plain `swrm.move()` inside a recording capturing exactly one state per step, and
  `move_swarms` capturing once per *step* rather than once per swarm;
- ✅ a swarm added mid-recording: its `first_capture` and its short first chunk.
  **Both spellings** — `envir.add_swarm(...)` and `planktos.Swarm(envir=envir)` —
  since a hook on either would have been wrong; the recorder discovers swarms at capture
  time instead (§2.3);
- ✅ `store=` omitting velocities warning at `record()`, and omitting positions raising
  (§2.4);
- ✅ restricting capture to a subset of swarms with `swarms=`;
- ✅ `calculate_FTLE` firing no captures, and contributing no swarm;
- ✅ the five refusals, which landed together because they are one concern: `reset()`
  while recording; a second `record()` on an environment already recording (§2.1);
  loading a new fluid while recording (§2.3); `record()` on a dynamically-loaded fluid
  whose window has already slid, and **not** raising under `INUM=None` (§2.1); and
  `move(update_time=False)` warning while a recorder is active (§2.2). Plus a
  **structural** check that every fluid setter's guard is an executable statement rather
  than text inside a docstring — a failure invisible to grep, and one that happened.

**Landed with A3b — the capture schedule:**

- ✅ `capture_interval=k` giving `len(time_history) == len(pos_history) ==
  len(vel_history)` and a capture count of `steps//k + 1`, with `time_history` holding
  only captured times, over several (steps, k) combinations including k not dividing
  steps (§2.2);
- ✅ capture *j* equalling `full_pos_history[j]` under a coarse schedule, which is the
  identity the whole design rests on;
- ✅ captures spaced exactly *k*·`dt` apart — the "as if `dt` were larger" framing,
  made executable;
- ✅ a failed step under `capture_interval=k` leaving the histories consistent (§2.2);
- ✅ `move_swarms` under `capture_interval=k` keeping every swarm's histories the same
  length as `time_history`;
- ✅ the histories returning to every step once recording stops — gating must not
  outlive the recorder;
- ✅ a recording started mid-run being evenly spaced, since captures count from the step
  recording began at rather than from step zero;
- ✅ `capture_interval` below 1 refused, leaving nothing gated.

**Landed with A4 — the reader** (`tests/test_run_reader.py`)**:**

- ✅ **reading one capture reads one chunk** — `np.load` is watched, and asking for a
  capture out of a ten-chunk archive must open exactly two files, the positions chunk
  and its mask. This is what makes §2.10 possible, and asserting on the count is what
  makes it a test rather than a hope;
- ✅ chunks are memmapped rather than read, and the open-file cache stays bounded;
- ✅ a `CaptureSeries` is **not** an ndarray, and hands back real masked arrays;
- ✅ a deliberately removed middle chunk refusing rather than short-reading, for both
  the time base and a swarm's own chunks, plus a chunk whose row count contradicts the
  capture count (§2.3);
- ✅ two swarms with the same default name: index access works, name access raises;
  a unique name resolves (§2.7);
- ✅ a swarm added mid-run coming back front-padded with masked rows, aligned to
  `run.times` (§2.3, §2.7);
- ✅ a fingerprint refusal against a differently-gridded environment naming the field
  that differs and the provenance of both sides, and a provenance-only difference
  warning rather than refusing (§2.8);
- ✅ an array that was not stored refused by name; a future format version refused; a
  directory that is not an archive refused;
- ✅ **reading never writes** — every file's mtime is unchanged across a full read,
  since a reader that mutates what it reads is the wrong shape (§2.7).

**Landed with B — the fluid half** (`tests/test_fluid_recording.py`, 47 tests;
plus the `_dataio` round-trips in `test_io_loaders.py` and the slider additions in
`test_dynamic_loading.py`)**:**

- ✅ the jump-to-start re-report leaving no duplicate fluid files — asserted by
  watching `write_dump_vorticity` itself, since a duplicate write would produce
  identical bytes and be invisible on disk;
- ✅ `fluid=` forced to `None` on a flow-free environment rather than failing, and
  in 3D, and **silently** in both (checked under `simplefilter('error')`);
- ✅ each of §3.3's three regimes selecting correctly, including that `INUM=True`
  lands with `INUM=None` and not with the streaming case;
- ✅ **no vorticity file anywhere** under a resident field — the top row's whole
  content, which would otherwise fail silently by costing disk nobody asked for;
- ✅ a blended read equalling the live curl, sourced *and* written, and the two
  agreeing with each other — plus a test that the blend is **not** the nearest
  dump, without which the first two would pass against a reader that snapped;
- ✅ a blended read refused under cubic splining, and clamped rather than
  extrapolated outside the data's bounds (matching `FluidData.__call__`);
- ✅ a partial source series warned about, left untouched, and written past;
- ✅ the fallback to `run_archive/fluid/` when the source cannot be written, with
  `meta.json` naming which happened;
- ✅ the two-slot read cache staying at two, and a monotone sweep in **either**
  direction reading each dump exactly once — which is what eviction-by-distance
  buys over eviction-by-age;
- ✅ the sidecar's rows being the reduction of their own dump and not a
  neighbour's, agreeing with `FluidData._dump_means`, and `NaN` for a dump that
  never loaded;
- ✅ the sidecar written in 3D, carrying no `vort_absmax` there — §0.2's 3D
  deliverable, and the only part of B a 3D run gets;
- ✅ quiver opt-in, storing exactly the strided slice `plot_all` draws, with the
  strides resolving the requested grid and never falling below 1;
- ✅ **recording the fluid costs no extra loads** — the loader-call sequence with
  `fluid='vort'` is identical to the same run with nothing recording. The B
  counterpart of A3a's headline, with the same guard that the window really slid;
- ✅ the observer unhooked when the recording stops, so it cannot outlive the
  archive it was writing for, and its staging directory removed with it;
- ✅ **a written vorticity file appears whole or not at all** — a write that fails
  partway leaves nothing in the source's dump directory and nothing the probe's
  own glob can see. Checked against the direct write, where the truncated file
  lands in the series;
- ✅ time-invariant flow captured once, since the observer never fires for it.

**Landed with C — the rendering** (`tests/test_archive_rendering.py`, plus the
moving-mesh regression in `test_plotting_smoke.py`)**:**

- ✅ **the headline: replaying a recorded windowed run costs zero loader calls**,
  with the same replay unrecorded costing a full second pass beside it so the
  zero means something, and a guard that the replay is more than one frame;
- ✅ the archive found without being named, including when recording redirected
  to a timestamped sibling — the path a plot has to read is the resolved one;
- ✅ an archive describing a fluid that has since been replaced reported and
  passed over, rather than serving statistics for the wrong dataset;
- ✅ the refusals: each backdrop asked for and not recorded, in both directions,
  so neither can be derived from the other; and that a resident field is refused
  nothing, since what is available follows from what is in memory;
- ✅ the re-read warning fired for a whole-run replay and **not** for a
  single-frame look-back, nor for a resident field;
- ✅ a blended backdrop equalling the curl of the velocity in use, sourced *and*
  written, and strided arrows equalling the strided slice of the interpolated
  field — §3.2's linearity through the render rather than through the writer;
- ✅ the stored-quiver read cache staying at two and reading each dump once on a
  monotone sweep;
- ✅ the colour limit equal to `nanmax(vort_absmax)` and never rescaled by a later
  frame; two renders of different stretches of one run agreeing with a recording
  and disagreeing without one; a NaN row not poisoning it; an explicit `clip`
  used as given; the arrow scale equal to `nanmax(vmax)` and **unmoved by a fluid
  access after the recording stopped**, where `fmax` moves;
- ✅ frames reaching past the dumps a recording covers refused at construction,
  and the re-read warning counting the dumps that exist rather than one more;
- ✅ the stored quiver grid winning over the figure's with a warning, not warning
  when the two are close, and the figure choosing when the field is resident;
- ✅ **the final frame drawing the present state** — its offsets, its time text
  and its per-agent colours read off the real artists, since that frame used to
  be drawn by a branch of its own;
- ✅ `plot_all=` refused at `record()` for more than one swarm and for a non-dict,
  leaving nothing recording; its dict passed through untouched; firing on an
  exception but not on a `KeyboardInterrupt` (which still flushes); a failure
  inside it warning rather than masking the run's exception; and a swarm joining
  later leaving the movie unmade;
- ✅ end to end with ffmpeg, since only a real encoder walks every frame: 2D with
  each backdrop, `dist='hist'`, `downsamp`, explicit `frames`, 3D, and the
  auto-render.

**Verification:** `pytest` (fast) plus `pytest --runslow` for the plotting smokes, which
exercise `plot_*` on the Agg backend and will catch signature breakage. The movie test
additionally needs ffmpeg on `PATH`.

### 6.3 Entry points for a cold start

Line numbers drift; search for the names.

**Step 0 / §5:**
- `Swarm._calc_basic_stats` (`planktos/_swarm.py`) — the `vel_data` derivation, and note
  **eight** unpack sites consume its tuple (`grep -n "_calc_basic_stats" planktos/_swarm.py`).
- `Swarm.move` — the `old_velocities` / `velocities` / `apply_boundary_conditions`
  ordering that §5.1 turns on.
- `Environment.reset` (`planktos/_environment.py`) — §5.2, and §2.2's raise (both landed; `reset` now also returns the step counter and capture phase).

**Step A — what A0–A3b built** (for reading the code, not for building it again):
- `planktos/archive.py` — the format and the writer (`_ArchiveWriter`), the fingerprint
  functions, and `RunRecorder`, which `Environment.record` returns.
- `planktos/_provenance.py` — `records_provenance` / `note_modifier` / `jsonable`, the
  decorators every fluid and mesh entry point carries.
- `Environment.record` / `flush_recording` / `stop_recording` /
  `_notify_step_complete` / `_records_this_step` / `_refuse_while_recording`
  (`planktos/_environment.py`).
- `Swarm._prev_positions`, set in `__init__`, in `Swarm.move`, and in both inlined loops
  in `Environment.calculate_FTLE` — the movement start point, which
  `apply_boundary_conditions` reads instead of `pos_history[-1]` (A0).
- `Swarm.move`'s `keep_state`, asked once at the top of a step and used for all three
  history appends, for the `time_history` append in the `update_time` block, **and in
  the `except BaseException` block** — which must gate too, or it closes the histories
  off inconsistently, the exact thing it exists to prevent.

**Step A — still to build (A4, A5):**
- `Swarm.full_pos_history` / `full_vel_history` and `Swarm._select_frames` — the index
  convention the reader has to match. Capture *j* is `full_pos_history[j]`.
- `Swarm.save_data` — the precedent for "props_history is not saved", and the existing
  model for a directory of run output; also the §2.11 checkpoint's nearest relative
  (it already writes `props` to json and `shared_props` to npz).
- `Swarm.__init__` — `rndState`, `store_prop_history`, `ib_condition`: the checkpoint
  inventory in §2.11.
- `archive._chunk_index_of` and `archive.compare_fingerprints` — what the reader scans
  and validates with; both exist already.

**Step B:**
- `FluidData._record_dump_means` (`planktos/fluid.py`) — the dump-arrival hook, called
  from `__init__` and from all three load sites in `update_spline`.
- `FluidData.get_vorticity` — the gradient math to extract. Two things to carry with it:
  the time-invariant branch, which is the case that never calls `_record_dump_means`;
  and `fluid._spatial_gradient`, which it calls per axis with `periodic_dim[axis]`, so
  an extracted `_vorticity_from_field` needs `periodic_dim` as an argument.
- `IB2dData._read_IB2d_dumpfiles` — its reference comment block names every quantity
  IB2d writes (`Omega`, `P`, `uMag`, `Fx`, `Fy`), and the `uX`/`uY` branch beside it is
  the scalar read path a vorticity reader reuses unchanged.
- `_dataio.write_vtk_2D_rectilinear_grid_scalars` — the existing scalar *writer* (used by
  `Environment.save_2D_vorticity`); the matching reader is what §3.6 says is missing.
- `_dataio.read_2DEulerian_Data_From_vtk` / `read_vtk_Structured_Points` — what must read
  back whatever gets written, unchanged, for the interoperability claim to hold.

**Step C — what C1 and C2 built** (for reading the code, not for building it again):
- `planktos/_frames.py` — `FrameSource`, which settles what the states are and where
  the fluid backdrop comes from. Everything else in step C is a call into it.
- `Environment._archive_path` — set by `record()`, kept after recording stops. This is
  how a plot finds the archive without being told.
- `Swarm.plot` and `Swarm.plot_all` (`planktos/_swarm.py`) — the two callers.
  `animate(n)` no longer has an `n >= len(pos_history)` branch at all.
- `Swarm._quiver_strides` — the figure-derived arrow density, now what a stored grid
  is compared *against* rather than what is used.
- `archive.RunRecorder._auto_plot` and `archive._check_plot_all` — `plot_all=`.

---

## 7. Obligations

**Changelog (1.1.0)**, all user-visible relative to 1.0.x:

- **[done]** fluid speed statistics replaced by agent-speed spread on plots;
- **[done]** `playback_rate` added and defaulting to 1, changing existing video output.
  One line; `fps`'s default did not change and `per_dump` was not built, so neither is
  changelog material;
- **[done]** vorticity backdrops no longer flashing (symmetric, non-shrinking colour
  limits; a supplied `clip` honoured). Filed under 1.0.3;
- **[done]** vorticity differenced across the wrap on periodic dimensions, changing the
  outermost ring of every vorticity plot. Filed under 1.0.3;
- **[done]** the `_calc_basic_stats` velocity fix (§5.1), on two lines: recorded
  velocities replacing differenced positions, and the initial frame showing the starting
  fluid drift instead of zero (§4.3). The `reset()` history fix (§5.2) took a third.
  All three are filed under **1.0.3**, which is prepared but not yet tagged and so is
  still open — `reset()` ported to `master` as a clean hunk, `_calc_basic_stats` as a
  genuine port, since `master`'s version of that method is shaped differently;
- **Owed at step A, and ⚠️ written at A4/A5 rather than as each piece lands.** The
  feature line is `Environment.record`, the run archive, and `planktos.load_run` — but
  until the reader exists the archive is **write-only**, and a changelog entry announcing
  it then would describe something a user cannot yet use. A0, A1, A2 and A3a are all
  invisible from outside for the same reason and none of them takes a line on its own.
  Dyload-only in its fluid half — it depends on `FluidData` — but the *agent* half does
  not, so check portability with `git diff master -- <file>` before assuming the whole
  feature is dyload-only. Two further lines belong to step A in their own right:
  **`capture_interval`**, which changes what `time_history`, `pos_history` and
  `vel_history` contain and so is visible to any script that indexes them; and the fact
  that recording into a **non-empty directory redirects to a timestamped sibling**
  (§2.1). ⚠️ *An earlier version of this bullet filed `capture_interval` at A3b, on the
  grounds that it is observable without a reader. True but beside the point: it is only
  **reachable** through `record()`, which is itself unannounced until A4/A5, so a line
  describing it would leave a reader asking where to set it. All three land together.* A0 is an internal refactor with no behavior change and gets no line —
  if it changes a trajectory, that is a bug, not a changelog entry;
- **Owed at step B:** if Planktos writes vorticity into the source directory (§3.3) that
  is user-visible in its own right and needs its own line;
- **[done]** step C: plots reading what a recording wrote, the colour limit becoming
  global rather than growing with the frames drawn, and `record(plot_all=)`. Three
  lines under 1.1.0, plus two bug-fix lines the merge of `animate`'s two branches
  turned up. All the fixes are `master`-applicable and are in `TODO.md`'s cherry-pick
  queue.

**Docs:** **[done]** the `fps`/`playback_rate` model and its `dt` ceiling, and the
seconds assumption, in `docs/quickstart.rst`. Still owed: `.mkv` guidance for long runs;
what the archive stores and when it is refused; and an API page for `RunArchive` /
`load_run` (§2.7) — a public class needs one, unlike everything else in this plan.

**Examples.** The call sites are done (§4.1 "As built") — each example names its
playback rate explicitly, chosen to be the old `dt × fps` product, and the stale "one
frame per time step" prose in `ex_ib2d_ibmesh.py` and its docs page is rewritten. What
remains is the wider prose pass. Current effective playback rates show the footgun's
fingerprint — a 27× spread with no evident intent:

| Example | `dt` | `fps` | Effective rate |
|---|---|---|---|
| `ex_ib2d_ibmesh.py` | 0.025 | 3 | 0.075 — 13× slow motion |
| `ex_ib2d_sticky.py` | 0.025 | 3 | 0.075 — 13× slow motion |
| `ex_ib2d_mvbnd_sticky.py` | 0.025 | 6 | 0.15 — 6.7× slow motion |
| `ex_ind_var.py` | 0.1 | 20 | 2.0 — 2× fast forward |

Under the old scheme `Δt_frame = dt` identically, so the effective rate was just
`dt × fps` — users could only choose `fps`, and the playback rate fell out wherever it
fell. That is why the spread is incoherent: nobody chose these rates. Stating the rate
they already had was the conservative starting point; the fluid examples genuinely want
slow motion for legible vortices. The real constraint when re-timing: at `dt = 0.025`,
`playback_rate = 0.075` permits at most 3 fps, so a smoother version of those examples
needs a **smaller `dt`**, not a different `fps`.

**`TODO.md`:** **[done]** the optional-history-retention item now records that
`capture_interval` subsumes most of it, leaving only the `store_pos_history=None`
residue (§8).

---

## 8. Deferred

- **Async frame writing.** Matplotlib rendering is slow and currently serializes with
  the physics. Matplotlib is not thread-safe, but rendering in the main thread and
  handing only the encode/write to a writer thread would hide most of the I/O cost.
  **Measure before building** — it may be irrelevant next to the physics.
- **A live one-pass render mode** (rendering without an archive). Only meaningful if a
  workflow appears that cannot afford the archive; it inherits the colour-normalization
  problem (§3.5).
- **Checkpoint / restart** (§2.11) — designed for, not built.
- **History-free running** (`store_pos_history=None`): keep no `pos_history` at all and
  rely on the archive. A0 removes the collision-path obstacle (§2.2) and
  `capture_interval` covers the rest of the `TODO.md` maybe-feature (§2.10), leaving
  only this residue — which needs its own pass, because live `plot_all` and
  `_calc_basic_stats` would then have nothing to read without an archive.
- **`save_*` re-expressed as archive exports** (§2.10) — deliberately not first-pass.

---

## 9. Component D — tiling and `extend`, as cleanup afterwards

Done once, for 2D **and** 3D together, with `tests/IBAMR_test_data/` available to verify
the 3D path end-to-end. This is the last thing gated off by the `FlowArray` removal
(Appendix A) that has not come back.

**It pairs naturally with the work above:** a tiled quiver wraps coordinates the same
way the interpolator does, so plotting never materializes a tiled array — and doing the
plotting work first gives tiling a working renderer to validate tiled visualization
against.

### 9.1 The design: position-wrapping, memory-free, dimension-agnostic

A tiled domain is a periodic extension, and `interpolate_flow` *already* implements
periodic extension by wrapping query positions (`positions[:,n] % flow_points[n][-1]`).
So tiling never needs a big array on any hot path:

- store `tiling = (tx, ty[, tz])` + the one base tile on `FluidData`;
- interpolation wraps agent positions into the base tile, then `interpn` against the base
  tile — identical in 2D and 3D;
- vorticity / `|u|`-gradient over a tiled domain = the base tile's field replicated, so
  compute on the tile and replicate the *result* only if a consumer needs the full field;
- reported `.shape` / domain extent = arithmetic on `base_shape × tiling`, no allocation.

**The naming rule, adopted when tiling was gated off and still binding:**

> **Public geometry (domain `L`, plot extent, reported grid shape) reflects the *tiled*
> domain. Stored data and all interpolation use the *base tile*. Nothing materializes
> the full tiled grid-data array on a hot path.**

If a consumer ever truly needs the big array, that is one explicit
`materialize_tiled()`-style method that documents its memory cost.

**Reconcile with `periodic_dim`:** a tiled dimension is effectively periodic for
interpolation. The implementation must define the interaction of `tiling` and
`periodic_dim` explicitly (a tiled dim implies wrapping regardless of the `periodic_dim`
flag for that dim), and test it.

**Revisit `extend` here.** `Environment.extend` (pad the fluid domain with copies of the
edge values) was removed on `dyload` in favor of extrapolation. Decide at this point
whether to bring it back for the specific fluid fields where padding is the physically
right answer — it is the same class of operation as tiling (reported domain ≠ stored
grid) and should share the mechanism rather than re-materializing arrays. If it returns,
un-skip `test_extend_grows_domain_and_copies_edges` in `tests/test_flow_generation.py`.

### 9.2 Why nothing may be virtualized — the constraint that killed the last attempt

**Do not reintroduce a materializing tiling stopgap, and do not reintroduce a virtualized
one either.** Both were tried; here is why each failed, because the failure modes are not
obvious and the second one is genuinely surprising.

**Virtualization is defeated by modern scipy.** The deleted `FlowArray` was an `ndarray`
subclass that overrode `.shape` and `__getitem__` to report a virtual `k×` grid off one
stored tile, so that `interpn` / `np.gradient` / matplotlib could index it as if it were
the big array. NumPy offers `__array_ufunc__` and `__array_function__`, but **neither can
intercept a bare `values.shape` read or a `values[index_tuple]` index** — which is
exactly what those consumers do — so overriding on the subclass is forced. That is the
path the NumPy docs warn against: C-level numpy code reads the true shape from the array
struct, not the Python property.

And it does not work. `scipy.interpolate.RegularGridInterpolator._check_values` calls
`np.asarray(values)` on any array-API object, **materializing the real (untiled) buffer
and discarding the virtual `.shape`/`__getitem__` entirely** (verified on scipy 1.17.1 /
numpy 2.4.6). So the tiled interpolation path never actually worked — no test ever called
`interpolate_flow` after `tile_domain`, which is why the breakage went unnoticed for as
long as it did. Meanwhile the subclass corrupted ordinary numpy operations on flow data.

**Materialization was considered as a 2D-only stopgap and dropped (2026-07-31)** because
it would not have bought much: of the tiling consumers in the tree,
`examples/ex_IBAMR_ibmesh.py` (`tile_domain(3,3)`), `examples/ex_sticky_seafan_3d.py`
(`tile_domain(x=13)`), and the tiling discussion in `examples/basic_ex_3d.py` are all
**3D** and would have hit `NotImplementedError` anyway. `tests/IBAMR_test_data/` is now
present, so the real dimension-agnostic implementation can be verified end-to-end against
actual 3D data — a throwaway 2D materializer would be work that had to be deleted again.
It also contradicts the naming rule above: better to have the feature clearly and loudly
unavailable than to have it quietly behave one way in 2D and another way eventually in 3D.

### 9.3 Restoration checklist — everything the gating touched

Gating tiling off left notices, stubs, and replaced tests across source, tests, examples,
docs, and prose. **This is the complete list**; work down it when tiling returns, and
delete this subsection once it is empty.

⚠️ **Read this first: the old bodies are preserved in place, commented out.**
`FluidData.tile_flow` and `Environment.tile_domain` both had their entire bodies replaced
by a `raise`, but the previous implementations sit directly beneath each raise under a
`PREVIOUS IMPLEMENTATION, KEPT FOR RESTORATION` banner. **Reuse them rather than
rewriting from memory** — parts of both are still correct:

- `tile_domain` — only its `self.flow.tile_flow(x,y)` call is superseded by
  position-wrapping. The ibmesh tiling (offsetting copies by `L[0]*ii`, `L[1]*jj`), the
  `self.L` scaling, and the `_reset_flow_deriv()` call are still correct verbatim.
- `tile_flow` — the `f.tiling` propagation is dead (`FlowArray` and the spline `tiling`
  attributes are gone), but the `fshape` arithmetic and the `flow_points` extension are
  the shape/geometry half of the naming rule and carry over as-is. The reported
  coordinate arrays still have to grow with the tiling even though the velocity data
  will not.

**Source — remove the gates:**
- [ ] `planktos/fluid.py` — `FluidData.tile_flow`: replace the `raise` and its
      `.. note::` with the position-wrapping implementation, reusing the commented-out
      `fshape`/`flow_points` handling. Delete the commented block once its useful parts
      are back in force.
- [ ] `planktos/_environment.py` — `Environment.tile_domain`: same, restoring the
      commented-out ibmesh/`L`/`_reset_flow_deriv` logic. Note the docstring currently
      explains *why* it raises before mutating anything — that rationale stops applying
      once the call succeeds. Delete the commented block afterward.
- [ ] `planktos/_swarm.py` — the `Swarm` class docstring example lost its
      `>>> envir.tile_domain(3,3)` line (it would have raised). Restore if you want the
      example to show tiling again.

**Tests — replace the interim contract with a behavioral one:**
- [ ] `tests/test_flow_generation.py` — delete `test_tile_domain_raises_not_implemented`,
      `test_tile_domain_leaves_environment_untouched`, and
      `test_tile_flow_raises_on_fluiddata_directly`, plus the section comment above them.
      Restore a real check: the pre-gating `test_tile_flow_replicates_and_resizes` is in
      git history and is a reasonable starting point, **but it only covered 2D and only
      the stored values** — the new implementation needs interpolation-through-tiling and
      3D coverage, which is exactly what was missing before (§9.2: no test ever called
      `interpolate_flow` after `tile_domain`, which is why the breakage went unnoticed).
- [ ] Add the `tiling` × `periodic_dim` interaction tests §9.1 calls for.

**Examples — delete the notices:**
- [ ] `examples/ex_IBAMR_ibmesh.py` — "!!! THIS EXAMPLE DOES NOT CURRENTLY RUN TO
      COMPLETION !!!" block in the module docstring.
- [ ] `examples/ex_sticky_seafan_3d.py` — same block in the module docstring.
- [ ] `examples/basic_ex_3d.py` — the `# NOTE: tile_domain currently raises ...` comment.
- [ ] `examples/old_examples/old_ex_pltcyl.py` calls `tile_domain(3,3)` and was
      **deliberately left unflagged** — it is archived record-keeping code whose own
      header says to skip it. Nothing to undo; listed so its absence above does not read
      as an oversight.

**Docs — delete the warnings:**
- [ ] `docs/examples/IBAMR_ibmesh.rst` — the `.. warning::` after the `tile_domain`
      snippet.
- [ ] `docs/examples/basic_3d.rst` — the `.. warning::` after the tiling paragraph.
- [ ] When `docs/api/FluidData.rst` finally exists (an open `TODO.md` item), make sure
      `tile_flow`'s docstring no longer carries the unavailability note.

**Prose — retract the "temporarily unavailable" framing:**
- [ ] `CLAUDE.md` — the "**Domain tiling currently raises `NotImplementedError`**"
      paragraph in "Fluid data architecture", and the `test_flow_generation.py` bullet in
      the Tests section.
- [ ] `TODO.md` — Phase 1 item **(E)**, and the deferred `Environment.extend` item.
- [ ] This note — §9 in its entirety, and the tiling row in §0.2.

### 9.4 Release coordination

`changelog.txt` under 1.1.0 carries
`- Domain tiling temporarily raises NotImplementedError; it returns with 2D and 3D support.`
That line is accurate **only if 1.1.0 ships before tiling returns.** If tiling lands
first, delete it and describe the new implementation instead. Do not let both statements
ship together.

---

## Appendix A — what the deleted fluid-interface note established

`docs/notes/flow_field_interface.md` was the design record for reshaping
`Environment.flow` on `dyload`. Its §1–§7 are **complete**; this appendix is all that
needs to survive as reference. The full analysis, the defect reproductions, and the
consumer audit are in git.

**What it concluded and what shipped:**

- `Environment.flow` is a `fluid.FluidData` (or subclass), or `None`. Container
  (`len(flow)`, `flow[dim]`, iteration, `np.array(flow)`) and callable (`flow(time)`,
  triggering `update_spline` on out-of-window times) contracts unchanged.
- **`FlowArray` was deleted.** Velocity components — stored static arrays and spline
  returns alike — are **plain `np.ndarray`**. Every numpy/scipy/matplotlib operation
  works natively; there are no interop caveats and **no `np.asarray()` defensive
  wrapping**. If you find such a wrapper, it is a leftover. Roughly 380 lines net came
  out: the ~165-line class, the tiling index-mapping branches inside
  `LinearSpline.__getitem__` / `fCubicSpline.__getitem__` (~73 lines each), and the
  `tiling`/`dshape` attributes throughout.
- **Its sole justification was tiling**, which it did not actually deliver (§9.2). Tiling
  and `Environment.extend` were therefore deferred wholesale rather than stopgapped;
  §9 is the real implementation.
- **`fmin`/`fmax` are tuples**, not single-use generators.
- Three live bugs surfaced and were fixed: `fmin`/`fmax` (above), `max_spd` /
  `get_mean_fluid_speed` corrupted by the subclass, and `get_raw_loaded_data` broken on
  the entire dynamic path (`LinearSpline.regenerate_data` did not exist).
- **`tests/test_flow_interface.py` is the safety net** for all of it, and pins the
  `Environment.flow` consumer contract: `interpolate_flow` /
  `interpolate_temporal_flow` values (2D + 3D, on/off-node, extrapolation), the container
  and spline-indexing surface, `fmin`/`fmax` tuples, `_calc_basic_stats` (including that
  it pulls **no** fluid field), `get_mean_velocity`, `get_mean_fluid_speed`,
  `calculate_mag_gradient`, `get_raw_loaded_data`, `fshape`, the plotting strided-slice
  path, the `LinearSpline`/`INUM` temporal path, and 3D vorticity. All closed-form. Keep
  it green.

  The `LinearSpline` coverage matters disproportionately: `test_temporal_interp.py` unit-
  tests `fCubicSpline` thoroughly but never touched `LinearSpline`, even though it is what
  *every* dynamically-loaded run interpolates with. `Environment(flow=...)` hardcodes
  `INUM=None`, so those tests construct `FluidData` directly.

**The durable lesson**, which is why §9.2 exists: the interface's blast radius is
`_environment.py` + a few `_swarm.py` accessors + plotting. `_geom.py`, `_ibc.py`,
`_dataio.py` and `__init__.py` do not touch the flow field, and `motion.py` never touches
`Environment.flow` directly — it goes through the four `Swarm` accessors
(`get_fluid_drift`, `get_dudt`, `get_DuDt`, `get_fluid_mag_gradient`), all of which funnel
into `Environment.interpolate_flow`.

## Appendix B — re-running the consumer audit

Line-number catalogues went stale the moment work landed and are not worth maintaining.
If an exhaustive per-site list is needed again, re-run the audit rather than trusting a
recorded one:

```
grep -rn "\.flow\|flow_points\|flow_times\|np\.asarray\|interpolat" \
    planktos/ tests/ examples/
```

Highest-signal sites, by role:

- **Spatial interpolation (hot path):** `_environment.py` `interpolate_flow`,
  `interpolate_temporal_flow`.
- **Swarm accessors → interpolation:** `_swarm.py` `get_fluid_drift`, `get_dudt`,
  `get_DuDt`, `get_fluid_mag_gradient`.
- **Gradient / analysis:** `calculate_mag_gradient`, `get_mean_fluid_speed`,
  `get_vorticity`, `get_dudt`, `calculate_DuDt`, `_calc_basic_stats`.
- **Save / round-trip:** `save_fluid`, `save_2D_vorticity`.
- **Plotting:** `plot_flow`, `plot_2D_vort`, `Swarm.plot` / `plot_all` (quiver strided
  slices `flow[k][::M,::N].T`, `fmax` unpack, `fshape[1:]` frame sizing).
