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

Planktos **held** an entire run in memory and could only write it out at the end.
`pos_history` and `vel_history` grew one masked `N×D` array per step forever, the three
save methods all required the whole history resident, and **nothing read any of them
back** — so a run could not be reloaded, only re-run. Meanwhile `plot_all` replayed a
finished run by pulling fluid data at every frame, which under dynamic loading
re-streamed the entire dataset a second time. Those looked like two problems. They were
one: **run state had nowhere to go except memory.** §1.1 has the detail.

The symmetry worth holding onto: **dynamic loading streams the fluid *in*; this streams
the agents *out*.** Same architecture, opposite direction, same reason — the thing is
too big to hold at once.

### 0.2 The components

| | What | Why | Status |
|---|---|---|---|
| **A** | **Run archive** — append-only, chunked, crash-valid on-the-fly capture of agent state, with a public reader and a capture schedule that also governs history retention | persistence: crash survival, later sessions, larger-than-RAM analysis, bounded history memory, run speed, and eventually restart | **[done]** — §6.1 A0–A5, 2026-08-21 to 2026-08-25 |
| **B** | **Fluid-side streaming** — per-dump means, vorticity by regime, whole-run extrema | dyload: never re-stream the dataset to draw a picture of it | **[done]** — §6.1 B1–B3, 2026-08-25 |
| **C** | **Rendering** — frame selection by time, archive-backed `plot_all`, global colour/arrow scales | consumes A and B | **[done]** — §6.1 C1–C2, 2026-08-27 |
| **R** | **Full-state reboot** — a checkpoint beside the archive, a reader that turns a directory back into an `Environment` and its `Swarm`s, and appending to the archive a run resumed from | the third problem this architecture solves, and the one A was built for: a run that outlives the process that made it | **[done]** — §6.1 R0–R6, 2026-08-31 to 2026-09-08 |
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

**Components A, B, C and R are built** (2026-08-19 to 2026-09-08): agent state streams
to disk as a run proceeds and reads back through `planktos.load_run`; the archive carries
what a plot needs from the fluid, so replaying a windowed run costs zero loader calls; and
a recorded run can be rebuilt into a live `Environment` and its `Swarm`s at any capture
and then go on being recorded into the same archive. §5's prerequisite bug fixes went
first, since each settled state the archive was going to store.

What is left in this note:

1. **§9 — tiling**, the one unbuilt component. It has its own restoration checklist
   (§9.3), because gating it off left notices across source, tests, examples and docs.
2. **§7 — the prose pass**, which rides on §9 because §9 decides what that prose
   describes. One new example is owed there in its own right: agents arriving mid-run
   (§8.1).

⚠️ **Branch-level priority this note cannot see:** check `TODO.md` before assuming
"next section in this note" means "next work to do."

---

## 1. Two things about this note

**It was written as a plan and is now mostly a record.** Sections describing built work
state what was decided and why, not what to do; the source and its docstrings are the
authority on behaviour. Where a section still specifies unbuilt work it says so.

**It began as a plot cache and was reframed** (2026-08-18) into a run archive with a
plotting consumer. The original framing is still visible in commit messages and a few
source comments, so: "the cache" in an older message means the archive. Almost all of the
design survived the reframe — crash validity, chunking, self-description, mmap, the
capture/render split, the linearity property (§3.2) — because none of it was ever
specific to plotting.

### 1.1 Two standing facts about the problem

**Plotting re-streams the fluid, and that is what components B and C exist to stop.**
`plot_all` replays a finished run frame by frame, and each frame pulled fluid data at
`envir.time_history[n]` — the statistics text unconditionally, vorticity or quiver in 2D.
Under dynamic loading every one of those goes through `FluidData.__call__`, which reloads
when the requested time leaves the resident window, so replaying a run slid the window
back to the start and forward again: a full second pass over a dataset that may be
~100 GB. In 3D that second pass bought **nothing but a text label**, since `fluid='vort'`
and `'quiver'` are 2D-only.

> ⚠️ **A correction worth not re-discovering.** The original outline claimed plotting was
> also a *memory* bottleneck, because "the whole animation is built before anything is
> written", and proposed streaming frames to disk as an independent win. **That is
> false.** `Animation.save()` already wraps `writer.saving(...)` and calls `grab_frame()`
> per frame, so `plot_all` has always streamed into the ffmpeg pipe with O(one frame)
> encoding memory; `FuncAnimation` holds a single figure and redraws it, and
> `cache_frame_data` caches only the frame *indices*. The 2D bottleneck is **time**, not
> memory. There is no "stream the video" work item and the video machinery needs no work.

**The three `save_*` methods still have no loader**, and `save_pos_to_csv` is the
worst-shaped of them for a long run: one `np.savetxt` of a dense `N+1 × (1+D)·T` matrix,
so the whole history must be resident *and* a full text copy is materialized — over a
gigabyte of ascii in a single call at 1000 agents × 10 000 steps in 3D. The archive is
now the way a run comes back; re-expressing those three as exports from it is deferred
(§8), not done.

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
                   store=('positions',), capture_interval=1,
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
| `store` | positions | which per-agent arrays to keep; velocities and the opt-in series are named here | §2.4 | **A3a**, default changed at **R4b** |
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

**The layout itself is in `planktos/archive.py`'s module docstring**, which is the one copy kept current; `docs/api/RunArchive.rst` renders it for users. What is here is why it is shaped that way.

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
possible** by decoupling collision handling from `pos_history`, but which is **still
not built**: it would leave live `plot_all` and `_calc_basic_stats` with nothing to read
without an archive, and that interaction wants its own pass. It is the whole of what is
left of the `TODO.md` maybe-feature (§8).

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

### 2.11 Component R — full-state reboot — **[done]**

**The goal, stated as a user would:** run a simulation streaming to disk, delete the
`Environment` and the `Swarm`, and rebuild both from the directory at the state the run
left off — same positions, same properties, same random stream — and carry on as if
nothing had happened. Built as §6.1's R0–R6; this section is the specification it was
built from, kept because it is the audit of *what a `Swarm` is made of*, which is what
anyone changing the checkpoint has to get right again.

The distinction that keeps it simple:

- an **archive** is append-only history — every capture, no state that history does not
  contain;
- a **checkpoint** is one latest state plus everything history cannot give you.

Same format, different file, written on the chunk boundary. §2.11.5 carries the
measurements and the container decisions — read it before changing what a checkpoint
holds or what it is written in.

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
| `ib_collision_idx` | int N | **required** | **none** | `after_move` overrides read it, within the step that set it. The state goes in the checkpoint; **no history is built** — see below |
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

**No history of `ib_collision_idx` is built** *(decided 2026-09-08, cutting what had been
specified as R5c)*. It is the one deliberate exception to §2.11.1's rule, and the reason is
that nothing needs it:

- **Planktos never reads one.** Every consumer — the `after_move` overrides in
  `ex_ib2d_sticky.py`, `ex_ib2d_mvbnd_sticky.py`, `ex_sticky_seafan_3d.py` — reads it
  inside the step that produced it, and `Swarm._apply_ib_result` rewrites every processed
  agent's entry each step, so nothing carries over.
- **A resume does not need it.** The first `move()` overwrites it before any `after_move`
  can look, which is why it sits under "everything else — end state" in §6.1 R5's table.
  The checkpoint keeps the latest value regardless: one int per agent, in a file that is
  O(N) by design.
- **The user-facing use is already reachable.** Collision statistics over a run come from
  `self.props['hit'] = self.ib_collision_idx` in `after_move`, recorded by the R4c props
  series. Both routes need the decision at `record()` time, so a dedicated series would
  buy only a cheaper channel for someone wanting collisions and nothing else from props.

If that last case ever turns up, the shape to build is `(capture, agent, element)` int32
triples in an ordinary chunk file — `12·E` against a dense `4·N·T`, break-even at a 33%
collision rate against measured rates of 6–9%.

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

#### 2.11.5 What the containers and the measurements settled

*(R0, 2026-08-31, run before anything wrote a checkpoint, so that the build started from
measurements rather than from this plan's assumptions.)*

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

✅ **The retargeted checklist tests were deleted when Step R closed** (R6, 2026-09-08),
as they said they would be. The behavioral tests cover the same ground — full coverage
of the checklist is what makes them pass — and are merely harder to read as a list. A
one-item-per-line checklist is worth having *while building* and is dead weight
afterwards. Recorded here rather than in the test file, which was the thing deleted.

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
  merge of two NaN-marked arrays. §6.1 Step R6 is what that is for;
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

Three defects found while reframing this plan, all predating it, all present on `master`,
and all touching state the archive was about to persist — so they were settled before
anything wrote to disk. §5.1 and §5.2 landed 2026-08-19, §5.3 on 2026-08-21. Each came
with tests, a changelog line under 1.0.3, and a cherry-pick entry in `TODO.md`, which is
where the port notes live.

**§5.1 — `_calc_basic_stats` finite-differenced positions instead of reading recorded
velocities.** These are not the same quantity. `Swarm.move` sets `velocities` from
**pre-boundary-condition** positions and then `apply_boundary_conditions` mutates
`positions`, so the two part company for any agent that collided with an immersed or
domain boundary — and on a periodic dimension a wrap makes the difference a spurious
near-domain-width velocity. **Per-agent velocity at a past time is therefore not
recoverable from stored positions**, which is why the archive stores or derives it rather
than reconstructing it, and why `store=` dropping velocities has to record what they were
needed for (§2.4). The same edit dropped a `t_indx == 0` branch that substituted a zero
vector; the recorded value there is the agents' initial fluid drift, and is the truth.

**§5.2 — `Environment.reset()` cleared `pos_history` only**, leaving `vel_history` and
`props_history` behind and permanently misaligned with it — which reaches the plotted
heading markers and, once §5.1 landed, the statistics too.

**§5.3 — a bare `Swarm.move()` froze the other swarms** in a multi-swarm environment
into an inconsistent history: it warned and advanced only itself, so the others' histories
fell behind `envir.time_history` and every index-aligned consumer was wrong thereafter.
It raises now, and `Environment.move_swarms` is the supported spelling.

---

## 6. Build order

### 6.1 Steps

⚠️ **The step letters and the component letters do not agree** — see §0.2. Step D here is
the prose pass (§7); component D is tiling (§9).

**What was built, in what order.** Everything below the last line is done; the source and
its docstrings are the authority on behaviour, and what is kept here is the reasoning a
reader of that source would not otherwise have.

| Step | What | Landed |
|---|---|---|
| **0** | §5's prerequisite bug fixes | 2026-08-19 to 08-21 |
| **A0** | decouple collision handling from `pos_history` | 2026-08-21 |
| **A1** | provenance at load time (§2.6) | 2026-08-21 |
| **A2–A3b** | the writer, `Environment.record`, `capture_interval` | 2026-08-21 to 08-24 |
| **A4–A5** | `planktos.load_run` and `RunArchive` (§2.7) | 2026-08-25 |
| **B1–B3** | fluid-side streaming (§3) — built B2, B1, B3 | 2026-08-25 |
| **C1–C2** | archive-backed rendering and global scales (§4) | 2026-08-27 |
| **R0–R6** | the full-state reboot (§2.11) | 2026-08-31 to 09-08 |
| **D** | examples and docs prose pass (§7) | **not built** — rides on §9 |

**A0 came first and alone**, because it touches the riskiest code in the project.
`apply_boundary_conditions` took each agent's movement start point from
`pos_history[-1]`; it reads `Swarm._prev_positions` now, set at all three sites that move
agents (`Swarm.move` and both inlined loops in `calculate_FTLE`). Without that,
`capture_interval` silently corrupts collisions — §2.2 carries the failure analysis.

> ⚠️ **The decoupling test was checked against the old coupling**, which is what made it
> worth having. With `prev_pos = self.pos_history[-1]` restored,
> `test_collisions_do_not_read_the_position_history` fails exactly as §2.2 predicts: the
> poisoned history makes the collision check miss entirely and all four agents pass
> **through** the wall. A test that passed both before and after would have proved
> nothing. The companion guard,
> `test_prev_positions_is_the_history_entry_while_capture_is_every_step`, fails the
> moment a capture schedule gates one history append and not the other.

**A3b is where `capture_interval` landed**, and its test — bit-identical trajectories at
a coarse schedule — is the one that protects the physics. It is pinned over **two** mesh
geometries: with a single one, a bug that only bites on a particular collision shape
passes. What is recorded must not change what happens.

**Two traps A1 left behind.** `_provenance.jsonable`'s type checks are ordered
**numpy-first**, because `np.float64` *is* a subclass of `float` — a plain
`isinstance(value, float)` branch ahead of the numpy ones passes numpy scalars straight
through while claiming to have converted them. And `functools.wraps` plus
`inspect.signature` following `__wrapped__` is what keeps Sphinx rendering a decorated
loader's argument list; losing that would silently empty the API reference for every
loader.

**B was built B2 → B1 → B3.** B2 (scalar rectilinear VTK I/O, §3.6) is independent of
everything and testable on its own with a round trip, and B3 needs it.

**C changed where per-frame data comes from, not how it is drawn.** `plot_all=` joined
`Environment.record`'s signature here rather than at A3: its whole value is that
`__exit__` renders *from the archive*, so landing it earlier would have given a version
that renders from live history and re-streams the fluid — precisely what it exists to
prevent.

---

**Step R — the full-state reboot (§2.11).** Specified in §2.11; built R0–R6.

*Why it went ahead of tiling.* The two are independent, so this was a scheduling call.
**The format had to grow, and archives were already being written**: a checkpoint file, a
Swarm-class name, `char_L`/`U`/`nu` in the environment provenance — every archive written
before those existed is one that cannot be rebooted. That is a one-way door for real runs,
and it was the only item on the queue with one.

- **R0 — pre-flight** *(2026-08-31)*. Verified §2.11.2's state list against a live
  `Swarm` and settled the container questions before either was baked into a format.
  §2.11.5 has what it found, including that `DataFrame.to_json` silently truncates.
- **R1 — the environment gaps** *(2026-09-02)*. `char_L`, `U`, `nu` and `ibmesh_color`
  into `provenance['environment']`. Additive, so old archives still read.
- **R2 — the checkpoint** *(2026-09-02)*. One latest state per swarm, rewritten whole and
  atomically on the chunk boundary, so it is never staler than the captures a hard kill
  would cost anyway.
- **R3 — the reader** *(2026-09-03)*. `RunArchive.restore()`, which delivers the whole
  user-visible claim. ⚠️ **It sets `Environment._archive_path`** — without that a restored
  run plotted its fluid by re-reading the dataset, silently, since with no archive linked
  there is nothing to warn about.
- **R4 — the derived quantities and the opt-in histories** *(2026-09-04)*. `store=`
  became `('positions',)` with the per-capture statistics and stored heading angle that
  let velocities drop — **48% off the archive and 59% off the recording overhead** — and
  `store=(…, 'props')` keeps the whole DataFrame per capture.

  ⚠️ **The capture buffer must copy the props frame, not reference it.** Captures are
  buffered until a chunk fills, so holding the live DataFrame lets an in-place edit
  rewrite every capture still waiting to flush — all of them showing the final value.
  Found by a probe reporting `stage=2` at capture 0, and pinned. **The same bug recurred
  at R5b** for `shared_props`, where `np.asarray` hands back the caller's own buffer.
- **R5 — resuming from an arbitrary capture** *(2026-09-08)*, below.
- **R6 — appending to the archive a run came from** *(2026-09-08)*, below.

*A per-capture `rndState` series was **dropped**: its only gain is a bit-exact resume from
an arbitrary capture, and a stochastically-different one is enough (2026-09-04). So was
an `ib_collision_idx` series (R5c, 2026-09-08) — nothing in Planktos reads such a history,
a resume takes the value from the end state because the first `move()` overwrites it
anyway, and the statistic is already reachable by copying it into props in `after_move`.*

**Step R5 — resuming from an arbitrary capture. ✅ [done 2026-09-08].**

`RunArchive.restore(capture=j)` rebuilds at any capture, not only the last. A
*stochastically different* continuation is the target — not a bit-exact one, which is why
no per-capture `rndState` series exists. Everything the recording did not keep per capture
comes from the checkpoint, i.e. the run's **final** state:

| | from capture *j*? | |
|---|---|---|
| `positions` | always | |
| `props`, `velocities`, `shared_props` | when `store` named them | |
| `velocities`, specifically | | irrelevant to the default Brownian model, which never reads an agent's own velocity; **`motion.inertial_particles` does** |
| everything else | end state | `accelerations` is recomputed on the first step; `ib_condition` and the class do not vary |

**R5a — `restore(capture=j)`.** Winds the state back and **prints what was and was not
recorded**, in the shape of the `store=` notice, so the caller can judge whether anything
time-varying is among the substitutions:

    Restoring at capture 340 of 1200 (t=17). Recorded per capture: positions.
    Taken from the end of the run instead: velocities, shared_props -- if any
    of them varied during the run, this resumes with their final values.

`capture=None` is unchanged and stays silent. The substitution sentence is dropped at the
last capture, where the checkpoint *is* capture *j*.

⚠️ **A swarm that had not joined the run by capture *j* is left out of the returned
list**, with a warning. Its series is front-padded with fully masked rows, so restoring it
would hand back a swarm reading "every agent has left the domain" rather than one that was
not there. The list is the roster the run held at *j*.

**R5b — the `shared_props` series.** The item that makes R5a honest, and **O(T), not
O(N·T)**: ~1.2 MB over 10 000 captures against 248 MB for the props series at N=1000.

⚠️ **It added no unconditional write.** Cheap on disk is not cheap in file-tree complexity
or in write time, and both are paid by every run whether or not anyone wants the series.
Two designs were rejected: its own always-on sidecar (free on disk, but another file every
run pays for), and sentinel rows in the props csv (`agent = -1`, one row per capture) —
which reuses the file, but every **agent** row then carries an empty cell for each shared
column: 6 columns × 1000 agents × 10 000 captures is **57 MB of commas** to store
something that is O(T). Reusing the file is not the same as reusing the row.

Instead `agents/swarmNN_stats.npz` was generalized into the per-swarm per-capture sidecar
and renamed **`swarmNN_series.npz`**: already accumulated in memory, already rewritten
whole on the chunk cadence, already exactly this shape. It gains `shared__<key>` beside
`avg_vel`/`avg_spd`/`std_spd`. No new file, no new write.

⚠️ **It is gated on a `'shared_props'` token of its own, not on `'props'`.** That file
exists only when velocities are **absent** (`_ArchiveWriter.derive`), so gating it on
`'props'` would have left `store=('positions', 'velocities', 'props')` with nowhere to
write. Its own token is the more honest knob regardless: a ramping `mu` is O(T) and has
nothing to do with whether the O(N·T) per-agent DataFrame was wanted. The file's existence
condition is `derive or 'shared_props' in store`.

*Three wrinkles*, all from `shared_props` being a mutable dict: a key that **appears**
mid-run, one that **vanishes**, and one whose value **changes shape**, which cannot be
stacked at all.

**How the padding is carried.** A padded slot is a hole, and a fill value cannot say so on
its own — a NaN or an empty string is a value a run could legitimately have held. So
`present__<key>` rides beside `shared__<key>`, and **only for a key that was not there at
every capture**, which is why a `shared_props` of fixed membership writes nothing extra.
`RunArchive.shared_props()` turns the pair into a masked array per key — the same statement
a masked position row makes — and `restore(capture=j)` **replaces** `shared_props`
wholesale from it rather than merging, so a key the run had deleted before *j* cannot come
back from the checkpoint's copy of the final dict. The fill under the mask is NaN for a
float column, for anyone reading the npz raw.

A **shape change** is refused by name. A value that cannot be stored without pickle is
warned about and dropped, once per key, which is what the checkpoint and `_split_props`
already do with the same value; refusing outright was tried and reverted, since it cannot
be stored either way and ending the run buys nothing the warning does not.

**Three defects turned up in the R5 review** *(2026-09-08)*, all in the sidecar and all
fixed with the step:

1. 🔴 **The sidecar was written only by `flush()` and `close()`.** Every chunk file, the
   heading series and the checkpoint land on the chunk boundary; this one did not, so a
   hard kill lost the *whole* of it — the speed statistics since R4a, and now the
   `shared_props` series — while everything around it survived. That contradicts §2.5's
   "valid with no finalizer having run" for the one file the claim was never tested
   against. `_write_series` is called from `_write_chunk` now, which puts it on the same
   boundary and, better, makes it cover *exactly* the captures the chunks do: the
   accumulators grow per capture, and a chunk closes before the capture that rolled it
   over is buffered.
2. **A short sidecar was not refused**, where a short chunk and a partial `ang` series both
   are. `_validate_chunks` checks its length now; absence stays ordinary, since the file
   is opt-in twice over.
3. **`restore(capture=j)` fell back silently** to the checkpoint's `shared_props` when
   `store` claimed a series that was not on disk, contradicting the notice it had just
   printed. It warns now.

**Two index conventions, and the swarm that joined mid-run — ✅ [fixed 2026-09-08].**
`positions` and `angles` come through `CaptureSeries`, whose contract is that a series is
`len(archive.times)` long and front-padded with masked rows for a swarm that joined
mid-run. `props()`, `agent_stats()` and `shared_props()` all start at that swarm's own
first capture instead. For the ordinary swarm, present from capture 0, the two agree.

**It cannot be fixed in the accessor**, and that is the whole of why it took a design pass.
`_frames.FrameSource` indexes by state index *n*, and *n* means two different things: for a
**live** swarm created at capture 5, `pos_history` starts empty, so `n=0` is its own first
state; for a **restored** one, `pos_history` is front-padded to the archive's index, so
`n=5` is archive capture 5. Front-padding the accessor fixes the second and breaks the
first.

**Resolved through the time base instead**, which is right in both and is the rule the
archive already states for anyone reading it — *"resolve by time, not by index into someone
else's list."* `FrameSource._archive_capture(n)` maps a state to a capture by matching
`times`, and `_series_row(n)` subtracts `RunArchive.first_capture(swarm)` (public, and the
one place the offset is named). It also settles a ragged edge no index arithmetic could: a
live mid-run swarm's **first state is one step before its first recorded capture**, because
the capture at that time was taken before it joined. Time says so and returns no row.

**Three further defects fell out of the same neighbourhood**, all fixed with it:

- 🔴 **`Swarm._calc_basic_stats` raised `ZeroDivisionError` for any restored mid-run
  swarm**, from every frame including the one `plot(t=)` draws, so drawing such a run was
  impossible. `num_orig` divided by the population of `pos_history[0]`, which for that
  swarm is the fully masked front-pad. It now takes the first history frame holding
  anybody. §8.1 records the `perc_left` decision this settles.
- 🔴 **`_frames._live_times` gave a mid-run swarm the *first* n of the environment's times
  rather than the last n**, so a live one's frames were labelled with times from before it
  existed — and the time-based resolution above depends on that being right.
- **`restore()` left `props_history` shorter than `pos_history`** for a mid-run swarm and
  for a swarm with no props at all (whose DataFrame has no rows, so its series has no
  frames). `move()` appends to both, so the misalignment was permanent. Both are now
  front-padded with one shared placeholder frame — unreadable by any plot, since every
  agent is masked out of the positions for exactly those states.

**Step R6 — appending to the archive a run was restored from. ✅ [done 2026-09-08].**

`record()` on the directory a run came from used to meet §2.1's non-empty rule and
redirect, so a resumed run sat beside its own history rather than continuing it. It
continues it now.

**The trigger is a checkable fact, not a remembered one: the archive's last capture is
exactly where the Environment now is** — `envir.time == archive.times[-1]` — with
`store`, `chunk_size`, `capture_interval` and the fluid quantities all matching
`meta.json`. That is better than "this Environment came from a restore" three ways: it is
verifiable from state; it **fails safe**, since restoring and then running before
recording leaves the clock past the last capture, so a separate archive is written rather
than a series with a hole in it (§2.8 makes a partial series a refusal, not a silent
fill); and it picks up the notebook workflow of `stop_recording()`, a look at the data,
and a second `record()`. **Nothing already in the archive is rewritten except the tail
chunk**, which is the one piece that has to grow.

*What it needed:* refill the tail chunk rather than starting a short one (`_validate_chunks`
refuses a short chunk mid-series) — the position `.npz`, the props csv and the `.npy`
files its array-valued columns spill to alike; skip the capture `RunRecorder.__init__`
otherwise always takes, which would duplicate the last capture at the same timestamp;
leave `meta.json` and `grid.npz` alone and validate the roster rather than adding to it;
bypass `_resolve_archive_path`'s redirect; and **seed the per-swarm series file and the
fluid `means`**, both of which are rewritten whole, so an append that does not read them
back first leaves an archive covering the appended stretch only — silently, since either
file is well formed that way. `_written` is seeded too, from the non-NaN rows of the
stored means, so vorticity and quiver already on disk are not written again.

*Why it went last.* **This step reconciles every file the archive writes**, so each series
added after it lands would be a second pass through the append path. Two of the items
above are what that already cost: the list was written 2026-09-03 and R4 landed
2026-09-04, so the per-swarm series file and the props chunks did not exist to be named.
R5 would have cost a third — its `shared__<key>` entries go into the very file this step
has to seed, under wrinkles that are strictly harder across an append boundary, where
"pad the earlier captures" means padding ones read back from disk. Hence the swap on
2026-09-08.

**As built.** `_appendable()` decides, before the writer exists, whether the directory is
an archive this run continues; `_ArchiveWriter._seed_from()` picks it up. Four things the
specification did not name:

- **The checks are ordered, and only one kind refuses.** The fingerprint is compared
  *first*: an archive that lines up in time but describes a different domain or fluid is
  not a continuation at all, so it warns and redirects the way any non-empty directory
  does. Only once the world matches does *how* the recording was made have to — and there
  a mismatch raises, as specified. The order matters for a real case: a fluid handed to
  `Environment(flow=[...])` as arrays cannot be replayed, so a restore gives an
  Environment with no fluid, and every configuration check would then fail for a reason
  that has nothing to do with configuration.
- **`capture_interval` had to be recorded.** The specification says to check it against
  `meta.json`, which never carried it — nothing else needed it, since it is an
  `Environment` concern rather than a writer one. It is written now. An archive from
  before that is **not appendable**: nothing in it says the timeline would stay evenly
  spaced, so it redirects, which is what it did anyway.
- **The fluid plan is checked too**, for the same reason `chunk_size` is. `meta.json` is
  not rewritten, so a second recording asking for a different `fluid=` or `quiver_shape`
  would leave the archive describing one thing and holding another. 🔴 **But only
  over what was *asked for*, not over where vorticity ended up** — which was the first
  version and was wrong. A windowed run whose source ships no vorticity writes one there
  (§3.3), and after the first stretch that series is **partial**, which
  `probe_stored_vorticity` correctly refuses to read; a freshly computed plan therefore
  moves the field from `'source'` to `'archive'` and the append was refused — the very
  append that recording was leading to. Where the field lives is settled once per archive
  and then followed: `_replan_as_recorded` puts the archive's own answer back, and the
  per-dump write never clobbers, so a series the source already shipped whole simply
  skips every one. Found by the test written to cover the write regime, which the
  byte-identical tests cannot reach.
- 🔴 **The checkpoint was not byte-reproducible**, and the headline test is what found it.
  `.npy` records the memory order in its header, so the same values in a Fortran-ordered
  array write different bytes; a `Swarm`'s arrays are C-ordered fresh but come back from a
  restore either way. `write_checkpoint` now writes `order='C'` throughout. ⚠️ **Not
  `ascontiguousarray`**, which promotes a 0-d array to shape `(1,)` and so turns a scalar
  `shared_props` entry into a one-element array on the way back — caught by the
  `shared_props` series refusing the shape change, which is the wrinkle R5b built.

*The headline holds across the whole file set*, parametrized over the split landing
mid-chunk and on a chunk boundary, `chunk_size=1`, all four series stored, a coarse
`capture_interval`, a quiver backdrop, no fluid recorded, and a windowed (`INUM=4`)
fluid: **every file byte-identical to the same run recorded in one go.** Two consecutive
appends are too.

⚠️ **The byte-identical tests hold the fluid source fixed**, starting from the fixture
that already ships an `Omega` series, and both runs of a comparison share one writable
copy of it — the loader call is recorded in the provenance, path and all, so two copies
would differ in `meta.json` for a reason that is not about appending. That also keeps the
suite from writing vorticity into the committed fixture directory, which it did until
this was noticed: a windowed run hands `_FluidWriter` the source it was given.
`test_an_append_does_not_rewrite_vorticity_already_on_disk` covers the write regime the
fixed source excludes.

*Two refusals the review added.* Dropping a swarm the archive holds is refused **at
`record()`** — the writer would otherwise complain about mismatched swarm sets at the
first step after it, which names the symptom rather than the cause. And a per-swarm series
the archive should hold but does not cannot be carried forward, so that warns rather than
quietly writing a file covering the appended stretch alone. One thing deliberately *not*
guarded: two same-sized swarms swapped in `envir.swarms` between restoring and recording
would write into each other's series, which is the same class of thing the fingerprint
cannot catch.

*Two tests it replaced rather than added to.*
`test_a_restored_run_records_to_a_new_directory` asserted the old behaviour and is now
`test_a_restored_run_appends_to_the_archive_it_came_from`.
`test_a_non_empty_directory_redirects_and_the_handle_says_where` recorded, stopped and
recorded again into the same directory — which is the notebook workflow this step exists
to pick up, so it now moves the clock first, and a second test covers a directory holding
something that is not an archive at all.

**Step D — examples and docs prose pass (§7).**

*Two earlier versions of this list are worth not repeating.* One had "stream the video"
as an independent step, on the false premise that `plot_all` held frames in memory (see
§1.1's correction). The other had "extract a shared frame renderer" as a prerequisite,
which §2.9's capture/render split removes: with exactly one rendering path there is
nothing to share.

### 6.2 Tests

**`CLAUDE.md` carries the map of the suite**; what belongs here is only the handful of
assertions this design stands on, so that a change which quietly breaks one is
recognizable as breaking the design rather than a test.

- **Recording costs no extra fluid reads.** A run recorded against a windowed
  `FluidData` makes *identically* many loader calls as the same run unrecorded — with a
  guard that the window actually slid, or it would pass against a dataset that never
  streamed. This is the property the whole design exists for.
  (`test_recording.py`, `test_fluid_recording.py`.)
- **Replaying a recorded run costs zero.** The same replay unrecorded costs a full
  second pass, asserted beside it so the zero means something.
  (`test_archive_rendering.py`.)
- **A capture schedule does not change the physics.** A run at `capture_interval=k`
  produces bit-identical trajectories to the same run captured every step; what is
  recorded must not change what happens. Over two mesh geometries — see §6.1 A3b.
- **A blended per-dump vorticity equals the live curl** to round-off, for the sourced
  and the written case, and the two agree with each other (§3.2). Under `INUM=None` the
  assertion is instead that **no vorticity file was written anywhere**, which is that
  regime's whole content and would otherwise fail silently by costing disk nobody asked
  for.
- **A run recorded, stopped, restored and appended is byte-identical** to the same run
  recorded in one go (§6.1 R6). If that holds, every consumer of the archive is
  automatically correct.
- **Crash validity is demonstrated, not argued**: `test_run_archive.py` `SIGKILL`s a
  subprocess mid-recording and reads the bytes back with raw `np.load`/`json.load`,
  since a round trip through our own reader can be self-consistently wrong.

⚠️ **`tests/test_data_streaming/` is opt-in** (`--runstreaming`) and is the adversarial
suite written from this note — four claims, end to end. Its own `README.md` is the
standing record of the verdict on each and of every defect it found. Run it, with
`--runslow`, after any change to the archive, the fluid streaming or the plotting paths.

### 6.3 Entry points for a cold start

`CLAUDE.md`'s package-layout table is the first stop; these are the few names that are
not obvious from it. Line numbers drift — search for the names.

- `Swarm._prev_positions` — the movement start point, which `apply_boundary_conditions`
  reads instead of `pos_history[-1]`. Set in `__init__`, in `move`, and in both inlined
  loops in `calculate_FTLE`. Decoupling those was what let a capture schedule exist at
  all (§6.1 A0).
- `Swarm.move`'s `keep_state` — asked once at the top of a step and used for all three
  history appends, for the `time_history` append, **and in the `except BaseException`
  block**, which must gate too or it closes the histories off inconsistently: the exact
  thing it exists to prevent.
- `Environment._records_this_step` — the one predicate gating both the history appends
  and the archive capture, which is what lets capture *j* be exactly
  `full_pos_history[j]` with no index translation anywhere.
- `Environment._archive_path` — set by `record()` and kept after recording stops. This
  is how a plot finds the archive without being told.
- `FluidData._dumps_arrived` — the one method called at all four load sites, which
  caches the per-dump means and fans out to observers. `_FluidWriter` hangs off it.
- `archive._appendable` — the whole of the decision to continue an archive rather than
  start one (§6.1 R6).

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

⚠️ **Also owed: the interpolation-cost numbers in Appendix C belong in `docs/api`.**
They answer "what does `INUM` cost me", which is a user's question, not an internal
one, and they currently sit in a dev note. Write them as a **self-contained** table —
the three datasets side by side, what was compared against what, and the standing
caveat that the absolute errors are a property of a cadence against a flow while the
convergence orders are what transfer — so that someone reading the table needs no
background from these notes to act on it.

**Examples.** ⚠️ **One new example is owed: agents arriving mid-run** — the fixed-*N*
pool with masked-until-released agents, specified in §8.1. It is the answer to a question
that has actually been asked (a predator's capture rate against a continuous influx),
the mechanism is not discoverable from the docstrings, and the one line that makes it
work — rebuild the masked array rather than assign into `.mask`, because the mask is
hardened — is exactly the kind of thing an example exists to show. Write it against a
2D immersed boundary so the capture count is the point, and record it, since the archive
handles the pool with no format change. Build it with §8.1's `release`/`retire` if those
land first; otherwise the example carries the recipe and §8.1 cites it.

The call sites are done (§4.1 "As built") — each example names its
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
- ~~**Checkpoint / restart** (§2.11)~~ — **built**, §6.1 R0–R6 (2026-09-08).
- **History-free running** (`store_pos_history=None`): keep no `pos_history` at all and
  rely on the archive. A0 removes the collision-path obstacle (§2.2) and
  `capture_interval` covers the rest of the `TODO.md` maybe-feature (§2.10), leaving
  only this residue — which needs its own pass, because live `plot_all` and
  `_calc_basic_stats` would then have nothing to read without an archive.
- **`save_*` re-expressed as archive exports** (§2.10) — deliberately not first-pass.
- **`Swarm.release` / `Swarm.retire`** — the open-system idiom, specified in §8.1 below
  and not built.

---

### 8.1 Agents arriving mid-run: fixed *N*, dynamic membership

*(Specified 2026-09-08, from the question "what if someone adds agents to a swarm
mid-run?" — asked for a predator whose capture rate is measured against a continuous
influx of prey past an immersed boundary.)*

**There is no API for growing a swarm, and there should not be one.** `Swarm.N` is a
read-only property over `positions.shape[0]`; growing it by hand means resizing
`positions`, `velocities`, `accelerations`, `props` and `ib_collision_idx` together.
Measured, that half-works in a way worse than failing: `move()` and `plot_all` carry on
(the histories are lists, so ragged frames are fine), `save_pos_to_csv` and `save_data`
fail with an opaque numpy concatenate error, and **recording refuses** — `swarm 0
positions has shape (5, 2), expected (3, 2)`, since a chunk is `(rows, N, D)` and `N` is
fixed at `add_swarm`.

**The idiom that works today is a fixed-*N* pool whose membership changes.** A masked row
already means "not in the domain" everywhere in Planktos, and `CaptureSeries` already
uses a fully masked row for "not yet in the run", so an arrival maps onto the archive
with **no format change at all** — verified end to end, recording included, agents
released one per step and read back correctly.

Two things make it worth wrapping in an API rather than leaving as a recipe:

- ⚠️ **The mask is hardened** (`Swarm.__setattr__`), so `arr.mask[i] = False` silently
  does nothing. Bringing a row back into the domain means building a fresh masked array
  and assigning it through `setattr`, which re-hardens it. That is the whole trick, and
  it is documented nowhere.
- The three arrays must be released together, or `move()`'s finite differences read a
  masked velocity against an unmasked position.

*The shape:*

    Swarm.release(idx, positions, velocities=None)
        Bring held-back agents into the domain at the given positions. Their
        velocities default to the local fluid drift, as Swarm.__init__ does.
    Swarm.retire(idx)
        Mask agents out of the domain, as leaving through a boundary does.

`props` needs nothing: a pool allocates the whole run's agents up front, so a per-agent
release time is an ordinary column, and a capture count is an ordinary counter an
`after_move` increments.

**What this does *not* fix, and is the honest limit:** the pool's size caps the run. An
influx experiment has to know its budget in advance, or recycle retired agents — which
is the cheaper answer anyway, and is how a steady-state capture rate is usually measured.
Recycling means `retire` then `release`, and the agent's `pos_history` then holds a
teleport that `plot_all` will draw as a straight line across the domain; a run that cares
should retire into a masked stretch of at least one capture, which draws nothing.

**`perc_left` reads above 100% under a releasing pool** — 250% for five present against
the two the run started with — and that is intended: the statistic is "how many are
present against how many you started with", which stays meaningful for an open system
(decided 2026-09-08). It is pinned by a test rather than left to be rediscovered.

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

## Appendix C — what dynamic loading costs, measured on the 2D sea fan

Kept here because `docs/notes/vti_loader.md`, where these were first written down, is
deletable once its load-bearing content has moved out. **These are user-facing and are
owed to `docs/api`** in a form a reader can act on without knowing what a withholding
study is — see §7 Obligations.

Measured 2026-09-15 on `tests/data/openfoam2D/` (40 dumps, 801×801, Δt = 0.1 s,
pulsatile at ~20 samples per pulse — the coarse end of the cadence range), reproducible
with `python tests/manual/quantify_seafan_interp.py [part ...]`. The counterparts are
`quantify_temporal_interp.py` (IB2d leaf data) and `vet_dynamic_loading_3d.py`, whose
numbers are in `TODO.md` Phase 1(C).

**The slider is exact, and the memory claim holds on real data.** Windowed (`INUM=4`)
and in-memory agree **exactly** — 0.0, not round-off — over forward and backward sweeps,
at every dump time against the stored values, and clamped past the end. A full sweep
reads each dump **once**. Over that sweep the windowed run held **41 MB**, against
411 MB for the series held linearly and **1604 MB** splined cubically.

**Interpolation error**, built on every 2nd dump and tested at the 19 withheld ones, so
these are for **Δt = 0.2 s**:

| | rms err | % of U_rms | max err | ratio |
|---|---|---|---|---|
| velocity, linear | 4.25e-4 | **5.49%** | 3.17e-3 | — |
| velocity, cubic | 1.16e-4 | **1.50%** | 2.92e-3 | 3.7× |
| ∂u/∂t, linear | 6.41e-4 | 3.74% | 8.53e-3 | — |
| ∂u/∂t, cubic | 3.39e-4 | 1.98% | 1.04e-2 | 1.9× |

∂u/∂t is where cubic's margin narrows to 1.9× and where its *worst* error is the larger
of the two — and that is the term feeding `get_dudt` → the material derivative → the
inertial models.

**Convergence order by error percentile** (Δt = 0.2, 0.3, 0.4 s; theory 2 and 4):

| percentile | linear | cubic |
|---|---|---|
| median | 1.91 | 4.86 |
| 90th | 1.83 | 3.35 |
| 99th | 1.55 | 2.21 |
| max | 1.51 | 1.80 |

The two-regime structure the other datasets show: both schemes reach their order where
the flow is smooth in time and both stall where it is not, so a single rms ratio
describes neither. Cubic's error spans 535× from median to max at Δt = 0.2 s. The
whole-field rms scales at 1.68 (linear) and 2.27 (cubic), putting the native 0.1 s
cadence near **1.7% and 0.3% of U_rms**.

**What an ensemble sees** (2025 tracers, pure advection, 120 Euler steps to t = 3.9):
mean and standard deviation of position agree to **≤ 0.09%**, mean net displacement to
0.51%, its spread to 1.3%, the 90th percentile of displacement to 2.9%. Individual
trajectories separate by 2.0% of the path travelled, which is the flow's own Lyapunov
growth under any perturbation rather than a property of the scheme.

**Reading these.** Every absolute number is this flow at this cadence; the orders are
what transfer. The plate's webs are exactly zero velocity and exact under both schemes,
but at 0.41% of the grid they neither carry nor rescue the averages.
