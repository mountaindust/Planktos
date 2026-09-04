Run archives
============

A **run archive** streams agent state to disk as the run proceeds --
the mirror image of what dynamic fluid loading does for the velocity field. 
Planktos otherwise holds an entire run in memory and can only write it out at
the end.

Start one with ``Environment.record`` and read it back with ``planktos.load_run``::

    import planktos

    envir = planktos.Environment()
    envir.read_IB2d_fluid_data('flow_data/', dt=1e-3, print_dump=100, INUM=4)
    swrm = planktos.Swarm(swarm_size=1000, envir=envir)

    with envir.record('run_archive/'):
        for _ in range(10000):
            swrm.move(0.001)

The loop is an ordinary Planktos loop, unchanged: captures fire automatically on
every environmental time step. The ``fluid`` argument says what a later *plot*
will need from the velocity field, so that replaying the run never streams the
dataset a second time -- see `The fluid half`_.

Author: Christopher Strickland

Email: cstric12@utk.edu

What an archive is good for
---------------------------

**Crash survival.** The format is valid with nothing having run at the end. 
Metadata is written when recording starts, every file appears atomically, and 
the reader reconstructs the timeline by scanning what is on disk. The most a 
kill can cost is the captures buffered since the last chunk.

**Analysis of a finished run, without re-running it**, in this session or a
later one, through ``planktos.load_run``.

**Plotting a dynamically loaded run for free.** What a plot needs to know about
the fluid is written as the run proceeds, so drawing it afterwards reads none of
the dataset again -- see `Plotting a recorded run`_.

**Simulations larger than RAM.** Reading one capture touches only the chunk it 
lives in, making simulations larger than RAM possible. Also, 
``capture_interval`` keeps state every *k*-th step rather than every step, both 
in the archive and in ``pos_history`` / ``vel_history`` / ``time_history``.

Recording
---------

The full signature is documented with the class that carries it:
:meth:`planktos.Environment.record`. Every parameter is fixed when recording
starts.

``Environment.record`` returns a handle which is also a context manager.
**Recording starts as soon as the call returns**, not when a ``with`` block is
entered -- the ``with`` form only adds the guaranteed stop. A bare
``envir.record(path)`` therefore records from that moment on, which is what you
want in a notebook, where a ``with`` block cannot span cells::

    rec = envir.record('run_archive/')
    for _ in range(200):
        swrm.move(0.1)
    envir.flush_recording()        # write what is buffered, keep recording
    swrm.plot_all()
    for _ in range(800):
        swrm.move(0.1)
    envir.stop_recording()

.. autoclass:: planktos.archive.RunRecorder
    :members:

Where the data goes
~~~~~~~~~~~~~~~~~~~

**Recording into a non-empty directory does not overwrite it.** The archive goes
to a timestamped sibling instead -- ``run_archive/`` becomes
``run_archive_20260825143052/`` -- with a warning naming the path actually
chosen. Read it back from the handle's ``.path`` rather than from the path you
asked for, or a later ``load_run('run_archive/')`` will quietly open the
*previous* run.

What is refused
~~~~~~~~~~~~~~~

An archive fixes the domain, the fluid grid and the timeline when recording
starts, and every capture is written against them. So while a recording is
active, ``Environment.reset`` and every fluid setter raise: rewinding the clock
or loading new fluid partway through would leave the archive describing a run
that no longer exists. So does a second ``record()`` on the same Environment,
since there is one recorder per Environment by construction.

One more, specific to dynamically loaded fluid: **start recording before the
fluid window has moved.** Dumps the sliding window has already passed are gone
and are never re-reported, so a recording started mid-run would have holes in
its fluid series. ``record()`` refuses at that point rather than at render time
-- before the run instead of after it. Either start before the loop, or load
with ``INUM=None``.

The fluid half
--------------

An archive stores what a later plot needs to know about the **fluid**, so that
replaying a run never re-streams the dataset a second time. Under dynamic
loading, redrawing a finished run used to cost a full second pass over ~100 GB;
it now costs nothing.

Three things are stored, each on its own terms.

**Per-dump statistics, always.** The spatial mean of each velocity component,
the per-component extrema, and (in 2D, when vorticity was asked for) the largest
absolute vorticity in each dump. A handful of floats per dump, written in 2D and
3D alike, since the statistics box on a plot shows the component means in every
case. Read them with ``RunArchive.dump_stats()``. ``NaN`` marks a dump the run
never loaded, which under a sliding window is an honest answer and not a gap to
be filled -- reduce over them with ``np.nanmax``, not ``np.max``.

**Vorticity is not cached -- it is sourced.** Which of three things happens is
decided by whether the fluid is being dynamically loaded and by whether the
source already ships the field:

============================  ===================================  ==============
Regime                        During the run                       At render
============================  ===================================  ==============
whole field in memory         nothing is written                   recomputed from
                                                                   the velocity
windowed, source has it       nothing is written                   read from the
                                                                   source
windowed, source has not      one file per dump, as it lands       read back
============================  ===================================  ==============

With the whole field resident, recomputing the curl is *cheaper* than reading it
back, so writing would cost about a gigabyte per five hundred dumps for negative
performance. With a window sliding, the velocity a render would need is no longer
in memory, so recomputing drags a full reload behind it -- and reads twice the
bytes to do it, since velocity is a vector and 2D vorticity a scalar.

Either way the answer is **exactly** the curl of the velocity actually in use,
not a second approximation stacked on the first. Under linear splining a field
evaluates as a weighted sum of two adjacent dumps and the curl is linear, so
blending per-dump vorticity with the same two weights gives the curl of the
blend.

**Where written vorticity goes.** Into the source's *own* fluid directory, in
the source's own naming -- ``Omega.0042.vtk`` beside ``u.0042.vtk`` -- so a later
run, ParaView, or the solver's own tooling reads it with no knowledge of
Planktos, and the source becomes indistinguishable from one whose solver had
printed vorticity all along. An existing file is never overwritten. If that
directory cannot be written, which for a read-only mount or a shared dataset is
normal, the archive's own ``fluid/`` is used instead, and ``meta.json`` records
which happened.

A source carrying the field for only *part* of the dump range is not used at all.
Serving one dump's field for another's is a plausible-looking wrong answer, so
Planktos warns and writes a complete series of its own rather than mixing the
two.

**Quiver is opt-in.** ``fluid='quiver'``, or ``fluid=('vort', 'quiver')``.
Vorticity is what almost every plot draws, and a quiver is a second
full-cadence array on disk for a backdrop most runs never use. What is stored is
the downsampled velocity, on a grid fixed by ``quiver_shape`` at record time --
``plot_all`` normally derives its arrow density from the figure size and axis
extent, neither of which exists while a simulation is running. Read one dump's
arrows with ``RunArchive.quiver(t_idx)``.

.. note::
   All of this beyond the statistics is **2D only**. ``fluid=`` is forced to
   ``None`` in 3D, where Planktos draws no fluid backdrop at all, and on a
   flow-free environment, which has no field to derive anything from. Neither is
   an error and neither warns, because in neither is there anything else the
   argument could have meant.

Plotting a recorded run
-----------------------

An Environment remembers the archive it recorded to, and keeps remembering after
the recording stops. So the ordinary loop needs nothing added to it::

    with envir.record('run_archive/', fluid='vort'):
        for _ in range(steps):
            swrm.move(dt)

    swrm.plot_all(movie_filename='out.mkv', fluid='vort')

**A whole movie of a dynamically loaded run then reads no fluid data at all.**
Drawn without a recording it costs a second streaming pass over the entire
dataset, which is the cost the archive exists to remove.

Three consequences worth knowing:

**A backdrop the archive does not hold is refused.** There is deliberately no
fallback, because the only fallback is re-reading the whole dataset to draw a
picture of it. Record with ``fluid=('vort','quiver')`` to keep the choice open.
This applies only where it has to: with the whole field in memory the curl is
derived from it and the arrows subsampled from it, which is free and exact.

**Vorticity colour limits cover the whole run.** Without a limit fixed up front
they grow with each frame drawn, so where they end up depends on how much of the
run was drawn and two plots of one run do not agree. The archive's largest
absolute vorticity fixes them before the first frame. An explicit ``clip`` is
still used as given.

**Plotting without a recording still works**, and warns once with an estimate of
what it will re-read. Nothing about the old workflow breaks.

.. note::
   Plotting a run in a *later session* needs the Environment and Swarm restored
   to where that run left off. ``RunArchive.restore()`` does that -- see
   `Restarting a run`_.

Rendering when the run ends
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``record(plot_all=...)`` takes a dict of ``Swarm.plot_all`` arguments and
renders when a ``with`` block ends::

    with envir.record('run_archive/',
                      plot_all=dict(movie_filename='out.mkv', fluid='vort')):
        for _ in range(steps):
            swrm.move(dt)

It renders when the run raises, since a crash is unexpected and the movie is
diagnostic, but not on a ``KeyboardInterrupt``, which is a request to stop
*now*. Both still flush. A failure inside the render is warned about rather than
allowed to mask the run's own exception, and the archive is complete either way.
Only for a recording covering one swarm, since ``plot_all`` is a ``Swarm``
method; that is refused at ``record()`` rather than at the end of the run.

.. note::
   Use ``.mkv`` for long or unattended runs. A hard kill leaves an ``.mp4``
   unplayable, since ffmpeg writes its index last; Matroska survives truncation
   and is playable while still being written. Remuxing afterwards is lossless:
   ``ffmpeg -i out.mkv -c copy out.mp4``.

Reading
-------

.. autofunction:: planktos.load_run

.. autoclass:: planktos.archive.RunArchive
    :members:

.. autoclass:: planktos.archive.CaptureSeries
    :members:

A worked read
~~~~~~~~~~~~~

::

    run = planktos.load_run('run_archive/')

    run.times                       # capture times: the archive's time base
    run.swarms                      # [('organism', 0)]

    pos = run.positions(0)          # a CaptureSeries, read on demand
    pos[run.capture_at(3.4)]        # where the agents were nearest t=3.4
    pos[10:20]                      # a span of captures
    pos.asarray()                   # the whole thing, if it fits in memory

Indexing a ``CaptureSeries`` reads only the chunks that index touches, and hands
back an ordinary masked array. A masked row means that agent is not in the
domain -- either because it left, or because the swarm had not joined the run
yet.

Two rules worth knowing before you rely on the wrong one:

**Resolve by time, not by index into someone else's list.** A swarm added
mid-run starts at a nonzero capture, and a recording started after ``t=0`` has
its capture 0 partway into the run. Matching on ``run.times`` is right in every
one of those cases; assuming archive index *j* equals history index *j* is right
only in the common one, and fails silently when it is not.

**Agent state is snapped, never interpolated.** ``capture_at`` returns the index
of the nearest capture and nothing blends between them. Interpolating positions
across a domain wrap or an immersed-boundary slide would invent trajectories
that never happened. Temporal interpolation belongs to the fluid, where the
field is smooth -- see :doc:`FluidData`.

**Swarms are addressed by index; names are a convenience.** The default
``Swarm`` name is ``'organism'`` for every swarm, so two swarms in one
Environment collide by name by default. ``run.positions('organism')`` raises
when the name is not unique rather than picking one, and ``run.swarms`` lists
name and index together so the collision is visible.

Restarting a run
----------------

``restore()`` turns an archive back into a live ``Environment`` and its
``Swarm``\ s, at the state the run left off::

    envir, swarms = planktos.load_run('run_archive/').restore()
    swrm, = swarms

    for _ in range(more_steps):     # carries on as if nothing had happened
        swrm.move(dt)

Same positions, same properties, same random stream, same clock. Nothing about
the fluid or the mesh is deserialized: the loader calls the archive recorded are
replayed, so the data is re-read from wherever it lives -- which means **the
fluid files have to still be there**.

``pos_history`` and ``vel_history`` are filled from the archive, and
``Environment.time_history`` with them. Pass ``history=False`` to skip that on a
long run; the physics is identical either way, but ``plot_all`` then has a
single frame to draw and the agent statistics count against the restored state
rather than the original swarm.

Three things fail differently, on purpose:

* **A fluid or mesh that cannot be replayed is an error.** Its files have moved,
  or the call was made with an array argument, whose contents a provenance
  record does not store.
* **A** ``Swarm`` **subclass that cannot be imported is an error.**
  ``apply_agent_model`` *is* the behavior of the run, so restoring a plain
  ``Swarm`` in its place would silently be a different simulation. Put the class
  on the import path and try again.
* **A lost** ``plot_structs`` **is a warning.** Those are function handles and
  cannot be recorded; a run that had them is restored without them, and said so.

A fluid handed to ``Environment(flow=[...])`` as arrays has no loader call to
replay. That warns rather than failing, and the Environment comes back without
fluid for you to set yourself.

.. note::
   Recording a restored run writes a **new** archive. ``record()`` on the
   directory it came from finds that directory non-empty and redirects to a
   timestamped sibling, so a resumed run sits beside the first rather than
   continuing it.

Validation
~~~~~~~~~~

See ``RunArchive.check_against(envir)``. Reloading the fluid at a different
``INUM`` -- or resident, which is the cheap thing to do now that a render does
not stream it -- is not a difference: ``INUM`` says how much of the dataset is
held at once and nothing about what the dataset is.

To be plain about what it cannot catch: two runs on the same mesh at the same
cadence fingerprint identically, and nothing cheap detects a dataset regenerated
in place at the same path.

What is on disk
---------------

A directory of ``.npy`` files plus a JSON sidecar, all of it readable without
Planktos::

    run_archive/
      meta.json                 written once at record(): format version, the
                                grid summary, dtype, chunk size, which fluid
                                quantities were recorded and where vorticity
                                lives, and the provenance of the fluid and the
                                mesh
      grid.npz                  flow_points, L, flow_times, periodic_dim
      agents/
        swarm00.json            name, N, D, first_capture
        swarm00_pos_0000.npy    (rows, N, D)
        swarm00_vel_0000.npy    (rows, N, D)
        swarm00_mask_0000.npy   (rows, N) bool
        times_0000.npy          (rows,) shared across swarms
      fluid/
        dump_stats.npz          per-dump component means and extrema
        quiver_00042.npy        downsampled velocity, only if asked for
        Omega.0042.vtk          vorticity, ONLY when it had to be written and
                                the source directory could not take it

``meta.json`` holds only what is known when recording starts and never changes.
Anything that accumulates during the run lives in the files that accumulate with
it, which is what lets the metadata be a single write that is never touched
again -- including when a swarm joins an hour into the run.

Agent captures are **chunked** rather than written one file per capture: a
10 000-step run would otherwise make 10 000 files, which is punishing on 
filesystems. Chunk indices are parsed as integers and sorted numerically, 
never lexically.
