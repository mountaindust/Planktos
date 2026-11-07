Run archives
============

A **run archive** streams agent state to disk as the run proceeds. 
Planktos otherwise holds an entire run in memory and can only write it out at
the end -- if the simulation crashes, the data is lost. It also allows for 
plotting a dynamically loaded run without re-reading the fluid dataset, and for 
running simulations that would not fit in RAM (including the option of a 
capture_interval that keeps state every *k*-th step rather than every step).

Start recording with ``Environment.record`` and read the data back with ``planktos.load_run``::

    import planktos

    envir = planktos.Environment()
    envir.read_IB2d_fluid_data('flow_data/', dt=1e-3, print_dump=100, INUM=4)
    swrm = planktos.Swarm(swarm_size=1000, envir=envir)

    with envir.record('run_archive/'):
        for _ in range(10000):
            swrm.move(0.001)

The simulation loop is the same either way, and data capture fires automatically 
on every time step. The ``fluid`` argument of record says what a later *plot*
will need from the velocity field, so that replaying the run never streams the
dataset a second time -- see `The fluid half`_.

Author: Christopher Strickland

Email: cstric12@utk.edu

Recording
---------

The full signature of the simulation is documented when recording starts, so 
environmental parameters must remain fixed from that point on.

``Environment.record`` returns a handle which is also a context manager.
**Recording starts as soon as the call returns**, not when a ``with`` block is
entered -- the ``with`` form only adds the guaranteed stop and cleanup at the 
end. A bare ``envir.record(path)`` therefore records from that moment on, which 
is what you want in a notebook where a ``with`` block cannot span cells::

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
to a timestamped sibling instead: ``run_archive/`` becomes
``run_archive_20260825143052/``, with a warning naming the new path. Read it 
back from the handle's ``.path`` rather than from the path you
asked for, or a later ``load_run('run_archive/')`` will quietly open the
*previous* run. **An exception is when the run continues the existing archive,** 
the one case where the directory is written into as it stands -- see
`Continuing an archive`_.

What is refused
~~~~~~~~~~~~~~~

An archive fixes the domain, the fluid grid and the timeline when recording
starts, and every capture is written against them. So while a recording is
active, ``Environment.reset`` and every fluid setter raise: rewinding the clock
or loading new fluid partway through would leave the archive describing a run
that no longer exists. So does a second ``record()`` on the same Environment,
since there is one recorder per Environment by construction.

Specific to dynamically loaded fluid: **start recording before the
fluid window has moved.** Dumps the sliding window has already passed are gone
and are never re-reported, so a recording started mid-run would have holes in
its fluid series. ``record()`` refuses at that point rather than at render time.

The fluid half
--------------

An archive stores what a later plot needs to know about the fluid so that
plotting a run doesn't re-stream the dataset a second time. Three things are 
stored:

**Per-dump fluid statistics.** The spatial mean of each velocity component,
the per-component extrema, and (in 2D, when vorticity was asked for) the largest
absolute vorticity in each dump. A handful of floats per dump, written in 2D and
3D alike, since the statistics box on a plot shows the component means in every
case. Read them with ``RunArchive.dump_stats()``. ``NaN`` marks a dump the run
never loaded -- reduce over them with ``np.nanmax``, not ``np.max``.

**Vorticity, conditional on the source.** What happens is decided by whether the 
fluid is being dynamically loaded and by whether the source already has the 
vorticity:

================================  ================================  ==============
Regime                            During the run                       At render
================================  ================================  ==============
whole field in memory             nothing is written                recomputed from
                                                                    the velocity
windowed, source has it           nothing is written                read from the
                                                                    source
windowed, source doesn't have it  one file per dump, as it lands    read back
================================  ================================  ==============

With the whole field resident, recomputing the curl is *cheaper* than reading it
back. With a window sliding, the velocity a render would need is no longer
in memory, so recomputing drags a full reload behind it. Either way the answer 
is **exactly** the curl of the velocity actually in use, not a second 
approximation stacked on the first.

Written velocity goes into the source's fluid directory, in the source's own 
naming -- ``Omega.0042.vtk`` beside ``u.0042.vtk`` -- so a later
run, ParaView, etc. can read it with no knowledge of Planktos. If that directory 
cannot be written, ``fluid/`` in the archive directory is used instead, and 
``meta.json`` records which happened. Written files are stamped with a vtk header 
line reading ``Planktos <version> vorticity <digest>``, where the digest is of 
the field the file holds. A later recording whose field differs -- another 
``vel_conv``, ``vec_name`` or ``periodic_dim``, or changed source data -- 
rewrites the series and warns that it is doing so. A recording that relied on 
the old files then refuses to draw its vorticity backdrop rather than draw the 
new field. A file without the stamp, such as the solver's own, is never 
overwritten. A source carrying the field for only *part* of the dump range is 
not used; Planktos warns and writes a complete series of its own rather than 
potentially mixing two results.

**Quiver, opt-in.** ``fluid='quiver'``, or ``fluid=('vort', 'quiver')``.
The downsampled velocity is stored on a grid fixed by ``quiver_shape`` at record 
time. Read one dump's arrows with ``RunArchive.quiver(t_idx)``.

.. note::
   All of this beyond the statistics is **2D only**. ``fluid=`` is forced to
   ``None`` in 3D, where Planktos draws no fluid backdrop at all (and on a
   flow-free environment).

Plotting a recorded run
-----------------------

An Environment remembers the archive it recorded to::

    with envir.record('run_archive/', fluid='vort'):
        for _ in range(steps):
            swrm.move(dt)

    swrm.plot_all(movie_filename='out.mkv', fluid='vort')

Three consequences worth knowing:

**A backdrop the archive does not hold is refused.** 
This applies only where it has to: with the whole field in memory the curl is
derived from it and the arrows subsampled from it, which is free and exact.

**Vorticity colour limits cover the whole run.** Without a limit fixed up front
they grow with each frame drawn. The archive's largest absolute vorticity fixes 
them before the first frame. An explicit ``clip`` is still used as given.

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

It also renders when the run crashes, since the movie could be diagnostic, but 
not on a ``KeyboardInterrupt``. Both still flush. A failure inside the render is 
warned about rather than allowed to mask the run's own exception, and the 
archive is complete either way. This is only for a recording covering one swarm, 
since ``plot_all`` is a ``Swarm`` method.

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

    pos = run.positions(0)          # a CaptureSeries for Swarm 0, read on demand
    pos[run.capture_at(3.4)]        # where the agents were nearest t=3.4
    pos[10:20]                      # a span of captures
    pos.asarray()                   # the whole thing

Indexing a ``CaptureSeries`` reads only the chunks that index touches and hands
back an ordinary masked array. A masked row means that agent is not in the
domain -- either because it left, or because the swarm had not joined the run
yet.

A swarm addedmid-run starts at a nonzero capture, and a recording started after 
``t=0`` has its capture 0 partway into the run. Matching on ``run.times`` is 
right in every one of those cases. ``capture_at`` returns the index
of the nearest capture and nothing blends between them (no interpolation). 
Swarms should be addressed by index, not by name -- the default ``Swarm`` name 
is ``'organism'`` for every swarm, so two swarms in one Environment collide by 
name by default.

Restarting a run
----------------

``restore()`` turns an archive back into a live ``Environment`` and its
``Swarm``\ s, at the state the run left off::

    envir, swarms = planktos.load_run('run_archive/').restore()
    swrm, = swarms

    for _ in range(more_steps):     # carries on as if nothing had happened
        swrm.move(dt)

The fluid files and all subclasses have to still be there and ``pos_history`` 
and ``vel_history`` are filled from the archive, ``Environment.time_history`` 
with them. Pass ``history=False`` to skip that on a long run; the physics is 
identical either way, but ``plot_all`` then has a single frame to draw and the 
agent statistics count against the restored state rather than the original swarm.

A fluid handed to ``Environment(flow=[...])`` as arrays has no loader call to
replay, so the Environment comes back withoutfluid for you to set yourself.

Restarting partway through
~~~~~~~~~~~~~~~~~~~~~~~~~~

``restore(capture=j)`` rebuilds at capture *j* rather than at the end of the
run. ``capture_at(t)`` turns a time into an index, and a negative index counts
back from the end::

    run = planktos.load_run('run_archive/')
    envir, (swrm,) = run.restore(capture=run.capture_at(17.0))

Only what the recording stored per capture comes from capture *j*. Everything 
else is the run's **final** state -- ``shared_props`` and the random stream 
among them -- so the continuation is stochastically different from the original 
run past *j*, not a bit-identical rerun of it. A notice printed at the call says 
which is which::

    Restoring at capture 340 of 1200 (t=17). Recorded per capture: positions.
    Taken from the end of the run instead: velocities, shared_props -- if any
    of them varied during the run, this resumes with their final values.

A swarm that had not joined the run by capture *j* did not exist then, and is
left out of the returned list with a warning rather than handed back with every
agent masked.

Continuing an archive
~~~~~~~~~~~~~~~~~~~~~

``record()`` on the directory a run came from **continues that archive** rather
than starting a second one beside it::

    envir, (swrm,) = planktos.load_run('run_archive/').restore()
    with envir.record('run_archive/'):        # the same directory
        for _ in range(more_steps):
            swrm.move(dt)

Nothing already written is rewritten except the tail chunk, which is the one
piece that has to grow. A run recorded, stopped, restored and appended gives an
archive **byte-identical** to the same run recorded in one go.

Validation
~~~~~~~~~~

See ``RunArchive.check_against(envir)``.

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
        swarm00_series.npz      per-capture values that do not scale with the
                                swarm: speed statistics when velocities were
                                not stored, shared__<key> when 'shared_props'
                                was, or both
        times_0000.npy          (rows,) shared across swarms
      fluid/
        dump_stats.npz          per-dump component means and extrema, and the
                                digest stamped into each vorticity file written
        quiver_00042.npy        downsampled velocity, only if asked for
        Omega.0042.vtk          vorticity, ONLY when it had to be written and
                                the source directory could not take it

``meta.json`` holds only what is known when recording starts and never changes.
Anything that accumulates during the run lives in the files that accumulate with
it. Agent captures are **chunked** rather than written one file per capture to 
reduce the number of files on disk.
