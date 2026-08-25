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
every environmental time step.

Author: Christopher Strickland

Email: cstric12@utk.edu

What an archive is good for
---------------------------

**Crash survival.** The format is valid with nothing having run at the end. 
Metadata is written when recording starts, every file appears atomically, and 
the reader reconstructs the timeline by scanning what is on disk. The most a 
kill can cost is the captures buffered since the last chunk.

**Analysis of a finished run, without re-running it.** This includes plotting.

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

Validation
~~~~~~~~~~

See ``RunArchive.check_against(envir)``.

To be plain about what it cannot catch: two runs on the same mesh at the same
cadence fingerprint identically, and nothing cheap detects a dataset regenerated
in place at the same path.

What is on disk
---------------

A directory of ``.npy`` files plus a JSON sidecar, all of it readable without
Planktos::

    run_archive/
      meta.json                 written once at record(): format version, the
                                grid summary, dtype, chunk size, and the
                                provenance of the fluid and the mesh
      grid.npz                  flow_points, L, flow_times, periodic_dim
      agents/
        swarm00.json            name, N, D, first_capture
        swarm00_pos_0000.npy    (rows, N, D)
        swarm00_vel_0000.npy    (rows, N, D)
        swarm00_mask_0000.npy   (rows, N) bool
        times_0000.npy          (rows,) shared across swarms
      fluid/                    per-dump fluid quantities

``meta.json`` holds only what is known when recording starts and never changes.
Anything that accumulates during the run lives in the files that accumulate with
it, which is what lets the metadata be a single write that is never touched
again -- including when a swarm joins an hour into the run.

Agent captures are **chunked** rather than written one file per capture: a
10 000-step run would otherwise make 10 000 files, which is punishing on 
filesystems. Chunk indices are parsed as integers and sorted numerically, 
never lexically.
