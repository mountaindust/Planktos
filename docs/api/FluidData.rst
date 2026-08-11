FluidData class
===============

``Environment.flow`` is a FluidData object. It holds the fluid velocity field,
the spatial grid the field is specified on, the times at which it is specified,
and the interpolation of the field in time.

Most simulations never touch it directly: agents get their fluid velocity
through ``Swarm.get_fluid_drift`` and the other ``Swarm`` accessors, which go
through ``Environment.interpolate_flow``. You will want this class when you need
the field itself -- to compute vorticity, to inspect the grid, or to enable
dynamic loading of a dataset too large to hold in memory.

Created: Thurs July 9 2025

Author: Christopher Strickland

Email: cstric12@utk.edu

Getting velocity data out
-------------------------

Call the object with a simulation time to get the field at that time,
interpolated in time::

    u, v = envir.flow(0.5)          # 2D
    u, v, w = envir.flow(0.5)       # 3D

Each component is a plain :class:`numpy.ndarray` indexed ``(i,j)`` in 2D or
``(i,j,k)`` in 3D, with i indexing x, j indexing y and k indexing z. Every
numpy, scipy and matplotlib operation works on them normally.

The object can also be indexed like a list of components, ``envir.flow[0]`` for
x-velocity and so on. For a steady field that returns the raw array; for a
time-varying field it returns that component's interpolant in time. Indexing is
**not** available while data is being loaded dynamically, because a time index
would then point into a shifting window rather than into the dataset -- call the
object with a time instead.

To interpolate in *space*, at agent positions, use
``Environment.interpolate_flow``.

.. _inum-tradeoff:

Dynamic loading and the ``INUM`` argument
-----------------------------------------

Time-dependent 3D fluid data is routinely far too large to hold in memory at
once. Planktos can therefore stream it, keeping only a sliding window of time
points resident and loading more from storage as the simulation advances. This
is controlled by ``INUM``, accepted by ``FluidData`` and by the
``Environment`` fluid readers that construct one:

============  ===============================  =========================
``INUM``      Held in memory                   Interpolation in time
============  ===============================  =========================
``None``      the entire dataset               cubic
``True``      the entire dataset               linear
an ``int``    a window of ``INUM`` + 1 times   linear
============  ===============================  =========================

``None`` is the default, and is what you want whenever the dataset fits in
memory.

**Enabling dynamic loading means accepting linear interpolation in time.** That
is a permanent property of the design rather than a placeholder: stitching a
cubic spline across a window that gains data at both ends is numerically
unstable, and re-splining each window makes the time derivative discontinuous at
the window boundaries. Linear interpolation is unconditionally stable, extends
to a new window by carrying two raw values with no derivatives to match, and
needs less data resident.

What linear-in-time costs
-------------------------

Formally, the velocity field goes from C\ :sup:`2` to C\ :sup:`0` in time --
there is a kink in the velocity at every timestamp -- accuracy between samples
goes from O(Δt\ :sup:`4`) to O(Δt\ :sup:`2`), and ∂u/∂t becomes a piecewise
constant step function. That last one propagates: ``get_dudt`` feeds the
material derivative, which feeds the inertial-particle motion models.

Measured on a 2D IB2d dataset (149 dumps, 129×193 grid, dump interval 10\
:sup:`-3` s), building both schemes on every second dump and testing against the
dumps withheld:

- **In the temporally smooth bulk of the flow**, both schemes achieve their
  theoretical convergence orders -- about 4 for cubic against about 2 for
  linear. Cubic is decisively better there, and its advantage grows the more
  often you dump.
- **Where the flow is temporally rough**, neither scheme converges at its
  nominal rate and the two are close to each other. There the sampling of the
  data, not the interpolation scheme, is what limits accuracy.
- **Ensemble statistics were essentially unaffected.** Advecting 1936 tracers
  under each scheme, individual trajectories decorrelate -- as they will under
  any perturbation, since the flow separates neighbouring particles
  exponentially -- but mean and spread of position and of net displacement, and
  the 10th, 50th and 90th percentiles of displacement, all agreed to within
  0.6%, most to within 0.3%.

The practical reading: for **dispersal and other ensemble statistics**, dynamic
loading is unlikely to change your answer. For anything that depends on the
smoothness of the velocity field or on ∂u/∂t -- inertial particles in
particular -- prefer ``INUM=None`` when the data fits, and be aware of the
tradeoff when it does not.

.. note::
   Absolute error figures are a property of a given dump interval relative to a
   given flow's own timescales, and do not transfer between datasets. The
   convergence orders do. If your dumps are more widely spaced relative to your
   flow's timescales, expect a larger error from both schemes and a smaller gap
   between them. These numbers come from one 2D dataset; the 3D case has not yet
   been characterized. The measurement is reproducible with
   ``tests/manual/quantify_temporal_interp.py``.

FluidData
---------

.. autoclass:: planktos.fluid.FluidData
    :members:

Per-source subclasses
---------------------

These handle reading a particular data format. In normal use they are
constructed for you by the corresponding ``Environment`` reader method rather
than directly.

.. autoclass:: planktos.fluid.IB2dData
    :members:
    :exclude-members: load_dumpfiles

.. autoclass:: planktos.fluid.VTK3dData
    :members:
    :exclude-members: load_dumpfiles

.. autoclass:: planktos.fluid.OpenFOAMData
    :members:
    :exclude-members: load_dumpfiles

.. autoclass:: planktos.fluid.ComsolVTUData
    :members:
