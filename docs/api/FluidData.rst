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

Tiled domains
-------------

After ``Environment.tile_domain``, the Environment's ``L`` spans the tiled
domain, but the FluidData still holds one stored tile: ``flow_points``,
``fshape`` and every field it returns -- ``flow(t)``, ``flow[i]``,
``get_vorticity``, ``get_dudt``, ``calculate_DuDt`` and the rest -- describe
that tile. ``Environment.interpolate_flow`` wraps agent positions onto it.
``tiling`` gives the tile counts, ``tiled_points`` the grid across the tiled
domain, and ``materialize_tiled`` replicates a field onto that grid, at the
memory cost of the whole tiled field.

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
memory. Use an ``int`` only when it does not: streaming from storage is slower
than holding the dataset. An ``int`` must be at least 2, and it does **not**
have to be odd.
``INUM`` + 1 time points are held, so a window of 4 intervals keeps 5. A smaller
window holds less in memory and, over a run moving forward in time, reads the
same data in more and smaller loads.

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

It has been measured on three datasets. In each, both schemes were built on
every second dump and tested at the dumps left out, so the errors are at twice
the dump interval:

=======================  ===========  ==========  ===============  =================
Dataset                  Grid         Dumps,      Velocity error,  Ensemble
                                      interval    linear / cubic   statistics agree
=======================  ===========  ==========  ===============  =================
2D IB2d flow             129×193      149,        1.3% / 0.5%      within 0.6%
                                      0.001 s
2D OpenFOAM sea fan,     801×801      40,         5.5% / 1.5%      within 0.1%
~20 dumps per pulse                   0.1 s
3D OpenFOAM, 10 dumps    68×68×180    17,         9.5% / 1.1%      within 0.35%
per pulse                             0.125 s
=======================  ===========  ==========  ===============  =================

Velocity error is the rms error as a percentage of the rms speed. The ensemble
column compares the mean and spread of the positions of advected tracers under
each scheme (1936, 2025 and 512 tracers respectively); individual trajectories
separate, as they do under any perturbation of a flow that separates
neighbouring particles, but the ensemble does not move.

**Absolute errors do not transfer between datasets; convergence orders do.** An
error figure is a property of a dump interval against a particular flow's own
timescales. The orders are what to carry to your data: where a flow is smooth in
time, linear converges at about 2 and cubic at about 4 to 5 (median point: 1.7
and 4.8 on the IB2d flow, 1.9 and 4.9 on the sea fan; the 3D export has too few
evenly spaced dumps to fit one), and where a flow is rough in time neither scheme
reaches its order and the two come close. The widely spaced datasets show the
*wider* gap between schemes, so widely spaced dumps are exactly when cubic is
worth keeping.

The memory saving is the point of the exercise: streaming the sea fan with
``INUM=4`` held 41 MB, against 1604 MB with the whole series splined cubically.

The practical reading: for **dispersal and other ensemble statistics**, dynamic
loading is unlikely to change your answer. For anything that depends on the
smoothness of the velocity field or on ∂u/∂t -- inertial particles in particular
-- prefer ``INUM=None`` when the data fits, and be aware of the tradeoff when it
does not.

The measurements are reproducible with ``tests/manual/quantify_temporal_interp.py``,
``tests/manual/quantify_seafan_interp.py`` and
``tests/manual/vet_dynamic_loading_3d.py``.

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

.. autoclass:: planktos.fluid.VTKXMLData
    :members:
    :exclude-members: load_dumpfiles

.. autoclass:: planktos.fluid.ComsolVTUData
    :members:
