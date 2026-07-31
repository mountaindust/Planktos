Environment class
=================

Every instance of the Environment class is functionally a rectangular spatial 
domain in either two or three dimensions. The lower left corner is located at 
the Euclidean origin. Boundary conditions are specified with respect to the 
agents on each side of the domain. A fluid velocity field can be specified on a 
regular mesh of grid points which always includes the domain boundaries. The 
fluid velocity may vary in time, but the spatial mesh on which it is specified
must remain constant. Analytical fluid velocity fields are also available.

The fluid velocity field itself lives in a separate object: ``Environment.flow``
is a :doc:`FluidData <FluidData>`. In particular, the ``INUM`` argument taken by
the fluid reader methods below selects whether the dataset is held in memory or
streamed from storage, and with it whether interpolation in time is cubic or
linear -- see :ref:`inum-tradeoff`.

Created on Tues Jan 24 2017

Author: Christopher Strickland

Email: cstric12@utk.edu

.. autoclass:: planktos.Environment
    :members:
