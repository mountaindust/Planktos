environment class
=================

Every instance of the environment class is functionally a rectangular spatial 
domain in either two or three dimensions. The lower left corner is located at 
the Euclidean origin. Boundary conditions are specified with respect to the 
agents on each side of the domain. A fluid velocity field can be specified on a 
regular mesh of grid points which always includes the domain boundaries. The 
fluid velocity may vary in time, but the spatial mesh on which it is specified 
must remain constant. Analytical fluid velocity fields are also available.

Created on Tues Jan 24 2017

Author: Christopher Strickland

Email: cstric12@utk.edu

.. autoclass:: planktos.environment
    :members:
