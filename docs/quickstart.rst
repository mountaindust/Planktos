Quickstart
==========

Dependencies and installation
-----------------------------

Installing dependencies using Anaconda Python is highly recommended.
However, the default package manager, conda, appears unable to handle the 
Planktos dependencies. The performance of conda has been degrading with time 
as the number of available Python libraries increases, and this is particularly 
true in the case of packages listed on the conda-forge channel. As of writing 
this (July, 2021), it is pretty much broken.

Instead, if you are using Anaconda Python, please first install the package manager 
`mamba <https://mamba.readthedocs.io/en/latest/>`_ and use it in place of conda.
The commands are the same (it's a drop-in replacement for conda) but it is a C++ 
solver based on libsolv which manages dependencies for RedHat, Debian, etc. Also, 
it has multi-threaded downloads and doesn't break when trying to obtain vtk 
and/or pyvista. Install with the following command::
    
    conda install -c conda-forge mamba

Having done that, the dependencies are as follows:

- Python 3.5+ 
- numpy/scipy
- matplotlib 3.x
- pandas
- ffmpeg from conda-forge (not from default anaconda). Use ::

    mamba install -c conda-forge ffmpeg

- vtk (if loading vtk data) from conda-forge. Use mamba!! conda seems to 
  break itself trying to install vtk for some reason, and takes an hour to try 
  and solve the dependencies in the process.
- pyvista (if saving vtk data) from conda-forge. Use mamba!! Same problem 
  as for vtk.
- numpy-stl (if loading stl data). Again, get it from conda-forge.
- pytest (if running tests)

Getting started
---------------

There are several working examples in the examples folder, including a 2D 
simulation, a 2D simulation demonstrating individual variation, a 3D simulation, 
a simulation utilizing VTK data obtained from IBAMR (pulled from the 
tests/IBAMR_test_data folder), and simulations demonstrating subclassing of the 
get_positions method for user-defined agent behavior. There are also examples 
demonstrating how to import vertex data (from IB2d and IBAMR), automatically
create immersed boundaries out of this data, and then simulate agent movement 
with these meshes as solid boundaries which the agents respect. More examples 
will be added as functionality is added. To run any of these examples, change 
your working directory to the examples directory and then run the desired script.

An important note about immersed boundary meshes: it is assumed that segments
of the boundary do not cross except at vertices. This is to keep computational
speed up and numerical complexity down. So, especially if you are auto-creating
boundaries from vertex data, be sure and check that boundary segments are not
intersecting each other away from specified vertices! A quick way to do this is
to call environment.plot_envir() after the mesh import is done to zoom in and 
visually check that the boundary formed correctly and doesn't cross itself in 
unexpected ways. There is also a method of the environment class called 
add_vertices_to_2D_ibmesh which will add vertices at all 2D mesh crossing points, 
however it's use is discouraged because it results in complex vertices that 
attach more than two mesh segments and leftover segments that do not contribute 
to the dynamics at all. Do not expect meshes resulting from this method to have 
undergone rigorous testing, and running the method will add significant 
computational overhead due to the need to search for collisions with each 
additional line segment.

Research that utilizes this framework can be seen in:  

- Ozalp, Miller, Dombrowski, Braye, Dix, Pongracz, Howell, Klotsa, Pasour, 
  Strickland (2020). Experiments and agent based models of zooplankton movement 
  within complex flow environments, *Biomimetics*, 5(1), 2.

Overview
--------

Currently, Planktos has built-in capabilities to load either time-independent or 
time-dependent 2D or 3D fluid velocity data specified on a regular mesh. ASCII 
vtk format is supported, as well as ASCII vtu files from COMSOL (single-time vtu
data only). More regular grid formats, especially if part of the open-source 
VTK format, may be supported in the future; please contact the author 
(cstric12@utk.edu) if you have a format you would like to see supported. A few 
analytical, 1D flow fields are also available and can be generated in either 2D 
or 3D environments; these include Brinkman flow, two layer channel flow, and 
canopy flow. Flow fields can also be extended and tiled in simple ways as 
appropriate. Mesh data must be time-invariant and loaded via IB2d/IBAMR-style 
vertex data (2D) or via stl file in 3D. Again, more (open source) formats may be 
considered if requested.

For agents, there is support for multiple species (swarms) along with individual 
variation though a pandas Dataframe property of the swarm class (swarm.props). 
Individual agents have access to the local flow field through interpolation of 
the spatial-temporal fluid velocity grid - specifically, Planktos implements a 
cubic spline in time with linear interpolation in space 
(future: tricubic spline in space). In addition to more custom behavior, 
included in Planktos is an Ito SDE solver (Euler-Maruyama method) for movement 
specified as an SDE of the type 
.. math::

    dX_t = \mu dt + \sigma dW_t 

and an inertial particle behavior for dynamics described by the linearized 
Maxey-Riley equation [1]_. These two may be combined, and other, user-supplied 
ODEs can also be fed into the drift term of the Ito SDE. Finally, agents will 
treat immersed boundary meshes as solid barriers. Upon encountering an immersed 
mesh boundary, any remaining movement will be projected onto the mesh. Both 
concanve and convex mesh joints are supported, and pains have been taken to make 
the projection algorithm as numerically stable as possible.

Single-time and animation plotting of results is available in 2D and 3D; support 
for plotting multiple agent species together has not yet been implemented, but 
is a TODO.

.. [1] Haller, G. and Sapsis, T. (2008). Where do inertial particles go in
   fluid flows? Physica D: Nonlinear Phenomena, 237(5), 573-583.