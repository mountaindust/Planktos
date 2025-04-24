Quickstart
==========

**Note**: that the vtk library takes a long time to load the first time you try to 
import it. This means that Planktos might hang for quite some time when it tries 
to import this library for the very first time, but the problem should not 
repeat after that.

Dependencies and installation
-----------------------------

Installing FFmpeg
~~~~~~~~~~~~~~~~~

Before using Planktos, FFmpeg must be installed and accessible via the `$PATH` 
environment variable in order to save video files of simulation results.

There are a variety of ways to install FFmpeg, such as the 
`official download links <https://ffmpeg.org/download.html>`_, or using your 
package manager of choice (e.g. "sudo apt install ffmpeg" on Debian/Ubuntu, 
"brew install ffmpeg" on OS X, etc.).

Regardless of how FFmpeg is installed, you can check if your environment path is 
set correctly by running the "ffmpeg" command from the terminal, in which case 
the version information should appear, as in the following example (truncated 
for brevity): ::

    $ ffmpeg
    ffmpeg version 4.3.1 Copyright (c) 2000-2020 the FFmpeg developers
      built with gcc 10.2.1 (GCC) 20200726

**Note**: The actual version information displayed here may vary from one 
system to another; but if a message such as "ffmpeg: command not found" appears 
instead of the version information, FFmpeg is not properly installed.

Installing Package Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing dependencies using Anaconda Python is highly recommended. Even better 
is Miniforge, since this library heavily depends upon packages only available 
through the conda-forge package repository, and conda-forge has largely become 
incompatible with packages provided in the default Anaconda channel. In either 
case, make sure you have the most updated version of conda before starting or 
things might break.

If Planktos crashes on import or when plotting, the problem almost always stems 
from the following places:
1. VTK. It sometimes will crash with a seg fault right as you import it, even in 
a clean Anaconda/Miniforge installation.
2. matplotlib's backend, especially Qt or Pyside, which are needed on MacOS for 
proper video rendering.

This is especially true on MacOS - Apple's operating system has always had 
terrible issues with plotting libraries in Python. My best advice is to avoid 
Python 3.12 and use Python 3.11 instead, use VTK 9.2 instead of the newer 
versions (or older ones, which are incompatible with newer versions of numpy), 
and make sure you are using only PyQt5 and do not have any PyQt6 or Pyside6 
libraries installed.

The dependencies are as follows:

- Python 3.8+ 
- numpy/scipy
- matplotlib 3.x
- pandas
- vtk :: 

    conda install conda-forge::vtk

- pyvista (from conda-forge. Note: pyvista may install vtk as a dependency, but 
    the version could be years old and broken in modern versions of numpy. This 
    may be an issue related to conda-forge and default Anaconda channel 
    incompatiblity)
- numpy-stl (if loading stl data). Again, get it from conda-forge.
- netCDF4 (if loading netCDF data)
- pytest (if running tests)

If you get _image DLL errors from pillow when trying to load matplotlib.pyplot, 
try using pip to reinstall using `pip install -U pillow`.

Installing Planktos
~~~~~~~~~~~~~~~~~~~

Once FFmpeg is installed, Planktos can be installed from source using `pip` on 
Python >= 3.8 from the Planktos directory. Navigate to the Planktos directory in 
a terminal and use the command: ::

    pip install .

Non-optional depdencencies (other than FFmpeg) should automatically be installed.

Planktos is still in active development and updates occur often. You should 
therefore pull the source repo often and then reinstall using the same command. 
To avoid needing to reinstall each time you pull the repo, you can instead 
install Planktos in "editable" mode (requires pip version >= 21.1): ::

    pip install -e .

Planktos can then be imported like any other Python package from any directory. 
Either approach also allows you to uninstall with the same command (from the 
Planktos directory): ::

    pip uninstall .

**Once you have installed, verify that things work** by trying to run 
basic_ex_2d.py in the examples folder. If it crashes, see the Dependencies 
section above for troubleshooting.

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
additional line segment. Finally, avoid mesh structures that intersect with a 
periodic boundary (w.r.t. agents); behavior related to this is not implemented.

If you use this software in your research, please cite it via the following paper: 

Strickland, W.C., Battista, N.A., Hamlet, C.L., Miller, L.A. (2022), 
Planktos: An agent-based modeling framework for small organism movement and 
dispersal in a fluid environment with immersed structures. 
*Bulletin of Mathematical Biology*, 84(72). 

A suggested BibTeX entry is included in the file 
:download:`Planktos.bib <../Planktos.bib>`.

Research that utilizes this framework can be seen in:  

- Ozalp, Miller, Dombrowski, Braye, Dix, Pongracz, Howell, Klotsa, Pasour, 
  Strickland (2020). Experiments and agent based models of zooplankton movement 
  within complex flow environments, *Biomimetics*, 5(1), 2.

Overview
--------

Currently, Planktos has built-in capabilities to load either time-independent or 
time-dependent 2D or 3D fluid velocity data specified on a regular mesh. ASCII 
vtk format is supported, as well as ASCII vtu files from COMSOL (single-time vtu
data only) and NetCDF. More regular grid formats, especially if part of  
open-source formats, may be supported in the future; please contact the author 
(cstric12@utk.edu) if you have a format you would like to see supported. A few 
analytical, 1D flow fields are also available and can be generated in either 2D 
or 3D environments; these include Brinkman flow, two layer channel flow, and 
canopy flow. Flow fields can also be extended and tiled in simple ways as 
appropriate. Mesh data must be time-invariant and loaded via IB2d/IBAMR-style 
vertex data (2D) or via stl file in 3D. Again, more (open source) formats may be 
considered if requested. Mesh data should never intersect any of the domain 
boundaries. This will not be checked, but is essential for correct preformance.

For agents, there is support for multiple species (swarms) along with individual 
variation though a pandas Dataframe property of the swarm class (swarm.props). 
Individual agents have access to the local flow field through interpolation of 
the spatial-temporal fluid velocity grid - specifically, Planktos implements a 
cubic spline in time with linear interpolation in space. In addition to more 
custom behavior, included in Planktos is an Ito SDE solver 
(Euler-Maruyama method) for movement specified as an SDE of the type 

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