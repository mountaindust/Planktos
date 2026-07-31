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

Using Miniforge as your Python distribution (instead of Anaconda Python) is 
highly recommended since this library heavily depends upon packages only available 
through the conda-forge package repository, and conda-forge has largely become 
incompatible with packages provided in the default Anaconda channel. In either 
case, make sure you have the most updated version of conda before installing.

If Planktos crashes on import or when plotting, the problem almost always stems 
from the following places:
1. VTK. It sometimes will crash with a seg fault right as you import it, even in 
a clean Anaconda/Miniforge installation.
2. matplotlib's backend, especially Qt or Pyside, which are needed on MacOS for 
proper video rendering.

This is especially true on MacOS - Apple's operating system has always had
terrible issues with plotting libraries in Python. If you hit trouble there, make
sure you are using only PyQt5 and do not have any PyQt6 or Pyside6 libraries
installed. Do not use VTK older than 9.2, which is incompatible with newer
versions of numpy. Newer VTK releases are fine; development is currently done on
9.4.2.

The dependencies are as follows, with the minimum supported version of each. If
you are using Miniforge, conda-forge is the default channel and you can install
vtk with just the command `conda install vtk`. Otherwise, it is recommended to
install as much as possible using the conda-forge channel via the command
`conda install conda-forge::<pkg name>`.

- Python 3.8+
- numpy >= 1.19
- scipy >= 1.10.1 (earlier versions have a broken interpn)
- matplotlib >= 3.0
- pandas
- vtk >= 9.2 ::

    conda install conda-forge::vtk

- pyvista >= 0.44 (from conda-forge. Note: pyvista may install vtk as a
    dependency, but the version could be years old and broken in modern versions
    of numpy. This may be an instance of conda-forge and default Anaconda channel
    incompatibility).
- numpy-stl >= 2.16.3 (if loading stl data). Again, get it from conda-forge.
- netCDF4 >= 1.5.7 (if loading netCDF data)
- pytest (if running tests)

These minimums are the ones declared in setup.cfg, so they are what `pip install`
will enforce. Development is currently done on Python 3.12.

If you get _image DLL errors from pillow when trying to load matplotlib.pyplot, 
try using pip to reinstall pillow using `pip install -U pillow`.

It is also highly recommended to install the jupyter package, which includes 
ipython and the necessary libraries to use jupyter lab.

Installing Planktos
~~~~~~~~~~~~~~~~~~~

Once these are installed, Planktos can be installed from source using `pip` on 
Python >= 3.8 from the Planktos directory. Navigate to the Planktos directory in 
a terminal and use the command: ::

    pip install .

Non-optional dependencies (other than FFmpeg) should automatically be installed.

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

If you use this software in your research, please cite it via the following paper: 

Strickland, W.C., Battista, N.A., Hamlet, C.L., Miller, L.A. (2022), 
Planktos: An agent-based modeling framework for small organism movement and 
dispersal in a fluid environment with immersed structures. 
*Bulletin of Mathematical Biology*, 84(72). 

A suggested BibTeX entry is included in the file 
:download:`Planktos.bib <../Planktos.bib>`.

There are several working examples in the examples folder. Start with
basic_ex_2d.py and basic_ex_3d.py, which are annotated minimal simulations.
ex_ind_var.py demonstrates individual variation between agents, and
ex_agent_behavior.py, ex_ode_gen.py, ex_vicsek_model_2d.py, and
ex_vicsek_model_3d.py demonstrate subclassing of the apply_agent_model method
for user-defined agent behavior. ex_poisson_search.py implements an intermittent
search strategy via Poisson process state switching.

Several examples show how to import vertex data (from IB2d and IBAMR),
automatically create immersed boundaries out of this data, and then simulate
agent movement with these meshes as solid boundaries which the agents respect:
ex_ib2d_ibmesh.py and ex_ib2d_sticky.py in 2D, ex_IBAMR_ibmesh.py using VTK data
obtained from IBAMR (pulled from the tests/IBAMR_test_data folder), and
ex_sticky_seafan_3d.py in 3D. In particular, ex_ib2d_mvbnd_sticky.py is the
showcase for **2D moving immersed boundaries**; it requires externally
downloaded data, so see the file header for the link. Finally,
ex_produce_ftle_2d.py demonstrates FTLE computation.

More examples will be added as functionality is added. To run any of these
examples, change your working directory to the examples directory and then run
the desired script.

An important note about immersed boundary meshes: it is assumed that segments
of the boundary do not cross except at vertices. This is to keep computational
speed up and numerical complexity down. So, especially if you are auto-creating
boundaries from vertex data, be sure and check that boundary segments are not
intersecting each other away from specified vertices! A quick way to do this is
to call Environment.plot_envir() after the mesh import is done to zoom in and 
visually check that the boundary formed correctly and doesn't cross itself in 
unexpected ways. There is also a method of the Environment class called
add_vertices_to_static_2D_ibmesh which will add vertices at all 2D mesh crossing points,
however it's use is discouraged because it results in complex vertices that 
attach more than two mesh segments and leftover segments that do not contribute 
to the dynamics at all. Do not expect meshes resulting from this method to have 
undergone rigorous testing, and running the method will add significant 
computational overhead due to the need to search for collisions with each 
additional line segment. Finally, avoid mesh structures that intersect with a 
periodic boundary (w.r.t. agents); behavior related to this is not implemented.

Research that utilizes this framework can be seen in:  

- Ozalp, Miller, Dombrowski, Braye, Dix, Pongracz, Howell, Klotsa, Pasour, 
  Strickland (2020). Experiments and agent based models of zooplankton movement 
  within complex flow environments, *Biomimetics*, 5(1), 2.

Overview
--------

Currently, Planktos has built-in capabilities to load either time-independent or 
time-dependent 2D or 3D fluid velocity data specified on a regular mesh. ASCII 
vtk format is supported, as well as one single-time ASCII vtu files from COMSOL 
and NetCDF. A few analytical 1D flow fields are also available and can be 
generated in either 2D or 3D environments; these include Brinkman flow, two layer 
channel flow, and canopy flow. Flow fields can also be tiled. Mesh data 
must be time-invariant in 3D but can be time-varying in 2D. They are loaded via 
IB2d/IBAMR-style vertex data (2D) or via stl file in 3D. 
More (open source) formats may be considered if requested. Mesh data should never 
intersect any of the domain boundaries. This will not be checked, but is essential
for correct performance.

For agents, there is support for individual variation though a pandas Dataframe 
property of the Swarm class (Swarm.props). Individual agents have access to the 
local flow field through interpolation of the spatial-temporal fluid velocity grid. 
Specifically, Planktos implements a cubic spline in time with linear interpolation 
in space. In addition to more custom behavior, an Ito SDE solver 
(Euler-Maruyama method) is included for movement specified as an SDE of the type 

.. math::
    dX_t = \mu dt + \sigma dW_t 

and inertial particle behavior for dynamics described by the linearized 
Maxey-Riley equation [1]_. These two may be combined, and other, user-supplied 
ODEs can also be fed into the drift term of the Ito SDE. Finally, agents will
treat immersed boundary meshes as solid barriers. The way an agent responds upon
encountering a mesh is configurable per Swarm via the ``ib_condition`` parameter,
and per move via ``move(..., ib_collisions=...)``:

* ``'sliding'`` (the default): there is no flux normal to the boundary, and any
  remaining movement for that step is projected onto the mesh. This is a
  recursive vector projection, so an agent that is slid into a further boundary
  is handled correctly.
* ``'sticky'``: the agent stops at the point of intersection for that step.
  Sticky interactions are supported on moving boundaries as well as static ones.
* ``None``: immersed boundaries are ignored entirely.

Elastic boundary conditions are not currently supported. Both concave and convex
mesh joints are handled, and pains have been taken to make the projection
algorithm as numerically stable as possible.

Collision detection is the main runtime bottleneck when immersed meshes are
present, and it can optionally be parallelized across agents. Attach any worker
pool exposing a ``.map`` method (``multiprocessing.Pool``,
``concurrent.futures.ProcessPoolExecutor``, or ``ThreadPoolExecutor``) to a Swarm
via the ``pool`` parameter. The default of ``None`` runs serially and reproduces
the unparallelized behavior exactly. Parallelization is mainly beneficial for the
expensive moving-boundary case; for cheap static-mesh collisions the per-agent
dispatch overhead can outweigh the work, so benchmark before relying on it.

Multiple agent species (Swarms) may share a single Environment. Advance them
together with ``Environment.move_swarms``, which moves every Swarm before
incrementing the environment time; calling ``Swarm.move`` directly on a single
Swarm will warn you that the others were not advanced. Each Swarm keeps its own
agent model, properties, and history, so different species can use entirely
different behaviors. Two caveats: there is no built-in mechanism for agents in
one Swarm to sense or interact with agents in another (a subclass can reach the
other Swarms through ``self.envir.swarms`` and implement this itself), and
plotting multiple Swarms together on the same axes is not yet implemented -- each
Swarm plots on its own. See `issue #49
<https://github.com/mountaindust/Planktos/issues/49>`_.

Beyond simulation, some analysis tools are available. Vorticity of the velocity
field can be computed, saved, and plotted via ``Environment.get_vorticity``,
``save_2D_vorticity``, and ``plot_2D_vort``.
Finite-time Lyapunov exponent fields are computed with
``Environment.calculate_FTLE`` and drawn with ``plot_2D_FTLE``. Forward-time FTLE
can be calculated for tracer particles, for user-supplied equations of motion
(including inertial particles), or for arbitrary agent behavior given by a Swarm
subclass; backward-time FTLE, requested with ``backward=True``, is supported for
tracer particles only. FTLE calculations respect static immersed boundaries, but
moving meshes are not supported.

Single-time and animation plotting of results is available in 2D and 3D.

.. [1] Haller, G. and Sapsis, T. (2008). Where do inertial particles go in
   fluid flows? Physica D: Nonlinear Phenomena, 237(5), 573-583.

Workflow
--------

This is outlined in more detail within the tutorial examples, but briefly, the 
following workflow is used to create simulatinos in Planktos:
1. Create an Environment object and load the fluid velocity data and any 
immersed mesh structures into it. Specify boundary conditions. Verify everything 
looks correct by plotting the environment.
2. Create a class for the agents you would like to simulate by subclassing 
planktos.Swarm. Create a model for your agents by implementing a method within 
your class called apply_agent_model. This method must expect the size of the 
time step as an argument and return the new positions of the agents as given 
by whatever model the user implements within the method. Boundary conditions will 
be automatically handled by Planktos after this method returns and therefore the 
user should NOT set the agent positions manually. apply_agent_model should also 
update agent states in any way necessary for the model. If such an update requires 
knowledge of the agents' final position after boundary interactions, the after_move 
method can be used, which is called only after everything else within the timestep 
has been done.
3. Create a Swarm object from your class and call it's move method in a loop to 
run the simulation.
4. Plot the results or export the data to examine elsewhere.