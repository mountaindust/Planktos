# Planktos Agent-based Modeling Framework

This project focuses on building a framework for ABMs of plankton and tiny
insects, or other small entities whose effect on the surrounding flow can be
considered negligable. Work is ongoing.

If you use this software in your project, please cite it as:  
Strickland, C. (2018), *Planktos agent-based modeling framework*. https://github.com/mountaindust/Planktos.  
A suggested BibTeX entry is included in the file Planktos.bib.

Also, check out the online documentation at https://planktos.readthedocs.io.

## Installation & Dependencies
I'm assuming you're using Anaconda, and if so, I ***strongly*** suggest that you ditch 
the default package manager conda (which is essentially broken at this point - 
particularly if you need packages from conda-forge, and we do) for mamba. 
The commands are the same (it's a drop-in replacement for conda) but it is a C++ 
solver based on libsolv which manages dependencies for RedHat, Debian, etc. Also, 
it has multi-threaded downloads and doesn't break when trying to obtain vtk 
and/or pyvista. Install with the following command:  
`conda install -c conda-forge mamba`  
Having done that, the dependencies are as follows:

- Python 3.5+ 
- numpy/scipy
- matplotlib 3.x
- pandas
- ffmpeg from conda-forge (not from default anaconda. Use `mamba install -c conda-forge ffmpeg`.)
- vtk (if loading vtk data, get from conda-forge and use mamba. conda seems to 
break itself trying to install vtk for some reason, and takes an hour to try and 
solve the dependencies in the process.)
- pyvista (if saving vtk data, get from conda-forge and use mamba. same problem 
as for vtk.)
- numpy-stl (if loading stl data, get from conda-forge)
- pytest (if running tests)

If you need to convert data from IBAMR into vtk, you will also need a Python 2.7 environment with numpy and VisIt installed (VisIt's Python API is written in
Python 2.7).

### Tests
All tests can be run by typing `pytest` into a terminal in the base directory.

## Overview
Currently, Planktos has built-in capabilities to load either time-independent or 
time-dependent 2D or 3D fluid velocity data specified on a regular mesh. ASCII 
vtk format is supported, as well as ASCII vtu files from COMSOL (single-time 
data only). More regular grid formats, especially if part of the open-source 
VTK format, may be supported in the future; please contact the author (cstric12@utk.edu) 
if you have a format you would like to see supported. A few analytical, 1D flow 
fields are also available and can be generated in either 2D or 3D environments; 
these include Brinkman flow, two layer channel flow, and canopy flow. Flow fields 
can also be extended and tiled in simple ways as appropriate. Mesh data must be 
time-invariant and loaded via IB2d/IBAMR-style vertex data (2D) or via stl file 
in 3D. Again, more (open source) formats may be considered if requested.

For agents, there is support for multiple species (swarms) along with individual 
variation though a Pandas Dataframe property of the swarm class (swarm.props). 
Individual agents have access to the local flow field through interpolation of 
the spatial-temporal fluid velocity grid - specifically, Planktos implements a 
cubic spline in time with linear interpolation in space 
(**Future: tricubic spline in space**). In addition to more custom behavior, 
included in Planktos is an Ito SDE solver (Euler-Maruyama method) for movement 
specified as an SDE of the type dX_t = \mu dt + \sigma dW_t and an inertial 
particle behavior for dynamics described by the linearized Maxey-Riley equation 
(Haller and Sapsis, 2008). These two may be combined, and other, user-supplied 
ODEs can also be fed into the drift term of the Ito SDE. Finally, agents will 
treat immersed boundary meshes as solid barriers. Upon encountering an immersed 
mesh boundary, any remaining movement will be projected onto the mesh. Both 
concanve and convex mesh joints are supported, and pains have been taken to make 
the projection algorithm as numerically stable as possible.

Single-time and animation plotting of results is available in 2D and 3D; support 
for plotting multiple agent species together has not yet been implemented.

## Quickstart

There are several working examples in the examples folder, including a 2D simulation, 
a 2D simulation demonstrating individual variation, a 3D simulation, 
a simulation utilizing vtk data obtained from IBAMR which is located in the 
tests/IBAMR_test_data folder, and a simulation demonstrating subclassing of the get_positions method for user-defined agent behavior. There are also two examples demonstrating how to import vertex data (from IB2d and IBAMR), automatically
create immersed boundaries out of this data, and then simulate agent movement with these meshes as solid boundaries which the agents respect. More examples will be added as functionality is added. To run any of these examples, change your working directory 
to the examples directory and then run the desired script.

An important note about immersed boundary meshes: it is assumed that segments
of the boundary do not cross except at vertices. This is to keep computational
speed up and numerical complexity down. So, especially if you are auto-creating
boundaries from vertex data, be sure and check that boundary segments are not
intersecting each other away from specified vertices! A quick way to do this is
to call environment.plot_envir() after the mesh import is done to visually check 
that the boundary formed correctly and doesn't cross itself in unexpected ways. 
There is also a method of the environment class called add_vertices_to_2D_ibmesh 
which will add vertices at all 2D mesh crossing points, however it's use is
discouraged because it results in complex vertices that attach more than two
mesh segments and leftover segments that do not contribute to the dynamics at all. 
Do not expect meshes resulting from this method to have undergone rigorous testing, 
and running the method will add significant computational overhead due to the 
need to search for crossings.

When experimenting with different agent behavior than what is prescribed in the
swarm class by default (e.g. different movement rules), it is strongly suggested 
that you subclass swarm (found in framework.py) in an appropriate subfolder. That 
way, you can keep track of everything you have tried and its outcome. 

Research that utilizes this framework can be seen in:  
- Ozalp, Miller, Dombrowski, Braye, Dix, Pongracz, Howell, Klotsa, Pasour, 
Strickland (2020). Experiments and agent based models of zooplankton movement 
within complex flow environments, *Biomimetics*, 5(1), 2.

## API
Class: environment
   
- Properties
    - `L` list, length 2 (2D) or 3 (3D) with length of each domain dimension
    - `bndry` list of tuples giving the boundary conditions (strings) of each dimension
    - `flow_times` list of times at which fluid data is specified
    - `flow_points` list of spatial points (as tuples) where flow data is specified. These are assumed to be the same across time points
    - `flow` fluid flow velocity data. List of length 2 or 3, where each entry is an ndarray giving fluid velocity in the x, y, (and z) directions. If the flow is time-dependent, the first dimension of each ndarray is time, with the others being space. This implies that the velocity field must be specified on a regular spatial grid, and it is also assumed that the outermost points on the grid are on the boundary for interpolation purposes.
    - `swarms` list of swarm objects in the environment
    - `time` current time of the simulation
    - `time_history` history of time points simulated
    - `ibmesh` Nx2x2 or Nx3x3 ndarray of mesh elements, given as line segment vertices (2D) or triangle vertices (3D)
    - `max_meshpt_dist` max distance between two vertices in ibmesh. Used internally.
    - `t_interp` scipy.interpolate PPoly instance giving a CubicSpline interpolation of flow in time
    - `t_interp` scipy.interpolate PPoly instance giving the time derivative of the flow velocity field
    - `struct_plots` additional items (structures) can be plotted along with the simulation by storing function handles in this list. The plotting routine will call each of them in order, passing the main axes handle as the first argument
    - `struct_plots_args` list of tuples supplying additional arguments to be passed to the struct_plots functions
    - `tiling` if the domain has been tiled, the amount of tiling is recorded here (x,y)
    - `orig_L` length of each domain dimension before tiling
    - `fluid_domain_LLC` if fluid was imported from data, the spatial coordinates of the lower left corner of the original data. This is used internally to aid subsequent translations
    - `char_L` optional parameter for characteristic length scale
    - `h_p` optional parameter for storing porous region height. If specified, the plotting routine will add some random grass with that height.
    - `rho` optional parameter for storing dynamic fluid velocity
    - `mu` optional parameter for dynamic viscosity
    - `nu` optional parameter kinematic viscosity
    - `U` optional parameter for characteristic fluid speed
    - `Re` read-only Reynolds number calculated from above parameters
    - `g` acceleration due to gravity (9.80665 m/s**2)
- Methods
    - `set_brinkman_flow` Given several (possibly time-dependent) fluid variables, calculate Brinkman flow on a regular grid with a given resolution and set that as the environment's fluid  velocity. Capable of handling both 2D and 3D domains.
    - `set_two_layer_channel_flow` Apply wide-channel flow with vegetation layer according to the two-layer model described in Defina and Bixio (2005) "Vegetated Open Channel Flow".
    - `set_canopy_flow` Apply flow within and above a uniform homogenous canopy according to the model described in Finnigan and Belcher (2004), "Flow over a hill covered with a plant canopy".
    - `read_IB2d_vtk_data` Read in 2D fluid velocity data from IB2d and set the environment's flow variable.
    - `read_IBAMR3d_vtk_data` Read in 3D fluid velocity data from vtk files obtained from IBAMR. See read_IBAMR3d_py27.py for converting IBAMR data to vtk format using VisIt.
    - `read_IBAMR3d_vtk_dataset` Read in multiple vtk files with naming scheme
    IBAMR_db_###.vtk where ### is the dump number (automatic format when using
    read_IBAMR3d_py27.py) for time varying flow.
    - `read_comsol_vtu_data` Read in 2D or 3D fluid velocity data from vtu files (either .vtu or .txt) obtained from COMSOL. This data must be on a regular grid and include a Grid specification at the top.
    - `read_stl_mesh_data` Reads in 3D immersed boundary data from an ascii or binary stl file. Only static meshes are supported.
    - `read_IB2d_vertex_data` Read in 2D immersed boundary data from a .vertex file used in IB2d. Will assume that vertices closer than half (+ epsilon) the Eulerian mesh resolution are connected linearly. Only static meshes are supported.
    - `read_vertex_data_to_convex_hull` Read in 2D or 3D vertex data from a vtk file or a .vertex file and create a structure by computing the convex hull. Only static meshes are supported.
    - `add_vertices_to_2D_ibmesh` Try to repair 2D mesh segments which intersect away from 
    specified vertices by adding vertices at the intersections.
    - `tile_flow` Tile the current fluid flow in the x and/or y directions. It is assumed that the flow is roughly periodic in the direction(s) specified - no checking will be done, and no errors thrown if not.
    - `extend` Extend the domain by duplicating the boundary flow a number of times in a given (or multiple) directions. Good when there is fully resolved fluid \
    flow before/after or on the sides of a structure.
    - `add_swarm` Add or initialize a swarm into the environment
    - `move_swarms` Call the move method of each swarm in the environment
    - `set_boundary_conditions` Check that each boundary condition is implemented before setting the bndry property.
    - `interpolate_flow` Interpolate a fluid velocity field in space at the given positions
    - `interpolate_temporal_flow` Linearly interpolate the flow field in time
    - `dudt` Returns a temporally interpolated time-derivative of the flow field
    - `get_mean_fluid_speed` Return the mean fluid speed at the current time, interpolating if necessary
    - `reset` Resets environment to time=0. Swarm history will be lost, and all swarms will maintain their last position. This is typically called automatically if the fluid flow has been altered by another method. If rm_swarms=True, remove all swarms.
    - `save_fluid` Save the fluid velocity field as one or more vtk files (one for each time point).
    - `get_2D_vorticity` Calculate and return the vorticity of a 2D flow field, 
    potentially interpolated in time.
    - `save_2D_vorticity` Calculate and save (as VTK) the vorticity of a flow field 
    at one or more (possibly interpolated) points in time.
    - `calculate_FTLE` Calculate an FTLE (finite-time Lagrangian exponent) field 
    using tracer particles, user supplied equations of motion, or arbitrary agent 
    behavior/motion.
    - `plot_envir` Just plots the bounding box and any ib meshes as a sanity check.
    - `plot_flow` Plot quiver velocity field in 2D or 3D, including time-varying flows. Probably not ever going to be pretty, but useful for a sanity check.
    - `plot_2D_vort` Plot vorticity for 2D fluid velocity fields, including time-varying vorticity.
    - `plot_2D_FTLE` Plot a generated 2D FTLE field.
    
Class: swarm

- Properties
    - `positions` list of current spatial positions, one for each agent
    - `pos_history` list of previous "positions" lists
    - `full_pos_history` list of both previous and current spatial positions
    - `velocities` list of current velocities, one for each agent (for use in
    projectile motion)
    - `accelerations` list of current accelerations, one for each agent (for use
    in projectile motion)
    - `envir` environment object that this swarm belongs to
    - `rndState` random number generator (for reproducability)
    - `shared_props` properties shared by all members of the swarm. Includes:
        - `mu` default mean for Gaussian walk (zeros)
        - `cov` covariance matrix for Gaussian walk (identity matrix)
        - `diam` characteristic length for agents' Reynolds number (if provided)
        - `m` mass of agents (if provided)
        - `Cd` drag coefficient of agents (if provided)
        - `cross_sec` cross sectional area of agents (if provided)
        - `R` density ratio (if provided)
    - `props` Pandas DataFrame of properties that vary by individual agent. Any
    of the properties mentioned under shared_props above can be provided here
    instead.
- Methods
    - `full_pos_history` return the full position history of the swarm, past and present
    - `save_data` save position, velocity, and accel data to csv, save agent property
    information to npz and json
    - `save_pos_to_csv` save all current and past agent positions to csv
    - `save_pos_to_vtk` save the positions at each time step to a vtk file. only the positions
    inside the domain are saved.
    - `calc_re` Calculate the Reynolds number based on environment variables.
    Requires rho and mu to be set in the environment, and diam to be set in swarm
    - `grid_init` return a grid of initial positions based on a grid. does not set
    the swarm to these positions.
    - `move` move each agent in the swarm. Do not override: see get_positions.
    - `get_positions` returns new physical locations for the agents. OVERRIDE THIS WHEN SUBCLASSING!
    - `get_prop` return the property requested as either a single value (if shared) or a numpy array (if varying by individual)
    - `add_prop` add a new property and check that it isn't in both props and shared_props
    - `get_fluid_drift` get the fluid velocity at each agent's position via interpolation
    - `get_fluid_gradient` get the gradient of the magnitude of the fluid velocity
    at each agent's position via interpolation
    - `apply_boundary_condition` method used to enforce the boundary conditions during a move
    - `plot` plot the swarm's current position or a previous position at the time provided
    - `plot_all` plot all of the swarm's positions up to the current time. can also be 
    used to save a video

