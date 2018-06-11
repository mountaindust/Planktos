# Planktos Agent Based Model Framework

This project focuses on building a framework for ABMs of plankton and tiny
insects, or other small entities whose effect on the surrounding flow can be
considered negligable. Work is ongoing.

### Dependencies
- Python 3.5+
- numpy/scipy
- matplotlib
- vtk
- pytest (if running tests)

You will also need a Python 2.7 environment with numpy and VisIt installed to convert IBAMR data into vtk data.

### Tests
All tests can be run by typing `py.test` into a terminal in the base directory.

## Quickstart

There are three working examples in the examples folder, including a 2D simulation, a 3D simulation, and a simulation utilizing vtk data obtained from IBAMR which is located in the tests/IBAMR_test_data folder. More will be added. When experimenting with different agent behavior than what is prescribed in swarm.py (e.g., different movement rules), it is strongly suggested that you subclass swarm (found in framework.py) in an appropriate subfolder. That way, you can keep track of everything you have tried and its outcome. See the code in the plankton folder for an example.

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
    - `struct_plots` additional items (structures) can be plotted along with the simulation by storing function handles in this list. The plotting routine will call each of them in order, passing the main axes handle as the first argument
    - `struct_plots_args` list of tuples supplying additional arguments to be passed to the struct_plots functions
    - `tiling` if the domain has been tiled, the amount of tiling is recorded here (x,y)
    - `orig_L` length of each domain dimension before tiling
    - `fluid_domain_LLC` if fluid was imported from data, the spatial coordinates of the lower left corner of the original data. This is used internally to aid subsequent translations
    - `a` optional parameter for storing porous region height. If specified, the plotting routine will add some random grass with that height.
    - `re` optional parameter for storing Reynolds number
    - `rho` optional parameter for storing dynamic fluid velocity
    - `char_L` optional parameter for storing the characteristic length
    - `mu` optional parameter for dynamic viscosity
    - `g` acceleration due to gravity (9.80665 m/s**2)
- Methods
    - `set_brinkman_flow` Given several (possibly time-dependent) fluid variables, calculate Brinkman flow on a regular grid with a given resolution and set that as the environment's fluid  velocity. Capable of handling both 2D and 3D domains.
    - `set_two_layer_channel_flow` Apply wide-channel flow with vegetation layer according to the two-layer model described in Defina and Bixio (2005) "Vegetated Open Channel Flow".
    - `set_canopy_flow` Apply flow within and above a uniform homogenous canopy according to the model described in Finnigan and Belcher (2004), "Flow over a hill covered with a plant canopy".
    - `read_IB2d_vtk_data` Read in 2D fluid velocity data from IB2d and set the environment's flow variable.
    - `read_IBAMR3d_vtk_data` Read in 3D fluid velocity data from vtk files obtained from IBAMR. See read_IBAMR3d_py27.py for converting IBAMR data to vtk format using VisIt.
    - `tile_flow` Tile the current fluid flow in the x and/or y directions. It is assumed that the flow is roughly periodic in the direction(s) specified - no checking will be done, and no errors thrown if not.
    - `add_swarm` Add or initialize a swarm into the environment
    - `move_swarms` Call the move method of each swarm in the environment
    - `set_boundary_conditions` Check that each boundary condition is implemented before setting the bndry property.
    - `reset` Resets environment to time=0. Swarm history will be lost, and all swarms will maintain their last position. This is typically called automatically if the fluid flow has been altered by another method. If rm_swarms=True, remove all swarms.
    
Class: swarm

- Properties
    - `positions` list of current spatial positions, one for each agent
    - `pos_history` list of previous "positions" lists
    - `envir` environment object that this swarm belongs to
- Methods
    - `move` move each agent in the swarm. Do not override: see update_positions.
    - `update_positions` defines the agent's movement behavior. OVERRIDE THIS WHEN SUBCLASSING!
    - `get_fluid_drift` get the fluid velocity at each agent's position via interpolation
    - `get_projectile_motion` Return acceleration using equations of projectile motion. Includes drag, inertia, and background flow velocity. Does not include gravity.
    - `apply_boundary_condition` method used to enforce the boundary conditions during a move
    - `plot` plot the swarm's current position or a previous position at the time provided
    - `plot_all` plot all of the swarm's positions up to the current time
    
