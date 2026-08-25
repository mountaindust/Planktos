'''
Swarm class of Planktos.

Created on Tues Jan 24 2017

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import sys, os, warnings
from pathlib import Path
import numpy as np
import numpy.ma as ma
from scipy import stats
import pandas as pd
if sys.platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.path import Path as mPath

from ._environment import Environment
from . import _dataio, _geom, _ibc, motion

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"


def _vorticity_norm(vort, clip=None, norm=None):
    '''Colour limits for the RdBu vorticity backdrop, symmetric about zero.

    RdBu is a diverging colormap, white at its midpoint, so limits that are not
    symmetric put zero somewhere other than white and tint the whole quiescent
    background red or blue. Rescaling to each frame's own extremes then makes
    that tint change frame to frame, which is what made vorticity movies flash.

    So limits are symmetric, and over a movie they only ever grow: pass the
    previous norm back in and a later, quieter frame cannot shrink the scale and
    re-tint everything. An explicit clip fixes them outright and is never
    rescaled.

    Parameters
    ----------
    vort : ndarray
        the vorticity field being drawn
    clip : float, optional
        symmetric pseudocolor limit; overrides everything else
    norm : matplotlib Normalize, optional
        the norm in use, to be grown rather than replaced

    Returns
    -------
    matplotlib Normalize
    '''

    if clip is not None:
        return colors.Normalize(-abs(clip), abs(clip), clip=True)
    finite = np.abs(np.asarray(vort)[np.isfinite(vort)])
    vmax = float(finite.max()) if finite.size > 0 else 0.
    if vmax == 0.:
        # A uniformly zero field should read as uniformly white. Limits of
        # (0, 0) would instead send every cell to the bottom of the colormap.
        vmax = 1.
    if norm is None:
        return colors.Normalize(-vmax, vmax)
    if norm.vmax is None or vmax > norm.vmax:
        norm.vmin, norm.vmax = -vmax, vmax
    return norm


class Swarm:
    '''
    Fundamental Planktos object describing a group of similar agents.

    The Swarm class (alongside the Environment class) provides the main agent 
    functionality of Planktos. Each Swarm object should be thought of as a group 
    of similar (though not necessarily identical) agents. Planktos implements 
    agents in this way rather than as individual objects for speed purposes; it 
    is easier to vectorize a swarm of agents than individual agent objects, and 
    also much easier to plot them all, get data on them all, etc.

    The Swarm object contains all information on the agents' positions, 
    properties, and movement algorithm. It also handles plotting of the agents 
    and saving of agent data to file for further analysis.

    Initial agent velocities will be set as the local fluid velocity if present,
    otherwise zero. Assignment to the velocities attribute can be made directly 
    for other initial conditions.

    NOTE: To customize agent behavior, subclass this class and re-implement the
    method apply_agent_model (do not change the call signature).

    Parameters
    ----------
    swarm_size : int, default=100
        Number of agents in the Swarm. ignored when using the 'grid' init method. 
    envir : Environment object, optional
        Environment for the Swarm to exist in. Defaults to a newly initialized 
        Environment with all of the defaults.
    init : {'random', 'grid', ndarray}, default='random'
        Method for initializing agent positions.

        * 'random': Uniform random distribution throughout the domain
        * 'grid': Uniform grid on interior of the domain, including capability
          to leave out closed immersed structures. In this case, swarm_size 
          is ignored since it is determined by the grid dimensions.
          Requires the additional keyword parameters grid_dim and testdir.
        * 1D array: All positions set to a single point given by the x,y,[z] 
          coordinates of this array
        * 2D array: All positions as specified. Shape of array should be NxD, 
          where N is the number of agents and D is spatial dimension. In this 
          case, swarm_size is ignored.
    ib_condition : {None, 'sliding' (default), 'sticky'}
        Boundary condition for immersed boundaries

        * None: Will turn off all interactions with immersed boundaries
        * 'sliding': (default) No flux in the direction normal to the boundary; 
          any movement across the boundary will be subject to vector projection 
          onto the boundary within a given time step
        * 'sticky': The velocity at the boundary is zero - anything that hits 
          the boundary stops for the remainder of the time step.
    seed : int, optional
        Seed for random number generator
    shared_props : dictionary, optional
        dictionary of properties shared by all agents as name-value pairs. If 
        none are provided, four default properties will be created, 'mu' and 'cov', 
        corresponding to intrinsic mean drift and a covariance matrix for 
        brownian motion respectively, and 'name' and 'color corresponding to the 
        name of the swarm and its default color for plotting. 'mu' will be set 
        to an array of zeros with length matching the spatial dimension, and 
        'cov' will be set to an identity matrix of appropriate size according to 
        the spatial dimension. This allows the default agent behavior to be 
        unbiased brownian motion.

        Examples:

        * diam: diameter of the particles
        * m: mass of the particles
        * Cd: drag coefficient of the particles
        * cross_sec: cross-sectional area of the particles
        * R: density ratio
    props : Pandas dataframe of individual agent properties, optional
        Pandas dataframe of individual agent properties that vary between agents. 
        This is the method by which individual variation among the agents should 
        be specified. The number of rows in the dataframe should match the 
        number of agents. If no dataframe is supplied, an empty one will be 
        created. A special property (column) can be specified called 'angle' 
        which, if props_history is being stored, will plot as the agent heading 
        in 2D.
    store_prop_history : bool
        Whether or not to keep a history of props at all time points
    name : string, optional
        Name of this swarm. Stored in shared_props.
    color : matplotlib color format 
        Default plotting color for swarm 
        (see https://matplotlib.org/stable/tutorials/colors/colors.html).
        Stored in shared_props. Can be overridden by supplying individual (and
        even time varying!) agent colors in a 'color' column of the props
        DataFrame.
    pool : object with a .map method, optional
        Worker pool used to parallelize per-agent immersed-boundary collision
        detection (the main runtime bottleneck when immersed meshes are present).
        Any object exposing ``.map(func, iterable)`` works, including
        ``multiprocessing.Pool``, ``concurrent.futures.ProcessPoolExecutor``, and
        ``concurrent.futures.ThreadPoolExecutor``; choose threads (shares memory,
        no pickling) or processes (true multi-core) just by which one you attach.
        Defaults to None, which runs serially and reproduces the unparallelized
        behavior exactly. The pool is created and owned by the user (Planktos does
        not shut it down). A process-based pool (e.g. ProcessPoolExecutor) is
        recommended and is mainly beneficial for the expensive moving-boundary
        case; for cheap static-mesh collisions, or for thread-based pools, the
        per-agent dispatch overhead can outweigh the work and reduce performance,
        so benchmark before relying on it. The pool is also accessible as
        ``self.pool``
        inside apply_agent_model for user subclasses, but the default agent model
        does not use it; note that parallelizing a custom stochastic agent model
        requires independent per-worker RNG streams (e.g.
        ``np.random.SeedSequence().spawn()``) to avoid shared-RNG contention.
    **kwargs : dict, optional
        keyword arguments to be used in the 'grid' initialization method or
        values to be set as a Swarm object property. In the latter case, these 
        values can be floats, ndarrays, or iterables, but keep in mind that
        problems will result with parsing if the number of agents is
        equal to the spatial dimension - this is to be avoided. This method of 
        specifying agent properties is depreciated: use the shared_props 
        dictionary instead.

    Other Parameters
    ----------------
    grid_dim : tuple of int (x, y, [z])
        number of grid points in x, y, [and z] directions for 'grid' initialization
    testdir : {'x0', 'x1', 'y0', 'y1', ['z0'], ['z1']}, optional
        two character string for testing if grid points are in the interior of 
        an immersed structure and if so, masking them in the grid initialization. 
        The first char is x,y, or z denoting the dimensional direction of the 
        search ray, the second is either 0 or 1 denoting the direction 
        (backward vs. forward) along that direction. See documentation of 
        Swarm.grid_init for more information.

    Attributes
    ----------
    envir : Environment object
        Environment that this Swarm belongs to
    positions : masked array, shape Nx2 (2D) or Nx3 (3D)
        spatial location of all the agents in the swarm. the mask is False for 
        any row corresponding to an agent that is within the spatial boundaries 
        of the environment, otherwise the mask for the row is set to True and 
        the position of that agent is no longer updated
    N : read only property, int
        The current number of agents in the swarm, based on positions.shape[0]
    pos_history : list of masked arrays
        all previous position arrays are stored here. to get their corresponding 
        times, check the time_history attribute of the Swarm's Environment.
    full_pos_history : list of masked arrays
        same as pos_history, but also includes the positions attribute as the 
        last entry in the list
    velocities : masked array, shape Nx2 (2D) or Nx3 (3D)
        velocity of all the agents in the swarm. same masking properties as 
        positions
    vel_history : list of masked arrays
        all previous velocity arrays are stored here. to get their corresponding 
        times, check the time_history attribute of the Swarm's Environment.
    full_vel_history : list of masked arrays
        same as vel_history, but also includes the velocities attribute as the 
        last entry in the list
    accelerations : masked array, shape Nx2 (2D) or Nx3 (3D)
        accelerations of all the agents in the swarm. same masking properties as 
        positions
    ib_collision_idx : 1D array of int with length equal to the swarm size
        For each agent, the index of the mesh element that the agent collided with
        in the most recent time it moved. -1 if no collision.
    props : pandas DataFrame
        Pandas dataframe of individual agent properties that vary between agents. 
        This is the method by which individual variation among the agents should 
        be specified. A special column can be specified called 'angle' which, if 
        props_history is being stored, will plot as the agent heading in 2D.
    props_history : List of past Pandas DataFrames or None
        If not None, this list records individual agent attributes at all 
        previous points in time corresponding to the time_history attribute of 
        the Swarm's Environment.
    full_props_history : List of Pandas DataFrames or None
        props_history plus the current time version of props
    shared_props : dictionary
        dictionary of properties shared by all agents as name-value pairs. Must 
        include 'name' and 'color' indicating the name of the swarm and its 
        default color for plotting. 'mu' and 'cov' are required for Brownian 
        motion, and other properties may be required for other physics.
    rndState : numpy Generator object
        random number generator for this Swarm, seeded by the "seed" parameter
    pool : object with a .map method or None
        Worker pool used to parallelize per-agent immersed-boundary collision
        detection, as supplied by the "pool" parameter. None (the default) runs
        serially. See the parameter documentation above for guidance on choosing
        a pool; it is also accessible as self.pool inside apply_agent_model.

    Notes
    -----
    If the agent behavior you are looking for is simply brownian motion with 
    fluid advection, all you need to do is change the 'cov' entry in the 
    shared_props dictionary to a covariance matrix that matches the amount of 
    jitter you are looking for. You can also add fixed directional bias by 
    editing the 'mu' shared_props entry. This default behavior is then 
    accomplished by solving the relevant SDE using Euler steps, where the step 
    size is the dt argument of the move method which you call in a loop, e.g.::
    
        for ii in range(50):
            swm.move(0.1)

    In order to accommodate general, user-defined behavior algorithms, all other 
    agent behaviors should be explicitly specified by subclassing this Swarm 
    class and overriding the apply_agent_model method. This is easy, and takes the 
    following form: ::

        class myagents(planktos.Swarm):
            
            def apply_agent_model(self, dt):
                #
                # Put any calculations here that are necessary to determine
                #   where the agents should end up after a time step of length 
                #   dt assuming they don't run into a boundary of any sort. 
                #   Boundary conditions, mesh crossings, etc. will be handled 
                #   automatically by Planktos after this function returns. The
                #   new positions you return should be an ndarray of shape NxD
                #   where N is the number of agents in the swarm and D is the 
                #   spatial dimension of the system. The params argument is 
                #   there in case you want this method to take in any external 
                #   info (e.g. time-varying forcing functions, user-controlled 
                #   behavior switching, etc.). Note that this method has full 
                #   access to all of the Swarm attributes via the "self" 
                #   argument. For example, self.positions will return an NxD 
                #   masked array of current agent positions. The one thing this 
                #   method SHOULD NOT do is set the positions, velocities, or 
                #   accelerations attributes of the Swarm. This will be handled 
                #   automatically after this method returns, and after boundary 
                #   conditions have been checked.

                return newpositions

    Then, when you create a Swarm object, create it using::
    
        swrm = myagents() # add Swarm parameters as necessary, as documented above

    This will create a Swarm object, but with your my_positions method instead 
    of the default one!

    Examples
    --------
    Create a default Swarm in an Environment with some fluid data loaded.

    >>> envir = planktos.Environment()
    >>> envir.read_IBAMR3d_vtk_data('../tests/IBAMR_test_data', d_start=5, d_finish=None)
    >>> swrm = Swarm(envir=envir)

    '''

    def __init_subclass__(cls, **kwargs):
        '''Warn when a subclass replaces move() instead of extending it.

        move() is the harness, not the behavior: it records the position,
        velocity and property histories, applies boundary conditions,
        recomputes velocity and acceleration by finite difference, and advances
        the Environment's time. Agent behavior belongs in apply_agent_model and
        post-step bookkeeping in after_move, both of which move() calls at the
        right moment. A subclass that replaces move() outright silently loses
        all of that, and the damage is quiet -- agents keep moving, but nothing
        is recorded and no boundary is enforced.

        Overriding in order to *extend* -- say, to change a default and then
        call super().move(...) -- keeps the harness intact and is fine, so it
        passes without a warning. Whether the override delegates is judged by
        looking for a reference to super or to move in its code, which an
        unusual spelling could slip past. That is the safe direction to be
        wrong in: this only ever warns, so a miss costs a warning that might
        have helped, never a working subclass that stops importing.
        '''

        super().__init_subclass__(**kwargs)
        move = cls.__dict__.get('move')
        if move is None:
            return
        code = getattr(move, '__code__', None)
        if code is not None and ('super' in code.co_names or
                                 'move' in code.co_names):
            return
        warnings.warn(
            "{} overrides Swarm.move without appearing to call it. move() is "
            "the harness that records history, applies boundary conditions, "
            "recomputes velocity and acceleration, and advances time; "
            "replacing it drops all of that silently. Override "
            "apply_agent_model to change how agents move, and after_move to "
            "act on the result. If you meant to extend move(), call "
            "super().move(...) from the override.".format(cls.__name__),
            UserWarning, stacklevel=3)



    def __init__(self, swarm_size=100, envir=None, init='random',
                 ib_condition='sliding', seed=None, shared_props=None,
                 props=None, store_prop_history=False, name='organism',
                 color='darkgreen', pool=None, **kwargs):

        # use a new, 3D default Environment if one was not given. Or infer
        #   dimension from init if possible.
        if envir is None:
            if isinstance(init,str):
                self.envir = Environment(init_swarms=self, Lz=10)
            elif isinstance(init,np.ndarray) and len(init.shape) == 2:
                if init.shape[1] == 2:
                    self.envir = Environment(init_swarms=self)
                else:
                    self.envir = Environment(init_swarms=self, Lz=10)
            else:
                if len(init) == 2:
                    self.envir = Environment(init_swarms=self)
                else:
                    self.envir = Environment(init_swarms=self, Lz=10)
        else:
            try:
                assert envir.__class__.__name__ == 'Environment'
                envir.swarms.append(self)
                self.envir = envir
            except AssertionError as ae:
                print("Error: invalid Environment object.")
                raise ae

        # initialize random number generator
        self.rndState = np.random.default_rng(seed=seed)

        # initialize agent locations
        if isinstance(init,np.ndarray) and len(init.shape) == 2:
            swarm_size = init.shape[0]
        self.positions = ma.zeros((swarm_size, len(self.envir.L)))
        if isinstance(init,str):
            if init == 'random':
                print('Initializing Swarm with uniform random positions...')
                for ii in range(len(self.envir.L)):
                    self.positions[:,ii] = self.rndState.uniform(0, 
                                        self.envir.L[ii], self.N)
            elif init == 'grid':
                assert 'grid_dim' in kwargs, "Required key word argument grid_dim missing for grid init."
                x_num = kwargs['grid_dim'][0]; y_num = kwargs['grid_dim'][1]
                if len(self.envir.L) > 2:
                    z_num = kwargs['grid_dim'][2]
                else:
                    z_num = None
                if 'testdir' in kwargs:
                    testdir = kwargs['testdir']
                else:
                    testdir = None
                print('Initializing Swarm with grid positions...')
                self.positions = self.grid_init(x_num, y_num, z_num, testdir)
                swarm_size = self.N
            else:
                print("Initialization method {} not implemented.".format(init))
                print("Exiting...")
                raise NameError
        else:
            if isinstance(init,np.ndarray) and len(init.shape) == 2:
                assert init.shape[1] == len(self.envir.L),\
                    "Initial location data must be Nx{} to match number of agents.".format(
                    len(self.envir.L))
                self.positions[:,:] = init[:,:]
            else:
                for ii in range(len(self.envir.L)):
                    self.positions[:,ii] = init[ii]

        # Due to overloading of the __setattr__ method below, positions, velocities, 
        #   and accelerations should always have a hard mask automatically.

        # initialize agent velocities
        if self.envir.flow is not None:
            self.velocities = ma.array(self.get_fluid_drift(), mask=self.positions.mask.copy())
        else:
            self.velocities = ma.array(np.zeros((swarm_size, len(self.envir.L))), 
                                       mask=self.positions.mask.copy())

        # initialize agent accelerations
        self.accelerations = ma.array(np.zeros((swarm_size, len(self.envir.L))),
                                      mask=self.positions.mask.copy())

        # Initialize position and velocity history
        self.pos_history = []
        self.vel_history = []

        # Where the agents started the current time step. This is the movement
        #   segment's start point: apply_boundary_conditions tests the line
        #   from here to self.positions against every mesh element to decide
        #   whether an agent crossed a boundary and where to put it if it did.
        #   It is control state for the physics, kept separate from the
        #   recording in pos_history so that what gets recorded, and how often,
        #   cannot change what happens. Set by every loop that moves agents and
        #   then applies boundary conditions -- Swarm.move and the two inlined
        #   loops in Environment.calculate_FTLE. Before the first step the
        #   agents have not moved, so they start where they were constructed.
        self._prev_positions = self.positions.copy()

        # Initialize IB collision detection
        self.ib_collision_idx = np.full(swarm_size, -1) # will be set to mesh index if collision occurs
        self.ib_condition = ib_condition

        # Optional worker pool for parallelizing immersed-boundary collision
        #   detection. Any object exposing a .map(func, iterable) method works
        #   (e.g. multiprocessing.Pool, concurrent.futures.ProcessPoolExecutor,
        #   or ThreadPoolExecutor). None (default) runs serially. See
        #   apply_boundary_conditions. Also accessible as self.pool inside
        #   apply_agent_model for user subclasses, though the default agent model
        #   does not use it.
        self.pool = pool

        # initialize Dataframe of non-shared properties
        if props is None:
            self.props = pd.DataFrame()
            # with random cov
            # self.props = pd.DataFrame(
            #     {'start_pos': [tuple(self.positions[ii,:]) for ii in range(swarm_size)],
            #     'cov': [np.eye(len(self.envir.L))*(0.5+np.random.rand()) for ii in range(swarm_size)]}
            # )
        else:
            self.props = props
        if store_prop_history:
            self.props_history = []
        else:
            self.props_history = None

        # Dictionary of shared properties
        if shared_props is None:
            self.shared_props = {}
        else:
            self.shared_props = shared_props

        # Include necessary default properties if they aren't already set
        if 'mu' not in self.shared_props and 'mu' not in self.props:
            self.shared_props['mu'] = np.zeros(len(self.envir.L))
        if 'cov' not in self.shared_props and 'cov' not in self.props:
            self.shared_props['cov'] = np.eye(len(self.envir.L))
        if 'name' not in self.shared_props:
            self.shared_props['name'] = name
        if 'color' not in self.shared_props:
            self.shared_props['color'] = color

        # Record any kwargs as Swarm parameters
        for name, obj in kwargs.items():
            try:
                if isinstance(obj,np.ndarray) and obj.shape[0] == swarm_size and\
                    obj.shape[0] != len(self.envir.L):
                    self.props[name] = obj
                elif isinstance(obj,np.ndarray):
                    self.shared_props[name] = obj
                elif len(obj) == swarm_size and len(obj) != len(self.envir.L):
                    self.props[name] = obj
                else:
                    # convert iterable to ndarray first
                    self.shared_props[name] = np.array(obj)
            except TypeError:
                # Called len on something that wasn't iterable
                self.shared_props[name] = obj
                    


    # Make sure mask is always hardened for positions, velocities, and accelerations
    def __setattr__(self, name, value):
        if name in ['positions', 'velocities', 'accelerations']:
            assert isinstance(value, np.ndarray), name+" must be an array or masked array."
            if not isinstance(value, ma.masked_array):
                value = ma.masked_array(value)
            value.harden_mask()
        super().__setattr__(name, value)



    def grid_init(self, x_num, y_num, z_num=None, testdir=None):
        '''Return a flattened array which describes a regular grid of locations, 
        except potentially masking any grid points in the interior of a closed, 
        immersed structure. 
        
        The full, unmasked grid will be x_num by y_num [by z_num] on the 
        interior and boundaries of the domain. The output of this method is 
        appropriate for finding FTLE, and that is its main purpose. It will 
        automatically be called by the Environment class's calculate_FTLE method, 
        and if you want to initialize a Swarm with a grid this is possible by 
        passing the init='grid' keyword argument when the Swarm is created. 
        So there is probably no reason to use this method directly.

        Grid list moves in the [Z direction], Y direction, then X direction (due 
        to C order of memory layout).

        Parameters
        ----------
        x_num, y_num, [z_num] : int
            number of grid points in each direction
        testdir : {'x0', 'x1', 'y0', 'y1', ['z0'], ['z1']}, optional
            to check if a point is an interior to an immersed structure, a line 
            will be drawn from the point to a domain boundary. If the number 
            of immersed boundary intersections is odd, the point will be 
            considered interior and masked. This check will not be run at all 
            if testdir is None. Otherwise, specify a direction with one of the 
            following: 'x0','x1','y0','y1','z0','z1' (the last two for 
            3D problems only) denoting the dimension (x,y, or z) and the 
            direction (0 for negative, 1 for positive).

        Notes
        -----
        This algorithm is meant as a heuristic only! It is not guaranteed to mask 
        all interior grid points, and will mask non-interior points if there is 
        not a clear line from the point to one of the boundaries of the domain. 
        If this method fails for your geometry and better accuracy is needed, 
        use this method as a starting point and mask/unmask as necessary.
        '''

        # Form initial grid
        x_pts = np.linspace(0, self.envir.L[0], x_num)
        y_pts = np.linspace(0, self.envir.L[1], y_num)
        if z_num is not None:
            z_pts = np.linspace(0, self.envir.L[2], z_num)
            X1, X2, X3 = np.meshgrid(x_pts, y_pts, z_pts, indexing='ij')
            xidx, yidx, zidx = np.meshgrid(np.arange(x_num), np.arange(y_num), 
                                           np.arange(z_num), indexing='ij')
            DIM = 3
        elif len(self.envir.L) > 2:
            raise RuntimeError("Must specify z_num for 3D problems.")
        else:
            X1, X2 = np.meshgrid(x_pts, y_pts, indexing='ij')
            xidx, yidx = np.meshgrid(np.arange(x_num), np.arange(y_num), 
                                     indexing='ij')
            DIM = 2

        if testdir is None:
            if DIM == 2:
                return ma.array([X1.flatten(), X2.flatten()]).T
            else:
                return ma.array([X1.flatten(), X2.flatten(), X3.flatten]).T
        elif testdir[0] == 'z' and len(self.envir.L) < 3:
            raise RuntimeError("z-direction unavailable in 2D problems.")

        # Translate directional input
        startdim = [0,0]
        if testdir[0] == 'x':
            startdim[0] = 0
            if DIM == 2:
                perp_idx = 1
            else:
                perp_idx = [1,2]
        elif testdir[0] == 'y':
            startdim[0] = 1
            if DIM == 2:
                perp_idx = 0
            else:
                perp_idx = [0,2]
        elif testdir[0] == 'z':
            startdim[0] = 2
            perp_idx = [0,1]
        else:
            raise RuntimeError("Unrecognized value in testdir, {}.".format(testdir))
        try:
            startdim[1] = int(testdir[1]) - 1
        except ValueError:
            raise RuntimeError("Unrecognized value in testdir, {}.".format(testdir))
        
        # Idea: start on opposite side of domain as given direction and take a
            #   full grid on the boundary. See if there are intersections.
            #   If none, none of the points along that ray need to be tested further
            #   Otherwise, we are also given the intersection points. Use to
            #   deduce the rest.

        # startdim gives the dimension and index on which to place a grid
        
        # Convert X1, X2, X3 to masked arrays
        X1 = ma.array(X1)
        X2 = ma.array(X2)
        if DIM == 3:
            X3 = ma.array(X3)
            grids = [X1, X2, X3]
            idx_list = [xidx, yidx, zidx]
        else:
            grids = [X1, X2]
            idx_list = [xidx, yidx]

        # get a list of the gridpoints on correct side of the domain
        firstpts = []
        first_idx_list = []
        for X, idx in zip(grids,idx_list):
            if startdim[0] == 0:
                firstpts.append(X[startdim[1],...])
                first_idx_list.append(idx[startdim[1],...])
            elif startdim[0] == 1:
                firstpts.append(X[:,startdim[1],...])
                first_idx_list.append(idx[:,startdim[1],...])
            else:
                firstpts.append(X[:,:,startdim[1]])
                first_idx_list.append(idx[:,:,startdim[1]])
        firstpts = np.array([X.flatten() for X in firstpts]).T
        idx_vals = np.array([idx.flatten() for idx in first_idx_list]).T

        mesh = self.envir.ibmesh
        meshptlist = mesh.reshape((mesh.shape[0]*mesh.shape[1],mesh.shape[2]))
        for pt, idx in zip(firstpts,idx_vals):
            # for each pt in the grid, get a list of eligibible mesh elements as
            #   those who have a point within a cylinder of diameter envir.max_meshpt_dist
            if DIM == 3:
                pt_bool = np.linalg.norm(meshptlist[:,perp_idx]-pt[perp_idx], 
                    axis=1)<=self.envir.max_meshpt_dist/2
            else:
                pt_bool = np.abs(meshptlist[:,perp_idx]-pt[perp_idx])\
                    <=self.envir.max_meshpt_dist/2
            pt_bool = pt_bool.reshape((mesh.shape[0], mesh.shape[1]))
            close_mesh = mesh[np.any(pt_bool, axis=1)]

            endpt = np.array(pt)
            if startdim[1] == -1:
                endpt[startdim[0]] = 0
            else:
                endpt[startdim[0]] = self.envir.L[startdim[0]]

            # Get intersections
            if DIM == 2:
                intersections = _geom.seg_intersect_2D(pt, endpt,
                    close_mesh[:,0,:], close_mesh[:,1,:], get_all=True)
            else:
                intersections = _geom.seg_intersect_3D_triangles(pt, endpt,
                    close_mesh[:,0,:], close_mesh[:,1,:], close_mesh[:,2,:], get_all=True)

            # For completeness, we should also worry about edge cases where 
            #   intersections are not of mesh facets but of triangle points, but
            #   as a heuristic, we will ignore this. A tweaking of the number of
            #   grid points used could fix this problem in most cases, or it
            #   could be fixed by hand.

            if intersections is not None:
                # Sort the intersections by distance away from pt
                intersections = sorted(intersections, key=lambda x: x[1])

                # get list of all x,y, or z values for points along the ray
                #   (where the dimension matches the direction of the ray)
                if startdim[0] == 0:
                    current_pt_val = pt[0] - 10e-7
                    if DIM == 3:
                        val_list = X1[:,idx[1],idx[2]]
                    else:
                        val_list = X1[:,idx[1]]
                elif startdim[0] == 1:
                    current_pt_val = pt[1] - 10e-7
                    if DIM == 3:
                        val_list = X2[idx[0],:,idx[2]]
                    else:
                        val_list = X2[idx[0],:]
                else:
                    current_pt_val = pt[2] - 10e-7
                    val_list = X3[idx[0],idx[1],:]

                while len(intersections) > 0:
                    n = len(intersections)
                    intersection = intersections.pop(0)
                    if startdim[0] == 0:
                        intersect_val = intersection[0][0]
                    elif startdim[0] == 1:
                        intersect_val = intersection[0][1]
                    else:
                        intersect_val = intersection[0][2]

                    if current_pt_val < intersect_val:
                        pair = [current_pt_val, intersect_val]
                    else:
                        pair = [intersect_val, current_pt_val]
                    
                    # gather all points between current one and intersection
                    #   This will always mask points exactly on a mesh boundary
                    bool_list = np.logical_and(pair[0]<=val_list,val_list<=pair[1])

                    # set mask if number of intersections is odd
                    if n%2 == 1:
                        if startdim[0] == 0:
                            if DIM == 3:
                                X1[bool_list,idx[1],idx[2]] = ma.masked
                                X2[bool_list,idx[1],idx[2]] = ma.masked
                                X3[bool_list,idx[1],idx[2]] = ma.masked
                            else:
                                X1[bool_list,idx[1]] = ma.masked
                                X2[bool_list,idx[1]] = ma.masked
                        elif startdim[0] == 1:
                            if DIM == 3:
                                X1[idx[0],bool_list,idx[2]] = ma.masked
                                X2[idx[0],bool_list,idx[2]] = ma.masked
                                X3[idx[0],bool_list,idx[2]] = ma.masked
                            else:
                                X1[idx[0],bool_list] = ma.masked
                                X2[idx[0],bool_list] = ma.masked
                        else:
                            X1[idx[0],idx[1],bool_list] = ma.masked
                            X2[idx[0],idx[1],bool_list] = ma.masked
                            X3[idx[0],idx[1],bool_list] = ma.masked

                    # Update current_pt_val to latest intersection
                    current_pt_val = intersect_val

        # all points done.
        if DIM == 2:
            return ma.array([X1.flatten(), X2.flatten()]).T
        else:
            return ma.array([X1.flatten(), X2.flatten(), X3.flatten()]).T



    @property
    def full_pos_history(self):
        '''History of self.positions, including present time.'''
        return [*self.pos_history, self.positions]
    


    @property
    def full_vel_history(self):
        '''History of self.positions, including present time.'''
        return [*self.vel_history, self.velocities]



    @property
    def full_props_history(self):
        '''History of self.props, including present time.'''
        if self.props_history is not None:
            return [*self.props_history, self.props]
        else:
            return None



    @property
    def N(self):
        '''Return the number of agents based on the number of entries in
        self.positions'''
        return self.positions.shape[0]



    def save_data(self, path, name, pos_fmt='%.18e'):
        '''Save the full position history (with mask and time stamps) along with 
        current velocity and acceleration to csv files. Save shared_props to a 
        npz file and save props to json.

        The output format for the position csv is the same as for the 
        save_pos_to_csv method.
        
        shared_props is saved as an npz file since it is likely to contain some 
        mixture of scalars and arrays, but does not vary between the agents so 
        is less likely to be loaded outside of Python. props is saved to json 
        since it is likely to contain a variety of types of data, may need to be 
        loaded outside of Python, and json will be human readable.

        props_history is not saved.

        Parameters
        ----------
        path : str
            directory for storing data
        name : str 
            prefix name for data files
        pos_fmt : str format, default='%.18e'
            format and precision for storing position, vel, and accel data

        See Also
        --------
        save_pos_to_csv
        save_pos_to_vtk
        '''

        path = Path(path)
        if not path.is_dir():
            os.makedirs(path)

        self.save_pos_to_csv(str(path/name), pos_fmt, sv_vel=True, sv_accel=True)

        props_file = path/(name+'_props.json')
        self.props.to_json(str(props_file))
        shared_props_file = path/(name+'_shared_props.npz')
        np.savez(str(shared_props_file), **self.shared_props)



    def save_pos_to_csv(self, filename, fmt='%.18e', sv_vel=False, sv_accel=False):
        '''Save the full position history including present time, with mask and 
        time stamps, to a csv.

        The output format for the position csv will be as follows:

        * The first row contains cycle and time information. The cycle is given, 
          and then each time stamp is repeated D times, where D is the spatial 
          dimension of the system.
        * Each subsequent row corresponds to a different agent in the Swarm.
        * Reading across the columns of an agent row: first, a boolean is given
          showing the state of the mask for that time step. Agents are masked
          when they have exited the domain. Then, the position vector is given
          as a group of D columns for the x, y, (and z) direction. Each set
          of 1+D columns then corresponds to a different cycle/time, as 
          labeled by the values in the first row.

        The result is a csv that is N+1 by (1+D)*T, where N is the number of 
        agents, D is the dimension of the system, and T is the number of 
        times recorded.

        Parameters
        ----------
        filename : str 
            path/name of the file to save the data to
        fmt : str format, default='%.18e'
            fmt argument to be passed to numpy.savetxt for format and precision 
            of numerical data
        sv_vel : bool, default=False
            whether or not to save the current time velocity data
        sv_accel : book, default=False
            whether or not to save the current time acceleration data

        See Also
        --------
        save_data
        save_pos_to_vtk
        '''
        if filename[-4:] != '.csv':
            filename = filename + '.csv'

        full_time = [*self.envir.time_history, self.envir.time]
        time_row = np.concatenate([[ii]+[jj]*self.positions.shape[1] 
                   for ii,jj in zip(range(len(full_time)), full_time)])

        fmtlist = ['%u'] + [fmt]*self.positions.shape[1]

        np.savetxt(filename, np.vstack((time_row, 
                   np.column_stack([mat for pos in self.full_pos_history for mat in (ma.getmaskarray(pos[:,0]), pos.data)]))),
                   fmt=fmtlist*len(full_time), delimiter=',')

        if sv_vel:
            vel_filename = filename[:-4] + '_vel.csv'
            np.savetxt(vel_filename, 
                   np.column_stack((self.velocities[:,0].mask, self.velocities.data)),
                   fmt=fmtlist, delimiter=',')
        if sv_accel:
            accel_filename = filename[:-4] + '_accel.csv'
            np.savetxt(accel_filename, 
                   np.column_stack((self.accelerations[:,0].mask, self.accelerations.data)),
                   fmt=fmtlist, delimiter=',')


    
    def save_pos_to_vtk(self, path, name, all=True):
        '''Save position data to vtk as point data (PolyData).
        A different file will be created for each time step in the history, or
        just one file of the current positions will be created if the all 
        argument is False.

        Parameters
        ----------
        path : str 
            location to save the data
        name : str 
            name of dataset
        all : bool 
            if True, save the entire history including the current time. 
            If false, save only the current time.

        See Also
        --------
        save_data
        save_pos_to_csv
        '''
        if len(self.envir.L) == 2:
            DIM2 = True
        else:
            DIM2 = False

        if not all or len(self.envir.time_history) == 0:
            if DIM2:
                data = np.zeros((self.positions[~ma.getmaskarray(self.positions[:,0]),:].shape[0],3))
                data[:,:2] = self.positions[~ma.getmaskarray(self.positions[:,0]),:]
                _dataio.write_vtk_point_data(path, name, data)
            else:
                _dataio.write_vtk_point_data(path, name, self.positions[~ma.getmaskarray(self.positions[:,0]),:])
        else:
            for cyc, time in enumerate(self.envir.time_history):
                # ma.getmaskarray (not .mask) is required: an unmasked history
                # entry has .mask == nomask (scalar), and ~nomask would index in
                # a new axis rather than select the unmasked rows.
                unmasked = self.pos_history[cyc][
                    ~ma.getmaskarray(self.pos_history[cyc][:,0]),:]
                if DIM2:
                    data = np.zeros((unmasked.shape[0],3))
                    data[:,:2] = unmasked
                    _dataio.write_vtk_point_data(path, name, data,
                                                 cycle=cyc, time=time)
                else:
                    _dataio.write_vtk_point_data(path, name, unmasked,
                                                 cycle=cyc, time=time)
            cyc = len(self.envir.time_history)
            if DIM2:
                data = np.zeros((self.positions[~ma.getmaskarray(self.positions[:,0]),:].shape[0],3))
                data[:,:2] = self.positions[~ma.getmaskarray(self.positions[:,0]),:]
                _dataio.write_vtk_point_data(path, name, data, cycle=cyc,
                                             time=self.envir.time)
            else:
                _dataio.write_vtk_point_data(path, name, 
                    self.positions[~ma.getmaskarray(self.positions[:,0]),:],
                    cycle=cyc, time=self.envir.time)



    def _change_envir(self, envir):
        '''Manages a change from one Environment to another.

        Reached through Environment.add_swarm when it is handed a Swarm object
        rather than a size -- which is how a Swarm built on its own (and so
        given a default Environment) is moved into the one it belongs in, and
        how Environment.calculate_FTLE adds its working copy.
        '''

        if self.positions.shape[1] != len(envir.L):
            if self.positions.shape[1] > len(envir.L):
                # Project swarm down to 2D
                self.positions = self.positions[:,:2]
                self.velocities = self.velocities[:,:2]
                self.accelerations = self.accelerations[:,:2]
                # Update known properties
                if 'mu' in self.shared_props:
                    self.shared_props['mu'] = self.shared_props['mu'][:2]
                    print('mu has been projected to 2D.')
                if 'mu' in self.props:
                    for n,mu in enumerate(self.props['mu']):
                        self.props['mu'][n] = mu[:2]
                    print('mu has been projected to 2D.')
                if 'cov' in self.shared_props:
                    self.shared_props['cov'] = self.shared_props['cov'][:2,:2]
                    print('cov has been projected to 2D.')
                if 'cov' in self.props:
                    for n,cov in enumerate(self.props['cov']):
                        self.props['cov'][n] = cov[:2,:2]
                    print('cov has been projected to 2D.')
                # warn about others
                other_props = [x for x in self.props if x not in ['mu', 'cov']]
                other_props += [x for x in self.shared_props if x not in ['mu', 'cov']]
                if len(other_props) > 0:
                    print('WARNING: other properties {} were not projected.'.format(other_props))
            else:
                raise RuntimeError("Swarm dimension smaller than new Environment dimension!"+
                    " Cannot scale up!")

        # Leave the old Environment before joining the new one. Without this a
        #   moved swarm stays in both lists: the old environment's move_swarms
        #   would go on moving it, and a recording there would go on capturing
        #   it. Identity, not equality -- Swarm defines no __eq__, and two
        #   swarms are never interchangeable anyway.
        # Nothing is removed when the swarm is not in that list to begin with,
        #   which is the case for calculate_FTLE's working copy: it is a shallow
        #   copy, so it already points at this Environment without being one of
        #   its swarms. That is also why the recording check sits inside the
        #   branch -- an FTLE field computed during a recording adds and drops a
        #   copy without changing which swarms are in the run, and must not be
        #   refused.
        old = self.envir
        if old is not None and any(s is self for s in old.swarms):
            old._refuse_while_recording()
            for n, s in enumerate(old.swarms):
                if s is self:
                    del old.swarms[n]
                    break

        self.envir = envir
        envir.swarms.append(self)



    def calc_re(self, u, diam=None):
        '''Calculate and return the Reynolds number as experienced by a swarm 
        with characteristic length 'diam' in a fluid moving with velocity u. All 
        other parameters will be pulled from the Environment's attributes. 
        
        If diam is not specified, this method will look for it in the 
        shared_props dictionary of this Swarm.
        
        Parameters
        ----------
        u : float
            characteristic fluid speed, m/s
        diam : float, optional
            characteristic length scale of a single agent, m

        Returns
        -------
        float
            Reynolds number
        '''

        if diam is None:
            diam = self.shared_props['diam']
        else:
            diam = diam
        if self.envir.rho is not None and self.envir.mu is not None and\
            'diam' in self.shared_props:
            return self.envir.rho*u*diam/self.envir.mu
        else:
            raise RuntimeError("Parameters necessary for Re calculation in Environment are undefined.")



    def move(self, dt=1.0, ib_collisions='default', 
             update_time=True, silent=False):
        '''Move all organisms in the swarm over one time step of length dt.
        DO NOT override this method when subclassing; override apply_agent_model
        instead!!!

        Performs a lot of utility tasks such as updating the positions and 
        pos_history attributes, checking boundary conditions, and recalculating 
        the current velocities and accelerations attributes.

        Parameters
        ----------
        dt : float
            length of time step to move all agents
        ib_collisions : {None, 'default', 'sliding', 'sticky'}
            Boundary condition for immersed boundaries. If 'default', use the 
            default found in self.ib_condition. If None, turn off all 
            interaction with immersed boundaries. In sliding collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        update_time : bool, default=True
            whether or not to update the Environment's time by dt. This exists
            for Environment.move_swarms, which moves every Swarm with
            update_time=False and then advances the time once for all of them.
            There is no reason for a user to pass False.
        silent : bool, default=False
            If True, suppress printing the updated time.

        Raises
        ------
        RuntimeError
            if the Environment holds more than one Swarm and update_time is
            True. One Swarm cannot advance the environment time on its own
            while the others stand still; call Environment.move_swarms instead.

        See Also
        --------
        apply_agent_model :
            method that returns (but does not assign) the new positions of the
            swarm after the time step dt, which Planktos users override in order
            to specify their own, custom agent behavior.
        Environment.move_swarms :
            moves every Swarm in the Environment and then advances the time.
        '''

        # A time of None marks an Environment left in an error state by a
        #   time step that raised partway through (see the except clause
        #   below). Everything the agents currently hold is a half-applied
        #   step, so there is nothing sensible to move on from.
        if self.envir.time is None:
            raise RuntimeError(
                "Cannot move: this Swarm/Environment is in an error state. A "
                "previous time step raised or was interrupted while boundary "
                "conditions were being applied, so agent positions, "
                "velocities, accelerations "
                "and ib_collision_idx hold a step that was applied to some "
                "agents but not others -- some may be inside an immersed "
                "boundary. envir.time was set to None to mark this.\n"
                "The recorded histories are complete and consistent up to and "
                "including the failed step. To back that step out and carry "
                "on, pop the last entry off each:\n"
                "    envir.time = envir.time_history.pop()\n"
                "    swrm.positions = swrm.pos_history.pop()\n"
                "    swrm.velocities = swrm.vel_history.pop()")

        # Advancing the environment clock on behalf of one Swarm while the
        #   others stand still is not supported. It used to warn and then
        #   freeze the other swarms by appending their current positions to
        #   pos_history -- but only to pos_history, so vel_history and
        #   props_history fell behind and full_vel_history no longer lined up
        #   with full_pos_history for the rest of the session. Every consumer
        #   that pairs the two (plot_all's heading arrows, _calc_basic_stats)
        #   then read the wrong entry or raised. There is no half-moved state
        #   worth recording, so this is an error rather than a repaired
        #   approximation of one.
        if update_time and len(self.envir.swarms) > 1:
            raise RuntimeError(
                "This Environment holds {} Swarms, so no single Swarm can "
                "advance its time. Call\n"
                "    envir.move_swarms(dt)\n"
                "instead, which moves every Swarm and then advances the time "
                "once for all of them.".format(len(self.envir.swarms)))

        if ib_collisions == 'default':
            ib_collisions = self.ib_condition

        # Is the state this step begins from one of the ones kept? Asked once,
        #   here, and used for every history append below and for the matching
        #   time_history append at the end.
        keep_state = self.envir._records_this_step()

        # Save current position to put in the history
        old_positions = self.positions.copy()
        old_velocities = self.velocities.copy()

        # ...and hand the same array to the boundary stage as this step's
        #   movement start point. It reads this rather than pos_history[-1] so
        #   that the physics does not depend on the recording; see __init__.
        self._prev_positions = old_positions

        # Conditionally save props to put in the history too
        if self.props_history is not None:
            old_props = self.props.copy()

        # Check that something is left in the domain to move, and move it.
        if not np.all(self.positions.mask):
            # Update positions, preserving mask
            self.positions[:,:] = self.apply_agent_model(dt)

        # Update history
        if keep_state:
            self.pos_history.append(old_positions)
            self.vel_history.append(old_velocities)
            if self.props_history is not None:
                self.props_history.append(old_props)
        
        # Update velocity and acceleration of swarm
        self.velocities[:,:] = (self.positions - old_positions)/dt
        self.accelerations[:,:] = (self.velocities - old_velocities)/dt

        # Apply boundary conditions (if anything was moving)
        if not np.all(self.positions.mask):
            try:
                self.apply_boundary_conditions(dt, ib_collisions=ib_collisions)
                self.after_move(dt)
            except BaseException as err:
                # Boundary conditions are applied one agent at a time, so this
                #   leaves the step applied to some agents and not others. The
                #   partial state is left alone -- it is what there is to debug
                #   -- but two things are done to make it legible. First,
                #   record the time this step started, so that time_history
                #   matches the pos_history entry appended above and the
                #   histories stay a consistent record. Second, set the time to
                #   None, which marks everything current as untrustworthy and
                #   is what move() refuses to run on.
                # BaseException rather than Exception: interrupting a long run
                #   with Ctrl-C is the most common way one ends, and a
                #   KeyboardInterrupt lands here exactly as an error does,
                #   leaving the same half-applied step. The state is marked the
                #   same way for both; only the reporting differs, because an
                #   interrupt has to keep propagating as itself instead of
                #   being wrapped into something an outer "except Exception"
                #   would swallow.
                if keep_state:
                    # Only when this step appended to pos_history above --
                    #   otherwise this closes the histories off *inconsistently*,
                    #   which is the exact thing it exists to prevent.
                    self.envir.time_history.append(self.envir.time)
                self.envir.time = None
                if not isinstance(err, Exception):
                    print("\nInterrupted partway through applying boundary "
                          "conditions or after_move, after agent positions had "
                          "already been updated. The step was applied to some "
                          "agents and not others, so positions, velocities, "
                          "accelerations and ib_collision_idx are all "
                          "unreliable and agents may be sitting inside an "
                          "immersed boundary. They are left as they are so the "
                          "state can be inspected.\n"
                          "envir.time has been set to None to mark this, and "
                          "no further moves are permitted until it is restored. "
                          "The histories were closed off consistently. To back "
                          "the step out and carry on:\n"
                          "    envir.time = envir.time_history.pop()\n"
                          "    swrm.positions = swrm.pos_history.pop()\n"
                          "    swrm.velocities = swrm.vel_history.pop()")
                    raise
                raise RuntimeError(
                    "Boundary conditions or after_move raised partway through this time "
                    "step, after agent positions had already been updated. "
                    "The step was applied to some agents and not others, so "
                    "positions, velocities, accelerations and "
                    "ib_collision_idx are all unreliable and agents may be "
                    "sitting inside an immersed boundary. They are left as "
                    "they are so the failure can be inspected.\n"
                    "envir.time has been set to None to mark this state, and "
                    "no further moves are permitted until it is restored. The "
                    "histories were closed off consistently: time_history now "
                    "matches pos_history, both ending with the state as it "
                    "was when this step began. To back the step out and carry "
                    "on, pop the last entry off each:\n"
                    "    envir.time = envir.time_history.pop()\n"
                    "    swrm.positions = swrm.pos_history.pop()\n"
                    "    swrm.velocities = swrm.vel_history.pop()") from err

        # Record new time
        if update_time:
            if keep_state:
                self.envir.time_history.append(self.envir.time)
            self.envir.time += dt
            if not silent:
                print('time = {}'.format(np.round(self.envir.time,11)))
            # The environment has advanced one step, with every swarm moved
            #   (there is only one, or this would have raised above).
            self.envir._notify_step_complete()
        elif not self.envir._in_move_swarms and self.envir._recorder is not None:
            # update_time=False exists for move_swarms to call. Reached any
            #   other way it means the caller intends to advance the clock by
            #   hand, and a hand-rolled advance fires no hook -- so the step
            #   happens but the archive never sees it.
            warnings.warn(
                "move(update_time=False) while recording: this step will not "
                "be captured, because nothing here advances the environment "
                "time and the capture hook rides that advance. Use "
                "envir.move_swarms(dt) to move every swarm and advance the "
                "time together.", UserWarning)



    def apply_agent_model(self, dt):
        '''Returns the new agent positions after a time step of dt.

        THIS IS THE METHOD TO OVERRIDE IF YOU WANT DIFFERENT MOVEMENT! Do not 
        change the call signature.

        This method returns the new positions of all agents following a time 
        step of length dt, whether due to behavior, drift, or anything else. It 
        should not set the self.positions attribute. Similarly, self.velocities 
        and self.accelerations will automatically be updated outside of this 
        method using finite differences. The only attributes it should change is 
        if there are any user-defined, time-varying agent properties that should 
        be different after the time step (whether shared among all agents, and 
        thus in self.shared_props, or individual to each agent, and thus in 
        self.props). These can be altered directly or by using the add_prop 
        method of this class.

        In this default implementation, movement is a random walk with drift
        as given by an Euler step solver of the appropriate SDE for this process.
        Drift is the local fluid velocity plus self.get_prop('mu') ('mu' is a 
        shared_prop attribute), and the stochasticity is determined by the 
        covariance matrix self.get_prop('cov') ('cov' is also a shared_prop 
        attribute).

        Parameters
        ----------
        dt : float
            length of time step

        Returns
        -------
        ndarray :
            NxD array of new agent positions after a time step of dt given that 
            the agents started at self.positions. N is the number of agents and 
            D is the spatial dimension of the system.

        Notes
        -----
        When writing code for this method, it can be helpful to make use of the 
        ode generators and solvers in the planktos.motion module. Please see the 
        documentation for the functions of this module for options. 
        
        To access the current positions of each agent, use self.positions. 
        self.positions is a masked, NxD array of agent positions where the mask 
        refers to whether or not the agent has exited the domain. You do not 
        want to accidentally edit self.positions directly, so make sure that you 
        get a value copy of self.positions using self.positions.copy() whenever 
        that copy will be modified. Direct assignment of self.positions is by 
        reference.

        Similarly,self.velocities and self.accelerations will provide initial 
        velocities and accelerations for the time step for each agent 
        respectively. Use .copy() as necessary and do not directly assign to 
        these variables; they will be automatically updated later in the 
        movement process. 
        
        The get_fluid_drift method will return the fluid velocity at each agent 
        location using interpolation. Call it once outside of a loop for speed. 
        Similarly, the get_dudt method will return the time derivative of the 
        fluid velocity at the location of each agent. The get_fluid_mag_gradient 
        method will return the gradient of the magnitude of the fluid velocity 
        at the location of each agent.

        See Also
        --------
        get_prop : 
            given an agent/Swarm property name, return the value(s). When 
            accessing a property in Swarm.props, this can be preferred over 
            accessing the property directly through the because instead of 
            returning a pandas Series object (for a column in the DataFrame), it 
            automatically converts to a numpy array first.
        add_prop : add a new agent/Swarm property or overwrite an old one
        get_fluid_drift : return the fluid velocity at each agent location
        get_dudt : return time derivative of fluid velocity at each agent
        get_fluid_mag_gradient : 
            return the gradient of the magnitude of the fluid velocity at each 
            agent
        '''

        # default behavior for Euler_brownian_motion is dift due to mu property
        #   plus local fluid velocity and diffusion given by cov property
        #   specifying the covariance matrix.
        return motion.Euler_brownian_motion(self, dt)



    def after_move(self, dt):
        '''This method is called after the Swarm's spatial positions have been 
        updated via apply_agent_model, but before the environment time has been 
        updated to the new time (prev time + dt).

        By default it does nothing, but you can override it in order to update 
        agent properties or other things that should be set based on the state 
        of the system at the end of the time step. For instance, you could use 
        it to color agents that satisfy certain criteria, or have them switch 
        state based upon their ending position.

        Parameters
        ----------
        dt : float
            length of time step
        '''
        pass



    def get_prop(self, prop_name):
        '''Return the property requested as either a scalar (if shared) or a 
        numpy array, ready for use in vectorized operations (left-most index
        specifies the agent).
        
        Parameters
        ----------
        prop_name : str
            name of the property to return

        Returns
        -------
        property : float or ndarray
        '''

        if prop_name in self.props:
            if prop_name in self.shared_props:
                warnings.warn('Property {} exists '.format(prop_name)+
                'in both props and shared_props. Using the props version.')
            return np.stack(self.props[prop_name].array, axis=0).squeeze()
        elif prop_name in self.shared_props:
            return self.shared_props[prop_name]
        else:
            raise KeyError('Property {} not found.'.format(prop_name))



    def add_prop(self, prop_name, value, shared=False):
        '''Method that will automatically delete any conflicting properties
        when adding a new one.
        
        Parameters
        ----------
        prop_name : str
            name of the property to add
        value : any
            value to set the property at
        shared : bool
            if False, set as a property that applies to all agents in the swarm. 
            if True, value should be an ndarray with a number of rows equal to 
            the number of agents in the swarm, and the property will be set as 
            a column in the Swarm.props DataFrame.
        '''
        if shared:
            self.shared_props[prop_name] = value
            if prop_name in self.props:
                del self.props[prop_name]
        else:
            self.props[prop_name] = value
            if prop_name in self.shared_props:
                del self.shared_props[prop_name]



    def get_fluid_drift(self, time=None, positions=None):
        '''Return fluid-based drift for all agents via interpolation.

        Current swarm position is used unless alternative positions are explicitly
        passed in. Any passed-in positions must be an NxD array where N is the
        number of points and D is the spatial dimension of the system.
        
        In the returned 2D ndarray, each row corresponds to an agent (in the
        same order as listed in self.positions) and each column is a dimension.

        Parameters
        ----------
        time : float, optional
            time at which to return the fluid drift. defaults to the current 
            environment time
        positions : ndarray, optional
            positions at which to return the fluid drift. defaults to the 
            locations of the swarm agents, self.positions

        Returns
        -------
        ndarray with shape NxD, where N is the number of agents and D the 
            spatial dimension
        '''

        # 3D?
        DIM3 = (len(self.envir.L) == 3)

        if positions is None:
            positions = self.positions

        # Interpolate fluid flow
        if self.envir.flow is None:
            return np.zeros(positions.shape)
        else:
            if time is None:
                return self.envir.interpolate_flow(positions, method='linear')
            else:
                return self.envir.interpolate_flow(positions, time=time,
                                                   method='linear')



    def get_dudt(self, time=None, positions=None):
        '''Return fluid time derivative at given positions via interpolation.

        Current swarm position is used unless alternative positions are explicitly
        passed in.
        
        In the returned 2D ndarray, each row corresponds to an agent (in the
        same order as listed in self.positions) and each column is a dimension.

        Parameters
        ----------
        time : float, optional
            time at which to return the data. defaults to the current 
            environment time
        positions : ndarray, optional
            positions at which to return the data. defaults to the locations of 
            the swarm agents, self.positions

        Returns
        -------
        ndarray with shape NxD, where N is the number of agents and D the 
            spatial dimension
        '''

        if positions is None:
            positions = self.positions

        return self.envir.interpolate_flow(positions, self.envir.get_dudt(time=time),
                                           method='linear')



    def get_fluid_mag_gradient(self, positions=None):
        '''Return the gradient of the magnitude of the fluid velocity at all
        agent positions (or at provided positions) via linear interpolation of 
        the gradient.

        The gradient is linearly interpolated from the fluid grid to the
        agent locations. The current environment time is always used, 
        interpolated from data if necessary

        Parameters
        ----------
        positions : ndarray, optional
            positions at which to return the data. defaults to the locations of 
            the swarm agents, self.positions

        Returns
        -------
        ndarray with shape NxD, where N is the number of agents and D the 
            spatial dimension
        '''

        if positions is None:
            positions = self.positions

        TIME_DEP = self.envir.flow.flow_times is not None
        flow_grad = None

        # If available, use the already calculated gradient (if it's at the
        #   correct time)
        if self.envir.mag_grad is not None:
            if not TIME_DEP:
                flow_grad = self.envir.mag_grad
            elif self.envir.mag_grad_time == self.envir.time:
                flow_grad = self.envir.mag_grad

        # Otherwise, calculate the gradient
        if flow_grad is None:
            self.envir.calculate_mag_gradient()
            flow_grad = self.envir.mag_grad

        # Interpolate the gradient at agent positions and return
        return self.envir.interpolate_flow(positions, flow_grad, method='linear')



    def get_DuDt(self, time=None, positions=None):
        '''Return the material derivative with respect to time of the fluid 
        velocity at all agent positions (or at provided positions) via linear 
        interpolation of the material gradient.
        
        Current swarm position is used unless alternative positions are explicitly
        passed in.
        
        In the returned 2D ndarray, each row corresponds to an agent (in the
        same order as listed in self.positions) and each column is a dimension.

        Parameters
        ----------
        time : float, optional
            time at which to return the data. defaults to the current 
            environment time
        positions : ndarray, optional
            positions at which to return the data. defaults to the locations of 
            the swarm agents, self.positions

        Returns
        -------
        ndarray with shape NxD, where N is the number of agents and D the 
            spatial dimension
        '''

        if positions is None:
            positions = self.positions

        if time is None:
            time = self.envir.time         

        # Interpolate at agent positions and return
        return self.envir.interpolate_flow(positions, 
                                           self.envir.calculate_DuDt(time=time), 
                                           method='linear')



    def apply_boundary_conditions(self, dt, ib_collisions='sliding'):
        '''Apply boundary conditions to self.positions.
        
        There is no reason for a user to call this method directly; it is 
        automatically called by self.move after updating agent positions 
        according to the algorithm found in self.apply_agent_model.

        This method compares current agent positions (self.positions) to the
        positions the agents started this time step from (self._prev_positions,
        set by whichever loop just moved them) in order to
        first: determine if the agent collided with any immersed structures and
        if so, to update self.positions using a sliding collision algorithm 
        based on vector projection and second: assess whether or not any agents 
        exited the domain and if so, update their positions based on the 
        boundary conditions as specified in the environment class (self.envir).

        For noflux boundary conditions such sliding projections are really simple 
        (since the domain is just a box), so we just do them directly/manually
        instead of folding them into the far more complex, recursive algorithm 
        used for internal mesh structures. Periodic boundary conditions will 
        recursively check for immersed boundary crossings after each crossing
        of the domain boundary.
        
        Parameters
        ----------
        dt : float
            Length of current time step. Necessary for updating velocity and 
            acceleration as a result of an IB collision.
        ib_collisions : {None, 'sliding' (default), 'sticky'}
            Type of interaction with immersed boundaries. If None, turn off all 
            interaction with immersed boundaries. In sliding collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        '''

        ##### Immersed mesh boundaries go first #####
        if self.envir.ibmesh is not None and ib_collisions is not None:

            # if all agents are masked (exited the domain), skip all IB checks
            if np.all(self.positions.mask):
                return
            # otherwise, gather the indices of agents still in the domain
            if np.any(self.positions.mask):
                active = np.arange(self.N)[~ma.getmaskarray(self.positions[:,0])]
            else:
                active = np.arange(self.N)

            if len(active) > 0:
                # Precompute the shared mesh data ONCE for this time step (for a
                #   moving mesh this avoids redundant per-agent interpolation).
                shared = self._precompute_ib_shared(dt, ib_collisions)
                # Build one small (idx, startpt, endpt) argument per active agent.
                #   Cast to plain ndarrays so masked arrays are not handed to a
                #   worker pool; .copy() matches the previous per-agent semantics.
                prev_pos = self._prev_positions
                args = [(int(n),
                         np.asarray(prev_pos[n,:]).copy(),
                         np.asarray(self.positions[n,:]).copy())
                        for n in active]
                # Dispatch: a user-supplied pool (any object with .map)
                #   parallelizes this embarrassingly-parallel work; otherwise the
                #   builtin map runs it serially. Both call the same pure worker,
                #   so the results are identical.
                worker = _ibc.make_ib_worker(shared)
                if self.pool is None:
                    results = map(worker, args)
                else:
                    results = self.pool.map(worker, args)
                # Apply results to swarm state in the parent process, keyed on the
                #   returned idx so correctness does not depend on result order.
                for idx, result in results:
                    self._apply_ib_result(idx, dt, result)

        ##### Environment Boundary Conditions #####
        self._domain_BC_loop(dt, ib_collisions=ib_collisions)


    #######################################################################
    #####          HELPER ROUTINES FOR BOUNDARY INTERACTIONS          #####
    #######################################################################


    def _IBC_routine(self, idx, dt, startpt, endpt, ib_collisions='sliding'):
        '''Serial single-agent IB check (used by the domain-boundary recursion).

        Delegates to the same precompute/worker/apply helpers used by the
        (optionally parallelized) per-agent loop in apply_boundary_conditions, so
        this path stays numerically identical to that loop.

        Parameters
        ----------
        idx : int
            Agent index
        dt : float
            Length of time step
        startpt : tuple
            Agent starting point
        endpt : tuple
            Agent ending point
        ib_collisions : {None, 'sliding' (default), 'sticky'}
            Type of interaction with immersed boundaries. If None, turn off all 
            interaction with immersed boundaries. In sliding collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        '''

        shared = self._precompute_ib_shared(dt, ib_collisions)
        worker = _ibc.make_ib_worker(shared)
        _, result = worker((idx, startpt, endpt))
        self._apply_ib_result(idx, dt, result)


    def _precompute_ib_shared(self, dt, ib_collisions):
        '''Compute the immersed-boundary data shared by all agents this step.

        Returns a tuple consumed by _ibc.make_ib_worker. For a static mesh this
        is cheap; for a moving mesh it interpolates the mesh at the start and end
        of the step and computes the max inter-vertex distance and the max vertex
        displacement once for the whole swarm (previously recomputed per agent).

        Parameters
        ----------
        dt : float
            Length of time step
        ib_collisions : {'sliding', 'sticky'}
            Type of interaction with immersed boundaries.

        Returns
        -------
        tuple
            ('static', mesh, max_meshpt_dist, ib_collisions) or
            ('moving', start_mesh, end_mesh, max_meshpt_dist, max_mov,
            ib_collisions)
        '''
        if self.envir.ibmesh.ndim == 3:
            # static mesh
            return ('static', self.envir.ibmesh, self.envir.max_meshpt_dist,
                    ib_collisions)
        else:
            # moving mesh. first get necessary info about it
            start_mesh = self.envir.interpolate_temporal_mesh()
            end_mesh = self.envir.interpolate_temporal_mesh(time=self.envir.time+dt)
            # The maximum distance between meshpoints will change in time.
            #   Calculate them here and pass it along.
            DIM = start_mesh.shape[1]
            max_meshpt_dist_start = np.concatenate(tuple(
                np.linalg.norm(start_mesh[:,ii,:]-start_mesh[:,(ii+1)%DIM,:], axis=1)
                for ii  in range(DIM))).max()
            max_meshpt_dist_end = np.concatenate(tuple(
                np.linalg.norm(end_mesh[:,ii,:]-end_mesh[:,(ii+1)%DIM,:], axis=1)
                for ii  in range(DIM))).max()
            max_meshpt_dist = np.max((max_meshpt_dist_start, max_meshpt_dist_end))
            # Calculate the maximum distance a mesh vertex moved
            max_mov = np.concatenate(tuple(
                np.linalg.norm(end_mesh[:,ii,:]-start_mesh[:,ii,:], axis=1)
                for ii  in range(DIM))).max()
            return ('moving', start_mesh, end_mesh, max_meshpt_dist, max_mov,
                    ib_collisions)


    def _apply_ib_result(self, idx, dt, result):
        '''Apply one agent's IB collision result to swarm state.

        Parameters
        ----------
        idx : int
            Agent index
        dt : float
            Length of time step
        result : tuple (new_loc, dx, mesh_idx)
            Output of apply_internal_static_BC / apply_internal_moving_BC. dx is
            None if the agent did not collide with the mesh during this step.
        '''
        new_loc, dx, mesh_idx = result
        self.positions[idx] = new_loc
        if dx is not None:
            self.accelerations[idx] = (dx/dt - self.velocities[idx])/dt
            self.velocities[idx] = dx/dt
            self.ib_collision_idx[idx] = mesh_idx
        else:
            self.ib_collision_idx[idx] = -1



    def _domain_BC_loop(self, dt, ib_collisions, idx_array=None):
        '''Loop over domain boundaries enforcing boundary conditions. Only 
        agents in idx_array will be checked, or all unmasked agents if idx_array 
        is not given. dt is necessary for moving immersed boundaries.
        '''

        if idx_array is None:
            idx_array = np.arange(self.N)

        status_BC = np.zeros((len(idx_array),len(self.envir.L)))

        ##### Mark all domain exits! -1 for left, 1 for right #####
        # skip masked entries
        if np.all(self.positions[idx_array].mask):
            return
        for dim in range(len(self.envir.L)):
            leftrow = np.logical_and(self.positions[idx_array,dim] < 0,
                                    ~self.positions[idx_array,dim].mask)
            rightrow = np.logical_and(self.positions[idx_array,dim] > self.envir.L[dim],
                                    ~self.positions[idx_array,dim].mask)
            status_BC[leftrow,dim] = -1
            status_BC[rightrow,dim] = 1

        ##### In cases where there are multiple exits, find the first #####
        BC_mult_bool = np.sum(np.abs(status_BC), axis=1) > 1
        if np.any(BC_mult_bool):
            # mark these for recursion
            mult_idx = idx_array[BC_mult_bool]
            # figure out which exit crossing occurred first and treat that as the 
            #   only one. Use the velocity to parameterize the movement.
            s_array = np.zeros((len(BC_mult_bool),len(self.envir.L)))
            for dim in range(len(self.envir.L)):
                # get multiple crossing entries that have crossing in this dim
                dim_bool = np.logical_and(BC_mult_bool, status_BC[:,dim] != 0) #full length
                dim_idx = idx_array[dim_bool] #reduced length
                right_1 = 1*(status_BC[dim_bool,dim] == 1) #reduced length
                s_array[dim_bool,dim] = (self.positions[dim_idx,dim]-right_1*
                        self.envir.L[dim])/self.velocities[dim_idx,dim]
            # for each row in s_array, the dimension with the largest value is
            #   now the one crossed first.
            first_dim = np.argmax(s_array[BC_mult_bool,:], axis=1)
            # remove the other crossings from the status_BC array
            status_vals = status_BC[BC_mult_bool, first_dim]
            status_BC[BC_mult_bool,:] = 0
            status_BC[BC_mult_bool, first_dim] = status_vals
            BC_bool = np.sum(np.abs(status_BC), axis=1) != 0
            BC_bool_check = np.sum(np.abs(status_BC), axis=1) == 1
            assert np.all(BC_bool == BC_bool_check), "Some multi crossings left over...?"
        else:
            mult_idx = None
            BC_bool = np.sum(np.abs(status_BC), axis=1) == 1

        if not np.any(BC_bool):
            return

        ##### Now apply BC to the first/only boundary crossing #####
        for dim, bndry in enumerate(self.envir.bndry):
            # Check for 3D
            if dim == 2 and len(self.envir.L) == 2:
                # Ignore last bndry condition; 2D environment.
                break

            ### Left boundary ###
            left_bool = np.logical_and(BC_bool, status_BC[:,dim]<0)
            left_idx = idx_array[left_bool]
            if len(left_idx) > 0:
                if bndry[0] == 'zero':
                    # mask anything that exited on the left.
                    self.positions[left_idx, :] = ma.masked
                    self.velocities[left_idx, :] = ma.masked
                    self.accelerations[left_idx, :] = ma.masked
                    # no further BC checks are made: masked entries are skipped
                elif bndry[0] == 'noflux':
                    # agent slides along flat boundary. pos/vel/accel in dir of 
                    #   boundary will be zero.
                    # additional IB crossings are possible, so first find 
                    #   point of intersection with the boundary to enable this 
                    #   check.
                    if self.envir.ibmesh is not None and ib_collisions is not None:
                        s_array = (self.positions[left_idx,dim]-0)/ \
                            self.velocities[left_idx,dim]
                        startpts = self.positions[left_idx,:] - (np.tile(s_array, 
                                   (self.velocities.shape[1],1)).T* 
                                    self.velocities[left_idx,:])
                    # now update pos/vel/accel
                    self.positions[left_idx, dim] = 0
                    self.velocities[left_idx, dim] = 0
                    self.accelerations[left_idx, dim] = 0
                    # now check for IB crossings. However, due to potential 
                    #   complex interactions with the noflux boundary, enforce 
                    #   sticky ib collisions in all cases.
                    if self.envir.ibmesh is not None and ib_collisions is not None:
                        for n, idx in enumerate(left_idx):
                            startpt = startpts[n]
                            endpt = self.positions[idx,:].copy()
                            self._IBC_routine(idx, dt, startpt, endpt, 'sticky')
                    # further domain crossings remain possible
                elif bndry[0] == 'periodic':
                    # wrap everything exiting on the left to the right
                    self.positions[left_idx, dim] =\
                        np.mod(self.positions[left_idx, dim],self.envir.L[dim])
                    # check for IB crossings. first, get the point of re-entry
                    #   using the velocity.
                    if self.envir.ibmesh is not None and ib_collisions is not None:
                        s_array = (self.positions[left_idx,dim]-
                            self.envir.L[dim])/self.velocities[left_idx,dim]
                        for n, idx in enumerate(left_idx):
                            startpt = self.positions[idx,:] - \
                                s_array[n]*self.velocities[idx,:]
                            endpt = self.positions[idx,:].copy()
                            self._IBC_routine(idx, dt, startpt, endpt, ib_collisions)
                    # further domain crossings are possible. if this happens, 
                    #   velocity should be the same as original velocity b/c 
                    #   immersed boundaries do not intersect with domain bndry,
                    #   so agent has slid off with original velocity heading.
                else:
                    raise NameError

            ### Right boundary ###
            right_bool = np.logical_and(BC_bool, status_BC[:,dim]>0)
            right_idx = idx_array[right_bool]
            if len(right_idx) > 0:
                if bndry[1] == 'zero':
                    # mask everything exiting on the right
                    self.positions[right_idx, :] = ma.masked
                    self.velocities[right_idx, :] = ma.masked
                    self.accelerations[right_idx, :] = ma.masked
                    # no further BC checks are made: masked entries are skipped
                elif bndry[1] == 'noflux':
                    # agent slides along flat boundary. pos/vel/accel in dir of 
                    #   boundary is zero.
                    # additional IB crossings are possible, so first find 
                    #   point of intersection with the boundary to enable this 
                    #   check.
                    if self.envir.ibmesh is not None and ib_collisions is not None:
                        s_array = (self.positions[right_idx,dim]-
                                   self.envir.L[dim])/self.velocities[right_idx,dim]
                        startpts = self.positions[right_idx,:] - (np.tile(s_array, 
                                   (self.velocities.shape[1],1)).T* 
                                    self.velocities[right_idx,:])
                    # now update pos/vel/accel
                    self.positions[right_idx, dim] = self.envir.L[dim]
                    self.velocities[right_idx, dim] = 0
                    self.accelerations[right_idx, dim] = 0
                    # now check for IB crossings. However, due to potential 
                    #   complex interactions with the noflux boundary, enforce 
                    #   sticky ib collisions in all cases.
                    if self.envir.ibmesh is not None and ib_collisions is not None:
                        for n, idx in enumerate(right_idx):
                            startpt = startpts[n]
                            endpt = self.positions[idx,:].copy()
                            self._IBC_routine(idx, dt, startpt, endpt, 'sticky')
                    # further domain crossings remain possible
                elif bndry[1] == 'periodic':
                    # wrap everything exiting on the right to the left
                    self.positions[right_idx, dim] =\
                        np.mod(self.positions[right_idx, dim],self.envir.L[dim])
                    # check for IB crossings. first, get the point of re-entry
                    #   using the velocity.
                    if self.envir.ibmesh is not None and ib_collisions is not None:
                        s_array = (self.positions[right_idx,dim]-0)/ \
                            self.velocities[right_idx,dim]
                        for n, idx in enumerate(right_idx):
                            startpt = self.positions[idx,:] - \
                                s_array[n]*self.velocities[idx,:]
                            endpt = self.positions[idx,:].copy()
                            self._IBC_routine(idx, dt, startpt, endpt, ib_collisions)
                    # further domain crossings are possible. if this happens, 
                    #   velocity should be the same as original velocity b/c 
                    #   immersed boundaries do not intersect with domain bndry,
                    #   so agent has slid off with original velocity heading.
                else:
                    raise NameError

        ##### All BC applied to first exit. Conduct recursion if necessary #####
        if mult_idx is not None:
            self._domain_BC_loop(dt, ib_collisions, idx_array=mult_idx)


    #######################################################################
    #####                      PLOTTING METHODS                       #####
    #######################################################################


    def _calc_basic_stats(self, DIM3, t_indx=None):
        ''' Return basic stats about % agents remaining, fluid velocity, and
        agent velocity for plot printing.

        No fluid velocity field is pulled here. The fluid component means come
        from FluidData's per-dump mean cache, which is exact at any time and
        needs no data loaded, so a plot or movie costs nothing in fluid I/O even
        when the run is dynamically loading. Whole-grid reductions over the
        fluid (mean and max fluid speed) were removed for that reason: they are
        nonlinear, so they cannot be cached that way, and they summarize the
        whole domain including regions holding no agents. The spread of agent
        speeds took their place -- it says directly whether the population is
        moving coherently. Whole-field values are still available on demand from
        FluidData.fmin/fmax and Environment.get_mean_fluid_speed().

        Agent velocities are read from vel_history, which records the velocity
        each agent actually had. They are not recovered by differencing
        pos_history: velocities are set in move() from pre-boundary-condition
        positions, so the two quantities part company for any agent that
        collided with an immersed boundary or the domain edge, and across a
        periodic wrap the difference of positions is meaningless.

        At t_indx=0 this reports the initial velocities, which Swarm.__init__
        sets to the local fluid drift where a flow exists. An earlier version
        reported the zero vector there on the grounds that velocity is
        undefined before the first step; the recorded value is the truth and
        agrees with what the agents were actually doing.

        Parameters
        ----------
        DIM3 : bool
            indicates the dimension of the domain (True for 3D)
        t_indx : int, optional
            The time index for pos_history or None for current time

        Returns
        -------
        perc_left : float
            percentage of agents left within the domain
        avg_spd_x : float
            average x-component of fluid velocity
        avg_spd_y : float
            average y-component of fluid velocity
        avg_spd_z : float, 3D only
            average z-component of fluid velocity
        avg_swrm_vel : array
            average agent velocity
        avg_swrm_spd : float
            average agent speed
        std_swrm_spd : float
            standard deviation of the agent speeds
        '''

        # get % of agents left in domain
        if t_indx is None:
            num_left = self.positions[:,0].compressed().size
        else:
            num_left = self.pos_history[t_indx][:,0].compressed().size
        if len(self.pos_history) > 0:
            num_orig = self.pos_history[0][:,0].compressed().size
        else:
            num_orig = num_left
        perc_left = 100*num_left/num_orig

        # get agent velocities. these are READ from the recorded history, not
        #   finite-differenced from positions: move() sets velocities from
        #   pre-boundary-condition positions and then apply_boundary_conditions
        #   mutates positions, so the two differ for any agent that collided
        #   with an immersed boundary or the domain edge. On a periodic
        #   dimension a wrap makes the difference of positions a spurious
        #   near-domain-width velocity.
        if t_indx is None:
            vel_data = self.velocities
        else:
            vel_data = self.full_vel_history[t_indx]

        # only agents still in the domain contribute. agents leave whole rows at
        # a time, so dropping masked rows loses nothing from the survivors.
        mask = np.ma.getmaskarray(vel_data)
        vel_data = np.asarray(np.ma.getdata(vel_data))[~mask.any(axis=1)]
        if vel_data.size == 0:
            # every agent has left; report zeros rather than masked values,
            # which the plot text cannot format.
            vel_data = np.zeros((1,len(self.envir.L)))

        avg_swrm_vel = vel_data.mean(axis=0)
        swrm_spd = np.linalg.norm(vel_data, axis=1)
        avg_swrm_spd = swrm_spd.mean()
        std_swrm_spd = swrm_spd.std()

        if self.envir.flow is None and not DIM3:
            return perc_left, 0, 0, avg_swrm_vel, avg_swrm_spd, std_swrm_spd
        elif self.envir.flow is None and DIM3:
            return perc_left, 0, 0, 0, avg_swrm_vel, avg_swrm_spd, std_swrm_spd

        # get the fluid component means from the cache
        if self.envir.flow.flow_times is None:
            # temporally constant flow
            fluid_means = self.envir.flow.get_mean_velocity()
        else:
            if t_indx is None:
                time = self.envir.time
            else:
                time = self.envir.time_history[t_indx]
            fluid_means = self.envir.flow.get_mean_velocity(time=time)

        if not DIM3:
            return (perc_left, fluid_means[0], fluid_means[1], avg_swrm_vel,
                    avg_swrm_spd, std_swrm_spd)
        else:
            return (perc_left, fluid_means[0], fluid_means[1], fluid_means[2],
                    avg_swrm_vel, avg_swrm_spd, std_swrm_spd)



    def plot(self, t=None, filename=None, blocking=True, dist='density', 
             fluid=None, clip=None, figsize=None, circ_rad=0.25, plot_heading=True,
             save_kwargs=None, azim=None, elev=None):
        '''Plot the position of the swarm at time t, or at the current time
        if no time is supplied. The actual time plotted will depend on the
        history of movement steps; the closest entry in
        Environment.time_history will be shown without interpolation.
        
        Parameters
        ----------
        t : float, optional
            time to plot. if None (default), the current time.
        filename : str, optional
            file name to save image as. Image will not be shown, only saved.
        blocking : bool, default True
            whether the plot should block execution or not
        dist : {'density' (default), 'cov', float, 'hist'}
            whether to plot Gaussian kernel density estimation or histogram.
            Options are:

            * 'density': plot Gaussian KDE using Scotts Factor from scipy.stats.gaussian_kde
            * 'cov': use the variance in each direction from self.shared_props['cov']
              to plot Gaussian KDE
            * float: plot Gaussian KDE using the given bandwidth factor to 
              multiply the KDE variance by
            * 'hist': plot histogram
        fluid : {'vort', 'quiver'}, optional
            Plot info on the fluid in the background. 2D only! If None, don't
            plot anything related to the fluid.
            Options are:

            * 'vort': plot vorticity in the background
            * 'quiver': quiver plot of fluid velocity in the background
        clip : float, optional
            if plotting vorticity, specifies the clip value for pseudocolor.
            this value is used for both negative and positive vorticity.
        figsize : tuple of length 2, optional
            figure size in inches, (width, height). default is a heurstic that 
            works... most of the time?
        circ_rad : float, default=0.25
            plotting size of the agent circles (in 2D only)
        plot_heading : bool, default=True
            whether or not to plot the direction (heading) of each agent as a 
            small line.
        save_kwargs : dict of keyword arguments, optional
            keys must be valid strings that match keyword arguments for the 
            matplotlib savefig function. These arguments will be passed to 
            savefig assuming that a filename has been specified.
        azim : float, optional
            In 3D plots, the azimuthal viewing angle. Defaults to -60.
        elev : float, optional
            In 3D plots, the elevation viewing angle. Defaults to 30.
        '''

        if t is not None and len(self.envir.time_history) != 0:
            loc = np.searchsorted(self.envir.time_history, t)
            if loc == len(self.envir.time_history):
                if (t-self.envir.time_history[-1]) > (self.envir.time-t):
                    loc = None
                else:
                    loc = -1
            elif t < self.envir.time_history[loc]:
                if (self.envir.time_history[loc]-t) > (t-self.envir.time_history[loc-1]):
                    loc -= 1
        else:
            loc = None

        # get time and positions
        if loc is None:
            time = self.envir.time
            positions = self.positions
        else:
            time = self.envir.time_history[loc]
            positions = self.pos_history[loc]

        if len(self.envir.L) == 2:
            # 2D plot
            if figsize is None:
                aspectratio = self.envir.L[0]/self.envir.L[1]
                if aspectratio > 1:
                    x_length = np.min((6*aspectratio,12))
                    y_length = 6
                elif aspectratio < 1:
                    x_length = 6
                    y_length = np.min((6/aspectratio,8))
                else:
                    x_length = 6
                    y_length = 6
                fig = plt.figure(figsize=(x_length,y_length))
            else:
                fig = plt.figure(figsize=figsize)
            ax, mesh_col, axHistx, axHisty = self.envir._plot_setup(fig)
            if figsize is None:
                # some final adjustments in a particular case
                if x_length == 12:
                    ax_pos = ax.get_position().get_points()
                    axHx_pos = np.array(axHistx.get_position().get_points())
                    axHy_pos = np.array(axHisty.get_position().get_points())
                    if ax_pos[0,1] > 0.1:
                        extra = 2*(ax_pos[0,1] - 0.1)*y_length
                        fig.set_size_inches(x_length,y_length-extra)
                        prop = (y_length-extra/4)/y_length
                        prop_wdth = (y_length-extra/2)/y_length
                        prop_len = (y_length-extra)/y_length
                        axHistx.set_position([axHx_pos[0,0],axHx_pos[0,1]*prop,
                                              axHx_pos[1,0]-axHx_pos[0,0],
                                              (axHx_pos[1,1]-axHx_pos[0,1])/prop_wdth])
                        axHisty.set_position([axHy_pos[0,0],axHy_pos[0,1]*prop_len,
                                              axHy_pos[1,0]-axHy_pos[0,0],
                                              (axHy_pos[1,1]-axHy_pos[0,1])/prop_len])

            # fluid visualization
            if fluid == 'vort' and self.envir.flow is not None:
                vort = self.envir.get_vorticity(t_indx=loc)
                norm = _vorticity_norm(vort, clip)
                ax.pcolormesh(self.envir.flow.flow_points[0], self.envir.flow.flow_points[1],
                              vort.T, shading='gouraud', cmap='RdBu',
                              norm=norm, alpha=0.9, antialiased=True)
            elif fluid == 'quiver' and self.envir.flow is not None:
                # get dimensions of axis to estimate a decent quiver density
                ax_pos = ax.get_position().get_points()
                fig_size = fig.get_size_inches()
                wdth_inch = fig_size[0]*(ax_pos[1,0]-ax_pos[0,0])
                height_inch = fig_size[1]*(ax_pos[1,1]-ax_pos[0,1])
                # use about 4.15/inch density of arrows
                x_num = round(4.15*wdth_inch)
                y_num = round(4.15*height_inch)
                M = int(round(len(self.envir.flow.flow_points[0])/x_num))
                N = int(round(len(self.envir.flow.flow_points[1])/y_num))
                # get worse case max velocity vector for scaling
                max_u, max_v = self.envir.flow.fmax
                max_mag = np.linalg.norm(np.array([max_u,max_v]))
                if self.envir.flow.flow_times is not None:
                    flow = self.envir.interpolate_temporal_flow(t_index=loc)
                else:
                    flow = self.envir.flow
                ax.quiver(self.envir.flow.flow_points[0][::M], self.envir.flow.flow_points[1][::N],
                          flow[0][::M,::N].T, flow[1][::M,::N].T, 
                          scale=max_mag*5, alpha=0.2)

            # ibmesh (if moving and not a current time - otherwise, done already)
            if mesh_col is not None and self.envir.ibmesh.ndim == 4 and t is not None:
                ibmesh = self.interpolate_temporal_mesh(time=t)
                mesh_col.set_segments(ibmesh)

            # Create marker headings to add to scatter
            paths = []
            circle = mPath.circle(radius=circ_rad)
            if plot_heading:
                line_codes = np.array([mPath.MOVETO, mPath.LINETO])
                codes = np.concatenate([circle.codes, line_codes])
                if 'angle' in self.props:
                    angles = self.props['angle']
                else:
                    # this is defined even for (0,0) by convention
                    angles = np.arctan2(self.velocities[:,1], self.velocities[:,0])
                for angle in angles:
                    if ma.is_masked(angle):
                        paths.append(circle)
                    else:
                        # make the heading marker stick out by one diameter
                        line_verts = np.array([[0,0],[circ_rad*3*np.cos(angle),
                                                    circ_rad*3*np.sin(angle)]])
                        # combine the circle and line vertices
                        verts = np.concatenate([circle.vertices, line_verts])
                        # append to path list
                        paths.append(mPath(verts, codes))
            else:
                paths.append(circle)

            # scatter plot
            if 'color' in self.props:
                if self.props_history is not None and loc is not None:
                    # Get color from history
                    color = self.props_history[loc]['color']
                else:
                    color = self.props['color']
                sc = ax.scatter(positions[:,0], positions[:,1], 
                           label=self.shared_props['name'], c=color)
            else:
                sc = ax.scatter(positions[:,0], positions[:,1], 
                           label=self.shared_props['name'], 
                           color=self.shared_props['color'])
            sc.set_paths(paths)

            # time text
            ax.text(0.02, 0.95, 'time = {:.2f}'.format(time),
                    transform=ax.transAxes, fontsize=12)

            # textual info
            perc_left, avg_spd_x, avg_spd_y, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                self._calc_basic_stats(DIM3=False, t_indx=loc)
            plt.figtext(0.77, 0.77,
                        '{:.1f}% remain\n'.format(perc_left)+
                        '\n  ------ Info ------\n'+
                        r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units)+
                        r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s\n'.format(avg_swrm_spd, std_swrm_spd, self.envir.units),
                        fontsize=10)
            axHistx.text(0.01, 0.98, r'Fluid $\overline{v}_x$'+': {:.2g} \n'.format(avg_spd_x)+
                         r'Agent $\overline{v}_x$'+': {:.2g}'.format(avg_swrm_vel[0]),
                         transform=axHistx.transAxes, verticalalignment='top',
                         fontsize=10)
            axHisty.text(0.02, 0.99, r'Fluid $\overline{v}_y$'+': {:.2g} \n'.format(avg_spd_y)+
                         r'Agent $\overline{v}_y$'+': {:.2g}'.format(avg_swrm_vel[1]),
                         transform=axHisty.transAxes, verticalalignment='top',
                         fontsize=10)

            if dist == 'hist':
                # histograms
                bins_x = np.linspace(0, self.envir.L[0], 26)
                bins_y = np.linspace(0, self.envir.L[1], 26)
                axHistx.hist(positions[:,0].compressed(), bins=bins_x)
                axHisty.hist(positions[:,1].compressed(), bins=bins_y,
                                orientation='horizontal')
            else:
                # Gaussian Kernel Density Estimation
                if dist == 'cov':
                    fac_x = self.shared_props['cov'][0,0]
                    fac_y = self.shared_props['cov'][1,1]
                else:
                    try:
                        fac_x = float(dist)
                        fac_y = fac_x
                    except:
                        fac_x = None
                        fac_y = None
                xmesh = np.linspace(0, self.envir.L[0], 1000)
                ymesh = np.linspace(0, self.envir.L[1], 1000)
                # deal with point sources
                pos_x = positions[:,0].compressed()
                pos_y = positions[:,1].compressed()
                try:
                    if len(pos_x) > 1:
                        x_density = stats.gaussian_kde(pos_x, fac_x)
                        x_density = x_density(xmesh)
                    elif len(pos_x) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        x_density = np.zeros_like(xmesh)
                except np.linalg.LinAlgError:
                    idx = (np.abs(xmesh - pos_x[0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                try:
                    if len(pos_y) > 1:
                        y_density = stats.gaussian_kde(pos_y, fac_y)
                        y_density = y_density(ymesh)
                    elif len(pos_y) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        y_density = np.zeros_like(ymesh)
                except np.linalg.LinAlgError:
                    idy = (np.abs(ymesh - pos_y[0])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idy] = 1
                axHistx.plot(xmesh, x_density)
                axHisty.plot(y_density, ymesh)
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_xaxis().set_ticks([])
                if np.max(x_density) != 0:
                    axHistx.set_ylim(bottom=0, top=np.max(x_density))
                else:
                    axHistx.set_ylim(bottom=0)
                if np.max(y_density) != 0:
                    axHisty.set_xlim(left=0, right=np.max(y_density))
                else:
                    axHisty.set_xlim(left=0)

        else:
            # 3D plot
            if figsize is None:
                fig = plt.figure(figsize=(10,5))
            else:
                fig = plt.figure(figsize=figsize)
            ax, mesh_col, axHistx, axHisty, axHistz = self.envir._plot_setup(fig)
            if azim is not None or elev is not None:
                ax.view_init(elev, azim)

            # scatter plot and time text
            if 'color' in self.props:
                if self.props_history is not None and loc is not None:
                    # Get color from history
                    color = self.props_history[loc]['color']
                else:
                    color = self.props['color']
                ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                           label=self.shared_props['name'], c=color)
            else:
                ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                           label=self.shared_props['name'], 
                           color=self.shared_props['color'])
            ax.text2D(0.02, 1, 'time = {:.2f}'.format(time),
                      transform=ax.transAxes, verticalalignment='top',
                      fontsize=12)

            # textual info
            perc_left, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                self._calc_basic_stats(DIM3=True, t_indx=loc)
            # anchored a little further left than the old fluid stats box: the
            # "mean +/- spread" line is wider than the lines it replaced.
            ax.text2D(0.65, 0.9, r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units)+
                      r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s'.format(avg_swrm_spd, std_swrm_spd, self.envir.units),
                      transform=ax.transAxes, horizontalalignment='left',
                      fontsize=10)
            ax.text2D(0.02, 0, '{:.1f}% remain\n'.format(perc_left),
                      transform=fig.transFigure, fontsize=10)
            axHistx.text(0.02, 0.98, r'Fluid $\overline{v}_x$'+': {:.2g} {}/s\n'.format(avg_spd_x,
                         self.envir.units)+
                         r'Agent $\overline{v}_x$'+': {:.2g} {}/s'.format(avg_swrm_vel[0],
                         self.envir.units),
                         transform=axHistx.transAxes, verticalalignment='top',
                         fontsize=10)
            axHisty.text(0.02, 0.98, r'Fluid $\overline{v}_y$'+': {:.2g} {}/s\n'.format(avg_spd_y,
                         self.envir.units)+
                         r'Agent $\overline{v}_y$'+': {:.2g} {}/s'.format(avg_swrm_vel[1],
                         self.envir.units),
                         transform=axHisty.transAxes, verticalalignment='top',
                         fontsize=10)
            axHistz.text(0.02, 0.98, r'Fluid $\overline{v}_z$'+': {:.2g} {}/s\n'.format(avg_spd_z,
                         self.envir.units)+
                         r'Agent $\overline{v}_z$'+': {:.2g} {}/s'.format(avg_swrm_vel[2],
                         self.envir.units),
                         transform=axHistz.transAxes, verticalalignment='top',
                         fontsize=10)

            if dist == 'hist':
                # histograms
                bins_x = np.linspace(0, self.envir.L[0], 26)
                bins_y = np.linspace(0, self.envir.L[1], 26)
                bins_z = np.linspace(0, self.envir.L[2], 26)
                axHistx.hist(positions[:,0].compressed(), bins=bins_x, alpha=0.8)
                axHisty.hist(positions[:,1].compressed(), bins=bins_y, alpha=0.8)
                axHistz.hist(positions[:,2].compressed(), bins=bins_z, alpha=0.8)
            else:
                # Gaussian Kernel Density Estimation
                if dist == 'cov':
                    fac_x = self.shared_props['cov'][0,0]
                    fac_y = self.shared_props['cov'][1,1]
                    fac_z = self.shared_props['cov'][2,2]
                else:
                    try:
                        fac_x = float(dist)
                        fac_y = fac_x
                        fac_z = fac_x
                    except:
                        fac_x = None
                        fac_y = None
                        fac_z = None
                xmesh = np.linspace(0, self.envir.L[0], 1000)
                ymesh = np.linspace(0, self.envir.L[1], 1000)
                zmesh = np.linspace(0, self.envir.L[2], 1000)
                # deal with point sources
                pos_x = positions[:,0].compressed()
                pos_y = positions[:,1].compressed()
                pos_z = positions[:,2].compressed()
                try:
                    if len(pos_x) > 1:
                        x_density = stats.gaussian_kde(pos_x, fac_x)
                        x_density = x_density(xmesh)
                    elif len(pos_x) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        x_density = np.zeros_like(xmesh)
                except np.linalg.LinAlgError:
                    idx = (np.abs(xmesh - pos_x[0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                try:
                    if len(pos_y) > 1:
                        y_density = stats.gaussian_kde(pos_y, fac_y)
                        y_density = y_density(ymesh)
                    elif len(pos_y) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        y_density = np.zeros_like(ymesh)
                except np.linalg.LinAlgError:
                    idy = (np.abs(ymesh - pos_y[0])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idy] = 1
                try:
                    if len(pos_z) > 1:
                        z_density = stats.gaussian_kde(pos_z, fac_z)
                        z_density = z_density(zmesh)
                    elif len(pos_z) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        z_density = np.zeros_like(zmesh)
                except np.linalg.LinAlgError:
                    idz = (np.abs(zmesh - pos_z[0])).argmin()
                    z_density = np.zeros_like(zmesh); z_density[idz] = 1
                axHistx.plot(xmesh, x_density)
                axHisty.plot(ymesh, y_density)
                axHistz.plot(zmesh, z_density)
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_yaxis().set_ticks([])
                axHistz.get_yaxis().set_ticks([])
                if np.max(x_density) != 0:
                    axHistx.set_ylim(bottom=0, top=np.max(x_density))
                else:
                    axHistx.set_ylim(bottom=0)
                if np.max(y_density) != 0:
                    axHisty.set_ylim(bottom=0, top=np.max(y_density))
                else:
                    axHisty.set_ylim(bottom=0)
                if np.max(z_density) != 0:
                    axHistz.set_ylim(bottom=0, top=np.max(z_density))
                else:
                    axHistz.set_ylim(bottom=0)

        # show the plot
        if filename is None:
            plt.show(block=blocking)
        else:
            if save_kwargs is not None:
                plt.savefig(filename, **save_kwargs)
            else:
                plt.savefig(filename)



    def _select_frames(self, fps, playback_rate):
        '''Choose which recorded states of the simulation become frames.

        Frames are laid down at a fixed interval of *simulated* time,
        dt_frame = playback_rate/fps, and each one shows the recorded state
        nearest to it. The first and last states are always included, so the
        animation spans the whole run. Frames cannot be produced in between
        recorded states, so a dt_frame finer than the interval between them is
        clamped to one frame per state, with a warning.

        Parameters
        ----------
        fps : int
            frames per second of the animation.
        playback_rate : float
            simulated seconds per second of animation.

        Returns
        -------
        ndarray of int
            frame indices into the position history, where the final index
            (equal to its length) means the present state.
        '''

        if fps <= 0 or playback_rate <= 0:
            raise ValueError('fps and playback_rate must both be positive.')

        # states available to draw: the position history, then the present.
        # frame index n means pos_history[n] at time_history[n], with index
        # len(pos_history) meaning the present positions at envir.time.
        n_hist = len(self.pos_history)
        if self.envir.time is None:
            # A time step raised partway through, so the present positions hold
            # a step applied to only some agents (see move()). The histories are
            # still a consistent record, so plot those and leave the incomplete
            # step out: there is no "present" frame to draw.
            warnings.warn("Environment.time is None: a time step failed partway "
                          "through, so the current agent positions are "
                          "incomplete and are left out. Plotting the recorded "
                          "history only.", stacklevel=3)
            times = np.asarray(self.envir.time_history[:n_hist], dtype=float)
        elif len(self.envir.time_history) < n_hist:
            # histories out of step (e.g. move(update_time=False) without a
            # matching environmental time update). there are no reliable times
            # to select against, so fall back on a frame per recorded state.
            return np.arange(n_hist+1)
        else:
            times = np.concatenate((self.envir.time_history[:n_hist],
                                    (self.envir.time,)))
        if len(times) < 2:
            return np.arange(len(times))

        dt_frame = playback_rate/fps
        span = times[-1] - times[0]
        # mean interval between recorded states; dt itself for a fixed timestep
        dt_state = span/(len(times)-1)

        # frame times, then the recorded state nearest to each of them
        frame_times = times[0] + np.arange(int(span/dt_frame)+1)*dt_frame
        hi = np.clip(np.searchsorted(times, frame_times), 1, len(times)-1)
        frames = np.where(frame_times-times[hi-1] <= times[hi]-frame_times,
                          hi-1, hi)

        unique_frames = np.unique(frames)
        # the tolerance keeps a deliberately exact choice (playback_rate/fps
        # meant to equal dt) from tripping the clamp on floating point roundoff
        if dt_frame < dt_state*(1-1e-9) or len(unique_frames) < len(frames):
            warnings.warn("Cannot draw a frame every {:.3g} s of simulated ".format(dt_frame)+
                "time: states were only recorded every {:.3g} s. ".format(dt_state)+
                "Using one frame per recorded state, which plays at {:.3g} ".format(dt_state*fps)+
                "rather than {:.3g} simulated s per s of video; ".format(playback_rate)+
                "an fps of {:.3g} or less gives the rate you asked for.".format(playback_rate/dt_state),
                stacklevel=3)
            frames = unique_frames
        elif np.max(np.abs(np.diff(times[frames])-dt_frame), initial=0) > dt_frame/6:
            # frames land on recorded states, so their spacing jitters by up to
            # one recording interval whenever dt_frame is not a whole multiple
            # of it. the video is encoded at a constant fps regardless, so this
            # shows up as uneven motion.
            warnings.warn("A frame every {:.3g} s of simulated time is only ".format(dt_frame)+
                "{:.3g}x the {:.3g} s between recorded states, and not a ".format(dt_frame/dt_state, dt_state)+
                "whole multiple of it, so motion will look slightly uneven.",
                stacklevel=3)

        # always finish on the final recorded state
        if frames[-1] != len(times)-1:
            frames = np.append(frames, len(times)-1)

        return frames



    def plot_all(self, movie_filename=None, frames=None, downsamp=None, fps=10,
                 playback_rate=1,
                 dist='density', fluid=None, clip=None, figsize=None, circ_rad=0.25,
                 plot_heading=True, save_kwargs=None, writer_kwargs=None, 
                 azim=None, elev=None):
        ''' Plot the history of the swarm's movement, incl. current time in 
        successively updating plots or saved as a movie file. A movie file is
        created if movie_filename is specified.

        Agent colors will be read from the 'color' column of props if it exists; 
        otherwise it will default to the color attribute of the Swarm.
        
        Parameters
        ----------
        movie_filename : string, optional
            file name to save movie as. file extension will determine the type
            of file saved.
        frames : iterable of integers, optional.
            If None, frames are chosen from fps and playback_rate (below). If
            an iterable, plot only the time steps of the swarm as indexed by
            the iterable, overriding playback_rate (note, this is an iterable
            of the time step indices, not the time in seconds at those time
            steps!).
        downsamp : iterable of int or int, optional
            If None, do not downsample the agents - plot them all. If an integer,
            plot only the first n agents (equivalent to range(downsamp)).
            If an iterable, plot only the agents specified. In all cases,
            statistics are reported for the TOTAL population, both shown and
            unshown. This includes the histograms/KDE plots.
        fps : int, default=10
            Frames per second of the animation: how *smooth* it is. Standard
            video rates are 24-30 fps.
        playback_rate : float, default=1
            Seconds of simulated time per second of animation: how *fast* it
            plays. 1 is real time (assuming simulated time is in seconds), 0.5
            is half-speed slow motion, 10 is ten times fast forward. Together 
            with fps this fixes the simulated time between frames,
            dt_frame = playback_rate/fps, and each frame shows the recorded
            state nearest to it in time. dt_frame must be at least the
            timestep dt.
        dist : {'density' (default), 'cov', float, 'hist'}
            whether to plot Gaussian kernel density estimation or histogram.
            Options are:

            * 'density': plot Gaussian KDE using Scotts Factor from scipy.stats.gaussian_kde
            * 'cov': use the variance in each direction from self.shared_props['cov']
              to plot Gaussian KDE
            * float: plot Gaussian KDE using the given bandwidth factor to 
              multiply the KDE variance by
            * 'hist': plot histogram
        fluid : {'vort', 'quiver'}, optional
            Plot info on the fluid in the background. 2D only! If None, don't
            plot anything related to the fluid.
            Options are:

            * 'vort': plot vorticity in the background
            * 'quiver': quiver plot of fluid velocity in the background
        clip : float, optional
            if plotting vorticity, specifies the clip value for pseudocolor.
            this value is used for both negative and positive vorticity.
        figsize : tuple of length 2, optional
            figure size in inches, (width, height). default is a heurstic that 
            works... most of the time?
        circ_rad : float, default=0.25
            plotting size of the agent circles (in 2D only)
        plot_heading : bool, default=True
            whether or not to plot the direction (heading) of each agent as a 
            small line.
        save_kwargs : dict of keyword arguments, optional
            keys must be valid strings that match keyword arguments for the 
            matplotlib animation.FFMpegWriter object. These arguments will be 
            used in the writer object initiation save assuming that a 
            movie_filename has been specified. Otherwise, defaults are the 
            passed in fps and metadata=dict(artist='Christopher Strickland')).
        writer_kwargs : dict of keyword arguments, optional
            keys must be valid strings that match keyword arguments for a  
            matplotlib 
        azim : float, optional
            In 3D plots, the azimuthal viewing angle. Defaults to -60.
        elev : float, optional
            In 3D plots, the elevation viewing angle. Defaults to 30.
        '''

        if len(self.envir.time_history) == 0:
            print('No position history! Plotting current position...')
            self.plot()
            return

        if movie_filename is not None:
            print("Creating video... this could take a long time!")
        
        DIM3 = (len(self.envir.L) == 3)

        if frames is None:
            frames = self._select_frames(fps, playback_rate)
        n0 = frames[0]

        if isinstance(downsamp, int):
            downsamp = range(downsamp)

        if not DIM3:
            ### 2D setup ###
            if figsize is None:
                aspectratio = self.envir.L[0]/self.envir.L[1]
                if aspectratio > 1:
                    x_length = np.min((6*aspectratio,12))
                    y_length = 6
                elif aspectratio < 1:
                    x_length = 6
                    y_length = np.min((6/aspectratio,8))
                else:
                    x_length = 6
                    y_length = 6
                fig = plt.figure(figsize=(x_length,y_length))
            else:
                fig = plt.figure(figsize=figsize)
            ax, mesh_col, axHistx, axHisty = self.envir._plot_setup(fig)
            if figsize is None:
                # some final adjustments in a particular case
                if x_length == 12:
                    ax_pos = ax.get_position().get_points()
                    axHx_pos = np.array(axHistx.get_position().get_points())
                    axHy_pos = np.array(axHisty.get_position().get_points())
                    if ax_pos[0,1] > 0.1:
                        extra = 2*(ax_pos[0,1] - 0.1)*y_length
                        fig.set_size_inches(x_length,y_length-extra)
                        prop = (y_length-extra/4)/y_length
                        prop_wdth = (y_length-extra/2)/y_length
                        prop_len = (y_length-extra)/y_length
                        axHistx.set_position([axHx_pos[0,0],axHx_pos[0,1]*prop,
                                              axHx_pos[1,0]-axHx_pos[0,0],
                                              (axHx_pos[1,1]-axHx_pos[0,1])/prop_wdth])
                        axHisty.set_position([axHy_pos[0,0],axHy_pos[0,1]*prop_len,
                                              axHy_pos[1,0]-axHy_pos[0,0],
                                              (axHy_pos[1,1]-axHy_pos[0,1])/prop_len])

            # fluid visualization
            if fluid == 'vort' and self.envir.flow is not None:
                # Limits start symmetric and are grown by each frame; see
                # _vorticity_norm. Without a clip the placeholder is (-1, 1),
                # which the first frame drawn replaces.
                norm = _vorticity_norm(np.zeros(1), clip)
                fld = ax.pcolormesh(self.envir.flow.flow_points[0], self.envir.flow.flow_points[1],
                           np.zeros(self.envir.flow.fshape[1:]).T, shading='gouraud',
                           cmap='RdBu', norm=norm, alpha=0.9)
            elif fluid == 'quiver' and self.envir.flow is not None:
                # get dimensions of axis to estimate a decent quiver density
                ax_pos = ax.get_position().get_points()
                fig_size = fig.get_size_inches()
                wdth_inch = fig_size[0]*(ax_pos[1,0]-ax_pos[0,0])
                height_inch = fig_size[1]*(ax_pos[1,1]-ax_pos[0,1])
                # use about 4.15/inch density of arrows
                x_num = round(4.15*wdth_inch)
                y_num = round(4.15*height_inch)
                M = round(len(self.envir.flow.flow_points[0])/x_num)
                N = round(len(self.envir.flow.flow_points[1])/y_num)
                # get worse case max velocity vector for scaling
                max_u, max_v = self.envir.flow.fmax
                max_mag = np.linalg.norm(np.array([max_u,max_v]))
                x_pts = self.envir.flow.flow_points[0][::M]
                y_pts = self.envir.flow.flow_points[1][::N]
                fld = ax.quiver(x_pts, y_pts, np.zeros((len(y_pts),len(x_pts))),
                                np.zeros((len(y_pts),len(x_pts))), 
                                scale=max_mag*5, alpha=0.2)

            # scatter plot
            scat = ax.scatter([], [], label=self.shared_props['name'], 
                              c=self.shared_props['color'])
            
            # set up marker headings to be added to the scatter plots
            circle = mPath.circle(radius=circ_rad)
            line_codes = np.array([mPath.MOVETO, mPath.LINETO])
            codes = np.concatenate([circle.codes, line_codes])

            # textual info
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                fontsize=12)
            perc_left, avg_spd_x, avg_spd_y, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                self._calc_basic_stats(DIM3=False, t_indx=n0)
            axStats = plt.axes([0.77, 0.77, 0.25, 0.2], frameon=False)
            axStats.set_axis_off()
            stats_text = axStats.text(0,1,
                         '{:.1f}% remain\n'.format(perc_left)+
                         '\n  ------ Info ------\n'+
                         r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units)+
                         r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s\n'.format(avg_swrm_spd, std_swrm_spd, self.envir.units),
                         fontsize=10, transform=axStats.transAxes,
                         verticalalignment='top')
            x_text = axHistx.text(0.01, 0.98, r'Fluid $\overline{v}_x$'+': {:.2g} \n'.format(avg_spd_x)+
                         r'Agent $\overline{v}_x$'+': {:.2g}'.format(avg_swrm_vel[0]),
                         transform=axHistx.transAxes, verticalalignment='top',
                         fontsize=10)
            y_text = axHisty.text(0.02, 0.99, r'Fluid $\overline{v}_y$'+': {:.2g} \n'.format(avg_spd_y)+
                         r'Agent $\overline{v}_y$'+': {:.2g}'.format(avg_swrm_vel[1]),
                         transform=axHisty.transAxes, verticalalignment='top',
                         fontsize=10)

            if dist == 'hist':
                # histograms
                data_x = self.pos_history[n0][:,0].compressed()
                data_y = self.pos_history[n0][:,1].compressed()
                bins_x = np.linspace(0, self.envir.L[0], 26)
                bins_y = np.linspace(0, self.envir.L[1], 26)
                n_x, bins_x, patches_x = axHistx.hist(data_x, bins=bins_x)
                n_y, bins_y, patches_y = axHisty.hist(data_y, bins=bins_y, 
                                                      orientation='horizontal')
            else:
                # Gaussian Kernel Density Estimation
                if dist == 'cov':
                        fac_x = self.shared_props['cov'][0,0]
                        fac_y = self.shared_props['cov'][1,1]
                else:
                    try:
                        fac_x = float(dist)
                        fac_y = fac_x
                    except:
                        # estimate covariance from Scotts Factor. HOWEVER: this 
                        #   estimation breaks if IC is point source.
                        fac_x = None
                        fac_y = None
                xmesh = np.linspace(0, self.envir.L[0], 1000)
                ymesh = np.linspace(0, self.envir.L[1], 1000)
                # deal with point sources
                pos_x = self.pos_history[n0][:,0].compressed()
                pos_y = self.pos_history[n0][:,1].compressed()
                try:
                    if len(pos_x) > 1:
                        x_density = stats.gaussian_kde(pos_x, fac_x)
                        x_density = x_density(xmesh)
                    elif len(pos_x) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        x_density = np.zeros_like(xmesh)
                except np.linalg.LinAlgError:
                    idx = (np.abs(xmesh - pos_x[0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                try:
                    if len(pos_y) > 1:
                        y_density = stats.gaussian_kde(pos_y, fac_y)
                        y_density = y_density(ymesh)
                    elif len(pos_y) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        y_density = np.zeros_like(ymesh)
                except np.linalg.LinAlgError:
                    idy = (np.abs(ymesh - pos_y[0])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idy] = 1
                xdens_plt, = axHistx.plot(xmesh, x_density)
                ydens_plt, = axHisty.plot(y_density, ymesh)
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_xaxis().set_ticks([])
                if np.max(xdens_plt.get_ydata()) != 0:
                    axHistx.set_ylim(bottom=0, top=np.max(xdens_plt.get_ydata()))
                else:
                    axHistx.set_ylim(bottom=0)
                if np.max(ydens_plt.get_xdata()) != 0:
                    axHisty.set_xlim(left=0, right=np.max(ydens_plt.get_xdata()))
                else:
                    axHisty.set_xlim(left=0)
            
        else:
            ### 3D setup ###
            if figsize is None:
                fig = plt.figure(figsize=(10,5))
            else:
                fig = plt.figure(figsize=figsize)
            ax, mesh_col, axHistx, axHisty, axHistz = self.envir._plot_setup(fig)
            if azim is not None or elev is not None:
                ax.view_init(elev, azim)
            # UNFORTUNATELY, 3D matplotlib plotting is very weird about masked 
            #   arrays. The implementation does not parallel 2D: it wants a color 
            #   list that is the same length as the number of points it will be 
            #   plotting, and not the length of the masked array in total. So, 
            #   we have to check for masking and adjust appropriately.
            if downsamp is None:
                if 'color' in self.props:
                    if self.props_history is not None:
                        # Get color from history
                        if ma.is_masked(self.pos_history[n0]):
                            not_msk = ~self.pos_history[n0][:,0].mask
                            color = self.props_history[n0].loc[not_msk, 'color']
                        else:
                            color = self.props_history[n0]['color']
                    else:
                        if ma.is_masked(self.pos_history[n0]):
                            not_msk = ~self.pos_history[n0][:,0].mask
                            color = self.props.loc[not_msk, 'color']
                        else:
                            color = self.props['color']
                    scat = ax.scatter(self.pos_history[n0][:,0], self.pos_history[n0][:,1],
                                    self.pos_history[n0][:,2], 
                                    label=self.shared_props['name'],
                                    c=color, animated=True)
                else:
                    scat = ax.scatter(self.pos_history[n0][:,0], self.pos_history[n0][:,1],
                                    self.pos_history[n0][:,2], 
                                    label=self.shared_props['name'],
                                    color=self.shared_props['color'], animated=True)
            else:
                if 'color' in self.props:
                    if self.props_history is not None:
                        # Get color from history
                        if ma.is_masked(self.pos_history[n0][downsamp,0]):
                            not_msk = ~self.pos_history[n0][downsamp,0].mask
                            color = self.props_history[n0].loc[downsamp,'color'][not_msk]
                        else:
                            color = self.props_history[n0].loc[downsamp,'color']
                    else:
                        if ma.is_masked(self.pos_history[n0][:,0]):
                            not_msk = ~self.pos_history[n0][:,0].mask
                            color = self.props.loc[downsamp,'color'][not_msk]
                        else:
                            color = self.props.loc[downsamp,'color']
                    scat = ax.scatter(self.pos_history[n0][downsamp,0],
                                    self.pos_history[n0][downsamp,1],
                                    self.pos_history[n0][downsamp,2],
                                    label=self.shared_props['name'], 
                                    color=color, animated=True)
                else:
                    scat = ax.scatter(self.pos_history[n0][downsamp,0],
                                    self.pos_history[n0][downsamp,1],
                                    self.pos_history[n0][downsamp,2],
                                    label=self.shared_props['name'], 
                                    color=self.shared_props['color'], animated=True)

            # textual info
            time_text = ax.text2D(0.02, 1, 'time = {:.2f}'.format(
                                  self.envir.time_history[n0]),
                                  transform=ax.transAxes, animated=True,
                                  verticalalignment='top', fontsize=12)
            perc_left, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                self._calc_basic_stats(DIM3=True, t_indx=n0)
            # see the note on the matching anchor in Swarm.plot
            flow_text = ax.text2D(0.65, 0.9,
                                  r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(
                                  np.linalg.norm(avg_swrm_vel), self.envir.units)+
                                  r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s'.format(
                                  avg_swrm_spd, std_swrm_spd, self.envir.units),
                                  transform=ax.transAxes, animated=True,
                                  horizontalalignment='left', fontsize=10)
            perc_text = ax.text2D(0.02, 0,
                                  '{:.1f}% remain\n'.format(perc_left),
                                  transform=fig.transFigure, animated=True,
                                  fontsize=10)
            x_text = axHistx.text(0.02, 0.98,
                                  r'Fluid $\overline{v}_x$'+': {:.2g} {}/s\n'.format(
                                  avg_spd_x, self.envir.units)+
                                  r'Agent $\overline{v}_x$'+': {:.2g} {}/s'.format(
                                  avg_swrm_vel[0], self.envir.units),
                                  transform=axHistx.transAxes, animated=True,
                                  verticalalignment='top', fontsize=10)
            y_text = axHisty.text(0.02, 0.98,
                                  r'Fluid $\overline{v}_y$'+': {:.2g} {}/s\n'.format(
                                  avg_spd_y, self.envir.units)+
                                  r'Agent $\overline{v}_y$'+': {:.2g} {}/s'.format(
                                  avg_swrm_vel[1], self.envir.units),
                                  transform=axHisty.transAxes, animated=True,
                                  verticalalignment='top', fontsize=10)
            z_text = axHistz.text(0.02, 0.98,
                                  r'Fluid $\overline{v}_z$'+': {:.2g} {}/s\n'.format(
                                  avg_spd_z, self.envir.units)+
                                  r'Agent $\overline{v}_z$'+': {:.2g} {}/s'.format(
                                  avg_swrm_vel[2], self.envir.units),
                                  transform=axHistz.transAxes, animated=True,
                                  verticalalignment='top', fontsize=10)

            if dist == 'hist':
                # histograms
                data_x = self.pos_history[n0][:,0].compressed()
                data_y = self.pos_history[n0][:,1].compressed()
                data_z = self.pos_history[n0][:,2].compressed()
                bins_x = np.linspace(0, self.envir.L[0], 26)
                bins_y = np.linspace(0, self.envir.L[1], 26)
                bins_z = np.linspace(0, self.envir.L[2], 26)
                n_x, bins_x, patches_x = axHistx.hist(data_x, bins=bins_x, alpha=0.8)
                n_y, bins_y, patches_y = axHisty.hist(data_y, bins=bins_y, alpha=0.8)
                n_z, bins_z, patches_z = axHistz.hist(data_z, bins=bins_z, alpha=0.8)
            else:
                # Gaussian Kernel Density Estimation
                if dist == 'cov':
                    fac_x = self.shared_props['cov'][0,0]
                    fac_y = self.shared_props['cov'][1,1]
                    fac_z = self.shared_props['cov'][2,2]
                else:
                    try:
                        # see if a float was passed
                        fac_x = float(dist)
                        fac_y = fac_x
                        fac_z = fac_x
                    except:
                        # estimate covariance from Scotts Factor. HOWEVER: this 
                        #   estimation breaks if IC is point source.
                        fac_x = None
                        fac_y = None
                        fac_z = None
                xmesh = np.linspace(0, self.envir.L[0], 1000)
                ymesh = np.linspace(0, self.envir.L[1], 1000)
                zmesh = np.linspace(0, self.envir.L[2], 1000)
                # deal with point sources
                pos_x = self.pos_history[n0][:,0].compressed()
                pos_y = self.pos_history[n0][:,1].compressed()
                pos_z = self.pos_history[n0][:,2].compressed()
                try:
                    if len(pos_x) > 1:
                        x_density = stats.gaussian_kde(pos_x, fac_x)
                        x_density = x_density(xmesh)
                    elif len(pos_x) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        x_density = np.zeros_like(xmesh)
                except np.linalg.LinAlgError:
                    idx = (np.abs(xmesh - pos_x[0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                try:
                    if len(pos_y) > 1:
                        y_density = stats.gaussian_kde(pos_y, fac_y)
                        y_density = y_density(ymesh)
                    elif len(pos_y) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        y_density = np.zeros_like(ymesh)
                except np.linalg.LinAlgError:
                    idy = (np.abs(ymesh - pos_y[0])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idy] = 1
                try:
                    if len(pos_z) > 1:
                        z_density = stats.gaussian_kde(pos_z, fac_z)
                        z_density = z_density(zmesh)
                    elif len(pos_z) == 1:
                        raise np.linalg.LinAlgError
                    else:
                        z_density = np.zeros_like(zmesh)
                except np.linalg.LinAlgError:
                    idz = (np.abs(zmesh - pos_z[0])).argmin()
                    z_density = np.zeros_like(zmesh); z_density[idz] = 1
                xdens_plt, = axHistx.plot(xmesh, x_density)
                ydens_plt, = axHisty.plot(ymesh, y_density)
                zdens_plt, = axHistz.plot(zmesh, z_density)
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_yaxis().set_ticks([])
                axHistz.get_yaxis().set_ticks([])
                if np.max(xdens_plt.get_ydata()) != 0:
                    axHistx.set_ylim(bottom=0, top=np.max(xdens_plt.get_ydata()))
                else:
                    axHistx.set_ylim(bottom=0)
                if np.max(ydens_plt.get_ydata()) != 0:
                    axHisty.set_ylim(bottom=0, top=np.max(ydens_plt.get_ydata()))
                else:
                    axHisty.set_ylim(bottom=0)
                if np.max(zdens_plt.get_ydata()) != 0:
                    axHistz.set_ylim(bottom=0, top=np.max(zdens_plt.get_ydata()))
                else:
                    axHistz.set_ylim(bottom=0)

        # animation function. Called sequentially
        angle_props_warned = [False]
        def animate(n):
            if n < len(self.pos_history):
                time_text.set_text('time = {:.2f}'.format(self.envir.time_history[n]))
                if not DIM3:
                    # 2D
                    perc_left, avg_spd_x, avg_spd_y, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                        self._calc_basic_stats(DIM3=False, t_indx=n)
                    stats_text.set_text('{:.1f}% remain\n'.format(perc_left)+
                         '\n  ------ Info ------\n'+
                         r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units)+
                         r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s\n'.format(avg_swrm_spd, std_swrm_spd, self.envir.units))
                    x_text.set_text(r'Fluid $\overline{v}_x$'+': {:.2g} \n'.format(avg_spd_x)+
                         r'Agent $\overline{v}_x$'+': {:.2g}'.format(avg_swrm_vel[0]))
                    y_text.set_text(r'Fluid $\overline{v}_y$'+': {:.2g} \n'.format(avg_spd_y)+
                         r'Agent $\overline{v}_y$'+': {:.2g}'.format(avg_swrm_vel[1]))
                    if fluid == 'vort' and self.envir.flow is not None:
                        vort = self.envir.get_vorticity(t_indx=n)
                        fld.set_array(vort.T)
                        # NOT autoscale(): it rescales to this frame's own
                        # min/max, which discards any clip the caller asked for
                        # and moves zero off the white centre of RdBu, differently
                        # every frame. That was the background flashing.
                        fld.norm = _vorticity_norm(vort, clip, fld.norm)
                        fld.changed()
                    elif fluid == 'quiver' and self.envir.flow is not None:
                        if self.envir.flow.flow_times is not None:
                            flow = self.envir.interpolate_temporal_flow(t_index=n)
                            fld.set_UVC(flow[0][::M,::N].T, flow[1][::M,::N].T)
                        else:
                            fld.set_UVC(self.envir.flow[0][::M,::N].T, self.envir.flow[1][::M,::N].T)
                    warning_msg = "Using velocity for heading markers "+\
                                  "and not the 'angles' property because "+\
                                  "the property history was not recorded."
                    if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                        ibmesh = self.envir.interpolate_temporal_mesh(time=self.envir.time_history[n])
                        mesh_col.set_segments(ibmesh)
                    if downsamp is None:
                        scat.set_offsets(self.pos_history[n])
                        if 'color' in self.props:
                            if self.props_history is not None:
                                scat.set_color(self.props_history[n]['color'])
                            else:
                                scat.set_color(self.props['color'])
                        # Grab angles for heading markers
                        if 'angle' in self.props and plot_heading:
                            if self.props_history is not None:
                                angles = self.props_history[n]['angle']
                            else:
                                if not angle_props_warned[0]:
                                    warnings.warn(warning_msg, stacklevel=9)
                                angle_props_warned[0] = True
                                angles = np.arctan2(self.vel_history[n][:,1], 
                                                    self.vel_history[n][:,0])
                        elif plot_heading:
                            # this is defined even for (0,0) by convention
                            angles = np.arctan2(self.vel_history[n][:,1], 
                                                self.vel_history[n][:,0])
                    else:
                        scat.set_offsets(self.pos_history[n][downsamp,:])
                        if 'color' in self.props:
                            if self.props_history is not None:
                                scat.set_color(self.props_history[n].loc[downsamp,'color'])
                            else:
                                scat.set_color(self.props.loc[downsamp,'color'])
                        # Grab angles for heading markers
                        if 'angle' in self.props and plot_heading:
                            if self.props_history is not None:
                                angles = self.props.loc[downsamp,'angle']
                            else:
                                if not angle_props_warned[0]:
                                    warnings.warn(warning_msg, stacklevel=9)
                                angle_props_warned[0] = True
                                angles = np.arctan2(self.vel_history[n][downsamp,1], 
                                                    self.vel_history[n][downsamp,0])
                        elif plot_heading:
                            # this is defined even for (0,0) by convention
                            angles = np.arctan2(self.vel_history[n][downsamp,1], 
                                                self.vel_history[n][downsamp,0])
                    # set heading markers
                    if plot_heading:
                        paths = []
                        for angle in angles:
                            if ma.is_masked(angle):
                                paths.append(circle)
                            else:
                                # make the heading marker stick out by one diameter
                                line_verts = np.array([[0,0],[circ_rad*3*np.cos(angle),
                                                            circ_rad*3*np.sin(angle)]])
                                # combine the circle and line vertices
                                verts = np.concatenate([circle.vertices, line_verts])
                                # append to path list
                                paths.append(mPath(verts, codes))
                        scat.set_paths(paths)
                    else:
                        scat.set_paths([circle])
                    
                    if dist == 'hist':
                        n_x, _ = np.histogram(self.pos_history[n][:,0].compressed(), bins_x)
                        n_y, _ = np.histogram(self.pos_history[n][:,1].compressed(), bins_y)
                        for rect, h in zip(patches_x, n_x):
                            rect.set_height(h)
                        for rect, h in zip(patches_y, n_y):
                            rect.set_width(h)
                        if fluid == 'vort' and self.envir.flow is not None:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, fld, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                            else:
                                return [fld, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                        else:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                            else:
                                return [scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                    else:
                        pos_x = self.pos_history[n][:,0].compressed()
                        pos_y = self.pos_history[n][:,1].compressed()
                        try:
                            if len(pos_x) > 1:
                                x_density = stats.gaussian_kde(pos_x, fac_x)
                                x_density = x_density(xmesh)
                            elif len(pos_x) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                x_density = np.zeros_like(xmesh)
                        except np.linalg.LinAlgError:
                            idx = (np.abs(xmesh - pos_x[0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                        try:
                            if len(pos_y) > 1:
                                y_density = stats.gaussian_kde(pos_y, fac_y)
                                y_density = y_density(ymesh)
                            elif len(pos_y) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                y_density = np.zeros_like(ymesh)
                        except np.linalg.LinAlgError:
                            idy = (np.abs(ymesh - pos_y[0])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idy] = 1
                        xdens_plt.set_ydata(x_density)
                        ydens_plt.set_xdata(y_density)
                        if np.max(xdens_plt.get_ydata()) != 0:
                            axHistx.set_ylim(bottom=0, top=np.max(xdens_plt.get_ydata()))
                        else:
                            axHistx.set_ylim(bottom=0)
                        if np.max(ydens_plt.get_xdata()) != 0:
                            axHisty.set_xlim(left=0, right=np.max(ydens_plt.get_xdata()))
                        else:
                            axHisty.set_xlim(left=0)
                        if fluid == 'vort' and self.envir.flow is not None:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, fld, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                            else:
                                return [fld, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                        else:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                            else:
                                return [scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                    
                else:
                    # 3D
                    perc_left, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                        self._calc_basic_stats(DIM3=True, t_indx=n)
                    # print(n)
                    # print(self.pos_history[n].all() is ma.masked)
                    flow_text.set_text(r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(
                                       np.linalg.norm(avg_swrm_vel), self.envir.units)+
                                       r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s'.format(
                                       avg_swrm_spd, std_swrm_spd, self.envir.units))
                    perc_text.set_text('{:.1f}% remain\n'.format(perc_left))
                    x_text.set_text(r'Fluid $\overline{v}_x$'+': {:.2g} {}/s\n'.format(
                                    avg_spd_x, self.envir.units)+
                                    r'Agent $\overline{v}_x$'+': {:.2g} {}/s'.format(
                                    avg_swrm_vel[0], self.envir.units))
                    y_text.set_text(r'Fluid $\overline{v}_y$'+': {:.2g} {}/s\n'.format(
                                    avg_spd_y, self.envir.units)+
                                    r'Agent $\overline{v}_y$'+': {:.2g} {}/s'.format(
                                    avg_swrm_vel[1], self.envir.units))
                    z_text.set_text(r'Fluid $\overline{v}_z$'+': {:.2g} {}/s\n'.format(
                                    avg_spd_z, self.envir.units)+
                                    r'Agent $\overline{v}_z$'+': {:.2g} {}/s'.format(
                                    avg_swrm_vel[2], self.envir.units))
                    # UNFORTUNATELY, 3D matplotlib plotting is very weird about masked 
                    #   arrays. The implementation does not parallel 2D: it wants a color 
                    #   list that is the same length as the number of points it will be 
                    #   plotting, and not the length of the masked array in total. So, 
                    #   we have to check for masking and adjust appropriately.
                    if downsamp is None:
                        scat._offsets3d = (np.ma.ravel(self.pos_history[n][:,0].compressed()),
                                        np.ma.ravel(self.pos_history[n][:,1].compressed()),
                                        np.ma.ravel(self.pos_history[n][:,2].compressed()))
                        if 'color' in self.props:
                            if self.props_history is not None:
                                if ma.is_masked(self.pos_history[n]):
                                    not_msk = ~self.pos_history[n][:,0].mask
                                    scat.set_color(self.props_history[n].loc[not_msk,'color'])
                                else:
                                    scat.set_color(self.props_history[n]['color'])
                            else:
                                if ma.is_masked(self.pos_history[n]):
                                    not_msk = ~self.pos_history[n][:,0].mask
                                    scat.set_color(self.props.loc[not_msk,'color'])
                                else:
                                    scat.set_color(self.props['color'])
                    else:
                        scat._offsets3d = (np.ma.ravel(self.pos_history[n][downsamp,0].compressed()),
                                        np.ma.ravel(self.pos_history[n][downsamp,1].compressed()),
                                        np.ma.ravel(self.pos_history[n][downsamp,2].compressed()))
                        if 'color' in self.props:
                            if self.props_history is not None:
                                if ma.is_masked(self.pos_history[n][downsamp,0]):
                                    not_msk = ~self.pos_history[n][downsamp,0].mask
                                    scat.set_color(self.props_history[n].loc[downsamp,'color'][not_msk])
                                else:
                                    scat.set_color(self.props_history[n].loc[downsamp,'color'])
                            else:
                                if ma.is_masked(self.pos_history[n][downsamp,0]):
                                    not_msk = ~self.pos_history[n][downsamp,0].mask
                                    scat.set_color(self.props.loc[downsamp,'color'][not_msk])
                                else:
                                    scat.set_color(self.props.loc[downsamp,'color'])
                    if dist == 'hist':
                        n_x, _ = np.histogram(self.pos_history[n][:,0].compressed(), bins_x)
                        n_y, _ = np.histogram(self.pos_history[n][:,1].compressed(), bins_y)
                        n_z, _ = np.histogram(self.pos_history[n][:,2].compressed(), bins_z)
                        for rect, h in zip(patches_x, n_x):
                            rect.set_height(h)
                        for rect, h in zip(patches_y, n_y):
                            rect.set_height(h)
                        for rect, h in zip(patches_z, n_z):
                            rect.set_height(h)
                        fig.canvas.draw()
                        return [scat, time_text, flow_text, perc_text, x_text, 
                            y_text, z_text] + list(patches_x) + list(patches_y) + list(patches_z)
                    else:
                        pos_x = self.pos_history[n][:,0].compressed()
                        pos_y = self.pos_history[n][:,1].compressed()
                        pos_z = self.pos_history[n][:,2].compressed()
                        try:
                            if len(pos_x) > 1:
                                x_density = stats.gaussian_kde(pos_x, fac_x)
                                x_density = x_density(xmesh)
                            elif len(pos_x) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                x_density = np.zeros_like(xmesh)
                        except np.linalg.LinAlgError:
                            idx = (np.abs(xmesh - pos_x[0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                        try:
                            if len(pos_y) > 1:
                                y_density = stats.gaussian_kde(pos_y, fac_y)
                                y_density = y_density(ymesh)
                            elif len(pos_y) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                y_density = np.zeros_like(ymesh)
                        except np.linalg.LinAlgError:
                            idy = (np.abs(ymesh - pos_y[0])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idy] = 1
                        try:
                            if len(pos_z) > 1:
                                z_density = stats.gaussian_kde(pos_z, fac_z)
                                z_density = z_density(zmesh)
                            elif len(pos_z) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                z_density = np.zeros_like(zmesh)
                        except np.linalg.LinAlgError:
                            idz = (np.abs(zmesh - pos_z[0])).argmin()
                            z_density = np.zeros_like(zmesh); z_density[idz] = 1
                        xdens_plt.set_ydata(x_density)
                        ydens_plt.set_ydata(y_density)
                        zdens_plt.set_ydata(z_density)
                        if np.max(xdens_plt.get_ydata()) != 0:
                            axHistx.set_ylim(bottom=0, top=np.max(xdens_plt.get_ydata()))
                        else:
                            axHistx.set_ylim(bottom=0)
                        if np.max(ydens_plt.get_ydata()) != 0:
                            axHisty.set_ylim(bottom=0, top=np.max(ydens_plt.get_ydata()))
                        else:
                            axHisty.set_ylim(bottom=0)
                        if np.max(zdens_plt.get_ydata()) != 0:
                            axHistz.set_ylim(bottom=0, top=np.max(zdens_plt.get_ydata()))
                        else:
                            axHistz.set_ylim(bottom=0)
                        fig.canvas.draw()
                        return [scat, time_text, flow_text, perc_text, x_text, 
                                y_text, z_text, xdens_plt, ydens_plt, zdens_plt]
                    
            else:
                time_text.set_text('time = {:.2f}'.format(self.envir.time))
                if not DIM3:
                    # 2D end
                    perc_left, avg_spd_x, avg_spd_y, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                        self._calc_basic_stats(DIM3=False, t_indx=None)
                    stats_text.set_text('{:.1f}% remain\n'.format(perc_left)+
                         '\n  ------ Info ------\n'+
                         r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units)+
                         r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s\n'.format(avg_swrm_spd, std_swrm_spd, self.envir.units))
                    x_text.set_text(r'Fluid $\overline{v}_x$'+': {:.2g} \n'.format(avg_spd_x)+
                         r'Agent $\overline{v}_x$'+': {:.2g}'.format(avg_swrm_vel[0]))
                    y_text.set_text(r'Fluid $\overline{v}_y$'+': {:.2g} \n'.format(avg_spd_y)+
                         r'Agent $\overline{v}_y$'+': {:.2g}'.format(avg_swrm_vel[1]))
                    if fluid == 'vort' and self.envir.flow is not None:
                        vort = self.envir.get_vorticity()
                        fld.set_array(vort.T)
                        fld.norm = _vorticity_norm(vort, clip, fld.norm)
                        fld.changed()
                    elif fluid == 'quiver' and self.envir.flow is not None:
                        if self.envir.flow.flow_times is not None:
                            flow = self.envir.interpolate_temporal_flow()
                            fld.set_UVC(flow[0][::M,::N].T, flow[1][::M,::N].T)
                        else:
                            fld.set_UVC(self.envir.flow[0][::M,::N].T, self.envir.flow[1][::M,::N].T)
                    if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                        ibmesh = self.envir.interpolate_temporal_mesh()
                        mesh_col.set_segments(ibmesh)
                    if downsamp is None:
                        scat.set_offsets(self.positions)
                        if self.props_history is not None and 'color' in self.props:
                            scat.set_color(self.props['color'])
                        # Grab angles for heading markers
                        if 'angle' in self.props and self.props_history is not None:
                            angles = self.props['angle']
                        else:
                            # this is defined even for (0,0) by convention
                            angles = np.arctan2(self.velocities[:,1], 
                                                self.velocities[:,0])
                    else:
                        scat.set_offsets(self.positions[downsamp,:])
                        if self.props_history is not None and 'color' in self.props:
                            scat.set_color(self.props.loc[downsamp,'color'])
                        # Grab angles for heading markers
                        if 'angle' in self.props and self.props_history is not None:
                            angles = self.props.loc[downsamp,'angle']
                        else:
                            # this is defined even for (0,0) by convention
                            angles = np.arctan2(self.velocities[downsamp,1], 
                                                self.velocities[downsamp,0])
                    # set heading markers
                    if plot_heading:
                        paths = []
                        for angle in angles:
                            if ma.is_masked(angle):
                                paths.append(circle)
                            else:
                                # make the heading marker stick out by one diameter
                                line_verts = np.array([[0,0],[circ_rad*3*np.cos(angle),
                                                            circ_rad*3*np.sin(angle)]])
                                # combine the circle and line vertices
                                verts = np.concatenate([circle.vertices, line_verts])
                                # append to path list
                                paths.append(mPath(verts, codes))
                        scat.set_paths(paths)
                    else:
                        scat.set_paths([circle])
                    if dist == 'hist':
                        n_x, _ = np.histogram(self.positions[:,0].compressed(), bins_x)
                        n_y, _ = np.histogram(self.positions[:,1].compressed(), bins_y)
                        for rect, h in zip(patches_x, n_x):
                            rect.set_height(h)
                        for rect, h in zip(patches_y, n_y):
                            rect.set_width(h)
                        if fluid == 'vort' and self.envir.flow is not None:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, fld, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                            else:
                                return [fld, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                        else:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                            else:
                                return [scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                    else:
                        pos_x = self.positions[:,0].compressed()
                        pos_y = self.positions[:,1].compressed()
                        try:
                            if len(pos_x) > 1:
                                x_density = stats.gaussian_kde(pos_x, fac_x)
                                x_density = x_density(xmesh)
                            elif len(pos_x) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                x_density = np.zeros_like(xmesh)
                        except np.linalg.LinAlgError:
                            idx = (np.abs(xmesh - pos_x[0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                        try:
                            if len(pos_y) > 1:
                                y_density = stats.gaussian_kde(pos_y, fac_y)
                                y_density = y_density(ymesh)
                            elif len(pos_y) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                y_density = np.zeros_like(ymesh)
                        except np.linalg.LinAlgError:
                            idy = (np.abs(ymesh - pos_y[0])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idy] = 1
                        xdens_plt.set_ydata(x_density)
                        ydens_plt.set_xdata(y_density)
                        if np.max(xdens_plt.get_ydata()) != 0:
                            axHistx.set_ylim(bottom=0, top=np.max(xdens_plt.get_ydata()))
                        else:
                            axHistx.set_ylim(bottom=0)
                        if np.max(ydens_plt.get_xdata()) != 0:
                            axHisty.set_xlim(left=0, right=np.max(ydens_plt.get_xdata()))
                        else:
                            axHisty.set_xlim(left=0)
                        if fluid == 'vort' and self.envir.flow is not None:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, fld, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                            else:
                                return [fld, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                        else:
                            if mesh_col is not None and self.envir.ibmesh.ndim == 4:
                                return [mesh_col, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                            else:
                                return [scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                    
                else:
                    # 3D end
                    perc_left, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel, avg_swrm_spd, std_swrm_spd = \
                        self._calc_basic_stats(DIM3=True)
                    flow_text.set_text(r'Agent $|\overline{v}|$'+': {:.2g} {}/s\n'.format(
                                       np.linalg.norm(avg_swrm_vel), self.envir.units)+
                                       r'Agent $\overline{|v|}$'+': {:.2g} $\\pm$ {:.2g} {}/s'.format(
                                       avg_swrm_spd, std_swrm_spd, self.envir.units))
                    perc_text.set_text('{:.1f}% remain\n'.format(perc_left))
                    x_text.set_text(r'Fluid $\overline{v}_x$'+': {:.2g} {}/s\n'.format(
                                    avg_spd_x, self.envir.units)+
                                    r'Agent $\overline{v}_x$'+': {:.2g} {}/s'.format(
                                    avg_swrm_vel[0], self.envir.units))
                    y_text.set_text(r'Fluid $\overline{v}_y$'+': {:.2g} {}/s\n'.format(
                                    avg_spd_y, self.envir.units)+
                                    r'Agent $\overline{v}_y$'+': {:.2g} {}/s'.format(
                                    avg_swrm_vel[1], self.envir.units))
                    z_text.set_text(r'Fluid $\overline{v}_z$'+': {:.2g} {}/s\n'.format(
                                    avg_spd_z, self.envir.units)+
                                    r'Agent $\overline{v}_z$'+': {:.2g} {}/s'.format(
                                    avg_swrm_vel[2], self.envir.units))
                    # UNFORTUNATELY, 3D matplotlib plotting is very weird about masked 
                    #   arrays. The implementation does not parallel 2D: it wants a color 
                    #   list that is the same length as the number of points it will be 
                    #   plotting, and not the length of the masked array in total. So, 
                    #   we have to check for masking and adjust appropriately.
                    if downsamp is None:
                        scat._offsets3d = (np.ma.ravel(self.positions[:,0].compressed()),
                                        np.ma.ravel(self.positions[:,1].compressed()),
                                        np.ma.ravel(self.positions[:,2].compressed()))
                        if 'color' in self.props:
                            if ma.is_masked(self.positions):
                                not_msk = ~self.positions[:,0].mask
                                scat.set_color(self.props.loc[not_msk,'color'])
                            else:
                                scat.set_color(self.props['color'])
                    else:
                        scat._offsets3d = (np.ma.ravel(self.positions[downsamp,0].compressed()),
                                        np.ma.ravel(self.positions[downsamp,1].compressed()),
                                        np.ma.ravel(self.positions[downsamp,2].compressed()))
                        if 'color' in self.props:
                            if ma.is_masked(self.positions[downsamp,0]):
                                not_msk = ~self.positions[downsamp,0].mask
                                scat.set_color(self.props.loc[downsamp,'color'][not_msk])
                            else:
                                scat.set_color(self.props.loc[downsamp,'color'])
                    if dist == 'hist':
                        n_x, _ = np.histogram(self.positions[:,0].compressed(), bins_x)
                        n_y, _ = np.histogram(self.positions[:,1].compressed(), bins_y)
                        n_z, _ = np.histogram(self.positions[:,2].compressed(), bins_z)
                        for rect, h in zip(patches_x, n_x):
                            rect.set_height(h)
                        for rect, h in zip(patches_y, n_y):
                            rect.set_height(h)
                        for rect, h in zip(patches_z, n_z):
                            rect.set_height(h)
                        fig.canvas.draw()
                        return [scat, time_text, flow_text, perc_text, x_text, 
                            y_text, z_text] + list(patches_x) + list(patches_y) + list(patches_z)
                    else:
                        pos_x = self.positions[:,0].compressed()
                        pos_y = self.positions[:,1].compressed()
                        pos_z = self.positions[:,2].compressed()
                        try:
                            if len(pos_x) > 1:
                                x_density = stats.gaussian_kde(pos_x, fac_x)
                                x_density = x_density(xmesh)
                            elif len(pos_x) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                x_density = np.zeros_like(xmesh)
                        except np.linalg.LinAlgError:
                            idx = (np.abs(xmesh - pos_x[0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                        try:
                            if len(pos_y) > 1:
                                y_density = stats.gaussian_kde(pos_y, fac_y)
                                y_density = y_density(ymesh)
                            elif len(pos_y) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                y_density = np.zeros_like(ymesh)
                        except np.linalg.LinAlgError:
                            idy = (np.abs(ymesh - pos_y[0])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idy] = 1
                        try:
                            if len(pos_z) > 1:
                                z_density = stats.gaussian_kde(pos_z, fac_z)
                                z_density = z_density(zmesh)
                            elif len(pos_z) == 1:
                                raise np.linalg.LinAlgError
                            else:
                                z_density = np.zeros_like(zmesh)
                        except np.linalg.LinAlgError:
                            idz = (np.abs(zmesh - pos_z[0])).argmin()
                            z_density = np.zeros_like(zmesh); z_density[idz] = 1
                        xdens_plt.set_ydata(x_density)
                        ydens_plt.set_ydata(y_density)
                        zdens_plt.set_ydata(z_density)
                        if np.max(xdens_plt.get_ydata()) != 0:
                            axHistx.set_ylim(bottom=0, top=np.max(xdens_plt.get_ydata()))
                        else:
                            axHistx.set_ylim(bottom=0)
                        if np.max(ydens_plt.get_ydata()) != 0:
                            axHisty.set_ylim(bottom=0, top=np.max(ydens_plt.get_ydata()))
                        else:
                            axHisty.set_ylim(bottom=0)
                        if np.max(zdens_plt.get_ydata()) != 0:
                            axHistz.set_ylim(bottom=0, top=np.max(zdens_plt.get_ydata()))
                        else:
                            axHistz.set_ylim(bottom=0)
                        fig.canvas.draw()
                        return [scat, time_text, flow_text, perc_text, x_text, 
                                y_text, z_text, xdens_plt, ydens_plt, zdens_plt]

        # on-screen playback: the frames were chosen to be fps apart in the
        # finished video, so display them that far apart too (in milliseconds).
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                    interval=1000/fps, repeat=False, blit=True)

        if movie_filename is not None:
            try:
                if writer_kwargs is None:
                    writer = animation.FFMpegWriter(fps=fps, 
                        metadata=dict(artist='Christopher Strickland'))#, bitrate=1800)
                else:
                    writer = animation.FFMpegWriter(**writer_kwargs)
                if save_kwargs is None:
                    anim.save(movie_filename, writer=writer, dpi=150)
                else:
                    anim.save(movie_filename, writer=writer, **save_kwargs)
                plt.close()
                print('Video saved to {}.'.format(movie_filename))
            except:
                print('Failed to save animation.')
                print('Check that you have ffmpeg or mencoder installed; these')
                print("aren't Python packages, but stand-alone applications.")
                print("An H.264 encoder is needed on the system's path in order")
                print('to save to that video format.')
                raise
        else:
            plt.show()

